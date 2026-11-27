#include "tav_pose/yolo_detector.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tav_pose {
namespace {
constexpr int kClassOffset = 7680;

void append_decoded_box(float x0,
                        float y0,
                        float x1,
                        float y1,
                        float score,
                        int cls_id,
                        int orig_w,
                        int orig_h,
                        float min_side,
                        std::vector<cv::Rect>& nms_boxes,
                        std::vector<cv::Rect2f>& raw_boxes,
                        std::vector<float>& scores,
                        std::vector<int>& class_ids) {
    if (!std::isfinite(x0) || !std::isfinite(y0) ||
        !std::isfinite(x1) || !std::isfinite(y1) || !std::isfinite(score)) {
        return;
    }
    x0 = std::clamp(x0, 0.0f, static_cast<float>(orig_w - 1));
    y0 = std::clamp(y0, 0.0f, static_cast<float>(orig_h - 1));
    x1 = std::clamp(x1, x0 + 1.0f, static_cast<float>(orig_w));
    y1 = std::clamp(y1, y0 + 1.0f, static_cast<float>(orig_h));
    if ((x1 - x0) < min_side || (y1 - y0) < min_side) return;

    raw_boxes.emplace_back(x0, y0, x1 - x0, y1 - y0);
    nms_boxes.emplace_back(
        static_cast<int>(std::round(x0)) + cls_id * kClassOffset,
        static_cast<int>(std::round(y0)) + cls_id * kClassOffset,
        std::max(1, static_cast<int>(std::round(x1 - x0))),
        std::max(1, static_cast<int>(std::round(y1 - y0))));
    scores.push_back(score);
    class_ids.push_back(cls_id);
}
}

YoloDetector::YoloDetector(const std::string& onnx_path,
                           std::vector<std::string> class_names,
                           int input_w,
                           int input_h,
                           float conf_thresh,
                           float nms_thresh,
                           bool use_gray_3ch,
                           GpuMode gpu_mode)
    : class_names_(std::move(class_names)),
      input_w_(input_w),
      input_h_(input_h),
      conf_thresh_(conf_thresh),
      nms_thresh_(nms_thresh),
      use_gray_3ch_(use_gray_3ch) {
    net_ = cv::dnn::readNetFromONNX(onnx_path);
    if (net_.empty()) {
        throw std::runtime_error("Failed to load ONNX model: " + onnx_path);
    }
    const auto cv_names = net_.getUnconnectedOutLayersNames();
    out_names_.assign(cv_names.begin(), cv_names.end());
    backend_status_ = configure_dnn_backend(
        net_, onnx_path, out_names_, cv::Size(input_w_, input_h_), 3, gpu_mode);
}

void YoloDetector::decode_yolov8(const cv::Mat& out_raw,
                                 int orig_w,
                                 int orig_h,
                                 float scale,
                                 int left,
                                 int top,
                                 std::vector<cv::Rect>& nms_boxes,
                                 std::vector<cv::Rect2f>& raw_boxes,
                                 std::vector<float>& scores,
                                 std::vector<int>& class_ids) const {
    cv::Mat out = out_raw.isContinuous() ? out_raw : out_raw.clone();
    if (out.depth() != CV_32F) out.convertTo(out, CV_32F);
    if (out.dims != 3 || out.size[0] != 1 || out.size[1] < 5) {
        throw std::runtime_error("Unexpected YOLOv8 ONNX output tensor shape");
    }
    const int channels = out.size[1];
    const int num_anchors = out.size[2];
    const int nc = channels - 4;
    const float* data = out.ptr<float>();

    for (int i = 0; i < num_anchors; ++i) {
        int best_cls = 0;
        float best_score = data[4 * num_anchors + i];
        for (int c = 1; c < nc; ++c) {
            const float s = data[(4 + c) * num_anchors + i];
            if (s > best_score) {
                best_score = s;
                best_cls = c;
            }
        }
        if (!std::isfinite(best_score) || best_score < conf_thresh_) continue;

        const float cx = data[0 * num_anchors + i];
        const float cy = data[1 * num_anchors + i];
        const float bw = data[2 * num_anchors + i];
        const float bh = data[3 * num_anchors + i];

        const float x0 = (cx - 0.5f * bw - static_cast<float>(left)) / scale;
        const float y0 = (cy - 0.5f * bh - static_cast<float>(top)) / scale;
        const float x1 = (cx + 0.5f * bw - static_cast<float>(left)) / scale;
        const float y1 = (cy + 0.5f * bh - static_cast<float>(top)) / scale;
        append_decoded_box(x0, y0, x1, y1, best_score, best_cls,
                           orig_w, orig_h, 1.0f,
                           nms_boxes, raw_boxes, scores, class_ids);
    }
}

void YoloDetector::decode_ifl_net(const std::vector<cv::Mat>& outs,
                                  int orig_w,
                                  int orig_h,
                                  float scale,
                                  int left,
                                  int top,
                                  std::vector<cv::Rect>& nms_boxes,
                                  std::vector<cv::Rect2f>& raw_boxes,
                                  std::vector<float>& scores,
                                  std::vector<int>& class_ids) const {
    const int nc = static_cast<int>(class_names_.size());
    std::vector<cv::Mat> hm_list, rg_list;
    for (const auto& m : outs) {
        if (m.dims != 4 || m.size[0] != 1) continue;
        cv::Mat mc = m.isContinuous() ? m : m.clone();
        if (mc.depth() != CV_32F) mc.convertTo(mc, CV_32F);
        if (mc.size[1] == nc && nc != 4) {
            hm_list.push_back(mc);
        } else if (mc.size[1] == 4) {
            rg_list.push_back(mc);
        }
    }
    if (hm_list.empty() || rg_list.empty()) {
        throw std::runtime_error("Unexpected IFL-Net ONNX output tensor shapes");
    }

    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    for (const auto& hm : hm_list) {
        const int H = hm.size[2];
        const int W = hm.size[3];
        if (H <= 0 || W <= 0) continue;
        const cv::Mat* rg_match = nullptr;
        for (const auto& rg : rg_list) {
            if (rg.size[2] == H && rg.size[3] == W) {
                rg_match = &rg;
                break;
            }
        }
        if (!rg_match) continue;

        const float stride = static_cast<float>(input_w_) / static_cast<float>(W);
        const int HW = H * W;
        const float* hm_ptr = hm.ptr<float>();
        const float* rg_ptr = rg_match->ptr<float>();

        for (int c = 0; c < nc; ++c) {
            cv::Mat hm_c(H, W, CV_32FC1, const_cast<float*>(hm_ptr + c * HW));
            cv::Mat dilated;
            cv::dilate(hm_c, dilated, kernel, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0.0));

            for (int y = 0; y < H; ++y) {
                const float* hrow = hm_c.ptr<float>(y);
                const float* drow = dilated.ptr<float>(y);
                for (int x = 0; x < W; ++x) {
                    const float sc = hrow[x];
                    if (sc < conf_thresh_ || sc < drow[x] - 1e-7f) continue;

                    const int idx = y * W + x;
                    const float l = rg_ptr[0 * HW + idx];
                    const float t = rg_ptr[1 * HW + idx];
                    const float r = rg_ptr[2 * HW + idx];
                    const float b = rg_ptr[3 * HW + idx];

                    const float cx = (static_cast<float>(x) + 0.5f) * stride;
                    const float cy = (static_cast<float>(y) + 0.5f) * stride;
                    const float x0 = (cx - l - static_cast<float>(left)) / scale;
                    const float y0 = (cy - t - static_cast<float>(top)) / scale;
                    const float x1 = (cx + r - static_cast<float>(left)) / scale;
                    const float y1 = (cy + b - static_cast<float>(top)) / scale;
                    append_decoded_box(x0, y0, x1, y1, sc, c,
                                       orig_w, orig_h, 4.0f,
                                       nms_boxes, raw_boxes, scores, class_ids);
                }
            }
        }
    }
}

std::vector<Detection> YoloDetector::detect(const cv::Mat& bgr) const {
    if (bgr.empty()) return {};

    cv::Mat src;
    if (use_gray_3ch_) {
        cv::Mat gray;
        if (bgr.channels() == 3) {
            cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = bgr;
        }
        cv::cvtColor(gray, src, cv::COLOR_GRAY2BGR);
    } else if (bgr.channels() == 1) {
        cv::cvtColor(bgr, src, cv::COLOR_GRAY2BGR);
    } else {
        src = bgr;
    }

    const int orig_w = src.cols;
    const int orig_h = src.rows;
    float scale = 1.0f;
    int left = 0;
    int top = 0;
    cv::Mat canvas;

    if (orig_w == input_w_ && orig_h == input_h_) {
        canvas = src;
    } else {
        scale = std::min(static_cast<float>(input_w_) / orig_w,
                         static_cast<float>(input_h_) / orig_h);
        const int nw = std::max(1, static_cast<int>(std::round(orig_w * scale)));
        const int nh = std::max(1, static_cast<int>(std::round(orig_h * scale)));

        const double pad_w = (input_w_ - nw) * 0.5;
        const double pad_h = (input_h_ - nh) * 0.5;
        top = std::max(0, static_cast<int>(std::round(pad_h - 0.1)));
        left = std::max(0, static_cast<int>(std::round(pad_w - 0.1)));
        const int bottom = std::max(0, input_h_ - nh - top);
        const int right = std::max(0, input_w_ - nw - left);

        cv::Mat resized;
        const int interp = (nw < orig_w || nh < orig_h) ? cv::INTER_AREA : cv::INTER_LINEAR;
        cv::resize(src, resized, cv::Size(nw, nh), 0, 0, interp);
        cv::copyMakeBorder(resized, canvas, top, bottom, left, right,
                           cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    }

    cv::Mat blob = cv::dnn::blobFromImage(
        canvas, 1.0 / 255.0, cv::Size(input_w_, input_h_), cv::Scalar(), true, false, CV_32F);
    net_.setInput(blob);

    std::vector<cv::Mat> outs;
    if (out_names_.empty()) {
        outs.push_back(net_.forward());
    } else {
        net_.forward(outs, out_names_);
    }

    std::vector<cv::Rect> nms_boxes;
    std::vector<cv::Rect2f> raw_boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    if (outs.size() == 1 && outs[0].dims == 3) {
        decode_yolov8(outs[0], orig_w, orig_h, scale, left, top,
                      nms_boxes, raw_boxes, scores, class_ids);
    } else {
        decode_ifl_net(outs, orig_w, orig_h, scale, left, top,
                       nms_boxes, raw_boxes, scores, class_ids);
    }

    std::vector<int> keep;
    if (!nms_boxes.empty()) {
        cv::dnn::NMSBoxes(nms_boxes, scores, conf_thresh_, nms_thresh_, keep);
    }

    std::vector<Detection> detections;
    detections.reserve(keep.size());
    for (int idx : keep) {
        Detection det;
        det.box = raw_boxes[idx];
        det.cls_id = class_ids[idx];
        det.cls_name = (det.cls_id >= 0 && det.cls_id < static_cast<int>(class_names_.size()))
                           ? class_names_[det.cls_id]
                           : std::to_string(det.cls_id);
        det.conf = scores[idx];
        detections.push_back(std::move(det));
    }
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) { return a.conf > b.conf; });
    if (detections.size() > 32) {
        detections.resize(32);
    }
    return detections;
}

}  // namespace tav_pose
