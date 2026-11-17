#include "tav_pose/yolo_detector.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tav_pose {

YoloDetector::YoloDetector(const std::string& onnx_path,
                           std::vector<std::string> class_names,
                           int imgsz,
                           float conf_thresh,
                           float nms_thresh,
                           bool use_gray_3ch)
    : class_names_(std::move(class_names)),
      imgsz_(imgsz),
      conf_thresh_(conf_thresh),
      nms_thresh_(nms_thresh),
      use_gray_3ch_(use_gray_3ch) {
    net_ = cv::dnn::readNetFromONNX(onnx_path);
    if (net_.empty()) {
        throw std::runtime_error("Failed to load ONNX model: " + onnx_path);
    }
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
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
    const float scale = std::min(static_cast<float>(imgsz_) / orig_w,
                                 static_cast<float>(imgsz_) / orig_h);
    const int nw = std::max(1, static_cast<int>(std::round(orig_w * scale)));
    const int nh = std::max(1, static_cast<int>(std::round(orig_h * scale)));

    const double pad_w = (imgsz_ - nw) * 0.5;
    const double pad_h = (imgsz_ - nh) * 0.5;
    const int top = std::max(0, static_cast<int>(std::round(pad_h - 0.1)));
    const int left = std::max(0, static_cast<int>(std::round(pad_w - 0.1)));
    const int bottom = std::max(0, imgsz_ - nh - top);
    const int right = std::max(0, imgsz_ - nw - left);

    cv::Mat resized, canvas;
    cv::resize(src, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
    cv::copyMakeBorder(resized, canvas, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    cv::Mat blob = cv::dnn::blobFromImage(
        canvas, 1.0 / 255.0, cv::Size(imgsz_, imgsz_), cv::Scalar(), true, false, CV_32F);
    net_.setInput(blob);
    cv::Mat out_raw = net_.forward();
    cv::Mat out = out_raw.isContinuous() ? out_raw : out_raw.clone();

    // Expected YOLOv8 detect output shape: (1, 4 + nc, num_anchors)
    if (out.dims != 3 || out.size[0] != 1 || out.size[1] < 5) {
        throw std::runtime_error("Unexpected ONNX output tensor shape");
    }
    const int channels = out.size[1];
    const int num_anchors = out.size[2];
    const int nc = channels - 4;
    const float* data = out.ptr<float>();

    std::vector<cv::Rect> nms_boxes;
    std::vector<cv::Rect2f> raw_boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    constexpr int kClassOffset = 7680;

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
        if (best_score < conf_thresh_) continue;

        const float cx = data[0 * num_anchors + i];
        const float cy = data[1 * num_anchors + i];
        const float bw = data[2 * num_anchors + i];
        const float bh = data[3 * num_anchors + i];

        float x0 = (cx - 0.5f * bw - static_cast<float>(left)) / scale;
        float y0 = (cy - 0.5f * bh - static_cast<float>(top)) / scale;
        float x1 = (cx + 0.5f * bw - static_cast<float>(left)) / scale;
        float y1 = (cy + 0.5f * bh - static_cast<float>(top)) / scale;

        x0 = std::clamp(x0, 0.0f, static_cast<float>(orig_w - 1));
        y0 = std::clamp(y0, 0.0f, static_cast<float>(orig_h - 1));
        x1 = std::clamp(x1, x0 + 1.0f, static_cast<float>(orig_w));
        y1 = std::clamp(y1, y0 + 1.0f, static_cast<float>(orig_h));

        raw_boxes.emplace_back(x0, y0, x1 - x0, y1 - y0);
        nms_boxes.emplace_back(
            static_cast<int>(std::round(x0)) + best_cls * kClassOffset,
            static_cast<int>(std::round(y0)) + best_cls * kClassOffset,
            std::max(1, static_cast<int>(std::round(x1 - x0))),
            std::max(1, static_cast<int>(std::round(y1 - y0))));
        scores.push_back(best_score);
        class_ids.push_back(best_cls);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(nms_boxes, scores, conf_thresh_, nms_thresh_, keep);

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
    return detections;
}

}  // namespace tav_pose
