#pragma once

#include "tav_pose/gpu_accel.hpp"
#include "tav_pose/types.hpp"
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

namespace tav_pose {

/// Unified ONNX detector supporting both YOLOv8 (3D tensor) and IFL-Net anchor-free (multi-scale 4D heads).
class YoloDetector {
public:
    YoloDetector(const std::string& onnx_path,
                 std::vector<std::string> class_names,
                 int input_w = 448,
                 int input_h = 336,
                 float conf_thresh = 0.45f,
                 float nms_thresh = 0.45f,
                 bool use_gray_3ch = true,
                 GpuMode gpu_mode = GpuMode::Auto);

    std::vector<Detection> detect(const cv::Mat& bgr) const;

    int input_width() const { return input_w_; }
    int input_height() const { return input_h_; }
    const DnnBackendStatus& backend_status() const { return backend_status_; }

private:
    void decode_yolov8(const cv::Mat& out_raw,
                       int orig_w,
                       int orig_h,
                       float scale,
                       int left,
                       int top,
                       std::vector<cv::Rect>& nms_boxes,
                       std::vector<cv::Rect2f>& raw_boxes,
                       std::vector<float>& scores,
                       std::vector<int>& class_ids) const;

    void decode_ifl_net(const std::vector<cv::Mat>& outs,
                        int orig_w,
                        int orig_h,
                        float scale,
                        int left,
                        int top,
                        std::vector<cv::Rect>& nms_boxes,
                        std::vector<cv::Rect2f>& raw_boxes,
                        std::vector<float>& scores,
                        std::vector<int>& class_ids) const;

    mutable cv::dnn::Net net_;
    std::vector<std::string> out_names_;
    std::vector<std::string> class_names_;
    int input_w_;
    int input_h_;
    float conf_thresh_;
    float nms_thresh_;
    bool use_gray_3ch_;
    DnnBackendStatus backend_status_;
};

}  // namespace tav_pose
