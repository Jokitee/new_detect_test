#pragma once

#include "tav_pose/types.hpp"
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

namespace tav_pose {

class YoloDetector {
public:
    YoloDetector(const std::string& onnx_path,
                 std::vector<std::string> class_names,
                 int imgsz = 416,
                 float conf_thresh = 0.25f,
                 float nms_thresh = 0.45f,
                 bool use_gray_3ch = true);

    std::vector<Detection> detect(const cv::Mat& bgr) const;

private:
    mutable cv::dnn::Net net_;
    std::vector<std::string> class_names_;
    int imgsz_;
    float conf_thresh_;
    float nms_thresh_;
    bool use_gray_3ch_;
};

}  // namespace tav_pose

