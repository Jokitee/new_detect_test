#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace tav_pose {

struct CameraIntrinsics {
    double fx = 1400.0;
    double fy = 1400.0;
    double cx = 320.0;
    double cy = 240.0;
};

struct ModelPoint {
    double px = 0.0;
    double py = 0.0;
    double tx = 0.0;
    double ty = 0.0;
    double nx = 0.0;
    double ny = 0.0;
};

struct CharTemplate {
    int cls_id = 0;
    std::string name;
    double width_mm = 62.7;
    double height_mm = 102.0;
    std::vector<ModelPoint> points;
};

struct Detection {
    cv::Rect2f box;
    int cls_id = 0;
    std::string cls_name;
    float conf = 0.0f;
};

struct PoseResult {
    bool valid = false;
    cv::Vec3d rvec{0.0, 0.0, 0.0};
    cv::Vec3d tvec{0.0, 0.0, 0.0};
    cv::Vec3d euler_yxz_deg{0.0, 0.0, 0.0};  // (yaw, pitch, roll) in degrees
    double edge_med_px = 0.0;
    double edge_frac = 0.0;
    double phot_contrast = 0.0;
    double mirror_gap = 0.0;
    double solve_ms = 0.0;
};

std::vector<CharTemplate> load_templates_csv(const std::string& csv_path);

}  // namespace tav_pose

