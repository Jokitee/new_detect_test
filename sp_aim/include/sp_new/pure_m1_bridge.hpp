#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include <list>
#include <memory>
#include <string>
#include <vector>

#include "tasks/auto_aim/armor.hpp"
#include "tav_pose/sparse_edge_pose.hpp"
#include "tav_pose/types.hpp"
#include "tav_pose/yolo_detector.hpp"

namespace sp_new {

struct SolvedArmor {
    auto_aim::Armor armor;
    tav_pose::PoseResult m1_pose;
    Eigen::Vector3d xyz_in_camera_m{0.0, 0.0, 0.0};
    Eigen::Matrix3d R_armor2camera = Eigen::Matrix3d::Identity();
    bool has_m1_pose = false;
};

struct BridgeDetectionStats {
    int raw_yolo_count = 0;
    int m1_accepted_count = 0;
    int m1_rejected_count = 0;
    double det_ms = 0.0;
    double m1_ms = 0.0;
};

/// Detector replacement using TAV IFL-Net V2-Slim (cv::dnn) + Pure-M1 6DoF sparse edge solver.
/// Maps detections into sp_vision_25::auto_aim::Armor + camera-frame 6DoF pose for EKF tracking.
class PureM1ArmorDetector {
public:
    explicit PureM1ArmorDetector(const std::string& config_path, bool debug = false);

    /// Apply undistortion (from current active intrinsics) + YOLO detect + Pure-M1 solve.
    std::list<SolvedArmor> detect(const cv::Mat& raw_bgr,
                                  const tav_pose::CameraIntrinsics* active_K = nullptr,
                                  BridgeDetectionStats* stats_out = nullptr);

    static std::string resolve_path(const std::string& path, const std::string& config_path);

    const tav_pose::CameraIntrinsics& base_intrinsics() const { return K_base_; }
    const tav_pose::CameraIntrinsics& last_intrinsics() const { return K_last_; }
    const cv::Mat& undistorted_last() const { return last_undistorted_bgr_; }
    const std::vector<tav_pose::CharTemplate>& templates() const { return templates_; }
    bool use_undistort() const { return use_undistort_; }

    bool tracking_enabled() const { return use_tracking_; }
    void set_tracking_enabled(bool en) { use_tracking_ = en; }
    void reset_tracking() { has_prev_track_ = false; }

private:
    bool debug_ = false;
    bool use_undistort_ = true;
    bool m1_gate_enable_ = true;
    double m1_min_edge_frac_ = 0.22;
    double m1_max_edge_med_px_ = 2.50;
    bool use_tracking_ = false;
    int redetect_every_ = 15;
    bool one_as_big_ = true;
    std::string color_mode_ = "auto";
    bool fallback_to_enemy_color_ = true;
    auto_aim::Color enemy_color_ = auto_aim::Color::red;

    tav_pose::CameraIntrinsics K_base_;
    tav_pose::CameraIntrinsics K_last_;

    cv::Size cached_map_size_{0, 0};
    double cached_fx_ = 0.0;
    double cached_fy_ = 0.0;
    double cached_cx_ = 0.0;
    double cached_cy_ = 0.0;
    cv::Mat map1_, map2_;
    cv::Mat last_undistorted_bgr_;

    std::unique_ptr<tav_pose::YoloDetector> detector_;
    std::vector<tav_pose::CharTemplate> templates_;

    bool has_prev_track_ = false;
    int prev_cls_id_ = -1;
    cv::Vec3d prev_rvec_{0.0, 0.0, 0.0};
    cv::Vec3d prev_tvec_{0.0, 0.0, 0.0};
    int frames_since_redetect_ = 0;

    void ensure_undistort_maps(const cv::Size& sz, const tav_pose::CameraIntrinsics& K);
    auto_aim::Color infer_lightbar_color(const cv::Mat& bgr, const cv::Rect2f& box) const;
    static void map_class_to_armor(int cls_id,
                                   const std::string& cls_name,
                                   bool one_as_big,
                                   auto_aim::ArmorName& name_out,
                                   auto_aim::ArmorType& type_out,
                                   auto_aim::ArmorPriority& prio_out);
    static std::vector<cv::Point2f> project_armor_corners(const Eigen::Vector3d& xyz_cam_m,
                                                          const Eigen::Matrix3d& R_armor2cam,
                                                          const tav_pose::CameraIntrinsics& K,
                                                          auto_aim::ArmorType type);
};

}  // namespace sp_new

