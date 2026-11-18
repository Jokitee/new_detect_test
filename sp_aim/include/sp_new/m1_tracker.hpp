#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include <chrono>
#include <list>
#include <string>
#include <vector>

#include "sp_new/pure_m1_bridge.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/target.hpp"
#include "tav_pose/types.hpp"

namespace sp_new {

/// Coordinate-transform & EKF adapter driving sp_vision_25::Target without IPPE corner coupling.
class PureM1Solver {
public:
    explicit PureM1Solver(const std::string& config_path);

    void set_R_gimbal2world(const Eigen::Quaterniond& q);
    Eigen::Matrix3d R_gimbal2world() const { return R_gimbal2world_; }

    /// Populate armor.{xyz_in_gimbal, xyz_in_world, ypr_in_gimbal, ypr_in_world, ypd_in_world, yaw_raw}
    /// from Pure-M1 6DoF camera-frame pose (or optional discrete reprojection yaw check).
    void solve(SolvedArmor& item, const tav_pose::CameraIntrinsics* active_K = nullptr) const;

    /// Project EKF target armor back onto image using the exact active (undistorted) camera intrinsics.
    std::vector<cv::Point2f> reproject_armor(const Eigen::Vector3d& xyz_in_world,
                                             double yaw,
                                             auto_aim::ArmorType type,
                                             auto_aim::ArmorName name,
                                             const tav_pose::CameraIntrinsics* active_K = nullptr) const;

    bool use_optimize_yaw() const { return use_optimize_yaw_; }
    void set_use_optimize_yaw(bool en) { use_optimize_yaw_ = en; }

private:
    auto_aim::Solver fallback_solver_;
    cv::Mat cam_matrix_;
    cv::Mat dist_coeffs_;
    bool undistorted_input_ = true;
    bool use_optimize_yaw_ = false;

    Eigen::Matrix3d R_gimbal2imubody_;
    Eigen::Matrix3d R_camera2gimbal_;
    Eigen::Vector3d t_camera2gimbal_;
    Eigen::Matrix3d R_gimbal2world_ = Eigen::Matrix3d::Identity();
};

/// Tracker driving sp_vision_25's exact 11-state Target EKF via PureM1Solver.
class PureM1Tracker {
public:
    PureM1Tracker(const std::string& config_path, PureM1Solver& solver);

    std::string state() const { return state_; }

    std::list<auto_aim::Target> track(std::list<SolvedArmor>& armors,
                                      std::chrono::steady_clock::time_point t,
                                      const cv::Size& img_size,
                                      bool use_enemy_color = true,
                                      const tav_pose::CameraIntrinsics* active_K = nullptr);

private:
    PureM1Solver& solver_;
    auto_aim::Color enemy_color_ = auto_aim::Color::red;
    int min_detect_count_ = 5;
    int max_temp_lost_count_ = 15;
    int outpost_max_temp_lost_count_ = 75;
    int normal_temp_lost_count_ = 15;

    int detect_count_ = 0;
    int temp_lost_count_ = 0;
    std::string state_ = "lost";
    auto_aim::Target target_;
    std::chrono::steady_clock::time_point last_timestamp_;

    void state_machine(bool found);
    bool set_target(std::list<SolvedArmor>& armors,
                    std::chrono::steady_clock::time_point t,
                    const tav_pose::CameraIntrinsics* active_K);
    bool update_target(std::list<SolvedArmor>& armors,
                       std::chrono::steady_clock::time_point t,
                       const tav_pose::CameraIntrinsics* active_K);
};

}  // namespace sp_new

