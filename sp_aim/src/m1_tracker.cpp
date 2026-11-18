#include "sp_new/m1_tracker.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <numeric>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace sp_new {
namespace {

constexpr double kLightbarLenM = 56e-3;
constexpr double kBigArmorWidthM = 230e-3;
constexpr double kSmallArmorWidthM = 135e-3;

template <typename T>
T read_yaml_or(const YAML::Node& node, const char* key, const T& fallback) {
    if (node[key] && !node[key].IsNull()) {
        return node[key].as<T>();
    }
    return fallback;
}

}  // namespace

PureM1Solver::PureM1Solver(const std::string& config_path)
    : fallback_solver_(config_path) {
    const YAML::Node yaml = YAML::LoadFile(config_path);

    auto R_g2i = yaml["R_gimbal2imubody"].as<std::vector<double>>();
    auto R_c2g = yaml["R_camera2gimbal"].as<std::vector<double>>();
    auto t_c2g = yaml["t_camera2gimbal"].as<std::vector<double>>();
    R_gimbal2imubody_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_g2i.data());
    R_camera2gimbal_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_c2g.data());
    t_camera2gimbal_ = Eigen::Matrix<double, 3, 1>(t_c2g.data());

    auto cam_vec = yaml["camera_matrix"].as<std::vector<double>>();
    auto dist_vec = yaml["distort_coeffs"].as<std::vector<double>>();
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K(cam_vec.data());
    cv::eigen2cv(K, cam_matrix_);
    dist_coeffs_ = cv::Mat(1, static_cast<int>(dist_vec.size()), CV_64FC1);
    for (size_t i = 0; i < dist_vec.size(); ++i) {
        dist_coeffs_.at<double>(0, static_cast<int>(i)) = dist_vec[i];
    }

    undistorted_input_ = read_yaml_or<bool>(yaml, "ifl_use_undistort", true);
    use_optimize_yaw_ = read_yaml_or<bool>(yaml, "use_optimize_yaw", false);
}

void PureM1Solver::set_R_gimbal2world(const Eigen::Quaterniond& q) {
    fallback_solver_.set_R_gimbal2world(q);
    const Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
    R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
}

void PureM1Solver::solve(SolvedArmor& item, const tav_pose::CameraIntrinsics* active_K) const {
    if (!item.has_m1_pose) {
        fallback_solver_.solve(item.armor);
        return;
    }

    item.armor.xyz_in_gimbal = R_camera2gimbal_ * item.xyz_in_camera_m + t_camera2gimbal_;
    item.armor.xyz_in_world = R_gimbal2world_ * item.armor.xyz_in_gimbal;

    const Eigen::Matrix3d R_armor2gimbal = R_camera2gimbal_ * item.R_armor2camera;
    const Eigen::Matrix3d R_armor2world = R_gimbal2world_ * R_armor2gimbal;
    item.armor.ypr_in_gimbal = tools::eulers(R_armor2gimbal, 2, 1, 0);
    item.armor.ypr_in_world = tools::eulers(R_armor2world, 2, 1, 0);
    item.armor.ypd_in_world = tools::xyz2ypd(item.armor.xyz_in_world);
    item.armor.yaw_raw = item.armor.ypr_in_world[0];

    if (use_optimize_yaw_ && item.armor.points.size() == 4) {
        const Eigen::Vector3d gimbal_ypr = tools::eulers(R_gimbal2world_, 2, 1, 0);
        constexpr int kSearchDeg = 140;
        const double yaw0 = tools::limit_rad(gimbal_ypr[0] - 0.5 * kSearchDeg * CV_PI / 180.0);
        double min_err = 1e12;
        double best_yaw = item.armor.ypr_in_world[0];
        for (int i = 0; i < kSearchDeg; ++i) {
            const double yaw = tools::limit_rad(yaw0 + i * CV_PI / 180.0);
            const auto proj = reproject_armor(
                item.armor.xyz_in_world, yaw, item.armor.type, item.armor.name, active_K);
            if (proj.size() != 4) continue;
            double err = 0.0;
            for (int k = 0; k < 4; ++k) err += cv::norm(item.armor.points[k] - proj[k]);
            if (err < min_err) {
                min_err = err;
                best_yaw = yaw;
            }
        }
        item.armor.ypr_in_world[0] = best_yaw;
    }
}

std::vector<cv::Point2f> PureM1Solver::reproject_armor(const Eigen::Vector3d& xyz_in_world,
                                                       double yaw,
                                                       auto_aim::ArmorType type,
                                                       auto_aim::ArmorName name,
                                                       const tav_pose::CameraIntrinsics* active_K) const {
    if (!undistorted_input_ && !active_K) {
        return fallback_solver_.reproject_armor(xyz_in_world, yaw, type, name);
    }

    const double sin_yaw = std::sin(yaw);
    const double cos_yaw = std::cos(yaw);
    const double pitch = (name == auto_aim::ArmorName::outpost) ? -15.0 * CV_PI / 180.0 : 15.0 * CV_PI / 180.0;
    const double sin_pitch = std::sin(pitch);
    const double cos_pitch = std::cos(pitch);

    Eigen::Matrix3d R_armor2world;
    R_armor2world << cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch,
                     sin_yaw * cos_pitch,  cos_yaw, sin_yaw * sin_pitch,
                     -sin_pitch,           0.0,     cos_pitch;

    const Eigen::Matrix3d R_armor2camera =
        R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * R_armor2world;
    const Eigen::Vector3d t_armor2camera =
        R_camera2gimbal_.transpose() * (R_gimbal2world_.transpose() * xyz_in_world - t_camera2gimbal_);

    cv::Vec3d rvec;
    cv::Mat R_cv;
    cv::eigen2cv(R_armor2camera, R_cv);
    cv::Rodrigues(R_cv, rvec);
    const cv::Vec3d tvec(t_armor2camera[0], t_armor2camera[1], t_armor2camera[2]);

    const double hw = (type == auto_aim::ArmorType::big ? kBigArmorWidthM : kSmallArmorWidthM) * 0.5;
    const double hh = kLightbarLenM * 0.5;
    const std::vector<cv::Point3f> obj_pts = {
        {0.0f,  static_cast<float>(hw),  static_cast<float>(hh)},
        {0.0f, -static_cast<float>(hw),  static_cast<float>(hh)},
        {0.0f, -static_cast<float>(hw), -static_cast<float>(hh)},
        {0.0f,  static_cast<float>(hw), -static_cast<float>(hh)},
    };

    cv::Mat K_mat = cam_matrix_;
    if (active_K && active_K->fx > 1.0 && active_K->fy > 1.0) {
        K_mat = (cv::Mat_<double>(3, 3) <<
            active_K->fx, 0.0, active_K->cx,
            0.0, active_K->fy, active_K->cy,
            0.0, 0.0, 1.0);
    }
    const cv::Mat dist = undistorted_input_ ? cv::Mat::zeros(1, 5, CV_64FC1) : dist_coeffs_;
    std::vector<cv::Point2f> img_pts;
    cv::projectPoints(obj_pts, rvec, tvec, K_mat, dist, img_pts);
    return img_pts;
}

PureM1Tracker::PureM1Tracker(const std::string& config_path, PureM1Solver& solver)
    : solver_(solver), last_timestamp_(std::chrono::steady_clock::now()) {
    const YAML::Node yaml = YAML::LoadFile(config_path);
    const std::string ec = read_yaml_or<std::string>(yaml, "enemy_color", "red");
    enemy_color_ = (ec == "blue") ? auto_aim::Color::blue : auto_aim::Color::red;
    min_detect_count_ = read_yaml_or<int>(yaml, "min_detect_count", 5);
    max_temp_lost_count_ = read_yaml_or<int>(yaml, "max_temp_lost_count", 15);
    outpost_max_temp_lost_count_ = read_yaml_or<int>(yaml, "outpost_max_temp_lost_count", 75);
    normal_temp_lost_count_ = max_temp_lost_count_;
}

std::list<auto_aim::Target> PureM1Tracker::track(std::list<SolvedArmor>& armors,
                                                 std::chrono::steady_clock::time_point t,
                                                 const cv::Size& img_size,
                                                 bool use_enemy_color,
                                                 const tav_pose::CameraIntrinsics* active_K) {
    const double dt = tools::delta_time(t, last_timestamp_);
    last_timestamp_ = t;

    if (state_ != "lost" && dt > 0.1) {
        tools::logger()->warn("[PureM1Tracker] Large dt: {:.3f}s -> reset to lost", dt);
        state_ = "lost";
    }

    if (use_enemy_color) {
        armors.remove_if([&](const SolvedArmor& a) { return a.armor.color != enemy_color_; });
    }

    const cv::Point2f img_center(
        static_cast<float>(img_size.width) * 0.5f,
        static_cast<float>(img_size.height) * 0.5f);

    armors.sort([&](const SolvedArmor& a, const SolvedArmor& b) {
        return cv::norm(a.armor.center - img_center) < cv::norm(b.armor.center - img_center);
    });
    armors.sort([](const SolvedArmor& a, const SolvedArmor& b) {
        return a.armor.priority < b.armor.priority;
    });

    bool found = false;
    if (state_ == "lost") {
        found = set_target(armors, t, active_K);
    } else {
        found = update_target(armors, t, active_K);
    }

    state_machine(found);

    if (state_ != "lost" && target_.diverged()) {
        tools::logger()->debug("[PureM1Tracker] Target diverged!");
        state_ = "lost";
        return {};
    }

    if (state_ != "lost" &&
        std::accumulate(target_.ekf().recent_nis_failures.begin(),
                        target_.ekf().recent_nis_failures.end(), 0) >=
            static_cast<int>(0.4 * target_.ekf().window_size)) {
        tools::logger()->debug("[PureM1Tracker] Bad Converge Found!");
        state_ = "lost";
        return {};
    }

    if (state_ == "lost") return {};
    return {target_};
}

void PureM1Tracker::state_machine(bool found) {
    if (state_ == "lost") {
        if (!found) return;
        state_ = "detecting";
        detect_count_ = 1;
    } else if (state_ == "detecting") {
        if (found) {
            detect_count_++;
            if (detect_count_ >= min_detect_count_) state_ = "tracking";
        } else {
            detect_count_ = 0;
            state_ = "lost";
        }
    } else if (state_ == "tracking") {
        if (found) return;
        temp_lost_count_ = 1;
        state_ = "temp_lost";
    } else if (state_ == "temp_lost") {
        if (found) {
            state_ = "tracking";
        } else {
            temp_lost_count_++;
            max_temp_lost_count_ = (target_.name == auto_aim::ArmorName::outpost)
                                       ? outpost_max_temp_lost_count_
                                       : normal_temp_lost_count_;
            if (temp_lost_count_ > max_temp_lost_count_) state_ = "lost";
        }
    }
}

bool PureM1Tracker::set_target(std::list<SolvedArmor>& armors,
                               std::chrono::steady_clock::time_point t,
                               const tav_pose::CameraIntrinsics* active_K) {
    if (armors.empty()) return false;

    auto& item = armors.front();
    solver_.solve(item, active_K);

    const bool is_balance =
        (item.armor.type == auto_aim::ArmorType::big) &&
        (item.armor.name == auto_aim::ArmorName::three ||
         item.armor.name == auto_aim::ArmorName::four ||
         item.armor.name == auto_aim::ArmorName::five);

    if (is_balance) {
        Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
        target_ = auto_aim::Target(item.armor, t, 0.2, 2, P0_dig);
    } else if (item.armor.name == auto_aim::ArmorName::outpost) {
        Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0, 0}};
        target_ = auto_aim::Target(item.armor, t, 0.2765, 3, P0_dig);
    } else if (item.armor.name == auto_aim::ArmorName::base) {
        Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0}};
        target_ = auto_aim::Target(item.armor, t, 0.3205, 3, P0_dig);
    } else {
        Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
        target_ = auto_aim::Target(item.armor, t, 0.2, 4, P0_dig);
    }
    return true;
}

bool PureM1Tracker::update_target(std::list<SolvedArmor>& armors,
                                  std::chrono::steady_clock::time_point t,
                                  const tav_pose::CameraIntrinsics* active_K) {
    target_.predict(t);

    int found_count = 0;
    for (const auto& item : armors) {
        if (item.armor.name != target_.name || item.armor.type != target_.armor_type) continue;
        found_count++;
    }
    if (found_count == 0) return false;

    for (auto& item : armors) {
        if (item.armor.name != target_.name || item.armor.type != target_.armor_type) continue;
        solver_.solve(item, active_K);
        target_.update(item.armor);
    }
    return true;
}

}  // namespace sp_new

