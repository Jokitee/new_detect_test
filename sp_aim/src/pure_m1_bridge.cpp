#include "sp_new/pure_m1_bridge.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <stdexcept>

#include "tools/logger.hpp"

namespace sp_new {
namespace fs = std::filesystem;

namespace {

constexpr double kLightbarLenM = 56e-3;
constexpr double kBigArmorWidthM = 230e-3;
constexpr double kSmallArmorWidthM = 135e-3;

// Orthogonal basis alignment from sp_vision_25 Armor frame
// (+X inward into target, +Y left, +Z up)
// to Pure-M1 template frame (+X right, +Y down, +Z optical forward):
//   [X_m1, Y_m1, Z_m1]^T = [-Y_armor, -Z_armor, +X_armor]^T
Eigen::Matrix3d make_R_armor_to_m1() {
    Eigen::Matrix3d R;
    R << 0.0, -1.0,  0.0,
         0.0,  0.0, -1.0,
         1.0,  0.0,  0.0;
    return R;
}

template <typename T>
T read_yaml_or(const YAML::Node& node, const char* key, const T& fallback) {
    if (node[key] && !node[key].IsNull()) {
        return node[key].as<T>();
    }
    return fallback;
}

bool has_distortion(const tav_pose::CameraIntrinsics& K) {
    return (K.k1 != 0.0 || K.k2 != 0.0 || K.p1 != 0.0 || K.p2 != 0.0 || K.k3 != 0.0);
}

std::vector<cv::Point2f> make_fallback_box_corners(const cv::Rect2f& b, int cols, int rows) {
    const float x0 = std::clamp(b.x, 0.0f, static_cast<float>(cols - 1));
    const float y0 = std::clamp(b.y, 0.0f, static_cast<float>(rows - 1));
    const float x1 = std::clamp(b.x + b.width, x0 + 1.0f, static_cast<float>(cols));
    const float y1 = std::clamp(b.y + b.height, y0 + 1.0f, static_cast<float>(rows));
    return {cv::Point2f(x0, y0), cv::Point2f(x1, y0), cv::Point2f(x1, y1), cv::Point2f(x0, y1)};
}

}  // namespace

std::string PureM1ArmorDetector::resolve_path(const std::string& path, const std::string& config_path) {
    if (path.empty()) return path;
    std::error_code ec;
    const fs::path p(path);
    if (p.is_absolute() && fs::exists(p, ec)) return p.string();
    if (fs::exists(p, ec)) return p.string();

    if (!config_path.empty()) {
        const fs::path cfg_dir = fs::path(config_path).parent_path();
        for (const auto& candidate : {cfg_dir / p, cfg_dir.parent_path() / p, cfg_dir.parent_path().parent_path() / p}) {
            if (fs::exists(candidate, ec)) return candidate.lexically_normal().string();
        }
    }
#ifdef SP_AIM_SOURCE_DIR
    {
        const fs::path c1 = fs::path(SP_AIM_SOURCE_DIR) / p;
        if (fs::exists(c1, ec)) return c1.lexically_normal().string();
    }
#endif
#ifdef NEW_DETECT_ROOT_DIR
    {
        const fs::path c2 = fs::path(NEW_DETECT_ROOT_DIR) / p;
        if (fs::exists(c2, ec)) return c2.lexically_normal().string();
    }
#endif
    return path;
}

PureM1ArmorDetector::PureM1ArmorDetector(const std::string& config_path, bool debug)
    : debug_(debug) {
    const YAML::Node yaml = YAML::LoadFile(config_path);

    const std::string enemy_str = read_yaml_or<std::string>(yaml, "enemy_color", "red");
    enemy_color_ = (enemy_str == "blue") ? auto_aim::Color::blue : auto_aim::Color::red;

    const std::string model_rel = read_yaml_or<std::string>(yaml, "ifl_model_path", "assets/ifl_net_v2_448x336.onnx");
    const std::string corner_rel = read_yaml_or<std::string>(yaml, "ifl_corner_model_path", "assets/roi_corner_v2_64x64.onnx");
    const std::string tpl_rel = read_yaml_or<std::string>(yaml, "ifl_templates_path", "assets/templates.csv");
    const int det_w = read_yaml_or<int>(yaml, "ifl_det_w", 448);
    const int det_h = read_yaml_or<int>(yaml, "ifl_det_h", 336);
    const float conf = read_yaml_or<float>(yaml, "ifl_conf", 0.45f);
    const float nms = read_yaml_or<float>(yaml, "ifl_nms", 0.45f);
    const bool use_gray = read_yaml_or<bool>(yaml, "ifl_use_gray", true);

    use_undistort_ = read_yaml_or<bool>(yaml, "ifl_use_undistort", true);
    m1_gate_enable_ = read_yaml_or<bool>(yaml, "m1_gate_enable", true);
    m1_min_edge_frac_ = read_yaml_or<double>(yaml, "m1_min_edge_frac", 0.22);
    m1_max_edge_med_px_ = read_yaml_or<double>(yaml, "m1_max_edge_med_px", 2.50);
    use_tracking_ = read_yaml_or<bool>(yaml, "m1_track", false);
    fast_1d_ = read_yaml_or<bool>(yaml, "m1_fast_1d", true);
    redetect_every_ = std::max(1, read_yaml_or<int>(yaml, "m1_redetect_every", 15));
    one_as_big_ = read_yaml_or<bool>(yaml, "ifl_one_as_big", true);
    color_mode_ = read_yaml_or<std::string>(yaml, "ifl_color_mode", "auto");
    fallback_to_enemy_color_ = read_yaml_or<bool>(yaml, "ifl_fallback_to_enemy_color", true);

    const std::string model_path = resolve_path(model_rel, config_path);
    const std::string tpl_path = resolve_path(tpl_rel, config_path);

    templates_ = tav_pose::load_templates_csv(tpl_path);
    std::vector<std::string> cls_names;
    cls_names.reserve(templates_.size());
    for (const auto& t : templates_) cls_names.push_back(t.name);
    if (cls_names.empty()) cls_names = {"1", "3", "sentary"};

    detector_ = std::make_unique<tav_pose::YoloDetector>(
        model_path, cls_names, det_w, det_h, conf, nms, use_gray);

    if (!corner_rel.empty()) {
        const std::string corner_path = resolve_path(corner_rel, config_path);
        std::error_code ec;
        if (fs::exists(fs::u8path(corner_path), ec)) {
            corner_prior_ = std::make_unique<tav_pose::RoiCornerPrior>(corner_path);
        }
    }

    if (yaml["camera_matrix"] && yaml["camera_matrix"].size() == 9) {
        const auto cv_vec = yaml["camera_matrix"].as<std::vector<double>>();
        K_base_.fx = cv_vec[0];
        K_base_.cx = cv_vec[2];
        K_base_.fy = cv_vec[4];
        K_base_.cy = cv_vec[5];
    }
    if (yaml["distort_coeffs"] && yaml["distort_coeffs"].size() >= 4) {
        const auto dv = yaml["distort_coeffs"].as<std::vector<double>>();
        K_base_.k1 = dv.size() > 0 ? dv[0] : 0.0;
        K_base_.k2 = dv.size() > 1 ? dv[1] : 0.0;
        K_base_.p1 = dv.size() > 2 ? dv[2] : 0.0;
        K_base_.p2 = dv.size() > 3 ? dv[3] : 0.0;
        K_base_.k3 = dv.size() > 4 ? dv[4] : 0.0;
    }
    K_last_ = K_base_;

    const std::string corner_tag =
        (corner_prior_ && !corner_prior_->empty())
            ? (std::to_string(corner_prior_->roi_sz()) + "x" + std::to_string(corner_prior_->roi_sz()) + "-hybrid")
            : "none";
    tools::logger()->info(
        "[PureM1Bridge] model='{}' corner='{}' conf={:.2f} templates={} K_base=({:.1f},{:.1f},{:.1f},{:.1f}) undistort={}",
        model_path, corner_tag,
        conf, templates_.size(), K_base_.fx, K_base_.fy, K_base_.cx, K_base_.cy,
        use_undistort_);
}

void PureM1ArmorDetector::ensure_undistort_maps(const cv::Size& sz, const tav_pose::CameraIntrinsics& K) {
    if (cached_map_size_ == sz &&
        std::abs(cached_fx_ - K.fx) < 1e-6 &&
        std::abs(cached_fy_ - K.fy) < 1e-6 &&
        std::abs(cached_cx_ - K.cx) < 1e-6 &&
        std::abs(cached_cy_ - K.cy) < 1e-6) {
        return;
    }
    const cv::Mat K_mat = (cv::Mat_<double>(3, 3) <<
        K.fx, 0.0, K.cx,
        0.0, K.fy, K.cy,
        0.0, 0.0, 1.0);
    const cv::Mat dist = (cv::Mat_<double>(1, 5) << K.k1, K.k2, K.p1, K.p2, K.k3);
    cv::initUndistortRectifyMap(K_mat, dist, cv::Mat(), K_mat, sz, CV_16SC2, map1_, map2_);
    cached_map_size_ = sz;
    cached_fx_ = K.fx;
    cached_fy_ = K.fy;
    cached_cx_ = K.cx;
    cached_cy_ = K.cy;
}

auto_aim::Color PureM1ArmorDetector::infer_lightbar_color(const cv::Mat& bgr, const cv::Rect2f& box) const {
    if (color_mode_ == "force_enemy") return enemy_color_;
    if (color_mode_ == "red") return auto_aim::Color::red;
    if (color_mode_ == "blue") return auto_aim::Color::blue;

    if (bgr.empty() || bgr.channels() < 3) {
        return fallback_to_enemy_color_ ? enemy_color_ : auto_aim::Color::extinguish;
    }

    const int W = bgr.cols;
    const int H = bgr.rows;
    const float bw = std::max(8.0f, box.width);
    const float bh = std::max(8.0f, box.height);
    const int y0 = std::clamp(static_cast<int>(std::floor(box.y + 0.08f * bh)), 0, H - 1);
    const int y1 = std::clamp(static_cast<int>(std::ceil(box.y + 0.92f * bh)), y0 + 1, H);

    const int lx0 = std::clamp(static_cast<int>(std::floor(box.x - 0.65f * bw)), 0, W);
    const int lx1 = std::clamp(static_cast<int>(std::ceil(box.x + 0.05f * bw)), 0, W);
    const int rx0 = std::clamp(static_cast<int>(std::floor(box.x + bw)), 0, W);
    const int rx1 = std::clamp(static_cast<int>(std::ceil(box.x + bw + 0.65f * bw)), 0, W);

    const std::array<cv::Rect, 2> rois = {
        cv::Rect(lx0, y0, std::max(0, lx1 - lx0), y1 - y0),
        cv::Rect(rx0, y0, std::max(0, rx1 - rx0), y1 - y0),
    };

    long long red_sum = 0;
    long long blue_sum = 0;
    int bright_cnt = 0;
    for (const auto& r : rois) {
        if (r.width <= 0 || r.height <= 0) continue;
        const cv::Mat patch = bgr(r);
        for (int py = 0; py < patch.rows; ++py) {
            const cv::Vec3b* row = patch.ptr<cv::Vec3b>(py);
            for (int px = 0; px < patch.cols; ++px) {
                const int b = row[px][0];
                const int g = row[px][1];
                const int rd = row[px][2];
                if (std::max({b, g, rd}) < 110) continue;
                if (std::abs(rd - b) < 12) continue;
                red_sum += rd;
                blue_sum += b;
                bright_cnt++;
            }
        }
    }

    if (bright_cnt >= 6) {
        if (blue_sum > red_sum + bright_cnt * 12LL) return auto_aim::Color::blue;
        if (red_sum > blue_sum + bright_cnt * 12LL) return auto_aim::Color::red;
    }
    return fallback_to_enemy_color_ ? enemy_color_ : auto_aim::Color::extinguish;
}

void PureM1ArmorDetector::map_class_to_armor(int cls_id,
                                               const std::string& cls_name,
                                               bool one_as_big,
                                               auto_aim::ArmorName& name_out,
                                               auto_aim::ArmorType& type_out,
                                               auto_aim::ArmorPriority& prio_out) {
    if (cls_name == "1" || cls_id == 0) {
        name_out = auto_aim::ArmorName::one;
        type_out = one_as_big ? auto_aim::ArmorType::big : auto_aim::ArmorType::small;
        prio_out = auto_aim::ArmorPriority::second;
    } else if (cls_name == "3" || cls_id == 1) {
        name_out = auto_aim::ArmorName::three;
        type_out = auto_aim::ArmorType::small;
        prio_out = auto_aim::ArmorPriority::first;
    } else {
        name_out = auto_aim::ArmorName::sentry;
        type_out = auto_aim::ArmorType::small;
        prio_out = auto_aim::ArmorPriority::third;
    }
}

std::vector<cv::Point2f> PureM1ArmorDetector::project_armor_corners(const Eigen::Vector3d& xyz_cam_m,
                                                                    const Eigen::Matrix3d& R_armor2cam,
                                                                    const tav_pose::CameraIntrinsics& K,
                                                                    auto_aim::ArmorType type) {
    const double hw = (type == auto_aim::ArmorType::big ? kBigArmorWidthM : kSmallArmorWidthM) * 0.5;
    const double hh = kLightbarLenM * 0.5;
    const std::vector<cv::Point3f> obj_pts = {
        {0.0f,  static_cast<float>(hw),  static_cast<float>(hh)},
        {0.0f, -static_cast<float>(hw),  static_cast<float>(hh)},
        {0.0f, -static_cast<float>(hw), -static_cast<float>(hh)},
        {0.0f,  static_cast<float>(hw), -static_cast<float>(hh)},
    };

    cv::Mat rmat_cv;
    cv::eigen2cv(R_armor2cam, rmat_cv);
    cv::Vec3d rvec, tvec(xyz_cam_m[0], xyz_cam_m[1], xyz_cam_m[2]);
    cv::Rodrigues(rmat_cv, rvec);

    const cv::Mat K_mat = (cv::Mat_<double>(3, 3) <<
        K.fx, 0.0, K.cx,
        0.0, K.fy, K.cy,
        0.0, 0.0, 1.0);
    const cv::Mat zero_dist = cv::Mat::zeros(1, 5, CV_64FC1);
    std::vector<cv::Point2f> img_pts;
    cv::projectPoints(obj_pts, rvec, tvec, K_mat, zero_dist, img_pts);
    return img_pts;
}

std::list<SolvedArmor> PureM1ArmorDetector::detect(const cv::Mat& raw_bgr,
                                                   const tav_pose::CameraIntrinsics* active_K,
                                                   BridgeDetectionStats* stats_out) {
    std::list<SolvedArmor> out;
    if (raw_bgr.empty()) return out;

    K_last_ = active_K ? *active_K : K_base_;

    cv::Mat bgr;
    if (use_undistort_ && has_distortion(K_last_) && K_last_.fx > 1.0 && K_last_.fy > 1.0) {
        ensure_undistort_maps(raw_bgr.size(), K_last_);
        cv::remap(raw_bgr, bgr, map1_, map2_, cv::INTER_LINEAR);
    } else {
        bgr = raw_bgr;
    }
    last_undistorted_bgr_ = bgr;

    cv::Mat gray;
    if (bgr.channels() >= 3) {
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = bgr;
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<tav_pose::Detection> dets;
    bool used_tracker = false;

    if (use_tracking_ && has_prev_track_ && frames_since_redetect_ < redetect_every_ &&
        prev_cls_id_ >= 0 && prev_cls_id_ < static_cast<int>(templates_.size())) {
        const auto m1_trk = tav_pose::SparseEdgePoseSolver::solve_tracking(
            gray, prev_rvec_, prev_tvec_, templates_[prev_cls_id_], K_last_, 16);
        if (m1_trk.valid && m1_trk.edge_frac >= 0.45 && m1_trk.edge_med_px <= 1.8) {
            used_tracker = true;
            frames_since_redetect_++;
            prev_rvec_ = m1_trk.rvec;
            prev_tvec_ = m1_trk.tvec;

            const auto pts = tav_pose::SparseEdgePoseSolver::project_points(
                templates_[prev_cls_id_], m1_trk.rvec, m1_trk.tvec, K_last_);
            float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
            for (const auto& p : pts) {
                x0 = std::min(x0, static_cast<float>(p.x));
                y0 = std::min(y0, static_cast<float>(p.y));
                x1 = std::max(x1, static_cast<float>(p.x));
                y1 = std::max(y1, static_cast<float>(p.y));
            }
            const cv::Rect2f trk_box(x0, y0, std::max(4.0f, x1 - x0), std::max(4.0f, y1 - y0));

            cv::Mat rmat_m1_cv;
            cv::Rodrigues(m1_trk.rvec, rmat_m1_cv);
            Eigen::Matrix3d R_m1_to_cam;
            cv::cv2eigen(rmat_m1_cv, R_m1_to_cam);
            const Eigen::Matrix3d R_armor2cam = R_m1_to_cam * make_R_armor_to_m1();
            const Eigen::Vector3d xyz_cam_m(
                m1_trk.tvec[0] * 1e-3, m1_trk.tvec[1] * 1e-3, m1_trk.tvec[2] * 1e-3);

            auto_aim::ArmorName aname;
            auto_aim::ArmorType atype;
            auto_aim::ArmorPriority aprio;
            map_class_to_armor(prev_cls_id_, templates_[prev_cls_id_].name, one_as_big_, aname, atype, aprio);
            auto pts4 = project_armor_corners(xyz_cam_m, R_armor2cam, K_last_, atype);
            if (pts4.size() != 4) pts4 = make_fallback_box_corners(trk_box, bgr.cols, bgr.rows);

            cv::Rect ibox(static_cast<int>(std::round(trk_box.x)),
                          static_cast<int>(std::round(trk_box.y)),
                          static_cast<int>(std::round(trk_box.width)),
                          static_cast<int>(std::round(trk_box.height)));
            auto_aim::Armor armor(0, 0.99f, ibox, pts4);
            armor.color = infer_lightbar_color(bgr, trk_box);
            armor.name = aname;
            armor.type = atype;
            armor.priority = aprio;
            armor.duplicated = false;
            armor.center_norm = cv::Point2f(armor.center.x / bgr.cols, armor.center.y / bgr.rows);

            if (stats_out) {
                stats_out->raw_yolo_count = 1;
                stats_out->m1_accepted_count = 1;
                stats_out->m1_rejected_count = 0;
                stats_out->det_ms = 0.0;
                stats_out->m1_ms = m1_trk.solve_ms;
            }
            out.push_back(SolvedArmor{armor, m1_trk, xyz_cam_m, R_armor2cam, true});
            return out;
        }
        has_prev_track_ = false;
    }

    if (!used_tracker) {
        dets = detector_->detect(gray);
        frames_since_redetect_ = 0;
    }
    const auto t_det = std::chrono::steady_clock::now();

    std::vector<tav_pose::PoseResult> poses(dets.size());
    const auto t_m1_0 = std::chrono::steady_clock::now();
    if (dets.size() == 1) {
        const auto& d = dets[0];
        if (d.cls_id >= 0 && d.cls_id < static_cast<int>(templates_.size())) {
            poses[0] = tav_pose::SparseEdgePoseSolver::solve_cold_start_hybrid(
                gray, d.box, templates_[d.cls_id], K_last_, corner_prior_.get(), 14, fast_1d_);
        }
    } else if (dets.size() > 1) {
        cv::parallel_for_(cv::Range(0, static_cast<int>(dets.size())), [&](const cv::Range& range) {
            for (int i = range.start; i < range.end; ++i) {
                const auto& d = dets[i];
                if (d.cls_id >= 0 && d.cls_id < static_cast<int>(templates_.size())) {
                    poses[i] = tav_pose::SparseEdgePoseSolver::solve_cold_start_hybrid(
                        gray, d.box, templates_[d.cls_id], K_last_, corner_prior_.get(), 14, fast_1d_);
                }
            }
        });
    }
    const auto t_m1_1 = std::chrono::steady_clock::now();

    const Eigen::Matrix3d R_armor_to_m1 = make_R_armor_to_m1();
    int accepted = 0;
    int rejected = 0;

    for (size_t i = 0; i < dets.size(); ++i) {
        const auto& det = dets[i];
        const auto& m1 = poses[i];

        const bool gate_pass = m1.valid &&
                               (!m1_gate_enable_ ||
                                (m1.edge_frac >= m1_min_edge_frac_ && m1.edge_med_px <= m1_max_edge_med_px_));
        if (!gate_pass) {
            rejected++;
            continue;
        }
        accepted++;

        cv::Mat rmat_m1_cv;
        cv::Rodrigues(m1.rvec, rmat_m1_cv);
        Eigen::Matrix3d R_m1_to_cam;
        cv::cv2eigen(rmat_m1_cv, R_m1_to_cam);
        const Eigen::Matrix3d R_armor2cam = R_m1_to_cam * R_armor_to_m1;
        const Eigen::Vector3d xyz_cam_m(
            m1.tvec[0] * 1e-3, m1.tvec[1] * 1e-3, m1.tvec[2] * 1e-3);

        auto_aim::ArmorName aname;
        auto_aim::ArmorType atype;
        auto_aim::ArmorPriority aprio;
        map_class_to_armor(det.cls_id, det.cls_name, one_as_big_, aname, atype, aprio);

        auto pts4 = project_armor_corners(xyz_cam_m, R_armor2cam, K_last_, atype);
        if (pts4.size() != 4) pts4 = make_fallback_box_corners(det.box, bgr.cols, bgr.rows);

        cv::Rect ibox(static_cast<int>(std::round(det.box.x)),
                      static_cast<int>(std::round(det.box.y)),
                      static_cast<int>(std::round(det.box.width)),
                      static_cast<int>(std::round(det.box.height)));
        auto_aim::Armor armor(0, det.conf, ibox, pts4);
        armor.color = infer_lightbar_color(bgr, det.box);
        armor.name = aname;
        armor.type = atype;
        armor.priority = aprio;
        armor.duplicated = false;
        armor.center_norm = cv::Point2f(armor.center.x / bgr.cols, armor.center.y / bgr.rows);

        out.push_back(SolvedArmor{armor, m1, xyz_cam_m, R_armor2cam, true});
    }

    if (use_tracking_ && dets.size() == 1 && out.size() == 1) {
        has_prev_track_ = true;
        prev_cls_id_ = dets[0].cls_id;
        prev_rvec_ = poses[0].rvec;
        prev_tvec_ = poses[0].tvec;
    } else {
        has_prev_track_ = false;
    }

    if (stats_out) {
        stats_out->raw_yolo_count = static_cast<int>(dets.size());
        stats_out->m1_accepted_count = accepted;
        stats_out->m1_rejected_count = rejected;
        stats_out->det_ms = std::chrono::duration<double, std::milli>(t_det - t0).count();
        stats_out->m1_ms = std::chrono::duration<double, std::milli>(t_m1_1 - t_m1_0).count();
    }
    return out;
}

}  // namespace sp_new

