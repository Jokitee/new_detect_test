#include <fmt/core.h>
#include <opencv2/opencv.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>

#include "sp_new/m1_tracker.hpp"
#include "sp_new/pure_m1_bridge.hpp"
#include "sp_new/vis_helper.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tav_pose/mv_camera.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"

namespace {

volatile std::sig_atomic_t g_stop_requested = 0;

void handle_signal(int) { g_stop_requested = 1; }

const std::string kKeys =
    "{help h usage ?   |                           | Show usage }"
    "{config c         | sp_aim/configs/sp_ekf_test.yaml | YAML config path }"
    "{camera           | false                     | Live MindVision camera + sp_vision EKF test }"
    "{video v          |                           | Offline video file path (.avi/.mp4) }"
    "{imu-txt t        |                           | Optional IMU quaternion log (.txt: t w x y z) }"
    "{out o            |                           | Optional output video path (.avi) }"
    "{exposure         | 8000.0                    | MindVision camera exposure (us) }"
    "{width            | 640                       | Requested camera width (0=full) }"
    "{height           | 480                       | Requested camera height (0=full) }"
    "{no-downsample    | false                     | Disable 448x336 INTER_AREA camera downsample }"
    "{start-index s    | 0                         | Video start frame index }"
    "{end-index e      | 0                         | Video end frame index (0=all) }"
    "{no-color-filter  | false                     | Disable red/blue filtering (for printed B/W targets) }"
    "{optimize-yaw     | false                     | Compare discrete Solver::optimize_yaw vs direct Pure-M1 yaw }"
    "{planner          | false                     | Run migrated TinyMPC Planner alongside Aimer }"
    "{track            | false                     | Enable temporal Pure-M1 keyframe tracking }"
    "{no-show          | false                     | Disable cv::imshow window }";

tav_pose::CameraIntrinsics scale_camera_intrinsics(const tav_pose::CameraIntrinsics& base,
                                                   const tav_pose::MvResolutionInfo& mv_res,
                                                   const cv::Size& proc_size) {
    tav_pose::CameraIntrinsics K = base;
    if (mv_res.valid && mv_res.fov_width > 0 && mv_res.fov_height > 0 &&
        mv_res.out_width > 0 && mv_res.out_height > 0) {
        const double sx_hw = static_cast<double>(mv_res.out_width) / mv_res.fov_width;
        const double sy_hw = static_cast<double>(mv_res.out_height) / mv_res.fov_height;
        K.fx *= sx_hw;
        K.fy *= sy_hw;
        K.cx = (K.cx - mv_res.h_offset) * sx_hw;
        K.cy = (K.cy - mv_res.v_offset) * sy_hw;

        if (proc_size.width > 0 && proc_size.height > 0 &&
            (proc_size.width != mv_res.out_width || proc_size.height != mv_res.out_height)) {
            const double sx_sw = static_cast<double>(proc_size.width) / mv_res.out_width;
            const double sy_sw = static_cast<double>(proc_size.height) / mv_res.out_height;
            K.fx *= sx_sw;
            K.fy *= sy_sw;
            K.cx *= sx_sw;
            K.cy *= sy_sw;
        }
    }
    return K;
}

}  // namespace

int main(int argc, char* argv[]) {
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);
    std::filesystem::create_directories("logs");

    cv::CommandLineParser cli(argc, argv, kKeys);
    if (cli.has("help")) {
        cli.printMessage();
        return 0;
    }

    const std::string cfg_raw = cli.get<std::string>("config");
    const std::string config_path = sp_new::PureM1ArmorDetector::resolve_path(cfg_raw, "");
    const std::string video_path_raw = cli.get<std::string>("video");
    const bool use_camera = cli.get<bool>("camera") || video_path_raw.empty();
    const std::string video_path = video_path_raw.empty()
                                       ? ""
                                       : sp_new::PureM1ArmorDetector::resolve_path(video_path_raw, config_path);

    std::string imu_path = cli.get<std::string>("imu-txt");
    if (!imu_path.empty()) {
        imu_path = sp_new::PureM1ArmorDetector::resolve_path(imu_path, config_path);
    }

    const std::string out_path = cli.get<std::string>("out");
    const double exposure_us = cli.get<double>("exposure");
    const int req_w = cli.get<int>("width");
    const int req_h = cli.get<int>("height");
    const bool downsample = !cli.get<bool>("no-downsample");
    const int start_index = cli.get<int>("start-index");
    const int end_index = cli.get<int>("end-index");
    const bool use_enemy_color = !cli.get<bool>("no-color-filter");
    const bool optimize_yaw = cli.get<bool>("optimize-yaw");
    const bool use_planner = cli.get<bool>("planner");
    const bool m1_track = cli.get<bool>("track");
    const bool no_show = cli.get<bool>("no-show");

    std::unique_ptr<tav_pose::MvCamera> mv_cam;
    cv::VideoCapture video_cap;
    std::ifstream imu_stream;
    bool has_imu = false;
    double stream_fps = 30.0;

    if (use_camera) {
        if (!tav_pose::MvCamera::sdk_available()) {
            tools::logger()->error("MVSDK not enabled in this build and no --video specified.");
            return 1;
        }
        mv_cam = std::make_unique<tav_pose::MvCamera>();
        mv_cam->open(exposure_us, req_w, req_h);
    } else {
        video_cap.open(video_path);
        if (!video_cap.isOpened()) {
            tools::logger()->error("Failed to open video: {}", video_path);
            return 1;
        }
        stream_fps = video_cap.get(cv::CAP_PROP_FPS);
        if (!std::isfinite(stream_fps) || stream_fps <= 1.0) stream_fps = 30.0;
        if (!imu_path.empty()) {
            imu_stream.open(imu_path);
            has_imu = imu_stream.is_open();
        }
        if (start_index > 0) {
            video_cap.set(cv::CAP_PROP_POS_FRAMES, start_index);
            if (has_imu) {
                double t, w, x, y, z;
                for (int i = 0; i < start_index; ++i) imu_stream >> t >> w >> x >> y >> z;
            }
        }
    }

    tools::Plotter plotter;
    sp_new::PureM1ArmorDetector detector(config_path, false);
    if (m1_track) detector.set_tracking_enabled(true);

    sp_new::PureM1Solver solver(config_path);
    if (optimize_yaw) solver.set_use_optimize_yaw(true);

    sp_new::PureM1Tracker tracker(config_path, solver);
    auto_aim::Aimer aimer(config_path);
    auto_aim::Shooter shooter(config_path);

    std::unique_ptr<auto_aim::Planner> planner;
    if (use_planner) {
        planner = std::make_unique<auto_aim::Planner>(config_path);
    }

    cv::VideoWriter writer;
    const auto t0 = std::chrono::steady_clock::now();
    cv::Mat frame, proc_frame;
    int cam_timeouts = 0;

    for (int frame_idx = use_camera ? 0 : start_index; !g_stop_requested; ++frame_idx) {
        if (!use_camera && end_index > 0 && frame_idx > end_index) break;

        tav_pose::MvResolutionInfo mv_res{};
        double t_sec = static_cast<double>(frame_idx) / stream_fps;
        Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
        std::chrono::steady_clock::time_point timestamp;

        if (use_camera) {
            if (!mv_cam->grab(frame, 80, &mv_res) || frame.empty()) {
                cam_timeouts++;
                if (cam_timeouts % 20 == 1) {
                    tools::logger()->warn("[sp_ekf_test] Waiting for camera frames (timeouts={})", cam_timeouts);
                }
                continue;
            }
            timestamp = std::chrono::steady_clock::now();
        } else {
            if (!video_cap.read(frame) || frame.empty()) break;
            if (has_imu) {
                double w, x, y, z;
                if (imu_stream >> t_sec >> w >> x >> y >> z) {
                    q = Eigen::Quaterniond(w, x, y, z).normalized();
                }
            }
            timestamp = t0 + std::chrono::microseconds(static_cast<long long>(t_sec * 1e6));
        }

        const tav_pose::CameraIntrinsics* active_K_ptr = nullptr;
        tav_pose::CameraIntrinsics scaled_K;
        if (use_camera) {
            if (downsample && (frame.cols != 448 || frame.rows != 336)) {
                cv::resize(frame, proc_frame, cv::Size(448, 336), 0.0, 0.0, cv::INTER_AREA);
            } else {
                proc_frame = frame;
            }
            scaled_K = scale_camera_intrinsics(detector.base_intrinsics(), mv_res, proc_frame.size());
            active_K_ptr = &scaled_K;
        } else {
            proc_frame = frame;
        }

        const auto loop_t0 = std::chrono::steady_clock::now();
        solver.set_R_gimbal2world(q);

        sp_new::BridgeDetectionStats st;
        auto armors = detector.detect(proc_frame, active_K_ptr, &st);
        const cv::Mat& vis_base = detector.undistorted_last().empty() ? proc_frame : detector.undistorted_last();
        auto targets = tracker.track(armors, timestamp, vis_base.size(), use_enemy_color, active_K_ptr);

        auto cmd = aimer.aim(targets, timestamp, 25.0, use_camera);
        const Eigen::Vector3d gimbal_pos = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
        cmd.shoot = shooter.shoot(cmd, aimer, targets, gimbal_pos);

        auto_aim::Plan plan_out{false};
        const auto_aim::Plan* plan_ptr = nullptr;
        if (planner) {
            std::optional<auto_aim::Target> opt_t =
                targets.empty() ? std::nullopt : std::optional<auto_aim::Target>(targets.front());
            plan_out = planner->plan(opt_t, 25.0);
            plan_ptr = &plan_out;
        }

        const double loop_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - loop_t0).count();

        if (frame_idx % 30 == 0 || (!use_camera && !targets.empty() && frame_idx % 5 == 0)) {
            tools::logger()->info(
                "[f={}] trk={} det={:.1f}ms m1={:.1f}ms loop={:.1f}ms raw={} acc={} rej={} nis={:.2f}",
                frame_idx, tracker.state(), st.det_ms, st.m1_ms, loop_ms,
                st.raw_yolo_count, st.m1_accepted_count, st.m1_rejected_count,
                targets.empty() ? 0.0 : targets.front().ekf().data.at("nis"));
        }

        if (!no_show || !out_path.empty()) {
            cv::Mat canvas = vis_base.clone();
            sp_new::draw_and_plot_ekf(canvas, frame_idx, loop_ms, st, armors, targets,
                                      cmd, plan_ptr, q, detector, solver, tracker, aimer,
                                      active_K_ptr, &plotter);
            if (!out_path.empty()) {
                if (!writer.isOpened()) {
                    writer.open(out_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), stream_fps, canvas.size());
                }
                if (writer.isOpened()) writer.write(canvas);
            }
            if (!no_show) {
                cv::Mat disp;
                if (canvas.cols > 960) cv::resize(canvas, disp, {}, 0.5, 0.5);
                else disp = canvas;
                cv::imshow("sp_ekf_test", disp);
                const int k = cv::waitKey(1);
                if (k == 'q' || k == 27) break;
            }
        }
    }

    if (mv_cam) mv_cam->close();
    if (writer.isOpened()) writer.release();
    return 0;
}

