#include "tav_pose/mv_camera.hpp"
#include "tav_pose/sparse_edge_pose.hpp"
#include "tav_pose/yolo_detector.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __linux__
#include <execinfo.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

volatile std::sig_atomic_t g_stop_requested = 0;

void handle_signal(int) {
    g_stop_requested = 1;
}

void handle_crash_signal(int sig) {
#ifdef __linux__
    const char msg[] = "\n[FATAL] Program crashed with signal (backtrace):\n";
    const ssize_t wrote = write(STDERR_FILENO, msg, sizeof(msg) - 1);
    (void)wrote;
    void* frames[32];
    const int n = backtrace(frames, 32);
    backtrace_symbols_fd(frames, n, STDERR_FILENO);
#else
    (void)sig;
#endif
    std::signal(sig, SIG_DFL);
    std::raise(sig);
}

void install_crash_handlers() {
#ifdef __linux__
    void* warm[1];
    (void)backtrace(warm, 1);  // Pre-load libgcc unwinder outside signal context.
    std::signal(SIGSEGV, handle_crash_signal);
    std::signal(SIGABRT, handle_crash_signal);
    std::signal(SIGFPE, handle_crash_signal);
    std::signal(SIGBUS, handle_crash_signal);
#endif
}

enum class InputMode {
    Camera,
    Video,
    Image,
};

struct Args {
    InputMode mode = InputMode::Camera;
    std::string image_path;
    std::string video_path;
    std::string model_path = "assets/ifl_net_v2_448x336.onnx";
    std::string templates_path = "assets/templates.csv";
    std::string out_path;
    std::string roi_str;
    std::string cls_name = "1";
    std::string codec;
    int imgsz = -1;
    int det_w = -1;
    int det_h = -1;
    float conf = 0.28f;
    bool use_bgr = false;
    bool no_show = false;
    bool force_show = false;
    int bench_runs = 0;
    double fx = -1.0;          // <= 0 -> calibrated 1757.35 on camera/1280x1024, else 1400.0
    double fy = -1.0;          // <= 0 -> calibrated 1754.99 on camera/1280x1024, else fx
    double cx = -1.0;
    double cy = -1.0;
    bool undistort = true;

    // Camera & video output options
    double exposure_us = 8000.0;  // 8 ms default exposure (only exposure differs from example1.cpp)
    int cam_width = 640;
    int cam_height = 480;
    bool downsample = true;       // 640x480 -> 448x336 (0.7x INTER_AREA, no crop)
    int stream_w = 448;
    int stream_h = 336;
    double fps = 0.0;       // 0 -> auto (30.0 for camera, source FPS for video)
    int max_frames = 0;     // 0 -> unlimited
    bool track = false;
    int redetect_every = 15;
    int max_targets = 0;    // <= 0 -> solve all detections (in parallel); > 0 -> cap top-K
    double vis_gain = 0.0;  // <= 0 -> auto brightness boost on dark frames
    int threads = 0;        // > 0 -> cv::setNumThreads(threads)
};

void print_usage() {
    std::cout
        << "Usage: new_detect_test [mode] [options]\n"
        << "Modes (defaults to --camera when neither --image nor --video is given):\n"
        << "  --camera             Live MindVision camera stream -> OpenCV VideoWriter output\n"
        << "  --video <path>       Offline video file -> OpenCV VideoWriter output\n"
        << "  --image <path>       Single image inference -> PNG output\n\n"
        << "Detection & Pure-M1 Pose options:\n"
        << "  --model <onnx>       ONNX path (default assets/ifl_net_v2_448x336.onnx)\n"
        << "  --templates <csv>    3D sparse contour CSV (default assets/templates.csv)\n"
        << "  --imgsz <int>        Square detector input size override (e.g. 416 for YOLOv8n)\n"
        << "  --det-w <int>        Detector input width (default 448, or 416 if YOLOv8 416)\n"
        << "  --det-h <int>        Detector input height (default 336, or 416 if YOLOv8 416)\n"
        << "  --conf <float>       Detector confidence threshold (default 0.28)\n"
        << "  --bgr                Feed raw BGR instead of grayscale-BGR\n"
        << "  --f <double>         Override focal length px (default calibrated 1757.35/1754.99 on camera)\n"
        << "  --fy <double>        Override fy px\n"
        << "  --cx <double>        Override principal point cx (default calibrated 639.50 on camera)\n"
        << "  --cy <double>        Override principal point cy (default calibrated 511.50 on camera)\n"
        << "  --no-undistort       Disable lens undistortion (default enabled for 1280x1024 camera)\n"
        << "  --roi x0,y0,x1,y1    Skip YOLO and use manual box\n"
        << "  --cls 1|3|sentary    Target class when using --roi (default 1)\n"
        << "  --track              Enable temporal Pure-M1 tracking between YOLO keyframes\n"
        << "  --redetect-every <N> YOLO refresh interval in --track mode (default 15)\n"
        << "  --max-targets <N>    Max targets to solve per frame (default 0 = all, sorted by conf)\n"
        << "  --bench <N>          Run N iterations in --image mode to benchmark C++ latency\n\n"
        << "MindVision Camera options (follows example1.cpp; 640x480 hardware -> 448x336 downsample + 8ms exposure):\n"
        << "  --exposure <us>      Manual exposure time in us (default 8000 = 8ms)\n"
        << "  --cam-w <int>        Hardware resolution width (default 640)\n"
        << "  --cam-h <int>        Hardware resolution height (default 480)\n"
        << "  --stream-w <int>     Downsampled working width without crop (default 448)\n"
        << "  --stream-h <int>     Downsampled working height without crop (default 336)\n"
        << "  --no-downsample      Keep raw capture resolution (640x480) instead of downsampling to 448x336\n\n"
        << "Output options:\n"
        << "  --out <path>         Output video (.avi/.mp4) or image (.png) path\n"
        << "  --fps <double>       Output video FPS (default 30.0 for camera, source FPS for video)\n"
        << "  --codec <FOURCC>     Optional 4-char VideoWriter codec (e.g. MJPG, mp4v, XVID)\n"
        << "  --max-frames <N>     Optional frame limit (default 0 = unlimited until Ctrl+C/q/ESC)\n"
        << "  --vis-gain <double>  Display brightness gain (<=0 = auto 3x on dark frames)\n"
        << "  --threads <int>      Optional cv::setNumThreads(N) (e.g. 1 if VM OpenCV hangs)\n"
        << "  --show               Display cv::imshow window (already enabled by default in camera/video mode)\n"
        << "  --no-show            Disable cv::imshow window (headless recording)\n";
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + k);
            return argv[++i];
        };
        if (k == "--") {
            continue;
        } else if (k == "--camera") {
            a.mode = InputMode::Camera;
        } else if (k == "--video") {
            a.mode = InputMode::Video;
            a.video_path = next();
        } else if (k == "--image") {
            a.mode = InputMode::Image;
            a.image_path = next();
        } else if (k == "--model") a.model_path = next();
        else if (k == "--templates") a.templates_path = next();
        else if (k == "--out") a.out_path = next();
        else if (k == "--roi") a.roi_str = next();
        else if (k == "--cls") a.cls_name = next();
        else if (k == "--codec") a.codec = next();
        else if (k == "--imgsz") a.imgsz = std::stoi(next());
        else if (k == "--det-w") a.det_w = std::stoi(next());
        else if (k == "--det-h") a.det_h = std::stoi(next());
        else if (k == "--conf") a.conf = std::stof(next());
        else if (k == "--f") a.fx = std::stod(next());
        else if (k == "--fy") a.fy = std::stod(next());
        else if (k == "--cx") a.cx = std::stod(next());
        else if (k == "--cy") a.cy = std::stod(next());
        else if (k == "--no-undistort") a.undistort = false;
        else if (k == "--bench") a.bench_runs = std::stoi(next());
        else if (k == "--exposure") a.exposure_us = std::stod(next());
        else if (k == "--cam-w") a.cam_width = std::stoi(next());
        else if (k == "--cam-h") a.cam_height = std::stoi(next());
        else if (k == "--stream-w") a.stream_w = std::stoi(next());
        else if (k == "--stream-h") a.stream_h = std::stoi(next());
        else if (k == "--no-downsample") a.downsample = false;
        else if (k == "--fps") a.fps = std::stod(next());
        else if (k == "--max-frames") a.max_frames = std::max(0, std::stoi(next()));
        else if (k == "--track") a.track = true;
        else if (k == "--redetect-every") a.redetect_every = std::max(1, std::stoi(next()));
        else if (k == "--max-targets") a.max_targets = std::max(0, std::stoi(next()));
        else if (k == "--vis-gain") a.vis_gain = std::stod(next());
        else if (k == "--threads") a.threads = std::max(1, std::stoi(next()));
        else if (k == "--bgr") a.use_bgr = true;
        else if (k == "--show") a.force_show = true;
        else if (k == "--no-show") a.no_show = true;
        else if (k == "-h" || k == "--help") {
            print_usage();
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown option: " + k + " (pass --help for usage)");
        }
    }
    return a;
}

cv::Rect2f parse_roi(const std::string& s) {
    std::stringstream ss(s);
    std::string tok;
    float v[4]{};
    for (int i = 0; i < 4; ++i) {
        if (!std::getline(ss, tok, ',')) throw std::runtime_error("Invalid --roi format");
        v[i] = std::stof(tok);
    }
    return cv::Rect2f(v[0], v[1], v[2] - v[0], v[3] - v[1]);
}

const tav_pose::CharTemplate* find_template(const std::vector<tav_pose::CharTemplate>& bank,
                                            const std::string& name,
                                            int cls_id) {
    for (const auto& t : bank) {
        if (!name.empty() && t.name == name) return &t;
    }
    for (const auto& t : bank) {
        if (t.cls_id == cls_id) return &t;
    }
    return bank.empty() ? nullptr : &bank[0];
}

struct PoseTarget {
    tav_pose::Detection det;
    const tav_pose::CharTemplate* tpl = nullptr;
    tav_pose::PoseResult pose;
};

struct TrackState {
    std::vector<PoseTarget> items;
    int frames_since_det = 0;
};

struct FrameMetrics {
    double det_ms = 0.0;
    double pose_ms = 0.0;
    bool used_tracking = false;
    std::vector<PoseTarget> targets;
};

cv::Rect2f projected_bbox(const std::vector<cv::Point2d>& pts, int width, int height, float pad = 6.0f) {
    double min_x = 1e9, min_y = 1e9, max_x = -1e9, max_y = -1e9;
    for (const auto& p : pts) {
        min_x = std::min(min_x, p.x);
        min_y = std::min(min_y, p.y);
        max_x = std::max(max_x, p.x);
        max_y = std::max(max_y, p.y);
    }
    const float x0 = std::max(0.0f, static_cast<float>(min_x) - pad);
    const float y0 = std::max(0.0f, static_cast<float>(min_y) - pad);
    const float x1 = std::min(static_cast<float>(width - 1), static_cast<float>(max_x) + pad);
    const float y1 = std::min(static_cast<float>(height - 1), static_cast<float>(max_y) + pad);
    return cv::Rect2f(x0, y0, std::max(1.0f, x1 - x0), std::max(1.0f, y1 - y0));
}

bool try_temporal_tracking(const cv::Mat& gray,
                           const tav_pose::CameraIntrinsics& K,
                           const Args& args,
                           TrackState& track,
                           FrameMetrics& metrics) {
    if (!args.track || !args.roi_str.empty() || track.items.empty() ||
        track.frames_since_det >= args.redetect_every) {
        return false;
    }

    const size_t n_trk = track.items.size();
    std::vector<PoseTarget> updated(n_trk);
    std::atomic<bool> any_lost{false};
    const auto tm0 = std::chrono::steady_clock::now();

    auto solve_one_trk = [&](size_t i) {
        const auto& prev = track.items[i];
        if (!prev.tpl) {
            any_lost.store(true);
            return;
        }
        const auto pose = tav_pose::SparseEdgePoseSolver::solve_tracking(
            gray, prev.pose.rvec, prev.pose.tvec, *prev.tpl, K);
        if (!pose.valid || pose.edge_frac < 0.55 || pose.edge_med_px > 3.5) {
            any_lost.store(true);
            return;
        }
        const auto pts = tav_pose::SparseEdgePoseSolver::project_points(
            *prev.tpl, pose.rvec, pose.tvec, K);
        PoseTarget cur = prev;
        cur.pose = pose;
        cur.det.box = projected_bbox(pts, gray.cols, gray.rows);
        updated[i] = std::move(cur);
    };

    if (n_trk > 1) {
        cv::parallel_for_(cv::Range(0, static_cast<int>(n_trk)), [&](const cv::Range& range) {
            for (int i = range.start; i < range.end; ++i) {
                solve_one_trk(static_cast<size_t>(i));
            }
        });
    } else {
        solve_one_trk(0);
    }

    if (any_lost.load()) return false;

    const auto tm1 = std::chrono::steady_clock::now();
    metrics.det_ms = 0.0;
    metrics.pose_ms = std::chrono::duration<double, std::milli>(tm1 - tm0).count();
    metrics.used_tracking = true;
    metrics.targets = updated;
    track.items = std::move(updated);
    ++track.frames_since_det;
    return true;
}

FrameMetrics process_frame(const cv::Mat& frame,
                           const cv::Mat& gray,
                           const tav_pose::YoloDetector* detector,
                           const std::vector<tav_pose::CharTemplate>& bank,
                           const tav_pose::CameraIntrinsics& K,
                           const Args& args,
                           TrackState* track_state,
                           cv::Mat& vis_out,
                           bool trace_frame0 = false) {
    FrameMetrics metrics;

    const bool tracked =
        track_state && try_temporal_tracking(gray, K, args, *track_state, metrics);

    if (!tracked) {
        std::vector<tav_pose::Detection> dets;
        if (!args.roi_str.empty()) {
            tav_pose::Detection d;
            d.box = parse_roi(args.roi_str);
            d.cls_name = args.cls_name;
            d.conf = 1.0f;
            dets.push_back(d);
        } else if (detector) {
            if (trace_frame0) {
                std::cout << "[Trace] Frame 0 calling detector->detect..." << std::endl;
            }
            const auto t0 = std::chrono::steady_clock::now();
            dets = detector->detect(frame);
            const auto t1 = std::chrono::steady_clock::now();
            metrics.det_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (trace_frame0) {
                std::cout << "[Trace] Frame 0 detector OK (dets=" << dets.size()
                          << ", " << metrics.det_ms << "ms)" << std::endl;
            }
        }

        std::vector<const tav_pose::CharTemplate*> tpls;
        std::vector<tav_pose::Detection> solve_dets;
        for (const auto& d : dets) {
            const auto* tpl = find_template(bank, d.cls_name, d.cls_id);
            if (!tpl) continue;
            solve_dets.push_back(d);
            tpls.push_back(tpl);
            if (args.max_targets > 0 && static_cast<int>(solve_dets.size()) >= args.max_targets) {
                break;
            }
        }

        const size_t n_solve = solve_dets.size();
        if (trace_frame0 && n_solve > 0) {
            std::cout << "[Trace] Frame 0 running M1 on " << n_solve << " det(s)..." << std::endl;
        }
        if (n_solve > 0) {
            metrics.targets.resize(n_solve);
            const auto tm0 = std::chrono::steady_clock::now();
            auto solve_one = [&](size_t i) {
                PoseTarget item;
                item.det = solve_dets[i];
                item.tpl = tpls[i];
                item.pose = tav_pose::SparseEdgePoseSolver::solve_cold_start(
                    gray, solve_dets[i].box, *tpls[i], K);
                metrics.targets[i] = std::move(item);
            };
            if (n_solve > 1) {
                cv::parallel_for_(cv::Range(0, static_cast<int>(n_solve)), [&](const cv::Range& range) {
                    for (int i = range.start; i < range.end; ++i) {
                        solve_one(static_cast<size_t>(i));
                    }
                });
            } else {
                solve_one(0);
            }
            const auto tm1 = std::chrono::steady_clock::now();
            metrics.pose_ms = std::chrono::duration<double, std::milli>(tm1 - tm0).count();
        }
        if (trace_frame0) {
            std::cout << "[Trace] Frame 0 M1 OK (" << metrics.pose_ms << "ms)" << std::endl;
        }

        if (track_state) {
            track_state->items.clear();
            for (const auto& t : metrics.targets) {
                if (t.pose.valid) track_state->items.push_back(t);
            }
            track_state->frames_since_det = 0;
        }
    }

    cv::Mat vis_bgr;
    if (frame.channels() == 1) {
        cv::cvtColor(frame, vis_bgr, cv::COLOR_GRAY2BGR);
    } else {
        vis_bgr = frame;
    }

    const double gain = (args.vis_gain > 0.0)
                            ? args.vis_gain
                            : ((cv::mean(gray)[0] < 60.0) ? 3.0 : 1.0);
    vis_bgr.convertTo(vis_out, -1, gain, 0.0);

    for (const auto& item : metrics.targets) {
        if (!item.pose.valid || !item.tpl) continue;

        const cv::Scalar box_color =
            metrics.used_tracking ? cv::Scalar(0, 220, 255) : cv::Scalar(255, 200, 0);
        cv::rectangle(vis_out, item.det.box, box_color, 2);

        const auto pts = tav_pose::SparseEdgePoseSolver::project_points(
            *item.tpl, item.pose.rvec, item.pose.tvec, K);
        for (const auto& q : pts) {
            if (!std::isfinite(q.x) || !std::isfinite(q.y) ||
                std::abs(q.x) > 10000.0 || std::abs(q.y) > 10000.0) {
                continue;
            }
            cv::circle(vis_out,
                       cv::Point(static_cast<int>(std::round(q.x)),
                                 static_cast<int>(std::round(q.y))),
                       2, cv::Scalar(0, 255, 0), -1);
        }

        char buf[160];
        std::snprintf(buf, sizeof(buf), "%s (%.2f) ypr=(%+.1f,%+.1f,%+.1f) z=%.0fmm",
                      item.det.cls_name.c_str(), item.det.conf,
                      item.pose.euler_yxz_deg[0], item.pose.euler_yxz_deg[1],
                      item.pose.euler_yxz_deg[2], item.pose.tvec[2]);
        int baseline = 0;
        const cv::Size tsz = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.48, 1, &baseline);
        const int tx = std::max(4, std::min(static_cast<int>(std::round(item.det.box.x)),
                                            std::max(4, vis_out.cols - tsz.width - 6)));
        const int ty = std::max(18, static_cast<int>(std::round(item.det.box.y)) - 6);
        cv::putText(vis_out, buf, cv::Point(tx, ty), cv::FONT_HERSHEY_SIMPLEX,
                    0.48, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    }

    return metrics;
}

void draw_hud(cv::Mat& vis,
              long long frame_idx,
              double fps,
              const FrameMetrics& metrics,
              size_t valid_count) {
    char hud[192];
    std::snprintf(hud, sizeof(hud),
                  "Frame %lld | FPS %.1f | %s %.1fms | M1 %.2fms | targets %zu",
                  frame_idx, fps, metrics.used_tracking ? "TRK" : "DET",
                  metrics.det_ms, metrics.pose_ms, valid_count);
    cv::putText(vis, hud, cv::Point(12, 26), cv::FONT_HERSHEY_SIMPLEX,
                0.6, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
}

std::string to_lower(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

int fourcc_from_str(const std::string& code) {
    if (code.size() != 4) {
        throw std::runtime_error("Codec must be a 4-character FOURCC (got '" + code + "')");
    }
    return cv::VideoWriter::fourcc(code[0], code[1], code[2], code[3]);
}

bool open_video_writer(cv::VideoWriter& writer,
                       std::string& out_path,
                       double fps,
                       const cv::Size& frame_size,
                       const std::string& requested_codec) {
    const fs::path p(out_path);
    if (p.has_parent_path()) {
        fs::create_directories(p.parent_path());
    }

    const std::string ext = to_lower(p.extension().string());
    std::vector<std::string> codecs;
    if (!requested_codec.empty()) {
        codecs.push_back(requested_codec);
    } else if (ext == ".mp4" || ext == ".mkv" || ext == ".mov") {
        codecs = {"mp4v", "avc1", "H264", "MJPG"};
    } else {
        codecs = {"MJPG", "XVID", "mp4v"};
    }

    for (const auto& c : codecs) {
        std::cout << "[VideoWriter] Trying codec " << c << " (" << frame_size.width << "x"
                  << frame_size.height << ") -> " << out_path << std::endl;
        if (writer.open(out_path, fourcc_from_str(c), fps, frame_size, true)) {
            std::cout << "[VideoWriter] OpenCV " << CV_VERSION << " recording "
                      << frame_size.width << "x" << frame_size.height
                      << " @ " << fps << " FPS (codec=" << c << ") -> " << out_path << std::endl;
            return true;
        }
    }

    if (ext != ".avi") {
        fs::path fallback = p;
        fallback.replace_extension(".avi");
        std::string fallback_str = fallback.string();
        if (writer.open(fallback_str, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                        fps, frame_size, true)) {
            std::cerr << "[VideoWriter] Warning: failed to open '" << out_path
                      << "'; fell back to MJPG AVI -> " << fallback_str << "\n";
            out_path = fallback_str;
            return true;
        }
    }
    return false;
}

tav_pose::CameraIntrinsics build_intrinsics(const Args& args,
                                            const cv::Size& frame_size,
                                            const tav_pose::MvResolutionInfo* mv_res = nullptr) {
    tav_pose::CameraIntrinsics K;
    const bool use_calibrated =
        (args.fx <= 0.0) &&
        (args.mode == InputMode::Camera || (frame_size.width == 1280 && frame_size.height == 1024));
    if (use_calibrated) {
        if (mv_res && mv_res->valid && mv_res->fov_width > 0 && mv_res->fov_height > 0 &&
            mv_res->out_width > 0 && mv_res->out_height > 0) {
            const double sx = static_cast<double>(mv_res->out_width) / mv_res->fov_width;
            const double sy = static_cast<double>(mv_res->out_height) / mv_res->fov_height;
            K.fx *= sx;
            K.fy *= sy;
            K.cx = (K.cx - mv_res->h_offset) * sx;
            K.cy = (K.cy - mv_res->v_offset) * sy;
        }
        if (args.cx >= 0.0) K.cx = args.cx;
        if (args.cy >= 0.0) K.cy = args.cy;
        return K;
    }
    K.fx = (args.fx > 0.0) ? args.fx : 1400.0;
    K.fy = (args.fy > 0.0) ? args.fy : K.fx;
    K.cx = (args.cx >= 0.0) ? args.cx : frame_size.width * 0.5;
    K.cy = (args.cy >= 0.0) ? args.cy : frame_size.height * 0.5;
    K.k1 = K.k2 = K.p1 = K.p2 = K.k3 = 0.0;
    return K;
}

class Undistorter {
public:
    void apply(const cv::Mat& src,
               cv::Mat& dst,
               const tav_pose::CameraIntrinsics& K,
               bool enabled) {
        if (src.empty() || !enabled || K.fx <= 1.0 || K.fy <= 1.0 ||
            !std::isfinite(K.fx) || !std::isfinite(K.fy) ||
            !std::isfinite(K.cx) || !std::isfinite(K.cy) ||
            (K.k1 == 0.0 && K.k2 == 0.0 && K.p1 == 0.0 && K.p2 == 0.0 && K.k3 == 0.0)) {
            dst = src;
            return;
        }
        if (cached_size_ != src.size()) {
            const cv::Mat K_mat = (cv::Mat_<double>(3, 3) << K.fx, 0.0, K.cx,
                                   0.0, K.fy, K.cy,
                                   0.0, 0.0, 1.0);
            const cv::Mat dist = (cv::Mat_<double>(1, 5) << K.k1, K.k2, K.p1, K.p2, K.k3);
            cv::initUndistortRectifyMap(
                K_mat, dist, cv::Mat(), K_mat, src.size(), CV_16SC2, map1_, map2_);
            cached_size_ = src.size();
        }
        cv::remap(src, dst, map1_, map2_, cv::INTER_LINEAR);
    }

private:
    cv::Size cached_size_;
    cv::Mat map1_;
    cv::Mat map2_;
};

int run_image_mode(const Args& args,
                   const tav_pose::YoloDetector* detector,
                   const std::vector<tav_pose::CharTemplate>& bank) {
    std::ifstream imf(fs::u8path(args.image_path), std::ios::binary | std::ios::ate);
    if (!imf.is_open()) {
        std::cerr << "Failed to open image: " << args.image_path << "\n";
        return 1;
    }
    std::vector<uchar> raw_bytes(static_cast<size_t>(imf.tellg()));
    imf.seekg(0, std::ios::beg);
    imf.read(reinterpret_cast<char*>(raw_bytes.data()), static_cast<std::streamsize>(raw_bytes.size()));
    cv::Mat bgr = cv::imdecode(raw_bytes, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        std::cerr << "Failed to decode image: " << args.image_path << "\n";
        return 1;
    }

    const auto K = build_intrinsics(args, bgr.size());
    cv::Mat work_bgr;
    Undistorter undistorter;
    undistorter.apply(bgr, work_bgr, K, args.undistort && (bgr.cols == 1280 && bgr.rows == 1024));

    cv::Mat gray;
    cv::cvtColor(work_bgr, gray, cv::COLOR_BGR2GRAY);

    if (args.bench_runs > 0 && detector) {
        detector->detect(work_bgr);
    }

    cv::Mat vis;
    const FrameMetrics metrics = process_frame(work_bgr, gray, detector, bank, K, args, nullptr, vis);

    std::printf("Image: %s (%dx%d), f=%.1f px, detections=%zu, det_time=%.2f ms\n",
                args.image_path.c_str(), gray.cols, gray.rows, K.fx,
                metrics.targets.size(), metrics.det_ms);

    for (size_t i = 0; i < metrics.targets.size(); ++i) {
        const auto& item = metrics.targets[i];
        if (!item.pose.valid) {
            std::printf("  [%zu] cls=%s conf=%.2f -> M1 failed\n",
                        i, item.det.cls_name.c_str(), item.det.conf);
            continue;
        }
        std::printf("  [%zu] cls=%s conf=%.2f box=[%.0f, %.0f, %.0f, %.0f]\n",
                    i, item.det.cls_name.c_str(), item.det.conf,
                    item.det.box.x, item.det.box.y,
                    item.det.box.x + item.det.box.width,
                    item.det.box.y + item.det.box.height);
        std::printf("      M1 (Pure C++): ypr=(%+.1f, %+.1f, %+.1f) deg  t=(%+.1f, %+.1f, %.1f)mm  "
                    "edge=%.2fpx (%.0f%%)  mirror_gap=%.3f  time=%.2f ms\n",
                    item.pose.euler_yxz_deg[0], item.pose.euler_yxz_deg[1],
                    item.pose.euler_yxz_deg[2], item.pose.tvec[0], item.pose.tvec[1],
                    item.pose.tvec[2], item.pose.edge_med_px, 100.0 * item.pose.edge_frac,
                    item.pose.mirror_gap, item.pose.solve_ms);

        if (args.bench_runs > 0 && item.tpl) {
            double sum_cold = 0.0, sum_trk = 0.0;
            for (int r = 0; r < args.bench_runs; ++r) {
                sum_cold += tav_pose::SparseEdgePoseSolver::solve_cold_start(
                                gray, item.det.box, *item.tpl, K)
                                .solve_ms;
                sum_trk += tav_pose::SparseEdgePoseSolver::solve_tracking(
                               gray, item.pose.rvec, item.pose.tvec, *item.tpl, K)
                               .solve_ms;
            }
            std::printf("      Benchmark (%d runs): cold_avg=%.3f ms  track_avg=%.3f ms\n",
                        args.bench_runs, sum_cold / args.bench_runs, sum_trk / args.bench_runs);
        }
    }

    std::string out_path = args.out_path;
    if (out_path.empty()) {
        out_path = "out/cpp_pred_" + fs::path(args.image_path).stem().string() + ".png";
    }
    const fs::path op(out_path);
    if (op.has_parent_path()) fs::create_directories(op.parent_path());

    std::vector<uchar> out_buf;
    if (cv::imencode(".png", vis, out_buf)) {
        std::ofstream ofs(fs::u8path(out_path), std::ios::binary);
        ofs.write(reinterpret_cast<const char*>(out_buf.data()),
                  static_cast<std::streamsize>(out_buf.size()));
    }
    std::cout << "Saved overlay -> " << out_path << "\n";

    if (args.force_show && !args.no_show) {
        cv::imshow("new_detect_test", vis);
        cv::waitKey(0);
    }
    return 0;
}

int run_stream_mode(const Args& args,
                    const tav_pose::YoloDetector* detector,
                    const std::vector<tav_pose::CharTemplate>& bank) {
    tav_pose::MvCamera mv_cam;
    cv::VideoCapture video_cap;
    double target_fps = args.fps;

    if (args.mode == InputMode::Camera) {
        std::cout << "[Main] Opening MindVision camera (req=" << args.cam_width << "x"
                  << args.cam_height << ", exposure=" << args.exposure_us << "us)..."
                  << std::endl;
        mv_cam.open(args.exposure_us, args.cam_width, args.cam_height);
        if (target_fps <= 0.0) target_fps = 30.0;
    } else {
        if (!video_cap.open(args.video_path)) {
            std::cerr << "Failed to open input video: " << args.video_path << "\n";
            return 1;
        }
        if (target_fps <= 0.0) {
            const double src_fps = video_cap.get(cv::CAP_PROP_FPS);
            target_fps = (src_fps > 1.0 && src_fps < 240.0) ? src_fps : 30.0;
        }
    }

    std::string out_path = args.out_path;
    if (out_path.empty()) {
        if (args.mode == InputMode::Camera) {
            out_path = "out/mv_detect.avi";
        } else {
            out_path = "out/video_pred_" + fs::path(args.video_path).stem().string() + ".avi";
        }
    }

    cv::VideoWriter writer;
    TrackState track_state;
    const bool show_window = !args.no_show;
    long long frame_idx = 0;
    long long total_timeouts = 0;
    double ema_fps = 0.0;
    auto prev_tick = std::chrono::steady_clock::now();

    std::cout << "[Main] Stream started (mode="
              << (args.mode == InputMode::Camera ? "MindVision Camera" : "Video File")
              << ", track=" << (args.track ? "ON" : "OFF")
              << ", show=" << (show_window ? "ON" : "OFF")
              << ", max_frames="
              << (args.max_frames > 0 ? std::to_string(args.max_frames) : std::string("unlimited"))
              << "). Press Ctrl+C"
              << (show_window ? " or q/ESC in preview window" : "")
              << " to stop and finalize video." << std::endl;

    cv::Mat raw_frame, undist_frame, frame, gray, vis;
    Undistorter undistorter;
    while (!g_stop_requested) {
        if (args.max_frames > 0 && frame_idx >= static_cast<long long>(args.max_frames)) break;

        bool got = false;
        tav_pose::MvResolutionInfo cur_mv_res{};
        if (args.mode == InputMode::Camera) {
            got = mv_cam.grab(raw_frame, 50, &cur_mv_res);
            if (!got || raw_frame.empty()) {
                ++total_timeouts;
                if (total_timeouts == 1 || total_timeouts % 20 == 0) {
                    std::cout << "[Main] Waiting for camera frames (timeouts="
                              << total_timeouts << ")" << std::endl;
                }
                if (show_window && frame_idx > 0) {
                    const int key = cv::waitKey(1) & 0xFF;
                    if (key == 'q' || key == 'Q' || key == 27) break;
                }
                continue;
            }
        } else {
            got = video_cap.read(raw_frame);
            if (!got || raw_frame.empty()) break;
        }

        const tav_pose::MvResolutionInfo* mv_res =
            (args.mode == InputMode::Camera) ? &cur_mv_res : nullptr;
        auto K = build_intrinsics(args, raw_frame.size(), mv_res);
        const bool do_undist = args.undistort &&
                               (args.mode == InputMode::Camera ||
                                (raw_frame.cols == 1280 && raw_frame.rows == 1024));
        undistorter.apply(raw_frame, undist_frame, K, do_undist);

        const bool did_downsample =
            args.downsample && args.stream_w > 0 && args.stream_h > 0 &&
            (undist_frame.cols != args.stream_w || undist_frame.rows != args.stream_h);
        if (did_downsample) {
            const double sx = static_cast<double>(args.stream_w) / undist_frame.cols;
            const double sy = static_cast<double>(args.stream_h) / undist_frame.rows;
            cv::resize(undist_frame, frame, cv::Size(args.stream_w, args.stream_h),
                       0.0, 0.0, cv::INTER_AREA);
            K.fx *= sx;
            K.fy *= sy;
            K.cx *= sx;
            K.cy *= sy;
        } else {
            frame = undist_frame;
        }

        if (frame_idx == 0) {
            std::cout << "[Main] First frame: raw=" << raw_frame.cols << "x" << raw_frame.rows
                      << " -> work=" << frame.cols << "x" << frame.rows
                      << " ch=" << frame.channels()
                      << " K=(" << K.fx << "," << K.fy << "," << K.cx << "," << K.cy << ")"
                      << (do_undist ? " [undistort ON]" : "")
                      << (did_downsample ? " [INTER_AREA downsample]" : "") << std::endl;
        }

        if (frame.channels() == 1) {
            gray = frame;
        } else {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        }

        if (!writer.isOpened()) {
            if (!open_video_writer(writer, out_path, target_fps, gray.size(), args.codec)) {
                std::cerr << "Failed to open OpenCV VideoWriter for " << out_path << "\n";
                return 1;
            }
        }

        const FrameMetrics metrics = process_frame(
            frame, gray, detector, bank, K, args,
            args.track ? &track_state : nullptr, vis, frame_idx == 0);

        const auto now_tick = std::chrono::steady_clock::now();
        const double loop_s = std::chrono::duration<double>(now_tick - prev_tick).count();
        prev_tick = now_tick;
        const double inst_fps = (loop_s > 1e-6) ? (1.0 / loop_s) : 0.0;
        ema_fps = (frame_idx == 0) ? inst_fps : (0.85 * ema_fps + 0.15 * inst_fps);

        size_t valid_count = 0;
        for (const auto& t : metrics.targets) {
            if (t.pose.valid) ++valid_count;
        }
        draw_hud(vis, frame_idx + 1, ema_fps, metrics, valid_count);
        if (frame_idx == 0) {
            std::cout << "[Trace] Frame 0 calling writer.write..." << std::endl;
        }
        writer.write(vis);
        if (frame_idx == 0) {
            std::cout << "[Trace] Frame 0 writer.write OK." << std::endl;
        }
        ++frame_idx;

        if (frame_idx == 1 || frame_idx % 30 == 0) {
            std::printf("[Frame %lld] fps=%.1f targets=%zu det=%.1fms m1=%.2fms (%s)\n",
                        frame_idx, ema_fps, valid_count, metrics.det_ms, metrics.pose_ms,
                        metrics.used_tracking ? "TRACK" : "COLD");
            std::fflush(stdout);
        }

        if (show_window) {
            if (frame_idx == 1) {
                std::cout << "[Trace] Frame 0 calling cv::imshow..." << std::endl;
            }
            cv::imshow("new_detect_test", vis);
            const int key = cv::waitKey(5) & 0xFF;
            if (frame_idx == 1) {
                std::cout << "[Trace] Frame 0 imshow OK." << std::endl;
            }
            if (key == 'q' || key == 'Q' || key == 27) {
                break;
            }
        }
    }

    if (writer.isOpened()) {
        writer.release();
    }
    mv_cam.close();
    if (video_cap.isOpened()) {
        video_cap.release();
    }
    if (show_window) {
        cv::destroyAllWindows();
    }

    std::cout << "[Done] Recorded " << frame_idx << " frame(s)"
              << (total_timeouts > 0 ? " (camera timeouts=" + std::to_string(total_timeouts) + ")" : "")
              << (frame_idx > 0 ? (" -> " + out_path) : " (no frames captured; video file not created)")
              << "\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    install_crash_handlers();
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    try {
        const Args args = parse_args(argc, argv);
        if (args.threads > 0) {
            cv::setNumThreads(args.threads);
        }
        std::cout << "[Init] OpenCV version: " << CV_VERSION
                  << " (threads=" << cv::getNumThreads() << ")"
                  << " | MVSDK: " << (tav_pose::MvCamera::sdk_available() ? "enabled" : "disabled")
                  << std::endl;

        const auto bank = tav_pose::load_templates_csv(args.templates_path);
        if (bank.empty()) {
            std::cerr << "No templates loaded from " << args.templates_path << "\n";
            return 1;
        }
        std::cout << "[Init] Templates (" << bank.size() << "): ";
        for (size_t i = 0; i < bank.size(); ++i) {
            std::cout << (i ? ", " : "") << bank[i].name;
        }
        std::cout << std::endl;

        std::unique_ptr<tav_pose::YoloDetector> detector;
        if (args.roi_str.empty()) {
            int dw = args.det_w;
            int dh = args.det_h;
            if (dw <= 0 || dh <= 0) {
                if (args.imgsz > 0) {
                    dw = args.imgsz;
                    dh = args.imgsz;
                } else if (args.model_path.find("416") != std::string::npos) {
                    dw = 416;
                    dh = 416;
                } else {
                    dw = 448;
                    dh = 336;
                }
            }
            std::vector<std::string> names;
            names.reserve(bank.size());
            for (const auto& t : bank) names.push_back(t.name);
            detector = std::make_unique<tav_pose::YoloDetector>(
                args.model_path, std::move(names), dw, dh, args.conf, 0.45f, !args.use_bgr);
            std::cout << "[Init] Detector: " << args.model_path << " (" << dw << "x" << dh
                      << ", conf=" << args.conf << ")" << std::endl;
        }

        if (args.mode == InputMode::Image) {
            return run_image_mode(args, detector.get(), bank);
        }
        return run_stream_mode(args, detector.get(), bank);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
