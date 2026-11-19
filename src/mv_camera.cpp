#include "tav_pose/mv_camera.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#ifdef __linux__
#include <fcntl.h>
#include <linux/usbdevice_fs.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#if defined(TAV_WITH_MVSDK) && TAV_WITH_MVSDK
#include "CameraApi.h"
#endif

namespace tav_pose {
namespace {

void sleep_interruptible(const std::atomic<bool>& running, int total_ms) {
    constexpr int kStepMs = 40;
    for (int elapsed = 0; elapsed < total_ms && running.load(); elapsed += kStepMs) {
        std::this_thread::sleep_for(std::chrono::milliseconds(std::min(kStepMs, total_ms - elapsed)));
    }
}

std::string read_sysfs_trimmed(const std::filesystem::path& p) {
    std::ifstream ifs(p);
    if (!ifs.is_open()) return {};
    std::string s;
    std::getline(ifs, s);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) {
        s.pop_back();
    }
    return s;
}

bool reset_mindvision_usb_linux(const std::string& sn, const std::string& product) {
#ifdef __linux__
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path usb_root("/sys/bus/usb/devices");
    if (!fs::exists(usb_root, ec)) return false;

    fs::path matched_dir;
    fs::path fallback_dir;
    for (const auto& entry : fs::directory_iterator(usb_root, ec)) {
        if (ec) break;
        const fs::path d = entry.path();
        const std::string bus_s = read_sysfs_trimmed(d / "busnum");
        const std::string dev_s = read_sysfs_trimmed(d / "devnum");
        if (bus_s.empty() || dev_s.empty()) continue;

        const std::string sys_sn = read_sysfs_trimmed(d / "serial");
        if (!sn.empty() && sys_sn == sn) {
            matched_dir = d;
            break;
        }
        const std::string sys_prod = read_sysfs_trimmed(d / "product");
        const std::string sys_mfg = read_sysfs_trimmed(d / "manufacturer");
        if ((!product.empty() && sys_prod.find(product) != std::string::npos) ||
            sys_mfg.find("MindVision") != std::string::npos ||
            sys_mfg.find("MINDVISION") != std::string::npos ||
            sys_prod.find("MV-") != std::string::npos) {
            fallback_dir = d;
        }
    }

    const fs::path target = !matched_dir.empty() ? matched_dir : fallback_dir;
    if (target.empty()) {
        std::printf("[MvCamera] USB reset skipped: matching /sys/bus/usb/devices node not found.\n");
        std::fflush(stdout);
        return false;
    }

    const int busnum = std::atoi(read_sysfs_trimmed(target / "busnum").c_str());
    const int devnum = std::atoi(read_sysfs_trimmed(target / "devnum").c_str());
    if (busnum <= 0 || devnum <= 0) return false;

    char dev_path[64];
    std::snprintf(dev_path, sizeof(dev_path), "/dev/bus/usb/%03d/%03d", busnum, devnum);
    const int fd = ::open(dev_path, O_WRONLY);
    if (fd < 0) {
        std::printf("[MvCamera] USB reset open(%s) failed: %s\n", dev_path, std::strerror(errno));
        std::fflush(stdout);
        return false;
    }
    const int rc = ::ioctl(fd, USBDEVFS_RESET, 0);
    const int saved_err = errno;
    ::close(fd);
    std::printf("[MvCamera] USBDEVFS_RESET(%s) -> rc=%d (%s)\n",
                dev_path, rc, (rc == 0) ? "OK" : std::strerror(saved_err));
    std::fflush(stdout);
    return rc == 0;
#else
    (void)sn;
    (void)product;
    return false;
#endif
}

#if defined(TAV_WITH_MVSDK) && TAV_WITH_MVSDK
void ensure_sdk_initialized() {
    static std::once_flag once;
    std::call_once(once, []() { CameraSdkInit(1); });
}

void apply_requested_resolution(int h_camera,
                                const tSdkCameraCapbility& cap,
                                int req_w,
                                int req_h,
                                int max_w,
                                int max_h,
                                bool log_presets) {
    const int min_w = std::max(16, cap.sResolutionRange.iWidthMin);
    const int min_h = std::max(16, cap.sResolutionRange.iHeightMin);
    if (log_presets) {
        std::printf("[MvCamera] Range: [%d..%d] x [%d..%d] (skip=0x%x binSum=0x%x binAvg=0x%x)\n",
                    min_w, max_w, min_h, max_h,
                    cap.sResolutionRange.uSkipModeMask,
                    cap.sResolutionRange.uBinSumModeMask,
                    cap.sResolutionRange.uBinAverageModeMask);
    }

    const int n_presets = std::clamp(cap.iImageSizeDesc, 0, 64);
    tSdkImageResolution target_res{};
    bool found_preset = false;
    if (cap.pImageSizeDesc != nullptr) {
        for (int i = 0; i < n_presets; ++i) {
            const tSdkImageResolution& p = cap.pImageSizeDesc[i];
            if (log_presets) {
                std::printf("[MvCamera] Preset %d: %.31s (%dx%d, FOV=%dx%d off=(%d,%d), bin=%u/%u skip=%u)\n",
                            p.iIndex, p.acDescription, p.iWidth, p.iHeight,
                            p.iWidthFOV, p.iHeightFOV, p.iHOffsetFOV, p.iVOffsetFOV,
                            p.uBinSumMode, p.uBinAverageMode, p.uSkipMode);
            }
            if (req_w > 0 && req_h > 0 && !found_preset &&
                p.iWidth == req_w && p.iHeight == req_h) {
                target_res = p;
                found_preset = true;
            }
        }
    }
    if (log_presets) std::fflush(stdout);

    if (req_w <= 0 || req_h <= 0) return;

    const char* mode_tag = "preset";
    if (!found_preset) {
        CameraGetImageResolution(h_camera, &target_res);
        target_res.iIndex = 0xFF;
        target_res.uBinSumMode = 0;
        target_res.uBinAverageMode = 0;
        target_res.uResampleMask = 0;
        target_res.iWidthZoomHd = 0;
        target_res.iHeightZoomHd = 0;
        target_res.iWidthZoomSw = 0;
        target_res.iHeightZoomSw = 0;
        if (req_w * 2 == max_w && req_h * 2 == max_h &&
            (cap.sResolutionRange.uSkipModeMask & 0x1U) != 0U) {
            mode_tag = "skip2x2";
            target_res.uSkipMode = 1;
            target_res.iHOffsetFOV = 0;
            target_res.iVOffsetFOV = 0;
            target_res.iWidthFOV = max_w;
            target_res.iHeightFOV = max_h;
            target_res.iWidth = req_w;
            target_res.iHeight = req_h;
        } else {
            mode_tag = "custom_roi";
            int w = std::clamp(req_w, min_w, max_w);
            int h = std::clamp(req_h, min_h, max_h);
            w = std::max(min_w, w & ~7);
            h = std::max(min_h, h & ~7);
            target_res.uSkipMode = 0;
            target_res.iHOffsetFOV = std::max(0, ((max_w - w) / 2) & ~7);
            target_res.iVOffsetFOV = std::max(0, ((max_h - h) / 2) & ~7);
            target_res.iWidthFOV = w;
            target_res.iHeightFOV = h;
            target_res.iWidth = w;
            target_res.iHeight = h;
        }
    }
    const int res_status = CameraSetImageResolution(h_camera, &target_res);
    std::printf("[MvCamera] CameraSetImageResolution(%dx%d, %s) status = %d\n",
                target_res.iWidth, target_res.iHeight, mode_tag, res_status);
    if (res_status != CAMERA_STATUS_SUCCESS) {
        std::printf("[MvCamera] Warning: CameraSetImageResolution rejected (status=%d); "
                    "check [MvCamera] Preset list or use --cam-w 0 --cam-h 0.\n", res_status);
    }
}

MvResolutionInfo query_active_resolution(int h_camera, bool log_active) {
    tSdkImageResolution cur_res{};
    MvResolutionInfo info{};
    if (CameraGetImageResolution(h_camera, &cur_res) == CAMERA_STATUS_SUCCESS) {
        info.valid = true;
        info.h_offset = cur_res.iHOffsetFOV;
        info.v_offset = cur_res.iVOffsetFOV;
        info.fov_width = (cur_res.iWidthFOV > 0) ? cur_res.iWidthFOV : cur_res.iWidth;
        info.fov_height = (cur_res.iHeightFOV > 0) ? cur_res.iHeightFOV : cur_res.iHeight;
        info.out_width = cur_res.iWidth;
        info.out_height = cur_res.iHeight;
        if (log_active) {
            std::printf("[MvCamera] Active resolution: %dx%d (FOV=%dx%d off=(%d,%d))\n",
                        info.out_width, info.out_height,
                        info.fov_width, info.fov_height,
                        info.h_offset, info.v_offset);
        }
    }
    return info;
}
#endif

}  // namespace

bool MvCamera::sdk_available() {
#if defined(TAV_WITH_MVSDK) && TAV_WITH_MVSDK
    return true;
#else
    return false;
#endif
}

MvCamera::~MvCamera() {
    close();
}

MvResolutionInfo MvCamera::resolution_info() const {
    std::lock_guard<std::mutex> lk(frame_mutex_);
    return res_info_;
}

void MvCamera::configure_hardware(bool is_mono_sensor) {
#if defined(TAV_WITH_MVSDK) && TAV_WITH_MVSDK
    const int ae_stat = CameraSetAeState(h_camera_, FALSE);
    const int wb_stat = CameraSetWbMode(h_camera_, FALSE);
    const int exp_stat = CameraSetExposureTime(h_camera_, req_exposure_us_);

    if (is_mono_sensor) {
        CameraSetIspOutFormat(h_camera_, CAMERA_MEDIA_TYPE_MONO8);
    } else {
        CameraSetIspOutFormat(h_camera_, CAMERA_MEDIA_TYPE_BGR8);
    }

    BOOL ae_now = TRUE;
    BOOL wb_now = TRUE;
    double exp_now = -1.0;
    CameraGetAeState(h_camera_, &ae_now);
    CameraGetWbMode(h_camera_, &wb_now);
    CameraGetExposureTime(h_camera_, &exp_now);
    std::printf("[MvCamera] Manual lock: ae_rc=%d(ae=%d) wb_rc=%d(wb_auto=%d) exp_rc=%d(req=%.1fus actual=%.1fus)\n",
                ae_stat, static_cast<int>(ae_now), wb_stat, static_cast<int>(wb_now),
                exp_stat, req_exposure_us_, exp_now);
    std::fflush(stdout);
#else
    (void)is_mono_sensor;
#endif
}

bool MvCamera::init_hardware(bool throw_on_error, bool quiet_enum_fail) {
#if defined(TAV_WITH_MVSDK) && TAV_WITH_MVSDK
    close_hardware();
    ensure_sdk_initialized();

    auto fail = [&](const char* msg) -> bool {
        close_hardware();
        if (throw_on_error) {
            throw std::runtime_error(msg);
        }
        return false;
    };

    int iCameraCounts = 1;
    tSdkCameraDevInfo tCameraEnumList{};
    tSdkCameraCapbility tCapability{};

    const int enum_status = CameraEnumerateDevice(&tCameraEnumList, &iCameraCounts);
    if (!quiet_enum_fail || iCameraCounts > 0) {
        std::printf("statue = %d\n", enum_status);
        std::printf("count = %d\n", iCameraCounts);
    }
    if (iCameraCounts == 0) {
        return fail("MindVision camera not found (iCameraCounts == 0).");
    }
    dev_sn_.assign(tCameraEnumList.acSn,
                   strnlen(tCameraEnumList.acSn, sizeof(tCameraEnumList.acSn)));
    dev_product_.assign(tCameraEnumList.acProductName,
                        strnlen(tCameraEnumList.acProductName, sizeof(tCameraEnumList.acProductName)));

    const int init_status = CameraInit(&tCameraEnumList, -1, -1, &h_camera_);
    std::printf("state = %d\n", init_status);
    if (init_status != CAMERA_STATUS_SUCCESS) {
        h_camera_ = -1;
        return fail("CameraInit failed.");
    }

    if (CameraGetCapability(h_camera_, &tCapability) != CAMERA_STATUS_SUCCESS) {
        return fail("CameraGetCapability failed.");
    }

    max_width_ = tCapability.sResolutionRange.iWidthMax;
    max_height_ = tCapability.sResolutionRange.iHeightMax;
    if (max_width_ <= 0 || max_height_ <= 0) {
        return fail("Invalid camera max resolution from CameraGetCapability.");
    }

    rgb_buffer_ = static_cast<unsigned char*>(std::malloc(
        static_cast<size_t>(max_height_) * static_cast<size_t>(max_width_) * 4U));
    if (!rgb_buffer_) {
        return fail("Failed to allocate camera ISP buffer.");
    }

    mono_sensor_ = (tCapability.sIspCapacity.bMonoSensor != 0);

    // Force continuous mode (0) and configure resolution BEFORE CameraPlay so presets/ROI latch cleanly.
    CameraSetTriggerMode(h_camera_, 0);
    if (prefer_low_speed_ && tCapability.iFrameSpeedDesc > 1) {
        const int sp_rc = CameraSetFrameSpeed(h_camera_, 0);
        if (!quiet_enum_fail) {
            std::printf("[MvCamera] Safe fallback FrameSpeed=0 (LOW) status = %d\n", sp_rc);
        }
    } else if (!quiet_enum_fail) {
        int cur_sp = -1;
        if (CameraGetFrameSpeed(h_camera_, &cur_sp) == CAMERA_STATUS_SUCCESS) {
            std::printf("[MvCamera] FrameSpeed index = %d / %d\n",
                        cur_sp, tCapability.iFrameSpeedDesc);
        }
    }

    apply_requested_resolution(h_camera_, tCapability, req_width_, req_height_,
                               max_width_, max_height_, !quiet_enum_fail);

    // Disable AE/AWB and lock manual exposure/gain/format BEFORE CameraPlay.
    configure_hardware(mono_sensor_);

    CameraClearBuffer(h_camera_);
    const int play_status = CameraPlay(h_camera_);
    if (play_status != CAMERA_STATUS_SUCCESS) {
        std::printf("[MvCamera] Warning: CameraPlay status = %d\n", play_status);
    }

    const MvResolutionInfo info = query_active_resolution(h_camera_, !quiet_enum_fail);
    {
        std::lock_guard<std::mutex> lk(frame_mutex_);
        res_info_ = info;
    }

    return true;
#else
    (void)throw_on_error;
    (void)quiet_enum_fail;
    throw std::runtime_error(
        "Binary was compiled without MVSDK (xmake f --with_mvsdk=true on Linux with MindVision SDK).");
#endif
}

bool MvCamera::try_recover_connection(int last_status, int& reconnect_failures) {
#if defined(TAV_WITH_MVSDK) && TAV_WITH_MVSDK
    if (h_camera_ >= 0) {
        const int conn_status = CameraConnectTest(h_camera_);
        std::printf("[MvCamera] Timeout/disconnect (status=%d); CameraConnectTest=%d (attempt=%d)\n",
                    last_status, conn_status, reconnect_failures + 1);
        std::fflush(stdout);

        // Try lightweight SDK reconnect first on first stall; escalate to full re-open if repeated.
        if (reconnect_failures == 0) {
            const int rec_status = CameraReConnect(h_camera_);
            std::printf("[MvCamera] CameraReConnect status = %d\n", rec_status);
            std::fflush(stdout);
            if (rec_status == CAMERA_STATUS_SUCCESS &&
                CameraConnectTest(h_camera_) == CAMERA_STATUS_SUCCESS) {
                CameraPause(h_camera_);
                CameraSetTriggerMode(h_camera_, 0);
                tSdkCameraCapbility cap{};
                if (req_width_ > 0 && req_height_ > 0 &&
                    CameraGetCapability(h_camera_, &cap) == CAMERA_STATUS_SUCCESS) {
                    apply_requested_resolution(h_camera_, cap, req_width_, req_height_,
                                               max_width_, max_height_, false);
                }
                configure_hardware(mono_sensor_);
                CameraClearBuffer(h_camera_);
                CameraPlay(h_camera_);
                const MvResolutionInfo info = query_active_resolution(h_camera_, false);
                {
                    std::lock_guard<std::mutex> lk(frame_mutex_);
                    res_info_ = info;
                }
                ++reconnect_failures;
                return true;
            }
        }
    }

    std::printf("[MvCamera] Resetting USB & reopening camera (switching to safe LOW USB speed)...\n");
    std::fflush(stdout);
    prefer_low_speed_ = true;
    close_hardware();
    reset_mindvision_usb_linux(dev_sn_, dev_product_);
    sleep_interruptible(running_, 600);
    if (!running_.load()) return false;

    if (init_hardware(false, true)) {
        std::printf("[MvCamera] Camera recovered via USB reset + full reopen.\n");
        std::fflush(stdout);
        reconnect_failures = 0;
        grab_err_logs_ = 0;
        return true;
    }

    ++reconnect_failures;
    return false;
#else
    (void)last_status;
    (void)reconnect_failures;
    return false;
#endif
}

void MvCamera::capture_loop() {
#if defined(TAV_WITH_MVSDK) && TAV_WITH_MVSDK
    int consec_timeouts = 0;
    int reconnect_failures = 0;
    int last_reopen_log = 0;

    while (running_.load()) {
        if (h_camera_ < 0 || !rgb_buffer_) {
            sleep_interruptible(running_, 300);
            if (!running_.load()) break;
            if (reconnect_failures > 0 && (reconnect_failures % 3) == 0) {
                reset_mindvision_usb_linux(dev_sn_, dev_product_);
                sleep_interruptible(running_, 400);
                if (!running_.load()) break;
            }
            if (init_hardware(false, true)) {
                std::printf("[MvCamera] Camera reconnected.\n");
                std::fflush(stdout);
                consec_timeouts = 0;
                reconnect_failures = 0;
                last_reopen_log = 0;
                grab_err_logs_ = 0;
            } else {
                ++reconnect_failures;
                if (reconnect_failures == 1 || reconnect_failures - last_reopen_log >= 10) {
                    last_reopen_log = reconnect_failures;
                    std::printf("[MvCamera] Waiting for camera USB reconnection (tries=%d)...\n",
                                reconnect_failures);
                    std::fflush(stdout);
                }
            }
            continue;
        }

        tSdkFrameHead sFrameInfo{};
        BYTE* pbyBuffer = nullptr;
        const int get_status = CameraGetImageBuffer(h_camera_, &sFrameInfo, &pbyBuffer, 250);
        if (get_status != CAMERA_STATUS_SUCCESS || pbyBuffer == nullptr) {
            if (!running_.load()) break;
            const bool hard_drop = (get_status != CAMERA_STATUS_TIME_OUT);
            ++consec_timeouts;
            if (hard_drop || consec_timeouts >= 4) {
                try_recover_connection(get_status, reconnect_failures);
                consec_timeouts = 0;
            }
            continue;
        }

        consec_timeouts = 0;
        reconnect_failures = 0;
        const int proc_status = CameraImageProcess(h_camera_, pbyBuffer, rgb_buffer_, &sFrameInfo);
        const int w = sFrameInfo.iWidth;
        const int h = sFrameInfo.iHeight;
        const bool dims_ok =
            (proc_status == CAMERA_STATUS_SUCCESS) &&
            (w > 0 && h > 0 && w <= max_width_ && h <= max_height_);

        if (dims_ok) {
            const int mat_type =
                (sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8) ? CV_8UC1 : CV_8UC3;
            cv::Mat tmp;
            cv::Mat(cv::Size(w, h), mat_type, rgb_buffer_).copyTo(tmp);
            {
                std::lock_guard<std::mutex> lk(frame_mutex_);
                latest_frame_ = std::move(tmp);
                ++frame_seq_;
            }
            frame_cv_.notify_one();
        } else if (grab_err_logs_ < 10) {
            ++grab_err_logs_;
            std::printf("[MvCamera] Invalid processed frame (proc=%d, size=%dx%d, max=%dx%d)\n",
                        proc_status, w, h, max_width_, max_height_);
            std::fflush(stdout);
        }

        CameraReleaseImageBuffer(h_camera_, pbyBuffer);
    }
#endif
}

void MvCamera::open(double exposure_us, int req_width, int req_height) {
#if defined(TAV_WITH_MVSDK) && TAV_WITH_MVSDK
    close();
    req_exposure_us_ = exposure_us;
    req_width_ = req_width;
    req_height_ = req_height;
    grab_err_logs_ = 0;
    prefer_low_speed_ = false;
    {
        std::lock_guard<std::mutex> lk(frame_mutex_);
        latest_frame_.release();
        frame_seq_ = 0;
        consumed_seq_ = 0;
    }

    init_hardware(true, false);

    running_.store(true);
    capture_thread_ = std::thread(&MvCamera::capture_loop, this);
#else
    (void)exposure_us;
    (void)req_width;
    (void)req_height;
    throw std::runtime_error(
        "Binary was compiled without MVSDK (xmake f --with_mvsdk=true on Linux with MindVision SDK).");
#endif
}

bool MvCamera::grab(cv::Mat& frame_out, int timeout_ms, MvResolutionInfo* res_out) {
#if defined(TAV_WITH_MVSDK) && TAV_WITH_MVSDK
    std::unique_lock<std::mutex> lk(frame_mutex_);
    const bool ready = frame_cv_.wait_for(
        lk,
        std::chrono::milliseconds(std::max(1, timeout_ms)),
        [this]() { return !running_.load() || frame_seq_ > consumed_seq_; });
    if (!ready || !running_.load() || frame_seq_ <= consumed_seq_ || latest_frame_.empty()) {
        return false;
    }
    frame_out = std::move(latest_frame_);
    consumed_seq_ = frame_seq_;
    if (res_out) {
        *res_out = res_info_;
    }
    return true;
#else
    (void)frame_out;
    (void)timeout_ms;
    (void)res_out;
    return false;
#endif
}

void MvCamera::close_hardware() {
#if defined(TAV_WITH_MVSDK) && TAV_WITH_MVSDK
    if (h_camera_ >= 0) {
        CameraPause(h_camera_);
        CameraStop(h_camera_);
        CameraUnInit(h_camera_);
        h_camera_ = -1;
    }
#else
    h_camera_ = -1;
#endif
    if (rgb_buffer_) {
        std::free(rgb_buffer_);
        rgb_buffer_ = nullptr;
    }
    max_width_ = 0;
    max_height_ = 0;
    std::lock_guard<std::mutex> lk(frame_mutex_);
    res_info_ = MvResolutionInfo{};
}

void MvCamera::close() {
    running_.store(false);
    frame_cv_.notify_all();
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
    close_hardware();
    std::lock_guard<std::mutex> lk(frame_mutex_);
    latest_frame_.release();
    frame_seq_ = 0;
    consumed_seq_ = 0;
}

}  // namespace tav_pose
