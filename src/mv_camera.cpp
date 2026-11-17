#include "tav_pose/mv_camera.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>

#if defined(TAV_WITH_MVSDK) && TAV_WITH_MVSDK
#include "CameraApi.h"
#endif

namespace tav_pose {

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

void MvCamera::open(const MvCameraConfig& cfg) {
#if defined(TAV_WITH_MVSDK) && TAV_WITH_MVSDK
    close();
    cfg_ = cfg;

    CameraSdkInit(1);

    constexpr int kMaxDevices = 8;
    tSdkCameraDevInfo dev_list[kMaxDevices]{};
    int camera_counts = kMaxDevices;
    const int enum_status = CameraEnumerateDevice(dev_list, &camera_counts);
    if (enum_status != CAMERA_STATUS_SUCCESS || camera_counts <= 0) {
        std::ostringstream oss;
        oss << "MindVision camera enumeration failed (status=" << enum_status
            << ", count=" << camera_counts << "). Check USB3.0 connection and permissions.";
        throw std::runtime_error(oss.str());
    }
    if (cfg_.camera_index < 0 || cfg_.camera_index >= camera_counts) {
        std::ostringstream oss;
        oss << "Requested camera index " << cfg_.camera_index
            << " out of range (detected " << camera_counts << " device(s)).";
        throw std::runtime_error(oss.str());
    }

    int init_status = CameraInit(&dev_list[cfg_.camera_index], -1, -1, &h_camera_);
    if (init_status != CAMERA_STATUS_SUCCESS) {
        std::ostringstream oss;
        oss << "CameraInit failed (status=" << init_status << ").";
        throw std::runtime_error(oss.str());
    }
    opened_ = true;

    tSdkCameraCapbility cap{};
    CameraGetCapability(h_camera_, &cap);
    max_width_ = cap.sResolutionRange.iWidthMax;
    max_height_ = cap.sResolutionRange.iHeightMax;
    if (max_width_ <= 0 || max_height_ <= 0) {
        close();
        throw std::runtime_error("Invalid camera max resolution from CameraGetCapability.");
    }

    isp_buffer_.assign(static_cast<size_t>(max_width_) * static_cast<size_t>(max_height_) * 3U, 0);

    // Configure exposure, gain, pixel format, continuous trigger mode and frame speed before CameraPlay
    CameraSetAeState(h_camera_, cfg_.auto_exposure);
    if (!cfg_.auto_exposure && cfg_.exposure_us > 0.0) {
        const int exp_status = CameraSetExposureTime(h_camera_, cfg_.exposure_us);
        if (exp_status != CAMERA_STATUS_SUCCESS) {
            std::cerr << "[MvCamera] Warning: CameraSetExposureTime returned " << exp_status << "\n";
        }
    }
    CameraSetGain(h_camera_, cfg_.gain_r, cfg_.gain_g, cfg_.gain_b);

    if (cap.sIspCapacity.bMonoSensor) {
        is_mono_ = true;
        CameraSetIspOutFormat(h_camera_, CAMERA_MEDIA_TYPE_MONO8);
    } else {
        is_mono_ = false;
        CameraSetIspOutFormat(h_camera_, CAMERA_MEDIA_TYPE_BGR8);
    }

    const int trig_status = CameraSetTriggerMode(h_camera_, 0);
    if (trig_status != CAMERA_STATUS_SUCCESS) {
        std::cerr << "[MvCamera] Warning: CameraSetTriggerMode(0) returned " << trig_status << "\n";
    }

    int applied_speed = cfg_.frame_speed;
    if (cap.iFrameSpeedDesc > 0) {
        applied_speed = std::max(0, std::min(cfg_.frame_speed, cap.iFrameSpeedDesc - 1));
        const int spd_status = CameraSetFrameSpeed(h_camera_, applied_speed);
        if (spd_status != CAMERA_STATUS_SUCCESS) {
            std::cerr << "[MvCamera] Warning: CameraSetFrameSpeed(" << applied_speed
                      << ") returned " << spd_status << "\n";
        }
    }

    const int play_status = CameraPlay(h_camera_);
    if (play_status != CAMERA_STATUS_SUCCESS) {
        close();
        std::ostringstream oss;
        oss << "CameraPlay failed (status=" << play_status << ").";
        throw std::runtime_error(oss.str());
    }

    int cur_trig = -1;
    int cur_speed = -1;
    CameraGetTriggerMode(h_camera_, &cur_trig);
    CameraGetFrameSpeed(h_camera_, &cur_speed);
    consecutive_timeouts_ = 0;

    std::cout << "[MvCamera] Opened device #" << cfg_.camera_index
              << " (" << max_width_ << "x" << max_height_
              << ", " << (is_mono_ ? "MONO8" : "BGR8")
              << ", AE=" << (cfg_.auto_exposure ? "ON" : "OFF")
              << ", exp=" << cfg_.exposure_us << "us"
              << ", gain=(" << cfg_.gain_r << "," << cfg_.gain_g << "," << cfg_.gain_b << ")"
              << ", trig=" << cur_trig
              << ", speed=" << cur_speed << ")" << std::endl;
#else
    (void)cfg;
    throw std::runtime_error(
        "Binary was compiled without MVSDK (xmake f --with_mvsdk=true on Linux with MindVision SDK).");
#endif
}

bool MvCamera::grab(cv::Mat& frame_out) {
#if defined(TAV_WITH_MVSDK) && TAV_WITH_MVSDK
    if (!opened_) return false;

    tSdkFrameHead frame_info{};
    BYTE* raw_buf = nullptr;
    const int status = CameraGetImageBuffer(h_camera_, &frame_info, &raw_buf, cfg_.grab_timeout_ms);
    if (status != CAMERA_STATUS_SUCCESS || raw_buf == nullptr) {
        ++consecutive_timeouts_;
        if (consecutive_timeouts_ == 1 || consecutive_timeouts_ % 5 == 0) {
            std::cerr << "[MvCamera] CameraGetImageBuffer waiting/failed (status=" << status
                      << ", count=" << consecutive_timeouts_ << ")" << std::endl;
        }
        if (consecutive_timeouts_ >= 3 && CameraConnectTest(h_camera_) != CAMERA_STATUS_SUCCESS) {
            std::cerr << "[MvCamera] Connection lost, attempting CameraReConnect..." << std::endl;
            CameraReConnect(h_camera_);
            consecutive_timeouts_ = 0;
        }
        return false;
    }

    consecutive_timeouts_ = 0;
    CameraImageProcess(h_camera_, raw_buf, isp_buffer_.data(), &frame_info);

    const bool ok = (frame_info.iWidth > 0 && frame_info.iHeight > 0 &&
                     frame_info.iWidth <= max_width_ && frame_info.iHeight <= max_height_);
    if (ok) {
        const int mat_type =
            (frame_info.uiMediaType == CAMERA_MEDIA_TYPE_MONO8) ? CV_8UC1 : CV_8UC3;
        cv::Mat wrapped(cv::Size(frame_info.iWidth, frame_info.iHeight), mat_type, isp_buffer_.data());
        wrapped.copyTo(frame_out);
    }

    CameraReleaseImageBuffer(h_camera_, raw_buf);
    return ok;
#else
    (void)frame_out;
    return false;
#endif
}

void MvCamera::close() {
#if defined(TAV_WITH_MVSDK) && TAV_WITH_MVSDK
    if (opened_ && h_camera_ >= 0) {
        CameraUnInit(h_camera_);
    }
#endif
    h_camera_ = -1;
    opened_ = false;
    isp_buffer_.clear();
}

}  // namespace tav_pose

