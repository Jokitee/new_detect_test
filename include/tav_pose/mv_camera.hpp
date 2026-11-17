#pragma once

#include <opencv2/core.hpp>
#include <cstdint>
#include <vector>

namespace tav_pose {

struct MvCameraConfig {
    int camera_index = 0;
    bool auto_exposure = false;
    double exposure_us = 5000.0;
    int gain_r = 100;
    int gain_g = 70;
    int gain_b = 50;
    int grab_timeout_ms = 1000;
};

/// RAII wrapper around the MindVision industrial camera SDK (MVSDK).
class MvCamera {
public:
    MvCamera() = default;
    ~MvCamera();

    MvCamera(const MvCamera&) = delete;
    MvCamera& operator=(const MvCamera&) = delete;

    /// Returns true if compiled with MVSDK support.
    static bool sdk_available();

    /// Initialize SDK, open camera device, allocate ISP buffer and start streaming.
    void open(const MvCameraConfig& cfg);

    /// Grab one processed frame (CV_8UC1 for mono sensors, CV_8UC3 BGR for color sensors).
    /// Returns false on timeout (and attempts automatic USB reconnect if disconnected).
    bool grab(cv::Mat& frame_out);

    void close();

    bool is_opened() const { return opened_; }
    bool is_mono() const { return is_mono_; }
    int max_width() const { return max_width_; }
    int max_height() const { return max_height_; }

private:
    MvCameraConfig cfg_;
    int h_camera_ = -1;
    bool opened_ = false;
    bool is_mono_ = false;
    int max_width_ = 0;
    int max_height_ = 0;
    int consecutive_timeouts_ = 0;
    std::vector<uint8_t> isp_buffer_;
};

}  // namespace tav_pose

