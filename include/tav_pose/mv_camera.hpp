#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

#include <opencv2/core.hpp>

namespace tav_pose {

struct MvResolutionInfo {
    bool valid = false;
    int h_offset = 0;
    int v_offset = 0;
    int fov_width = 0;
    int fov_height = 0;
    int out_width = 0;
    int out_height = 0;
};

/// MindVision industrial camera wrapper with dedicated capture thread,
/// latest-frame ringless buffer, and automatic timeout/disconnect recovery.
class MvCamera {
public:
    MvCamera() = default;
    ~MvCamera();

    MvCamera(const MvCamera&) = delete;
    MvCamera& operator=(const MvCamera&) = delete;

    static bool sdk_available();

    void open(double exposure_us = 8000.0, int req_width = 640, int req_height = 480);
    bool grab(cv::Mat& frame_out, int timeout_ms = 50, MvResolutionInfo* res_out = nullptr);
    void close();

    MvResolutionInfo resolution_info() const;

private:
    bool init_hardware(bool throw_on_error, bool quiet_enum_fail = false);
    void configure_hardware(bool is_mono_sensor);
    void close_hardware();
    bool try_recover_connection(int last_status, int& reconnect_failures);
    void capture_loop();

    int h_camera_ = -1;
    int max_width_ = 0;
    int max_height_ = 0;
    int grab_err_logs_ = 0;
    bool mono_sensor_ = false;
    bool prefer_low_speed_ = false;
    double req_exposure_us_ = 8000.0;
    int req_width_ = 640;
    int req_height_ = 480;
    std::string dev_sn_;
    std::string dev_product_;
    unsigned char* rgb_buffer_ = nullptr;

    mutable std::mutex frame_mutex_;
    std::condition_variable frame_cv_;
    cv::Mat latest_frame_;
    MvResolutionInfo res_info_;
    uint64_t frame_seq_ = 0;
    uint64_t consumed_seq_ = 0;

    std::atomic<bool> running_{false};
    std::thread capture_thread_;
};

}  // namespace tav_pose
