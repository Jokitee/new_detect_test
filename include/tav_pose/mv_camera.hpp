#pragma once

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

/// MindVision industrial camera wrapper following MdVison_edu/demo/example1.cpp,
/// with hardware 640x480 resolution selection and 4 ms (4000 us) default exposure.
class MvCamera {
public:
    MvCamera() = default;
    ~MvCamera();

    MvCamera(const MvCamera&) = delete;
    MvCamera& operator=(const MvCamera&) = delete;

    static bool sdk_available();

    void open(double exposure_us = 4000.0, int req_width = 640, int req_height = 480);
    bool grab(cv::Mat& frame_out);
    void close();

    const MvResolutionInfo& resolution_info() const { return res_info_; }

private:
    int h_camera_ = -1;
    unsigned char* rgb_buffer_ = nullptr;
    MvResolutionInfo res_info_;
};

}  // namespace tav_pose
