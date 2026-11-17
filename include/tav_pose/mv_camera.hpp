#pragma once

#include <opencv2/core.hpp>

namespace tav_pose {

/// MindVision industrial camera wrapper following MdVison_edu/demo/example1.cpp.
/// Only exposure (default 4000 us = 4 ms) is changed; all other camera settings match example1.cpp.
class MvCamera {
public:
    MvCamera() = default;
    ~MvCamera();

    MvCamera(const MvCamera&) = delete;
    MvCamera& operator=(const MvCamera&) = delete;

    static bool sdk_available();

    void open(double exposure_us = 4000.0);
    bool grab(cv::Mat& frame_out);
    void close();

private:
    int h_camera_ = -1;
    unsigned char* rgb_buffer_ = nullptr;
};

}  // namespace tav_pose
