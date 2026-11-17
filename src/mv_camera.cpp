#include "tav_pose/mv_camera.hpp"

#include <cstdio>
#include <cstdlib>
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

void MvCamera::open(double exposure_us) {
#if defined(TAV_WITH_MVSDK) && TAV_WITH_MVSDK
    close();

    int iCameraCounts = 1;
    int iStatues = -1;
    int setStatues = -1;
    tSdkCameraDevInfo tCameraEnumList{};
    tSdkCameraCapbility tCapability{};

    CameraSdkInit(1);

    iStatues = CameraEnumerateDevice(&tCameraEnumList, &iCameraCounts);
    std::printf("statue = %d\n", iStatues);
    std::printf("count = %d\n", iCameraCounts);
    if (iCameraCounts == 0) {
        throw std::runtime_error("MindVision camera not found (iCameraCounts == 0).");
    }

    iStatues = CameraInit(&tCameraEnumList, -1, -1, &h_camera_);
    std::printf("state = %d\n", iStatues);
    if (iStatues != CAMERA_STATUS_SUCCESS) {
        h_camera_ = -1;
        throw std::runtime_error("CameraInit failed.");
    }

    CameraGetCapability(h_camera_, &tCapability);

    rgb_buffer_ = static_cast<unsigned char*>(std::malloc(
        static_cast<size_t>(tCapability.sResolutionRange.iHeightMax) *
        static_cast<size_t>(tCapability.sResolutionRange.iWidthMax) * 3U));
    if (!rgb_buffer_) {
        close();
        throw std::runtime_error("Failed to allocate camera ISP buffer.");
    }

    CameraPlay(h_camera_);

    CameraSetAeState(h_camera_, false);
    setStatues = CameraSetExposureTime(h_camera_, exposure_us);
    CameraSetGain(h_camera_, 100, 70, 50);
    std::printf("statue = %d\n", setStatues);
    std::fflush(stdout);

    if (tCapability.sIspCapacity.bMonoSensor) {
        CameraSetIspOutFormat(h_camera_, CAMERA_MEDIA_TYPE_MONO8);
    } else {
        CameraSetIspOutFormat(h_camera_, CAMERA_MEDIA_TYPE_BGR8);
    }
#else
    (void)exposure_us;
    throw std::runtime_error(
        "Binary was compiled without MVSDK (xmake f --with_mvsdk=true on Linux with MindVision SDK).");
#endif
}

bool MvCamera::grab(cv::Mat& frame_out) {
#if defined(TAV_WITH_MVSDK) && TAV_WITH_MVSDK
    if (h_camera_ < 0 || !rgb_buffer_) return false;

    tSdkFrameHead sFrameInfo{};
    BYTE* pbyBuffer = nullptr;
    if (CameraGetImageBuffer(h_camera_, &sFrameInfo, &pbyBuffer, 1000) == CAMERA_STATUS_SUCCESS) {
        CameraImageProcess(h_camera_, pbyBuffer, rgb_buffer_, &sFrameInfo);

        cv::Mat matImage(
            cv::Size(sFrameInfo.iWidth, sFrameInfo.iHeight),
            sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8 ? CV_8UC1 : CV_8UC3,
            rgb_buffer_);
        matImage.copyTo(frame_out);

        CameraReleaseImageBuffer(h_camera_, pbyBuffer);
        return true;
    }
    return false;
#else
    (void)frame_out;
    return false;
#endif
}

void MvCamera::close() {
#if defined(TAV_WITH_MVSDK) && TAV_WITH_MVSDK
    if (h_camera_ >= 0) {
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
}

}  // namespace tav_pose
