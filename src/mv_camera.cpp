#include "tav_pose/mv_camera.hpp"

#include <algorithm>
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

void MvCamera::open(double exposure_us, int req_width, int req_height) {
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

    if (req_width > 0 && req_height > 0) {
        tSdkImageResolution target_res{};
        bool found_preset = false;
        if (tCapability.pImageSizeDesc != nullptr) {
            for (int i = 0; i < tCapability.iImageSizeDesc; ++i) {
                const tSdkImageResolution& p = tCapability.pImageSizeDesc[i];
                std::printf("[MvCamera] Preset %d: %.31s (%dx%d, FOV=%dx%d off=(%d,%d), bin=%u/%u skip=%u)\n",
                            p.iIndex, p.acDescription, p.iWidth, p.iHeight,
                            p.iWidthFOV, p.iHeightFOV, p.iHOffsetFOV, p.iVOffsetFOV,
                            p.uBinSumMode, p.uBinAverageMode, p.uSkipMode);
                if (!found_preset && p.iWidth == req_width && p.iHeight == req_height) {
                    target_res = p;
                    found_preset = true;
                }
            }
        }
        if (!found_preset) {
            CameraGetImageResolution(h_camera_, &target_res);
            const int w = std::min(req_width, tCapability.sResolutionRange.iWidthMax);
            const int h = std::min(req_height, tCapability.sResolutionRange.iHeightMax);
            target_res.iIndex = 0xFF;
            target_res.uBinSumMode = 0;
            target_res.uBinAverageMode = 0;
            target_res.uSkipMode = 0;
            target_res.uResampleMask = 0;
            target_res.iWidthFOV = w;
            target_res.iHeightFOV = h;
            target_res.iHOffsetFOV = std::max(0, (tCapability.sResolutionRange.iWidthMax - w) / 2);
            target_res.iVOffsetFOV = std::max(0, (tCapability.sResolutionRange.iHeightMax - h) / 2);
            target_res.iWidth = w;
            target_res.iHeight = h;
            target_res.iWidthZoomHd = 0;
            target_res.iHeightZoomHd = 0;
            target_res.iWidthZoomSw = 0;
            target_res.iHeightZoomSw = 0;
        }
        const int res_status = CameraSetImageResolution(h_camera_, &target_res);
        std::printf("[MvCamera] CameraSetImageResolution(%dx%d, %s) status = %d\n",
                    target_res.iWidth, target_res.iHeight,
                    found_preset ? "preset" : "custom_roi", res_status);
    }

    tSdkImageResolution cur_res{};
    if (CameraGetImageResolution(h_camera_, &cur_res) == CAMERA_STATUS_SUCCESS) {
        res_info_.valid = true;
        res_info_.h_offset = cur_res.iHOffsetFOV;
        res_info_.v_offset = cur_res.iVOffsetFOV;
        res_info_.fov_width = (cur_res.iWidthFOV > 0) ? cur_res.iWidthFOV : cur_res.iWidth;
        res_info_.fov_height = (cur_res.iHeightFOV > 0) ? cur_res.iHeightFOV : cur_res.iHeight;
        res_info_.out_width = cur_res.iWidth;
        res_info_.out_height = cur_res.iHeight;
        std::printf("[MvCamera] Active resolution: %dx%d (FOV=%dx%d off=(%d,%d))\n",
                    res_info_.out_width, res_info_.out_height,
                    res_info_.fov_width, res_info_.fov_height,
                    res_info_.h_offset, res_info_.v_offset);
    } else {
        res_info_ = MvResolutionInfo{};
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
    (void)req_width;
    (void)req_height;
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
    res_info_ = MvResolutionInfo{};
}

}  // namespace tav_pose
