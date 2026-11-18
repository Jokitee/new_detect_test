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

    if (CameraGetCapability(h_camera_, &tCapability) != CAMERA_STATUS_SUCCESS) {
        close();
        throw std::runtime_error("CameraGetCapability failed.");
    }

    max_width_ = tCapability.sResolutionRange.iWidthMax;
    max_height_ = tCapability.sResolutionRange.iHeightMax;
    if (max_width_ <= 0 || max_height_ <= 0) {
        close();
        throw std::runtime_error("Invalid camera max resolution from CameraGetCapability.");
    }

    rgb_buffer_ = static_cast<unsigned char*>(std::malloc(
        static_cast<size_t>(max_height_) * static_cast<size_t>(max_width_) * 4U));
    if (!rgb_buffer_) {
        close();
        throw std::runtime_error("Failed to allocate camera ISP buffer.");
    }

    const int min_w = std::max(16, tCapability.sResolutionRange.iWidthMin);
    const int min_h = std::max(16, tCapability.sResolutionRange.iHeightMin);
    std::printf("[MvCamera] Range: [%d..%d] x [%d..%d] (skip=0x%x binSum=0x%x binAvg=0x%x)\n",
                min_w, max_width_, min_h, max_height_,
                tCapability.sResolutionRange.uSkipModeMask,
                tCapability.sResolutionRange.uBinSumModeMask,
                tCapability.sResolutionRange.uBinAverageModeMask);

    const int n_presets = std::clamp(tCapability.iImageSizeDesc, 0, 64);
    tSdkImageResolution target_res{};
    bool found_preset = false;
    if (tCapability.pImageSizeDesc != nullptr) {
        for (int i = 0; i < n_presets; ++i) {
            const tSdkImageResolution& p = tCapability.pImageSizeDesc[i];
            std::printf("[MvCamera] Preset %d: %.31s (%dx%d, FOV=%dx%d off=(%d,%d), bin=%u/%u skip=%u)\n",
                        p.iIndex, p.acDescription, p.iWidth, p.iHeight,
                        p.iWidthFOV, p.iHeightFOV, p.iHOffsetFOV, p.iVOffsetFOV,
                        p.uBinSumMode, p.uBinAverageMode, p.uSkipMode);
            if (req_width > 0 && req_height > 0 && !found_preset &&
                p.iWidth == req_width && p.iHeight == req_height) {
                target_res = p;
                found_preset = true;
            }
        }
    }
    std::fflush(stdout);

    // Follow example1.cpp & Catalyst-MDVS: enter CameraPlay before resolution/format calls.
    const int play_status = CameraPlay(h_camera_);
    if (play_status != CAMERA_STATUS_SUCCESS) {
        std::printf("[MvCamera] Warning: CameraPlay status = %d\n", play_status);
    }

    if (req_width > 0 && req_height > 0) {
        if (!found_preset) {
            CameraGetImageResolution(h_camera_, &target_res);
            int w = std::clamp(req_width, min_w, max_width_);
            int h = std::clamp(req_height, min_h, max_height_);
            w = std::max(min_w, w & ~7);
            h = std::max(min_h, h & ~7);
            const int h_off = std::max(0, ((max_width_ - w) / 2) & ~7);
            const int v_off = std::max(0, ((max_height_ - h) / 2) & ~7);
            target_res.iIndex = 0xFF;
            target_res.uBinSumMode = 0;
            target_res.uBinAverageMode = 0;
            target_res.uSkipMode = 0;
            target_res.uResampleMask = 0;
            target_res.iHOffsetFOV = h_off;
            target_res.iVOffsetFOV = v_off;
            target_res.iWidthFOV = w;
            target_res.iHeightFOV = h;
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
        if (res_status != CAMERA_STATUS_SUCCESS) {
            std::printf("[MvCamera] Warning: CameraSetImageResolution rejected (status=%d); "
                        "check [MvCamera] Preset list or use --cam-w 0 --cam-h 0.\n", res_status);
        }
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

    CameraSetAeState(h_camera_, false);
    setStatues = CameraSetExposureTime(h_camera_, exposure_us);
    CameraSetGain(h_camera_, 100, 70, 50);
    std::printf("statue = %d\n", setStatues);

    if (tCapability.sIspCapacity.bMonoSensor) {
        CameraSetIspOutFormat(h_camera_, CAMERA_MEDIA_TYPE_MONO8);
    } else {
        CameraSetIspOutFormat(h_camera_, CAMERA_MEDIA_TYPE_BGR8);
    }
    std::fflush(stdout);
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
    const int get_status = CameraGetImageBuffer(h_camera_, &sFrameInfo, &pbyBuffer, 1000);
    if (get_status != CAMERA_STATUS_SUCCESS || pbyBuffer == nullptr) {
        if (get_status != CAMERA_STATUS_TIME_OUT && grab_err_logs_ < 10) {
            ++grab_err_logs_;
            std::printf("[MvCamera] CameraGetImageBuffer failed (status=%d)\n", get_status);
            std::fflush(stdout);
        }
        return false;
    }

    const int proc_status = CameraImageProcess(h_camera_, pbyBuffer, rgb_buffer_, &sFrameInfo);
    const int w = sFrameInfo.iWidth;
    const int h = sFrameInfo.iHeight;
    const bool dims_ok =
        (proc_status == CAMERA_STATUS_SUCCESS) &&
        (w > 0 && h > 0 && w <= max_width_ && h <= max_height_);

    if (dims_ok) {
        const int mat_type =
            (sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8) ? CV_8UC1 : CV_8UC3;
        cv::Mat matImage(cv::Size(w, h), mat_type, rgb_buffer_);
        matImage.copyTo(frame_out);
    } else if (grab_err_logs_ < 10) {
        ++grab_err_logs_;
        std::printf("[MvCamera] Invalid processed frame (proc=%d, size=%dx%d, max=%dx%d)\n",
                    proc_status, w, h, max_width_, max_height_);
        std::fflush(stdout);
    }

    CameraReleaseImageBuffer(h_camera_, pbyBuffer);
    return dims_ok;
#else
    (void)frame_out;
    return false;
#endif
}

void MvCamera::close() {
#if defined(TAV_WITH_MVSDK) && TAV_WITH_MVSDK
    if (h_camera_ >= 0) {
        CameraPause(h_camera_);
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
    grab_err_logs_ = 0;
    res_info_ = MvResolutionInfo{};
}

}  // namespace tav_pose
