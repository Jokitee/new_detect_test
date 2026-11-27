#pragma once

#include "tav_pose/gpu_accel.hpp"
#include "tav_pose/types.hpp"
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <array>
#include <mutex>
#include <string>
#include <vector>

namespace tav_pose {

/// Thread-safe L1/L2-resident isotropic ROI virtual-corner predictor (`RoiCornerNetV2`, 64x64 or 48x48).
class RoiCornerPrior {
public:
    RoiCornerPrior() = default;
    explicit RoiCornerPrior(const std::string& onnx_path, GpuMode gpu_mode = GpuMode::Auto);

    RoiCornerPrior(const RoiCornerPrior&) = delete;
    RoiCornerPrior& operator=(const RoiCornerPrior&) = delete;

    bool empty() const noexcept { return net_.empty(); }
    int roi_sz() const noexcept { return roi_sz_; }
    const DnnBackendStatus& backend_status() const noexcept { return backend_status_; }

    /// Predicts 4 ordered image-space corners corresponding to template rect3:
    /// [-W/2,-H/2], [W/2,-H/2], [W/2,H/2], [-W/2,H/2].
    bool predict(const cv::Mat& gray_u8,
                 const cv::Rect2f& box,
                 int cls_id,
                 std::array<cv::Point2d, 4>& out_corners) const;

private:
    mutable cv::dnn::Net net_;
    mutable std::mutex mu_;
    std::vector<std::string> out_names_;
    int roi_sz_ = 64;
    DnnBackendStatus backend_status_;
};

class SparseEdgePoseSolver {
public:
    /// Pure M1 cold-start pose solver directly from detector bounding box (no minAreaRect / IPPE).
    static PoseResult solve_cold_start(const cv::Mat& gray,
                                       const cv::Rect2f& box,
                                       const CharTemplate& tpl,
                                       const CameraIntrinsics& K,
                                       int margin = 14);

    /// Hybrid Option-A cold-start: 64x64 RoiCornerPrior + Dual-IPPE + adaptive 1~3-step Lie 1D Edge,
    /// with automatic safety fallback to Tier-1 `solve_cold_start` if residual or coverage fails gate.
    /// When `sparse_1d_raw` is true (O1n_pe, default), skips full-ROI float/LED-mask/GaussianBlur
    /// and samples raw uint8 along contour normals with 5-tap [1,4,6,4,1]/16 1D smoothing.
    static PoseResult solve_cold_start_hybrid(const cv::Mat& gray,
                                              const cv::Rect2f& box,
                                              const CharTemplate& tpl,
                                              const CameraIntrinsics& K,
                                              const RoiCornerPrior* corner_prior,
                                              int margin = 14,
                                              bool sparse_1d_raw = true);

    /// Temporal tracking mode: refines directly from previous frame's (rvec, tvec).
    static PoseResult solve_tracking(const cv::Mat& gray,
                                     const cv::Vec3d& prev_rvec,
                                     const cv::Vec3d& prev_tvec,
                                     const CharTemplate& tpl,
                                     const CameraIntrinsics& K,
                                     int margin = 16);

    static std::vector<cv::Point2d> project_points(const CharTemplate& tpl,
                                                   const cv::Vec3d& rvec,
                                                   const cv::Vec3d& tvec,
                                                   const CameraIntrinsics& K);

    static cv::Vec3d rvec_to_euler_yxz_deg(const cv::Vec3d& rvec);
};

}  // namespace tav_pose

