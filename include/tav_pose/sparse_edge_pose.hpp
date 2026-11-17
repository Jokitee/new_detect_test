#pragma once

#include "tav_pose/types.hpp"
#include <opencv2/core.hpp>
#include <vector>

namespace tav_pose {

class SparseEdgePoseSolver {
public:
    /// Pure M1 cold-start pose solver directly from detector bounding box (no minAreaRect / IPPE).
    static PoseResult solve_cold_start(const cv::Mat& gray,
                                       const cv::Rect2f& box,
                                       const CharTemplate& tpl,
                                       const CameraIntrinsics& K,
                                       int margin = 14);

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

