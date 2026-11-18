#pragma once

#include <fmt/core.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <list>
#include <optional>
#include <string>
#include <vector>

#include "sp_new/m1_tracker.hpp"
#include "sp_new/pure_m1_bridge.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tools/img_tools.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"

namespace sp_new {

inline int template_index_for(auto_aim::ArmorName name,
                              const std::vector<tav_pose::CharTemplate>& templates) {
    const std::string want = (name == auto_aim::ArmorName::one) ? "1"
                           : (name == auto_aim::ArmorName::three) ? "3"
                                                                  : "sentary";
    for (size_t i = 0; i < templates.size(); ++i) {
        if (templates[i].name == want) return static_cast<int>(i);
    }
    return 0;
}

inline void draw_and_plot_ekf(cv::Mat& canvas,
                              int frame_idx,
                              double loop_ms,
                              const BridgeDetectionStats& st,
                              const std::list<SolvedArmor>& armors,
                              const std::list<auto_aim::Target>& targets,
                              const io::Command& cmd,
                              const auto_aim::Plan* plan_opt,
                              const Eigen::Quaterniond& gimbal_q,
                              const PureM1ArmorDetector& detector,
                              const PureM1Solver& solver,
                              const PureM1Tracker& tracker,
                              const auto_aim::Aimer& aimer,
                              const tav_pose::CameraIntrinsics* active_K,
                              tools::Plotter* plotter) {
    if (cv::mean(canvas)[0] < 45.0) {
        canvas.convertTo(canvas, -1, 2.2, 6.0);
    }

    for (const auto& item : armors) {
        const auto& a = item.armor;
        tools::draw_points(canvas, a.points, {255, 220, 0}, 2);
        if (item.has_m1_pose && !detector.templates().empty()) {
            const int tidx = template_index_for(a.name, detector.templates());
            const auto pts = tav_pose::SparseEdgePoseSolver::project_points(
                detector.templates()[tidx],
                item.m1_pose.rvec,
                item.m1_pose.tvec,
                detector.last_intrinsics());
            for (const auto& q : pts) {
                cv::circle(canvas,
                           cv::Point(static_cast<int>(std::round(q.x)),
                                     static_cast<int>(std::round(q.y))),
                           2, {0, 255, 0}, -1);
            }
        }
        const std::string det_lbl = fmt::format(
            "{}({:.2f}) {} d={:.2f}m yaw={:+.1f}d e={:.2f}px",
            auto_aim::ARMOR_NAMES[a.name], a.confidence, auto_aim::COLORS[a.color],
            a.ypd_in_world[2], a.ypr_in_world[0] * 57.3, item.m1_pose.edge_med_px);
        tools::draw_text(canvas, det_lbl,
                         cv::Point(std::max(5, a.box.x), std::max(18, a.box.y - 6)),
                         {0, 255, 255}, 0.50, 1);
    }

    const std::string hdr = fmt::format(
        "[{}] trk={} det={:.1f}ms m1={:.1f}ms loop={:.1f}ms cmd=({}, y:{:+.1f}d, p:{:+.1f}d, fire:{})",
        frame_idx, tracker.state(), st.det_ms, st.m1_ms, loop_ms,
        cmd.control, cmd.yaw * 57.3, cmd.pitch * 57.3, cmd.shoot);
    tools::draw_text(canvas, hdr, {8, 22}, {255, 255, 255}, 0.52, 1);

    if (!targets.empty()) {
        const auto& target = targets.front();
        const auto xyza_list = target.armor_xyza_list();
        for (const auto& xyza : xyza_list) {
            auto pts = solver.reproject_armor(
                xyza.head(3), xyza[3], target.armor_type, target.name, active_K);
            tools::draw_points(canvas, pts, {0, 255, 0}, 2);
        }

        const auto aim_pt = aimer.debug_aim_point;
        if (aim_pt.valid) {
            auto aim_pts = solver.reproject_armor(
                aim_pt.xyza.head(3), aim_pt.xyza[3], target.armor_type, target.name, active_K);
            tools::draw_points(canvas, aim_pts, {0, 0, 255}, 2);
        }

        const Eigen::VectorXd x = target.ekf_x();
        const std::string ekf_lbl = fmt::format(
            "EKF xyz=({:.2f},{:.2f},{:.2f}) v=({:.2f},{:.2f}) w={:+.2f}rad/s r={:.3f} l={:.3f} nis={:.2f}",
            x[0], x[2], x[4], x[1], x[3], x[7], x[8], x[9], target.ekf().data.at("nis"));
        tools::draw_text(canvas, ekf_lbl, {8, 44}, {0, 255, 128}, 0.50, 1);
    }

    if (plan_opt && plan_opt->control) {
        const std::string pl_lbl = fmt::format(
            "TinyMPC y={:+.1f}d yv={:+.2f} ya={:+.1f} fire={}",
            plan_opt->yaw * 57.3, plan_opt->yaw_vel, plan_opt->yaw_acc, plan_opt->fire);
        tools::draw_text(canvas, pl_lbl, {8, 66}, {255, 180, 80}, 0.50, 1);
    }

    if (plotter) {
        nlohmann::json data;
        data["frame"] = frame_idx;
        data["det_ms"] = st.det_ms;
        data["m1_ms"] = st.m1_ms;
        data["loop_ms"] = loop_ms;
        data["armor_num"] = armors.size();
        data["m1_rejected"] = st.m1_rejected_count;
        if (!armors.empty()) {
            const auto& first = armors.front();
            data["armor_x"] = first.armor.xyz_in_world[0];
            data["armor_y"] = first.armor.xyz_in_world[1];
            data["armor_z"] = first.armor.xyz_in_world[2];
            data["armor_yaw"] = first.armor.ypr_in_world[0] * 57.3;
            data["armor_yaw_raw"] = first.armor.yaw_raw * 57.3;
            data["m1_edge_med"] = first.m1_pose.edge_med_px;
            data["m1_edge_frac"] = first.m1_pose.edge_frac;
        }
        const auto gimbal_ypr = tools::eulers(gimbal_q, 2, 1, 0);
        data["gimbal_yaw"] = gimbal_ypr[0] * 57.3;
        data["cmd_yaw"] = cmd.yaw * 57.3;
        data["cmd_pitch"] = cmd.pitch * 57.3;
        data["shoot"] = cmd.shoot;
        if (plan_opt) {
            data["plan_control"] = plan_opt->control;
            data["plan_yaw"] = plan_opt->yaw * 57.3;
            data["plan_yaw_vel"] = plan_opt->yaw_vel;
            data["plan_fire"] = plan_opt->fire;
        }
        if (!targets.empty()) {
            const auto& target = targets.front();
            const Eigen::VectorXd x = target.ekf_x();
            data["x"] = x[0];
            data["vx"] = x[1];
            data["y"] = x[2];
            data["vy"] = x[3];
            data["z"] = x[4];
            data["vz"] = x[5];
            data["a"] = x[6] * 57.3;
            data["w"] = x[7];
            data["r"] = x[8];
            data["l"] = x[9];
            data["h"] = x[10];
            data["last_id"] = target.last_id;
            data["nis"] = target.ekf().data.at("nis");
            data["recent_nis_failures"] = target.ekf().data.at("recent_nis_failures");
        }
        plotter->plot(data);
    }
}

}  // namespace sp_new

