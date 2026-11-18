#include "tav_pose/sparse_edge_pose.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

namespace tav_pose {
namespace {

constexpr double kPi = 3.14159265358979323846;

inline double deg2rad(double deg) { return deg * (kPi / 180.0); }
inline double rad2deg(double rad) { return rad * (180.0 / kPi); }

cv::Matx33d euler_yxz_mat(double yaw_deg, double pitch_deg, double roll_deg) {
    const double a = deg2rad(pitch_deg);
    const double b = deg2rad(yaw_deg);
    const double c = deg2rad(roll_deg);

    const double ca = std::cos(a), sa = std::sin(a);
    const double cb = std::cos(b), sb = std::sin(b);
    const double cc = std::cos(c), sc = std::sin(c);

    const cv::Matx33d Rx(1, 0, 0, 0, ca, -sa, 0, sa, ca);
    const cv::Matx33d Ry(cb, 0, sb, 0, 1, 0, -sb, 0, cb);
    const cv::Matx33d Rz(cc, -sc, 0, sc, cc, 0, 0, 0, 1);
    return Rx * Ry * Rz;
}

cv::Vec3d mat_to_rvec(const cv::Matx33d& R) {
    cv::Vec3d rvec;
    cv::Rodrigues(R, rvec);
    return rvec;
}

cv::Matx33d rvec_to_mat(const cv::Vec3d& rvec) {
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    return cv::Matx33d(R);
}

double rot_err_deg(const cv::Matx33d& R1, const cv::Matx33d& R2) {
    const cv::Matx33d D = R1.t() * R2;
    const double tr = D(0, 0) + D(1, 1) + D(2, 2);
    const double c = std::clamp((tr - 1.0) * 0.5, -1.0, 1.0);
    return rad2deg(std::acos(c));
}

inline bool is_finite_pose(const cv::Matx33d& R, const cv::Vec3d& t) {
    for (int r = 0; r < 3; ++r) {
        if (!std::isfinite(t[r])) return false;
        for (int c = 0; c < 3; ++c) {
            if (!std::isfinite(R(r, c))) return false;
        }
    }
    return t[2] > 1.0;
}

inline float sample_bilinear_f32(const cv::Mat& img, float x, float y) {
    if (!std::isfinite(x) || !std::isfinite(y) || img.empty()) return 0.0f;
    const int w = img.cols;
    const int h = img.rows;
    const float xc = std::clamp(x, 0.0f, static_cast<float>(w - 1));
    const float yc = std::clamp(y, 0.0f, static_cast<float>(h - 1));
    const int x0 = static_cast<int>(std::floor(xc));
    const int y0 = static_cast<int>(std::floor(yc));
    const int x1 = std::min(x0 + 1, w - 1);
    const int y1 = std::min(y0 + 1, h - 1);
    const float fx = xc - x0;
    const float fy = yc - y0;

    const float* r0 = img.ptr<float>(y0);
    const float* r1 = img.ptr<float>(y1);
    const float v0 = r0[x0] * (1.0f - fx) + r0[x1] * fx;
    const float v1 = r1[x0] * (1.0f - fx) + r1[x1] * fx;
    return v0 * (1.0f - fy) + v1 * fy;
}

void compute_hist_percentiles_u8(const cv::Mat& core, double& p50, double& p95) {
    std::array<int, 256> hist{};
    const int total = core.rows * core.cols;
    for (int y = 0; y < core.rows; ++y) {
        const uchar* row = core.ptr<uchar>(y);
        for (int x = 0; x < core.cols; ++x) {
            hist[row[x]]++;
        }
    }
    const int target50 = static_cast<int>(std::round(0.50 * (total - 1)));
    const int target95 = static_cast<int>(std::round(0.95 * (total - 1)));
    int acc = 0;
    bool got50 = false, got95 = false;
    p50 = 0.0;
    p95 = 0.0;
    for (int v = 0; v < 256; ++v) {
        acc += hist[v];
        if (!got50 && acc > target50) {
            p50 = static_cast<double>(v);
            got50 = true;
        }
        if (!got95 && acc > target95) {
            p95 = static_cast<double>(v);
            got95 = true;
            break;
        }
    }
}

double pca_roll_deg(const std::vector<cv::Point2d>& pts) {
    if (pts.empty()) return 0.0;
    double mx = 0.0, my = 0.0;
    for (const auto& p : pts) {
        mx += p.x;
        my += p.y;
    }
    mx /= pts.size();
    my /= pts.size();
    double cxx = 0.0, cxy = 0.0, cyy = 0.0;
    for (const auto& p : pts) {
        const double dx = p.x - mx;
        const double dy = p.y - my;
        cxx += dx * dx;
        cxy += dx * dy;
        cyy += dy * dy;
    }
    const double n = static_cast<double>(pts.size());
    cxx /= n;
    cxy /= n;
    cyy /= n;

    // Major eigenvector of 2x2 symmetric covariance matrix
    const double diff = cxx - cyy;
    const double disc = std::sqrt(diff * diff + 4.0 * cxy * cxy);
    const double lam1 = 0.5 * (cxx + cyy + disc);
    double vx = 0.0, vy = 1.0;
    if (std::abs(cxy) > 1e-12) {
        vx = lam1 - cyy;
        vy = cxy;
    } else if (cxx >= cyy) {
        vx = 1.0;
        vy = 0.0;
    }
    if (vy < 0.0) {
        vx = -vx;
        vy = -vy;
    }
    return rad2deg(std::atan2(-vx, vy));
}

bool solve_6x6(const cv::Matx66d& H, const cv::Vec6d& g, cv::Vec6d& dx) {
    cv::Mat Hm(H), gm(-g), xm;
    if (!cv::solve(Hm, gm, xm, cv::DECOMP_CHOLESKY) &&
        !cv::solve(Hm, gm, xm, cv::DECOMP_LU)) {
        return false;
    }
    for (int i = 0; i < 6; ++i) dx[i] = xm.at<double>(i);
    return true;
}

struct EdgeSearchResult {
    std::vector<double> off;
    std::vector<uint8_t> valid;
    int n_valid = 0;
};

EdgeSearchResult edge_search(const cv::Mat& gs,
                             const std::vector<cv::Point2d>& pts_img,
                             const std::vector<cv::Point2d>& n_img,
                             int rad) {
    const int n_pts = static_cast<int>(pts_img.size());
    const int len = 2 * rad + 1;
    const int dlen = len - 2;

    EdgeSearchResult out;
    out.off.assign(n_pts, 0.0);
    out.valid.assign(n_pts, 0);

    std::vector<float> v(len);
    std::vector<float> d(dlen);
    std::vector<float> dmins(n_pts, 0.0f);
    std::vector<std::vector<float>> all_d(n_pts, std::vector<float>(dlen));

    for (int i = 0; i < n_pts; ++i) {
        const float px = static_cast<float>(pts_img[i].x);
        const float py = static_cast<float>(pts_img[i].y);
        const float nx = static_cast<float>(n_img[i].x);
        const float ny = static_cast<float>(n_img[i].y);
        for (int k = 0; k < len; ++k) {
            const float o = static_cast<float>(k - rad);
            v[k] = sample_bilinear_f32(gs, px + o * nx, py + o * ny);
        }
        float dmin = std::numeric_limits<float>::infinity();
        for (int k = 0; k < dlen; ++k) {
            const float dv = 0.5f * (v[k + 2] - v[k]);
            all_d[i][k] = dv;
            if (dv < dmin) dmin = dv;
        }
        dmins[i] = dmin;
    }

    std::vector<float> neg_dmins(n_pts);
    for (int i = 0; i < n_pts; ++i) neg_dmins[i] = -dmins[i];
    std::nth_element(neg_dmins.begin(), neg_dmins.begin() + n_pts / 2, neg_dmins.end());
    const float contrast = neg_dmins[n_pts / 2];
    const float thr = -std::max(0.35f * contrast, 0.8f);

    for (int i = 0; i < n_pts; ++i) {
        if (!(dmins[i] < thr)) continue;
        const auto& di = all_d[i];
        int best_j = -1;
        float best_abs_o = std::numeric_limits<float>::infinity();
        const float half_min = 0.5f * dmins[i];
        for (int j = 1; j < dlen - 1; ++j) {
            if (di[j] <= di[j - 1] && di[j] <= di[j + 1] && di[j] < half_min) {
                const float o = static_cast<float>(j - (rad - 1));
                const float ao = std::abs(o);
                if (ao < best_abs_o) {
                    best_abs_o = ao;
                    best_j = j;
                }
            }
        }
        if (best_j < 0) continue;
        const float dm = di[best_j - 1];
        const float d0 = di[best_j];
        const float dp = di[best_j + 1];
        const float den = dm - 2.0f * d0 + dp;
        float sub = 0.0f;
        if (std::abs(den) > 1e-9f) {
            sub = std::clamp(0.5f * (dm - dp) / den, -0.5f, 0.5f);
        }
        const double o_j = static_cast<double>(best_j - (rad - 1));
        out.off[i] = o_j + sub;
        out.valid[i] = 1;
        out.n_valid++;
    }
    return out;
}

void project_with_lie_jac(const CharTemplate& tpl,
                          const cv::Matx33d& R,
                          const cv::Vec3d& t,
                          const CameraIntrinsics& K,
                          std::vector<cv::Point2d>& uv,
                          std::vector<cv::Point2d>& n_img,
                          std::vector<cv::Matx<double, 2, 6>>& J) {
    const size_t N = tpl.points.size();
    uv.resize(N);
    n_img.resize(N);
    J.resize(N);

    for (size_t i = 0; i < N; ++i) {
        const auto& mp = tpl.points[i];
        const double px_r = R(0, 0) * mp.px + R(0, 1) * mp.py;
        const double py_r = R(1, 0) * mp.px + R(1, 1) * mp.py;
        const double pz_r = R(2, 0) * mp.px + R(2, 1) * mp.py;

        const double X = px_r + t[0];
        const double Y = py_r + t[1];
        const double Z = std::max(pz_r + t[2], 1e-6);
        const double iz = 1.0 / Z;
        const double iz2 = iz * iz;

        uv[i] = cv::Point2d(K.fx * X * iz + K.cx, K.fy * Y * iz + K.cy);

        const double fx_iz = K.fx * iz;
        const double fy_iz = K.fy * iz;
        const double fx_X_iz2 = -K.fx * X * iz2;
        const double fy_Y_iz2 = -K.fy * Y * iz2;

        // SE(3) left-perturbation Jacobian: dPc = domega x Pr + dt
        cv::Matx<double, 2, 6>& Ji = J[i];
        Ji(0, 0) = fx_X_iz2 * py_r;
        Ji(0, 1) = fx_iz * pz_r - fx_X_iz2 * px_r;
        Ji(0, 2) = -fx_iz * py_r;
        Ji(0, 3) = fx_iz;
        Ji(0, 4) = 0.0;
        Ji(0, 5) = fx_X_iz2;

        Ji(1, 0) = -fy_iz * pz_r + fy_Y_iz2 * py_r;
        Ji(1, 1) = -fy_Y_iz2 * px_r;
        Ji(1, 2) = fy_iz * px_r;
        Ji(1, 3) = 0.0;
        Ji(1, 4) = fy_iz;
        Ji(1, 5) = fy_Y_iz2;

        // Analytical tangent -> outward image normal
        const double tcx = R(0, 0) * mp.tx + R(0, 1) * mp.ty;
        const double tcy = R(1, 0) * mp.tx + R(1, 1) * mp.ty;
        const double tcz = R(2, 0) * mp.tx + R(2, 1) * mp.ty;
        const double tux = fx_iz * tcx + fx_X_iz2 * tcz;
        const double tuy = fy_iz * tcy + fy_Y_iz2 * tcz;
        const double inv_tn = 1.0 / (std::hypot(tux, tuy) + 1e-12);
        double nix = tuy * inv_tn;
        double niy = -tux * inv_tn;

        const double ncx = R(0, 0) * mp.nx + R(0, 1) * mp.ny;
        const double ncy = R(1, 0) * mp.nx + R(1, 1) * mp.ny;
        const double ncz = R(2, 0) * mp.nx + R(2, 1) * mp.ny;
        const double nux = fx_iz * ncx + fx_X_iz2 * ncz;
        const double nuy = fy_iz * ncy + fy_Y_iz2 * ncz;
        if (nix * nux + niy * nuy < 0.0) {
            nix = -nix;
            niy = -niy;
        }
        n_img[i] = cv::Point2d(nix, niy);
    }
}

bool apply_lie_step(cv::Matx33d& R, cv::Vec3d& t, const cv::Vec6d& dx) {
    for (int k = 0; k < 6; ++k) {
        if (!std::isfinite(dx[k])) return false;
    }
    const cv::Matx33d R_next = rvec_to_mat(cv::Vec3d(dx[0], dx[1], dx[2])) * R;
    const cv::Vec3d t_next = t + cv::Vec3d(dx[3], dx[4], dx[5]);
    if (!is_finite_pose(R_next, t_next)) return false;
    R = R_next;
    t = t_next;
    return true;
}

void refine_chamfer_gated(const cv::Mat& dt,
                          const cv::Mat& dtx,
                          const cv::Mat& dty,
                          cv::Matx33d& R,
                          cv::Vec3d& t,
                          const CharTemplate& tpl,
                          const CameraIntrinsics& K,
                          int iters = 6,
                          double gate = 8.0,
                          double huber = 3.0) {
    std::vector<cv::Point2d> uv, n_dummy;
    std::vector<cv::Matx<double, 2, 6>> J;
    const size_t N = tpl.points.size();
    std::vector<double> ds(N), gxs(N), gys(N);

    for (int it = 0; it < iters; ++it) {
        project_with_lie_jac(tpl, R, t, K, uv, n_dummy, J);
        int vis_cnt = 0;
        for (size_t i = 0; i < N; ++i) {
            const float x = static_cast<float>(uv[i].x);
            const float y = static_cast<float>(uv[i].y);
            ds[i] = sample_bilinear_f32(dt, x, y);
            gxs[i] = sample_bilinear_f32(dtx, x, y);
            gys[i] = sample_bilinear_f32(dty, x, y);
            if (ds[i] <= gate) ++vis_cnt;
        }
        const bool use_gate = (vis_cnt >= 12);

        cv::Matx66d H = cv::Matx66d::zeros();
        cv::Vec6d g = cv::Vec6d::zeros();
        for (size_t i = 0; i < N; ++i) {
            if (use_gate && ds[i] > gate) continue;
            const double d = ds[i];
            const double w = (d <= huber) ? 1.0 : huber / std::max(d, 1e-9);
            double Jr[6];
            for (int c = 0; c < 6; ++c) {
                Jr[c] = gxs[i] * J[i](0, c) + gys[i] * J[i](1, c);
            }
            for (int r = 0; r < 6; ++r) {
                g[r] += w * Jr[r] * d;
                for (int c = 0; c < 6; ++c) {
                    H(r, c) += w * Jr[r] * Jr[c];
                }
            }
        }
        double tr = 0.0;
        for (int k = 0; k < 6; ++k) tr += H(k, k);
        const double damp = 1e-3 * tr / 6.0 + 1e-9;
        for (int k = 0; k < 6; ++k) H(k, k) += damp;

        cv::Vec6d dx;
        if (!solve_6x6(H, g, dx)) break;
        if (!apply_lie_step(R, t, dx)) break;

        double max_step = 0.0;
        for (int k = 0; k < 6; ++k) max_step = std::max(max_step, std::abs(dx[k]));
        if (max_step < 2e-4) break;
    }
}

void refine_edges_lie(const cv::Mat& gs,
                      cv::Matx33d& R,
                      cv::Vec3d& t,
                      const CharTemplate& tpl,
                      const CameraIntrinsics& K,
                      const std::vector<int>& radii,
                      double& out_med,
                      double& out_frac,
                      double huber = 1.5) {
    out_med = std::numeric_limits<double>::quiet_NaN();
    out_frac = 0.0;
    const int min_valid = std::min(12, std::max(6, static_cast<int>(tpl.points.size()) / 2));
    std::vector<cv::Point2d> uv, n_img;
    std::vector<cv::Matx<double, 2, 6>> J;

    for (int rad : radii) {
        project_with_lie_jac(tpl, R, t, K, uv, n_img, J);
        EdgeSearchResult es = edge_search(gs, uv, n_img, rad);
        if (es.n_valid < min_valid) break;

        cv::Matx66d H = cv::Matx66d::zeros();
        cv::Vec6d g = cv::Vec6d::zeros();
        std::vector<double> abs_e;
        abs_e.reserve(es.n_valid);

        for (size_t i = 0; i < tpl.points.size(); ++i) {
            if (!es.valid[i]) continue;
            const double e = -es.off[i];
            const double ae = std::abs(e);
            abs_e.push_back(ae);
            const double w = (ae <= huber) ? 1.0 : huber / std::max(ae, 1e-9);
            double Jr[6];
            for (int c = 0; c < 6; ++c) {
                Jr[c] = n_img[i].x * J[i](0, c) + n_img[i].y * J[i](1, c);
            }
            for (int r = 0; r < 6; ++r) {
                g[r] += w * Jr[r] * e;
                for (int c = 0; c < 6; ++c) {
                    H(r, c) += w * Jr[r] * Jr[c];
                }
            }
        }
        double tr = 0.0;
        for (int k = 0; k < 6; ++k) tr += H(k, k);
        const double damp = 1e-6 * tr / 6.0 + 1e-12;
        for (int k = 0; k < 6; ++k) H(k, k) += damp;

        cv::Vec6d dx;
        if (!solve_6x6(H, g, dx)) break;
        if (!apply_lie_step(R, t, dx)) break;

        if (!abs_e.empty()) {
            std::nth_element(abs_e.begin(), abs_e.begin() + abs_e.size() / 2, abs_e.end());
            out_med = abs_e[abs_e.size() / 2];
        }
        out_frac = static_cast<double>(es.n_valid) / static_cast<double>(tpl.points.size());
    }
}

double boundary_phot_contrast(const cv::Mat& gs,
                                const CharTemplate& tpl,
                                const cv::Matx33d& R,
                                const cv::Vec3d& t,
                                const CameraIntrinsics& K) {
    std::vector<cv::Point2d> uv, n_img;
    std::vector<cv::Matx<double, 2, 6>> J;
    project_with_lie_jac(tpl, R, t, K, uv, n_img, J);
    double sum = 0.0;
    for (size_t i = 0; i < uv.size(); ++i) {
        const float vi = sample_bilinear_f32(
            gs, static_cast<float>(uv[i].x - n_img[i].x), static_cast<float>(uv[i].y - n_img[i].y));
        const float vo = sample_bilinear_f32(
            gs, static_cast<float>(uv[i].x + n_img[i].x), static_cast<float>(uv[i].y + n_img[i].y));
        sum += static_cast<double>(vi - vo);
    }
    return sum / std::max<size_t>(uv.size(), 1);
}

struct SeedHyp {
    double score;
    cv::Matx33d R;
    int yaw_deg;
};

struct RefinedCand {
    cv::Matx33d R;
    cv::Vec3d t;
    double score;
    double phot;
    double med;
    double frac;
};

}  // namespace

std::vector<CharTemplate> load_templates_csv(const std::string& csv_path) {
    std::ifstream ifs(csv_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open templates CSV: " + csv_path);
    }
    std::map<int, CharTemplate> bank;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string tok;
        std::vector<std::string> cols;
        while (std::getline(ss, tok, ',')) cols.push_back(tok);
        if (cols.size() < 10) continue;

        const int cid = std::stoi(cols[0]);
        auto& tpl = bank[cid];
        tpl.cls_id = cid;
        tpl.name = cols[1];
        tpl.width_mm = std::stod(cols[2]);
        tpl.height_mm = std::stod(cols[3]);

        ModelPoint mp;
        mp.px = std::stod(cols[4]);
        mp.py = std::stod(cols[5]);
        mp.tx = std::stod(cols[6]);
        mp.ty = std::stod(cols[7]);
        mp.nx = std::stod(cols[8]);
        mp.ny = std::stod(cols[9]);
        tpl.points.push_back(mp);
    }
    std::vector<CharTemplate> out;
    for (auto& kv : bank) out.push_back(std::move(kv.second));
    return out;
}

cv::Vec3d SparseEdgePoseSolver::rvec_to_euler_yxz_deg(const cv::Vec3d& rvec) {
    const cv::Matx33d R = rvec_to_mat(rvec);
    const double yaw = rad2deg(std::asin(std::clamp(R(0, 2), -1.0, 1.0)));
    const double pitch = rad2deg(std::atan2(-R(1, 2), R(2, 2)));
    const double roll = rad2deg(std::atan2(-R(0, 1), R(0, 0)));
    return cv::Vec3d(yaw, pitch, roll);
}

std::vector<cv::Point2d> SparseEdgePoseSolver::project_points(const CharTemplate& tpl,
                                                              const cv::Vec3d& rvec,
                                                              const cv::Vec3d& tvec,
                                                              const CameraIntrinsics& K) {
    const cv::Matx33d R = rvec_to_mat(rvec);
    std::vector<cv::Point2d> out;
    out.reserve(tpl.points.size());
    for (const auto& mp : tpl.points) {
        const double X = R(0, 0) * mp.px + R(0, 1) * mp.py + tvec[0];
        const double Y = R(1, 0) * mp.px + R(1, 1) * mp.py + tvec[1];
        const double Z = std::max(R(2, 0) * mp.px + R(2, 1) * mp.py + tvec[2], 1e-6);
        out.emplace_back(K.fx * X / Z + K.cx, K.fy * Y / Z + K.cy);
    }
    return out;
}

PoseResult SparseEdgePoseSolver::solve_cold_start(const cv::Mat& gray,
                                                  const cv::Rect2f& box,
                                                  const CharTemplate& tpl,
                                                  const CameraIntrinsics& K,
                                                  int margin) {
    const auto t_start = std::chrono::steady_clock::now();
    PoseResult res;
    if (gray.empty() || tpl.points.empty()) return res;

    const int W_im = gray.cols;
    const int H_im = gray.rows;
    const double bx0 = box.x;
    const double by0 = box.y;
    const double bx1 = box.x + box.width;
    const double by1 = box.y + box.height;
    const double bw = std::max(bx1 - bx0, 4.0);
    const double bh = std::max(by1 - by0, 4.0);

    const int ix0 = std::clamp(static_cast<int>(std::floor(bx0)), 0, W_im - 1);
    const int iy0 = std::clamp(static_cast<int>(std::floor(by0)), 0, H_im - 1);
    const int ix1 = std::clamp(static_cast<int>(std::ceil(bx1)), ix0 + 1, W_im);
    const int iy1 = std::clamp(static_cast<int>(std::ceil(by1)), iy0 + 1, H_im);
    const cv::Mat core = gray(cv::Rect(ix0, iy0, ix1 - ix0, iy1 - iy0));
    if (core.total() < 16) return res;

    double bg_med = 0.0, p95_in = 0.0;
    compute_hist_percentiles_u8(core, bg_med, p95_in);
    const double led_thr = std::max(p95_in * 1.5, 80.0);

    // Padded segmentation ROI with bright external LED bars masked out
    const int pad_x = std::max(4, static_cast<int>(0.08 * bw));
    const int pad_y = std::max(4, static_cast<int>(0.08 * bh));
    const int rx0 = std::max(0, ix0 - pad_x);
    const int ry0 = std::max(0, iy0 - pad_y);
    const int rx1 = std::min(W_im, ix1 + pad_x);
    const int ry1 = std::min(H_im, iy1 + pad_y);

    cv::Mat roi_seg = gray(cv::Rect(rx0, ry0, rx1 - rx0, ry1 - ry0)).clone();
    const uchar bg_u8 = static_cast<uchar>(std::clamp(std::round(bg_med), 0.0, 255.0));
    const int in_y0 = iy0 - ry0, in_y1 = iy1 - ry0;
    const int in_x0 = ix0 - rx0, in_x1 = ix1 - rx0;
    for (int y = 0; y < roi_seg.rows; ++y) {
        uchar* row = roi_seg.ptr<uchar>(y);
        const bool y_out = (y < in_y0 || y >= in_y1);
        for (int x = 0; x < roi_seg.cols; ++x) {
            if ((y_out || x < in_x0 || x >= in_x1) && row[x] > led_thr) {
                row[x] = bg_u8;
            }
        }
    }

    const bool solid_target = (tpl.height_mm / std::max(tpl.width_mm, 1e-6)) >= 1.25;
    cv::Mat tight, bin;
    cv::GaussianBlur(roi_seg, tight, cv::Size(3, 3), 0.0);
    cv::threshold(tight, bin, 0, 1, cv::THRESH_BINARY | cv::THRESH_OTSU);
    if (solid_target && std::min(bin.rows, bin.cols) >= 16) {
        cv::morphologyEx(bin, bin, cv::MORPH_OPEN,
                         cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    }

    std::vector<std::vector<cv::Point>> cnts;
    std::vector<cv::Vec4i> hier;
    cv::findContours(bin, cnts, hier, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
    if (cnts.empty()) return res;

    std::vector<double> areas(cnts.size(), 0.0);
    auto in_core = [&](const std::vector<cv::Point>& c) {
        const cv::Rect bb = cv::boundingRect(c);
        const double mx = bb.x + 0.5 * bb.width;
        const double my = bb.y + 0.5 * bb.height;
        return mx >= (in_x0 - 2) && mx <= (in_x1 + 2) &&
               my >= (in_y0 - 2) && my <= (in_y1 + 2);
    };

    std::vector<int> top_inner;
    for (size_t i = 0; i < cnts.size(); ++i) {
        areas[i] = cv::contourArea(cnts[i]);
        const bool is_top = (hier.size() != cnts.size()) || (hier[i][3] < 0);
        if (is_top && in_core(cnts[i])) {
            top_inner.push_back(static_cast<int>(i));
        }
    }
    if (top_inner.empty()) {
        for (size_t i = 0; i < cnts.size(); ++i) {
            const bool is_top = (hier.size() != cnts.size()) || (hier[i][3] < 0);
            if (is_top) top_inner.push_back(static_cast<int>(i));
        }
    }
    if (top_inner.empty()) return res;

    int k_max = top_inner[0];
    double max_area = areas[k_max];
    for (int idx : top_inner) {
        if (areas[idx] > max_area) {
            max_area = areas[idx];
            k_max = idx;
        }
    }
    if (max_area < 16.0) return res;

    std::vector<cv::Point2d> cnt;
    double min_cx = 1e9, max_cx = -1e9, min_cy = 1e9, max_cy = -1e9;
    auto push_cnt = [&](const std::vector<cv::Point>& pts_local) {
        for (const auto& pt : pts_local) {
            const double gx = pt.x + rx0;
            const double gy = pt.y + ry0;
            cnt.emplace_back(gx, gy);
            min_cx = std::min(min_cx, gx);
            max_cx = std::max(max_cx, gx);
            min_cy = std::min(min_cy, gy);
            max_cy = std::max(max_cy, gy);
        }
    };
    for (int oi : top_inner) {
        if (oi == k_max || areas[oi] >= 0.12 * max_area) {
            push_cnt(cnts[oi]);
            if (hier.size() == cnts.size()) {
                for (size_t c = 0; c < cnts.size(); ++c) {
                    if (hier[c][3] == oi && areas[c] > 0.015 * areas[oi]) {
                        push_cnt(cnts[c]);
                    }
                }
            }
        }
    }
    if (cnt.empty()) return res;

    // Optimization ROI patch with external LED suppression
    const int x0 = std::max(0, static_cast<int>(std::floor(std::min(bx0, min_cx))) - margin);
    const int y0 = std::max(0, static_cast<int>(std::floor(std::min(by0, min_cy))) - margin);
    const int x1 = std::min(W_im, static_cast<int>(std::ceil(std::max(bx1, max_cx))) + margin + 1);
    const int y1 = std::min(H_im, static_cast<int>(std::ceil(std::max(by1, max_cy))) + margin + 1);
    if (x1 <= x0 + 1 || y1 <= y0 + 1 || K.fx <= 1.0 || K.fy <= 1.0) return res;

    cv::Mat sub_u8 = gray(cv::Rect(x0, y0, x1 - x0, y1 - y0));
    cv::Mat sub;
    sub_u8.convertTo(sub, CV_32F);
    const int oy0 = iy0 - y0, oy1 = iy1 - y0;
    const int ox0 = ix0 - x0, ox1 = ix1 - x0;
    for (int y = 0; y < sub.rows; ++y) {
        float* row = sub.ptr<float>(y);
        const bool y_out = (y < oy0 || y >= oy1);
        for (int x = 0; x < sub.cols; ++x) {
            if ((y_out || x < ox0 || x >= ox1) && row[x] > led_thr) {
                row[x] = static_cast<float>(bg_med);
            }
        }
    }

    CameraIntrinsics Kc = K;
    Kc.cx -= x0;
    Kc.cy -= y0;

    cv::Mat edge(sub.rows, sub.cols, CV_8UC1, cv::Scalar(255));
    for (auto& pt : cnt) {
        pt.x -= x0;
        pt.y -= y0;
        const int px = std::clamp(static_cast<int>(std::round(pt.x)), 0, edge.cols - 1);
        const int py = std::clamp(static_cast<int>(std::round(pt.y)), 0, edge.rows - 1);
        edge.at<uchar>(py, px) = 0;
    }
    cv::Mat dt, dtx, dty;
    cv::distanceTransform(edge, dt, cv::DIST_L2, 3);
    cv::Sobel(dt, dtx, CV_32F, 1, 0, 3, 0.125);
    cv::Sobel(dt, dty, CV_32F, 0, 1, 3, 0.125);

    const double roll_deg = pca_roll_deg(cnt);
    const double ch = max_cy - min_cy;
    double cx_c = 0.0, cy_c = 0.0, h_c = bh;
    if (ch < 0.8 * bh) {
        cx_c = 0.5 * (bx0 + bx1) - x0;
        cy_c = 0.5 * (by0 + by1) - y0;
        h_c = bh;
    } else {
        cx_c = 0.5 * (min_cx + max_cx) - x0;
        cy_c = 0.5 * (min_cy + max_cy) - y0;
        h_c = std::max({ch, bh * 0.8, 4.0});
    }
    const double z0 = Kc.fy * tpl.height_mm / h_c;
    const cv::Vec3d t0((cx_c - Kc.cx) * z0 / Kc.fx, (cy_c - Kc.cy) * z0 / Kc.fy, z0);

    // Coarse multi-yaw/pitch rotation seeds (no minAreaRect, no IPPE)
    std::vector<double> rolls;
    if (solid_target) {
        rolls.push_back(roll_deg);
        if (std::abs(roll_deg) > 4.0) rolls.push_back(0.0);
    } else {
        rolls = {0.0, -18.0, 18.0};
    }

    std::vector<SeedHyp> scored;
    const std::array<int, 5> yaws{-42, -16, 0, 16, 42};
    const std::array<int, 3> pitches{-20, 0, 20};
    scored.reserve(rolls.size() * yaws.size() * pitches.size());

    for (double rl : rolls) {
        for (int y : yaws) {
            for (int p : pitches) {
                const cv::Matx33d R = euler_yxz_mat(static_cast<double>(y), static_cast<double>(p), rl);
                double sum_d = 0.0;
                for (const auto& mp : tpl.points) {
                    const double X = R(0, 0) * mp.px + R(0, 1) * mp.py + t0[0];
                    const double Y = R(1, 0) * mp.px + R(1, 1) * mp.py + t0[1];
                    const double Z = std::max(R(2, 0) * mp.px + R(2, 1) * mp.py + t0[2], 1e-6);
                    const float u = static_cast<float>(Kc.fx * X / Z + Kc.cx);
                    const float v = static_cast<float>(Kc.fy * Y / Z + Kc.cy);
                    sum_d += std::min(static_cast<double>(sample_bilinear_f32(dt, u, v)), 8.0);
                }
                scored.push_back({sum_d / tpl.points.size(), R, y});
            }
        }
    }
    std::sort(scored.begin(), scored.end(),
              [](const SeedHyp& a, const SeedHyp& b) { return a.score < b.score; });

    std::vector<SeedHyp> cands{scored[0]};
    auto add_branch = [&](auto pred) {
        int added = 0;
        for (const auto& s : scored) {
            if (!pred(s.yaw_deg)) continue;
            bool distinct = true;
            for (const auto& c : cands) {
                if (rot_err_deg(s.R, c.R) <= 10.0) {
                    distinct = false;
                    break;
                }
            }
            if (distinct) {
                cands.push_back(s);
                ++added;
                if (added >= 2 || cands.size() >= 6) break;
            }
        }
    };
    add_branch([](int y) { return y < 0; });
    add_branch([](int y) { return y > 0; });
    add_branch([](int y) { return y == 0; });

    cv::Mat gs;
    cv::GaussianBlur(sub, gs, cv::Size(0, 0), 1.0);

    std::vector<RefinedCand> refined;
    const double chamfer_gate = std::max(8.0, 0.06 * h_c);
    const int edge_r0 = std::clamp(static_cast<int>(std::round(0.04 * h_c)), 6, 14);
    std::vector<int> radii;
    if (edge_r0 > 6) {
        radii = {edge_r0, 6, 4, 3, 2};
    } else {
        radii = {6, 4, 3, 2};
    }
    for (const auto& cand : cands) {
        cv::Matx33d R = cand.R;
        cv::Vec3d t = t0;
        refine_chamfer_gated(dt, dtx, dty, R, t, tpl, Kc, 8, chamfer_gate, 3.0);
        double med = 0.0, frac = 0.0;
        refine_edges_lie(gs, R, t, tpl, Kc, radii, med, frac, 1.5);
        if (!std::isfinite(med) || !is_finite_pose(R, t)) continue;
        const double phot = boundary_phot_contrast(gs, tpl, R, t, Kc);
        if (!std::isfinite(phot)) continue;
        const double sc = phot * (frac * frac) - 2.0 * std::max(med - 0.5, 0.0);
        refined.push_back({R, t, sc, phot, med, frac});
    }
    if (refined.empty()) return res;

    std::sort(refined.begin(), refined.end(),
              [](const RefinedCand& a, const RefinedCand& b) { return a.score > b.score; });

    const auto& best = refined[0];
    res.valid = true;
    res.rvec = mat_to_rvec(best.R);
    res.tvec = best.t;
    res.euler_yxz_deg = rvec_to_euler_yxz_deg(res.rvec);
    res.edge_med_px = best.med;
    res.edge_frac = best.frac;
    res.phot_contrast = best.phot;
    res.mirror_gap = std::numeric_limits<double>::infinity();
    for (size_t i = 1; i < refined.size(); ++i) {
        if (rot_err_deg(best.R, refined[i].R) > 10.0) {
            res.mirror_gap = (best.phot - refined[i].phot) / std::max(std::abs(best.phot), 1.0);
            break;
        }
    }
    const auto t_end = std::chrono::steady_clock::now();
    res.solve_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    return res;
}

PoseResult SparseEdgePoseSolver::solve_tracking(const cv::Mat& gray,
                                                const cv::Vec3d& prev_rvec,
                                                const cv::Vec3d& prev_tvec,
                                                const CharTemplate& tpl,
                                                const CameraIntrinsics& K,
                                                int margin) {
    const auto t_start = std::chrono::steady_clock::now();
    PoseResult res;
    if (gray.empty() || tpl.points.empty()) return res;

    auto pts = project_points(tpl, prev_rvec, prev_tvec, K);
    double min_x = 1e9, max_x = -1e9, min_y = 1e9, max_y = -1e9;
    for (const auto& p : pts) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y)) return res;
        min_x = std::min(min_x, p.x);
        max_x = std::max(max_x, p.x);
        min_y = std::min(min_y, p.y);
        max_y = std::max(max_y, p.y);
    }
    if (min_x > max_x || min_y > max_y || K.fx <= 1.0 || K.fy <= 1.0) return res;
    const int x0 = std::clamp(static_cast<int>(std::floor(min_x)) - margin, 0, gray.cols - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(min_y)) - margin, 0, gray.rows - 1);
    const int x1 = std::clamp(static_cast<int>(std::ceil(max_x)) + margin + 1, x0 + 1, gray.cols);
    const int y1 = std::clamp(static_cast<int>(std::ceil(max_y)) + margin + 1, y0 + 1, gray.rows);
    if (x1 <= x0 + 1 || y1 <= y0 + 1) return res;

    cv::Mat sub, gs;
    gray(cv::Rect(x0, y0, x1 - x0, y1 - y0)).convertTo(sub, CV_32F);
    cv::GaussianBlur(sub, gs, cv::Size(0, 0), 1.0);

    CameraIntrinsics Kc = K;
    Kc.cx -= x0;
    Kc.cy -= y0;

    cv::Matx33d R = rvec_to_mat(prev_rvec);
    cv::Vec3d t = prev_tvec;
    double med = 0.0, frac = 0.0;
    refine_edges_lie(gs, R, t, tpl, Kc, {4, 2, 2}, med, frac, 1.5);
    if (!std::isfinite(med) || !is_finite_pose(R, t)) return res;

    res.valid = true;
    res.rvec = mat_to_rvec(R);
    res.tvec = t;
    res.euler_yxz_deg = rvec_to_euler_yxz_deg(res.rvec);
    res.edge_med_px = med;
    res.edge_frac = frac;
    res.phot_contrast = boundary_phot_contrast(gs, tpl, R, t, Kc);
    res.mirror_gap = std::numeric_limits<double>::infinity();
    const auto t_end = std::chrono::steady_clock::now();
    res.solve_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    return res;
}

}  // namespace tav_pose
