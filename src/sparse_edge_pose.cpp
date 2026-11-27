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

inline cv::Matx33d exp_so3(double wx, double wy, double wz) {
    // Closed-form SO(3) Rodrigues exponential map without cv::Mat allocations.
    const double th2 = wx * wx + wy * wy + wz * wz;
    double c1, c2;
    if (th2 < 1e-8) {
        c1 = 1.0 - th2 * (1.0 / 6.0);
        c2 = 0.5 - th2 * (1.0 / 24.0);
    } else {
        const double th = std::sqrt(th2);
        c1 = std::sin(th) / th;
        c2 = (1.0 - std::cos(th)) / th2;
    }
    return cv::Matx33d(
        1.0 - c2 * (wy * wy + wz * wz), -c1 * wz + c2 * wx * wy,        c1 * wy + c2 * wx * wz,
        c1 * wz + c2 * wx * wy,         1.0 - c2 * (wx * wx + wz * wz), -c1 * wx + c2 * wy * wz,
        -c1 * wy + c2 * wx * wz,        c1 * wx + c2 * wy * wz,         1.0 - c2 * (wx * wx + wy * wy));
}

inline double mat_yaw_deg(const cv::Matx33d& R) {
    return rad2deg(std::asin(std::clamp(R(0, 2), -1.0, 1.0)));
}

inline double est_pixel_step(const cv::Vec6d& dx, double r_mm, double tz, const CameraIntrinsics& K) {
    const double f = 0.5 * (K.fx + K.fy);
    const double inv_z = 1.0 / std::max(std::abs(tz), 1e-3);
    const double w_norm = std::sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);
    const double rot_px = w_norm * f * r_mm * inv_z;
    const double trans_xy_px = (std::abs(dx[3]) * K.fx + std::abs(dx[4]) * K.fy) * inv_z;
    const double trans_z_px = std::abs(dx[5]) * f * r_mm * (inv_z * inv_z);
    return rot_px + trans_xy_px + trans_z_px;
}

inline void sample_chamfer_triplet(const cv::Mat& dt,
                                   const cv::Mat& dtx,
                                   const cv::Mat& dty,
                                   float x,
                                   float y,
                                   double& d_out,
                                   double& gx_out,
                                   double& gy_out) {
    if (!std::isfinite(x) || !std::isfinite(y) || dt.empty()) {
        d_out = gx_out = gy_out = 0.0;
        return;
    }
    const int w = dt.cols;
    const int h = dt.rows;
    const float xc = std::clamp(x, 0.0f, static_cast<float>(w - 1));
    const float yc = std::clamp(y, 0.0f, static_cast<float>(h - 1));
    const int x0 = static_cast<int>(std::floor(xc));
    const int y0 = static_cast<int>(std::floor(yc));
    const int x1 = std::min(x0 + 1, w - 1);
    const int y1 = std::min(y0 + 1, h - 1);
    const float fx = xc - x0;
    const float fy = yc - y0;
    const float w00 = (1.0f - fx) * (1.0f - fy);
    const float w10 = fx * (1.0f - fy);
    const float w01 = (1.0f - fx) * fy;
    const float w11 = fx * fy;

    const float* d0 = dt.ptr<float>(y0);
    const float* d1 = dt.ptr<float>(y1);
    d_out = w00 * d0[x0] + w10 * d0[x1] + w01 * d1[x0] + w11 * d1[x1];

    const float* x0r = dtx.ptr<float>(y0);
    const float* x1r = dtx.ptr<float>(y1);
    gx_out = w00 * x0r[x0] + w10 * x0r[x1] + w01 * x1r[x0] + w11 * x1r[x1];

    const float* y0r = dty.ptr<float>(y0);
    const float* y1r = dty.ptr<float>(y1);
    gy_out = w00 * y0r[x0] + w10 * y0r[x1] + w01 * y1r[x0] + w11 * y1r[x1];
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

inline float sample_bilinear_u8_masked(const cv::Mat& raw_u8,
                                       float x,
                                       float y,
                                       bool mask_led,
                                       int ix0,
                                       int iy0,
                                       int ix1,
                                       int iy1,
                                       float led_thr,
                                       float bg_med) {
    if (!std::isfinite(x) || !std::isfinite(y) || raw_u8.empty()) return 0.0f;
    const int w = raw_u8.cols;
    const int h = raw_u8.rows;
    const float xc = std::clamp(x, 0.0f, static_cast<float>(w - 1));
    const float yc = std::clamp(y, 0.0f, static_cast<float>(h - 1));
    const int x0 = static_cast<int>(std::floor(xc));
    const int y0 = static_cast<int>(std::floor(yc));
    const int x1 = std::min(x0 + 1, w - 1);
    const int y1 = std::min(y0 + 1, h - 1);
    const float fx = xc - x0;
    const float fy = yc - y0;

    const uchar* r0 = raw_u8.ptr<uchar>(y0);
    const uchar* r1 = raw_u8.ptr<uchar>(y1);
    const float v0 = static_cast<float>(r0[x0]) * (1.0f - fx) + static_cast<float>(r0[x1]) * fx;
    const float v1 = static_cast<float>(r1[x0]) * (1.0f - fx) + static_cast<float>(r1[x1]) * fx;
    float v = v0 * (1.0f - fy) + v1 * fy;
    if (mask_led &&
        (x < static_cast<float>(ix0) || x >= static_cast<float>(ix1) ||
         y < static_cast<float>(iy0) || y >= static_cast<float>(iy1)) &&
        v > led_thr) {
        v = bg_med;
    }
    return v;
}

void compute_hist_percentiles_u8(const cv::Mat& core, double& p50, double& p95) {
    if (core.empty()) {
        p50 = 0.0;
        p95 = 0.0;
        return;
    }
    std::array<int, 256> hist{};
    // Bound sampling to ~64x64 grid so huge close-up ROIs (e.g. 1280x1024) stay O(1).
    const int step_y = std::max(1, core.rows / 64);
    const int step_x = std::max(1, core.cols / 64);
    int total = 0;
    for (int y = 0; y < core.rows; y += step_y) {
        const uchar* row = core.ptr<uchar>(y);
        for (int x = 0; x < core.cols; x += step_x) {
            hist[row[x]]++;
            ++total;
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
    // Zero-allocation Gaussian elimination with partial pivoting for H * dx = -g.
    double A[6][7];
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) A[r][c] = H(r, c);
        A[r][6] = -g[r];
    }
    for (int col = 0; col < 6; ++col) {
        int piv = col;
        double best = std::abs(A[col][col]);
        for (int r = col + 1; r < 6; ++r) {
            const double v = std::abs(A[r][col]);
            if (v > best) {
                best = v;
                piv = r;
            }
        }
        if (!(best > 1e-18)) return false;
        if (piv != col) {
            for (int c = col; c < 7; ++c) std::swap(A[col][c], A[piv][c]);
        }
        const double inv_piv = 1.0 / A[col][col];
        for (int r = col + 1; r < 6; ++r) {
            const double f = A[r][col] * inv_piv;
            if (f == 0.0) continue;
            for (int c = col + 1; c < 7; ++c) {
                A[r][c] -= f * A[col][c];
            }
        }
    }
    for (int r = 5; r >= 0; --r) {
        double sum = A[r][6];
        for (int c = r + 1; c < 6; ++c) {
            sum -= A[r][c] * dx[c];
        }
        dx[r] = sum / A[r][r];
        if (!std::isfinite(dx[r])) return false;
    }
    return true;
}

struct EdgeSearchScratch {
    std::vector<float> v;
    std::vector<float> ext_v;
    std::vector<float> dmins;
    std::vector<float> neg_dmins;
    std::vector<float> all_d;
    std::vector<double> off;
    std::vector<uint8_t> valid;
    int n_valid = 0;
};

struct EdgeImgSource {
    const cv::Mat* gs_f32 = nullptr;
    const cv::Mat* raw_u8 = nullptr;
    bool mask_led = false;
    int ix0 = 0;
    int iy0 = 0;
    int ix1 = 0;
    int iy1 = 0;
    float led_thr = 0.0f;
    float bg_med = 0.0f;

    static EdgeImgSource from_blur(const cv::Mat& gs) {
        EdgeImgSource s;
        s.gs_f32 = &gs;
        return s;
    }
};

struct SolverWorkspace {
    std::vector<cv::Point2d> uv;
    std::vector<cv::Point2d> n_img;
    std::vector<cv::Vec3d> pr;
    std::vector<double> X, Y, iz;
    std::vector<double> ds, gxs, gys;
    std::vector<double> abs_e;
    EdgeSearchScratch es;

    void reserve(size_t n) {
        uv.resize(n);
        n_img.resize(n);
        pr.resize(n);
        X.resize(n);
        Y.resize(n);
        iz.resize(n);
        ds.resize(n);
        gxs.resize(n);
        gys.resize(n);
        abs_e.reserve(n);
    }
};

inline void accum_scalar_triple_eq(double px_r,
                                   double py_r,
                                   double pz_r,
                                   double X,
                                   double Y,
                                   double iz,
                                   double nx,
                                   double ny,
                                   const CameraIntrinsics& K,
                                   double residual,
                                   double weight,
                                   cv::Matx66d& H,
                                   cv::Vec6d& g) {
    // Scalar normal Jacobian via triple product: n^T J_pi (domega x Pr + dt) = [Pr x g, g] * [domega; dt].
    const double alpha = K.fx * nx * iz;
    const double beta = K.fy * ny * iz;
    const double gamma = -(alpha * X + beta * Y) * iz;
    const double Jr[6] = {
        py_r * gamma - pz_r * beta,
        pz_r * alpha - px_r * gamma,
        px_r * beta - py_r * alpha,
        alpha,
        beta,
        gamma,
    };
    const double wr = weight * residual;
    for (int r = 0; r < 6; ++r) {
        g[r] += wr * Jr[r];
        const double wjr = weight * Jr[r];
        for (int c = r; c < 6; ++c) {
            H(r, c) += wjr * Jr[c];
        }
    }
}

inline void mirror_upper_sym_6x6(cv::Matx66d& H) {
    for (int r = 1; r < 6; ++r) {
        for (int c = 0; c < r; ++c) {
            H(r, c) = H(c, r);
        }
    }
}

void edge_search_src(const EdgeImgSource& src,
                     const std::vector<cv::Point2d>& pts_img,
                     const std::vector<cv::Point2d>& n_img,
                     int rad,
                     EdgeSearchScratch& sc,
                     double* out_phot = nullptr) {
    const int n_pts = static_cast<int>(pts_img.size());
    const int len = 2 * rad + 1;
    const int dlen = len - 2;

    sc.off.assign(n_pts, 0.0);
    sc.valid.assign(n_pts, 0);
    sc.n_valid = 0;
    sc.v.resize(len);
    sc.dmins.resize(n_pts);
    sc.neg_dmins.resize(n_pts);
    sc.all_d.resize(static_cast<size_t>(n_pts) * static_cast<size_t>(dlen));
    if (src.raw_u8) sc.ext_v.resize(len + 4);

    double phot_sum = 0.0;
    for (int i = 0; i < n_pts; ++i) {
        const float px = static_cast<float>(pts_img[i].x);
        const float py = static_cast<float>(pts_img[i].y);
        const float nx = static_cast<float>(n_img[i].x);
        const float ny = static_cast<float>(n_img[i].y);
        if (src.raw_u8) {
            const int ext_len = len + 4;
            for (int k = 0; k < ext_len; ++k) {
                const float o = static_cast<float>(k - rad - 2);
                sc.ext_v[k] = sample_bilinear_u8_masked(
                    *src.raw_u8, px + o * nx, py + o * ny,
                    src.mask_led, src.ix0, src.iy0, src.ix1, src.iy1, src.led_thr, src.bg_med);
            }
            constexpr float kW0 = 1.0f / 16.0f;
            constexpr float kW1 = 4.0f / 16.0f;
            constexpr float kW2 = 6.0f / 16.0f;
            for (int k = 0; k < len; ++k) {
                sc.v[k] = kW0 * (sc.ext_v[k] + sc.ext_v[k + 4]) +
                          kW1 * (sc.ext_v[k + 1] + sc.ext_v[k + 3]) +
                          kW2 * sc.ext_v[k + 2];
            }
        } else {
            for (int k = 0; k < len; ++k) {
                const float o = static_cast<float>(k - rad);
                sc.v[k] = sample_bilinear_f32(*src.gs_f32, px + o * nx, py + o * ny);
            }
        }
        if (out_phot && rad >= 1) {
            phot_sum += static_cast<double>(sc.v[rad - 1] - sc.v[rad + 1]);
        }
        float* di = sc.all_d.data() + static_cast<size_t>(i) * static_cast<size_t>(dlen);
        float dmin = std::numeric_limits<float>::infinity();
        for (int k = 0; k < dlen; ++k) {
            const float dv = 0.5f * (sc.v[k + 2] - sc.v[k]);
            di[k] = dv;
            if (dv < dmin) dmin = dv;
        }
        sc.dmins[i] = dmin;
        sc.neg_dmins[i] = -dmin;
    }
    if (out_phot) {
        *out_phot = phot_sum / std::max(n_pts, 1);
    }

    std::nth_element(sc.neg_dmins.begin(), sc.neg_dmins.begin() + n_pts / 2, sc.neg_dmins.end());
    const float contrast = sc.neg_dmins[n_pts / 2];
    const float thr = -std::max(0.35f * contrast, 0.8f);

    for (int i = 0; i < n_pts; ++i) {
        if (!(sc.dmins[i] < thr)) continue;
        const float* di = sc.all_d.data() + static_cast<size_t>(i) * static_cast<size_t>(dlen);
        int best_j = -1;
        float best_abs_o = std::numeric_limits<float>::infinity();
        const float half_min = 0.5f * sc.dmins[i];
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
        sc.off[i] = o_j + sub;
        sc.valid[i] = 1;
        sc.n_valid++;
    }
}

void project_points_chamfer(const CharTemplate& tpl,
                            const cv::Matx33d& R,
                            const cv::Vec3d& t,
                            const CameraIntrinsics& K,
                            SolverWorkspace& ws) {
    const size_t N = tpl.points.size();
    ws.uv.resize(N);
    ws.pr.resize(N);
    ws.X.resize(N);
    ws.Y.resize(N);
    ws.iz.resize(N);

    for (size_t i = 0; i < N; ++i) {
        const auto& mp = tpl.points[i];
        const double px_r = R(0, 0) * mp.px + R(0, 1) * mp.py;
        const double py_r = R(1, 0) * mp.px + R(1, 1) * mp.py;
        const double pz_r = R(2, 0) * mp.px + R(2, 1) * mp.py;

        const double X = px_r + t[0];
        const double Y = py_r + t[1];
        const double Z = std::max(pz_r + t[2], 1e-6);
        const double iz = 1.0 / Z;

        ws.pr[i] = cv::Vec3d(px_r, py_r, pz_r);
        ws.X[i] = X;
        ws.Y[i] = Y;
        ws.iz[i] = iz;
        ws.uv[i] = cv::Point2d(K.fx * X * iz + K.cx, K.fy * Y * iz + K.cy);
    }
}

void project_points_and_normals(const CharTemplate& tpl,
                                const cv::Matx33d& R,
                                const cv::Vec3d& t,
                                const CameraIntrinsics& K,
                                SolverWorkspace& ws,
                                bool need_jac_state) {
    const size_t N = tpl.points.size();
    ws.uv.resize(N);
    ws.n_img.resize(N);
    if (need_jac_state) {
        ws.pr.resize(N);
        ws.X.resize(N);
        ws.Y.resize(N);
        ws.iz.resize(N);
    }

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

        ws.uv[i] = cv::Point2d(K.fx * X * iz + K.cx, K.fy * Y * iz + K.cy);
        if (need_jac_state) {
            ws.pr[i] = cv::Vec3d(px_r, py_r, pz_r);
            ws.X[i] = X;
            ws.Y[i] = Y;
            ws.iz[i] = iz;
        }

        const double fx_iz = K.fx * iz;
        const double fy_iz = K.fy * iz;
        const double fx_X_iz2 = -K.fx * X * iz2;
        const double fy_Y_iz2 = -K.fy * Y * iz2;

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
        ws.n_img[i] = cv::Point2d(nix, niy);
    }
}

bool apply_lie_step(cv::Matx33d& R, cv::Vec3d& t, const cv::Vec6d& dx) {
    for (int k = 0; k < 6; ++k) {
        if (!std::isfinite(dx[k])) return false;
    }
    const cv::Matx33d R_next = exp_so3(dx[0], dx[1], dx[2]) * R;
    const cv::Vec3d t_next = t + cv::Vec3d(dx[3], dx[4], dx[5]);
    if (!is_finite_pose(R_next, t_next)) return false;
    R = R_next;
    t = t_next;
    return true;
}

double refine_chamfer_gated(const cv::Mat& dt,
                            const cv::Mat& dtx,
                            const cv::Mat& dty,
                            cv::Matx33d& R,
                            cv::Vec3d& t,
                            const CharTemplate& tpl,
                            const CameraIntrinsics& K,
                            SolverWorkspace& ws,
                            double r_mm,
                            int iters = 5,
                            double gate = 8.0,
                            double huber = 3.0) {
    const size_t N = tpl.points.size();
    ws.ds.resize(N);
    ws.gxs.resize(N);
    ws.gys.resize(N);

    for (int it = 0; it < iters; ++it) {
        project_points_chamfer(tpl, R, t, K, ws);
        int vis_cnt = 0;
        double sum_all = 0.0;
        double sum_vis = 0.0;
        for (size_t i = 0; i < N; ++i) {
            sample_chamfer_triplet(dt, dtx, dty,
                                   static_cast<float>(ws.uv[i].x),
                                   static_cast<float>(ws.uv[i].y),
                                   ws.ds[i], ws.gxs[i], ws.gys[i]);
            sum_all += ws.ds[i];
            if (ws.ds[i] <= gate) {
                ++vis_cnt;
                sum_vis += ws.ds[i];
            }
        }
        const bool use_gate = (vis_cnt >= 12);
        const double mean_d = use_gate ? (sum_vis / vis_cnt) : (sum_all / std::max<size_t>(N, 1));

        cv::Matx66d H = cv::Matx66d::zeros();
        cv::Vec6d g = cv::Vec6d::zeros();
        for (size_t i = 0; i < N; ++i) {
            if (use_gate && ws.ds[i] > gate) continue;
            const double d = ws.ds[i];
            const double w = (d <= huber) ? 1.0 : huber / std::max(d, 1e-9);
            accum_scalar_triple_eq(ws.pr[i][0], ws.pr[i][1], ws.pr[i][2],
                                   ws.X[i], ws.Y[i], ws.iz[i],
                                   ws.gxs[i], ws.gys[i], K, d, w, H, g);
        }
        mirror_upper_sym_6x6(H);
        double tr = 0.0;
        for (int k = 0; k < 6; ++k) tr += H(k, k);
        const double damp = 1e-3 * tr / 6.0 + 1e-9;
        for (int k = 0; k < 6; ++k) H(k, k) += damp;

        cv::Vec6d dx;
        if (!solve_6x6(H, g, dx)) break;
        // 0.7x damping inside the Chamfer crease avoids V-valley period-2 limit cycles.
        if (mean_d < 1.2) {
            for (int k = 0; k < 6; ++k) dx[k] *= 0.7;
        }
        if (!apply_lie_step(R, t, dx)) break;
        if (est_pixel_step(dx, r_mm, t[2], K) < 0.05) break;
    }

    // Final truncated Chamfer residual score for post-Chamfer branch deduplication
    project_points_chamfer(tpl, R, t, K, ws);
    double sum_trunc = 0.0;
    for (size_t i = 0; i < N; ++i) {
        const double d = static_cast<double>(sample_bilinear_f32(
            dt, static_cast<float>(ws.uv[i].x), static_cast<float>(ws.uv[i].y)));
        sum_trunc += std::min(d, gate);
    }
    return sum_trunc / std::max<size_t>(N, 1);
}

bool run_single_edge_step(const EdgeImgSource& src,
                          const CharTemplate& tpl,
                          const CameraIntrinsics& K,
                          int rad,
                          double huber,
                          double damp_scale,
                          double damp_eps,
                          cv::Matx33d& R,
                          cv::Vec3d& t,
                          SolverWorkspace& ws,
                          double& out_med,
                          double& out_p90,
                          double& out_frac,
                          cv::Vec6d& out_dx) {
    const int min_valid = std::min(12, std::max(6, static_cast<int>(tpl.points.size()) / 2));
    project_points_and_normals(tpl, R, t, K, ws, true);
    edge_search_src(src, ws.uv, ws.n_img, rad, ws.es);
    if (ws.es.n_valid < min_valid) return false;

    cv::Matx66d H = cv::Matx66d::zeros();
    cv::Vec6d g = cv::Vec6d::zeros();
    ws.abs_e.clear();

    for (size_t i = 0; i < tpl.points.size(); ++i) {
        if (!ws.es.valid[i]) continue;
        const double e = -ws.es.off[i];
        const double ae = std::abs(e);
        ws.abs_e.push_back(ae);
        const double w = (ae <= huber) ? 1.0 : huber / std::max(ae, 1e-9);
        accum_scalar_triple_eq(ws.pr[i][0], ws.pr[i][1], ws.pr[i][2],
                               ws.X[i], ws.Y[i], ws.iz[i],
                               ws.n_img[i].x, ws.n_img[i].y, K, e, w, H, g);
    }
    mirror_upper_sym_6x6(H);
    double tr = 0.0;
    for (int k = 0; k < 6; ++k) tr += H(k, k);
    const double damp = damp_scale * tr / 6.0 + damp_eps;
    for (int k = 0; k < 6; ++k) H(k, k) += damp;

    if (!solve_6x6(H, g, out_dx)) return false;
    if (!apply_lie_step(R, t, out_dx)) return false;

    if (!ws.abs_e.empty()) {
        std::sort(ws.abs_e.begin(), ws.abs_e.end());
        out_med = ws.abs_e[ws.abs_e.size() / 2];
        const size_t idx90 = static_cast<size_t>(std::round(0.90 * static_cast<double>(ws.abs_e.size() - 1)));
        out_p90 = ws.abs_e[idx90];
    }
    out_frac = static_cast<double>(ws.es.n_valid) / static_cast<double>(tpl.points.size());
    return true;
}

void refine_edges_lie(const cv::Mat& gs,
                      cv::Matx33d& R,
                      cv::Vec3d& t,
                      const CharTemplate& tpl,
                      const CameraIntrinsics& K,
                      const std::vector<int>& radii,
                      SolverWorkspace& ws,
                      double r_mm,
                      double& out_med,
                      double& out_frac,
                      double huber = 1.5) {
    out_med = std::numeric_limits<double>::quiet_NaN();
    out_frac = 0.0;
    const EdgeImgSource src = EdgeImgSource::from_blur(gs);
    for (int rad : radii) {
        double p90 = 0.0;
        cv::Vec6d dx;
        if (!run_single_edge_step(src, tpl, K, rad, huber, 1e-6, 1e-12, R, t, ws,
                                  out_med, p90, out_frac, dx)) {
            break;
        }
        if (rad <= 3 && est_pixel_step(dx, r_mm, t[2], K) < 0.04) break;
    }
}

void eval_edge_residual_and_phot(const EdgeImgSource& src,
                                 const cv::Matx33d& R,
                                 const cv::Vec3d& t,
                                 const CharTemplate& tpl,
                                 const CameraIntrinsics& K,
                                 int rad,
                                 SolverWorkspace& ws,
                                 double& out_med,
                                 double& out_frac,
                                 double& out_phot) {
    out_med = std::numeric_limits<double>::quiet_NaN();
    out_frac = 0.0;
    out_phot = std::numeric_limits<double>::quiet_NaN();
    project_points_and_normals(tpl, R, t, K, ws, false);
    edge_search_src(src, ws.uv, ws.n_img, rad, ws.es, &out_phot);
    if (ws.es.n_valid <= 0) return;
    ws.abs_e.clear();
    for (size_t i = 0; i < tpl.points.size(); ++i) {
        if (ws.es.valid[i]) ws.abs_e.push_back(std::abs(ws.es.off[i]));
    }
    if (ws.abs_e.empty()) return;
    std::nth_element(ws.abs_e.begin(), ws.abs_e.begin() + ws.abs_e.size() / 2, ws.abs_e.end());
    out_med = ws.abs_e[ws.abs_e.size() / 2];
    out_frac = static_cast<double>(ws.es.n_valid) / static_cast<double>(tpl.points.size());
}

void refine_edges_lie_adaptive(const EdgeImgSource& src,
                               cv::Matx33d& R,
                               cv::Vec3d& t,
                               const CharTemplate& tpl,
                               const CameraIntrinsics& K,
                               int edge_steps,
                               int r0,
                               SolverWorkspace& ws,
                               double r_mm,
                               double& out_med,
                               double& out_frac,
                               double& out_phot,
                               int eval_rad = 4) {
    int cur_rad = std::max(3, r0);
    const int next_max = std::clamp(eval_rad + 1, 5, 8);
    for (int it = 0; it < edge_steps; ++it) {
        double step_med = 0.0, step_p90 = 0.0, step_frac = 0.0;
        cv::Vec6d dx;
        if (!run_single_edge_step(src, tpl, K, cur_rad, 1.5, 1e-5, 1e-11, R, t, ws,
                                  step_med, step_p90, step_frac, dx)) {
            break;
        }
        if (it + 1 < edge_steps) {
            if (cur_rad <= 3 && est_pixel_step(dx, r_mm, t[2], K) < 0.03) break;
            cur_rad = std::clamp(static_cast<int>(std::ceil(std::max(2.0 * step_p90, 3.0))),
                                 3, std::min(next_max, r0));
        }
    }
    eval_edge_residual_and_phot(src, R, t, tpl, K, eval_rad, ws, out_med, out_frac, out_phot);
}

cv::Mat build_led_masked_sub_f32(const cv::Mat& gray,
                                 int ix0,
                                 int iy0,
                                 int ix1,
                                 int iy1,
                                 int x0,
                                 int y0,
                                 int x1,
                                 int y1,
                                 double led_thr,
                                 double bg_med) {
    cv::Mat sub;
    gray(cv::Rect(x0, y0, x1 - x0, y1 - y0)).convertTo(sub, CV_32F);
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
    return sub;
}

void linear_quantile_range(std::vector<float>& vals, double q_lo, double q_hi, float& out_lo, float& out_hi) {
    if (vals.empty()) {
        out_lo = 0.0f;
        out_hi = 0.0f;
        return;
    }
    std::sort(vals.begin(), vals.end());
    auto at_q = [&](double q) -> float {
        const double pos = std::clamp(q, 0.0, 1.0) * static_cast<double>(vals.size() - 1);
        const size_t lo = static_cast<size_t>(std::floor(pos));
        const size_t hi = std::min(lo + 1, vals.size() - 1);
        const double frac = pos - static_cast<double>(lo);
        return static_cast<float>(vals[lo] * (1.0 - frac) + vals[hi] * frac);
    };
    out_lo = at_q(q_lo);
    out_hi = at_q(q_hi);
}

double boundary_phot_contrast(const cv::Mat& gs,
                                const CharTemplate& tpl,
                                const cv::Matx33d& R,
                                const cv::Vec3d& t,
                                const CameraIntrinsics& K,
                                SolverWorkspace& ws) {
    project_points_and_normals(tpl, R, t, K, ws, false);
    double sum = 0.0;
    for (size_t i = 0; i < ws.uv.size(); ++i) {
        const float vi = sample_bilinear_f32(
            gs, static_cast<float>(ws.uv[i].x - ws.n_img[i].x), static_cast<float>(ws.uv[i].y - ws.n_img[i].y));
        const float vo = sample_bilinear_f32(
            gs, static_cast<float>(ws.uv[i].x + ws.n_img[i].x), static_cast<float>(ws.uv[i].y + ws.n_img[i].y));
        sum += static_cast<double>(vi - vo);
    }
    return sum / std::max<size_t>(ws.uv.size(), 1);
}

struct SeedHyp {
    double score;
    cv::Matx33d R;
    int yaw_deg;
};

struct ChamferCand {
    double dt_score;
    cv::Matx33d R;
    cv::Vec3d t;
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
    const cv::Matx33d R = exp_so3(rvec[0], rvec[1], rvec[2]);
    const double yaw = rad2deg(std::asin(std::clamp(R(0, 2), -1.0, 1.0)));
    const double pitch = rad2deg(std::atan2(-R(1, 2), R(2, 2)));
    const double roll = rad2deg(std::atan2(-R(0, 1), R(0, 0)));
    return cv::Vec3d(yaw, pitch, roll);
}

std::vector<cv::Point2d> SparseEdgePoseSolver::project_points(const CharTemplate& tpl,
                                                              const cv::Vec3d& rvec,
                                                              const cv::Vec3d& tvec,
                                                              const CameraIntrinsics& K) {
    const cv::Matx33d R = exp_so3(rvec[0], rvec[1], rvec[2]);
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

    cv::Mat sub = build_led_masked_sub_f32(gray, ix0, iy0, ix1, iy1, x0, y0, x1, y1, led_thr, bg_med);

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

    SolverWorkspace ws;
    ws.reserve(tpl.points.size());
    const double r_mm = 0.5 * std::hypot(tpl.width_mm, tpl.height_mm);
    const double chamfer_gate = std::max(8.0, 0.06 * h_c);
    const int edge_r0 = std::clamp(static_cast<int>(std::round(0.04 * h_c)), 6, 14);
    std::vector<int> radii;
    if (edge_r0 > 6) {
        radii = {edge_r0, 6, 4, 3, 2};
    } else {
        radii = {6, 4, 3, 2};
    }

    // Stage A: Damped Chamfer alignment (max 5 iters, pixel-step early stop) + 4-deg deduplication
    std::vector<ChamferCand> ch_results;
    ch_results.reserve(cands.size());
    for (const auto& cand : cands) {
        cv::Matx33d R = cand.R;
        cv::Vec3d t = t0;
        const double ch_dt = refine_chamfer_gated(dt, dtx, dty, R, t, tpl, Kc, ws, r_mm, 5, chamfer_gate, 3.0);
        ch_results.push_back({ch_dt, R, t});
    }
    std::sort(ch_results.begin(), ch_results.end(),
              [](const ChamferCand& a, const ChamferCand& b) { return a.dt_score < b.dt_score; });

    std::vector<ChamferCand> edge_cands;
    edge_cands.reserve(4);
    for (const auto& item : ch_results) {
        bool dup = false;
        for (const auto& kept : edge_cands) {
            if (rot_err_deg(item.R, kept.R) <= 4.0) {
                dup = true;
                break;
            }
        }
        if (dup) continue;
        edge_cands.push_back(item);
        if (edge_cands.size() >= 4) break;
    }

    // Stage B: 1D Edge IRLS on top <=4 distinct Chamfer basins; early exit once both yaw signs covered + locked
    std::vector<RefinedCand> refined;
    refined.reserve(edge_cands.size());
    bool high_conf_lock = false;
    bool best_pos_yaw = false;
    bool best_neg_yaw = false;
    for (const auto& cand : edge_cands) {
        if (high_conf_lock && best_pos_yaw && best_neg_yaw) break;
        cv::Matx33d R = cand.R;
        cv::Vec3d t = cand.t;
        double med = 0.0, frac = 0.0;
        refine_edges_lie(gs, R, t, tpl, Kc, radii, ws, r_mm, med, frac, 1.5);
        if (!std::isfinite(med) || !is_finite_pose(R, t)) continue;
        const double phot = boundary_phot_contrast(gs, tpl, R, t, Kc, ws);
        if (!std::isfinite(phot)) continue;
        const double sc = phot * (frac * frac) - 2.0 * std::max(med - 0.5, 0.0);
        refined.push_back({R, t, sc, phot, med, frac});
        const double yaw = mat_yaw_deg(R);
        if (yaw >= -3.0) best_pos_yaw = true;
        if (yaw <= 3.0) best_neg_yaw = true;
        if (frac >= 0.80 && med <= 0.45 && phot > 8.0) {
            high_conf_lock = true;
        }
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

cv::Mat ensure_gray_u8(const cv::Mat& src) {
    if (src.channels() == 1 && src.depth() == CV_8U) return src;
    cv::Mat gray1;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray1, cv::COLOR_BGR2GRAY);
        if (gray1.depth() != CV_8U) gray1.convertTo(gray1, CV_8U);
    } else {
        src.convertTo(gray1, CV_8U);
    }
    return gray1;
}

RoiCornerPrior::RoiCornerPrior(const std::string& onnx_path, GpuMode gpu_mode) {
    net_ = cv::dnn::readNetFromONNX(onnx_path);
    if (net_.empty()) {
        throw std::runtime_error("Failed to load corner prior ONNX: " + onnx_path);
    }
    const auto cv_names = net_.getUnconnectedOutLayersNames();
    out_names_.assign(cv_names.begin(), cv_names.end());

    if (onnx_path.find("32x32") != std::string::npos) {
        roi_sz_ = 32;
    } else if (onnx_path.find("48x48") != std::string::npos) {
        roi_sz_ = 48;
    } else if (onnx_path.find("64x64") != std::string::npos) {
        roi_sz_ = 64;
    } else if (!out_names_.empty()) {
        roi_sz_ = 64;
        for (int cand : {64, 48, 32}) {
            try {
                cv::dnn::Net probe = cv::dnn::readNetFromONNX(onnx_path);
                probe.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                probe.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                const int dims[4] = {1, 1, cand, cand};
                cv::Mat dummy = cv::Mat::zeros(4, dims, CV_32F);
                probe.setInput(dummy);
                cv::Mat y = probe.forward(out_names_[0]);
                if (!y.empty() && y.total() >= 24) {
                    roi_sz_ = cand;
                    break;
                }
            } catch (...) {
            }
        }
    }
    backend_status_ = configure_dnn_backend(
        net_, onnx_path, out_names_, cv::Size(roi_sz_, roi_sz_), 1, gpu_mode);
}

bool RoiCornerPrior::predict(const cv::Mat& gray_u8,
                             const cv::Rect2f& box,
                             int cls_id,
                             std::array<cv::Point2d, 4>& out_corners) const {
    if (net_.empty() || out_names_.empty() || gray_u8.empty()) return false;

    const cv::Mat gray1 = ensure_gray_u8(gray_u8);

    const int kRoiSz = roi_sz_;
    constexpr double kPadScale = 1.48;
    const double bw = std::max(static_cast<double>(box.width), 4.0);
    const double bh = std::max(static_cast<double>(box.height), 4.0);
    const double cx = static_cast<double>(box.x) + 0.5 * static_cast<double>(box.width);
    const double cy = static_cast<double>(box.y) + 0.5 * static_cast<double>(box.height);
    const double S = std::max(kPadScale * std::max(bw, bh), 12.0);
    const double x0 = cx - 0.5 * S;
    const double y0 = cy - 0.5 * S;

    const float scale = static_cast<float>(kRoiSz / S);
    const cv::Matx23f M(scale, 0.0f, static_cast<float>(-x0 * kRoiSz / S),
                        0.0f, scale, static_cast<float>(-y0 * kRoiSz / S));
    cv::Mat patch_u8;
    const int interp = (S > static_cast<double>(kRoiSz)) ? cv::INTER_AREA : cv::INTER_LINEAR;
    cv::warpAffine(gray1, patch_u8, M, cv::Size(kRoiSz, kRoiSz), interp, cv::BORDER_REPLICATE);

    const int ix0 = std::clamp(static_cast<int>(std::floor((box.x - x0) * kRoiSz / S)), 0, kRoiSz - 2);
    const int iy0 = std::clamp(static_cast<int>(std::floor((box.y - y0) * kRoiSz / S)), 0, kRoiSz - 2);
    const int ix1 = std::clamp(static_cast<int>(std::ceil((box.x + box.width - x0) * kRoiSz / S)), ix0 + 2, kRoiSz);
    const int iy1 = std::clamp(static_cast<int>(std::ceil((box.y + box.height - y0) * kRoiSz / S)), iy0 + 2, kRoiSz);

    std::vector<float> core_vals;
    core_vals.reserve(static_cast<size_t>((iy1 - iy0) * (ix1 - ix0)));
    for (int y = iy0; y < iy1; ++y) {
        const uchar* row = patch_u8.ptr<uchar>(y);
        for (int x = ix0; x < ix1; ++x) {
            core_vals.push_back(static_cast<float>(row[x]));
        }
    }
    float lo = 0.0f, hi = 255.0f;
    linear_quantile_range(core_vals, 0.05, 0.95, lo, hi);
    const float inv_span = 1.0f / std::max(hi - lo, 12.0f);

    const int dims[4] = {1, 1, kRoiSz, kRoiSz};
    cv::Mat blob(4, dims, CV_32F);
    float* dst = blob.ptr<float>();
    for (int y = 0; y < kRoiSz; ++y) {
        const uchar* row = patch_u8.ptr<uchar>(y);
        for (int x = 0; x < kRoiSz; ++x) {
            const float norm_v = (static_cast<float>(row[x]) - lo) * inv_span;
            dst[y * kRoiSz + x] = std::clamp(norm_v, 0.0f, 1.0f);
        }
    }

    cv::Mat out_raw;
    {
        std::lock_guard<std::mutex> lock(mu_);
        net_.setInput(blob);
        out_raw = net_.forward(out_names_[0]);
    }
    if (out_raw.empty() || out_raw.total() < 24) return false;
    cv::Mat out_f = out_raw.isContinuous() ? out_raw : out_raw.clone();
    if (out_f.depth() != CV_32F) out_f.convertTo(out_f, CV_32F);

    const int cid = std::clamp(cls_id, 0, 2);
    const float* coords = out_f.ptr<float>() + cid * 8;
    for (int k = 0; k < 4; ++k) {
        const double ux = static_cast<double>(coords[2 * k]);
        const double uy = static_cast<double>(coords[2 * k + 1]);
        if (!std::isfinite(ux) || !std::isfinite(uy)) return false;
        out_corners[k] = cv::Point2d(x0 + ux * S, y0 + uy * S);
    }
    return true;
}

PoseResult SparseEdgePoseSolver::solve_cold_start_hybrid(const cv::Mat& gray,
                                                         const cv::Rect2f& box,
                                                         const CharTemplate& tpl,
                                                         const CameraIntrinsics& K,
                                                         const RoiCornerPrior* corner_prior,
                                                         int margin,
                                                         bool sparse_1d_raw) {
    if (!corner_prior || corner_prior->empty()) {
        return solve_cold_start(gray, box, tpl, K, margin);
    }
    using Clock = std::chrono::steady_clock;
    auto ms_between = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    const auto t_start = Clock::now();
    PoseResult res;
    if (gray.empty() || tpl.points.empty() || K.fx <= 1.0 || K.fy <= 1.0) return res;

    const cv::Mat gray1 = ensure_gray_u8(gray);
    auto fallback = [&]() {
        PoseResult fb = solve_cold_start(gray1, box, tpl, K, margin);
        if (fb.valid) {
            fb.used_fallback = true;
            fb.timing = res.timing;
            fb.solve_ms = ms_between(t_start, Clock::now());
        }
        return fb;
    };

    std::array<cv::Point2d, 4> p4{};
    if (!corner_prior->predict(gray1, box, tpl.cls_id, p4)) {
        res.timing.corner_ms = ms_between(t_start, Clock::now());
        return fallback();
    }
    const auto t_corner = Clock::now();
    res.timing.corner_ms = ms_between(t_start, t_corner);

    const double hw = 0.5 * tpl.width_mm;
    const double hh = 0.5 * tpl.height_mm;
    const std::vector<cv::Point3d> rect3 = {
        cv::Point3d(-hw, -hh, 0.0),
        cv::Point3d( hw, -hh, 0.0),
        cv::Point3d( hw,  hh, 0.0),
        cv::Point3d(-hw,  hh, 0.0),
    };
    const std::vector<cv::Point2d> img4(p4.begin(), p4.end());
    const cv::Mat K_mat = (cv::Mat_<double>(3, 3) <<
        K.fx, 0.0, K.cx,
        0.0, K.fy, K.cy,
        0.0, 0.0, 1.0);
    std::vector<cv::Mat> rvecs, tvecs;
    const int n_sol = cv::solvePnPGeneric(
        rect3, img4, K_mat, cv::noArray(), rvecs, tvecs, false, cv::SOLVEPNP_IPPE);
    const auto t_ippe = Clock::now();
    res.timing.ippe_ms = ms_between(t_corner, t_ippe);
    if (n_sol <= 0 || rvecs.empty() || tvecs.empty()) {
        return fallback();
    }

    const int W_im = gray1.cols;
    const int H_im = gray1.rows;
    const double bx0 = box.x;
    const double by0 = box.y;
    const double bx1 = box.x + box.width;
    const double by1 = box.y + box.height;
    const double bh = std::max(by1 - by0, 4.0);

    cv::Mat gs;
    EdgeImgSource edge_src;
    CameraIntrinsics Kc = K;
    std::array<cv::Point2d, 4> pts_ref = p4;

    if (sparse_1d_raw) {
        edge_src.raw_u8 = &gray1;
        edge_src.ix0 = std::clamp(static_cast<int>(std::floor(bx0)), 0, W_im - 1);
        edge_src.iy0 = std::clamp(static_cast<int>(std::floor(by0)), 0, H_im - 1);
        edge_src.ix1 = std::clamp(static_cast<int>(std::ceil(bx1)), edge_src.ix0 + 1, W_im);
        edge_src.iy1 = std::clamp(static_cast<int>(std::ceil(by1)), edge_src.iy0 + 1, H_im);
        const cv::Mat core = gray1(cv::Rect(edge_src.ix0, edge_src.iy0,
                                            edge_src.ix1 - edge_src.ix0,
                                            edge_src.iy1 - edge_src.iy0));
        if (core.total() >= 16) {
            double bg_med = 0.0, p95_in = 0.0;
            compute_hist_percentiles_u8(core, bg_med, p95_in);
            edge_src.mask_led = true;
            edge_src.bg_med = static_cast<float>(bg_med);
            edge_src.led_thr = static_cast<float>(std::max(p95_in * 1.5, 80.0));
        }
    } else {
        double min_px = bx0, max_px = bx1, min_py = by0, max_py = by1;
        for (const auto& pt : p4) {
            min_px = std::min(min_px, pt.x);
            max_px = std::max(max_px, pt.x);
            min_py = std::min(min_py, pt.y);
            max_py = std::max(max_py, pt.y);
        }
        const int roi_pad = std::max(margin, 18);
        const int x0 = std::max(0, static_cast<int>(std::floor(min_px)) - roi_pad);
        const int y0 = std::max(0, static_cast<int>(std::floor(min_py)) - roi_pad);
        const int x1 = std::min(W_im, static_cast<int>(std::ceil(max_px)) + roi_pad + 1);
        const int y1 = std::min(H_im, static_cast<int>(std::ceil(max_py)) + roi_pad + 1);
        if (x1 <= x0 + 2 || y1 <= y0 + 2) return fallback();

        const int ix0 = std::clamp(static_cast<int>(std::floor(bx0)), x0, x1 - 1);
        const int iy0 = std::clamp(static_cast<int>(std::floor(by0)), y0, y1 - 1);
        const int ix1 = std::clamp(static_cast<int>(std::ceil(bx1)), ix0 + 1, x1);
        const int iy1 = std::clamp(static_cast<int>(std::ceil(by1)), iy0 + 1, y1);
        const cv::Mat core = gray1(cv::Rect(ix0, iy0, ix1 - ix0, iy1 - iy0));

        cv::Mat sub;
        if (core.total() >= 16) {
            double bg_med = 0.0, p95_in = 0.0;
            compute_hist_percentiles_u8(core, bg_med, p95_in);
            const double led_thr = std::max(p95_in * 1.5, 80.0);
            sub = build_led_masked_sub_f32(gray1, ix0, iy0, ix1, iy1, x0, y0, x1, y1, led_thr, bg_med);
        } else {
            gray1(cv::Rect(x0, y0, x1 - x0, y1 - y0)).convertTo(sub, CV_32F);
        }
        cv::GaussianBlur(sub, gs, cv::Size(0, 0), 1.0);
        edge_src = EdgeImgSource::from_blur(gs);
        Kc.cx -= x0;
        Kc.cy -= y0;
        for (int k = 0; k < 4; ++k) {
            pts_ref[k] = cv::Point2d(p4[k].x - x0, p4[k].y - y0);
        }
    }
    const auto t_prep = Clock::now();
    res.timing.roi_prep_ms = ms_between(t_ippe, t_prep);

    const int r0 = std::clamp(static_cast<int>(std::round(0.035 * bh)), 4, 16);
    const int eval_rad = std::clamp(static_cast<int>(std::round(0.012 * bh)), 4, 8);
    const int edge_steps = (bh < 58.0) ? 1 : 3;
    const double r_mm = 0.5 * std::hypot(tpl.width_mm, tpl.height_mm);
    SolverWorkspace ws;
    ws.reserve(tpl.points.size());

    std::vector<RefinedCand> refined;
    refined.reserve(2);
    const int n_try = std::min(n_sol, 2);
    for (int si = 0; si < n_try; ++si) {
        cv::Mat rm = rvecs[si].isContinuous() ? rvecs[si] : rvecs[si].clone();
        cv::Mat tm = tvecs[si].isContinuous() ? tvecs[si] : tvecs[si].clone();
        if (rm.depth() != CV_64F) rm.convertTo(rm, CV_64F);
        if (tm.depth() != CV_64F) tm.convertTo(tm, CV_64F);
        if (rm.total() < 3 || tm.total() < 3) continue;

        const double* rp = rm.ptr<double>();
        const double* tp = tm.ptr<double>();
        cv::Matx33d R = exp_so3(rp[0], rp[1], rp[2]);
        cv::Vec3d t(tp[0], tp[1], tp[2]);
        if (!is_finite_pose(R, t)) continue;

        double med = 0.0, frac = 0.0, phot = 0.0;
        refine_edges_lie_adaptive(edge_src, R, t, tpl, Kc, edge_steps, r0, ws, r_mm, med, frac, phot, eval_rad);
        if (!std::isfinite(med) || !std::isfinite(phot) || !is_finite_pose(R, t)) continue;

        double c_rep = 0.0;
        for (int k = 0; k < 4; ++k) {
            const double X = R(0, 0) * rect3[k].x + R(0, 1) * rect3[k].y + t[0];
            const double Y = R(1, 0) * rect3[k].x + R(1, 1) * rect3[k].y + t[1];
            const double Z = std::max(R(2, 0) * rect3[k].x + R(2, 1) * rect3[k].y + t[2], 1e-6);
            const double px = Kc.fx * (X / Z) + Kc.cx;
            const double py = Kc.fy * (Y / Z) + Kc.cy;
            c_rep += std::hypot(px - pts_ref[k].x, py - pts_ref[k].y);
        }
        c_rep *= 0.25;

        const double sc = phot * (frac * frac)
                        - 2.0 * std::max(med - 0.5, 0.0)
                        - 60.0 * (c_rep / std::max(bh, 20.0));
        refined.push_back({R, t, sc, phot, med, frac});
    }
    const auto t_edge = Clock::now();
    res.timing.edge_ms = ms_between(t_prep, t_edge);

    if (refined.empty()) return fallback();

    std::sort(refined.begin(), refined.end(),
              [](const RefinedCand& a, const RefinedCand& b) { return a.score > b.score; });
    const auto& best = refined[0];
    const double med_gate = std::max(0.70, 0.0025 * bh);
    if (best.med > med_gate || best.frac < 0.58) {
        return fallback();
    }

    res.valid = true;
    res.used_fallback = false;
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
    res.solve_ms = ms_between(t_start, Clock::now());
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

    cv::Matx33d R = exp_so3(prev_rvec[0], prev_rvec[1], prev_rvec[2]);
    cv::Vec3d t = prev_tvec;
    SolverWorkspace ws;
    ws.reserve(tpl.points.size());
    const double r_mm = 0.5 * std::hypot(tpl.width_mm, tpl.height_mm);
    double med = 0.0, frac = 0.0;
    refine_edges_lie(gs, R, t, tpl, Kc, {4, 2, 2}, ws, r_mm, med, frac, 1.5);
    if (!std::isfinite(med) || !is_finite_pose(R, t)) return res;

    res.valid = true;
    res.rvec = mat_to_rvec(R);
    res.tvec = t;
    res.euler_yxz_deg = rvec_to_euler_yxz_deg(res.rvec);
    res.edge_med_px = med;
    res.edge_frac = frac;
    res.phot_contrast = boundary_phot_contrast(gs, tpl, R, t, Kc, ws);
    res.mirror_gap = std::numeric_limits<double>::infinity();
    const auto t_end = std::chrono::steady_clock::now();
    res.solve_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    return res;
}

}  // namespace tav_pose
