#include "tav_pose/gpu_accel.hpp"

#include <opencv2/core/ocl.hpp>
#if __has_include(<opencv2/core/cuda.hpp>)
#include <opencv2/core/cuda.hpp>
#define TAV_HAS_CUDA_HEADER 1
#else
#define TAV_HAS_CUDA_HEADER 0
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <vector>

namespace tav_pose {
namespace {

std::string to_lower_copy(std::string s) {
    for (char& ch : s) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return s;
}

bool contains_backend(const std::vector<std::pair<cv::dnn::Backend, cv::dnn::Target>>& list,
                      cv::dnn::Backend be,
                      cv::dnn::Target tgt) {
    for (const auto& item : list) {
        if (item.first == be && item.second == tgt) return true;
    }
    return false;
}

bool is_software_opencl_name(const std::string& name) {
    const std::string low = to_lower_copy(name);
    return low.find("llvmpipe") != std::string::npos ||
           low.find("softpipe") != std::string::npos ||
           low.find("pthread") != std::string::npos ||
           low.find("cpu") != std::string::npos;
}

double bench_net_median_ms(cv::dnn::Net& net,
                           const std::vector<std::string>& out_names,
                           const cv::Mat& dummy,
                           int warmup_iters,
                           int bench_iters) {
    auto run_once = [&]() -> bool {
        net.setInput(dummy);
        if (out_names.empty()) {
            cv::Mat y = net.forward();
            return !y.empty();
        }
        std::vector<cv::Mat> outs;
        net.forward(outs, out_names);
        return !outs.empty() && !outs[0].empty();
    };
    for (int i = 0; i < warmup_iters; ++i) {
        if (!run_once()) return -1.0;
    }
    std::vector<double> ts;
    ts.reserve(static_cast<size_t>(bench_iters));
    for (int i = 0; i < bench_iters; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = run_once();
        const auto t1 = std::chrono::steady_clock::now();
        if (!ok) return -1.0;
        ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ts.begin(), ts.end());
    return ts[ts.size() / 2];
}

void set_cpu_backend(cv::dnn::Net& net) {
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

}  // namespace

GpuMode parse_gpu_mode(const std::string& text) {
    const std::string low = to_lower_copy(text);
    if (low == "off" || low == "cpu" || low == "false" || low == "0") return GpuMode::Off;
    if (low == "on" || low == "gpu" || low == "true" || low == "1") return GpuMode::ForceOn;
    return GpuMode::Auto;
}

GpuDeviceInfo detect_gpu_hardware() {
    GpuDeviceInfo info;
    const auto backends = cv::dnn::getAvailableBackends();

    const bool dnn_cuda = contains_backend(backends, cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA);
    const bool dnn_cuda_fp16 =
        contains_backend(backends, cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA_FP16);
    if (dnn_cuda || dnn_cuda_fp16) {
#if TAV_HAS_CUDA_HEADER
        try {
            const int n_cuda = cv::cuda::getCudaEnabledDeviceCount();
            if (n_cuda > 0) {
                info.has_cuda = true;
                info.cuda_fp16 = dnn_cuda_fp16;
                cv::cuda::DeviceInfo dev(0);
                info.cuda_name = std::string(dev.name());
            }
        } catch (...) {
        }
#else
        // Without cuda.hpp, probe CUDA availability safely during configure_dnn_backend dry-run.
        info.has_cuda = true;
        info.cuda_fp16 = dnn_cuda_fp16;
        info.cuda_name = "CUDA GPU";
#endif
    }

    const bool dnn_ocl = contains_backend(backends, cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_OPENCL);
    const bool dnn_ocl_fp16 =
        contains_backend(backends, cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_OPENCL_FP16);
    if (dnn_ocl || dnn_ocl_fp16) {
        try {
            if (cv::ocl::haveOpenCL()) {
                cv::ocl::setUseOpenCL(true);
                const cv::ocl::Context& ctx = cv::ocl::Context::getDefault();
                if (!ctx.empty() && ctx.ndevices() > 0) {
                    const cv::ocl::Device& dev = cv::ocl::Device::getDefault();
                    const int dtype = dev.type();
                    const bool is_hw_gpu =
                        dev.available() &&
                        ((dtype & cv::ocl::Device::TYPE_GPU) != 0) &&
                        ((dtype & cv::ocl::Device::TYPE_CPU) == 0) &&
                        !is_software_opencl_name(dev.name());
                    if (is_hw_gpu) {
                        info.has_opencl_gpu = true;
                        info.opencl_fp16 = dnn_ocl_fp16 && (dev.halfFPConfig() != 0);
                        info.opencl_name = dev.name();
                    }
                }
            }
        } catch (...) {
        }
    }

    return info;
}

DnnBackendStatus configure_dnn_backend(cv::dnn::Net& net,
                                       const std::string& onnx_path,
                                       const std::vector<std::string>& out_names,
                                       const cv::Size& input_sz,
                                       int channels,
                                       GpuMode mode) {
    DnnBackendStatus status;
    set_cpu_backend(net);
    if (mode == GpuMode::Off) {
        status.summary = "CPU (forced)";
        return status;
    }

    const GpuDeviceInfo hw = detect_gpu_hardware();
    struct Candidate {
        cv::dnn::Backend backend;
        cv::dnn::Target target;
        const char* tag;
        std::string dev_name;
    };
    std::vector<Candidate> candidates;
    if (hw.has_cuda) {
        if (hw.cuda_fp16) {
            candidates.push_back({cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA_FP16, "CUDA-FP16", hw.cuda_name});
        }
        candidates.push_back({cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA, "CUDA", hw.cuda_name});
    }
    if (hw.has_opencl_gpu) {
        if (hw.opencl_fp16) {
            candidates.push_back({cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_OPENCL_FP16, "OpenCL-FP16", hw.opencl_name});
        }
        candidates.push_back({cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_OPENCL, "OpenCL", hw.opencl_name});
    }

    if (candidates.empty()) {
        status.summary = "CPU (no hardware GPU detected)";
        return status;
    }

    const int dims[4] = {1, channels, input_sz.height, input_sz.width};
    cv::Mat dummy = cv::Mat::zeros(4, dims, CV_32F);

    double cpu_ms = -1.0;
    if (mode == GpuMode::Auto) {
        try {
            cpu_ms = bench_net_median_ms(net, out_names, dummy, 2, 5);
        } catch (...) {
            cpu_ms = -1.0;
        }
    }

    for (const auto& c : candidates) {
        try {
            net.setPreferableBackend(c.backend);
            net.setPreferableTarget(c.target);
            const double gpu_ms = bench_net_median_ms(net, out_names, dummy, 2, 5);
            if (gpu_ms <= 0.0) continue;

            if (mode == GpuMode::ForceOn || cpu_ms <= 0.0 || gpu_ms < cpu_ms) {
                status.backend = c.backend;
                status.target = c.target;
                status.using_gpu = true;
                char buf[256];
                if (cpu_ms > 0.0) {
                    std::snprintf(buf, sizeof(buf), "%s [%s] (gpu=%.2fms < cpu=%.2fms)",
                                  c.tag, c.dev_name.c_str(), gpu_ms, cpu_ms);
                } else {
                    std::snprintf(buf, sizeof(buf), "%s [%s] (gpu=%.2fms)",
                                  c.tag, c.dev_name.c_str(), gpu_ms);
                }
                status.summary = buf;
                return status;
            }

            // GPU ran cleanly, but CPU was faster (common for tiny 64x64/48x48 nets on OpenCL iGPU).
            net = cv::dnn::readNetFromONNX(onnx_path);
            set_cpu_backend(net);
            char buf[256];
            std::snprintf(buf, sizeof(buf), "CPU (auto: cpu=%.2fms <= %s=%.2fms)",
                          cpu_ms, c.tag, gpu_ms);
            status.summary = buf;
            return status;
        } catch (...) {
            // Driver/kernel/op error on GPU candidate -> reload clean CPU net and continue trying next candidate.
            net = cv::dnn::readNetFromONNX(onnx_path);
            set_cpu_backend(net);
        }
    }

    status.summary = "CPU (GPU probe failed, fallback to CPU)";
    return status;
}

}  // namespace tav_pose
