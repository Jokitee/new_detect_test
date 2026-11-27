#pragma once

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

namespace tav_pose {

enum class GpuMode {
    Auto,
    ForceOn,
    Off,
};

struct GpuDeviceInfo {
    bool has_cuda = false;
    bool cuda_fp16 = false;
    std::string cuda_name;

    bool has_opencl_gpu = false;
    bool opencl_fp16 = false;
    std::string opencl_name;
};

struct DnnBackendStatus {
    cv::dnn::Backend backend = cv::dnn::DNN_BACKEND_OPENCV;
    cv::dnn::Target target = cv::dnn::DNN_TARGET_CPU;
    bool using_gpu = false;
    std::string summary;
};

GpuMode parse_gpu_mode(const std::string& text);

/// Detects whether a real hardware GPU (CUDA or OpenCL iGPU/dGPU, excluding software CPU OpenCL)
/// is available to OpenCV DNN on this host.
GpuDeviceInfo detect_gpu_hardware();

/// Configures `net` for CPU or GPU according to `mode`.
/// In Auto mode, verifies GPU dry-run execution and benchmarks GPU vs CPU to avoid regressions on tiny nets.
DnnBackendStatus configure_dnn_backend(cv::dnn::Net& net,
                                       const std::string& onnx_path,
                                       const std::vector<std::string>& out_names,
                                       const cv::Size& input_sz,
                                       int channels,
                                       GpuMode mode);

}  // namespace tav_pose

