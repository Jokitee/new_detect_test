# new_detect_test (C++17 / xmake / OpenCV 4.13 / MindVision SDK)

Pure-M1 character detection & 6DoF pose estimation in C++17 (`IFL-Net V2-Slim 448x336` / `YOLOv8n ONNX via cv::dnn` + `64-pt Sparse Edge Refinement`, without `minAreaRect` / `IPPE`), supporting unlimited live MindVision camera capture (`MVSDK`) and OpenCV `VideoWriter` video output.

## Structure

- `xmake.lua` — xmake build script (uses local/system OpenCV by default without downloading; links `MVSDK`, C++17, `fastest`)
- `include/tav_pose/` — `types.hpp`, `yolo_detector.hpp`, `sparse_edge_pose.hpp`, `mv_camera.hpp`
- `src/` — `yolo_detector.cpp`, `sparse_edge_pose.cpp`, `mv_camera.cpp`, `main.cpp`
- `assets/ifl_net_v2_448x336.onnx` — Default 3-scale (`P2+P3+P4`), 3-class (`1`, `3`, `sentary`) IFL-Net V2-Direct ONNX (`448x336`, `opset=12`)
- `assets/roi_corner_v2_64x64.onnx` — Default `64x64` RoiCornerNetV2 ONNX (`Hybrid Option-A + O1n_pe`)
- `assets/roi_corner_v2_48x48.onnx` — Optional compact `48x48` corner prior ONNX (`tiny48_fc`, ~2.4x faster CNN)
- `assets/char3_det_416.onnx` — Optional 3-class YOLOv8n detect ONNX (`416x416`, `opset=12`)
- `assets/templates.csv` — 64-point 3D planar contour templates with tangents and outward normals

## Build (Raspberry Pi 5 `aarch64` & Intel NUC / Linux VM `x86_64`)

`xmake.lua` automatically adapts to both **Raspberry Pi 5 (`arm64` / `aarch64`)** and **Intel NUC (`x86_64`)**:
- Searches local/system OpenCV first (`--opencv_dir`, `$OpenCV_DIR`, `~/install`, `/usr/local`, `/usr`, `/usr/include/{aarch64,x86_64}-linux-gnu/opencv4`, `pkg-config`, `cmake`) without downloading.
- Searches MindVision SDK headers (`CameraApi.h`) and architecture-matched `libMVSDK.so` (`lib/arm64` on Pi 5, `lib/x64` on NUC, or `/usr/lib`, `/usr/local/lib`) with automatic `rpath` injection.

### 1. Install Dependencies
```bash
# Ubuntu / Debian / Raspberry Pi OS (NUC & Raspberry Pi 5)
sudo apt update
sudo apt install -y build-essential git pkg-config cmake libopencv-dev

# Optional: only needed if building sp_ekf_test inside sp_aim/
sudo apt install -y libeigen3-dev libfmt-dev libspdlog-dev libyaml-cpp-dev nlohmann-json3-dev
```
> **MindVision SDK (`MVSDK`) note**: Download MindVision Linux SDK and install the library matching your board architecture (`arm64` for Raspberry Pi 5, `x64` for Intel NUC), or pass `--mvsdk_dir=/path/to/linuxSDK` directly without global installation. Add MindVision udev rules (`88-mvusb.rules`) or run with USB write permission so `USBDEVFS_RESET` recovery works without `sudo`.

### 2. Configure & Build
```bash
cd ~/RM/new_detect_test
git pull
xmake f -c -m release   # Always re-configure (-c) when switching between Pi 5 (arm64) and NUC (x86_64)
xmake -j$(nproc)
```

Useful configure options:
- Specify custom OpenCV 4.10~4.13 install prefix (recommended on Pi 5 if system `apt` OpenCV is `< 4.8`): `xmake f -c -m release --opencv_dir=$HOME/install`
- Custom unpacked MindVision SDK directory (auto-picks `lib/arm64` on Pi 5 or `lib/x64` on NUC): `xmake f -c -m release --mvsdk_dir=$HOME/linuxSDK_V2`
- Build without MindVision SDK (offline `--image` / `--video` testing on any board): `xmake f -c -m release --with_mvsdk=false`
- Build from `xmake-repo` (NUC recommended only; **avoid on 4GB Raspberry Pi 5** due to slow/OOM source compilation): `xmake f -c -m release --use_xrepo=true --opencv_ver=4.13.x`

> **Raspberry Pi 5 / NUC GPU runtime tips**:
> - GPU acceleration is automatic (`--gpu auto`, default): probes NVIDIA CUDA (`CUDA-FP16` / `CUDA`) and hardware OpenCL iGPU/dGPU (`OpenCL-FP16` / `OpenCL`, ignoring software CPU OpenCL), benchmarks GPU vs CPU on startup, and automatically falls back to CPU if no GPU is faster or available. Force modes via `--gpu on` / `--gpu off` (`--cpu`) or `ifl_gpu_mode` in `sp_aim/configs/sp_ekf_test.yaml`.
> - Under Wayland / SSH headless environments, pass `--no-show`.
> - Lock threads if needed with `--threads 4` (Pi 5 has 4 Cortex-A76 cores), and optionally test `--corner-model assets/roi_corner_v2_48x48.onnx` for lower ARM64 CPU load.

## Run Modes

### 1. Live MindVision Camera -> Video Output (Default, Unlimited Frames)

Captures `640x480` frames continuously without frame limit (`max_frames=0`) using `CameraSetImageResolution` in the `MdVison_edu/demo/example1.cpp` MindVision SDK flow (8 ms default exposure, automatic intrinsics & distortion adaptation from calibrated `1280x1024`), uniformly downsamples via `cv::INTER_AREA` to `448x336` (zero cropping, `K` scaled `0.7x` automatically), runs **IFL-Net V2-Slim (`448x336`)** + Pure-M1 pose estimation, displays `cv::imshow` directly (unless `--no-show` is passed), and writes annotated video via OpenCV `VideoWriter` until you stop with `Ctrl+C`, `q`, or `ESC`:

```bash
# Default pipeline: hardware 640x480 -> 448x336 INTER_AREA -> IFL-Net V2-Slim + Pure-M1
xmake run new_detect_test --camera --out out/mv_detect.mp4 --fps 30

# Live preview + unlimited video recording + temporal tracking
xmake run new_detect_test --camera --track --out out/mv_detect.avi

# Pure headless recording (disable cv::imshow window) or fallback to YOLOv8n 416
xmake run new_detect_test --camera --no-show --out out/mv_detect.avi
xmake run new_detect_test --camera --no-downsample --model assets/char3_det_416.onnx --out out/mv_yolo.avi
```

Note: if OpenCV lacks FFmpeg/H.264 support when requesting `.mp4`, the writer automatically falls back to `MJPG` `.avi` so video saving succeeds.

### 2. Offline Video File -> Video Output

```bash
xmake run new_detect_test --video input.mp4 --out out/pred.avi --show
```

### 3. Single Image Test & Latency Benchmark

```bash
xmake run new_detect_test --image ../../Photo/261118.jpg --bench 50
xmake run new_detect_test --image ../../Photo/IMG_20261117141856.jpg --roi 305,108,464,431 --cls 3
```
