# new_detect_test (C++17 / xmake / OpenCV 4.13 / MindVision SDK)

Pure-M1 character detection & 6DoF pose estimation in C++17 (`YOLOv8n ONNX via cv::dnn` + `64-pt Sparse Edge Refinement`, without `minAreaRect` / `IPPE`), supporting unlimited live MindVision camera capture (`MVSDK`) and OpenCV `VideoWriter` video output.

## Structure

- `xmake.lua` — xmake build script (uses local/system OpenCV by default without downloading; links `MVSDK`, C++17, `fastest`)
- `include/tav_pose/` — `types.hpp`, `yolo_detector.hpp`, `sparse_edge_pose.hpp`, `mv_camera.hpp`
- `src/` — `yolo_detector.cpp`, `sparse_edge_pose.cpp`, `mv_camera.cpp`, `main.cpp`
- `assets/char3_det_416.onnx` — 3-class (`3`, `4`, `7`) YOLOv8n detect ONNX (`opset=12`)
- `assets/templates.csv` — 64-point 3D planar contour templates with tangents and outward normals

## Build (Linux VM / NUC)

By default `xmake.lua` automatically searches your local/system OpenCV (`--opencv_dir`, `$OpenCV_DIR`, `~/install`, `/usr/local`, `/usr`, `pkg-config`, `cmake`) and will **not** download OpenCV from `xmake-repo`.

```bash
cd ~/RM/new_detect_test
git pull
xmake f -c -m release
xmake
```

Useful configure options:
- Specify your custom OpenCV 4.13 install directory explicitly: `xmake f -c -m release --opencv_dir=$HOME/install`
- Only if you want xmake to download OpenCV from `xmake-repo`: `xmake f -c -m release --use_xrepo=true --opencv_ver=4.13.x`
- Build without MindVision SDK (offline `--image` / `--video` only): `xmake f -c -m release --with_mvsdk=false`
- Custom MindVision SDK directory: `xmake f -c -m release --mvsdk_dir=/path/to/MindVisionSDK`

## Run Modes

### 1. Live MindVision Camera -> Video Output (Default, Unlimited Frames)

Captures frames continuously without frame limit (`max_frames=0`) using the MindVision SDK (`MONO8` or `BGR8`), runs YOLO + Pure-M1 pose estimation, and writes an annotated video via OpenCV `VideoWriter` until you stop with `Ctrl+C` (or `q` / `ESC` in preview mode):

```bash
# Headless unlimited recording -> out/mv_detect.mp4 (or automatic .avi fallback)
xmake run new_detect_test -- --camera --out out/mv_detect.mp4 --fps 30 --exposure 5000 --gain 100,70,50

# Live preview window + unlimited video recording + temporal tracking
xmake run new_detect_test -- --camera --show --track --out out/mv_detect.avi
```

Note: if OpenCV lacks FFmpeg/H.264 support when requesting `.mp4`, the writer automatically falls back to `MJPG` `.avi` so video saving succeeds.

### 2. Offline Video File -> Video Output

```bash
xmake run new_detect_test -- --video input.mp4 --out out/pred.avi --show
```

### 3. Single Image Test & Latency Benchmark

```bash
xmake run new_detect_test -- --image ../../Photo/261118.jpg --bench 50
xmake run new_detect_test -- --image ../../Photo/IMG_20261117141856.jpg --roi 305,108,464,431 --cls 3
```
