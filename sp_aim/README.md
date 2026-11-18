# sp_aim — 从 sp_vision_25 迁移的非识别算法 xmake 实验子构建

本目录位于 `new_detect_test/sp_aim/` 下，使用 **`xmake`** 构建，保留了我们完整的识别/位姿/相机栈（`IFL-Net V2-Slim`、`Pure-M1`、`MvCamera`），并将 `sp_vision_25` 除识别以外的全部核心算法迁移至此进行对接实测：

- **已迁移模块（`sp_port/`）**：
  - `tools/`：`extended_kalman_filter`（11维 EKF 核心）、`math_tools`、`trajectory`、`img_tools`、`logger`、`plotter`、`exiter`、`recorder`、`yaml`
  - `tasks/auto_aim/`：`armor`、`solver`、`target`（11维整车运动与多装甲板匹配）、`aimer`、`shooter`、`voter`、`planner` + `tinympc`（QP 轨迹规划器）
  - **已彻底剔除**：原 `sp_vision_25` `detector / classifier / yolo* / mt_detector / auto_buff / omniperception` 以及 OpenVINO / SocketCAN / ROS 依赖。
- **新对接层（`include/sp_new/`, `src/`）**：
  - `PureM1ArmorDetector`：直接从 YAML/MindVision 读取标定内参做 `cv::remap` 去畸变 -> `IFL-Net V2-Slim (conf=0.45)` -> `Pure-M1` 64 点稀疏边缘求解 + 残差门控 + ROI 两侧灯条红蓝推断。
  - `PureM1Solver` + `PureM1Tracker`：将 Pure-M1 `tvec(mm) -> m` 与正交系对齐旋转 $\mathbf{R}_{\text{armor}\to\text{m1}}$ 转入世界/云台系，绕过原版 4 角点 IPPE PnP 直接喂给原版 `Target` EKF（亦保留 `--optimize-yaw` 开关做对照）。

> **如何一键取消**：默认 `xmake` 仍只编原有 `new_detect_test`。若本测试链路实测不通过，直接 `rm -rf sp_aim`，主工程零残留恢复原状。

---

## Ubuntu 依赖与构建命令

```bash
# 安装 sp_vision_25 算法栈需要的公共 C++ 库（OpenCV 沿用你现有的配置）
sudo apt install -y libeigen3-dev libfmt-dev libspdlog-dev libyaml-cpp-dev nlohmann-json3-dev

cd ~/RM/new_detect_test
xmake build sp_ekf_test          # 或进入独立子目录: cd sp_aim && xmake
```

## 实测运行命令

```bash
# 1. MindVision 实机相机 + 11维 EKF + Aimer + TinyMPC Planner 测试（无灯条打印靶加 --no-color-filter）
xmake run sp_ekf_test -- --camera --no-color-filter --planner --out out/ekf_cam.avi

# 2. 离线视频测试（无 IMU txt 时自动以静态世界系运行）
xmake run sp_ekf_test -- --video out/mv_skip640x512.avi --no-color-filter --planner --out out/ekf_offline.avi

# 3. 与原版 discrete optimize_yaw 做 A/B 对照
xmake run sp_ekf_test -- --video <video> --no-color-filter --optimize-yaw
```

