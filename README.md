# CyberDog Competition

六赛段比赛代码：狗端运行 `dog/main.py`，电脑端提供鱼眼测线与 YOLO 识别服务。

## 赛段逻辑

1. 第一赛段：右鱼眼测量黄线，控制出球位置。
2. 第二赛段：雷达、橙蓝球视觉和坐标规划结合，依次完成撞球路径。
3. 第三赛段：按曲线路径行走，鱼眼持续校正黄线距离。
4. 第四赛段：依次通过三个通道；YOLO 识别限高杆、不可跨越障碍、可乐瓶、足球和橙球，触发播报、低姿、绕行或撞球。
5. 第五赛段：按预设坐标通过独木桥并从末端跳下。
6. 第六赛段：按固定路线完成第一次推球，再以 YOLO 和前深度相机定位足球，继续推向缺口，最后到终点趴下。

## 狗端

### 依赖

- CyberDog ROS 2 环境：`/etc/mi/ros2_env.conf`。
- Python 3.6、`rclpy`、`numpy`、`opencv-python`、LCM Python 模块。
- 相机节点已启动，且机器狗具有 `/params/camera/calibration` 标定目录。
- 已部署本仓库 `locomotion/` 编译得到的匹配运控程序；第四、六赛段的低姿和姿态控制依赖它。

### 部署

将 `dog/` 的内容复制到机器狗的 `/home/mi/cyberdog_competition/`。第四赛段的推流和语音脚本还必须复制到固定目录：

```bash
scp -r dog/* mi@<DOG_IP>:/home/mi/cyberdog_competition/
ssh mi@<DOG_IP> 'mkdir -p /home/mi/stage4/vision'
scp dog/support/stage4_vision/*.py mi@<DOG_IP>:/home/mi/stage4/vision/
ssh mi@<DOG_IP> 'chmod +x /home/mi/cyberdog_competition/support/stop_motion_manager.sh'
```

首次覆盖前请备份机器狗原目录与当前可用的运控文件。

### 运行

先在电脑启动四个视觉服务，再在机器狗执行。将 `<PC_IP>` 替换为运行视觉服务的电脑 IPv4：

```bash
source /etc/mi/ros2_env.conf
cd /home/mi/cyberdog_competition
python3 main.py --stand --arm \
  --vision-url http://<PC_IP>:9876/measure \
  --stage3-vision-url http://<PC_IP>:9877/measure \
  --stage4-pc-host <PC_IP> \
  --stage6-pc-host <PC_IP> \
  --stage4-detour right
```

`main.py` 默认从第一赛段连续运行到第六赛段。可用 `--start-at` 从指定位置续跑，后续赛段仍会继续执行：

```text
stage1              完整一至六赛段
stage2-upper-left   第二赛段左上四球中心
stage4              第四赛段走廊入口
stage5              第五赛段起点
stage5-turn1        第五赛段第一个转向点后
stage5-turn2        第五赛段第二个转向点后
stage5-turn3        第五赛段第三个转向点后
stage5-turn4        第五赛段第四个转向点后
```

例如从第四赛段开始：

```bash
python3 main.py --start-at stage4 --arm \
  --stage4-pc-host <PC_IP> \
  --stage6-pc-host <PC_IP>
```

第四赛段默认右绕；传入 `--stage4-detour left` 可选择左绕。第四赛段的狗端语音服务会由联合脚本自动启动，第六赛段没有语音播报。

## 电脑端

### 依赖

- Windows Python、NVIDIA CUDA GPU 对应的 PyTorch。
- `ultralytics`、`opencv-python`、`numpy`、`PyYAML`。
- 保留已有 GPU PyTorch；不要用 `pip` 安装 CPU 版 `torch` 覆盖它。

若只缺其余依赖，在 `pc_vision/` 目录执行：

```powershell
python -m pip install -r .\requirements.txt
```

电脑防火墙必须放行 TCP `9876`、`9877`、`9891`、`9892`，并确保电脑和机器狗相互可达。

### 运行

在 `pc_vision/` 中打开四个 PowerShell 窗口，均将 `<DOG_IP>` 改为机器狗当前 IP：

```powershell
python .\stage1_vision.py --dog-ip <DOG_IP>
```

```powershell
python .\stage3_vision.py --dog-ip <DOG_IP>
```

```powershell
python .\stage4_yolo.py --port 9891 --dog-ip <DOG_IP> --push-ip <DOG_IP>
```

```powershell
python .\stage6_yolo.py --port 9892 --push-ip <DOG_IP>
```

四个服务均保持运行。模型位于 `pc_vision/models/`，鱼眼标定位于 `pc_vision/calibration/`。

## 运控编译与部署

`locomotion/` 是比赛使用的运控源码。若你已有原始 `locomotion` 仓库，应以原仓库为基础，合并本包中对应的源码改动后再编译，不要复制任何旧的 `onboard-build/` 生成目录。

比赛相关的主要改动在：

```text
locomotion/control/src/convex_mpc/convex_mpc_loco_gaits.cpp
locomotion/control/src/convex_mpc/convex_mpc_motion_gaits.cpp
```

其中将最大抬腿高度放宽为 `0.20m`，并允许约 `+/-0.47rad` 的 roll 指令。

需要 Docker 和小米 ARM64 工具链镜像。进入原始或已合并的 `locomotion` 仓库所在目录后启动容器：

```bash
docker run -it --rm \
  -v D:/path/to/locomotion:/work/build_farm/workspace/cyberdog \
  cr.d.xiaomi.net/athena/athena_cheetah_arm64:2.0 \
  /bin/bash
```

容器内编译：

```bash
cd /work/build_farm/workspace/cyberdog
mkdir -p onboard-build
cd onboard-build
cmake \
  -DCMAKE_TOOLCHAIN_FILE=/usr/xcc/aarch64-openwrt-linux-gnu/Toolchain.cmake \
  -DONBOARD_BUILD=ON \
  -DBUILD_FACTORY=ON \
  -DBUILD_CYBERDOG2=ON \
  ..
make -j4
```

先备份机器狗现有运控、停止原服务，再上传新生成物：

```bash
ssh mi@<DOG_IP> 'sudo systemctl stop cyberdog_bringup'
scp -r onboard-build/control/user/cyberdog_control mi@<DOG_IP>:/home/mi/cyberdog_locomotion/onboard-build/control/user/
scp -r onboard-build/hardware/ mi@<DOG_IP>:/home/mi/cyberdog_locomotion/onboard-build/
scp -r onboard-build/robot-software/ mi@<DOG_IP>:/home/mi/cyberdog_locomotion/onboard-build/
```

部署后先在空场验证站立、停止、低姿、抬腿和 roll，再进入赛道测试。
