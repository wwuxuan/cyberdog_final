# CyberDog Competition

当前正式比赛版本：一至六赛段联合运行。

## 正式文件

### 狗端 `dog/`

根目录只有七个正式入口：

- `main.py`：一至六赛段联合入口，也是比赛时应运行的脚本。
- `stage1.py` 至 `stage6.py`：六个正式赛段模块。`main.py` 会按赛段顺序调用它们的现行实现。

其余目录不参与默认比赛流程：

- `core/`：正式赛段的适配器、LCM、深度/鱼眼、步态、校准与联合赛段交接实现。
- `support/stage4_vision/`：狗端第四赛段 RGB 推流与语音播报辅助脚本。
- `support/stage5/`：第五赛段独立适配器与跳下路线实现。
- `tools/`：相机、鱼眼、第二/三赛段诊断与测试脚本。
- `alternatives/`：第四赛段左绕与第六赛段历史备选方案；不会被 `main.py` 默认调用。
- `legacy/`：早期 LCM 主控方案，仅保留参考。

### 电脑端 `pc_vision/`

四个正式服务入口：

- `stage1_vision.py`：第一赛段右鱼眼黄线服务，端口 `9876`。
- `stage3_vision.py`：第三赛段鱼眼黄线服务，端口 `9877`。
- `stage4_yolo.py`：第四赛段 YOLO，端口 `9891`。
- `stage6_yolo.py`：第六赛段足球 YOLO，端口 `9892`，不回传语音。

模型在 `models/`，鱼眼标定在 `calibration/`，服务内部实现放在 `support/`；`tools/` 只放一键启动批处理。

## 依赖

### 狗端

- CyberDog ROS 2 环境：`/etc/mi/ros2_env.conf`。
- Python 3.6、`rclpy`、`numpy`、`opencv-python`、LCM Python 模块。
- 机器狗本机相机节点和本机标定目录 `/params/camera/calibration`。
- 与 `locomotion/` 源码匹配的 LCM 运控；它支持比赛使用的低姿、抬腿和 roll 指令。

### 电脑端

- Windows Python、CUDA GPU PyTorch、`ultralytics`、`opencv-python`、`numpy`、`PyYAML`。
- 保持已有 GPU PyTorch，不要安装 CPU 版 `torch`。
- 若只缺其余库：

```powershell
python -m pip install ultralytics opencv-python numpy PyYAML
```

- Windows 防火墙放行 TCP `9876`、`9877`、`9891`、`9892`；电脑与机器狗必须相互可达。

## 部署狗端

将 `dog/` 内的内容复制到机器狗 `/home/mi/cyberdog_competition/`。第四赛段语音/推流脚本还必须额外部署到固定目录：

```bash
scp -r dog/* mi@<DOG_IP>:/home/mi/cyberdog_competition/
ssh mi@<DOG_IP> 'mkdir -p /home/mi/stage4/vision'
scp dog/support/stage4_vision/*.py mi@<DOG_IP>:/home/mi/stage4/vision/
ssh mi@<DOG_IP> 'chmod +x /home/mi/cyberdog_competition/support/stop_motion_manager.sh'
```

部署会覆盖同名文件。首次替换前请备份机器狗原目录和当前可用运控文件。

## 运行电脑端视觉

在电脑上进入 `pc_vision/`，打开四个 PowerShell 窗口并分别运行。把 `<DOG_IP>` 替换为当前机器狗 IP：

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

四个窗口均保持打开；未用到的服务保持运行不会影响比赛。

## 运行联合赛段

在机器狗执行。将 `<PC_IP>` 改为电脑的 IPv4：

```bash
source /etc/mi/ros2_env.conf
cd /home/mi/cyberdog_competition
python3 main.py --stand --arm \
  --vision-url http://<PC_IP>:9876/measure \
  --stage3-vision-url http://<PC_IP>:9877/measure \
  --stage4-pc-host <PC_IP> \
  --stage6-pc-host <PC_IP> \
  --stage6-stream-port 9892 \
  --stage4-detour right
```

`--stage4-detour right` 是默认右绕；传 `--stage4-detour left` 仍可选择左绕。

可从指定场地点继续，后续赛段会连续执行：

```text
--start-at stage1              完整一至六赛段
--start-at stage2-upper-left   第二赛段左上四球中心
--start-at stage4              第四赛段走廊入口
--start-at stage5              第五赛段起点
--start-at stage5-turn1        第五赛段第一个转向点后
--start-at stage5-turn2        第五赛段第二个转向点后
--start-at stage5-turn3        第五赛段第三个转向点后
--start-at stage5-turn4        第五赛段第四个转向点后
```

第四赛段默认自动启动狗端语音播报；第六赛段不播报。第一至第五赛段的局部保护超时会停止或报错，不会趴下；当前第六赛段在正常完成或四轮 YOLO 扫描都没有稳定足球时会前往终点趴下。

## 运控源码

`locomotion/` 是交叉编译源码，未包含 `onboard-build/` 生成物。编译和部署步骤见 [`docs/运控编译与部署.md`](docs/运控编译与部署.md)。

模型权重已配置 Git LFS；上传 GitHub 前执行 `git lfs install`，并确认系统源码和模型权重可以公开发布。
