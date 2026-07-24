# 小米 CyberDog 机器狗开发指南

> **比赛**: 2026 小米杯决赛  
> **设备**: 小米 CyberDog (四足机器人)  
> **IP**: 192.168.43.247  
> **SSH**: `mi` / `123`  
> **日期**: 2026-07-24

---

## 目录

1. [系统概览](#1-系统概览)
2. [硬件架构](#2-硬件架构)
3. [软件架构](#3-软件架构)
4. [ROS2 架构详解](#4-ros2-架构详解)
5. [网络与通信](#5-网络与通信)
6. [开发环境与工具链](#6-开发环境与工具链)
7. [代码部署方式](#7-代码部署方式)
8. [Python API 参考](#8-python-api-参考)
9. [C++ 开发指南](#9-c-开发指南)
10. [Visual Programming (VP) 系统](#10-visual-programming-vp-系统)
11. [系统启动流程](#11-系统启动流程)
12. [常见任务示例](#12-常见任务示例)
13. [调试与日志](#13-调试与日志)
14. [注意事项与最佳实践](#14-注意事项与最佳实践)

---

## 1. 系统概览

| 项目 | 详情 |
|------|------|
| **操作系统** | Ubuntu 18.04.5 LTS (Bionic Beaver) |
| **内核** | Linux 4.9.253-tegra (NVIDIA Jetson 定制内核) |
| **架构** | ARM64 (aarch64) |
| **ROS 版本** | ROS2 Galactic |
| **DDS 实现** | Eclipse Cyclone DDS (仅 localhost 通信) |
| **Python** | Python 3.6.9 |
| **内存** | 约 8 GB (7.96 GB) |
| **可用内存** | 约 4.5 GB |
| **用户名** | mi |
| **主机名** | mi-desktop |

---

## 2. 硬件架构

### 计算平台
- **主控**: NVIDIA Jetson (Tegra T19x)，含 GPU 加速
- **电机控制器**: 通过 CAN 总线通信 (can0, can1)
- **协处理器**: BES (蓝牙音频) 芯片

### 传感器配置
| 传感器类型 | 说明 | ROS Topic |
|-----------|------|-----------|
| Intel RealSense (x2) | 深度相机 | `/camera/*` |
| 激光雷达 (LiDAR) | 单线激光雷达 | `/scan` |
| 超声波传感器 | 前后超声波 | `/ultrasonic_payload` |
| ToF 传感器 | 头部/尾部 | `/head_tof_payload`, `/rear_tof_payload` |
| IMU | 惯性测量单元 | `/imu_out` |
| GPS | 全球定位 | `/gps_payload` |
| 触摸板 | 头部触摸传感器 | `/touch_status` |
| 电子皮肤 | 身体变色LED | CAN 总线控制 |
| UWB | 超宽带定位 | `/uwb_*` |
| 麦克风阵列 | 语音交互 | `/asr_text` |

### 外设接口
- **CAN 总线**: can0, can1 (电机控制、电子皮肤)
- **I2C**: 9 个 I2C 总线 (i2c-0 至 i2c-8)
- **SPI**: spidev0.1, spidev2.0, spidev2.1
- **GPIO**: 5 个 GPIO 芯片 (gpiochip0-4)
- **摄像头**: 6 个 video 设备 (video0-video5)
- **网络**: WiFi (wlan0: 192.168.43.247), 以太网 (eth0: 192.168.44.1), USB RNDIS

---

## 3. 软件架构

```
┌─────────────────────────────────────────────────────────┐
│                    应用层 (User Code)                     │
│  /home/mi/.cyberdog/cyberdog_vp/workspace/               │
│  ├── task/          (可执行任务 .py)                      │
│  ├── module/        (可复用模块 .py)                      │
│  └── choreographer/ (舞蹈编排 .py)                        │
├─────────────────────────────────────────────────────────┤
│                ROS2 可视化编程层 (VP Layer)                │
│  /opt/ros2/cyberdog/share/cyberdog_vp/                   │
│  ├── cyberdog_vp_engine      (VP 引擎)                   │
│  ├── cyberdog_vp_abilityset  (能力集接口)                 │
│  └── cyberdog_vp_terminal    (交互终端)                   │
├─────────────────────────────────────────────────────────┤
│              ROS2 核心功能层 (Core Services)               │
│  ├── cyberdog_manager   (系统管理)                        │
│  ├── motion_manager     (运动控制)                        │
│  ├── sensor_manager     (传感器管理)                       │
│  ├── device_manager     (设备管理)                         │
│  ├── cyberdog_audio     (语音)                            │
│  ├── cyberdog_face      (人脸识别)                        │
│  ├── cyberdog_action    (动作执行)                         │
│  ├── cyberdog_interactive (交互系统)                      │
│  ├── cyberdog_train     (训练系统)                         │
│  ├── cyberdog_ai_sports (AI运动)                          │
│  ├── cyberdog_vision    (视觉)                            │
│  └── connector          (连接管理)                         │
├─────────────────────────────────────────────────────────┤
│              ROS2 导航层 (Navigation Layer)                │
│  ├── navigation_bringup (导航启动)                        │
│  ├── nav2_*             (Nav2 导航栈)                     │
│  ├── laser_slam         (激光SLAM)                        │
│  ├── vins               (视觉惯性里程计)                   │
│  ├── mcr_*              (多传感器融合)                     │
│  └── pose_graph         (位姿图优化)                       │
├─────────────────────────────────────────────────────────┤
│                底层通信层 (Low-Level)                       │
│  ├── motion_bridge       (电机桥接)                        │
│  ├── bes_transmit        (BES芯片通信)                     │
│  ├── cyberdog_grpc       (gRPC 服务)                       │
│  ├── cyberdog_bluetooth  (蓝牙)                            │
│  └── cyberdog_embed_protocol (嵌入式协议)                   │
├─────────────────────────────────────────────────────────┤
│               操作系统层 (OS Layer)                         │
│  Ubuntu 18.04 + NVIDIA Jetson Linux 4.9.253-tegra         │
│  CAN / I2C / SPI / GPIO / USB / Video                     │
└─────────────────────────────────────────────────────────┘
```

---

## 4. ROS2 架构详解

### 环境配置
所有 ROS2 环境配置在 `/etc/mi/ros2_env.conf`:

```bash
export ROS_VERSION=2
export ROS_PYTHON_VERSION=3
export ROS_DISTRO=galactic
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///etc/mi/cyclonedds.xml
```

### 关键目录

| 路径 | 说明 |
|------|------|
| `/opt/ros2/galactic/` | ROS2 Galactic 标准安装 |
| `/opt/ros2/cyberdog/` | CyberDog 自定义 ROS2 包 |
| `/opt/ros2/cyberdog/share/` | 所有 ROS2 包的 share 目录 |
| `/opt/ros2/cyberdog/lib/` | 编译后的库文件 |
| `/opt/ros2/cyberdog/include/` | C++ 头文件 |
| `/opt/ros2/cyberdog/lib/python3.6/site-packages/mi/` | Python API 包 |

### 核心 ROS2 节点

执行 `ros2 launch cyberdog_bringup main.launch.py` 时启动的节点:

| 包名 | 可执行文件 | 功能 |
|------|-----------|------|
| `connector` | `connector` | WiFi/连接管理 |
| `cyberdog_audio` | `cyberdog_audio` | 语音播报/语音交互 |
| `device_manager` | `device_manager` | 硬件设备管理 |
| `sensor_manager` | `sensor_manager` | 传感器管理 |
| `motion_manager` | `motion_manager` | 运动控制 |
| `realsense2_camera` | `realsense2_actuator_node` | RealSense 相机 |
| `cyberdog_manager` | `cyberdog_manager` | 系统主管理 |
| `motion_bridge` | `odom_out_publisher` | 里程计发布 |
| `motion_bridge` | `motor_bridge` | 电机桥接 |
| `cyberdog_grpc` | `app_server` | gRPC 服务 |
| `bes_transmit` | `bes_transmit_waiter` | BES 通信 |
| `cyberdog_vp_engine` | `cyberdog_vp_engine` | VP 引擎 |
| `cyberdog_face` | `cyberdog_face` | 人脸识别 |
| `cyberdog_action` | `cyberdog_action` | 动作执行 |
| `cyberdog_interactive` | `cyberdog_interactive` | 交互系统 |
| `cyberdog_train` | `cyberdog_train` | 训练系统 |
| `cyberdog_ai_sports` | `cyberdog_ai_sports` | AI 运动 |

### 主要 ROS2 Topic

| Topic | 类型 | 说明 |
|-------|------|------|
| `/scan` | LaserScan | 激光雷达数据 |
| `/imu_out` | IMU | 惯性测量数据 |
| `/odom_out` | Odometry | 里程计 |
| `/motion_status` | - | 运动状态 |
| `/bms_status` | - | 电池状态 |
| `/touch_status` | - | 触摸板状态 |
| `/ultrasonic_payload` | - | 超声波数据 |
| `/head_tof_payload` | - | 头部 ToF |
| `/rear_tof_payload` | - | 尾部 ToF |
| `/gps_payload` | - | GPS 数据 |
| `/asr_text` | - | 语音识别文本 |
| `/face_rec_msg` | - | 人脸识别结果 |
| `/gesture_action_msg` | - | 手势识别结果 |
| `/algo_task_status` | - | 导航算法状态 |

---

## 5. 网络与通信

### 网络接口

| 接口 | IP 地址 | 用途 |
|------|---------|------|
| wlan0 | 192.168.43.247 (DHCP) | WiFi 连接 (你的开发机通过此连接) |
| eth0 | 192.168.44.1/24 | 以太网 (ROS2 节点间通信) |
| docker0 | 172.17.0.1/16 | Docker 桥接 (未使用) |
| can0/can1 | - | CAN 总线 (电机、电子皮肤) |

### ROS2 DDS 配置

DDS 配置在 `/etc/mi/cyclonedds.xml`，仅使用 **localhost** 通信：

```xml
<Domain id="42">
    <General>
        <NetworkInterfaceAddress>lo</NetworkInterfaceAddress>
        <AllowMulticast>false</AllowMulticast>
    </General>
    <Discovery>
        <Peers>
            <Peer address="localhost"/>
        </Peers>
    </Discovery>
</Domain>
```

> ⚠️ **重要**: 所有 ROS2 节点运行在同一台机器上，使用 localhost 通信。如果你需要从外部 PC 连接到 ROS2 网络，需要修改 CycloneDDS 配置。

---

## 6. 开发环境与工具链

### 已安装的开发工具

| 工具 | 版本 | 用途 |
|------|------|------|
| **Python 3** | 3.6.9 | Python 脚本执行 |
| **GCC** | ARM64 交叉编译 | C++ 编译 |
| **CMake** | - | C++ 构建系统 |
| **colcon** | - | ROS2 包构建工具 |
| **OpenCV** | 4.1.1 | 计算机视觉 |
| **VS Code Server** | 已安装 | 远程开发 |
| **Docker** | 已安装 | 容器化(未使用) |

### 开发工作流

#### 方法 1: 通过 VS Code Remote-SSH（推荐）

```bash
# 在你的开发机上
code --remote ssh-remote+mi@192.168.43.247 /home/mi/
```

VS Code Server 已安装在机器人上（`.vscode-server` 目录），可以直接连接。

#### 方法 2: 通过 SSH + SCP

```bash
# 连接到机器人
ssh mi@192.168.43.247

# 传输文件
scp my_code.py mi@192.168.43.247:/home/mi/.cyberdog/cyberdog_vp/workspace/task/src/
```

#### 方法 3: 直接在机器人上编辑

```bash
ssh mi@192.168.43.247
# 使用 vim/nano 编辑代码
vim /home/mi/.cyberdog/cyberdog_vp/workspace/task/src/my_task.py
```

---

## 7. 代码部署方式

CyberDog 支持 **4 种** 代码部署方式：

### 方式 A: Python VP 任务（最简单，推荐入门）

将 Python 脚本放入 VP 工作空间，通过 VP 引擎加载运行。

**步骤**:
1. 编写 Python 任务脚本
2. 放入 `/home/mi/.cyberdog/cyberdog_vp/workspace/task/src/`
3. 在 `task.toml` 中注册任务
4. 通过 VP 引擎或语音命令触发

**示例**: 见 [Visual Programming 系统](#10-visual-programming-vp-系统) 章节

### 方式 B: Python VP 模块（可复用组件）

创建可复用的模块，供任务调用。

**步骤**:
1. 编写 Python 模块
2. 放入 `/home/mi/.cyberdog/cyberdog_vp/workspace/module/src/`
3. 在 `module.toml` 中注册模块

### 方式 C: ROS2 Python 包（独立 ROS2 节点）

创建完整的 ROS2 Python 包，作为独立节点运行。

**步骤**:
1. 创建 ROS2 包结构:
   ```
   my_package/
   ├── my_package/
   │   └── __init__.py
   │   └── my_node.py
   ├── setup.py
   ├── setup.cfg
   └── package.xml
   ```
2. 构建: `colcon build --packages-select my_package`
3. Source: `source install/setup.bash`
4. 运行: `ros2 run my_package my_node`

### 方式 D: ROS2 C++ 包（高性能节点）

创建 C++ ROS2 包，需要交叉编译。

**步骤**:
1. 创建 C++ ROS2 包结构
2. 在开发机上使用 Jetson 交叉编译工具链编译
3. 将编译产物复制到机器人
4. 或直接在机器人上编译（速度较慢）

---

## 8. Python API 参考

### 核心导入

```python
from mi.cyberdog_vp.abilityset import Cyberdog
from mi.cyberdog_vp.abilityset import StateCode
from mi.cyberdog_vp.abilityset import LedConstraint
from mi.cyberdog_vp.abilityset import MotionSequence
from mi.cyberdog_vp.abilityset import MotionSequenceGait
from mi.cyberdog_vp.abilityset import MotionSequencePace
from mi.cyberdog_vp.terminal import Visual
from mi.cyberdog_vp import decorator
from mi.cyberdog_bringup.manual import get_namespace
from mi.cyberdog_vp.utils import get_argv
```

### Cyberdog 对象 - 主要接口

```python
# 创建 Cyberdog 实例
cyberdog = Cyberdog(task_id, namespace, use_ros, task_parameters)

# 运动控制
cyberdog.motion.get_down()           # 趴下
cyberdog.motion.resume_standing()    # 恢复站立
cyberdog.motion.run_sequence(seq)    # 执行运动序列

# 语音
cyberdog.audio.play_text("你好")     # 播放语音
cyberdog.audio.set_volume(80)        # 设置音量

# LED 控制
cyberdog.led.execute(...)            # LED 灯效控制

# 任务控制
cyberdog.task.block()                # 阻塞等待任务完成
cyberdog.task.start()                # 启动任务
cyberdog.task.set_log(True)          # 启用日志

# 传感器
# 通过订阅 ROS2 topic 获取传感器数据
```

### 运动控制 - MotionSequence API

```python
from mi.cyberdog_vp.abilityset import MotionSequence, MotionSequenceGait, MotionSequencePace

# 创建运动序列
sequ = MotionSequence()
sequ.name = 'my_motion'
sequ.describe = '我的自定义动作'

# 步态控制 (每条腿的接触状态)
gait_meta = MotionSequenceGait()
gait_meta.right_forefoot = 1    # 右前腿着地
gait_meta.left_forefoot = 1     # 左前腿着地
gait_meta.right_hindfoot = 1    # 右后腿着地
gait_meta.left_hindfoot = 1     # 左后腿着地
gait_meta.duration = 500        # 持续时间(ms)
sequ.gait_list.push_back(gait_meta)

# 步伐控制 (腿部运动轨迹)
pace_meta = MotionSequencePace()
pace_meta.twist.linear.x = 0.05          # x 方向速度
pace_meta.twist.linear.y = 0.0           # y 方向速度
pace_meta.centroid.position.x = 0.0      # 质心位置
pace_meta.centroid.position.y = 0.0
pace_meta.centroid.position.z = 0.25     # 站立高度
pace_meta.duration = 1000                # 持续时间(ms)
sequ.pace_list.push_back(pace_meta)

# 执行
cyberdog.motion.run_sequence(sequ)
```

### 预置动作

预置动作配置在 `/opt/ros2/cyberdog/share/motion_action/preset/` 目录:
- `0.toml`: 站立
- `1.toml`: 趴下
- `2.toml`, `3.toml`: 握左手/右手 (伸懒腰/握手)
- `101-181.toml`: 各种动作/舞蹈
- `201-212.toml`: 步态
- `301-305.toml`: 特殊动作
- `400.toml`: 用户自定义步态

---

## 9. C++ 开发指南

### 编译环境

机器人上已安装完整的 ROS2 Galactic 和 colcon 构建系统。

### 构建步骤

```bash
# 1. Source ROS2 环境
source /opt/ros2/galactic/setup.bash
source /opt/ros2/cyberdog/setup.bash

# 2. 创建工作空间
mkdir -p ~/my_ws/src
cd ~/my_ws

# 3. 放入你的 ROS2 包到 src 目录

# 4. 构建
colcon build --packages-select your_package_name

# 5. Source
source install/setup.bash

# 6. 运行
ros2 run your_package_name your_executable
```

### 可用的库

- **OpenCV 4.1.1**: 计算机视觉
- **PCL**: 点云处理 (`pcl_ros`, `pcl_conversions`)
- **Nav2**: 导航栈
- **gRPC**: RPC 通信 (`cyberdog_grpc`)
- **Protobuf**: 序列化 (`protos`)
- **ZXing**: 二维码/条形码识别
- **NvInference**: NVIDIA 推理引擎
- **RealSense SDK**: 深度相机

---

## 10. Visual Programming (VP) 系统

VP 系统是 CyberDog 的高层应用框架，允许你通过 Python 脚本快速开发应用。

### 工作空间结构

```
/home/mi/.cyberdog/cyberdog_vp/workspace/
├── task/
│   ├── task.toml          # 任务配置
│   └── src/
│       └── your_task.py   # 你的任务代码
├── module/
│   ├── module.toml        # 模块配置
│   └── src/
│       └── your_module.py # 你的模块代码
├── choreographer/
│   └── dancer.py          # 舞蹈编排
└── log/
    └── README.md          # 日志目录
```

### 任务编写模板

```python
# my_task.py - 放在 task/src/ 下
import os
import sys
import time
import threading

import mi.cyberdog_vp.decorator
from mi.cyberdog_bringup.manual import get_namespace
from mi.cyberdog_vp.utils import get_argv
from mi.cyberdog_vp.abilityset import Cyberdog
from mi.cyberdog_vp.abilityset import StateCode

# 获取任务 ID 和参数
now_task_id, now_task_parameters = get_argv()
task_id = now_task_id if len(now_task_id) != 0 else 'my_task'
task_parameters = now_task_parameters if len(now_task_parameters) != 0 else ''

# 创建 Cyberdog 实例
print(time.strftime("任务开始时间为：%Y年%m月%d日 %H点%M分%S秒", time.localtime()))
cyberdog = Cyberdog(task_id, get_namespace(), True, task_parameters)
cyberdog.set_log(True)

# ==== 你的任务逻辑 ====

# 示例1: 让狗站立
cyberdog.motion.resume_standing()
time.sleep(2)

# 示例2: 语音播报
cyberdog.audio.play_text("你好，我是CyberDog")

# 示例3: 执行预置动作 (握左手)
# cyberdog.action.execute_preset(2)

# ==== 任务结束 ====
cyberdog.task.block()  # 阻塞等待
```

### 模块编写模板

```python
# my_module.py - 放在 module/src/ 下
import os
import sys
import threading
import mi.cyberdog_vp.decorator
import __main__

def my_function():
    """
    Describe: 我的模块功能描述
    """
    print("任务线程标识符：%d" % threading.get_ident())
    print("当前装饰器版本：%s" % mi.cyberdog_vp.decorator.version())
    cyberdog = __main__.cyberdog  # 引用任务的 Cyberdog 实例
    
    # 你的模块逻辑
    cyberdog.audio.play_text("模块被调用了")
    return True
```

### 注册任务

编辑 `task/task.toml`:

```toml
[task]
[task.my_task]
operate = []
style = "style"
condition = "now"
state = "run_wait"
be_depended = []
describe = "我的自定义任务"
mode = "cycle"
dependent = []
file = "my_task.py"
```

### 注册模块

编辑 `module/module.toml`:

```toml
[module]
[module.my_module]
file = "my_module.py"
mode = "common"
condition = "my_function()"
describe = "我的模块"
style = "style"
state = "active"
dependent = []
be_depended = []
operate = []
```

### 舞蹈编排 (Choreographer)

舞蹈编排系统允许你创建复杂的步态和动作序列。

```python
# 示例: 在 dancer.py 中定义舞蹈
from mi.cyberdog_vp.abilityset import MotionSequence, MotionSequenceGait, MotionSequencePace

def show(cyberdog_motion_id):
    """自定义舞蹈"""
    import _ctypes
    cyberdog_motion = _ctypes.PyObj_FromPtr(cyberdog_motion_id)
    
    sequ = MotionSequence()
    sequ.name = 'my_dance'
    sequ.describe = '我的舞蹈'
    
    # 定义步态和步伐...
    
    return cyberdog_motion.run_sequence(sequ)
```

### 语音命令映射

编辑 `/home/mi/.cyberdog/interaction/train_plan.json`:

```json
{
    "system": [
        {"伸懒腰": ["motion", "1"]},
        {"握左手": ["motion", "2"]},
        {"握右手": ["motion", "3"]},
        {"终止任务": ["vp_task", "shutdown"]},
        {"暂停任务": ["vp_task", "suspend"]},
        {"继续任务": ["vp_task", "recover"]},
        {"训练词一": ["unknow", "1"]},
        {"训练词二": ["unknow", "2"]},
        {"训练词三": ["unknow", "3"]}
    ],
    "user": [
        {"你的自定义指令": ["unknow", "1"]}
    ]
}
```

---

## 11. 系统启动流程

### Systemd 服务

| 服务 | 说明 | 状态 |
|------|------|------|
| `cyberdog_bringup.service` | 主服务，启动 ROS2 节点 | enabled |
| `cyberdog_sudo.service` | 特权服务（权限/蓝牙/WiFi/OTA） | enabled |
| `cyberdog_autodock.service` | 自动回充服务 | enabled |
| `cyberdog_factory.service` | 工厂模式服务 | disabled |
| `mi_preset.service` | 小米预置服务 | enabled |

### 启动顺序

```
mi_preset.service (系统预置)
    └── cyberdog_sudo.service (特权服务: 权限、WiFi、蓝牙、OTA)
    └── cyberdog_bringup.service (主服务: ROS2 核心节点)
            └── main.launch.py
                    ├── base.launch.py (核心节点)
                    │   ├── connector
                    │   ├── device_manager
                    │   ├── sensor_manager
                    │   ├── motion_manager
                    │   ├── cyberdog_manager
                    │   ├── cyberdog_vp_engine
                    │   ├── cyberdog_face
                    │   ├── cyberdog_action
                    │   ├── cyberdog_interactive
                    │   ├── cyberdog_train
                    │   ├── cyberdog_ai_sports
                    │   └── ...
                    └── navigation.launch.py (导航节点)
                        ├── nav2_base
                        ├── laser_slam / vins
                        ├── tracking
                        └── ...
```

### 手动控制服务

```bash
# 查看服务状态
sudo systemctl status cyberdog_bringup

# 停止机器人服务
sudo systemctl stop cyberdog_bringup cyberdog_sudo

# 启动机器人服务
sudo systemctl start cyberdog_sudo cyberdog_bringup

# 手动启动 (用于调试)
source /etc/mi/ros2_env.conf
rm -rf ~/.ros/log/*
ros2 launch cyberdog_bringup main.launch.py
```

---

## 12. 常见任务示例

### 示例 1: 让机器狗走路

```python
import time
from mi.cyberdog_vp.abilityset import Cyberdog, MotionSequence, MotionSequencePace
from mi.cyberdog_bringup.manual import get_namespace

cyberdog = Cyberdog("walk_task", get_namespace(), True, "")

# 加载用户自定义步态
sequ = MotionSequence()
sequ.name = 'walk_forward'
sequ.describe = '向前走'

pace = MotionSequencePace()
pace.twist.linear.x = 0.05  # 前进速度 0.05 m/s
pace.centroid.position.z = 0.25  # 站立高度
pace.duration = 5000  # 持续 5 秒
sequ.pace_list.push_back(pace)

cyberdog.motion.run_sequence(sequ)
cyberdog.task.block()
```

### 示例 2: 通过语音控制

语音命令通过 `train_plan.json` 映射到动作。说"握左手"，机器人会执行 motion preset 2。

要添加自定义语音命令：
1. 在 `train_plan.json` 的 `user` 数组中添加映射
2. 或通过 cyberdog_train 系统训练新词汇

### 示例 3: 获取传感器数据并响应

```python
# 订阅激光雷达数据检测前方障碍物
import rclpy
from sensor_msgs.msg import LaserScan

class ObstacleDetector:
    def __init__(self, cyberdog_instance):
        self.cyberdog = cyberdog_instance
        self.node = rclpy.create_node('obstacle_detector')
        self.sub = self.node.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10)
    
    def scan_callback(self, msg):
        if min(msg.ranges) < 0.3:  # 前方 30cm 有障碍物
            self.cyberdog.audio.play_text("前方有障碍物")
```

### 示例 4: 人脸识别交互

```python
# 人脸识别结果通过 topic /face_rec_msg 获取
# 配合 cyberdog_face 节点的服务进行人脸注册/识别
```

### 示例 5: 导航到指定位置

```bash
# 使用 Nav2 导航栈
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {position: {x: 1.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}"
```

---

## 13. 调试与日志

### 查看 ROS2 运行状态

```bash
# Source 环境
source /etc/mi/ros2_env.conf

# 查看所有节点
ros2 node list

# 查看所有 Topic
ros2 topic list

# 查看 Topic 数据
ros2 topic echo /motion_status
ros2 topic echo /scan
ros2 topic echo /imu_out

# 查看节点信息
ros2 node info /cyberdog_manager
```

### 查看系统日志

```bash
# ROS2 日志
ls ~/.ros/log/

# 系统服务日志
journalctl -u cyberdog_bringup -f
journalctl -u cyberdog_sudo -f

# VP 任务日志
ls /home/mi/.cyberdog/cyberdog_vp/workspace/log/

# VSCode Server 日志
ls ~/.vscode-server/
```

### 调试工具

```bash
# rviz2 可视化 (如果在本地有显示器)
rviz2

# 查看 TF 树
ros2 run tf2_tools view_frames

# 检查 DDS 通信
ros2 daemon status
```

---

## 14. 注意事项与最佳实践

### 安全警告

1. **⚠️ 紧急停止**: 机器人有物理紧急停止按钮，测试运动代码时请有人在旁
2. **⚠️ 运动范围**: 确保机器狗有足够的活动空间（至少 2m x 2m）
3. **⚠️ 高度限制**: 站立高度约 0.25m，注意上方空间
4. **⚠️ 电池**: 注意电量，低电量时停止高强度运动

### 开发建议

1. **先仿真后真机**: 如果有仿真环境，先在仿真中测试
2. **小步前进**: 运动测试从小到大（先站立、再慢走、再快走）
3. **日志先行**: 每个任务开始时打印日志，便于追踪
4. **任务命名**: 给每个任务唯一的名字，避免冲突
5. **资源释放**: 确保 ROS2 节点正常退出时释放资源
6. **DDS 限制**: 由于使用 localhost DDS，所有节点必须在机器人本地运行

### 性能优化

1. **Python vs C++**: 计算密集型任务用 C++，业务逻辑用 Python
2. **传感器频率**: 不需要高频率的订阅请使用较低的 QoS
3. **内存管理**: 8GB 内存中约 4.5GB 可用，注意大数据处理
4. **GPU 加速**: 视觉相关任务可使用 NVIDIA GPU (Jetson)

### 常见问题

| 问题 | 解决方案 |
|------|---------|
| ROS2 命令找不到 | `source /etc/mi/ros2_env.conf` |
| 节点无法通信 | 检查 ROS_DOMAIN_ID=42 和 CycloneDDS 配置 |
| 服务无法启动 | `systemctl status cyberdog_bringup` 查看日志 |
| 语音不识别 | 检查麦克风是否被遮挡 |
| 运动不响应 | 检查紧急停止是否被按下 |
| SSH 连接断开 | WiFi 不稳定，建议用有线或靠近路由器 |

### 比赛开发流程建议

```
1. 本地开发 (你的笔记本)
   ├── 编写 Python/C++ 代码
   ├── 本地测试逻辑
   └── 通过 VS Code Remote-SSH 编辑机器人上的代码

2. 部署到机器人
   ├── scp 或 VS Code 保存文件
   ├── 如需要，注册到 task.toml 或 module.toml
   └── 重启相关服务或手动触发

3. 真机测试
   ├── 通过 SSH 远程启动任务
   ├── 观察日志输出
   ├── 远程监控 ROS2 topics
   └── 调整参数

4. 迭代优化
   └── 重复 1-3
```

---

## 附录: 文件路径速查表

```
/etc/mi/ros2_env.conf              # ROS2 环境变量
/etc/mi/cyclonedds.xml             # DDS 配置
/home/mi/.cyberdog/                # CyberDog 配置与工作空间
/home/mi/.cyberdog/cyberdog_vp/workspace/  # VP 工作空间
/home/mi/.cyberdog/interaction/train_plan.json  # 语音命令映射
/home/mi/.cyberdog/connector/wifi.toml         # WiFi 配置
/opt/ros2/galactic/                # ROS2 Galactic
/opt/ros2/cyberdog/                # CyberDog 自定义包
/opt/ros2/cyberdog/share/          # ROS2 包 share 目录
/opt/ros2/cyberdog/lib/python3.6/site-packages/mi/  # Python API
/opt/ros2/cyberdog/share/cyberdog_bringup/launch/   # Launch 文件
/opt/ros2/cyberdog/share/motion_action/preset/      # 预置动作
/opt/ros2/cyberdog/share/cyberdog_vp/config/        # VP 配置
/opt/ros2/cyberdog/share/cyberdog_vp/script/        # VP 脚本
```

---

> **文档版本**: 1.0  
> **生成日期**: 2026-07-24  
> **基于**: 直接探索 CyberDog 系统生成  
> **许可**: Apache License 2.0 (与 CyberDog 软件一致)
