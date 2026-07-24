# 初赛代码 → 真机移植指南

## 项目概况

- **初赛**: Gazebo 仿真环境，6 个关卡（直线走/打橙球/曲线跟随/障碍赛/独木桥跳/推足球）
- **真机**: 小米 CyberDog，Ubuntu 18.04 + ROS2 Galactic
- **总代码量**: Python ~3500 行 + C++ ~7400 行（C++ 部分全部替换）

---

## 1. 最大改动：通信层

这是**最核心的改动**。初赛用了两套通信，真机只有 ROS2。

```
初赛架构:                          真机架构:
┌──────────┐                      ┌──────────┐
│  main.py │                      │  main.py │
└────┬─────┘                      └────┬─────┘
     │                                 │
  LCM 发指令                       ROS2 topic/action
  LCM 收状态                       或 VP Python API
     │                                 │
┌────┴─────┐                      ┌────┴─────────────┐
│ Gazebo   │                      │ cyberdog_manager  │
│ Plugin   │                      │ motion_manager    │
│ MPC C++  │                      │ 真机控制器         │
└──────────┘                      └──────────────────┘
```

**需要改的**：

| 初赛文件 | 初赛方式 | 真机方式 |
|----------|---------|---------|
| `dog_ctrl.py` | LCM `robot_control_cmd` | 直接用 `Navigation` action 或 `cyberdog.motion.*` API |
| `monitor.py` | LCM 两个端口收状态 | 订阅 ROS2 topic `/odom_out`, `/motion_status` 等 |

---

## 2. 各模块移植分析

### dog_ctrl.py → 直接废弃

```python
# 初赛: LCM 发步态指令
msg = robot_control_cmd_lcmt()
msg.mode = 11; msg.gait_id = 27; msg.vel_des = [0.3, 0, 0]
lcm.publish("robot_control_cmd", msg)

# 真机: 方法 A - VP API
cyberdog.motion.run_sequence(sequ)  # 运动序列

# 真机: 方法 B - Navigation Action
goal = Navigation.Goal()
goal.nav_type = Navigation.Goal.NAVIGATION_TYPE_START_AB
goal.poses = [target_pose]
nav_client.send_goal(goal)
```

### monitor.py → 订阅 ROS2 topic

```python
# 初赛: LCM 监听两个端口
lcm.subscribe("simulator_state", _pos_loop)

# 真机: 订阅 ROS2 topic
node.create_subscription(Odometry, '/odom_out', callback, 10)
node.create_subscription(MotionStatus, '/motion_status', callback, 10)
```

### stage2.py (打橙球) → 需要改摄像头

| 项目 | 初赛 | 真机 |
|------|------|------|
| 摄像头 | `/rgb_camera/image_raw` 320×180 | `/camera/color/image_raw` 或 `/image_rgb` |
| HSV 阈值 | 仿真调好的 | **需要重新标定**（真机光照不同） |
| 激光 | `/scan` 180采样 ±90° | `/scan`（参数可能不同） |
| 路径行走 | LCM 发速度指令 | `Navigation` action 发坐标目标 |

### stage3.py (MPC曲线跟随) → MPC可复用，控制接口要换

```python
# MPC 数学部分完全可复用:
# - 路径加载、最近点查找、误差计算、OSQP求解

# 需要改的:
# 初赛: dog.vel_des = [vx, vy, wz]  # 直接改 LCM 消息
# 真机: cyberdog.motion 发速度，或 Navigation action 发连续坐标点
```

### stage4.py (障碍赛) → YOLO 模型要验证

| 项目 | 初赛 | 真机 |
|------|------|------|
| YOLO 模型 | `full_classes.pt` 仿真训练的 | **需真机图片验证**，可能要重新标注 |
| 限高杆检测 | YOLO bbox 面积阈值 | 阈值需重新标定 |
| 蓝色方块 | YOLO + 激光 1.67m 判断 | 激光阈值可能不同 |
| 低姿态行走 | `gait_id=83, z=-0.08` | 需映射到 CyberDog 的低姿步态 |

### stage5.py (独木桥+跳) → 特殊步态需映射

| 初赛步态 | 真机对应 |
|----------|---------|
| `GAIT_TILT` (roll=-0.4) 倾斜走 | 查 CyberDog 是否支持 roll 偏移步态 |
| `GAIT_LEFT_WALK` 横向走 | `MotionSequencePace` 设置 y 方向速度 |
| `GAIT_JUMP` (mode=16, jump3D) | 查 `motion_action/preset/` 中的跳跃动作 |

### stage6.py (推足球) → 最复杂，重写量最大

初赛用一个 7 状态 FSM：
```
SCAN → APPROACH → CIRCLE → AIM_SCAN → ATTACK_APPROACH → PUSH → BACK_OFF
```

真机移植要点：
- 状态机逻辑 **可复用**
- 所有 `_send(dog, vx, vy, wz)` 调用需要换成真机接口
- `_estimate_ball_world()` 深度估计依赖仿真相机内参，真机需要重新标定 `fx`
- `FB_WALL_*` 边界常量需要根据真机地图更新
- 足球检测模型 `football.pt` 可能需要在真机图片上验证

---

## 3. 移植步骤（按优先级）

### 第一步：封装统一控制接口（最重要）

把初赛中所有 `_send(dog, vx, vy, wz)` 调用封装到一个适配层：

```python
# adapter.py - 统一控制接口
class CyberDogAdapter:
    def __init__(self):
        # 方式 A: VP API
        self.cyberdog = Cyberdog("task_id", get_namespace(), True, "")
        # 方式 B: ROS2 Navigation Action
        self.nav_client = ActionClient(node, Navigation, 'navigation')
    
    def walk(self, vx, vy=0.0, wz=0.0, duration=None):
        """替代初赛的 _send() 函数"""
        # 速度控制 → 映射到 CyberDog motion API
        sequ = MotionSequence()
        pace = MotionSequencePace()
        pace.twist.linear.x = vx
        pace.twist.linear.y = vy
        pace.twist.angular.z = wz
        pace.duration = duration * 1000 if duration else 500
        sequ.pace_list.push_back(pace)
        self.cyberdog.motion.run_sequence(sequ)
    
    def walk_to(self, x, y, yaw=None):
        """替代初赛的 walk_to_xy() 函数"""
        goal = Navigation.Goal()
        goal.nav_type = Navigation.Goal.NAVIGATION_TYPE_START_AB
        goal.poses = [make_pose(x, y, yaw or 0.0)]
        self.nav_client.send_goal(goal)
    
    def turn_to(self, target_yaw):
        """替代初赛的 turn_to_yaw() 函数"""
        # 用 Navigation action 原地旋转
        ...
    
    def get_position(self):
        """替代初赛的 Monitor.msg()"""
        # 订阅 /odom_out 或 /tf 获取当前位置
        return {'x': ..., 'y': ..., 'yaw': ...}
```

### 第二步：改 monitor.py → 真机状态获取

```python
# 真机 monitor.py
import rclpy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

class RobotMonitor:
    def __init__(self, node):
        self.position = [0, 0, 0]
        self.rpy = [0, 0, 0]
        node.create_subscription(Odometry, '/odom_out', self._odom_cb, 10)
        node.create_subscription(Imu, '/imu_out', self._imu_cb, 10)
    
    def msg(self):
        return {'position': self.position, 'rpy_deg': self.rpy, 'gait_mode': None}
```

### 第三步：更新摄像头和激光 topic

```python
# 初赛
self.rgb_sub = node.create_subscription(Image, '/rgb_camera/image_raw', cb, 10)
self.scan_sub = node.create_subscription(LaserScan, '/scan', cb, 10)

# 真机（topic 名可能不同，需确认）
self.rgb_sub = node.create_subscription(Image, '/camera/color/image_raw', cb, 10)
self.scan_sub = node.create_subscription(LaserScan, '/scan', cb, 10)
```

### 第四步：重新标定视觉参数

| 参数 | 初赛值 | 真机需要 |
|------|--------|---------|
| HSV 橙色范围 | 仿真调好的 | 真机光照下重新标定 |
| YOLO bbox 阈值 | 15000 px² | 根据真机分辨率调整 |
| 相机 fx | 179.8 px | 从 RealSense 标定文件获取 |
| 激光采样数/FOV | 180/±90° | 查真机 `/scan` 实际参数 |
| 球体直径 | 0.22m | 如果比赛用球相同则不变 |

### 第五步：更新坐标常量为真机地图

初赛所有硬编码坐标都基于仿真地图：

```python
# 初赛坐标（来自仿真地图）
STAGE1_END_X = 3.07
STAGE2_EXIT = (0.159, 4.28)
STAGE3_ENTRY = (-0.340, 4.315)
FB_WALL_X = [0.10, 2.90]
...

# 真机 → 全部替换为建图后的实际坐标
# 建议用 map_label_server 替代硬编码坐标
```

### 第六步：C++ 代码 → 全部废弃

`legged_plugin.cpp`、`convex_mpc_loco_gaits.cpp`、`convex_mpc_motion_gaits.cpp` — 这三个是 Gazebo 仿真专用的，真机不需要编译。

---

## 4. 可以复用的 vs 必须重写的

| 模块 | 可复用度 | 说明 |
|------|---------|------|
| stage2 BFS路径规划 | ✅ 100% | 纯算法，不依赖仿真 |
| stage2 360度扫描逻辑 | ✅ 80% | 换摄像头 topic，重标 HSV |
| stage3 MPC 数学 | ✅ 95% | 只改控制输出接口 |
| stage4 YOLO 检测 | ✅ 70% | 模型可能需验证/重训 |
| stage4 导航原语 | ⚠️ 40% | 逻辑可复用，接口全换 |
| stage5 区域状态机 | ✅ 80% | 换坐标常量+步态映射 |
| stage5 跳跃序列 | ⚠️ 30% | 完全依赖真机跳跃能力 |
| stage6 FSM 状态机 | ✅ 80% | 逻辑可复用，接口全换 |
| stage6 单目深度估计 | ⚠️ 50% | 需重新标定相机内参 |
| main.py 关卡调度 | ⚠️ 50% | 改坐标常量+状态接口 |
| dog_ctrl.py | ❌ 0% | 完全重写为 CyberDog 接口 |
| monitor.py | ❌ 10% | 从 LCM 改为 ROS2 topic |
| C++ 代码 | ❌ 0% | Gazebo 专用，全部废弃 |

---

## 5. 建议的移植顺序

```
第1天: 建图 + 标定
  ├── 真机场地建图（绕场一圈）
  ├── 标定关键坐标（用 map_label_server）
  ├── 验证摄像头/激光 topic
  └── 标定 HSV + YOLO 在真机图片上的参数

第2天: 适配层 + 基础控制
  ├── 写 CyberDogAdapter（统一控制接口）
  ├── 改 monitor.py → ROS2 topic
  ├── 验证 walk_to() 和 turn_to()
  └── 跑通 stage1（直线走）

第3天: 逐个关卡移植
  ├── stage2: 换摄像头 topic + 重标 HSV
  ├── stage3: 接 MPC 输出到 adapter
  ├── stage4: 验证 YOLO 模型
  ├── stage5: 检查真机是否支持倾斜/横走/跳跃
  └── stage6: 最复杂，最后搞

第4天: 联调 + 全流程
  ├── 跑通全流程
  ├── 调参数（速度、阈值、超时）
  └── 处理异常情况
```

---

## 6. 关键风险点

| 风险 | 影响 | 对策 |
|------|------|------|
| CyberDog 不支持某些特殊步态（倾斜走/跳跃） | stage5 可能跑不了 | 提前确认步态能力，备选方案：绕行 |
| YOLO 模型在真机上误检/漏检 | stage4/6 失败 | 提前采集真机图片验证模型 |
| 真机建图原点与仿真地图不对齐 | 所有坐标都要改 | 用 map_label_server 替代硬编码 |
| Navigation action 导航精度不如仿真 | 可能撞障碍物 | 降低速度，增加安全距离 |
| 真机摄像头视野/FPS 不同 | 检测窗口不匹配 | 重新标定所有视觉参数 |

---

> **核心结论**: 初赛约 60% 的 Python 逻辑代码可以复用（算法/状态机/规划），但 100% 的通信层和底层控制需要重写。先写一个 `CyberDogAdapter` 统一接口层，然后逐个关卡接上去即可。
