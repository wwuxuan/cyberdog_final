
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

import tf2_ros
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import math
import threading

def quaternion_to_euler_yaw(q_x, q_y, q_z, q_w):
    """
    将四元数转换为欧拉角中的偏航角 (Yaw)。
    返回弧度制的 Yaw 角。
    """
    siny_cosp = 2 * (q_w * q_z + q_x * q_y)
    cosy_cosp = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw

class RobotPoseMonitor(Node):
    """
    一个纯粹的TF位姿监听节点。
    它依赖外部节点来广播完整的TF树。
    """
    def __init__(self, node_name='robot_pose_monitor'):
        super().__init__(node_name)
        self.get_logger().info(f"位姿监听节点 '{self.get_name()}' 已启动 (纯监听模式)。")

        # --- TF监听器初始化 ---
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # --- 状态变量与配置 ---
        self._current_x_tf = None
        self._current_y_tf = None
        self._current_z_tf = None
        self._current_yaw_from_tf_rad = None
        self._data_lock = threading.Lock()
        
        # --- 初始偏移值，用于模拟从零点启动 ---
        self._initial_x_offset = 0.0
        self._initial_y_offset = 0.0
        self._initial_z_offset = 0.0
        self._initial_yaw_offset_rad = 0.0
        self._offset_set = False  # 标记是否已设置初始偏移值
        
        # 监听的目标坐标系
        self.reference_frame = 'odom'
        self.robot_base_frame = 'base_link_leg'

        # 创建一个定时器，定期查询TF变换
        self.tf_lookup_timer = self.create_timer(0.05, self.lookup_tf_transform) # 20 Hz
        self.get_logger().info(f"将监听并获取 '{self.robot_base_frame}' 相对于 '{self.reference_frame}' 的位姿。")

    def lookup_tf_transform(self):
        """定时器回调函数，查询最新的TF变换并更新内部状态。"""
        try:
            # 查询从 'odom' 到 'base_link' 的变换
            # TF2会自动处理 odom -> base_link_leg -> base_link 的链式关系
            transform_stamped = self.tf_buffer.lookup_transform(
                self.reference_frame, 
                self.robot_base_frame, 
                Time(), # 获取最新可用的变换
                timeout=Duration(seconds=0.1)
            )
            
            # 线程安全地更新位姿数据
            with self._data_lock:
                self._current_x_tf = transform_stamped.transform.translation.x
                self._current_y_tf = transform_stamped.transform.translation.y
                self._current_z_tf = transform_stamped.transform.translation.z
                q = transform_stamped.transform.rotation
                self._current_yaw_from_tf_rad = quaternion_to_euler_yaw(q.x, q.y, q.z, q.w)
                
                # 每秒输出一次当前的位置信息
                current_time = self.get_clock().now()
                if not hasattr(self, '_last_log_time'):
                    self._last_log_time = current_time
                
                time_diff = (current_time - self._last_log_time).nanoseconds / 1e9
                if time_diff >= 2.0:  # 每2秒输出一次
                    x = self._current_x_tf - self._initial_x_offset
                    y = self._current_y_tf - self._initial_y_offset
                    z = self._current_z_tf - self._initial_z_offset
                    yaw_deg = math.degrees(self._current_yaw_from_tf_rad - self._initial_yaw_offset_rad)
                    
                    self.get_logger().info(
                        f"当前位置: x={x:.3f}m, y={y:.3f}m, z={z:.3f}m, yaw={yaw_deg:.1f}°"
                    )   
                    self._last_log_time = current_time

        except TransformException as ex:
            # 使用带节流的日志，避免在TF暂时不可用时刷屏
            self.get_logger().warn(
                f"TF查询失败: {ex}", 
                throttle_duration_sec=1.0
            )

    # --- 线程安全的Getter方法，供外部调用 ---

    def get_coordinates_from_tf(self):
        """获取机器人的三维坐标 (x, y, z)，自动减去初始偏移值，如果不可用则返回 (None, None, None)。"""
        with self._data_lock:
            if None not in [self._current_x_tf, self._current_y_tf, self._current_z_tf]:
                x = self._current_x_tf - self._initial_x_offset
                y = self._current_y_tf - self._initial_y_offset
                z = self._current_z_tf - self._initial_z_offset
                return (x, y, z)
            return (None, None, None)

    def get_yaw_from_tf_rad(self):
        """获取机器人的偏航角 (弧度)，自动减去初始偏移值，如果不可用则返回 None。"""
        with self._data_lock:
            if self._current_yaw_from_tf_rad is not None:
                return self._current_yaw_from_tf_rad - self._initial_yaw_offset_rad
            return None

    def get_yaw_from_tf_degrees(self):
        """获取机器人的偏航角 (度)，自动减去初始偏移值，如果不可用则返回 None。"""
        yaw_rad = self.get_yaw_from_tf_rad()
        return math.degrees(yaw_rad) if yaw_rad is not None else None
    
    def get_current_pose(self):
        """获取机器人的当前位姿 (x, y, z, yaw)，自动减去初始偏移值，如果不可用则返回 (None, None, None, None)。"""
        with self._data_lock:
            if None not in [self._current_x_tf, self._current_y_tf, self._current_z_tf, self._current_yaw_from_tf_rad]:
                x = self._current_x_tf - self._initial_x_offset
                y = self._current_y_tf - self._initial_y_offset
                z = self._current_z_tf - self._initial_z_offset
                yaw_rad = self._current_yaw_from_tf_rad - self._initial_yaw_offset_rad
                return (x, y, z, yaw_rad)
            return (None, None, None, None)

    def set_initial_pose_offset(self, x=None, y=None, z=None, yaw_rad=None):
        """设置初始位姿偏移值。如果参数为None，则使用当前位姿作为偏移值。"""

        self.reset_initial_offset()
        # 尝试获取当前位姿数据，最多重试10次
        for attempt in range(10):
            self.lookup_tf_transform()
            with self._data_lock:
                if None not in [self._current_x_tf, self._current_y_tf, self._current_z_tf, self._current_yaw_from_tf_rad]:
                    break
            if attempt < 9:  # 不是最后一次尝试
                import time
                time.sleep(0.1)  # 等待100ms后重试
        
        with self._data_lock:
            if x is not None:
                self._initial_x_offset = x
            elif self._current_x_tf is not None:
                self._initial_x_offset = self._current_x_tf
                
            if y is not None:
                self._initial_y_offset = y
            elif self._current_y_tf is not None:
                self._initial_y_offset = self._current_y_tf
                
            if z is not None:
                self._initial_z_offset = z
            elif self._current_z_tf is not None:
                self._initial_z_offset = self._current_z_tf
                
            if yaw_rad is not None:
                self._initial_yaw_offset_rad = yaw_rad
            elif self._current_yaw_from_tf_rad is not None:
                self._initial_yaw_offset_rad = self._current_yaw_from_tf_rad
                
            self._offset_set = True
            
        self.get_logger().info(
            f"已设置初始位姿偏移值: X={self._initial_x_offset:.3f}, Y={self._initial_y_offset:.3f}, "
            f"Z={self._initial_z_offset:.3f}, Yaw={math.degrees(self._initial_yaw_offset_rad):.2f}°"
        )

    def set_current_pose_as_origin(self):
        """将当前位姿设置为原点（零点）。"""
        self.set_initial_pose_offset()
        self.get_logger().info("已将当前位姿设置为原点")

    def reset_initial_offset(self):
        """重置初始偏移值为零。"""
        with self._data_lock:
            self._initial_x_offset = 0.0
            self._initial_y_offset = 0.0
            self._initial_z_offset = 0.0
            self._initial_yaw_offset_rad = 0.0
            self._offset_set = False
        self.get_logger().info("已重置初始偏移值为零")

    def get_initial_offset(self):
        """获取当前的初始偏移值。"""
        with self._data_lock:
            return {
                'x': self._initial_x_offset,
                'y': self._initial_y_offset,
                'z': self._initial_z_offset,
                'yaw_rad': self._initial_yaw_offset_rad,
                'yaw_deg': math.degrees(self._initial_yaw_offset_rad),
                'offset_set': self._offset_set
            }

    def is_offset_set(self):
        """检查是否已设置初始偏移值。"""
        with self._data_lock:
            return self._offset_set

def main(args=None):
    """一个简单的main函数，用于独立测试此节点的监听功能。"""
    rclpy.init(args=args)
    pose_monitor_node = RobotPoseMonitor()
    
    # 定义一个日志打印函数，用于验证
    def log_pose_data():
        coords = pose_monitor_node.get_coordinates_from_tf()
        yaw_tf_deg = pose_monitor_node.get_yaw_from_tf_degrees()
        if coords[0] is not None and yaw_tf_deg is not None:
             pose_monitor_node.get_logger().info(f"当前位姿: X={coords[0]:.3f}, Y={coords[1]:.3f}, Yaw={yaw_tf_deg:.2f}°")
        else:
             pose_monitor_node.get_logger().info("正在等待有效的位姿数据...")

    # 创建一个1秒周期的定时器来打印日志
    log_timer = pose_monitor_node.create_timer(1.0, log_pose_data)
    
    # 使用多线程执行器，确保TF监听线程和主回调不互相阻塞
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(pose_monitor_node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        pose_monitor_node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()