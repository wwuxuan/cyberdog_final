import os
import time
import math

# 站立
def stand_up(ctrl, wait_timeout_sec=10.0, pitch = 0.0):
    """
    命令机器人执行站立动作 (Mode 12, Gait ID 0 - RECOVERY_STAND).
    这是一个阻塞式函数，会等待动作完成或超时。

    Args:
        ctrl (RobotCtrl): RobotCtrl 的实例。
        wait_timeout_sec (float): 等待动作完成的超时时间（秒）。

    Returns:
        bool: 如果动作成功完成则为 True，否则为 False。
    """
    ctrl.logger.info("基础动作: 请求站立...")
    # 根据文档附录表1，恢复站立是 Mode 12, Gait ID 0
    # duration=0 表示让机器人自行决定完成时间，依赖 order_process_bar
    success = ctrl.execute_discrete_action(
        mode=12,
        gait_id=0,
        duration_ms=0, # 让机器人自行决定完成时间
        wait_for_completion=True,
        wait_timeout_sec=wait_timeout_sec,
        pos_des = [0.0,pitch,0.0]
    )
    if success:
        ctrl.logger.info("基础动作: 站立完成。")
        time.sleep(1.0) # 站立后稳定1秒
    else:
        ctrl.logger.warn("基础动作: 站立失败或超时。")
    return success

def speed_stand(ctrl, wait_timeout_sec=10.0, pitch = 0.0, height=0.28):
    ctrl.logger.info("基础动作: 请求站立...")
    # 根据文档附录表1，恢复站立是 Mode 12, Gait ID 0
    # duration=0 表示让机器人自行决定完成时间，依赖 order_process_bar
    success = ctrl.execute_discrete_action(
        mode=11,
        gait_id=1,
        duration_ms=0, # 让机器人自行决定完成时间
        wait_for_completion=True,
        wait_timeout_sec=wait_timeout_sec,
        pos_des = [0.0,pitch,0.0],
        rpy_des = [0.0,0.0,height]
    )
    if success:
        ctrl.logger.info("基础动作: 站立完成。")
        time.sleep(1.0) # 站立后稳定1秒
    else:
        ctrl.logger.warn("基础动作: 站立失败或超时。")
    return success

# 趴下
def lie_down(ctrl, wait_timeout_sec=10.0):
    """
    命令机器人执行趴下动作 (Mode 7, Gait ID 0 - PureDamper).
    这是一个阻塞式函数，会等待动作完成或超时。

    Args:
        ctrl (RobotCtrl): RobotCtrl 的实例。
        wait_timeout_sec (float): 等待动作完成的超时时间（秒）。

    Returns:
        bool: 如果动作成功完成则为 True，否则为 False。
    """
    ctrl.logger.info("基础动作: 请求趴下...")
    # Mode 7, Gait ID 0 是 PureDamper 模式，通常用于趴下
    # 使用 set_robot_to_pure_damper 方法，它内部调用了 execute_discrete_action
    success = ctrl.set_robot_to_pure_damper(wait=True, timeout=wait_timeout_sec)

    if success:
        ctrl.logger.info("基础动作: 趴下完成。")
        time.sleep(1.0) # 趴下后稳定1秒
    else:
        ctrl.logger.warn("基础动作: 趴下失败或超时。")
    return success

# 急停
def stop_motion(ctrl, wait_timeout_sec=10.0):
    """
    命令机器人立即停止所有通过速度指令产生的运动。

    Args:
        ctrl (RobotCtrl): RobotCtrl 的实例。
    """
    ctrl.logger.info("基础动作: 急停...")
    # 根据文档附录表1，恢复站立是 Mode 12, Gait ID 0
    # duration=0 表示让机器人自行决定完成时间，依赖 order_process_bar
    success = ctrl.execute_discrete_action(
        mode=0,
        gait_id=0,
        duration_ms=0, # 让机器人自行决定完成时间
        wait_for_completion=True,
        wait_timeout_sec=wait_timeout_sec
    )
    if success:
        ctrl.logger.info("基础动作: 急停完成。")
        time.sleep(1.0) # 站立后稳定1秒
    else:
        ctrl.logger.warn("基础动作: 急停失败或超时。")
    return success

# 直线行走
def move_straight_timed(ctrl, distance_m, speed_mps,
                        body_height=0.28, gait_id_walk=26,pitch=0.0,
                        add_final_stabilization_delay=True): # 新增参数
    """
    【基于时间的开环控制】命令机器人直线前进或后退指定距离。
    正距离表示前进，负距离表示后退。速度应为正值。
    """
    ctrl.logger.info(f"基础开环动作: 直线移动 {distance_m:.2f}m, 速度 {speed_mps:.2f}m/s。")

    if speed_mps <= 0:
        # ctrl.logger.error("移动速度必须为正值。")
        return False
    if distance_m == 0:
        # ctrl.logger.info("移动距离为0，不执行动作。")
        return True

    duration_s = abs(distance_m / speed_mps)
    if duration_s < 0.01 and distance_m != 0 : # 避免几乎为零的无效移动
        duration_s = 0.01

    actual_linear_x_speed = speed_mps if distance_m > 0 else -speed_mps

    ctrl.logger.info(f"  预计运动时间: {duration_s:.3f}s, 发送速度: {actual_linear_x_speed:.2f}m/s")

    ctrl.set_velocity_command(
        linear_x=actual_linear_x_speed, linear_y=0.0, angular_z=0.0,
        body_height=body_height, gait_id=gait_id_walk, pitch=pitch,
    )

    start_time = time.monotonic()
    while time.monotonic() - start_time < duration_s:
        time.sleep(0.01)
        if hasattr(ctrl, '_is_running') and not ctrl._is_running:
            ctrl.logger.warn("RobotCtrl 停止运行，直线移动中断。")
            ctrl.set_velocity_command(linear_x=0.0, linear_y=0.0, angular_z=0.0, body_height=body_height, gait_id=gait_id_walk)
            return False

    ctrl.set_velocity_command(
        linear_x=0.0, linear_y=0.0, angular_z=0.0,
        body_height=body_height, gait_id=gait_id_walk
    )
    ctrl.logger.info(f"  开环直线移动（持续{duration_s:.3f}s）完成，已发送停止指令。")

    if add_final_stabilization_delay:
        time.sleep(0.5) # 默认0.3s
    
    return True

# 原地旋转
def turn_in_place_timed(ctrl, angle_degrees, angular_speed_dps,
                        body_height=0.28, gait_id_turn=26,
                        add_final_stabilization_delay=True): # 新增参数
    """
    【基于时间的开环控制】命令机器人原地旋转指定角度。
    正角度表示逆时针/左转，负角度表示顺时针/右转。角速度应为正值。
    """
    ctrl.logger.info(f"基础开环动作: 请求原地旋转 {angle_degrees:.1f}°, 角速度 {angular_speed_dps:.1f}dps。") # 日志由调用者处理或按需保留

    if angular_speed_dps <= 0:
        # ctrl.logger.error("旋转角速度必须为正值。") # 日志由调用者处理
        return False # 或者抛出异常
    if angle_degrees == 0:
        # ctrl.logger.info("旋转角度为0，不执行动作。")
        return True

    angle_rad = math.radians(angle_degrees)
    angular_speed_rps = math.radians(angular_speed_dps)

    duration_s = abs(angle_rad / angular_speed_rps)
    if duration_s < 0.01 and angle_degrees != 0: # 避免几乎为零的无效旋转
        duration_s = 0.01 # 最小执行一小段时间
        
    actual_angular_z_speed = angular_speed_rps if angle_degrees > 0 else -angular_speed_rps

    ctrl.logger.info(f"  预计旋转时间: {duration_s:.3f}s, 发送角速度: {actual_angular_z_speed:.3f}rad/s")

    ctrl.set_velocity_command(
        linear_x=0.0, linear_y=0.0, angular_z=actual_angular_z_speed,
        body_height=body_height, gait_id=gait_id_turn
    )

    start_time = time.monotonic()
    while time.monotonic() - start_time < duration_s:
        time.sleep(0.01) # 短暂sleep允许其他线程运行
        if hasattr(ctrl, '_is_running') and not ctrl._is_running:
            ctrl.logger.warn("RobotCtrl 停止运行，原地旋转中断。")
            ctrl.set_velocity_command(linear_x=0.0, linear_y=0.0, angular_z=0.0, body_height=body_height, gait_id=gait_id_turn)
            return False

    ctrl.set_velocity_command(
        linear_x=0.0, linear_y=0.0, angular_z=0.0,
        body_height=body_height, gait_id=gait_id_turn
    )
    # ctrl.logger.info(f"  开环旋转动作（持续{duration_s:.3f}s）完成，已发送停止指令。")

    if add_final_stabilization_delay:
        time.sleep(0.5) # 仅当需要时才添加稳定延时，默认0.3s
    
    return True

# 侧向行走
def move_lateral_timed(ctrl, distance_m, speed_mps, body_height=0.28, gait_id_lateral=26,add_final_stabilization_delay=True):
    """
    【基于时间的开环控制】命令机器人横向平移指定距离。
    注意：这是一个开环控制方法，实际移动距离可能因机器人能力、打滑等因素与期望距离有差异。
    正距离表示向机器人左侧平移，负距离表示向机器人右侧平移。速度应为正值。
    这个函数假设机器人支持通过设置 vel_des[1] (身体Y轴速度) 来进行横向平移。

    Args:
        ctrl (RobotCtrl): RobotCtrl 的实例。
        distance_m (float): 横向平移距离 (米)。正为向左，负为向右。
        speed_mps (float): 横向平移速度 (米/秒)，应为正值。
        body_height (float): 平移时的身体高度。
        gait_id_lateral (int): 横向平移时使用的步态ID。

    Returns:
        bool: 操作指令是否成功发送并等待了预定时间。
    """
    ctrl.logger.info(f"基础动作: 请求横向平移 {distance_m:.2f}m, 速度 {speed_mps:.2f}m/s (基于时间开环)。")

    if not hasattr(ctrl._current_lcm_cmd, 'vel_des') or len(ctrl._current_lcm_cmd.vel_des) < 2:
        ctrl.logger.error("RobotCtrl 的 _current_lcm_cmd.vel_des 结构不正确，无法执行横向平移。")
        return False

    if speed_mps <= 0:
        ctrl.logger.error("横向平移速度必须为正值。")
        return False
    if distance_m == 0:
        ctrl.logger.info("横向平移距离为0，不执行动作。")
        return True

    duration_s = abs(distance_m / speed_mps)
    # vel_des[1] 控制身体 Y 轴速度 (通常左为正)
    actual_linear_y_speed = speed_mps if distance_m > 0 else -speed_mps

    ctrl.logger.info(f"  预计平移时间: {duration_s:.2f}s, 发送身体Y轴速度: {actual_linear_y_speed:.2f}m/s")

    # 发送横向平移指令
    # Mode 11 通常是 LOCOMOTION 模式
    ctrl.set_velocity_command(
        linear_x=0.0, # X方向速度为0
        linear_y=actual_linear_y_speed,
        angular_z=0.0, # 角速度为0
        body_height=body_height,
        gait_id=gait_id_lateral
    )

    start_time = time.monotonic()
    while time.monotonic() - start_time < duration_s:
        time.sleep(0.01)
        if not hasattr(ctrl, '_is_running') or not ctrl._is_running:
            ctrl.logger.warn("RobotCtrl 停止运行，横向平移中断。")
            ctrl.set_velocity_command(linear_x=0.0, linear_y=0.0, angular_z=0.0, body_height=body_height, gait_id=gait_id_lateral)
            return False

    ctrl.logger.info(f"  预计时间已到，发送停止指令。")
    ctrl.set_velocity_command(
        linear_x=0.0,
        linear_y=0.0,
        angular_z=0.0,
        body_height=body_height,
        gait_id=gait_id_lateral
    )
    if add_final_stabilization_delay:
        time.sleep(0.5) # 仅当需要时才添加稳定延时，默认0.3s

    ctrl.logger.info(f"基础动作: 横向平移 {distance_m:.2f}m (基于时间) 指令序列完成。")
    return True

# 曲线行走
def walk_in_arc_timed(ctrl, linear_speed_mps, angular_speed_dps, duration_s, body_height=0.28, gait_id_walk=26):
    """
    【基于时间的开环控制】命令机器人以指定的线速度和角速度进行弧线行走。
    正的角速度表示逆时针/左转。

    Args:
        ctrl (RobotCtrl): RobotCtrl 的实例。
        linear_speed_mps (float): 机器人的前进线速度 (米/秒)。
        angular_speed_dps (float): 机器人的旋转角速度 (度/秒)。
        duration_s (float): 弧线行走的持续时间 (秒)。
        body_height (float): 行走时的身体高度。
        gait_id_walk (int): 行走时使用的步态ID。

    Returns:
        bool: 操作指令是否成功发送并等待了预定时间。
    """
    ctrl.logger.info(f"基础动作: 请求弧线行走, 线速度 {linear_speed_mps:.2f}m/s, 角速度 {angular_speed_dps:.1f}dps, 持续 {duration_s:.2f}s (开环)。")

    if duration_s <= 0:
        ctrl.logger.info("弧线行走持续时间为0或负，不执行动作。")
        return True

    angular_speed_rps = math.radians(angular_speed_dps)

    ctrl.logger.info(f"  发送指令: 线速度X={linear_speed_mps:.2f}m/s, 角速度Z={angular_speed_rps:.3f}rad/s")

    # 发送弧线行走指令
    ctrl.set_velocity_command(
        linear_x=float(linear_speed_mps),
        linear_y=0.0, # 假设弧线行走主要通过前进和旋转组合，不使用横向平移
        angular_z=float(angular_speed_rps),
        body_height=body_height,
        gait_id=gait_id_walk
    )

    start_time = time.monotonic()
    while time.monotonic() - start_time < duration_s:
        time.sleep(0.01)
        if not hasattr(ctrl, '_is_running') or not ctrl._is_running:
            ctrl.logger.warn("RobotCtrl 停止运行，弧线行走中断。")
            ctrl.set_velocity_command(linear_x=0.0, linear_y=0.0, angular_z=0.0, body_height=body_height, gait_id=gait_id_walk)
            return False

    ctrl.logger.info(f"  预计时间已到，发送停止指令。")
    ctrl.set_velocity_command(
        linear_x=0.0,
        linear_y=0.0,
        angular_z=0.0, # 停止旋转和前进
        body_height=body_height,
        gait_id=gait_id_walk
    )
    time.sleep(0.5)

    ctrl.logger.info(f"基础动作: 弧线行走指令序列完成。")
    return True

# 自定义步态
def execute_custom_gait_sequence(ctrl,
                                 gait_params_file_path,
                                 gait_def_file_path,
                                 user_gait_list_file_path,
                                 base_working_dir=".",
                                 wait_for_file_ack_timeout_sec=5.0):
    """
    加载、发送并执行一个完整的自定义步态序列。
    这个函数是对 RobotCtrl 类中 load_and_execute_custom_gait 方法的封装调用。

    Args:
        ctrl (RobotCtrl): RobotCtrl 的实例。
        gait_params_file_path (str): 原始步态参数文件的路径 (例如 "Gait_Params_moonwalk.toml")。
        gait_def_file_path (str): 步态定义文件的路径 (例如 "Gait_Def_moonwalk.toml")。
        user_gait_list_file_path (str): 用户步态序列文件的路径 (例如 "Usergait_List.toml")。
        base_working_dir (str): 用于存放生成的 "full" 参数文件的目录。
                                 默认为当前目录 "."。
        wait_for_file_ack_timeout_sec (float): 等待步态文件发送ACK的超时时间（秒）。

    Returns:
        bool: 如果整个自定义步态序列成功加载并执行则为 True，否则为 False。
    """
    ctrl.logger.info(f"基础动作: 请求执行自定义步态序列。")
    ctrl.logger.info(f"  定义文件: {gait_def_file_path}")
    ctrl.logger.info(f"  参数文件: {gait_params_file_path}")
    ctrl.logger.info(f"  列表文件: {user_gait_list_file_path}")

    # 确保所有文件路径存在，以避免在 RobotCtrl 内部才发现问题
    if not os.path.exists(gait_params_file_path):
        ctrl.logger.error(f"自定义步态错误: 参数文件未找到 '{gait_params_file_path}'")
        return False
    if not os.path.exists(gait_def_file_path):
        ctrl.logger.error(f"自定义步态错误: 定义文件未找到 '{gait_def_file_path}'")
        return False
    if not os.path.exists(user_gait_list_file_path):
        ctrl.logger.error(f"自定义步态错误: 列表文件未找到 '{user_gait_list_file_path}'")
        return False

    success = ctrl.load_and_execute_custom_gait(
        gait_params_file_path=gait_params_file_path,
        gait_def_file_path=gait_def_file_path,
        user_gait_list_file_path=user_gait_list_file_path,
        base_working_dir=base_working_dir,
        wait_for_file_ack_timeout_sec=wait_for_file_ack_timeout_sec
    )

    if success:
        ctrl.logger.info("基础动作: 自定义步态序列执行完成。")
    else:
        ctrl.logger.warn("基础动作: 自定义步态序列执行失败或中途中断。")
    
    return success