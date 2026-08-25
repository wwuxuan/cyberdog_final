import rclpy
import math
import time
from motion.utils.basic_action import turn_in_place_timed, move_straight_timed, move_lateral_timed

# 假设 RobotPoseMonitor 和 RobotCtrl 类已正确导入
# 以及开环动作函数 turn_in_place_timed 和 move_straight_timed, move_lateral_timed

def _normalize_angle_for_basic_action(angle_rad):
    while angle_rad > math.pi:
        angle_rad -= 2 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2 * math.pi
    return angle_rad

def align_yaw_to_target(logger, robot_ctrl, pose_monitor,
                        target_yaw_degrees,
                        tolerance_degrees=2.0,
                        max_attempts=10,
                        min_angular_speed_dps=3.0,    # 最小转动速度
                        max_angular_speed_dps=15.0,   # 最大转动速度
                        angular_speed_scale_factor=1.0, # 角速度缩放因子
                        max_rotation_per_step_degrees=15.0):
    """
    【闭环-动态速度】精确调整机器人偏航角至目标角度。
    根据偏差距离动态计算转动速度。
    """
    target_yaw_rad = math.radians(target_yaw_degrees)

    logger.info(f"校准偏航角 -> 目标: {target_yaw_degrees:.2f}° (容差: ±{tolerance_degrees:.2f}°)")
    logger.info(f"  最小转动速度: {min_angular_speed_dps}°/s, 最大转动速度: {max_angular_speed_dps}°/s")

    for attempt_count in range(max_attempts):
        # 获取当前偏航角
        current_yaw_rad = pose_monitor.get_yaw_from_tf_rad()  # tf 获取
        total_yaw_error_rad = _normalize_angle_for_basic_action(target_yaw_rad - current_yaw_rad)
        total_yaw_error_deg_abs = abs(math.degrees(total_yaw_error_rad)) # 用于日志和速度选择

        logger.info(f"  尝试 {attempt_count + 1}/{max_attempts}: "
                         f"目标Yaw: {target_yaw_degrees:.2f}°, "
                         f"当前Yaw: {math.degrees(current_yaw_rad):.2f}°, "
                         f"误差: {math.degrees(total_yaw_error_rad):.2f}°")

        if total_yaw_error_deg_abs <= tolerance_degrees: # 直接用度数比较容差
            logger.info(f"  成功对准目标偏航角 {target_yaw_degrees:.2f}°。")
            robot_ctrl.set_velocity_command(0.0,0.0,0.0)
            return True

        # 根据偏差距离动态计算角速度
        # 角速度与偏差成正比，但限制在最小和最大值之间
        dynamic_angular_speed_dps = total_yaw_error_deg_abs * angular_speed_scale_factor
        dynamic_angular_speed_dps = max(min_angular_speed_dps, min(dynamic_angular_speed_dps, max_angular_speed_dps))

        # 限制单步旋转角度
        rotation_this_step_degrees = math.degrees(total_yaw_error_rad)
        if abs(rotation_this_step_degrees) > max_rotation_per_step_degrees:
            rotation_this_step_degrees = math.copysign(max_rotation_per_step_degrees, rotation_this_step_degrees)
        
        logger.info(f"    执行单步旋转: {rotation_this_step_degrees:.2f}° @ {dynamic_angular_speed_dps:.1f}°/s (动态速度)")
        step_success = turn_in_place_timed(
            ctrl=robot_ctrl,
            angle_degrees=rotation_this_step_degrees,
            angular_speed_dps=dynamic_angular_speed_dps, # 使用动态计算的速度
            add_final_stabilization_delay=False
        )
        time.sleep(1)
        if not step_success:
            logger.warn("    单步开环旋转执行失败。")

    logger.warn(f"达到最大尝试次数 ({max_attempts}) 未能对准目标偏航角 {target_yaw_degrees:.2f}°。")
    robot_ctrl.set_velocity_command(0.0,0.0,0.0)
    return False

def align_axis_by_driving_forward(logger, robot_ctrl, pose_monitor,
                                  target_coord_value,
                                  axis_to_align, # "X" 或 "Y"
                                  direction= 1, # "+1" 或 "-1"
                                  tolerance_m=0.02,
                                  max_attempts=20,
                                  fast_linear_speed_mps=0.05,
                                  slow_down_xy_threshold_m=0.1, 
                                  slow_linear_speed_mps=0.03,  
                                  max_move_step_m=0.07):
    """
    通过前进/后退的方式，校准指定世界轴方向的位置。
    线速度会根据距离目标的远近分阶段调整。
    """
    if axis_to_align not in ["X", "Y"]:
        logger.error(f"无效的轴: {axis_to_align}。必须是 'X' 或 'Y'。")
        return False

    logger.info(f"校准世界{axis_to_align}轴（前进/后退） -> 目标: {target_coord_value:.3f}m")

    for attempt_count in range(max_attempts):
        # 获取当前坐标
        current_coords = pose_monitor.get_coordinates_from_tf()
        current_x, current_y, _ = current_coords
        current_coord_val_on_axis = current_x if axis_to_align == "X" else current_y
        coord_error_m = target_coord_value - current_coord_val_on_axis
        abs_coord_error_m = abs(coord_error_m)
        # 输出当前状态
        logger.info(f"  尝试 {attempt_count + 1}/{max_attempts} ({axis_to_align}轴): "
                         f"目标: {target_coord_value:.3f}, 当前: {current_coord_val_on_axis:.3f}, "
                         f"误差: {coord_error_m:.3f}m")
        # 判断是否成功
        if abs_coord_error_m <= tolerance_m:
            logger.info(f"  成功对准世界{axis_to_align}轴至 {target_coord_value:.3f}m。")
            robot_ctrl.set_velocity_command(0.0,0.0,0.0)
            return True

        # 根据误差选择线速度
        current_fixed_linear_speed_mps = fast_linear_speed_mps if abs_coord_error_m > slow_down_xy_threshold_m else slow_linear_speed_mps

        if abs_coord_error_m > max_move_step_m:
            coord_error_m = math.copysign(max_move_step_m, coord_error_m)
        
        logger.info(f"    执行单步沿世界{axis_to_align}轴移动: {abs_coord_error_m:.3f}m @ {current_fixed_linear_speed_mps}m/s (当前使用 {'快速' if current_fixed_linear_speed_mps == fast_linear_speed_mps else '慢速'} 模式)")
        step_move_success = move_straight_timed(
            ctrl=robot_ctrl,
            distance_m=coord_error_m * direction,
            speed_mps=current_fixed_linear_speed_mps, # 使用当前阶段的速度
            add_final_stabilization_delay=False
        )
        if not step_move_success:
            logger.warn(f"    单步沿世界{axis_to_align}轴移动执行失败。")
        time.sleep(1)

    logger.warn(f"达到最大尝试次数 ({max_attempts}) 未能对准世界{axis_to_align}轴至 {target_coord_value:.3f}m。")
    robot_ctrl.set_velocity_command(0.0,0.0,0.0)
    return False

def align_axis_by_strafing(logger, robot_ctrl, pose_monitor,
                           target_coord_value,
                           axis_to_adjust_laterally, # "X" 或 "Y"
                           direction= 1, # "+1" 或 "-1"
                           tolerance_m=0.02,
                           max_attempts=20,
                           fast_lateral_speed_mps=0.05,
                           slow_down_lateral_threshold_m=0.1, # 误差小于5cm时用慢速侧移
                           slow_lateral_speed_mps=0.01,
                           max_lateral_move_step_m=0.1,
                           stabilization_time_s=1.0):
    """
    通过侧向平移校准指定的世界坐标轴。
    侧移线速度会根据距离目标的远近分阶段调整。
    """
    if axis_to_adjust_laterally not in ["X", "Y"]:
        logger.error(f"无效的侧向调整轴: {axis_to_adjust_laterally}。")
        return False

    logger.info(f"校准世界{axis_to_adjust_laterally}轴（侧向平移） -> 目标: {target_coord_value:.3f}m")
    logger.info(f"  快速侧移速度: {fast_lateral_speed_mps}m/s (误差 > {slow_down_lateral_threshold_m}m 时)")
    logger.info(f"  慢速侧移速度: {slow_lateral_speed_mps}m/s (误差 <= {slow_down_lateral_threshold_m}m 时)")

    for attempt_count in range(max_attempts):
        # 获取当前坐标
        current_coords = pose_monitor.get_coordinates_from_tf()
        current_x, current_y, _ = current_coords
        current_coord_val_on_axis = current_x if axis_to_adjust_laterally == "X" else current_y
        world_error_on_axis = target_coord_value - current_coord_val_on_axis
        abs_world_error_on_axis = abs(world_error_on_axis)
        # 输出当前状态
        logger.info(f"  尝试 {attempt_count + 1}/{max_attempts} ({axis_to_adjust_laterally}轴侧移): "
                         f"目标: {target_coord_value:.3f}, 当前: {current_coord_val_on_axis:.3f}, "
                         f"误差: {world_error_on_axis:.3f}m")
        # 判断是否成功
        if abs_world_error_on_axis <= tolerance_m:
            logger.info(f"  成功通过侧向平移对准世界{axis_to_adjust_laterally}轴。")
            robot_ctrl.set_velocity_command(0.0,0.0,0.0)
            return True
        
        # 根据误差选择侧移速度
        current_fixed_lateral_speed_mps = fast_lateral_speed_mps if abs_world_error_on_axis > slow_down_lateral_threshold_m else slow_lateral_speed_mps
        if abs_world_error_on_axis > max_lateral_move_step_m:
            world_error_on_axis = math.copysign(max_lateral_move_step_m, world_error_on_axis)

        
        logger.info(f"    执行单步侧向平移 (身体Y轴): {abs_world_error_on_axis:.3f}m @ {current_fixed_lateral_speed_mps}m/s (当前使用 {'快速' if current_fixed_lateral_speed_mps == fast_lateral_speed_mps else '慢速'} 模式)")
        step_success = move_lateral_timed(
            ctrl=robot_ctrl,
            distance_m=world_error_on_axis * direction,
            speed_mps=current_fixed_lateral_speed_mps, # 使用当前阶段的侧移速度
            add_final_stabilization_delay=False
        )
        if not step_success:
            logger.warn(f"    单步侧向平移执行失败。")
        time.sleep(stabilization_time_s)
            
    logger.warn(f"达到最大尝试次数 ({max_attempts}) 未能通过侧向平移对准世界{axis_to_adjust_laterally}轴。")
    robot_ctrl.set_velocity_command(0.0,0.0,0.0)
    return False

# navigate_to_exact_pose 函数也需要更新其调用这些校准函数时的参数
# 它将传递快速/慢速阶段的速度和阈值给这些底层的校准函数
def navigate_to_exact_pose(logger, robot_ctrl, pose_monitor,
                           target_x, target_y, target_yaw_degrees,
                           primary_axis_to_align_first, # "X" 或 "Y"
                           xy_tolerance_m=0.03,
                           yaw_tolerance_degrees=3.0,
                           max_attempts_per_main_stage=20,
                           # --- 速度和阈值参数 ---
                           # Yaw对准参数
                           fast_angular_speed_dps=10.0,
                           slow_down_yaw_threshold_degrees=20.0,
                           slow_angular_speed_dps=3.0,
                           max_rotation_per_step_degrees=10.0, # Yaw调整时单步最大旋转
                           # 直线驱动对准轴参数
                           fast_linear_speed_mps=0.05,
                           slow_down_xy_threshold_m=0.1, # 对应align_axis_by_driving_forward
                           slow_linear_speed_mps=0.03,
                           # 侧向平移对准轴参数
                           fast_lateral_speed_mps=0.05,
                           slow_down_lateral_threshold_m=0.1, # 对应align_axis_by_strafing
                           slow_lateral_speed_mps=0.03,
                           max_lateral_move_step_m=0.03, # 侧向平移单步最大移动
                           stabilization_time_s=1.0):
    """
    【最终分阶段导航-指定优先轴-阶段速度】
    1. 校准优先选择的世界轴 (X或Y) 通过先旋转与轴平行然后前进/后退。
    2. 为侧向平移，重新对准最终目标偏航角。
    3. 校准另一个世界轴通过侧向平移。
    4. 最终微调朝向。
    所有调整都使用阶段性速度。
    """
    if primary_axis_to_align_first not in ["X", "Y"]:
        logger.error(f"无效的优先对准轴: {primary_axis_to_align_first}。")
        return False

    secondary_axis_to_align_laterally = "Y" if primary_axis_to_align_first == "X" else "X"
    target_coord_primary = target_x if primary_axis_to_align_first == "X" else target_y
    target_coord_secondary = target_y if primary_axis_to_align_first == "X" else target_x

    logger.info(f"=== 开始精确导航 (优先 {primary_axis_to_align_first}轴, 阶段速度): X={target_x:.3f}, Y={target_y:.3f}, Yaw={target_yaw_degrees:.2f}° ===")

    # 阶段1: 校准优先轴 (旋转+直行)
    logger.info(f"\n--- 导航阶段 1: 校准优先轴 {primary_axis_to_align_first} 到 {target_coord_primary:.3f}m ---")
    success_primary_axis = align_axis_by_driving_forward(
        logger, robot_ctrl, pose_monitor,
        target_coord_value=target_coord_primary,
        axis_to_align=primary_axis_to_align_first,
        tolerance_m=xy_tolerance_m,
        max_attempts=max_attempts_per_main_stage,
        fast_linear_speed_mps=fast_linear_speed_mps,
        slow_down_xy_threshold_m=slow_down_xy_threshold_m,
        slow_linear_speed_mps=slow_linear_speed_mps
    )
    logger.info(f"导航阶段 1 (校准 {primary_axis_to_align_first}轴) 完成。")
    time.sleep(stabilization_time_s)

    # 阶段3: 校准另一个世界轴 (通过侧向平移)
    logger.info(f"\n--- 导航阶段 2: 通过侧向平移校准 {secondary_axis_to_align_laterally}轴到 {target_coord_secondary:.3f}m ---")
    current_yaw_for_lateral_rad = pose_monitor.get_yaw_from_tf_rad() or pose_monitor.get_yaw_from_imu_rad()
    success_secondary_axis = align_axis_by_strafing(
        logger, robot_ctrl, pose_monitor,
        target_coord_value=target_coord_secondary,
        axis_to_adjust_laterally=secondary_axis_to_align_laterally,
        robot_current_yaw_rad=current_yaw_for_lateral_rad,
        tolerance_m=xy_tolerance_m,
        max_attempts=max_attempts_per_main_stage,
        fast_lateral_speed_mps=fast_lateral_speed_mps,
        slow_down_lateral_threshold_m=slow_down_lateral_threshold_m,
        slow_lateral_speed_mps=slow_lateral_speed_mps,
        max_lateral_move_step_m=max_lateral_move_step_m,
        stabilization_time_s=stabilization_time_s
    )
    if not success_secondary_axis:
        logger.error(f"导航失败：未能完成 {secondary_axis_to_align_laterally}轴的侧向校准。")
        return False
    logger.info(f"导航阶段 2 (校准 {secondary_axis_to_align_laterally}轴) 完成。")
    time.sleep(stabilization_time_s)

    # 阶段4: 最终微调目标偏航角
    logger.info(f"\n--- 导航阶段 3: 最终微调目标偏航角 ({target_yaw_degrees:.2f}°) ---")
    success_final_yaw = align_yaw_to_target(
        logger, robot_ctrl, pose_monitor,
        target_yaw_degrees=target_yaw_degrees,
        tolerance_degrees=yaw_tolerance_degrees,
        max_attempts=max_attempts_per_main_stage,
        fast_angular_speed_dps=slow_angular_speed_dps, # 最终微调使用慢速档的参数
        slow_down_yaw_threshold_degrees=slow_down_yaw_threshold_degrees, # 切换到更慢速的阈值更小
        slow_angular_speed_dps=slow_angular_speed_dps , # 最终微调使用更慢的角速度
        max_rotation_per_step_degrees=max_rotation_per_step_degrees,
        stabilization_time_s=stabilization_time_s
    )
    if not success_final_yaw:
        logger.warn("精确导航警告：最终偏航角微调未达最佳精度。")

    # 最终验证
    time.sleep(stabilization_time_s)
    final_coords_check = pose_monitor.get_coordinates_from_tf()
    final_yaw_rad_check = pose_monitor.get_yaw_from_tf_rad() or pose_monitor.get_yaw_from_imu_rad()

    if final_coords_check[0] is None or final_yaw_rad_check is None:
        logger.error("最终状态获取失败，无法确认导航结果。")
        return False

    final_x_error_check = abs(target_x - final_coords_check[0])
    final_y_error_check = abs(target_y - final_coords_check[1])
    final_yaw_error_deg_check = abs(math.degrees(_normalize_angle_for_basic_action(math.radians(target_yaw_degrees) - final_yaw_rad_check)))
    
    logger.info(f"导航任务结束。最终误差：X={final_x_error_check:.4f}m, Y={final_y_error_check:.4f}m, Yaw={final_yaw_error_deg_check:.2f}°")

    if final_x_error_check <= xy_tolerance_m and \
       final_y_error_check <= xy_tolerance_m and \
       final_yaw_error_deg_check <= yaw_tolerance_degrees:
        logger.info("=== 精确导航任务 (指定优先轴，阶段速度) 成功！ ===")
        return True
    else:
        logger.error("=== 精确导航任务 (指定优先轴，阶段速度) 未能达到所有精度要求。 ===")
        return False