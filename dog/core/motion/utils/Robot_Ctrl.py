import copy
import lcm
import threading
import time
import math
import toml # 请确保已安装此库: pip install toml
import os # 用于文件路径操作

# 使用你提供的确切导入路径
from motion.utils.robot_control_cmd_lcmt import robot_control_cmd_lcmt
from motion.utils.robot_control_response_lcmt import robot_control_response_lcmt
from motion.utils.file_send_lcmt import file_send_lcmt
from motion.utils.localization_lcmt import localization_lcmt
from motion.utils.file_recv_lcmt import file_recv_lcmt # 确保这个文件在你提供的路径下

class ConsoleLogger:
    # ... (保持 ConsoleLogger 不变) ...
    def _log(self, level, msg):
        print(f"[{level}] {time.strftime('%H:%M:%S', time.localtime())}: {msg}")
    def info(self, msg):
        self._log("INFO", msg)
    def warn(self, msg):
        self._log("WARN", msg)
    def error(self, msg):
        self._log("ERROR", msg)
    def debug(self, msg): # RobotCtrl 中可能用到debug日志
        self._log("DEBUG", msg)

class RobotCtrl:
    def __init__(self,
                 ros2_logger,
                 lcm_cmd_url="udpm://239.255.76.67:7671?ttl=255",
                 lcm_cmd_channel="robot_control_cmd",
                 lcm_response_url="udpm://239.255.76.67:7670?ttl=255",
                 lcm_response_channel="robot_control_response",
                 lcm_odom_url="udpm://239.255.76.67:7667?ttl=255",
                 lcm_odom_channel="global_to_robot",
                 cmd_heartbeat_hz=20.0,
                 enable_odom_lcm=True,
                 user_gait_file_channel="user_gait_file",       # 这个通道用于发送步态文件
                 user_gait_result_channel="user_gait_result"): # 这个通道用于接收步态文件处理结果

        self.logger = ros2_logger
        self._lcm_cmd_url = lcm_cmd_url
        self._lcm_cmd_channel = lcm_cmd_channel
        self._lcm_response_url = lcm_response_url
        self._lcm_response_channel = lcm_response_channel
        self._lcm_odom_url = lcm_odom_url
        self._lcm_odom_channel = lcm_odom_channel
        self._cmd_heartbeat_period = 1.0 / cmd_heartbeat_hz
        self._enable_odom_lcm = enable_odom_lcm
        self._user_gait_file_channel = user_gait_file_channel
        self._lcm_user_gait_result_channel = user_gait_result_channel

        self._lc_publisher = None # 用于发送命令和步态文件
        self._lc_response_subscriber = None
        self._lc_odom_subscriber = None
        self._lc_gait_file_ack_subscriber = None # 用于接收步态文件处理结果的ACK

        self._current_lcm_cmd = robot_control_cmd_lcmt()
        self._current_lcm_cmd.life_count = 0
        self._cmd_lock = threading.Lock()

        self._latest_robot_status = robot_control_response_lcmt()
        self._status_lock = threading.Lock()

        self._latest_odom_data = localization_lcmt() if self._enable_odom_lcm else None
        self._odom_lock = threading.Lock() if self._enable_odom_lcm else None

        self._action_done_event = threading.Event()
        self._expected_completion_mode = -1
        self._expected_completion_gait_id = -1

        self._gait_file_ack_event = threading.Event()
        self._gait_file_ack_success = False
        self._gait_file_ack_lock = threading.Lock()


        self._is_running = False
        self._cmd_send_thread = None
        self._response_receive_thread = None
        self._odom_receive_thread = None
        self._gait_file_ack_thread = None


        self.logger.info("RobotCtrl: 实例已创建。")

    def _ensure_int8(self, value):
        # ... (保持 _ensure_int8 不变) ...
        try:
            value = int(value)
            if value > 127: return 127
            if value < -128: return -128
            return value
        except (ValueError, TypeError):
            self.logger.warn(f"无法将值 '{value}' 转换为 int8_t, 使用0替代。")
            return 0

    def start(self):
        if self._is_running:
            self.logger.warn("RobotCtrl: 通信线程已在运行。")
            return True
        try:
            # _lc_publisher 用于发送 robot_control_cmd 和 user_gait_file
            # 官方示例中 lcm_cmd 和 lcm_usergait 使用了相同的 URL，所以一个 publisher 应该就够了
            # 如果通道不同，但URL相同，一个LCM实例可以publish到多个channel
            # 如果URL也不同，则需要为 self._user_gait_file_channel 单独创建LCM实例
            # 这里假设都用 self._lcm_cmd_url
            self._lc_publisher = lcm.LCM(self._lcm_cmd_url)
            self._lc_response_subscriber = lcm.LCM(self._lcm_response_url)
            if self._enable_odom_lcm:
                self._lc_odom_subscriber = lcm.LCM(self._lcm_odom_url)
            # Gait file ACK 订阅器，使用命令URL (或其专用URL)
            self._lc_gait_file_ack_subscriber = lcm.LCM(self._lcm_cmd_url) # 假设ACK也发到命令相关的LCM网络
            self.logger.info("RobotCtrl: LCM实例初始化成功。")
        except Exception as e:
            self.logger.error(f"RobotCtrl: 初始化LCM实例失败: {e}")
            return False

        self._is_running = True

        self._cmd_send_thread = threading.Thread(target=self._cmd_send_loop, name="LcmCmdSendLoop", daemon=True)
        self._cmd_send_thread.start()

        self._response_receive_thread = threading.Thread(target=self._threaded_lcm_handler_loop,
                                                         args=(self._lc_response_subscriber, self._lcm_response_channel, self._response_handler, "StateResponse"), daemon=True)
        self._response_receive_thread.start()

        if self._enable_odom_lcm and self._lc_odom_subscriber:
            self._odom_receive_thread = threading.Thread(target=self._threaded_lcm_handler_loop,
                                                         args=(self._lc_odom_subscriber, self._lcm_odom_channel, self._odom_handler, "Odom"), daemon=True)
            self._odom_receive_thread.start()

        if self._lc_gait_file_ack_subscriber:
            self._gait_file_ack_thread = threading.Thread(target=self._threaded_lcm_handler_loop,
                                                          args=(self._lc_gait_file_ack_subscriber, self._lcm_user_gait_result_channel, self._gait_file_ack_handler, "GaitFileAck"), daemon=True)
            self._gait_file_ack_thread.start()
        else:
            self.logger.warn("RobotCtrl: GaitFileAck LCM 订阅器未初始化，自定义步态文件ACK功能将不可用。")

        self.logger.info("RobotCtrl: 所有LCM通信线程已启动。")
        return True

    def stop(self):
        # ... (保持 stop 不变) ...
        if not self._is_running:
            self.logger.info("RobotCtrl: 通信线程已停止。")
            return
        self.logger.info("RobotCtrl: 正在停止通信线程...")
        self._is_running = False
        try:
            self.set_robot_to_pure_damper(wait=False)
            time.sleep(0.2)
        except Exception as e:
            self.logger.warn(f"RobotCtrl: 停止时发送 PureDamper 指令出错: {e}")

        if self._cmd_send_thread and self._cmd_send_thread.is_alive(): self._cmd_send_thread.join(timeout=1.0)
        if self._response_receive_thread and self._response_receive_thread.is_alive(): self._response_receive_thread.join(timeout=1.5)
        if self._odom_receive_thread and self._odom_receive_thread.is_alive(): self._odom_receive_thread.join(timeout=1.5)
        if self._gait_file_ack_thread and self._gait_file_ack_thread.is_alive(): self._gait_file_ack_thread.join(timeout=1.5)
        self.logger.info("RobotCtrl: 通信线程已停止。")


    def _get_next_master_life_count(self):
        # ... (保持 _get_next_master_life_count 不变) ...
        with self._cmd_lock:
            current_val = self._current_lcm_cmd.life_count
            next_val = current_val + 1
            if next_val > 127: next_val = -128
            self._current_lcm_cmd.life_count = next_val
            return next_val

    def _cmd_send_loop(self):
        # ... (保持 _cmd_send_loop 不变) ...
        self.logger.info("RobotCtrl: LCM指令发送循环已启动。")
        cmd_to_send_on_loop = robot_control_cmd_lcmt()
        while self._is_running:
            with self._cmd_lock:
                current_master_lc = self._current_lcm_cmd.life_count
                next_master_lc = current_master_lc + 1
                if next_master_lc > 127: next_master_lc = -128
                self._current_lcm_cmd.life_count = next_master_lc
                for field_name in robot_control_cmd_lcmt.__slots__:
                    setattr(cmd_to_send_on_loop, field_name, getattr(self._current_lcm_cmd, field_name))
            if self._lc_publisher:
                try:
                    payload = cmd_to_send_on_loop.encode()
                    self._lc_publisher.publish(self._lcm_cmd_channel, payload)
                except Exception as e:
                    self.logger.error(f"RobotCtrl: 在发送循环中发布指令时出错 (life_count: {cmd_to_send_on_loop.life_count}): {e}")
            time.sleep(self._cmd_heartbeat_period)
        self.logger.info("RobotCtrl: LCM指令发送循环已结束。")

    def _threaded_lcm_handler_loop(self, lcm_instance, channel, handler_method, name="LCMHandler"):
        # ... (保持 _threaded_lcm_handler_loop 不变) ...
        if not lcm_instance: self.logger.error(f"RobotCtrl: {name} 的 LCM 实例为 None。"); return
        subscription = None
        self.logger.info(f"RobotCtrl: {name} 接收循环启动，通道 '{channel}'。")
        try:
            subscription = lcm_instance.subscribe(channel, handler_method)
            self.logger.info(f"RobotCtrl: 已订阅 LCM 通道: {channel} (用于 {name})")
            while self._is_running: lcm_instance.handle_timeout(100)
        except Exception as e:
            if self._is_running: self.logger.error(f"RobotCtrl: {name} LCM 接收循环出错，通道 '{channel}': {e}")
        finally:
            if subscription and lcm_instance:
                try: lcm_instance.unsubscribe(subscription)
                except Exception as e_unsub: self.logger.error(f"RobotCtrl: 取消订阅 {name} 出错，通道 '{channel}': {e_unsub}")
            self.logger.info(f"RobotCtrl: {name} 接收循环结束，通道 '{channel}'。")

    def _gait_file_ack_handler(self, channel, data):
        # ... (保持 _gait_file_ack_handler 不变) ...
        try:
            msg = file_recv_lcmt.decode(data) # file_recv_lcmt 来自 utils
            with self._gait_file_ack_lock:
                if msg.result == 0: self._gait_file_ack_success = True; self.logger.info(f"机器人已成功处理自定义步态文件 (result={msg.result})。")
                else: self._gait_file_ack_success = False; self.logger.error(f"机器人处理自定义步态文件失败 (result={msg.result})。")
            self._gait_file_ack_event.set()
        except Exception as e:
            self.logger.error(f"处理自定义步态文件ACK时出错: {e}")
            with self._gait_file_ack_lock: self._gait_file_ack_success = False
            self._gait_file_ack_event.set()


    def _response_handler(self, channel, data):
        try:
            msg = robot_control_response_lcmt.decode(data)
            with self._status_lock: self._latest_robot_status = copy.deepcopy(msg)
            if self._expected_completion_mode != -1 and \
               msg.mode == self._expected_completion_mode and \
               msg.gait_id == self._expected_completion_gait_id and \
               msg.order_process_bar >= 95: # 官方示例中进度条似乎用百分比，这里假设95%算完成
                self.logger.info(f"RobotCtrl: 动作 {self._expected_completion_mode}/{self._expected_completion_gait_id} 完成 (进度 {msg.order_process_bar}%).")
                self._action_done_event.set()
                self._expected_completion_mode = -1; self._expected_completion_gait_id = -1
        except Exception as e: self.logger.error(f"RobotCtrl: 处理LCM状态响应失败，通道 '{channel}': {e}")


    def _odom_handler(self, channel, data):
        # ... (保持 _odom_handler 不变) ...
        if not self._enable_odom_lcm: return
        try:
            msg = localization_lcmt.decode(data)
            with self._odom_lock: self._latest_odom_data = copy.deepcopy(msg)
        except Exception as e: self.logger.error(f"RobotCtrl: 处理LCM里程计数据失败，通道 '{channel}': {e}")

    def set_velocity_command(self, linear_x, linear_y, angular_z, body_height=0.28, gait_id=26, pitch=0.0):
        with self._cmd_lock:
            self._current_lcm_cmd.mode = self._ensure_int8(11)
            self._current_lcm_cmd.gait_id = self._ensure_int8(gait_id)
            self._current_lcm_cmd.vel_des = [float(linear_x), float(linear_y), float(angular_z)]
            self._current_lcm_cmd.pos_des = [0.0, 0.0, float(body_height)]
            self._current_lcm_cmd.rpy_des = [0.0, pitch, 0.0]
            self._current_lcm_cmd.duration = 0
            self._current_lcm_cmd.step_height = [0.05, 0.05] # 默认的步高，可以根据需要调整或作为参数

    def execute_discrete_action(self, mode, gait_id,
                                rpy_des=None, pos_des=None, duration_ms=0,
                                step_height=None, contact=None, value=None,
                                wait_for_completion=False, wait_timeout_sec=10.0):
        self.logger.info(f"RobotCtrl: 请求执行离散动作: mode={mode}, gait_id={gait_id}, duration={duration_ms}ms, 等待={wait_for_completion}")
        with self._cmd_lock:
            self._current_lcm_cmd.mode = self._ensure_int8(mode)
            self._current_lcm_cmd.gait_id = self._ensure_int8(gait_id)
            self._current_lcm_cmd.duration = int(duration_ms)
            self._current_lcm_cmd.vel_des = [0.0, 0.0, 0.0] # 离散动作通常不直接设速度
            self._current_lcm_cmd.rpy_des = list(rpy_des) if rpy_des is not None else [0.0, 0.0, 0.0]
            # 保留当前身体高度，除非在pos_des中显式指定或模式为7
            current_body_height = self._current_lcm_cmd.pos_des[2] if len(self._current_lcm_cmd.pos_des) == 3 else 0.28
            if pos_des is not None: self._current_lcm_cmd.pos_des = list(pos_des)
            else: self._current_lcm_cmd.pos_des = [0.0, 0.0, current_body_height if mode != 7 else 0.0] # Mode 7 (PureDamper) 通常身体高度为0

            self._current_lcm_cmd.step_height = list(step_height) if step_height is not None else [0.0, 0.0] # 根据需要设置
            if contact is not None: self._current_lcm_cmd.contact = self._ensure_int8(contact)
            if value is not None: self._current_lcm_cmd.value = int(value)

        if wait_for_completion:
            self.logger.info(f"RobotCtrl: 等待动作 mode={mode}, gait_id={gait_id} 完成...")
            self._action_done_event.clear()
            self._expected_completion_mode = int(mode)
            self._expected_completion_gait_id = int(gait_id)
            completed = self._action_done_event.wait(timeout=wait_timeout_sec)
            self._expected_completion_mode = -1 # 重置期望值
            self._expected_completion_gait_id = -1
            if completed:
                self.logger.info(f"RobotCtrl: 动作 mode={mode}, gait_id={gait_id} 已确认完成。")
                return True
            else:
                self.logger.warn(f"RobotCtrl: 等待动作 mode={mode}, gait_id={gait_id} 超时。")
                return False
        return True


    def set_robot_to_pure_damper(self, wait=True, timeout=5.0):
        return self.execute_discrete_action(mode=7, gait_id=0, pos_des=[0.0,0.0,0.0], duration_ms=0, wait_for_completion=wait, wait_timeout_sec=timeout)


    def get_latest_status_copy(self):
        with self._status_lock: return copy.deepcopy(self._latest_robot_status)

    def get_latest_odom_data_copy(self):
        # ... (保持 get_latest_odom_data_copy 不变) ...
        if not self._enable_odom_lcm or not self._odom_lock: return None
        with self._odom_lock: return copy.deepcopy(self._latest_odom_data) if self._latest_odom_data else None

    # --- 新增的自定义步态加载和执行方法 ---
    def load_and_execute_custom_gait(self,
                                     gait_params_file_path: str,
                                     gait_def_file_path: str,
                                     user_gait_list_file_path: str,
                                     base_working_dir: str = ".",
                                     wait_for_file_ack_timeout_sec: float = 5.0):
        """
        加载、发送并执行自定义步态。
        1. 处理 gait_params_file，生成 full_params 文件。
        2. 发送 gait_def_file。
        3. 发送 full_params 文件。
        4. 执行 user_gait_list_file 中的步态序列。

        :param gait_params_file_path: 原始步态参数文件路径 (例如 Gait_Params_moonwalk.toml)
        :param gait_def_file_path: 步态定义文件路径 (例如 Gait_Def_moonwalk.toml)
        :param user_gait_list_file_path: 用户步态序列文件路径 (例如 Usergait_List.toml)
        :param base_working_dir: 用于存放生成的 "full" 参数文件的目录。
        :param wait_for_file_ack_timeout_sec: 等待步态文件发送ACK的超时时间。
        :param action_wait_timeout_sec: 等待每个步态动作完成的超时时间。
        :return: 如果所有步骤成功则返回 True，否则返回 False。
        """
        if not self._is_running or not self._lc_publisher:
            self.logger.error("RobotCtrl: LCM 未运行或发布器未初始化，无法加载自定义步态。")
            return False

        self.logger.info(f"开始加载和执行自定义步态: 定义='{gait_def_file_path}', 参数='{gait_params_file_path}', 列表='{user_gait_list_file_path}'")

        # 准备一个空的 robot_cmd 字典模板，用于填充 full_steps
        robot_cmd_template = {
            'mode':0, 'gait_id':0, 'contact':0, 'life_count':0, # life_count 会在发送时更新
            'vel_des':[0.0, 0.0, 0.0], 'rpy_des':[0.0, 0.0, 0.0], 'pos_des':[0.0, 0.0, 0.0],
            'acc_des':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'ctrl_point':[0.0, 0.0, 0.0],
            'foot_pose':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'step_height':[0.0, 0.0],
            'value':0,  'duration':0
        }
        generated_full_params_file_path = os.path.join(base_working_dir, os.path.basename(gait_params_file_path).replace(".toml", "_full.toml"))

        try:
            # 1. 处理 gait_params_file，生成 full_params 文件
            self.logger.info(f"正在处理步态参数文件: {gait_params_file_path}")
            original_params = toml.load(gait_params_file_path)
            full_steps_data = {'step': []}

            for i_param in original_params.get('step', []):
                cmd_entry = copy.deepcopy(robot_cmd_template)
                cmd_entry['duration'] = i_param.get('duration',0)
                if i_param.get('type') == 'usergait':
                    cmd_entry['mode'] = 11  # LOCOMOTION
                    cmd_entry['gait_id'] = 110 # USERGAIT (根据示例，这个值固定为110)
                    cmd_entry['vel_des'] = i_param.get('body_vel_des', [0.0,0.0,0.0])
                    body_pos_des = i_param.get('body_pos_des', [0.0]*6)
                    cmd_entry['rpy_des'] = body_pos_des[0:3]
                    cmd_entry['pos_des'] = body_pos_des[3:6]

                    landing_pos_des = i_param.get('landing_pos_des', [0.0]*11)
                    cmd_entry['foot_pose'][0:2] = landing_pos_des[0:2]
                    cmd_entry['foot_pose'][2:4] = landing_pos_des[3:5]
                    cmd_entry['foot_pose'][4:6] = landing_pos_des[6:8]
                    cmd_entry['ctrl_point'][0:2] = landing_pos_des[9:11]

                    step_h = i_param.get('step_height', [0.0]*4)
                    cmd_entry['step_height'][0] = math.ceil(step_h[0] * 1e3) + math.ceil(step_h[1] * 1e3) * 1e3
                    cmd_entry['step_height'][1] = math.ceil(step_h[2] * 1e3) + math.ceil(step_h[3] * 1e3) * 1e3

                    cmd_entry['acc_des'] = i_param.get('weight', [0.0]*6) # weight 对应 acc_des
                    cmd_entry['value'] = i_param.get('use_mpc_traj', 0)
                    cmd_entry['contact'] = math.floor(i_param.get('landing_gain', 0.0) * 1e1)
                    cmd_entry['ctrl_point'][2] = i_param.get('mu', 0.0)
                else:
                    # 如果有其他类型的参数，这里可以添加处理逻辑或警告
                    self.logger.warn(f"在 {gait_params_file_path} 中遇到未知类型 '{i_param.get('type')}' 的参数，已跳过。")
                    continue
                full_steps_data['step'].append(cmd_entry)

            with open(generated_full_params_file_path, 'w') as f:
                f.write(f"# Gait Params (Full - Generated by RobotCtrl for {os.path.basename(gait_params_file_path)})\n")
                toml.dump(full_steps_data, f)
            self.logger.info(f"已生成 full 步态参数文件: {generated_full_params_file_path}")

        except FileNotFoundError:
            self.logger.error(f"错误: 步态参数文件 '{gait_params_file_path}' 未找到。")
            return False
        except Exception as e:
            self.logger.error(f"处理步态参数文件 '{gait_params_file_path}' 时出错: {e}")
            return False

        usergait_msg = file_send_lcmt()

        # 2. 发送 gait_def_file
        try:
            self.logger.info(f"正在发送步态定义文件: {gait_def_file_path}")
            with open(gait_def_file_path, 'r') as f_def:
                usergait_msg.data = f_def.read()

            self._gait_file_ack_event.clear()
            self._gait_file_ack_success = False
            self._lc_publisher.publish(self._user_gait_file_channel, usergait_msg.encode())
            self.logger.info(f"已发布步态定义文件到通道 '{self._user_gait_file_channel}'。等待ACK...")

            if self._gait_file_ack_event.wait(timeout=wait_for_file_ack_timeout_sec):
                if self._gait_file_ack_success:
                    self.logger.info("步态定义文件发送成功并收到ACK。")
                else:
                    self.logger.error("机器人处理步态定义文件失败 (收到NACK)。")
                    return False
            else:
                self.logger.warn("等待步态定义文件ACK超时。")
                # 根据实际需求，这里可以选择返回False或继续（如果ACK不是严格必需的）
                return False # 严格要求ACK

        except FileNotFoundError:
            self.logger.error(f"错误: 步态定义文件 '{gait_def_file_path}' 未找到。")
            return False
        except Exception as e:
            self.logger.error(f"发送步态定义文件 '{gait_def_file_path}' 时出错: {e}")
            return False

        time.sleep(0.5) # 根据官方示例，在两次文件发送之间有延时

        # 3. 发送 generated_full_params_file
        try:
            self.logger.info(f"正在发送 full 步态参数文件: {generated_full_params_file_path}")
            with open(generated_full_params_file_path, 'r') as f_params:
                usergait_msg.data = f_params.read()

            self._gait_file_ack_event.clear()
            self._gait_file_ack_success = False
            self._lc_publisher.publish(self._user_gait_file_channel, usergait_msg.encode())
            self.logger.info(f"已发布 full 步态参数文件到通道 '{self._user_gait_file_channel}'。等待ACK...")

            if self._gait_file_ack_event.wait(timeout=wait_for_file_ack_timeout_sec):
                if self._gait_file_ack_success:
                    self.logger.info("Full 步态参数文件发送成功并收到ACK。")
                else:
                    self.logger.error("机器人处理 Full 步态参数文件失败 (收到NACK)。")
                    return False
            else:
                self.logger.warn("等待 Full 步态参数文件ACK超时。")
                return False # 严格要求ACK
        except FileNotFoundError: # 理论上此文件是刚生成的，不应找不到
            self.logger.error(f"错误: 生成的 full 步态参数文件 '{generated_full_params_file_path}' 未找到。")
            return False
        except Exception as e:
            self.logger.error(f"发送 full 步态参数文件 '{generated_full_params_file_path}' 时出错: {e}")
            return False

        time.sleep(0.1) # 官方示例延时

        # 4. 执行 user_gait_list_file 中的步态序列
        try:
            self.logger.info(f"正在执行用户步态列表: {user_gait_list_file_path}")
            gait_sequence = toml.load(user_gait_list_file_path)
            for step_idx, step_cmd_data in enumerate(gait_sequence.get('step', [])):
                mode = step_cmd_data.get('mode')
                gait_id = step_cmd_data.get('gait_id')
                duration_ms = step_cmd_data.get('duration', 0)

                if mode is None or gait_id is None:
                    self.logger.warn(f"跳过步骤 {step_idx + 1} (在 {user_gait_list_file_path} 中): mode 或 gait_id 未定义。")
                    continue

                self.logger.info(f"执行步态序列中的步骤 {step_idx + 1}: mode={mode}, gait_id={gait_id}, duration={duration_ms}ms")

                # 根据 gait_id 区分处理
                if gait_id == 110 :
                    self.logger.info(f"  检测到自定义步态 (gait_id=110)。直接设置命令。")
                    with self._cmd_lock:
                        self._current_lcm_cmd.mode = self._ensure_int8(mode)
                        self._current_lcm_cmd.gait_id = self._ensure_int8(gait_id) # 应为 110
                        self._current_lcm_cmd.duration = int(duration_ms)
                        # 从 step_cmd_data 加载可能覆盖或指定的参数
                        if 'vel_des' in step_cmd_data: self._current_lcm_cmd.vel_des = list(step_cmd_data['vel_des'])
                        if 'rpy_des' in step_cmd_data: self._current_lcm_cmd.rpy_des = list(step_cmd_data['rpy_des'])
                        else: self._current_lcm_cmd.rpy_des = [0.0,0.0,0.0] # 自定义步态通常有自己的姿态
                        if 'pos_des' in step_cmd_data: self._current_lcm_cmd.pos_des = list(step_cmd_data['pos_des'])
                        else: # 维持或使用full_params中定义的身体高度
                            current_body_height = self._current_lcm_cmd.pos_des[2] if len(self._current_lcm_cmd.pos_des) == 3 else 0.28
                            self._current_lcm_cmd.pos_des = [0.0, 0.0, current_body_height]

                        if 'step_height' in step_cmd_data: self._current_lcm_cmd.step_height = list(step_cmd_data['step_height'])
                        # contact 和 value 通常在自定义步态的 full_params 文件中定义，但也可以在这里覆盖
                        if 'contact' in step_cmd_data: self._current_lcm_cmd.contact = self._ensure_int8(step_cmd_data['contact'])
                        if 'value' in step_cmd_data: self._current_lcm_cmd.value = int(step_cmd_data['value'])
                        # acc_des, ctrl_point, foot_pose 通常在 full_params 文件中详细定义
                        # 如果 Usergait_List.toml 要覆盖它们，也需要在这里添加赋值逻辑
                        if 'acc_des' in step_cmd_data: self._current_lcm_cmd.acc_des = list(step_cmd_data['acc_des'])
                        if 'ctrl_point' in step_cmd_data: self._current_lcm_cmd.ctrl_point = list(step_cmd_data['ctrl_point'])
                        if 'foot_pose' in step_cmd_data: self._current_lcm_cmd.foot_pose = list(step_cmd_data['foot_pose'])

                    self._get_next_master_life_count() # 更新 life_count，确保下一条是新指令
                    self.logger.info(f"    已设置自定义步态指令 (mode={self._current_lcm_cmd.mode}, gait_id={self._current_lcm_cmd.gait_id}, duration={self._current_lcm_cmd.duration})。等待其定义的持续时间。")
                    if duration_ms > 0:
                        time.sleep(duration_ms / 1000.0)
                    else:
                        time.sleep(0.1) # 默认最小延时
                else:
                    # 其他 gait_id，视为预定义动作，使用 execute_discrete_action
                    self.logger.info(f"  检测到预定义动作 (mode={mode}, gait_id={gait_id})。使用 execute_discrete_action。")
                    success = self.execute_discrete_action(mode, gait_id, 
                                          wait_for_completion=True)
                    if success:
                        self.logger.info("自定义动作完成。")
                        time.sleep(1.0) # 站稳后稳定
                    else:
                        self.logger.warn("自定义动作失败或超时。")
            self.logger.info("自定义步态序列中的所有指令已处理。")
            return True

        except FileNotFoundError:
            self.logger.error(f"错误: 用户步态列表文件 '{user_gait_list_file_path}' 未找到。")
            return False
        except Exception as e:
            self.logger.error(f"执行用户步态列表 '{user_gait_list_file_path}' 时出错: {e}")
            return False