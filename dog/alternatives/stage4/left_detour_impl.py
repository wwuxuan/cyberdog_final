#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第四赛段 真机版（完整流程，从第四赛段起点单跑）
障碍物绕行方向：本文件 DETOUR_SIDE=-1，从障碍物左侧绕；从右侧绕的版本见 stage4_real_right.py。

坐标系说明（重要）：
  - 第四赛段对外只使用绝对坐标：x=左下角蓝球原点向右的横向距离，y=距赛段入口向前的距离。
  - RealDogAdapter 的局部坐标仅用于执行运动：局部 +x 沿赛道 +y，局部 +y 指向赛道左侧（即全局 x 减小）。

运动：复用队友 RealDogAdapter（LCM 20Hz 心跳 + global_to_robot 里程计）。
低姿（限高杆）：旧狗自定义步态 mode=62/gait=110（gait/ 目录上传）。
视觉：电脑 YOLO（live_detect_server.py --push-ip <狗IP>）→ 本机 RemoteDetector(9890)。
语音：电脑端 live_detect_server 置信度+面积触发 → 狗端 speak_on_detect.py(9888) 播报。

运行（电脑识别服务 + 狗上主程序）：
  狗①: source /etc/mi/ros2_env.conf
       python3 /home/mi/cyberdog_competition/stage4_real_left.py --start-at-corridor
       （主程序默认自动启动 RGB 推流）
  电脑: python live_detect_server.py --dog-ip <狗IP> --push-ip <狗IP> \
        --targets cola,football,orange_ball,obstacle,limit_bar
  语音(可选): 狗③ python3 /home/mi/stage4/vision/speak_on_detect.py
"""
import argparse
import math
import os
import subprocess
import sys
import threading
import time

CORE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core"))
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from adapter import RealDogAdapter
from remote import RemoteDetector
from low_gait import LowGait
from s4_line import LineCalib, TARGET_LINE


# ================= CONFIG =================
# 第四赛段坐标：x/y 都是赛道平面图的绝对坐标。
# adapter.set_origin() 后立即把当前位姿映射为场地绝对坐标，后续 fwd 即绝对 y。
STAGE4_START_X = 3.200
STAGE4_FINISH_X = 0.325
STAGE4_ENTRY_Y = 6.580
RIGHT_START_LINE_TARGET = 0.30
RIGHT_FINISH_LINE_TARGET = 0.25
CORRIDOR_LINE_TARGET = TARGET_LINE
THIRD_CHANNEL_LEFT_LINE_TARGET = 0.50

# 按赛道平面图的绝对 x：第1通道在右侧，第3通道在左侧。
CHANNEL_X = [2.200, 1.100, 0.000]
CORRIDOR_Y = 7.200
DETECT_CEILING_Y = 11.020
DETECT_FLOOR_Y = 7.500
BB_UP_WP1_Y = 8.279
BB_UP_WP2_Y = 8.980
BB_UP_WP3_Y = 9.190
BB_DN_START_Y = 9.280
BB_DN_WP1_Y = 8.770
BB_DN_WP2_Y = 8.279
BB_DN_WP3_Y = 7.820

# 返回段先盲走到这里，再启用右鱼眼；右鱼眼循线走到终点校准前的位置。
RETURN_LINE_START_X = 1.00
FINISH_APPROACH_X = 2.70

FWD_YAW = 0.0                        # 面向通道前进方向（= 适配器 yaw 0）
RIGHT_YAW = -90.0                    # 面向右侧（适配器 -y 方向）

WALK_VX = 0.25                       # 正常前进速度（真机用，初赛 0.9 太快）
LOW_VX = 0.14                        # 低姿前进速度
TURN_TOL = 4.0

BBOX_THRESHOLD = 15000.0             # 距离判据(px²)——实机相机分辨率不同需重新标定！
LIMIT_BAR_AREA_NORM = 0.65           # 限高杆触发低姿和播报的占画面比例；可 --bar-norm 调
OBSTACLE1_CENTER_TOL = 0.18
EXTRA_SECS_BALL = 1.5
EXTRA_SECS_BAR = 13.8                # (不再用于限高杆，保留兼容)
LOW_DIST_M = 1.2                      # 限高杆低姿行走距离（米），走完恢复正常步态
FOOTBALL_TRIGGER_NORM = 0.07    # 足球占画面比例≥此值才触发'识别到足球'+推球动作
COLA_TRIGGER_NORM = 0.08      # 可乐瓶占画面比例>此值才触发'识别到可乐瓶'
ORANGE_TRIGGER_NORM = 0.05   # 橙色小球占画面比例>此值才触发'识别到橙色小球'
OBSTACLE2_WALK_M = 1.3       # 障碍2（可乐/橙球）：对中后正常前走距离（米），不超上行上限；与足球低姿推球(1.3)一致
FOOTBALL_PUSH_M = 1.3        # 足球：对中后低姿推球距离（米）
LIMIT_BAR_DOWN_LEAD = 0.1    # 下行：杆前多远(上方)切低姿
LIMIT_BAR_DOWN_CLEAR = 0.0   # 下行：过杆多远后站起

OBSTACLE1 = ["limit_bar", "obstacle"]          # 初赛 blue_block -> 决赛 obstacle
OBSTACLE2 = ["orange_ball", "cola", "football"]  # 初赛 blue_ball -> 决赛 orange_ball

LANE_K = 0.5
LANE_MAX_VY = 0.10
LANE_K_YAW = 0.03
LANE_MAX_WZ = 0.15

BB_DX_OUT = 0.9                    # 绕障碍时横向斜出偏移（绝对 x）
DETOUR_SIDE = -1                       # 障碍绕行方向：+1=从右侧绕；-1=从左侧绕（本文件为左侧绕版）

WP_TOL = 0.08
WP_MAX_V = 0.25
WP_K = 1.5
WP_TIMEOUT = 40.0

STALE_WAIT_TIMEOUT = 10.0            # 电脑断连时最多等多久（秒），超时终止
ODOM_STALE_TIMEOUT = 3.0             # 里程计多久没更新视为异常
NO_PROGRESS_TIMEOUT = 6.0            # 行走阶段多少秒位置没变化则异常
SPEECH_MUTE_FILE = "/tmp/stage4_speech_mute"   # 语音播报静音标记：上行写 up，下行写 down，speak_on_detect 读它
DEFAULT_PC_HOST = "192.168.43.102"
DEFAULT_STREAM_PORT = 9891
DEFAULT_STREAM_SCRIPT = "/home/mi/stage4/vision/stream_images.py"


class Stage4Real(object):
    def __init__(self, adapter, det, low, line=None, cfg=None):
        self.adp = adapter
        self.det = det
        self.low = low
        self.line = line
        self.cfg = cfg or {}
        self.bar_norm = float(self.cfg.get("bar_norm", LIMIT_BAR_AREA_NORM))
        self.low_active = False
        self._pending_obstacle2 = None
        self._obstacle1_detect_fwd = None   # 上行检测到障碍物时的 fwd（下行绕行最后回中心线用）
        self._last_odom = None
        self._last_odom_t = 0.0

    # ---------------- 基础工具 ----------------
    def _pos(self):          # 适配器坐标 (x=前进, y=左)
        return self.adp.get_position()

    def _stage_y(self):
        return self._pos()[0]

    def _fwd(self):          # 兼容旧调用：现在返回第四赛段绝对 y
        return self._stage_y()

    def _stage_x(self):
        return STAGE4_START_X - self._pos()[1]

    def _lat(self):          # 兼容旧调用：现在返回第四赛段绝对 x
        return self._stage_x()

    def _yaw(self):
        return self.adp.get_yaw_deg()

    def _walk(self, vx, vy=0.0, wz=0.0):   # vx=前, vy=左, wz=转
        self.adp.walk(vx, vy, wz)

    def _stop(self):
        self.adp.stop()

    def _set_speech_mute(self, state):
        """上行='up'(可播报) / 下行='down'(静音)：写入 SPEECH_MUTE_FILE 供 speak_on_detect 读取"""
        try:
            with open(SPEECH_MUTE_FILE, "w") as f:
                f.write(state)
        except Exception:
            pass

    def _check_odom(self, label):
        pos = self._pos()
        now = time.time()
        if self._last_odom is not None:
            d = math.hypot(pos[0] - self._last_odom[0], pos[1] - self._last_odom[1])
            if d < 1e-6 and now - self._last_odom_t > ODOM_STALE_TIMEOUT:
                raise RuntimeError("[%s] 里程计长时间无更新 pos=%s" %
                                   (label, (round(pos[0], 3), round(pos[1], 3))))
        self._last_odom = pos
        self._last_odom_t = now

    def _wait_detector(self, timeout=STALE_WAIT_TIMEOUT):
        t0 = time.time()
        while not self.det.alive():
            if time.time() - t0 > timeout:
                raise RuntimeError("[stage4] 电脑识别链路断连超过 %.0fs，终止" % timeout)
            print("[stage4] 等待电脑识别链路 ...")
            self._stop()
            time.sleep(1.0)
    # ---------------- 黄线校准 + 循线（换通道 / 回程） ----------------
    def line_ready(self):
        return self.line is not None and getattr(self.line, "ready", False)

    def line_calibrate_here(self, side="left"):
        """原地（面朝 +fwd）：轠90°（线到身侧）→ 线校准 → 转回 +fwd"""
        turn_yaw = 90.0 if side == "left" else -90.0
        if not self.adp.align_yaw(turn_yaw, tol=2.0, timeout=8.0):
            print("[stage4] 转向 %+.0f° 失败" % turn_yaw)
            return False
        self.line.calibrate_fwd(
            self.adp,
            side=side,
            target_dist=CORRIDOR_LINE_TARGET,
            mapped_fwd=CORRIDOR_Y,
        )
        self.adp.align_yaw(FWD_YAW, tol=2.0, timeout=8.0)
        return True

    def line_calibrate_stage_x(self, target_dist, target_x):
        """用右鱼眼校准绝对 x，并把里程计 x 写回目标坐标。"""
        if not self.line.end_calibrate(self.adp, side="right", target_dist=target_dist):
            return False
        pos_x, _pos_y, _z = self._pos()
        self.adp.set_mapped_pose(pos_x, STAGE4_START_X - target_x, self._yaw())
        print("[stage4] 右鱼眼 x 校准完成：线距=%.2fm，绝对 x=%.3f" %
              (target_dist, target_x), flush=True)
        return True

    def line_transition(self, side, stop_x, target_yaw, pre_walk_m=0.0):
        """转向后用黄线校准 y，再循线走到绝对 x。"""
        if not self.adp.align_yaw(target_yaw, tol=2.0, timeout=8.0):
            print("[stage4] 转向 %+.0f° 失败" % target_yaw)
            return False
        if pre_walk_m > 0:
            print("[stage4] 转向后先直走 %.2fm ..." % pre_walk_m)
            self._walk_forward_dist(pre_walk_m)
        self.line.calibrate_fwd(
            self.adp,
            side=side,
            target_dist=CORRIDOR_LINE_TARGET,
            mapped_fwd=CORRIDOR_Y,
        )
        ok = self.line.line_follow(
            self.adp,
            side=side,
            stop_x=stop_x,
            x_origin=STAGE4_START_X,
            target_yaw=target_yaw,
        )
        self.adp.align_yaw(FWD_YAW, tol=2.0, timeout=8.0)
        return ok

    def calibrate_third_channel_return(self):
        """在第三通道回到走廊后用左鱼眼校边线，不改写里程计坐标。"""
        print("[stage4] 第三通道回到 +y：左鱼眼校准左线距=%.2fm（不重写坐标）" %
              THIRD_CHANNEL_LEFT_LINE_TARGET, flush=True)
        if not self.adp.align_yaw(FWD_YAW, tol=2.0, timeout=8.0):
            print("[stage4] 第三通道回到 +y 对准失败，跳过左线校准", flush=True)
            return False
        ok = self.line.end_calibrate(
            self.adp,
            side="left",
            target_dist=THIRD_CHANNEL_LEFT_LINE_TARGET,
        )
        print("[stage4] 第三通道左线校准 %s：坐标保持 x=%.3f y=%.3f yaw=%.1f" %
              ("完成" if ok else "未达标",
               self._stage_x(), self._stage_y(), self._yaw()), flush=True)
        return ok

    def _phase_i_odom(self, target_x=FINISH_APPROACH_X):
        """面向全局 +x，按里程计走到指定绝对 x。"""
        print("[stage4] 返回段面向 +x，先走到 x=%.3f（暂不启用右鱼眼）" % target_x)
        self.turn_to_yaw(RIGHT_YAW, tol=TURN_TOL)
        t0 = time.time()
        while True:
            if time.time() - t0 > 30.0:
                raise RuntimeError("phase I 超时")
            self._check_odom("phaseI")
            stage_x = self._stage_x()
            if stage_x >= target_x:
                print("[stage4] 到达 x=%.3f" % stage_x)
                break
            f = self._fwd()
            vy = LANE_K * (CORRIDOR_Y - f)      # 保持走廊 y
            vy = max(-LANE_MAX_VY, min(LANE_MAX_VY, vy))
            yaw_err = ((RIGHT_YAW - self._yaw()) + 180.0) % 360.0 - 180.0
            wz = LANE_K_YAW * yaw_err
            wz = max(-LANE_MAX_WZ, min(LANE_MAX_WZ, wz))
            self._walk(WALK_VX, vy, wz)
            time.sleep(0.07)
        self._stop()


    def _lane(self, target_x):
        """面向 +y 时保持绝对 x=target_x。"""
        yaw = self._yaw()
        x_err = target_x - self._stage_x()
        vy = -LANE_K * x_err
        vy = max(-LANE_MAX_VY, min(LANE_MAX_VY, vy))
        yaw_err = ((FWD_YAW - yaw) + 180.0) % 360.0 - 180.0
        wz = LANE_K_YAW * yaw_err
        wz = max(-LANE_MAX_WZ, min(LANE_MAX_WZ, wz))
        return vy, wz

    # ---------------- 运动原语（全部用适配器坐标计算） ----------------
    def turn_to_yaw(self, target, tol=TURN_TOL, timeout=8.0):
        return self.adp.align_yaw(target, tol=tol, timeout=timeout)

    def walk_to_y(self, target_y, vx=WALK_VX, lock_x=None, timeout=60.0):
        """沿前进方向走到绝对 y=target_y。"""
        going_up = vx > 0
        if going_up:
            target_y = min(target_y, DETECT_CEILING_Y)
        t0 = time.time()
        last_progress = [self._fwd(), time.time()]
        while True:
            if time.time() - t0 > timeout:
                raise RuntimeError("walk_to_y(y=%s) 超时" % target_y)
            self._check_odom("walk_to_y")
            f = self._fwd()
            if going_up and f >= target_y:
                break
            if not going_up and f <= target_y:
                break
            if lock_x is not None:
                vy, wz = self._lane(lock_x)
            else:
                vy, wz = 0.0, 0.0
            if abs(f - last_progress[0]) < 0.01 and time.time() - last_progress[1] > NO_PROGRESS_TIMEOUT:
                raise RuntimeError("walk_to_y 无位移 fwd=%.3f" % f)
            if abs(f - last_progress[0]) >= 0.01:
                last_progress = [f, time.time()]
            self._walk(vx, vy, wz)
            time.sleep(0.07)
        self._stop()

    def walk_to_x(self, target_x, vx=WALK_VX, going_left=True, timeout=30.0):
        """横向走到绝对 x=target_x。"""
        t0 = time.time()
        while True:
            if time.time() - t0 > timeout:
                raise RuntimeError("walk_to_x(x=%s) 超时" % target_x)
            self._check_odom("walk_to_x")
            stage_x = self._stage_x()
            if going_left and stage_x <= target_x:
                break
            if not going_left and stage_x >= target_x:
                break
            spd = vx if abs(stage_x - target_x) > 0.3 else 0.14
            self._walk(spd, 0.0, 0.0)
            time.sleep(0.07)
        self._stop()

    def walk_to_xy(self, target_y, target_x, target_yaw=None,
                   max_v=WP_MAX_V, timeout=WP_TIMEOUT):
        """斜走到第四赛段绝对坐标 (x=target_x, y=target_y)。"""
        if target_yaw is None:
            target_yaw = FWD_YAW
        target_y = min(target_y, DETECT_CEILING_Y)
        t0 = time.time()
        last_log = 0.0
        while True:
            if time.time() - t0 > timeout:
                raise RuntimeError("walk_to_xy(x=%s,y=%s) 超时" % (target_x, target_y))
            self._check_odom("walk_to_xy")
            x_fwd = self._fwd()
            yaw = self._yaw()
            df = target_y - x_fwd
            dl = target_x - self._stage_x()
            dist = math.hypot(df, dl)
            if dist <= WP_TOL:
                break
            yaw_rad = math.radians(yaw)
            body_vx = df * math.cos(yaw_rad) - dl * math.sin(yaw_rad)
            body_vy = -df * math.sin(yaw_rad) - dl * math.cos(yaw_rad)
            vx = WP_K * body_vx
            vy = WP_K * body_vy
            speed = math.hypot(vx, vy)
            if speed > max_v:
                scale = max_v / speed
                vx *= scale
                vy *= scale
            yaw_err = ((target_yaw - yaw) + 180.0) % 360.0 - 180.0
            wz = LANE_K_YAW * yaw_err
            wz = max(-LANE_MAX_WZ, min(LANE_MAX_WZ, wz))
            now = time.time()
            if now - last_log >= 1.0:
                print("[stage4] walk_to_xy target=(%.3f,%.3f) pos=(%.3f,%.3f) dist=%.3f" %
                      (target_x, target_y, self._stage_x(), x_fwd, dist))
                last_log = now
            self._walk(vx, vy, wz)
            time.sleep(0.07)
        self._stop()
    def walk_timed(self, vx, vy, wz, secs):
        t_end = time.time() + secs
        while time.time() < t_end:
            self._check_odom("walk_timed")
            self._walk(vx, vy, wz)
            time.sleep(0.07)
        self._stop()

    # ---------------- 低姿（自定义步态） ----------------
    def _low_on(self, vx):
        # 与 low_gait_test 验证过的流程一致：
        #   mode12 站定 -> 上传步态(Full_Params 带 vel_x) -> 切 mode62
        # 注意1：速度由 Full_Params 驱动，指令 vel_des 必须为 [0,0,0]。
        # 注意2：adapter 心跳是独立进程，改完 cmd 必须 _sync_heartbeat_command() 才会真正下发！
        with self.adp._cmd_lock:
            self.adp.cmd.mode = 12
            self.adp.cmd.gait_id = 0
            self.adp.cmd.contact = 15
            self.adp.cmd.value = 0
            self.adp.cmd.duration = 0
            self.adp.cmd.vel_des = [0.0, 0.0, 0.0]
            self.adp.cmd.pos_des = [0.0, 0.0, 0.28]
            self.adp.cmd.step_height = [0.09, 0.09]
        self.adp._sync_heartbeat_command()
        print("[stage4] 低姿准备：站定 ...")
        time.sleep(1.5)

        if not self.low.ensure_uploaded(vel_x=vx):
            raise RuntimeError("[stage4] 低姿步态上传失败")

        with self.adp._cmd_lock:
            self.adp.cmd.mode = 62
            self.adp.cmd.gait_id = 110
            self.adp.cmd.vel_des = [0.0, 0.0, 0.0]
            self.adp.cmd.pos_des = [0.0, 0.0, 0.0]
            self.adp.cmd.step_height = [0.0, 0.0]
            self.adp.cmd.contact = 15
            self.adp.cmd.value = 0
            self.adp.cmd.duration = 9600
        self.adp._sync_heartbeat_command()
        self.low_active = True
        print("[stage4] 低姿开启 vx=%.2f（速度由步态参数驱动）" % vx)

    def _low_off(self):
        # 直接用 adapter.walk(0,0,0)：恢复 mode11/gait26 并同步到心跳进程
        self.adp.walk(0.0, 0.0, 0.0)
        self.low_active = False
        print("[stage4] 低姿关闭")

    def recover_stand(self):
        print("[stage4] 恢复站立")
        self.adp.stand()
        time.sleep(0.5)

    # ---------------- 视觉判据 ----------------
    def _limit_bar_close(self):
        for d in self.det.get_detections():
            if d.get("class_name") == "limit_bar" and d.get("bbox_area_norm", 0.0) >= self.bar_norm:
                return True
        return False

    def _best_norm(self, cls):
        norms = [d.get("bbox_area_norm", 0.0) for d in self.det.get_detections()
                 if d.get("class_name") == cls]
        return max(norms) if norms else None

    def _obstacle2_close(self):
        """检查障碍2（足球/可乐/橙球）是否已'够近'，返回类别名或 None"""
        for d in self.det.get_detections():
            nm = d.get("class_name")
            if nm not in OBSTACLE2:
                continue
            if nm == "football":
                if d.get("bbox_area_norm", 0.0) >= FOOTBALL_TRIGGER_NORM:
                    return nm
            elif nm == "cola":
                if d.get("bbox_area_norm", 0.0) >= COLA_TRIGGER_NORM:
                    return nm
            elif nm == "orange_ball":
                if d.get("bbox_area_norm", 0.0) >= ORANGE_TRIGGER_NORM:
                    return nm
            else:
                if d.get("bbox_area", 0.0) >= BBOX_THRESHOLD:
                    return nm
        return None

    def _face_obstacle(self, cls, tol=0.02, max_vy=0.04, timeout=10.0,
                       stable_frames=3, lost_frames=5):
        """横向移动使 cls 居于镜头中心（正向面对它）。
        - 对中前要求目标连续 stable_frames 帧出现才启动；
        - 横移中短时丢失会停下等待；连续丢失 lost_frames 帧后，按最后一次 cx 小幅盲移约0.1m；
          盲移后仍看不到则放弃（返回 False，由调用方继续前进1.5m）；看到了则继续精准对中。
        """
        t0 = time.time()
        seen_cnt = 0
        center_cnt = 0
        lost_cnt = 0
        last_cx = None
        small_shot_done = False
        while time.time() - t0 < timeout:
            self._wait_detector()
            best = None
            for d in self.det.get_detections():
                if d.get("class_name") != cls:
                    continue
                cx = d.get("bbox_cx_norm", 0.5)
                if best is None or abs(cx - 0.5) < abs(best[1] - 0.5):
                    best = (d, cx)
            if best is not None:
                # 目标可见
                lost_cnt = 0
                last_cx = best[1]
                seen_cnt += 1
                cx = best[1]
                err = cx - 0.5          # >0 目标偏右 -> 右移(vy<0)
                if seen_cnt < stable_frames:
                    # 先确认目标稳定出现，再开始横移
                    self._stop()
                    time.sleep(0.05)
                    continue
                if abs(err) <= tol:
                    center_cnt += 1
                    if center_cnt >= stable_frames:
                        print("[stage4] %s 已对中 cx=%.3f" % (cls, cx))
                        self._stop()
                        return True
                    # 已在容差内但还需稳定几帧 -> 不动
                    self._stop()
                    time.sleep(0.05)
                    continue
                center_cnt = 0
                vy = -LANE_K * err
                vy = max(-max_vy, min(max_vy, vy))
                self._walk(0.0, vy, 0.0)
                time.sleep(0.07)
            else:
                # 目标当前帧不在画面
                lost_cnt += 1
                if last_cx is None:
                    # 从头到尾没看到：等几帧就放弃
                    if lost_cnt >= lost_frames:
                        print("[stage4] %s 始终不在画面中，放弃对中" % cls)
                        self._stop()
                        return False
                    self._stop()
                    time.sleep(0.05)
                    continue
                if not small_shot_done and lost_cnt >= lost_frames:
                    # 第一次持续丢失：按最后 cx 小幅盲移约 0.1m，再找
                    small_shot_done = True
                    err_last = last_cx - 0.5
                    vy_shot = 0.10 if err_last < 0 else -0.10   # 偏左->左移(vy>0)
                    print("[stage4] %s 丢失，按最后 cx=%.3f 盲移 0.1m ..." % (cls, last_cx))
                    t_end = time.time() + 1.0
                    while time.time() < t_end:
                        self._walk(0.0, vy_shot, 0.0)
                        time.sleep(0.07)
                    self._stop()
                    lost_cnt = 0
                    continue
                if small_shot_done and lost_cnt >= lost_frames:
                    print("[stage4] %s 盲移后仍看不到，放弃对中（直接前进）" % cls)
                    self._stop()
                    return False
                # 短暂丢失：停下等它回来
                self._stop()
                time.sleep(0.05)
        self._stop()
        print("[stage4] %s 对中超时" % cls)
        return False

    # ---------------- 赛段逻辑（照搬初赛，坐标已换算） ----------------
    def walk_until_detected(self, classes, vx=WALK_VX, going_up=True, lock_x=None):
        t0 = time.time()
        last_dbg_t = 0.0
        last_center_reject_t = 0.0
        require_center = all(cls in OBSTACLE1 for cls in classes)
        while True:
            self._wait_detector()
            if time.time() - t0 > 60.0:
                raise RuntimeError("walk_until_detected 整体超时")
            f = self._fwd()
            if going_up and f >= DETECT_CEILING_Y:
                print("[stage4] 上行未检测到，到达上限 y=%.3f" % f)
                return "none"
            if not going_up and f <= DETECT_FLOOR_Y:
                print("[stage4] 下行未检测到，到达下限 y=%.3f" % f)
                return "none"
            action_cx = None
            if require_center:
                centered = []
                rejected = []
                for detection in self.det.get_detections():
                    if detection.get("class_name") not in classes:
                        continue
                    cx = float(detection.get("bbox_cx_norm", 0.5))
                    if abs(cx - 0.5) <= OBSTACLE1_CENTER_TOL:
                        centered.append(detection)
                    else:
                        rejected.append((detection.get("class_name"),
                                         detection.get("bbox_area", 0.0), cx))
                if rejected and time.time() - last_center_reject_t >= 0.5:
                    details = " ".join(
                        "%s(area=%.0f,cx=%.2f)" % item for item in rejected
                    )
                    print("[stage4] 忽略非中心障碍 %s（要求 cx=0.50±%.2f）" %
                          (details, OBSTACLE1_CENTER_TOL), flush=True)
                    last_center_reject_t = time.time()
                if centered:
                    best = max(centered, key=lambda item: item.get("bbox_area", 0.0))
                    result = (best["class_name"], best.get("bbox_area", 0.0))
                    action_cx = float(best.get("bbox_cx_norm", 0.5))
                else:
                    result = None
            else:
                result = self.det.nearest_of(classes)
            if result is not None:
                cls_name, area = result
                if cls_name == "limit_bar":
                    close_enough = self._limit_bar_close()
                    if not close_enough:
                        # 调试：打印当前限高杆 norm，便于标定 --bar-norm
                        norms = [d.get("bbox_area_norm", 0.0) for d in self.det.get_detections()
                                 if d.get("class_name") == "limit_bar"]
                        if norms and max(norms) >= 0.05:
                            print("[stage4] limit_bar norm=%.3f (阈值 %.3f) y=%.3f" %
                                  (max(norms), self.bar_norm, f))
                elif cls_name == "football":
                    fbn = self._best_norm("football")
                    close_enough = fbn is not None and fbn >= FOOTBALL_TRIGGER_NORM
                elif cls_name == "cola":
                    cbn = self._best_norm("cola")
                    close_enough = cbn is not None and cbn >= COLA_TRIGGER_NORM
                elif cls_name == "orange_ball":
                    obn = self._best_norm("orange_ball")
                    close_enough = obn is not None and obn >= ORANGE_TRIGGER_NORM
                else:
                    close_enough = area >= BBOX_THRESHOLD
                if close_enough:
                    center_text = "" if action_cx is None else " cx=%.2f" % action_cx
                    print("[stage4] 检测到 %s area=%.0f%s y=%.3f" %
                          (cls_name, area, center_text, f))
                    return cls_name
            # 调试：搜索期间打印各类别 norm（≥0.03，每0.5s一次），定位障碍2是否被看到
            now = time.time()
            if now - last_dbg_t >= 0.5:
                dbg = []
                for d in self.det.get_detections():
                    if d.get("class_name") in classes:
                        n = d.get("bbox_area_norm", 0.0)
                        if n >= 0.03:
                            dbg.append("%s=%.3f" % (d.get("class_name"), n))
                if dbg:
                    print("[stage4] 检测debug %s y=%.3f" % (" ".join(dbg), f))
                    last_dbg_t = now
            if lock_x is not None:
                vy, wz = self._lane(lock_x)
            else:
                vy, wz = 0.0, 0.0
            self._walk(vx, vy, wz)
            time.sleep(0.07)
        return "none"

    def handle_limit_bar(self, lock_x=None):
        # 语音触发(60%)后：低姿走 LOW_DIST_M 米 -> 恢复正常步态。
        # 注意：低姿阶段【不能调 adapter.walk()】，mode62 由心跳持续下发，
        #       速度来自上传的 Full_Params(vel_x)，狗自动低姿前进。
        # 返回低姿起点 fwd，作为限高杆位置参考（下行用它算低姿区间）。
        start_fwd = self._fwd()
        target_fwd = min(start_fwd + LOW_DIST_M, DETECT_CEILING_Y)
        print("[stage4] limit_bar 低姿行走 %.2fm（起点 y=%.3f，上限 y=%.3f）..." %
              (LOW_DIST_M, start_fwd, DETECT_CEILING_Y))
        t0 = time.time()
        last_f, last_t = start_fwd, time.time()
        while True:
            if time.time() - t0 > 25.0:
                raise RuntimeError("handle_limit_bar 低姿行走超时")
            self._check_odom("limit_bar_low")
            f = self._fwd()
            if f >= target_fwd:
                print("[stage4] 低姿行走完成 y=%.3f" % f)
                break
            # 低姿中同步检测障碍2：只记录，不提前结束低姿（避免碰杆），低姿结束后再处理
            hit2 = self._obstacle2_close()
            if hit2 is not None and self._pending_obstacle2 is None:
                self._pending_obstacle2 = hit2
                print("[stage4] 低姿中识别到障碍2: %s，待低姿结束后处理 fwd=%.3f" % (hit2, f))
            # 位移停滞(步态可能结束)则提前恢复，避免干等超时
            if abs(f - last_f) < 0.005 and time.time() - last_t > 1.5:
                print("[stage4] 低姿位移停滞，提前恢复 fwd=%.3f" % f)
                break
            if abs(f - last_f) >= 0.005:
                last_f, last_t = f, time.time()
            time.sleep(0.07)
        end_fwd = self._fwd()
        self._stop()
        self._low_off()
        self.recover_stand()
        return start_fwd, end_fwd

    def handle_limit_bar_down(self, lock_x, low_start_ref, low_end_ref):
        # 上行低姿段 [low_start_ref, low_end_ref] 覆盖限高杆。
        # 下行：在低姿段终点上方 LEAD 就切低姿，倒着低姿走过整段，
        #       到低姿段起点下方 CLEAR 再站起——保证倒着过杆全程低姿。
        down_low_start = low_end_ref + LIMIT_BAR_DOWN_LEAD
        down_low_end = low_start_ref - LIMIT_BAR_DOWN_CLEAR
        print("[stage4] limit_bar 下行 低姿区间 [%.3f, %.3f]（上行低姿段 [%.3f, %.3f]）" %
              (down_low_start, down_low_end, low_start_ref, low_end_ref))
        # 正常姿态倒走到低姿区间上方
        self.walk_to_y(down_low_start, vx=-WALK_VX, lock_x=lock_x)
        # 低姿倒走过杆：不能调 adapter.walk()，mode62 心跳驱动（vel_x=-LOW_VX 已上传）
        self._low_on(vx=-LOW_VX)
        t0 = time.time()
        while True:
            if time.time() - t0 > 20.0:
                raise RuntimeError("handle_limit_bar_down 低姿倒走超时")
            self._check_odom("limit_bar_down")
            f = self._fwd()
            if f <= down_low_end:
                break
            time.sleep(0.07)
        self._stop()
        self._low_off()
        self.recover_stand()

    def _walk_forward_dist(self, dist, speed=WALK_VX, timeout=15.0):
        """沿当前朝向直走 dist 米（用 odometry 位移判断，适合面朝 +lat/-lat 时走侧向距离）"""
        x0, y0, _z = self.adp.get_position()
        t0 = time.time()
        while time.time() - t0 < timeout:
            self._check_odom("walk_forward_dist")
            x, y, _z = self.adp.get_position()
            if math.hypot(x - x0, y - y0) >= dist:
                break
            self._walk(speed, 0.0, 0.0)
            time.sleep(0.07)
        self._stop()

    def _walk_lat_to(self, target_x, speed=WALK_VX, forward=True, timeout=15.0, facing=-1):
        """横向走到绝对 x=target_x；facing=-1 朝 -x，facing=+1 朝 +x。"""
        t0 = time.time()
        yaw_target = 90.0 if facing == -1 else -90.0
        while time.time() - t0 < timeout:
            self._check_odom("walk_lat_to")
            stage_x = self._stage_x()
            if forward:
                reached = (stage_x <= target_x + 0.02) if facing == -1 else (stage_x >= target_x - 0.02)
            else:
                reached = (stage_x >= target_x - 0.02) if facing == -1 else (stage_x <= target_x + 0.02)
            if reached:
                break
            vx = speed if forward else -speed
            yaw_err = ((yaw_target - self._yaw()) + 180.0) % 360.0 - 180.0
            wz = max(-LANE_MAX_WZ, min(LANE_MAX_WZ, LANE_K_YAW * yaw_err))
            self._walk(vx, 0.0, wz)
            time.sleep(0.07)
        self._stop()

    def _strafe_fwd(self, delta, speed=0.15, timeout=10.0, facing=-1):
        """横向移动 ±y（不纵向走）：facing=-1 面朝全局 -x，facing=+1 面朝全局 +x。"""
        target = self._fwd() + delta
        t0 = time.time()
        vy_sign = (1.0 if delta > 0 else -1.0) * facing
        yaw_target = 90.0 if facing == -1 else -90.0
        while time.time() - t0 < timeout:
            self._check_odom("strafe_fwd")
            f = self._fwd()
            if (delta > 0 and f >= target - 0.01) or (delta < 0 and f <= target + 0.01):
                break
            yaw_err = ((yaw_target - self._yaw()) + 180.0) % 360.0 - 180.0
            wz = max(-LANE_MAX_WZ, min(LANE_MAX_WZ, LANE_K_YAW * yaw_err))
            self._walk(0.0, vy_sign * speed, wz)
            time.sleep(0.07)
        self._stop()

    def handle_obstacle(self, channel_x, going_up):
        """绕不可跨越障碍物（初赛 blue_block）。绕行侧由模块常量 DETOUR_SIDE 决定：
        +1 从全局 +x 侧绕、-1 从全局 -x 侧绕（方向镜像）。
        上行：斜出到侧边 -> 沿侧边直走 -> 横向回到通道中心 -> 沿 y 前移 0.3m。
        下行：按上行路线反向返回通道中心。"""
        side_x = channel_x + DETOUR_SIDE * BB_DX_OUT
        facing = -DETOUR_SIDE          # 回中心线时，右侧绕行面朝 -x，左侧绕行面朝 +x
        print("[stage4] obstacle 绕行 up=%s x=%.3f（侧边 x=%.3f，%s侧）" %
              (going_up, channel_x, side_x, "右" if DETOUR_SIDE > 0 else "左"))
        if going_up:
            self.walk_to_xy(BB_UP_WP1_Y, side_x)         # ① 斜出到侧边
            self.walk_to_xy(BB_UP_WP2_Y, side_x)         # ② 沿侧边直走
            self.turn_to_yaw(90.0 * DETOUR_SIDE)         # 右侧绕行面朝 -x；左侧绕行面朝 +x
            self._walk_lat_to(channel_x, forward=True, facing=facing)   # ④ 直走回中心线
            self._strafe_fwd(0.3, facing=facing)         # ⑤ 沿 y 前移 0.3m
            self.turn_to_yaw(FWD_YAW)                    # ⑥ 转回 +fwd
        else:
            # 下行：把上行路径倒着走（跟上行反着来）
            self.turn_to_yaw(90.0 * DETOUR_SIDE)         # ① 右侧绕行面朝 -x；左侧绕行面朝 +x
            self._strafe_fwd(-0.3, facing=facing)        # ② 横向移 -0.3m（-fwd）
            self._walk_lat_to(side_x, forward=False, facing=facing)  # ③ 后退退到侧边
            self.turn_to_yaw(FWD_YAW)                    # ④ 转回 +fwd
            self.walk_to_y(BB_DN_WP1_Y, vx=-WALK_VX, lock_x=side_x)   # ⑤ 后退沿侧边直下
            self.walk_to_xy(self._obstacle1_detect_fwd if self._obstacle1_detect_fwd is not None else BB_DN_WP1_Y, channel_x)  # ⑥ 斜着倒退回中心线
    def handle_obstacle2_pass(self, cls, hold_x=None):
        """统一的障碍2处理：向前走 OBSTACLE2_WALK_M 米（不超上行上限），然后进入下行"""
        start_fwd = self._fwd()
        target_fwd = min(start_fwd + OBSTACLE2_WALK_M, DETECT_CEILING_Y)
        print("[stage4] 处理 %s：向前走 %.2fm（目标 y=%.3f，上限 y=%.3f）..." %
              (cls, OBSTACLE2_WALK_M, target_fwd, DETECT_CEILING_Y))
        t0 = time.time()
        while True:
            if time.time() - t0 > 25.0:
                raise RuntimeError("handle_obstacle2_pass 超时")
            self._check_odom("obstacle2_walk")
            f = self._fwd()
            if f >= target_fwd:
                print("[stage4] %s 处理完成，停止于 y=%.3f" % (cls, f))
                break
            if hold_x is not None:
                vy, wz = self._lane(hold_x)
            else:
                vy, wz = 0.0, 0.0
            self._walk(WALK_VX, vy, wz)
            time.sleep(0.07)
        self._stop()

    def handle_football_low_push(self, cls, target_dist=FOOTBALL_PUSH_M):
        """足球：对中完成后用低姿（mode62）向前推 target_dist 米，再恢复正常步态。
        原因：正常姿态底盘太高，足球会卡在身子下面；低姿推球更稳。
        """
        start_fwd = self._fwd()
        target_fwd = min(start_fwd + target_dist, DETECT_CEILING_Y)
        self._low_on(vx=LOW_VX)
        print("[stage4] %s 低姿推球：向前走 %.2fm（目标 y=%.3f，上限 y=%.3f）..." %
              (cls, target_dist, target_fwd, DETECT_CEILING_Y))
        t0 = time.time()
        last_f, last_t = self._fwd(), time.time()
        while True:
            if time.time() - t0 > 30.0:
                raise RuntimeError("handle_football_low_push 超时")
            self._check_odom("football_low_push")
            f = self._fwd()
            if f >= target_fwd:
                print("[stage4] %s 低姿推球完成 fwd=%.3f" % (cls, f))
                break
            # 位移停滞(步态可能结束)则提前恢复，避免干等
            if abs(f - last_f) < 0.005 and time.time() - last_t > 1.5:
                print("[stage4] %s 低姿位移停滞，提前恢复 fwd=%.3f" % (cls, f))
                break
            if abs(f - last_f) >= 0.005:
                last_f, last_t = f, time.time()
            time.sleep(0.07)
        self._stop()
        self._low_off()
        self.recover_stand()

    def traverse_channel(self, channel_x):
        self._pending_obstacle2 = None
        self._set_speech_mute("up")
        print("[stage4] ===== 通道 x=%.3f 上行 =====" % channel_x)
        cls1 = self.walk_until_detected(OBSTACLE1, vx=WALK_VX, going_up=True, lock_x=channel_x)
        self._stop()
        self._obstacle1_detect_fwd = self._fwd()   # 记录障碍物检测时的 fwd

        low_seg = None
        if cls1 == "limit_bar":
            self._low_on(vx=LOW_VX)
            low_seg = self.handle_limit_bar(lock_x=channel_x)
        elif cls1 == "obstacle":
            self.handle_obstacle(channel_x, going_up=True)

        if self._pending_obstacle2 is not None:
            cls2 = self._pending_obstacle2
            self._pending_obstacle2 = None
            print("[stage4] 低姿中已识别障碍2: %s，直接处理" % cls2)
        else:
            cls2 = self.walk_until_detected(OBSTACLE2, vx=WALK_VX, going_up=True, lock_x=channel_x)
        self._stop()
        if cls2 == "football":
            # 足球：正常姿态对中→低姿推球前进 1.5m→恢复正常
            self._face_obstacle(cls2)
            self.handle_football_low_push(cls2)
        elif cls2 in ("orange_ball", "cola"):
            self._face_obstacle(cls2)
            self.handle_obstacle2_pass(cls2, hold_x=self._stage_x())

        # 下行（下行不播报）
        self._set_speech_mute("down")
        if cls1 == "obstacle":
            print("[stage4] 下行绕障碍")
            self.walk_to_y(BB_DN_START_Y, vx=-WALK_VX, lock_x=channel_x)
            self.handle_obstacle(channel_x, going_up=False)
        elif cls1 == "limit_bar" and low_seg is not None:
            self.handle_limit_bar_down(lock_x=channel_x,
                                       low_start_ref=low_seg[0], low_end_ref=low_seg[1])

        print("[stage4] 回到走廊 y=%.3f" % CORRIDOR_Y)
        self.walk_to_y(CORRIDOR_Y, vx=-WALK_VX, lock_x=channel_x)
        print("[stage4] ===== 通道 x=%.3f 完成 =====" % channel_x)

    # ---------------- 整段 ----------------
    def run(self, max_channels=None):
        print("[stage4] ==== 第四赛段开始 ====")
        self.turn_to_yaw(FWD_YAW, tol=TURN_TOL)

        if self.line_ready():
            print("[stage4] 起点右鱼眼校准：目标 x=%.3f，右线距=%.2fm" %
                  (STAGE4_START_X, RIGHT_START_LINE_TARGET), flush=True)
            if not self.line_calibrate_stage_x(RIGHT_START_LINE_TARGET, STAGE4_START_X):
                print("[stage4] WARN: 起点 x 校准未达到目标，继续第四赛段", flush=True)

        if self.cfg.get("start_at_corridor"):
            print("[stage4] 已在走廊起点（跳过 approach walk）")
        else:
            print("[stage4] 走到走廊 y=%.3f" % CORRIDOR_Y)
            self.walk_to_y(CORRIDOR_Y, vx=WALK_VX)

        if self.cfg.get("wait_detector_before_channels"):
            self._wait_detector(timeout=20.0)

        channels = CHANNEL_X if max_channels is None else CHANNEL_X[:max_channels]
        for i, channel_x in enumerate(channels, 1):
            if i == 1:
                print("[stage4] 前往通道 1 x=%.3f（斜走，不做左鱼眼校准）" % channel_x)
                self.walk_to_xy(CORRIDOR_Y, channel_x)
            else:
                print("[stage4] 前往通道 %d x=%.3f（左鱼眼循线）" % (i, channel_x))
                if self.line_ready():
                    ok = self.line_transition(side="left", stop_x=channel_x, target_yaw=90.0)
                    if not ok:
                        print("[stage4] 循线失败，回退斜走")
                        self.walk_to_xy(CORRIDOR_Y, channel_x)
                else:
                    self.walk_to_xy(CORRIDOR_Y, channel_x)
            self.traverse_channel(channel_x)

        # 第三通道完成后先在 +y 朝向用左鱼眼校正左线距；不改坐标。
        if self.line_ready() and channels and abs(channels[-1] - CHANNEL_X[-1]) < 1e-6:
            self.calibrate_third_channel_return()

        # 先按里程计走到 x=1.0，再启用右鱼眼校 y，并循线走到 x=2.7。
        if self.line_ready():
            self._phase_i_odom(target_x=RETURN_LINE_START_X)
            ok = self.line_transition(side="right", stop_x=FINISH_APPROACH_X, target_yaw=-90.0)
            if not ok:
                print("[stage4] 回程循线失败，按里程计走到终点校准前 x=%.3f" % FINISH_APPROACH_X)
                self._phase_i_odom(target_x=FINISH_APPROACH_X)
        else:
            self._phase_i_odom(target_x=FINISH_APPROACH_X)

        # 终点：右鱼眼校准（狗面向 +fwd，右侧为地图边界黄线）——只横移+旋转，不动 fwd
        if self.line_ready():
            self.adp.align_yaw(FWD_YAW, tol=2.0, timeout=8.0)
            print("[stage4] 终点右鱼眼校准：目标 x=%.3f，右线距=%.2fm" %
                  (STAGE4_FINISH_X, RIGHT_FINISH_LINE_TARGET))
            if self.line_calibrate_stage_x(RIGHT_FINISH_LINE_TARGET, STAGE4_FINISH_X):
                print("[stage4] 终点线距已校准，回正面向 +y", flush=True)
                if not self.adp.align_yaw(FWD_YAW, tol=2.0, timeout=8.0):
                    print("[stage4] WARNING: 终点回正 +y 未在时限内完成", flush=True)

        # 第四赛段结束（不再去交棒点；直接停住退出）
        self._stop()
        print("[stage4] ==== 第四赛段完成 ====")
        return True


def main():
    ap = argparse.ArgumentParser(description="第四赛段真机（完整流程）")
    ap.add_argument("--no-wait-detector", action="store_true",
                    help="不强制等待电脑识别链路（调试用）")
    ap.add_argument("--start-at-corridor", action="store_true",
                    help="狗已摆在走廊起点，跳过 approach walk（推荐单跑用）")
    ap.add_argument("--max-channels", type=int, default=None,
                    help="只跑前 N 条通道（调试用）")
    ap.add_argument("--bar-norm", type=float, default=None,
                    help="限高杆触发低姿的占画面比例阈值（默认0.65；现场看日志里的 norm 调）")
    ap.add_argument("--det-port", type=int, default=9890)
    ap.add_argument("--pc-host", default=DEFAULT_PC_HOST,
                    help="运行 live_detect_server.py 的电脑 IP（默认192.168.43.102）")
    ap.add_argument("--stream-port", type=int, default=DEFAULT_STREAM_PORT,
                    help="RGB 推流连接的电脑端口（默认9891）")
    ap.add_argument("--no-stream", action="store_true",
                    help="不自动启动 RGB 推流（联跑或手动推流时使用）")
    ap.add_argument("--stream-script", default=DEFAULT_STREAM_SCRIPT,
                    help=argparse.SUPPRESS)
    ap.add_argument("--gait-dir", default=os.path.join(BASE_DIR, "gait"))
    ap.add_argument("--handoff-dir", default="",
                    help="internal: stage3-to-stage4 standing handoff directory")
    ap.add_argument("--handoff-timeout", type=float, default=30.0,
                    help="internal: maximum handoff wait, seconds")
    args = ap.parse_args()

    adapter = RealDogAdapter(None)
    det = RemoteDetector(port=args.det_port)
    low = LowGait(gait_dir=args.gait_dir)
    stream_process = None
    line = None
    try:
        line = LineCalib()
    except Exception as e:
        print("[stage4] 黄线校准模块初始化失败: %s（将全程用里程计）" % e)
    cfg = {
        "start_at_corridor": args.start_at_corridor,
        "bar_norm": args.bar_norm if args.bar_norm is not None else LIMIT_BAR_AREA_NORM,
        "wait_detector_before_channels": not args.no_wait_detector,
        "handoff_line_calibration": bool(args.handoff_dir),
    }

    runner = Stage4Real(adapter, det, low, line=line, cfg=cfg)
    pos_stop = threading.Event()
    def _pos_printer():
        while not pos_stop.is_set():
            try:
                print("[pos] x=%.3f y=%.3f yaw=%.1f" %
                      (runner._stage_x(), runner._stage_y(), runner._yaw()), flush=True)
            except Exception:
                pass
            time.sleep(0.5)
    threading.Thread(target=_pos_printer, daemon=True).start()

    try:
        if not args.no_stream:
            print("[stage4] 启动 RGB 推流 -> %s:%d" %
                  (args.pc_host, args.stream_port), flush=True)
            stream_process = subprocess.Popen([
                sys.executable,
                args.stream_script,
                "--host",
                args.pc_host,
                "--port",
                str(args.stream_port),
            ])
        if not adapter.wait_odom(timeout=5.0):
            print("[stage4] 警告：未收到里程计，继续（坐标可能无效）")
        if args.handoff_dir:
            print("[stage4] handoff: keep existing stand posture", flush=True)
        else:
            adapter.stand()
        adapter.set_origin()
        print("[stage4] 起点已设为原点，面朝 yaw=%.1f" % adapter.get_yaw_deg())

        if args.handoff_dir:
            os.makedirs(args.handoff_dir, exist_ok=True)
            ready_path = os.path.join(args.handoff_dir, "stage4_ready")
            release_path = os.path.join(args.handoff_dir, "release_stage3")
            with open(ready_path, "w"):
                pass
            print("[stage4] handoff ready; waiting for stage3 heartbeat release", flush=True)
            deadline = time.time() + max(1.0, args.handoff_timeout)
            while not os.path.exists(release_path):
                if time.time() >= deadline:
                    raise RuntimeError("stage3 handoff release timeout")
                time.sleep(0.05)
            print("[stage4] handoff released; stage4 takes control", flush=True)

        start_y = CORRIDOR_Y if args.start_at_corridor else STAGE4_ENTRY_Y
        adapter.set_mapped_pose(start_y, 0.0, adapter.get_yaw_deg())
        print("[stage4] 起点映射为绝对 y=%.3f、x=%.3f" %
              (start_y, STAGE4_START_X), flush=True)

        ok = runner.run(max_channels=args.max_channels)
        print("[stage4] 完成，ok=%s" % ok)
    except KeyboardInterrupt:
        print("\n[stage4] 用户中断")
    except Exception as e:
        print("[stage4] 异常: %s" % e)
        import traceback
        traceback.print_exc()
        adapter.stop()
        try:
            if runner.low_active:
                runner._low_off()
        except Exception:
            pass
    finally:
        pos_stop.set()
        adapter.stop()
        time.sleep(0.3)
        adapter.shutdown()
        det.stop()
        if line is not None:
            try:
                line.shutdown()
            except Exception:
                pass
        if stream_process is not None and stream_process.poll() is None:
            stream_process.terminate()
            try:
                stream_process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                stream_process.kill()
                stream_process.wait()
        print("[stage4] end")


if __name__ == "__main__":
    main()
