#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第六赛段·攘金建功（真机版）

任务（赛题）：
  1) 把足球从"出口/缺口"位置踢出（球从独木桥上飞出后不许放回）；
  2) 机器狗来到终点位置（= 缺口处的终点圈），停下（或趴下）结束。
  全程自主，启动后不允许再触碰电脑/远程操控。

坐标（以第六赛段起点为原点，起点 = 狗跳下独木桥的位置 = 初赛世界 (2.007, 13.050)）：
  局部 fwd = 初赛世界 y - 13.050
  局部 lat = 初赛世界 x - 2.007
  场地：lat ∈ [-1.91, +0.89]，fwd ∈ [-0.35, +1.95]
  缺口/终点圈：lat=+0.89, fwd=-0.10
  球出界判据：lat > +1.19（= 初赛世界 x > 3.20）

流程：
  set_origin(面朝 +fwd) -> 扫球(转圈找 football) -> 接近+居中 ->
  绕到球后(behind 点 = 球-缺口连线的反向) -> 面向缺口推球(球出界) ->
  去缺口终点圈 -> 停住结束。

依赖（与第四赛段同一套）：
  - adapter.py 的 RealDogAdapter（LCM 运动/里程计）
  - remote_detector.py（电脑端 YOLO 全量检测回传，TCP 9890）
  - 电脑端 live_detect_server.py --targets football（用第四赛段训练的模型，football 类）

⚠ 待现场标定（占位参数，标完再调）：
  BALL_DIST_K(=A) / BALL_DIST_B / BALL_FOV_DEG（球"面积占比->距离"模型，A/B 已标定）
  BALL_*_NORM（接近/够近阈值）、PUSH_VX（推球速度，初赛 0.5~0.9 m/s）
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def _wrap_deg(angle):
    return ((float(angle) + 180.0) % 360.0) - 180.0

from adapter import RealDogAdapter
from depth import DEFAULT_DEPTH_TOPIC, DepthBallTracker, ensure_depth_camera
from remote import RemoteDetector


DEFAULT_STREAM_PC_HOST = "192.168.43.102"
DEFAULT_STREAM_PORT = 9891
DEFAULT_STREAM_FPS = 15.0


def _start_rgb_stream(pc_host, port, fps):
    stream_script = os.path.join(BASE_DIR, "stream.py")
    if not os.path.exists(stream_script):
        raise RuntimeError("[stage6] missing RGB stream script: %s" % stream_script)
    command = [
        sys.executable,
        "-u",
        stream_script,
        "--host",
        pc_host,
        "--port",
        str(port),
        "--fps",
        str(fps),
    ]
    print("[stage6] start RGB stream -> %s:%d" % (pc_host, port))
    return subprocess.Popen(command)

# ================= CONFIG =================
# ---- 场地（局部坐标，起点为原点） ----
_START_X, _START_Y = 2.007, 13.050          # 初赛世界坐标的第六赛段起点（狗跳下桥位置）
FIELD_LAT_MIN = 0.10 - _START_X             # -1.907
FIELD_LAT_MAX = 2.90 - _START_X             # +0.893
FIELD_FWD_MIN = 12.70 - _START_Y            # -0.350
FIELD_FWD_MAX = 15.00 - _START_Y            # +1.950
# 足球推向的缺口瞄准点，和狗穿出边界的位置分开处理。
PUSH_GAP_WORLD_X = 3.000
PUSH_GAP_WORLD_Y = 12.500
PUSH_GAP_LAT = PUSH_GAP_WORLD_X - _START_X
PUSH_GAP_FWD = PUSH_GAP_WORLD_Y - _START_Y

# 狗以 y=12.75 穿过右侧开口，再到终点趴下。x=2.70 是入洞前的对齐点，
# 因此经过 x=2.90 时已经面向世界 +x，且 y 已经稳定在 12.75。
EXIT_CROSS_WORLD_X = 2.900
EXIT_CROSS_WORLD_Y = 12.750
EXIT_CROSS_LAT = EXIT_CROSS_WORLD_X - _START_X
EXIT_CROSS_FWD = EXIT_CROSS_WORLD_Y - _START_Y
EXIT_ALIGN_WORLD_X = 2.700
EXIT_ALIGN_LAT = EXIT_ALIGN_WORLD_X - _START_X
END_WORLD_X = 3.250
END_WORLD_Y = 12.750
END_LAT = END_WORLD_X - _START_X
END_FWD = END_WORLD_Y - _START_Y

# Existing depth-gap helpers retain these names for the dog exit geometry.
GAP_LAT = EXIT_CROSS_LAT
GAP_FWD = EXIT_CROSS_FWD
GAP_FWD_MIN = 12.70 - _START_Y              # 缺口 fwd 范围
GAP_FWD_MAX = 13.20 - _START_Y
BALL_OUT_LAT = 3.20 - _START_X              # 球出界判据 lat>1.193
END_CIRCLE = (END_LAT, END_FWD)             # 终点圈

# ---- 第五赛段跳下后的实际启动位姿 / 固定足球位置（世界坐标） ----
# 参考原点仍保持 _START_X/_START_Y，避免改变既有场地与出口坐标。
START_WORLD_X = 2.710
START_WORLD_Y = 13.250
START_WORLD_YAW = 90.0                       # 世界 -x，在本地 fwd/lat 坐标中为 +90deg
START_FWD = START_WORLD_Y - _START_Y
START_LAT = START_WORLD_X - _START_X
START_ADAPTER_Y = -START_LAT

BALL_INITIAL_WORLD_X = 0.600
BALL_INITIAL_WORLD_Y = 14.400
BALL_INITIAL_FWD = BALL_INITIAL_WORLD_Y - _START_Y
BALL_INITIAL_LAT = BALL_INITIAL_WORLD_X - _START_X
INITIAL_BALL_SCAN_WZ = -0.08                 # 从面向世界 -x 开始顺时针找球
INITIAL_BALL_SCAN_TIMEOUT = 6.0
INITIAL_BALL_AIM_TIMEOUT = 5.0
INITIAL_BALL_CONFIRM_FRAMES = 3
INITIAL_BALL_CONFIRM_CX_TOL = 0.12
INITIAL_BALL_HIGH_CONF = 0.55
INITIAL_BALL_OBSERVATION_WINDOW = 2.5
INITIAL_BALL_DIRECTION_TOL_DEG = 12.0
AIM_STABLE_FRAMES = 2
AIM_OBSERVATION_WINDOW = 1.0
AIM_LOST_HOLD_SECS = 0.70
AIM_LOST_WZ_MAX = 0.12

# ---- 斜坡防护 ----
RAMP_SLOW_DISTANCE = 0.65                    # 前深度发现上坡后开始限速的距离
RAMP_BLOCK_DISTANCE = 0.30                   # 仅抑制继续朝坡前进，仍允许转向/横移/后退
RAMP_SLOW_VX = 0.07
FIELD_RAMP_SLOW_MARGIN = 0.40                # 坐标边界辅助保护：靠近坡前先减速
FIELD_RAMP_BLOCK_MARGIN = 0.16

# ---- 起点预备路线（先移至左侧，再到球后方的球-缺口连线） ----
# set_origin 后 yaw=0 即面朝赛场 +y（本地 +fwd）。这段路线在正式的
# 视觉扫球前执行：移到左侧、走到球后方、按狗-球-缺口三点一线对准。
PREP_LEFT_LAT = 0.320 - _START_X
PREP_TOP_FWD = 14.750 - _START_Y
PREP_FACE_FWD_YAW = 0.0
PREP_CRAB_MAX_V = 0.175
PREP_FORWARD_MAX_V = 0.20
PREP_RUSH_VX = 0.55
PREP_RUSH_SECS = 1.20
PREP_RUSH_WZ_K = 0.025
PREP_RUSH_WZ_MAX = 0.35
PREP_BALL_RIGHT_CX = 0.56
PREP_BALL_RIGHT_VY_MIN = 0.035
PREP_BALL_RIGHT_VY_MAX = 0.10
PREP_BALL_RIGHT_VY_K = 0.70
PREP_BALL_RIGHT_LOG_INTERVAL = 0.50
PREP_BEHIND_TIMEOUT = 20.0
PREP_LINE_CONFIRM_FRAMES = 3
PREP_LINE_CONFIRM_WINDOW = 2.0
PREP_LINE_POSITION_SPREAD = 0.22
PREP_LINE_CORRECTION_TOL = 0.10
PREP_LINE_CORRECTION_MAX_V = 0.12
PREP_LINE_CENTER_CX_TOL = 0.08
PREP_LINE_MAX_PASSES = 2
BALL_TRACK_HOLD_SECS = 0.50
BALL_TRACK_LOG_INTERVAL = 0.60
BALL_POSITION_HOLD_SECS = 8.0
PREP_BALL_LAT_OFFSET = PREP_LEFT_LAT - BALL_INITIAL_LAT
PREP_BALL_FWD_OFFSET = PREP_TOP_FWD - BALL_INITIAL_FWD
FIXED_CORNER_FWD = PREP_TOP_FWD
FIXED_CORNER_LAT = PREP_LEFT_LAT
FIXED_PUSH_OUT_MARGIN = 0.05
FIXED_FIRST_PUSH_RATIO = 0.50
FIXED_PUSH_MAX_V = 0.60
FIXED_PUSH_TIMEOUT = 10.0
FIXED_PUSH_LEFT_OFFSET = 0.08
FIXED_PUSH_LEFT_MAX_VY = 0.08
FIXED_PUSH_LEFT_TIMEOUT = 4.0

# ---- 足球识别（电脑 YOLO，RemoteDetector） ----
BALL_CLASS = "football"
BALL_CONF = 0.40                            # 置信度阈值（初赛用 0.4）
BALL_MIN_NORM = 0.004                       # 检测到球的最低占画面比例（扫描判定）——需标定
BALL_APPROACH_NORM = 0.012                  # 接近阈值（进入减速/居中）——需标定
BALL_CLOSE_NORM = 0.03                      # 够近可推阈值——需标定

# ---- 球 面积占比->距离 模型（占位，需现场标定） ----
BALL_DIST_K = 0.1635                        # A：dist = A/sqrt(area)+B ——标定 json 覆盖
BALL_DIST_B = 0.3009                        # B：补偿相机高度/球贴地偏差 ——标定 json 覆盖
BALL_FOV_DEG = 90.0                         # 前方 RGB 相机水平视场角(°)——标定
BALL_BEHIND_DIST = 0.55                     # 绕到球后时，与球的期望距离(m)
PUSH_LINE_MIN_BEHIND = 0.10                 # 狗至少位于球后方这么远才可直接推
PUSH_LINE_MAX_CROSS = 0.12                  # 狗到“球-缺口”直线的最大横向误差

# ---- 运动 ----
WALK_VX = 0.25
TURN_TOL = 3.0
PUSH_VX = 0.60                              # 推球速度（初赛 0.5~0.9，实机要试）
PUSH_VX_FAST = 0.80
PUSH_ACCEL_SECS = 1.2
PUSH_WZ_MAX = 0.50                          # 推球时转向限幅
PUSH_K_CX = 1.2                             # 推球时 球cx偏差 -> wz
PUSH_PITCH = 0.30                            # 推球时俯角（正值低头，约17.2°）
PUSH_TIMEOUT = 12.0                         # 推球最长时长(s)
PUSH_NO_PROGRESS = 3.0                      # 推球位移停滞判定(s)
APPROACH_VX = 0.30
APPROACH_VX_SLOW = 0.12
APPROACH_K_CX = 0.8                         # 接近时 cx偏差 -> wz
SCAN_TIMEOUT = 10.0                         # 扫描找球超时(s)
SCAN_WZ = 0.5

STALE_WAIT_TIMEOUT = 10.0                   # 电脑链路断连等待
DETECTOR_STALE_TIMEOUT = 1.5                 # 容忍电脑 YOLO 单帧推理/网络短暂停顿
ODOM_STALE_TIMEOUT = 3.0
NO_PROGRESS_TIMEOUT = 6.0

# ---- 推球后使用前深度相机贴右边界寻找缺口 ----
GAP_EDGE_YAW = -90.0                         # 面向世界 +x 的右侧边界
GAP_EDGE_TARGET_DISTANCE = 0.38
GAP_EDGE_OPEN_DISTANCE = 0.95
GAP_EDGE_OPEN_VALID_RATIO = 0.05
GAP_EDGE_OPEN_STABLE_FRAMES = 3
GAP_EDGE_FOLLOW_SPEED = 0.08
GAP_EDGE_DISTANCE_K = 0.55
GAP_EDGE_MAX_NORMAL_V = 0.08
GAP_EDGE_SEARCH_TIMEOUT = 14.0
GAP_EDGE_COORD_MARGIN = 0.24
GAP_EDGE_OPEN_DISTANCE = 1.30
GAP_ENTRY_LAT = GAP_LAT + 0.18
GAP_ENTRY_MAX_V = 0.14
GAP_ENTRY_TIMEOUT = 8.0
END_STAGE_MAX_V = 0.22
END_CROSS_MAX_V = 0.18
DEPTH_STARTUP_TIMEOUT = 4.0

# ---- 异常恢复/兜底 ----
FIELD_CENTER_FWD = (FIELD_FWD_MIN + FIELD_FWD_MAX) / 2.0   # 场地中心 fwd
FIELD_CENTER_LAT = (FIELD_LAT_MIN + FIELD_LAT_MAX) / 2.0   # 场地中心 lat
BACKUP_STEP = 0.4                    # 恢复时后退的距离(m)
PUSH_LOST_FRAMES = 20                # 推球连续多少帧看不到球判定为“太近出画面”
PUSH_LOST_BLIND_SECS = 2.0            # 球离开镜头后，按初始三点一线继续推的时长
PUSH_MAX_ATTEMPTS = 10               # 推球最大尝试次数
APPROACH_OVERALL_TIMEOUT = 120.0     # 接近阶段总时长上限(s)，超过按球已推出处理
AIM_WZ_MAX = 0.4              # 对准球时最大角速度(rad/s)
AIM_K = 1.5                   # 球 cx 偏差 -> 角速度 比例
AIM_TOL = 0.05                # 对准容差：cx 距画面中央 0.5 的误差
AIM_TIMEOUT = 6.0             # 对准超时(s)



# ================= 球距标定（stage6_ball_calib.py 生成） =================
def _load_ball_calib(glb):
    p = os.path.join(BASE_DIR, "stage6_ball_calib.json")
    if not os.path.exists(p):
        print("[stage6] 未找到球距标定 %s，用默认 BALL_DIST_K=%.3f" % (p, glb.get("BALL_DIST_K")))
        return
    try:
        with open(p, "r") as f:
            data = json.load(f)
        a = data.get("A") or data.get("K")
        if a and a > 0:
            glb["BALL_DIST_K"] = float(a)
        b = data.get("B", 0.0)
        if b:
            glb["BALL_DIST_B"] = float(b)
        print("[stage6] 已加载球距标定: A=%.4f B=%.4f (dist=A/sqrt(area)+B)" % (glb["BALL_DIST_K"], glb["BALL_DIST_B"]))
        fov = data.get("BALL_FOV_DEG")
        if fov and 10.0 < float(fov) < 170.0:
            glb["BALL_FOV_DEG"] = float(fov)
            print("[stage6] 已加载球距标定: BALL_FOV_DEG=%.1f" % float(fov))
    except Exception as e:
        print("[stage6] 读取球距标定失败（用默认值）: %s" % e)

_load_ball_calib(globals())

class Stage6Real(object):
    def __init__(self, adapter, det, depth_tracker=None,
                 assume_out_after_fixed_push=False):
        self.adp = adapter
        self.det = det
        self.depth = depth_tracker
        self._last_range_log_t = 0.0
        self._last_range_source = None
        self._ball_track = None
        self._last_ball_position = None
        self._last_ball_track_log_t = 0.0
        self._last_prep_ball_guard_log_t = 0.0
        self._last_depth_failure_log_t = 0.0
        self._last_detector_wait_log_t = 0.0
        self._last_ball_out_log_t = 0.0
        self._prep_push_yaw = None
        self._last_push_yaw = None
        self.assume_out_after_fixed_push = bool(assume_out_after_fixed_push)

    # ---------------- 基础 ----------------
    def _pos(self):
        return self.adp.get_position()

    def _fwd(self):
        return self._pos()[0]

    def _lat(self):
        return -self._pos()[1]

    def _yaw(self):
        return self.adp.get_yaw_deg()

    def _wait_detector(self, timeout=STALE_WAIT_TIMEOUT):
        t0 = time.time()
        waited = False
        while not self.det.alive():
            if time.time() - t0 > timeout:
                raise RuntimeError("[stage6] \u7535\u8111\u8bc6\u522b\u94fe\u8def\u65ad\u8fde\u8d85\u8fc7 %.0fs\uff0c\u7ec8\u6b62" % timeout)
            waited = True
            now = time.monotonic()
            if now - self._last_detector_wait_log_t >= 1.0:
                status = self.det.status() if hasattr(self.det, "status") else {}
                age = status.get("age")
                age_text = "never" if age is None or not math.isfinite(age) else "%.2fs" % age
                print("[stage6] 等待电脑识别链路：last_frame_age=%s stale_limit=%.2fs" % (
                    age_text, status.get("timeout", DETECTOR_STALE_TIMEOUT)))
                self._last_detector_wait_log_t = now
            self._stop()
            time.sleep(0.10)
        if waited:
            status = self.det.status() if hasattr(self.det, "status") else {}
            print("[stage6] 电脑识别链路恢复 waited=%.2fs frame_age=%.2fs detections=%d；保持当前姿态继续" % (
                time.time() - t0,
                status.get("age", 0.0),
                status.get("detections", 0),
            ))

    def _walk(self, vx, vy=0.0, wz=0.0, pitch=0.0):
        vx, vy = self._ramp_guard(vx, vy)
        self.adp.walk(vx, vy, wz, pitch=pitch)

    def _stop(self):
        self.adp.walk(0.0, 0.0, 0.0)

    def _face_yaw(self, target, tol=TURN_TOL, timeout=8.0):
        return self.adp.align_yaw(target, tol=tol, timeout=timeout)

    def _detector_frame(self):
        status = self.det.status() if hasattr(self.det, "status") else {}
        return status.get("frames")

    def _field_ramp_guard(self, vx, vy):
        """Keep commanded motion inside the field, except through the gap."""
        fwd, adapter_y, _z = self._pos()
        lat = -adapter_y
        yaw_rad = math.radians(self._yaw())
        local_df = vx * math.cos(yaw_rad) - vy * math.sin(yaw_rad)
        local_dlat = -(vx * math.sin(yaw_rad) + vy * math.cos(yaw_rad))
        gap_open = GAP_FWD_MIN - 0.05 <= fwd <= GAP_FWD_MAX + 0.05 and lat > GAP_LAT - 0.30

        def limit_toward_boundary(value, distance):
            if distance <= FIELD_RAMP_BLOCK_MARGIN:
                return 0.0
            if distance < FIELD_RAMP_SLOW_MARGIN:
                scale = (distance - FIELD_RAMP_BLOCK_MARGIN) / (
                    FIELD_RAMP_SLOW_MARGIN - FIELD_RAMP_BLOCK_MARGIN
                )
                return value * max(0.0, min(1.0, scale))
            return value

        if local_df < 0.0:
            local_df = -limit_toward_boundary(-local_df, fwd - FIELD_FWD_MIN)
        elif local_df > 0.0:
            local_df = limit_toward_boundary(local_df, FIELD_FWD_MAX - fwd)
        if not gap_open and local_dlat > 0.0:
            local_dlat = limit_toward_boundary(local_dlat, FIELD_LAT_MAX - lat)
        elif local_dlat < 0.0:
            local_dlat = -limit_toward_boundary(-local_dlat, lat - FIELD_LAT_MIN)

        guarded_vx = local_df * math.cos(yaw_rad) - local_dlat * math.sin(yaw_rad)
        guarded_vy = -local_df * math.sin(yaw_rad) - local_dlat * math.cos(yaw_rad)
        return guarded_vx, guarded_vy

    def _ramp_guard(self, vx, vy):
        vx, vy = self._field_ramp_guard(vx, vy)
        if self.depth is None or vx <= 0.0:
            return vx, vy
        observation = self.depth.ramp_ahead()
        if observation is None:
            return vx, vy
        distance = observation["distance"]
        if distance <= RAMP_BLOCK_DISTANCE:
            guarded_vx = 0.0
        elif distance <= RAMP_SLOW_DISTANCE:
            guarded_vx = min(vx, RAMP_SLOW_VX)
        else:
            guarded_vx = vx
        if abs(guarded_vx - vx) > 1e-3:
            now = time.monotonic()
            if now - self._last_range_log_t >= 0.6:
                print("[stage6] ramp guard distance=%.2fm tilt=%.1fdeg vx=%.2f->%.2f" % (
                    distance, observation["tilt_deg"], vx, guarded_vx))
                self._last_range_log_t = now
        return guarded_vx, vy

    def _walk_to(self, target_fwd, target_lat, max_v=0.35, tol=0.07,
                 timeout=40.0, ball_offset=None, behind_ball=False,
                 keep_ball_right=False, require_detector=True):
        """\u659c\u8d70\u5230 (fwd, lat)\uff0c\u9762\u671d\u4fdd\u6301\u5f53\u524d\u671d\u5411"""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if require_detector:
                self._wait_detector()
            if keep_ball_right:
                ball = self._ball()
                if (ball is not None and ball[1] >= BALL_MIN_NORM and
                        ball[2] < PREP_BALL_RIGHT_CX):
                    error = PREP_BALL_RIGHT_CX - ball[2]
                    vy = min(PREP_BALL_RIGHT_VY_MAX, max(
                        PREP_BALL_RIGHT_VY_MIN,
                        PREP_BALL_RIGHT_VY_K * error,
                    ))
                    yaw_error = _wrap_deg(PREP_FACE_FWD_YAW - self._yaw())
                    wz = max(-PREP_RUSH_WZ_MAX, min(PREP_RUSH_WZ_MAX, 0.03 * yaw_error))
                    self._walk(0.0, vy, wz)
                    now = time.monotonic()
                    if now - self._last_prep_ball_guard_log_t >= PREP_BALL_RIGHT_LOG_INTERVAL:
                        print("[stage6] prep ball guard: ball cx=%.2f, left-shift vy=+%.2f before +y" %
                              (ball[2], vy))
                        self._last_prep_ball_guard_log_t = now
                    time.sleep(0.05)
                    continue
            active_target_fwd = target_fwd
            active_target_lat = target_lat
            if ball_offset is not None or behind_ball:
                track = self._update_ball_track()
                if track is not None:
                    if behind_ball:
                        active_target_fwd, active_target_lat = self._behind_point(
                            track["fwd"], track["lat"])
                        phase = "behind"
                    else:
                        offset_fwd, offset_lat, phase = ball_offset
                        if offset_fwd is not None:
                            active_target_fwd = track["fwd"] + offset_fwd
                        if offset_lat is not None:
                            active_target_lat = track["lat"] + offset_lat
                    self._log_ball_track(
                        track, active_target_fwd, active_target_lat, phase)
            x, y, _z = self._pos()
            yaw = self._yaw()
            df = active_target_fwd - x
            dl = (-active_target_lat) - y
            dist = math.hypot(df, dl)
            if dist <= tol:
                break
            yaw_rad = math.radians(yaw)
            bvx = df * math.cos(yaw_rad) + dl * math.sin(yaw_rad)
            bvy = -df * math.sin(yaw_rad) + dl * math.cos(yaw_rad)
            spd = min(max_v, max(0.06, 0.8 * dist))
            vx = bvx / max(dist, 0.01) * spd
            vy = bvy / max(dist, 0.01) * spd
            self._walk(vx, vy, 0.0)
            time.sleep(0.05)
        self._stop()

    def _walk_fixed_to(self, target_fwd, target_lat, max_v, label,
                       tol=0.06, timeout=25.0):
        """Coordinate-only move: do not consult RGB or D430 and bypass their guards."""
        print("[stage6] fixed %s target=(fwd=%.2f, lat=%+.2f)" % (
            label, target_fwd, target_lat))
        deadline = time.monotonic() + timeout
        last_log_t = 0.0
        while time.monotonic() < deadline:
            fwd, adapter_y, _z = self._pos()
            lat = -adapter_y
            df = target_fwd - fwd
            dlat = target_lat - lat
            distance = math.hypot(df, dlat)
            if distance <= tol:
                self._stop()
                print("[stage6] fixed %s reached fwd=%.2f lat=%+.2f" % (
                    label, fwd, lat))
                return True
            yaw_rad = math.radians(self._yaw())
            target_adapter_y = -target_lat
            dy = target_adapter_y - adapter_y
            bvx = df * math.cos(yaw_rad) + dy * math.sin(yaw_rad)
            bvy = -df * math.sin(yaw_rad) + dy * math.cos(yaw_rad)
            speed = min(max_v, max(0.05, 0.8 * distance))
            vx = bvx / max(distance, 0.01) * speed
            vy = bvy / max(distance, 0.01) * speed
            self.adp.walk(vx, vy, 0.0)
            now = time.monotonic()
            if now - last_log_t >= 0.5:
                print("[stage6] fixed %s fwd=%.2f lat=%+.2f err=(%+.2f,%+.2f) cmd=(%.2f,%+.2f)" % (
                    label, fwd, lat, df, dlat, vx, vy))
                last_log_t = now
            time.sleep(0.05)
        self._stop()
        print("[stage6] fixed %s timeout fwd=%.2f lat=%+.2f" % (
            label, self._fwd(), self._lat()))
        return False

    def _side_step_fixed_push_left(self, heading):
        """Keep the push heading and shift the body slightly to its left."""
        heading_rad = math.radians(heading)
        start_fwd, start_lat = self._fwd(), self._lat()
        deadline = time.monotonic() + FIXED_PUSH_LEFT_TIMEOUT
        last_log_t = 0.0
        while time.monotonic() < deadline:
            delta_fwd = self._fwd() - start_fwd
            delta_lat = self._lat() - start_lat
            left_progress = (
                delta_fwd * -math.sin(heading_rad)
                + delta_lat * -math.cos(heading_rad)
            )
            remaining = FIXED_PUSH_LEFT_OFFSET - left_progress
            if remaining <= 0.015:
                self._stop()
                print("[stage6] fixed push left-offset complete=%.3fm yaw=%.1f" % (
                    left_progress, self._yaw()))
                return True
            vy = min(FIXED_PUSH_LEFT_MAX_VY, max(0.03, 0.8 * remaining))
            yaw_error = _wrap_deg(heading - self._yaw())
            wz = max(-0.25, min(0.25, 0.035 * yaw_error))
            self.adp.walk(0.0, vy, wz)
            now = time.monotonic()
            if now - last_log_t >= 0.5:
                print("[stage6] fixed push left-offset=%.3f/%.3fm vy=+%.2f yaw=%.1f" % (
                    left_progress, FIXED_PUSH_LEFT_OFFSET, vy, self._yaw()))
                last_log_t = now
            time.sleep(0.05)
        self._stop()
        print("[stage6] fixed push left-offset timeout=%.3fm; keep current offset and push" % (
            FIXED_PUSH_LEFT_OFFSET,))
        return False

    def close(self):
        pass

    def _aim_ball(self, tol=AIM_TOL, timeout=AIM_TIMEOUT):
        """把球转到画面中央，允许短暂丢帧并要求连续居中确认。"""
        print("[stage6] 对准球：把球转到画面中央 ...")
        t0 = time.time()
        centered_observations = []
        last_err = None
        last_seen_t = 0.0
        last_frame = self._detector_frame()
        while time.time() - t0 < timeout:
            self._wait_detector()
            frame = self._detector_frame()
            fresh_frame = frame is None or frame != last_frame
            if fresh_frame:
                last_frame = frame
            b = self._ball()
            if b is not None and fresh_frame:
                err = b[2] - 0.5
                last_err = err
                last_seen_t = time.time()
                if abs(err) <= tol:
                    centered_observations.append(time.monotonic())
                    centered_observations = [
                        stamp for stamp in centered_observations
                        if time.monotonic() - stamp <= AIM_OBSERVATION_WINDOW
                    ]
                    self._walk(0.0, 0.0, 0.0)
                    if len(centered_observations) >= AIM_STABLE_FRAMES:
                        self._stop()
                        print("[stage6] 球已稳定居中 cx=%.2f frames=%d" % (
                            b[2], len(centered_observations)))
                        return True
                else:
                    centered_observations = []
                    wz = max(-AIM_WZ_MAX, min(AIM_WZ_MAX, -AIM_K * err))
                    self._walk(0.0, 0.0, wz)
            elif b is not None:
                self._stop()
            elif last_err is not None and time.time() - last_seen_t <= AIM_LOST_HOLD_SECS:
                hold_wz = max(-AIM_LOST_WZ_MAX, min(AIM_LOST_WZ_MAX, -0.45 * AIM_K * last_err))
                self._walk(0.0, 0.0, hold_wz)
            else:
                self._stop()
            time.sleep(0.05)
        self._stop()
        return False

    def _shift_ball_to_right(self, timeout=PREP_BEHIND_TIMEOUT):
        """面朝 +y 时，先把前方足球横移到画面右侧；此处才允许横移。"""
        t0 = time.time()
        while time.time() - t0 < timeout:
            self._wait_detector()
            ball = self._ball()
            if (ball is None or ball[1] < BALL_MIN_NORM or
                    ball[2] >= PREP_BALL_RIGHT_CX):
                self._stop()
                return True
            error = PREP_BALL_RIGHT_CX - ball[2]
            vy = min(PREP_BALL_RIGHT_VY_MAX, max(
                PREP_BALL_RIGHT_VY_MIN,
                PREP_BALL_RIGHT_VY_K * error,
            ))
            yaw_error = _wrap_deg(PREP_FACE_FWD_YAW - self._yaw())
            wz = max(-PREP_RUSH_WZ_MAX, min(PREP_RUSH_WZ_MAX, 0.03 * yaw_error))
            self._walk(0.0, vy, wz)
            now = time.monotonic()
            if now - self._last_prep_ball_guard_log_t >= PREP_BALL_RIGHT_LOG_INTERVAL:
                print("[stage6] prep ball guard: ball cx=%.2f, left-shift vy=+%.2f before +y" %
                      (ball[2], vy))
                self._last_prep_ball_guard_log_t = now
            time.sleep(0.05)
        self._stop()
        print("[stage6] prep ball guard timeout; continue with fixed ball-behind geometry")
        return False

    def _walk_forward_to(self, target_fwd, target_lat, target_yaw,
                         max_v=PREP_FORWARD_MAX_V, timeout=PREP_BEHIND_TIMEOUT):
        """保持指定朝向，只用身体前进 vx 到目标；不发送横移 vy。"""
        heading_rad = math.radians(target_yaw)
        t0 = time.time()
        last_log_t = 0.0
        reached = False
        while time.time() - t0 < timeout:
            df = target_fwd - self._fwd()
            dlat = target_lat - self._lat()
            distance = math.hypot(df, dlat)
            along = df * math.cos(heading_rad) - dlat * math.sin(heading_rad)
            if distance <= 0.07 or along <= 0.04:
                reached = distance <= 0.12
                break
            speed = min(max_v, max(0.06, 0.8 * along))
            yaw_error = _wrap_deg(target_yaw - self._yaw())
            wz = max(-PREP_RUSH_WZ_MAX, min(PREP_RUSH_WZ_MAX, 0.03 * yaw_error))
            self._walk(speed, 0.0, wz)
            now = time.monotonic()
            if now - last_log_t >= 0.5:
                print("[stage6] prep forward-to-behind dist=%.2f along=%.2f vx=%.2f vy=0.00 yaw=%.1f" % (
                    distance, along, speed, self._yaw()))
                last_log_t = now
            time.sleep(0.05)
        self._stop()
        if reached:
            print("[stage6] prep forward-to-behind reached dist=%.2f pos=(fwd=%.2f, lat=%+.2f)" % (
                math.hypot(target_fwd - self._fwd(), target_lat - self._lat()),
                self._fwd(), self._lat()))
        else:
            print("[stage6] prep forward-to-behind timeout dist=%.2f pos=(fwd=%.2f, lat=%+.2f)" % (
                math.hypot(target_fwd - self._fwd(), target_lat - self._lat()),
                self._fwd(), self._lat()))
        return reached

    def _recover_ball(self):
        """球丢失/扫不到时：原地重扫；一圈扫不到就后退几步再扫。找到后先对准再返回"""
        for i in range(2):
            if self.scan_ball():
                if self._aim_ball():
                    return True
                continue
            print("[stage6] 重扫第 %d 轮没扫到，后退 %.2f m 再扫" % (i + 1, BACKUP_STEP))
            self._walk_to(self._fwd() - BACKUP_STEP, self._lat())
        if self.scan_ball():
            self._aim_ball()
            return True
        return False

    def _initial_ball_anchor(self):
        """Use the known initial football position to correct the local yaw."""
        expected_yaw = math.degrees(math.atan2(
            -(BALL_INITIAL_LAT - START_LAT),
            BALL_INITIAL_FWD - START_FWD,
        ))
        print("[stage6] initial ball anchor world=(%.2f,%.2f) expected_yaw=%.1f; clockwise scan" % (
            BALL_INITIAL_WORLD_X, BALL_INITIAL_WORLD_Y, expected_yaw))
        t0 = time.time()
        observations = []
        last_observation_log_t = 0.0
        last_frame = self._detector_frame()
        while time.time() - t0 < INITIAL_BALL_SCAN_TIMEOUT:
            self._wait_detector()
            ball = self._ball()
            now = time.monotonic()
            frame = self._detector_frame()
            fresh_frame = frame is None or frame != last_frame
            if fresh_frame:
                last_frame = frame
            observations = [
                item for item in observations
                if now - item[0] <= INITIAL_BALL_OBSERVATION_WINDOW
            ]
            if (fresh_frame and ball is not None and
                    ball[0] >= INITIAL_BALL_HIGH_CONF and ball[1] >= BALL_MIN_NORM):
                heading = self._ball_heading_deg(ball)
                observations.append((now, heading, ball[2]))
                _mean_heading, spread = self._heading_consensus(observations)
                if now - last_observation_log_t >= 0.4:
                    print("[stage6] initial ball observation conf=%.2f heading=%+.1f spread=%.1f observations=%d/%d" % (
                        ball[0], heading, spread, len(observations), INITIAL_BALL_CONFIRM_FRAMES))
                    last_observation_log_t = now
                if len(observations) < INITIAL_BALL_CONFIRM_FRAMES or spread > INITIAL_BALL_DIRECTION_TOL_DEG:
                    self._walk(0.0, 0.0, INITIAL_BALL_SCAN_WZ)
                    time.sleep(0.05)
                    continue
                self._stop()
                mean_heading, _spread = self._heading_consensus(observations)
                old_yaw = self._yaw()
                print("[stage6] initial ball confirmed heading=%.1f; skip frame-by-frame centering" %
                      mean_heading)
                self._face_yaw(expected_yaw, tol=4.0, timeout=5.0)
                self.adp.set_mapped_pose(START_FWD, START_ADAPTER_Y, expected_yaw)
                correction = ((expected_yaw - old_yaw + 180.0) % 360.0) - 180.0
                print("[stage6] initial ball anchor applied correction=%+.1fdeg expected=%.1f" % (
                    correction,
                    expected_yaw,
                ))
                return True
            self._walk(0.0, 0.0, INITIAL_BALL_SCAN_WZ)
            time.sleep(0.05)
        self._stop()
        print("[stage6] initial ball anchor not found; retain configured start pose")
        return False

    def _capture_ramp_reference(self):
        if self.depth is None:
            return False
        for _unused in range(5):
            reference = self.depth.capture_ground_reference()
            if reference is not None:
                return True
            time.sleep(0.12)
        status = self.depth.status()
        age = status.get("frame_age")
        print("[stage6] ramp guard: no D430 ground reference; frames=%d age=%s camera_info=%s coordinate guard remains active" % (
            status.get("frames", 0),
            "n/a" if age is None else "%.2fs" % age,
            status.get("camera_info", False),
        ))
        return False

    def _report_depth_startup(self):
        if self.depth is None:
            print("[stage6_depth] disabled by --no-depth")
            return False
        ready = self.depth.wait_ready(DEPTH_STARTUP_TIMEOUT)
        status = self.depth.status()
        age = status.get("frame_age")
        print("[stage6_depth] startup ready=%s frames=%d age=%s encoding=%s size=%sx%s valid=%.0f%% median=%s camera_info=%s" % (
            ready,
            status.get("frames", 0),
            "n/a" if age is None else "%.3fs" % age,
            status.get("encoding") or "n/a",
            status.get("width") or "n/a",
            status.get("height") or "n/a",
            status.get("valid_ratio", 0.0) * 100.0,
            "n/a" if status.get("median_depth") is None else "%.2fm" % status["median_depth"],
            status.get("camera_info", False),
        ))
        if not ready:
            print("[stage6_depth] ERROR no fresh D430 frame; ball range and gap search will use guarded fallbacks")
        return ready

    def run_preparation_route(self):
        """Use only fixed field coordinates to reach the ball-behind push line."""
        print("[stage6] 固定预备：面朝 +y，先到左侧角落 world=(%.2f, %.2f)；不使用 RGB/D430/左鱼眼" % (
            _START_X + FIXED_CORNER_LAT, _START_Y + FIXED_CORNER_FWD))
        self._face_yaw(PREP_FACE_FWD_YAW)
        lateral_ok = self._walk_fixed_to(
            self._fwd(),
            FIXED_CORNER_LAT,
            max_v=PREP_CRAB_MAX_V,
            label="left-corner lateral",
            timeout=24.0,
        )
        self._face_yaw(PREP_FACE_FWD_YAW)
        corner_ok = self._walk_fixed_to(
            FIXED_CORNER_FWD,
            FIXED_CORNER_LAT,
            max_v=PREP_FORWARD_MAX_V,
            label="left-corner forward",
            timeout=20.0,
        )

        behind_fwd, behind_lat = self._behind_point(
            BALL_INITIAL_FWD,
            BALL_INITIAL_LAT,
        )
        behind_ok = self._walk_fixed_to(
            behind_fwd,
            behind_lat,
            max_v=PREP_FORWARD_MAX_V,
            label="ball-behind point",
            timeout=10.0,
        )
        push_yaw = math.degrees(math.atan2(
            -(PUSH_GAP_LAT - BALL_INITIAL_LAT),
            PUSH_GAP_FWD - BALL_INITIAL_FWD))
        print("[stage6] 固定预备：三点一线 dog->ball->gap，面向推球 yaw=%.1f" % push_yaw)
        self._face_yaw(push_yaw)
        self._side_step_fixed_push_left(push_yaw)
        self._prep_push_yaw = push_yaw
        self._last_push_yaw = push_yaw
        self._stop()
        print("[stage6] 固定预备完成（身位左偏 %.2fm）pos=(fwd=%.2f, lat=%+.2f) yaw=%.1f，下一步固定前推" % (
            FIXED_PUSH_LEFT_OFFSET,
            self._fwd(), self._lat(), self._yaw()))
        return lateral_ok and corner_ok and behind_ok

    # ---------------- 足球检测 ----------------
    def _ball(self):
        """\u8fd4\u56de\u6700\u4f73 football \u68c0\u6d4b (conf, area_norm, cx) \u6216 None"""
        best = None
        for d in self.det.get_detections():
            if d.get("class_name") != BALL_CLASS:
                continue
            conf = d.get("confidence", 0.0)
            if conf < BALL_CONF:
                continue
            area = d.get("bbox_area_norm", 0.0)
            cx = d.get("bbox_cx_norm", 0.5)
            if best is None or conf > best[0]:
                best = (conf, area, cx, d)
        return best

    def _ball_dist(self, area_norm):
        """\u9762\u79ef\u5360\u6bd4 -> \u8ddd\u79bb\uff1adist = A/sqrt(area) + B\uff08A/B \u7531 stage6_ball_calib.py \u6807\u5b9a\uff09"""
        if area_norm <= 1e-6:
            return None
        return BALL_DIST_K / math.sqrt(area_norm) + BALL_DIST_B

    def _ball_range(self, ball):
        """Prefer D430 depth near the RGB ball box; retain RGB-area fallback."""
        if ball is None:
            return None, None, None
        rgb_distance = self._ball_dist(ball[1])
        depth_result = None
        if self.depth is not None:
            depth_result = self.depth.measure(ball[3], expected_distance=rgb_distance)
        if depth_result is not None:
            bearing = math.radians((ball[2] - 0.5) * BALL_FOV_DEG)
            distance = depth_result["distance"] / max(math.cos(bearing), 0.80)
            now = time.monotonic()
            if now - self._last_range_log_t >= 0.6 or self._last_range_source != "depth":
                print("[stage6] ball range=%.2fm source=depth support=%.0f%% spread=%.2fm age=%.0fms" % (
                    distance,
                    depth_result["support"] * 100.0,
                    depth_result["spread"],
                    depth_result["age"] * 1000.0,
                ))
                self._last_range_log_t = now
                self._last_range_source = "depth"
            return distance, "depth", depth_result
        now = time.monotonic()
        if rgb_distance is not None and (
                now - self._last_range_log_t >= 1.5 or self._last_range_source != "rgb"):
            reason = "disabled"
            status = None
            if self.depth is not None:
                status = self.depth.status()
                reason = status.get("last_measure", {}).get("reason", "unknown")
            age = None if status is None else status.get("frame_age")
            print("[stage6] ball range=%.2fm source=rgb-area depth_reason=%s depth_frames=%d depth_age=%s" % (
                rgb_distance,
                reason,
                0 if status is None else status.get("frames", 0),
                "n/a" if age is None else "%.3fs" % age,
            ))
            self._last_range_log_t = now
            self._last_range_source = "rgb"
        return rgb_distance, "rgb", None

    def _confirm_prepared_ball_heading(self, timeout=4.0):
        """Use distinct RGB frames plus D430 range to finish the ball-gap line-up."""
        print("[stage6] 三点一线后用 D430 多帧球位复核；修正移动只允许 vx，vy=0")
        for pass_index in range(PREP_LINE_MAX_PASSES):
            observation = self._collect_prepared_ball_pose(timeout)
            if observation is None:
                print("[stage6] 三点一线后未取得稳定 D430 球位，才进入常规扫描")
                return False

            behind_fwd, behind_lat = self._behind_point(
                observation["fwd"], observation["lat"])
            correction_distance = math.hypot(
                behind_fwd - self._fwd(), behind_lat - self._lat())
            print("[stage6] prep line pass=%d ball=(%.2f,%+.2f) behind=(%.2f,%+.2f) correction=%.2fm depth=%.2fm" % (
                pass_index + 1,
                observation["fwd"],
                observation["lat"],
                behind_fwd,
                behind_lat,
                correction_distance,
                observation["distance"],
            ))
            if correction_distance > PREP_LINE_CORRECTION_TOL:
                correction_yaw = math.degrees(math.atan2(
                    -(behind_lat - self._lat()), behind_fwd - self._fwd()))
                print("[stage6] prep line correction: face yaw=%.1f then forward-only vx<=%.2f vy=0.00" % (
                    correction_yaw, PREP_LINE_CORRECTION_MAX_V))
                self._face_yaw(correction_yaw, tol=3.0, timeout=6.0)
                if not self._walk_forward_to(
                        behind_fwd,
                        behind_lat,
                        correction_yaw,
                        max_v=PREP_LINE_CORRECTION_MAX_V):
                    return False

            ball_yaw = math.degrees(math.atan2(
                -(observation["lat"] - self._lat()),
                observation["fwd"] - self._fwd(),
            ))
            before_turn_frame = self._detector_frame()
            print("[stage6] prep line face ball/gap yaw=%.1f; wait for fresh centered frames" % ball_yaw)
            self._face_yaw(ball_yaw, tol=2.5, timeout=6.0)
            if self._confirm_fresh_ball_center(before_turn_frame, timeout):
                self._prep_push_yaw = self._yaw()
                return True
        print("[stage6] prep line could not center ball after %d passes; use regular recovery" %
              PREP_LINE_MAX_PASSES)
        return False

    def _collect_prepared_ball_pose(self, timeout):
        observations = []
        last_frame = self._detector_frame()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._wait_detector()
            frame = self._detector_frame()
            if frame is not None and frame == last_frame:
                self._stop()
                time.sleep(0.03)
                continue
            last_frame = frame
            observation = self._ball_observation(require_depth=True)
            now = time.monotonic()
            observations = [
                item for item in observations
                if now - item["timestamp"] <= PREP_LINE_CONFIRM_WINDOW
            ]
            if observation is not None:
                observations.append(observation)
                if len(observations) >= PREP_LINE_CONFIRM_FRAMES:
                    recent = observations[-PREP_LINE_CONFIRM_FRAMES:]
                    mean_fwd = sum(item["fwd"] for item in recent) / len(recent)
                    mean_lat = sum(item["lat"] for item in recent) / len(recent)
                    spread = max(math.hypot(
                        item["fwd"] - mean_fwd,
                        item["lat"] - mean_lat,
                    ) for item in recent)
                    if spread <= PREP_LINE_POSITION_SPREAD:
                        return {
                            "fwd": mean_fwd,
                            "lat": mean_lat,
                            "distance": sum(item["distance"] for item in recent) / len(recent),
                            "spread": spread,
                        }
            self._stop()
            time.sleep(0.03)
        return None

    def _confirm_fresh_ball_center(self, after_frame, timeout):
        centered = 0
        last_frame = after_frame
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._wait_detector()
            frame = self._detector_frame()
            if frame is not None and frame == last_frame:
                self._stop()
                time.sleep(0.03)
                continue
            last_frame = frame
            ball = self._ball()
            if ball is not None and ball[1] >= BALL_MIN_NORM:
                error = ball[2] - 0.5
                if abs(error) <= PREP_LINE_CENTER_CX_TOL:
                    centered += 1
                    self._stop()
                    if centered >= AIM_STABLE_FRAMES:
                        print("[stage6] prep line confirmed ball centered cx=%.2f fresh_frames=%d yaw=%.1f" % (
                            ball[2], centered, self._yaw()))
                        return True
                else:
                    centered = 0
                    print("[stage6] prep line fresh frame ball cx=%.2f not centered; recompute line" % ball[2])
                    return False
            else:
                centered = 0
            self._stop()
            time.sleep(0.03)
        return False

    def _ball_world(self):
        """\u4f30\u7b97\u7403\u7684\u5c40\u90e8 (fwd, lat)\uff0c\u57fa\u4e8e\u72d7\u4f4d\u7f6e+\u8ddd\u79bb\u6a21\u578b+cx\u65b9\u5411"""
        observation = self._ball_observation()
        if observation is None:
            return None
        return observation["fwd"], observation["lat"]

    def _ball_heading_deg(self, ball):
        """Return the field heading from the dog to the detected football."""
        bearing = (ball[2] - 0.5) * BALL_FOV_DEG
        return _wrap_deg(self._yaw() - bearing)

    @staticmethod
    def _heading_consensus(observations):
        if not observations:
            return 0.0, 180.0
        sin_sum = sum(math.sin(math.radians(item[1])) for item in observations)
        cos_sum = sum(math.cos(math.radians(item[1])) for item in observations)
        mean_heading = math.degrees(math.atan2(sin_sum, cos_sum))
        spread = max(abs(_wrap_deg(item[1] - mean_heading)) for item in observations)
        return mean_heading, spread

    def _ball_observation(self, require_depth=False):
        ball = self._ball()
        if ball is None:
            return None
        distance, source, detail = self._ball_range(ball)
        if distance is None or (require_depth and source != "depth"):
            return None
        yaw = math.radians(self._yaw())
        bearing = math.radians((ball[2] - 0.5) * BALL_FOV_DEG)
        fwd, adapter_y, _z = self._pos()
        heading = yaw - bearing
        ball_fwd = fwd + distance * math.cos(heading)
        ball_lat = -adapter_y - distance * math.sin(heading)
        observation = {
            "fwd": ball_fwd,
            "lat": ball_lat,
            "distance": distance,
            "source": source,
            "detail": detail,
            "cx": ball[2],
            "confidence": ball[0],
            "timestamp": time.monotonic(),
        }
        self._last_ball_position = dict(observation)
        return observation

    def _recent_ball_world(self, phase):
        """Use a current observation or a short-lived ball position cache for motion only."""
        ball_world = self._ball_world()
        if ball_world is not None:
            return ball_world, False
        cached = self._last_ball_position
        if cached is None:
            return None, False
        age = time.monotonic() - cached["timestamp"]
        if age > BALL_POSITION_HOLD_SECS:
            print("[stage6] %s: last ball position expired age=%.1fs; cannot continue push" % (
                phase, age))
            return None, False
        print("[stage6] %s: ball briefly left frame; use %.1fs-old %s position world=(%.2f,%.2f) for line-up only" % (
            phase,
            age,
            cached["source"],
            _START_X + cached["lat"],
            _START_Y + cached["fwd"],
        ))
        return (cached["fwd"], cached["lat"]), True

    def _update_ball_track(self):
        """Fuse RGB football identity with the newest D430 range measurement."""
        now = time.monotonic()
        observation = self._ball_observation(require_depth=True)
        if observation is not None:
            previous = self._ball_track
            if previous is not None and now - previous["timestamp"] <= 1.0:
                alpha = 0.35
                for key in ("fwd", "lat", "distance"):
                    observation[key] = (
                        alpha * observation[key] + (1.0 - alpha) * previous[key]
                    )
            observation["timestamp"] = now
            self._ball_track = observation
            return observation
        if self._ball_track is not None and now - self._ball_track["timestamp"] <= BALL_TRACK_HOLD_SECS:
            return self._ball_track
        return None

    def _log_ball_track(self, track, target_fwd, target_lat, phase):
        now = time.monotonic()
        if now - self._last_ball_track_log_t < BALL_TRACK_LOG_INTERVAL:
            return
        marker = " center-right" if track["cx"] >= PREP_BALL_RIGHT_CX else ""
        print("[stage6] ball-track %s depth fwd=%.2f lat=%+.2f range=%.2fm cx=%.2f%s target=(%.2f,%+.2f)" % (
            phase,
            track["fwd"],
            track["lat"],
            track["distance"],
            track["cx"],
            marker,
            target_fwd,
            target_lat,
        ))
        self._last_ball_track_log_t = now

    def _behind_point(self, ball_fwd, ball_lat):
        dx = ball_fwd - PUSH_GAP_FWD
        dy = ball_lat - PUSH_GAP_LAT
        norm = math.hypot(dx, dy)
        if norm < 1e-3:
            norm = 1.0
        unit_fwd = dx / norm
        unit_lat = dy / norm
        behind_distance = BALL_BEHIND_DIST
        low_fwd, high_fwd = FIELD_FWD_MIN + 0.1, FIELD_FWD_MAX - 0.1
        low_lat, high_lat = FIELD_LAT_MIN + 0.1, FIELD_LAT_MAX - 0.1
        for coordinate, direction, low, high in (
                (ball_fwd, unit_fwd, low_fwd, high_fwd),
                (ball_lat, unit_lat, low_lat, high_lat)):
            if direction > 1e-6:
                behind_distance = min(behind_distance, max(0.0, (high - coordinate) / direction))
            elif direction < -1e-6:
                behind_distance = min(behind_distance, max(0.0, (low - coordinate) / direction))
        behind_fwd = ball_fwd + unit_fwd * behind_distance
        behind_lat = ball_lat + unit_lat * behind_distance
        return behind_fwd, behind_lat

    # ---------------- 流程 ----------------
    def scan_ball(self):
        """\u8f6c\u5708\u626b\u63cf\u627e\u7403\uff0c\u627e\u5230\u8fd4\u56de True"""
        print("[stage6] \u626b\u63cf\u627e\u7403 ...")
        t0 = time.time()
        while time.time() - t0 < SCAN_TIMEOUT:
            self._wait_detector()
            b = self._ball()
            if b is not None and b[1] >= BALL_MIN_NORM:
                distance, source, _detail = self._ball_range(b)
                if distance is None:
                    print("[stage6] \u627e\u5230\u7403 area_norm=%.4f cx=%.2f" % (b[1], b[2]))
                else:
                    print("[stage6] \u627e\u5230\u7403 area_norm=%.4f cx=%.2f range=%.2fm(%s)" % (
                        b[1], b[2], distance, source))
                self._stop()
                return True
            self._walk(0.0, 0.0, SCAN_WZ)
            time.sleep(0.05)
        self._stop()
        print("[stage6] \u626b\u63cf\u8d85\u65f6\u672a\u627e\u5230\u7403")
        return False

    def center_ball(self, area_thresh, vx, wz_max, k_cx, timeout=8.0,
                    overall_timeout=APPROACH_OVERALL_TIMEOUT):
        """走向球并保持居中，直到 area_norm>=area_thresh；8s 没到位不结束：
        原地重扫/后退重扫后继续接近（反复找不到球则返回 False，由上层按球已推出处理）"""
        t0 = time.time()
        while time.time() - t0 < overall_timeout:
            win_t0 = time.time()
            last_area = -1
            while time.time() - win_t0 < timeout:
                self._wait_detector()
                b = self._ball()
                if b is not None:
                    area, cx = b[1], b[2]
                    distance, source, _detail = self._ball_range(b)
                    close_distance = self._ball_dist(area_thresh)
                    close_by_depth = (
                        source == "depth"
                        and distance is not None
                        and close_distance is not None
                        and distance <= close_distance
                    )
                    if area >= area_thresh or close_by_depth:
                        print("[stage6] \u63a5\u8fd1\u5230\u4f4d area_norm=%.4f range=%.2fm(%s)" % (
                            area, distance if distance is not None else -1.0, source or "none"))
                        self._stop()
                        return True
                    last_area = area
                    # 接近时越近越慢
                    approach_distance = self._ball_dist(BALL_APPROACH_NORM)
                    slow_by_depth = (
                        source == "depth"
                        and distance is not None
                        and approach_distance is not None
                        and distance <= approach_distance
                    )
                    spd = APPROACH_VX_SLOW if area >= BALL_APPROACH_NORM or slow_by_depth else APPROACH_VX
                    wz = max(-wz_max, min(wz_max, -k_cx * (cx - 0.5)))
                    self._walk(spd, 0.0, wz)
                else:
                    self._stop()
                    time.sleep(0.1)
                time.sleep(0.05)
            # 一个 8s 窗口没到位：不结束，重扫后继续接近
            print("[stage6] 接近窗口(%.0fs)未到位 last_area=%.4f，重扫后继续接近" % (timeout, last_area))
            self._stop()
            if not self._recover_ball():
                print("[stage6] 接近阶段反复找不到球")
                break
        self._stop()
        return False

    def go_behind(self, ball_fwd, ball_lat):
        """\u7ed5\u5230\u7403\u540e\uff1abehind\u70b9 = \u7403 + (ball-gap)\u5355\u4f4d\u5411\u91cf * BEHIND_DIST\uff08\u5373\u80cc\u5bf9\u7f3a\u53e3\u4fa7\uff09"""
        behind_fwd, behind_lat = self._behind_point(ball_fwd, ball_lat)
        print("[stage6] \u7ed5\u5230\u7403\u540e (%+.2f, %+.2f)" % (behind_lat, behind_fwd))
        self._walk_to(behind_fwd, behind_lat, behind_ball=True)
        # \u9762\u5411\u7f3a\u53e3\uff08\u63a8\u7403\u65b9\u5411\uff09
        gap_yaw = math.degrees(math.atan2(
            -(PUSH_GAP_LAT - self._lat()),
            PUSH_GAP_FWD - self._fwd(),
        ))
        self._face_yaw(gap_yaw)
        self._last_push_yaw = gap_yaw
        return True

    def _push_line_status(self, ball_fwd, ball_lat):
        gap_fwd = PUSH_GAP_FWD - ball_fwd
        gap_lat = PUSH_GAP_LAT - ball_lat
        gap_norm = math.hypot(gap_fwd, gap_lat)
        if gap_norm < 1e-3:
            return {
                "ready": False,
                "behind": 0.0,
                "cross": float("inf"),
                "distance": 0.0,
            }
        behind_unit_fwd = -gap_fwd / gap_norm
        behind_unit_lat = -gap_lat / gap_norm
        dog_fwd = self._fwd() - ball_fwd
        dog_lat = self._lat() - ball_lat
        behind = dog_fwd * behind_unit_fwd + dog_lat * behind_unit_lat
        cross = abs(dog_fwd * behind_unit_lat - dog_lat * behind_unit_fwd)
        distance = math.hypot(dog_fwd, dog_lat)
        return {
            "ready": behind >= PUSH_LINE_MIN_BEHIND and cross <= PUSH_LINE_MAX_CROSS,
            "behind": behind,
            "cross": cross,
            "distance": distance,
        }

    def _lineup_for_push(self, ball_fwd, ball_lat, phase, allow_cached_ball=False):
        status = self._push_line_status(ball_fwd, ball_lat)
        print("[stage6] %s line-check behind=%.2fm cross=%.2fm ball_dist=%.2fm ready=%s" % (
            phase,
            status["behind"],
            status["cross"],
            status["distance"],
            status["ready"],
        ))
        if not status["ready"]:
            print("[stage6] %s: 狗不在球后共线上，才执行绕后" % phase)
            return self.go_behind(ball_fwd, ball_lat)

        ball_yaw = math.degrees(math.atan2(
            -(ball_lat - self._lat()), ball_fwd - self._fwd()))
        before_turn_frame = self._detector_frame()
        print("[stage6] %s: 已在球后共线上，不绕后；对准球 yaw=%.1f 后直接推" % (
            phase, ball_yaw))
        self._face_yaw(ball_yaw, tol=2.5, timeout=6.0)
        self._last_push_yaw = ball_yaw
        if allow_cached_ball:
            print("[stage6] %s: use cached ball position; yaw set, continue visual push without another wait" % phase)
            return True
        if self._confirm_fresh_ball_center(before_turn_frame, timeout=3.0):
            return True
        print("[stage6] %s: 新画面未确认居中，执行小角度视觉对准，不改变站位" % phase)
        if self._aim_ball(tol=PREP_LINE_CENTER_CX_TOL, timeout=4.0):
            return True
        return False

    def _fixed_push_target(self):
        """Return the dog position behind a ball already pushed past the gap."""
        line_fwd = PUSH_GAP_FWD - BALL_INITIAL_FWD
        line_lat = PUSH_GAP_LAT - BALL_INITIAL_LAT
        line_norm = math.hypot(line_fwd, line_lat)
        if line_norm < 1e-6 or abs(line_lat) < 1e-6:
            raise RuntimeError("[stage6] invalid fixed ball-to-gap geometry")
        unit_fwd = line_fwd / line_norm
        unit_lat = line_lat / line_norm
        ball_exit_lat = BALL_OUT_LAT + FIXED_PUSH_OUT_MARGIN
        travel = (ball_exit_lat - BALL_INITIAL_LAT) / unit_lat
        ball_exit_fwd = BALL_INITIAL_FWD + unit_fwd * travel
        full_fwd = ball_exit_fwd - unit_fwd * BALL_BEHIND_DIST
        full_lat = ball_exit_lat - unit_lat * BALL_BEHIND_DIST
        behind_fwd, behind_lat = self._behind_point(
            BALL_INITIAL_FWD, BALL_INITIAL_LAT)
        dog_fwd = behind_fwd + FIXED_FIRST_PUSH_RATIO * (full_fwd - behind_fwd)
        dog_lat = behind_lat + FIXED_FIRST_PUSH_RATIO * (full_lat - behind_lat)
        return dog_fwd, dog_lat

    def push_ball_fixed_line(self):
        """Push from the fixed behind point through the fixed ball-to-gap line."""
        target_fwd, target_lat = self._fixed_push_target()
        push_yaw = self._prep_push_yaw
        if push_yaw is None:
            push_yaw = math.degrees(math.atan2(
                -(PUSH_GAP_LAT - BALL_INITIAL_LAT),
                PUSH_GAP_FWD - BALL_INITIAL_FWD,
            ))
        print("[stage6] 固定首推：按原计划 %.0f%% 距离到 dog=(world x=%.2f, y=%.2f) yaw=%.1f；不使用 RGB/D430" % (
            FIXED_FIRST_PUSH_RATIO * 100.0,
            _START_X + target_lat, _START_Y + target_fwd, push_yaw))
        self._face_yaw(push_yaw, tol=3.0, timeout=6.0)
        self._last_push_yaw = push_yaw
        heading_rad = math.radians(push_yaw)
        deadline = time.monotonic() + FIXED_PUSH_TIMEOUT
        last_log_t = 0.0
        while time.monotonic() < deadline:
            df = target_fwd - self._fwd()
            dlat = target_lat - self._lat()
            along = df * math.cos(heading_rad) - dlat * math.sin(heading_rad)
            cross = df * math.sin(heading_rad) + dlat * math.cos(heading_rad)
            if along <= 0.05:
                self._stop()
                print("[stage6] 固定推球到达线终点 fwd=%.2f lat=%+.2f cross=%+.2f" % (
                    self._fwd(), self._lat(), cross))
                return True
            speed = min(FIXED_PUSH_MAX_V, max(0.10, 0.8 * along))
            yaw_error = _wrap_deg(push_yaw - self._yaw())
            wz = max(-PUSH_WZ_MAX, min(PUSH_WZ_MAX, 0.03 * yaw_error))
            self.adp.walk(speed, 0.0, wz, pitch=PUSH_PITCH)
            now = time.monotonic()
            if now - last_log_t >= 0.5:
                print("[stage6] fixed push along=%.2f cross=%+.2f vx=%.2f vy=0.00 yaw=%.1f" % (
                    along, cross, speed, self._yaw()))
                last_log_t = now
            time.sleep(0.05)
        self._stop()
        print("[stage6] 固定推球超时 fwd=%.2f lat=%+.2f；随后用视觉/D430确认球是否出界" % (
            self._fwd(), self._lat()))
        return False

    def _ball_out_confirmed(self, phase):
        """Confirm exit only from a current RGB-associated ball position estimate."""
        observation = self._ball_observation()
        if observation is None:
            now = time.monotonic()
            if now - self._last_ball_out_log_t >= 0.8:
                print("[stage6] %s: 未取得当前足球位置，不能确认出界" % phase)
                self._last_ball_out_log_t = now
            return False
        world_x = _START_X + observation["lat"]
        world_y = _START_Y + observation["fwd"]
        if observation["lat"] > BALL_OUT_LAT:
            print("[stage6] %s: 球出界已确认 world=(%.2f,%.2f) source=%s；允许进入缺口" % (
                phase, world_x, world_y, observation["source"]))
            return True
        now = time.monotonic()
        if now - self._last_ball_out_log_t >= 0.8:
            print("[stage6] %s: 球仍在场内 world=(%.2f,%.2f) source=%s；禁止进入缺口" % (
                phase, world_x, world_y, observation["source"]))
            self._last_ball_out_log_t = now
        return False

    def _push_once(self):
        """单次推球：球估算 lat>BALL_OUT_LAT 返回 'out'；超时/卡住/连续丢球返回对应原因"""
        print("[stage6] 推球（俯角=%.2frad） ..." % PUSH_PITCH)
        t0 = time.time()
        last_fwd = self._fwd()
        last_t = time.time()
        accel_end = time.time() + PUSH_ACCEL_SECS
        lost_frames = 0
        while time.time() - t0 < PUSH_TIMEOUT:
            self._wait_detector()
            f = self._fwd()
            if self._ball_out_confirmed("push"):
                self._stop()
                return "out"
            # 推球速度：先 0.6 再加速到 0.8
            vx = PUSH_VX_FAST if time.time() >= accel_end else PUSH_VX
            b = self._ball()
            if b is not None:
                lost_frames = 0
                cx = b[2]
                wz = max(-PUSH_WZ_MAX, min(PUSH_WZ_MAX, -PUSH_K_CX * (cx - 0.5)))
                self._walk(vx, 0.0, wz, pitch=PUSH_PITCH)
            else:
                lost_frames += 1
                if lost_frames >= PUSH_LOST_FRAMES:
                    print("[stage6] 球连续 %d 帧检测不到（可能太近出画面），停止重扫" % lost_frames)
                    self._stop()
                    return "lost"
                # 短暂丢帧：保持最后方向继续走
                self._walk(vx, 0.0, 0.0, pitch=PUSH_PITCH)
            # 卡住判断：fwd 停滞
            if abs(f - last_fwd) < 0.01 and time.time() - last_t > PUSH_NO_PROGRESS:
                print("[stage6] 推球卡住（fwd 停滞 %.1fs），停止" % PUSH_NO_PROGRESS)
                self._stop()
                return "stall"
            if abs(f - last_fwd) >= 0.01:
                last_fwd, last_t = f, time.time()
            time.sleep(0.05)
        self._stop()
        print("[stage6] 推球超时（%.0fs）" % PUSH_TIMEOUT)
        return "timeout"

    def _recover_push(self, reason):
        """Recover a failed visual push without treating a lost ball as an exit."""
        if reason in ("lost", "timeout"):
            self._blind_push_from_anchor(reason)
            self._wait_detector()
            if self._ball_out_confirmed("blind-push"):
                return "out"
        if reason == "stall":
            print("[stage6] 卡住恢复：反方向后退 %.2f m" % BACKUP_STEP)
            self._walk_to(self._fwd() - BACKUP_STEP, self._lat())
        b = self._ball()
        if b is not None:
            if self._ball_out_confirmed("push-recovery-visible"):
                return "out"
            self._aim_ball()
            bw = self._ball_world()
            if bw is not None and self._lineup_for_push(
                    bw[0], bw[1], "push-recovery-visible"):
                return "ready"
        for i in range(2):
            print("[stage6] 恢复：后退 %.2f m 后重扫（第 %d/2 轮）" % (BACKUP_STEP, i + 1))
            self._walk_to(self._fwd() - BACKUP_STEP, self._lat())
            if self.scan_ball():
                if self._ball_out_confirmed("push-recovery-rescan"):
                    return "out"
                self._aim_ball()
                bw = self._ball_world()
                if bw is not None and self._lineup_for_push(
                        bw[0], bw[1], "push-recovery-rescan"):
                    return "ready"
        print("[stage6] 反复找不到球，未确认球出界；禁止进入缺口")
        return "failed"

    def _blind_push_from_anchor(self, reason):
        """Continue a short push on the most recently confirmed push line."""
        push_yaw = self._last_push_yaw
        if push_yaw is None:
            push_yaw = math.degrees(math.atan2(
                -(PUSH_GAP_LAT - BALL_INITIAL_LAT),
                PUSH_GAP_FWD - BALL_INITIAL_FWD,
            ))
        print("[stage6] 球丢失原因=%s；不后退不重扫，沿最近确认的三点一线 yaw=%.1f 盲推 %.1fs" %
              (reason, push_yaw, PUSH_LOST_BLIND_SECS))
        self._face_yaw(push_yaw, tol=4.0, timeout=5.0)
        deadline = time.monotonic() + PUSH_LOST_BLIND_SECS
        last_log_t = 0.0
        while time.monotonic() < deadline:
            self._walk(PUSH_VX, 0.0, 0.0, pitch=PUSH_PITCH)
            now = time.monotonic()
            if now - last_log_t >= 0.5:
                print("[stage6] blind push vx=%.2f vy=0.00 yaw=%.1f remaining=%.1fs" % (
                    PUSH_VX, self._yaw(), max(0.0, deadline - now)))
                last_log_t = now
            time.sleep(0.05)
        self._stop()
        print("[stage6] blind push complete；重新用视觉/D430确认球位置")

    def push_ball(self):
        """Repeat visual pushes until the ball is positively confirmed outside the gap."""
        for attempt in range(1, PUSH_MAX_ATTEMPTS + 1):
            print("[stage6] 推球尝试 %d/%d" % (attempt, PUSH_MAX_ATTEMPTS))
            if self._ball_out_confirmed("push-precheck"):
                return True
            reason = self._push_once()
            if reason == "out":
                return True
            self._stop()
            recovery = self._recover_push(reason)
            if recovery == "out":
                return True
            if recovery != "ready":
                print("[stage6] 推球恢复失败，球未确认出界；停止在场内")
                return False
        print("[stage6] 推球 %d 次仍未确认球出界；停止在场内" % PUSH_MAX_ATTEMPTS)
        return False

    def confirm_and_push_until_out(self):
        """Resume RGB+D430 pushing after the fixed first push."""
        print("[stage6] 固定首推结束；现在启用 YOLO+D430 确认球状态")
        self._wait_detector()
        if self._ball_out_confirmed("fixed-push postcheck"):
            return True
        if not self.scan_ball():
            print("[stage6] 固定首推后未找到球，不能把球丢失当作出界")
            return False
        initial_ball_world = self._ball_world()
        if initial_ball_world is None:
            print("[stage6] post-fixed scan had no usable range; keep trying visual alignment")
        if not self._aim_ball():
            print("[stage6] 固定首推后未稳定居中；保留最新 RGB/D430 定位继续判断共线")
        if self._ball_out_confirmed("fixed-push visual-check"):
            return True
        ball_world, used_cached_ball = self._recent_ball_world("post-fixed-push")
        if ball_world is None:
            print("[stage6] 固定首推后无法由 RGB/D430定位足球，停止在场内")
            return False
        if not self._lineup_for_push(
                ball_world[0], ball_world[1], "post-fixed-push",
                allow_cached_ball=used_cached_ball):
            print("[stage6] 固定首推后未能建立球后共线，停止在场内")
            return False
        return self.push_ball()

    def to_end_circle(self):
        return self._finish_in_known_gap()

    def _finish_in_known_gap(self):
        """Reach the endpoint through a fixed, aligned right-side crossing."""
        print("[stage6] 终点：斜走至 world=(%.2f, %.2f)，再面向 +x 穿过 x=%.2f" % (
            EXIT_ALIGN_WORLD_X, END_WORLD_Y, EXIT_CROSS_WORLD_X))
        if not self._walk_fixed_to(
                END_FWD,
                EXIT_ALIGN_LAT,
                max_v=END_STAGE_MAX_V,
                label="end-stage diagonal",
                timeout=18.0):
            print("[stage6] 终点对齐点未到达，不执行趴下")
            return False
        if not self._face_yaw(GAP_EDGE_YAW, tol=2.5, timeout=8.0):
            print("[stage6] 无法面向 +x，不穿过右侧开口")
            return False
        print("[stage6] 终点：已在 y=%.2f 且面向 +x；只前进到 world=(%.2f, %.2f)" % (
            END_WORLD_Y, END_WORLD_X, END_WORLD_Y))
        if not self._walk_forward_to(
                END_FWD,
                END_LAT,
                GAP_EDGE_YAW,
                max_v=END_CROSS_MAX_V,
                timeout=10.0):
            print("[stage6] 终点前进未到达，不执行趴下")
            return False
        self._stop()
        print("[stage6] 已到终点 world=(%.2f, %.2f)，执行趴下" % (
            _START_X + self._lat(), _START_Y + self._fwd()))
        self.adp.lie_down()
        time.sleep(1.0)
        return True

    def _walk_gap(self, vx, vy, wz):
        """Use the coordinate boundary guard, but not the ramp classifier."""
        vx, vy = self._field_ramp_guard(vx, vy)
        self.adp.walk(vx, vy, wz)

    def _follow_known_gap(self):
        """Use D430 only for edge-distance corrections while y follows a fixed target."""
        if self.depth is None or not self.depth.wait_ready(1.5):
            return False
        print("[stage6] 终点：面向右边界，D430 实时保持 %.2fm；按坐标 fwd=%.2f 寻找缺口" % (
            GAP_EDGE_TARGET_DISTANCE, GAP_FWD))
        self._face_yaw(GAP_EDGE_YAW, tol=3.0, timeout=8.0)
        deadline = time.monotonic() + GAP_EDGE_SEARCH_TIMEOUT
        last_log_t = 0.0
        last_edge_distance = None
        open_latched = False
        while time.monotonic() < deadline:
            clearance = self.depth.front_clearance()
            status = self.depth.status()
            fwd_error = GAP_FWD - self._fwd()
            if abs(fwd_error) <= 0.05:
                self._stop()
                print("[stage6_depth] gap y-coordinate reached fwd=%.2f lat=%+.2f open_latched=%s" % (
                    self._fwd(), self._lat(), open_latched))
                return True

            distance = None if clearance is None else clearance["distance"]
            valid_ratio = 0.0 if clearance is None else clearance["valid_ratio"]
            in_gap_window = (
                GAP_FWD_MIN - GAP_EDGE_COORD_MARGIN <= self._fwd() <=
                GAP_FWD_MAX + GAP_EDGE_COORD_MARGIN
            )
            normal_v = 0.0
            if distance is not None:
                if in_gap_window and distance >= GAP_EDGE_OPEN_DISTANCE:
                    if not open_latched:
                        print("[stage6_depth] gap opening observed fwd=%.2f edge=%.2fm；锁存位置，后续丢帧不改道" % (
                            self._fwd(), distance))
                    open_latched = True
                else:
                    last_edge_distance = distance
                    normal_v = max(-GAP_EDGE_MAX_NORMAL_V, min(
                        GAP_EDGE_MAX_NORMAL_V,
                        GAP_EDGE_DISTANCE_K * (distance - GAP_EDGE_TARGET_DISTANCE),
                    ))
            elif not status.get("ready"):
                # A late frame removes only this correction; coordinate travel continues.
                normal_v = 0.0

            along_v = max(-GAP_EDGE_FOLLOW_SPEED, min(
                GAP_EDGE_FOLLOW_SPEED, 0.8 * fwd_error))
            yaw_error = _wrap_deg(GAP_EDGE_YAW - self._yaw())
            wz = max(-0.18, min(0.18, 0.03 * yaw_error))
            self._walk_gap(normal_v, along_v, wz)
            now = time.monotonic()
            if now - last_log_t >= 0.5:
                print("[stage6_depth] gap-follow fwd=%.2f target_fwd=%.2f lat=%+.2f edge=%s last=%.2f valid=%.0f%% vx=%+.2f vy=%+.2f open_latched=%s" % (
                    self._fwd(), GAP_FWD, self._lat(),
                    "open" if distance is None else "%.2fm" % distance,
                    -1.0 if last_edge_distance is None else last_edge_distance,
                    valid_ratio * 100.0, normal_v, along_v, open_latched))
                last_log_t = now
            time.sleep(0.05)
        self._stop()
        print("[stage6_depth] gap y-coordinate timeout; use direct entry coordinate")
        return False

    def _enter_gap(self):
        """Cross into the known gap while retaining D430 distance micro-corrections."""
        print("[stage6] 进入缺口：目标 world=(%.2f, %.2f)，D430 仅作实时微调" % (
            _START_X + GAP_ENTRY_LAT, _START_Y + GAP_FWD))
        self._face_yaw(GAP_EDGE_YAW, tol=3.0, timeout=6.0)
        deadline = time.monotonic() + GAP_ENTRY_TIMEOUT
        last_log_t = 0.0
        while time.monotonic() < deadline:
            fwd_error = GAP_FWD - self._fwd()
            lat_error = GAP_ENTRY_LAT - self._lat()
            if lat_error <= 0.05 and abs(fwd_error) <= 0.07:
                self._stop()
                return True
            clearance = self.depth.front_clearance() if self.depth is not None else None
            distance = None if clearance is None else clearance["distance"]
            valid_ratio = 0.0 if clearance is None else clearance["valid_ratio"]
            vx = min(GAP_ENTRY_MAX_V, max(0.05, 0.7 * max(lat_error, 0.0)))
            vy = max(-GAP_EDGE_FOLLOW_SPEED, min(
                GAP_EDGE_FOLLOW_SPEED, 0.8 * fwd_error))
            if distance is not None and distance < GAP_EDGE_TARGET_DISTANCE:
                vx = min(vx, 0.06)
            yaw_error = _wrap_deg(GAP_EDGE_YAW - self._yaw())
            wz = max(-0.18, min(0.18, 0.03 * yaw_error))
            self._walk_gap(vx, vy, wz)
            now = time.monotonic()
            if now - last_log_t >= 0.5:
                print("[stage6_depth] gap-entry fwd=%.2f lat=%+.2f err=(%+.2f,%+.2f) edge=%s valid=%.0f%% vx=%.2f vy=%+.2f" % (
                    self._fwd(), self._lat(), fwd_error, lat_error,
                    "open" if distance is None else "%.2fm" % distance,
                    valid_ratio * 100.0, vx, vy))
                last_log_t = now
            time.sleep(0.05)
        self._stop()
        print("[stage6] 进入缺口超时 fwd=%.2f lat=%+.2f" % (
            self._fwd(), self._lat()))
        return False

    def _legacy_depth_follow_to_gap(self):
        """Face the right boundary, follow it, and identify the open gap."""
        if self.depth is None or not self.depth.wait_ready(1.5):
            return False
        print("[stage6] 终点：面向右边界，用前深度保持 %.2fm 并贴边寻找缺口" %
              GAP_EDGE_TARGET_DISTANCE)
        self._face_yaw(GAP_EDGE_YAW, tol=3.0, timeout=8.0)
        deadline = time.monotonic() + GAP_EDGE_SEARCH_TIMEOUT
        stable_open = 0
        last_log_t = 0.0
        saw_clearance = False
        search_targets = [
            GAP_FWD,
            GAP_FWD_MIN + 0.06,
            GAP_FWD_MAX - 0.06,
            GAP_FWD,
        ]
        search_index = 0
        target_reached_t = None
        while time.monotonic() < deadline:
            clearance = self.depth.front_clearance()
            status = self.depth.status()
            if not status.get("ready"):
                self._stop()
                print("[stage6_depth] gap search lost depth frame age=%s" % (
                    "n/a" if status.get("frame_age") is None else
                    "%.2fs" % status["frame_age"]))
                return False
            if clearance is None:
                self._stop()
                time.sleep(0.05)
                continue
            saw_clearance = True
            distance = clearance["distance"]
            in_gap_window = (
                GAP_FWD_MIN - GAP_EDGE_COORD_MARGIN <= self._fwd() <=
                GAP_FWD_MAX + GAP_EDGE_COORD_MARGIN
            )
            open_depth = (
                distance is None or
                clearance["valid_ratio"] <= GAP_EDGE_OPEN_VALID_RATIO or
                distance >= GAP_EDGE_OPEN_DISTANCE
            )
            stable_open = stable_open + 1 if in_gap_window and open_depth else 0
            if stable_open >= GAP_EDGE_OPEN_STABLE_FRAMES:
                self._stop()
                print("[stage6_depth] gap confirmed fwd=%.2f distance=%s valid=%.0f%% frames=%d" % (
                    self._fwd(),
                    "open" if distance is None else "%.2fm" % distance,
                    clearance["valid_ratio"] * 100.0,
                    stable_open,
                ))
                return True

            search_fwd = search_targets[search_index]
            fwd_error = search_fwd - self._fwd()
            if abs(fwd_error) <= 0.035 and not open_depth:
                if target_reached_t is None:
                    target_reached_t = time.monotonic()
                elif time.monotonic() - target_reached_t >= 0.45:
                    search_index = min(search_index + 1, len(search_targets) - 1)
                    search_fwd = search_targets[search_index]
                    fwd_error = search_fwd - self._fwd()
                    target_reached_t = None
            else:
                target_reached_t = None
            along_v = max(-GAP_EDGE_FOLLOW_SPEED, min(
                GAP_EDGE_FOLLOW_SPEED, 0.8 * fwd_error))
            normal_v = 0.0
            if distance is not None and distance < GAP_EDGE_OPEN_DISTANCE:
                normal_v = max(-GAP_EDGE_MAX_NORMAL_V, min(
                    GAP_EDGE_MAX_NORMAL_V,
                    GAP_EDGE_DISTANCE_K * (distance - GAP_EDGE_TARGET_DISTANCE),
                ))
            yaw_error = _wrap_deg(GAP_EDGE_YAW - self._yaw())
            wz = max(-0.18, min(0.18, 0.03 * yaw_error))
            self._walk(normal_v, along_v, wz)
            now = time.monotonic()
            if now - last_log_t >= 0.5:
                print("[stage6_depth] gap-follow fwd=%.2f target_fwd=%.2f lat=%+.2f edge=%s valid=%.0f%% vx=%+.2f vy=%+.2f open=%d/%d" % (
                    self._fwd(), search_fwd, self._lat(),
                    "open" if distance is None else "%.2fm" % distance,
                    clearance["valid_ratio"] * 100.0,
                    normal_v, along_v, stable_open,
                    GAP_EDGE_OPEN_STABLE_FRAMES,
                ))
                last_log_t = now
            time.sleep(0.05)
        self._stop()
        if saw_clearance:
            print("[stage6_depth] gap search timeout; use coordinate fallback")
        return False

    # ---------------- \u6574\u6bb5 ----------------
    def run(self):
        print("[stage6] ==== 第六赛段开始：固定首推后恢复 YOLO+D430 推球闭环 ====")
        prepared = self.run_preparation_route()
        if not prepared:
            print("[stage6] 固定预备路线未完全到位，仍按固定三点一线继续前推")
        fixed_push_ok = self.push_ball_fixed_line()
        if self.assume_out_after_fixed_push:
            print("[stage6] 固定首推完成 ok=%s；--assume-out 已启用，直接前往终点" % (
                fixed_push_ok,))
            finished = self.to_end_circle()
            print("[stage6] ==== 第六赛段完成 ====")
            return finished
        print("[stage6] 固定首推完成 ok=%s；继续 YOLO+D430 推球确认" % fixed_push_ok)
        self._report_depth_startup()
        self._capture_ramp_reference()
        if not self.confirm_and_push_until_out():
            self._stop()
            print("[stage6] 足球未确认出界；不进入缺口、不趴下")
            return False
        finished = self.to_end_circle()
        print("[stage6] ==== 第六赛段完成 ====")
        return finished


def main():
    ap = argparse.ArgumentParser(description="\u7b2c\u516d\u8d5b\u6bb5\u771f\u673a\uff08\u8db3\u7403\u63a8\u51fa\u7f3a\u53e3\uff09")
    ap.add_argument("--det-port", type=int, default=9890)
    ap.add_argument("--depth-topic", default=DEFAULT_DEPTH_TOPIC,
                    help="D430 depth topic (default: %(default)s)")
    ap.add_argument("--no-depth", action="store_true",
                    help="disable D430 ball-range confirmation")
    ap.add_argument("--pc-host", default=DEFAULT_STREAM_PC_HOST,
                    help="PC running stage6_yolo.py (default: %(default)s)")
    ap.add_argument("--stream-port", type=int, default=DEFAULT_STREAM_PORT,
                    help="PC YOLO stream port (default: %(default)s)")
    ap.add_argument("--stream-fps", type=float, default=DEFAULT_STREAM_FPS,
                    help="RGB stream FPS (default: %(default)s)")
    ap.add_argument("--no-stream", action="store_true",
                    help="do not start stream_images.py (manual stream debugging only)")
    ap.add_argument("--no-wait-detector", action="store_true")
    ap.add_argument("--assume-out", action="store_true",
                    help="固定首推完成后直接前往终点，不再确认足球是否出界")
    ap.add_argument("--skip-odom-wait", action="store_true",
                    help="assume LCM odometry is already available")
    ap.add_argument("--test", choices=["full", "scan", "approach", "behind", "push"], default="full",
                    help="分步测试：scan=只扫球 / approach=扫球+接近 / behind=扫球+接近+绕后面向缺口(不推球) / push=到推球为止 / full=整段")
    args = ap.parse_args()

    adapter = RealDogAdapter(None)
    det = RemoteDetector(port=args.det_port, stale_timeout=DETECTOR_STALE_TIMEOUT)
    depth = None
    if not args.no_depth:
        ensure_depth_camera(args.depth_topic)
        depth = DepthBallTracker(args.depth_topic)
    runner = Stage6Real(
        adapter,
        det,
        depth_tracker=depth,
        assume_out_after_fixed_push=args.assume_out,
    )
    stream_process = None
    try:
        if not args.no_stream:
            stream_process = _start_rgb_stream(
                args.pc_host,
                args.stream_port,
                args.stream_fps,
            )
        elif args.test == "full":
            print("[stage6] full 模式使用固定坐标推球；不启动 RGB 推流")
        if not args.skip_odom_wait and not adapter.wait_odom(timeout=5.0):
            print("[stage6] \u8b66\u544a\uff1a\u672a\u6536\u5230\u91cc\u7a0b\u8ba1")
        adapter.stand()
        adapter.set_mapped_pose(START_FWD, START_ADAPTER_Y, START_WORLD_YAW)
        print("[stage6] \u8d77\u70b9\u6620\u5c04\u4e3a world=(%.3f,%.3f) local=(%.3f,%.3f) yaw=%.1f" % (
            START_WORLD_X, START_WORLD_Y, START_FWD, START_LAT, adapter.get_yaw_deg()))
        if args.test != "full" and not args.no_wait_detector:
            runner._wait_detector(timeout=20.0)
        if args.test == "full":
            ok = runner.run()
        else:
            ok = True
            if args.test in ("scan", "approach", "behind", "push"):
                ok = runner.scan_ball()
                if not ok:
                    print("[stage6] 测试[scan]失败：未找到球")
                else:
                    ok = runner._aim_ball()
                    if not ok:
                        print("[stage6] 测试[aim]失败：未把球转到画面中央")
            if ok and args.test in ("approach", "behind", "push"):
                ok = runner.center_ball(BALL_CLOSE_NORM, APPROACH_VX, 0.5, APPROACH_K_CX)
                if not ok:
                    print("[stage6] 测试[approach]失败")
            if ok and args.test in ("behind", "push"):
                bw = runner._ball_world()
                if bw is None:
                    ok = False
                    print("[stage6] 测试[behind]：估算球位置失败")
                else:
                    runner._lineup_for_push(bw[0], bw[1], "test-behind")
            if ok and args.test == "push":
                ok = runner.push_ball()
                if not ok:
                    print("[stage6] 测试[push]失败")
            print("[stage6] 测试[%s]完成，ok=%s" % (args.test, ok))
        print("[stage6] \u5b8c\u6210\uff0cok=%s" % ok)
        if not ok:
            print("[stage6] task stopped without completion; no lie_down command was sent")
    except KeyboardInterrupt:
        print("\n[stage6] \u7528\u6237\u4e2d\u65ad")
    except Exception as e:
        print("[stage6] \u5f02\u5e38: %s" % e)
        import traceback
        traceback.print_exc()
        runner._stop()
    finally:
        runner._stop()
        time.sleep(0.3)
        runner.close()
        adapter.shutdown()
        det.stop()
        if depth is not None:
            depth.stop()
        if stream_process is not None:
            stream_process.terminate()
            try:
                stream_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                stream_process.kill()
        print("[stage6] end")


if __name__ == "__main__":
    main()
