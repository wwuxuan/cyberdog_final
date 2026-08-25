#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第四赛段·黄线校准 + 循线（换通道 / 最后通道后回程）

供 stage4_real.py 调用。测距复用 fisheye_line_distance.py（MEI 反投影+地面交点，
现支持同时返回"线距"和"线角"），标定参数读同目录 line_calib.json。

- calibrate_fwd(adp, side)：狗已转好向（黄线在身侧）。
    1) 横移(vy)把线距(=真实fwd)校准到 target(0.60)；
    2) 用线角微转(wz)把身体转到与黄线平行（视觉校 yaw，不依赖陀螺仪）；
    3) 成功后 set_mapped_pose：里程计 fwd=target，yaw=该侧正确朝向（左+90/右-90）。
- line_follow(adp, side, stop_x, target_yaw)：沿当前朝向直走，线距闭环保持目标距离，
    线角闭环保持身体与黄线平行（wz 以视觉为主），走到绝对 x 越过 stop_x 停；
    线距/线角不可信（丢线、距离越界、角度异常）时按里程计直走，超时则中止。

方向约定（与 stage4_real 一致）：fwd=adapter x；lat=-adapter y；yaw 左转=+90，右转=-90。
线在身侧时：side="left" 则 err>0 向左移(vy>0)；side="right" 则 err>0 向右移(vy<0)。
线角：地面拟合线 y=a*x+b（x=前,y=左）→ angle=atan(a)，狗身与线平行时 angle≈0。
"""
import json
import math
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

from fish_line import CAMS, measure_vectorized

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON = os.path.join(HERE, "line.json")

TARGET_LINE = 0.60        # 走廊中心线 = 到边界黄线的目标距离(m)
BODY_HEIGHT = 0.235       # 正常站立机身中心离地高度(m)，与标定时一致
SETTLE_TOL = 0.02         # 距离校准允许误差(m)
SETTLE_TIMEOUT = 10.0     # 距离校准超时(s)
K_VY = 0.6                # 线距偏差 -> vy 增益
VY_MAX = 0.05             # vy 限幅(m/s)
K_WZ_ANGLE = 0.05         # 线角偏差 -> wz 增益（视觉朝向反馈，主）
ANGLE_TOL = 3.0           # yaw 校准允许线角误差(°)；线角测量噪声约±2~3°，太严格会很难收敛
ANGLE_PLAUSIBLE = 30.0    # 线角超过此值视为不可信（交叉线/角落）
YAW_ALIGN_TIMEOUT = 8.0   # yaw 校准超时(s)
K_WZ = 0.03               # 里程计 yaw 增益（丢线/无角度时保持朝向）
WZ_MAX = 0.15
DIST_MIN = 0.10           # 线距低于此值视为不可信（如贴近线/线到头）
DIST_MAX = 1.5            # 线距高于此值视为不可信
SPEED = 0.12              # 循线前进速度(m/s)
LOST_STOP_SECS = 5.0      # 循线连续不可信超过该时间则中止(s)
FOLLOW_TIMEOUT = 60.0     # 循线整体超时(s)
LOOP_DT = 0.05
PRINT_DT = 0.5

HSV = {"h_low": 10, "h_high": 50, "s_min": 50, "v_min": 50}


class Latest(object):
    def __init__(self):
        self._lock = threading.Lock()
        self._data = {}

    def set(self, key, value):
        with self._lock:
            self._data[key] = value

    def get(self, key, default=None):
        with self._lock:
            return self._data.get(key, default)


def image_to_bgr(msg):
    enc = (msg.encoding or "").lower()
    w, h, step = int(msg.width), int(msg.height), int(msg.step)
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    rows = raw.reshape((h, step))
    if enc in ("rgb8", "bgr8"):
        px = rows[:, :w * 3].reshape((h, w, 3))
        return px[:, :, ::-1] if enc == "rgb8" else px
    if enc in ("rgba8", "bgra8"):
        px = rows[:, :w * 4].reshape((h, w, 4))[:, :, :3]
        return px[:, :, ::-1] if enc == "rgba8" else px
    if enc in ("mono8", "8uc1"):
        return rows[:, :w].copy()
    raise ValueError("unsupported encoding %s" % msg.encoding)


def detect_namespace(node, timeout=15.0):
    t0 = time.time()
    topics = {}
    while time.time() - t0 < timeout:
        topics = dict(node.get_topic_names_and_types())
        for name in topics:
            if name.endswith("/image_left"):
                return name.rsplit("/", 1)[0]
        time.sleep(0.3)
    prefixes = {}
    for n in topics:
        parts = n.split("/")
        if len(parts) >= 3:
            pref = "/" + parts[1] + "/" + parts[2]
            prefixes[pref] = prefixes.get(pref, 0) + 1
    return max(prefixes, key=prefixes.get) if prefixes else ""


def activate_stereo(ns):
    import subprocess
    for cmd in (
        ["ros2", "lifecycle", "set", "%s/camera/camera" % ns, "configure"],
        ["ros2", "lifecycle", "set", "%s/camera/camera" % ns, "activate"],
        ["ros2", "lifecycle", "set", "%s/stereo_camera" % ns, "activate"],
    ):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            print("[line] $ %s -> rc=%d" % (" ".join(cmd), r.returncode))
        except Exception as e:
            print("[line] $ %s -> ERR %s" % (" ".join(cmd), e))


def wrap_deg(a):
    a = float(a) % 360.0
    if a > 180.0:
        a -= 360.0
    return a


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class LineCalib(object):
    def __init__(self, json_path=DEFAULT_JSON, target=TARGET_LINE,
                 body_height=BODY_HEIGHT, timeout_frames=12.0):
        self.target = target
        self.body_height = body_height
        self.ready = False
        self.node = None
        self.cal = self._load_cal(json_path)
        try:
            rclpy.init()
            self.node = rclpy.create_node("stage4_line")
        except Exception as e:
            print("[line] rclpy 初始化失败: %s（黄线校准不可用，将用里程计）" % e)
            return
        self.ns = detect_namespace(self.node)
        self.frames = Latest()
        if self.ns:
            self.node.create_subscription(Image, self.ns + "/image_left",
                                          lambda m: self.frames.set("left", m), 1)
            self.node.create_subscription(Image, self.ns + "/image_right",
                                          lambda m: self.frames.set("right", m), 1)
        self._spin = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self._spin.start()
        t0 = time.time()
        while time.time() - t0 < timeout_frames and not (self.frames.get("left") or self.frames.get("right")):
            time.sleep(0.2)
        if self.frames.get("left") or self.frames.get("right"):
            self.ready = True
        else:
            print("[line] 没有鱼眼帧，尝试激活 stereo_camera ...")
            activate_stereo(self.ns)
            time.sleep(1.0)
            t0 = time.time()
            while time.time() - t0 < timeout_frames and not (self.frames.get("left") or self.frames.get("right")):
                time.sleep(0.2)
            if self.frames.get("left") or self.frames.get("right"):
                self.ready = True
        if not self.ready:
            print("[line] 仍无鱼眼帧：黄线校准/循线不可用，全程用里程计")
        else:
            print("[line] 黄线校准就绪: %s" % self.cal)

    def _load_cal(self, path):
        cal = {"left": {"scale": 1.0, "offset": 0.0},
               "right": {"scale": 1.0, "offset": 0.0}}
        loaded = False
        try:
            d = json.load(open(path))
            for side in ("left", "right"):
                s = d.get(side)
                if s and "scale" in s and "offset" in s:
                    cal[side] = {"scale": float(s["scale"]), "offset": float(s["offset"])}
                    loaded = True
        except Exception as e:
            print("[line] 读取标定 %s 失败: %s（用 scale=1/offset=0）" % (path, e))
        if not loaded:
            print("[line] 未找到标定文件 %s（用 scale=1/offset=0）" % path)
        if cal["right"]["scale"] == 1.0 and cal["right"]["offset"] == 0.0 and cal["left"]["scale"] != 1.0:
            cal["right"] = dict(cal["left"])
            print("[line] 警告：右侧未标定，暂用左侧标定值（建议先跑 stage4_line_calib.py --side right 标定）")
        return cal

    def measure(self, side, n=1):
        """采 n 帧，返回最后一次非 None 的 (线距, 线角)；线角单位°，与狗身平行时≈0"""
        if not self.ready:
            return None, None
        val = ang = None
        for _ in range(n):
            msg = self.frames.get(side)
            if msg is None:
                time.sleep(0.08)
                continue
            try:
                img = image_to_bgr(msg)
            except Exception:
                time.sleep(0.08)
                continue
            sc = self.cal[side]["scale"]
            off = self.cal[side]["offset"]
            res = measure_vectorized(img, side, body_height=self.body_height,
                                     roi_bottom=0.5, thresh_k=1.6, hsv=HSV,
                                     debug_path=None, step=2, scale=sc, offset=off,
                                     line_mode="all", return_angle=True)
            if res[0] is not None:
                val, ang = res[0], res[2]
            time.sleep(0.08)
        return val, ang

    def _align_yaw_to_line(self, adp, side, timeout_):
        """线角 -> 与黄线平行；返回是否对齐"""
        t0 = time.time()
        while time.time() - t0 < timeout_:
            d, ang = self.measure(side, n=2)
            if d is None or ang is None or abs(ang) > ANGLE_PLAUSIBLE:
                print("[line]   校准 yaw：线角不可信(ang=%s)，跳过 yaw 对齐" %
                      ("n/a" if ang is None else "%.1f" % ang), flush=True)
                return False
            print("[line]   校准 yaw：线角=%.2f°" % ang, flush=True)
            if abs(ang) <= ANGLE_TOL:
                return True
            wz = _clamp(K_WZ_ANGLE * ang, -WZ_MAX, WZ_MAX)
            adp.walk(0.0, 0.0, wz)
            time.sleep(LOOP_DT)
        return False

    def calibrate_fwd(self, adp, side, tol=SETTLE_TOL, timeout=SETTLE_TIMEOUT,
                      target_dist=None, mapped_fwd=None):
        """狗已转好向（黄线在身侧）。
        顺序：线角校 yaw -> 横移校 fwd（横移会扰动朝向）-> 再线角校 yaw -> 同步。
        同步前再量一次线角：只有当前线角在容差内，才把里程计 yaw 设为该侧正确朝向（左+90/右-90），
        否则保持原值——避免把横移带歪的角度写进里程计。"""
        sign = 1.0 if side == "left" else -1.0
        target_yaw = 90.0 if side == "left" else -90.0
        line_target = self.target if target_dist is None else float(target_dist)
        mapped_target = line_target if mapped_fwd is None else float(mapped_fwd)
        if not DIST_MIN <= line_target <= DIST_MAX:
            print("[line] ERROR: 物理黄线目标 %.3fm 超出 %.2f-%.2fm，拒绝横移" %
                  (line_target, DIST_MIN, DIST_MAX), flush=True)
            adp.stop()
            return False

        # 1) yaw 校准（线角 -> 与黄线平行）
        yaw_aligned = self._align_yaw_to_line(adp, side, YAW_ALIGN_TIMEOUT)
        adp.stop()

        # 2) 距离校准（横移，把线距=真实fwd校准到 target）
        ok_dist = False
        last_err = None
        t1 = time.time()
        while time.time() - t1 < timeout:
            d, ang = self.measure(side, n=2)
            if d is None:
                print("[line] 校准中未检测到黄线(side=%s)，停下等待 ..." % side, flush=True)
                adp.stop()
                time.sleep(0.3)
                continue
            err = d - line_target
            last_err = err
            print("[line]   校准 side=%s 线距=%.3f target=%.3f err=%+.3f 线角=%s" %
                  (side, d, line_target, err,
                   "%.1f°" % ang if ang is not None else "--"), flush=True)
            if abs(err) <= tol:
                ok_dist = True
                break
            vy = _clamp(sign * K_VY * err, -VY_MAX, VY_MAX)
            adp.walk(0.0, vy, 0.0)
            time.sleep(LOOP_DT)
        adp.stop()

        # 3) 横移会扰动朝向 -> 再校一次 yaw，确保同步前角度是准的
        if ok_dist:
            print("[line]   横移后重新对齐 yaw ...", flush=True)
            if self._align_yaw_to_line(adp, side, YAW_ALIGN_TIMEOUT):
                yaw_aligned = True
            adp.stop()

        # 4) 同步里程计：只把 fwd 设为 target；yaw 保持当前值。
        #    狗此时已物理平行于黄线，保持当前 odom yaw，后续“转回 +fwd”的旋转量才正确；
        #    强行设 90 等于假设“线垂直于通道+脚螺仪很准”，这两点在这台狗上都不成立（会导致右转过多/偏右）。
        if ok_dist:
            x, y, _z = adp.get_position()
            yaw_now = adp.get_yaw_deg()
            _df, ang_final = self.measure(side, n=2)
            adp.set_mapped_pose(mapped_target, y, yaw_now)
            print("[line] 校准完成：里程计 fwd=%.3f 线距=%.3f yaw=%.1f（lat=%.3f，同步前线角=%s，保持当前yaw）" %
                  (mapped_target, line_target, yaw_now, -y,
                   ("%.1f°" % ang_final) if ang_final is not None else "n/a"), flush=True)
        else:
            print("[line] 校准失败/超时（最后 err=%s），保持里程计不变" %
                  ("%.3f" % last_err if last_err is not None else "n/a"))
        return ok_dist

    def end_calibrate(self, adp, side, target_dist):
        """终点校准：狗面向 +fwd、黄线在身侧（如右侧地图边界线）。
        只旋转(wz)校角度（与黄线平行）+ 横移(vy)校距离到 target_dist；
        全程不纵向动(vx)、不改变/同步 fwd，避免破坏 fwd 坐标。"""
        sign = 1.0 if side == "left" else -1.0

        # 1) yaw 对齐（与黄线平行）
        self._align_yaw_to_line(adp, side, YAW_ALIGN_TIMEOUT)
        adp.stop()

        # 2) 横向校准距离（只 vy，不动 vx / fwd）
        t1 = time.time()
        ok = False
        while time.time() - t1 < SETTLE_TIMEOUT:
            d, ang = self.measure(side, n=2)
            if d is None:
                print("[line] 终点校准中未检测到黄线(side=%s)，停下等待 ..." % side, flush=True)
                adp.stop()
                time.sleep(0.3)
                continue
            err = d - target_dist
            print("[line]   终点校准 side=%s 线距=%.3f err=%+.3f" % (side, d, err), flush=True)
            if abs(err) <= SETTLE_TOL:
                ok = True
                break
            vy = _clamp(sign * K_VY * err, -VY_MAX, VY_MAX)
            adp.walk(0.0, vy, 0.0)
            time.sleep(LOOP_DT)
        adp.stop()

        # 3) 横移会扰动朝向 -> 再校一次 yaw
        self._align_yaw_to_line(adp, side, YAW_ALIGN_TIMEOUT)
        adp.stop()
        print("[line] 终点校准完成：距%s黄线 %.3f m（fwd 未改变）" % (side, target_dist), flush=True)
        return ok

    def line_follow(self, adp, side, stop_x=None, target_yaw=0.0, speed=SPEED,
                    timeout=FOLLOW_TIMEOUT, lost_stop=LOST_STOP_SECS,
                    x_origin=0.0):
        """沿当前朝向直走：线距闭环保持目标距离，线角闭环保持身体与黄线平行；
        走到目标绝对 x 停；目标在身后时倒走到目标。
        线距/线角不可信时按里程计直走，超时中止。"""
        sign = 1.0 if side == "left" else -1.0
        if stop_x is not None:
            stop_local_lat = float(stop_x) - float(x_origin)
            stop_label = "x=%.3f" % float(stop_x)
        else:
            raise ValueError("line_follow 需要绝对 stop_x")
        x, y, _z = adp.get_position()
        current_x = float(x_origin) - float(y)
        going_up = current_x <= float(stop_x)
        target_delta_x = float(stop_x) - current_x
        forward_x = -math.sin(math.radians(float(target_yaw)))
        if abs(forward_x) < 0.5:
            forward_x = -math.sin(math.radians(adp.get_yaw_deg()))
        travel_sign = 1.0 if target_delta_x * forward_x >= 0.0 else -1.0
        hold_yaw = adp.get_yaw_deg()   # 循线期间保持的里程计朝向（丢线时回退用，=开始时的平行朝向）
        print("[line] 循线开始 side=%s stop=%s (%s)" %
              (side, stop_label, "x增大" if going_up else "x减小"), flush=True)
        t0 = time.time()
        last_print = 0.0
        lost_t0 = None
        while time.time() - t0 < timeout:
            x, y, _z = adp.get_position()
            lat = -y
            yaw = adp.get_yaw_deg()
            if (going_up and lat >= stop_local_lat) or (not going_up and lat <= stop_local_lat):
                print("[line] 循线到达 %s" %
                      ("x=%.3f" % (float(x_origin) + lat) if stop_x is not None
                       else "lat=%.3f" % lat), flush=True)
                break
            d, ang = self.measure(side, n=1)
            now = time.time()
            reliable = (d is not None and DIST_MIN <= d <= DIST_MAX and
                        ang is not None and abs(ang) <= ANGLE_PLAUSIBLE)
            if not reliable:
                if lost_t0 is None:
                    lost_t0 = now
                    print("[line] 丢线/读数不可信(dist=%s ang=%s)，按里程计直走 ..." % (
                        "%.3f" % d if d is not None else "n/a",
                        "%.1f" % ang if ang is not None else "n/a"), flush=True)
                elif now - lost_t0 > lost_stop:
                    print("[line] 连续不可信 %.1fs，循线中止" % lost_stop, flush=True)
                    adp.stop()
                    return False
                vy = 0.0
                wz = _clamp(K_WZ * wrap_deg(hold_yaw - yaw), -WZ_MAX, WZ_MAX)
                d_disp = "noline" if d is None else "%.3f" % d
            else:
                lost_t0 = None
                err = d - self.target
                vy = _clamp(sign * K_VY * err, -VY_MAX, VY_MAX)
                # 视觉线角为主 + 少量里程计 yaw 辅助
                wz = _clamp(K_WZ_ANGLE * ang, -WZ_MAX, WZ_MAX)
                d_disp = "%.3f" % d
            adp.walk(travel_sign * speed, vy, wz)
            if now - last_print >= PRINT_DT:
                last_print = now
                print("[line]   odom(fwd=%.3f lat=%.3f yaw=%.1f) 线测fwd=%s 线角=%s" %
                      (x, lat, yaw, d_disp,
                       ("%.1f°" % ang) if ang is not None else "--"), flush=True)
            time.sleep(LOOP_DT)
        else:
            print("[line] 循线超时", flush=True)
            adp.stop()
            return False
        adp.stop()
        return True

    def line_follow_fwd_distance(self, adp, side, distance, target_dist,
                                 speed=SPEED, timeout=15.0,
                                 lost_stop=LOST_STOP_SECS, stop_callback=None):
        """面向当前前方循线指定距离，持续校正侧线距离和线角。"""
        sign = 1.0 if side == "left" else -1.0
        start_fwd = adp.get_position()[0]
        target_fwd = start_fwd + float(distance)
        hold_yaw = adp.get_yaw_deg()
        t0 = time.time()
        last_print = 0.0
        lost_t0 = None
        print("[line] 前向循线开始 side=%s dist=%.2fm target=%.2fm" %
              (side, distance, target_dist), flush=True)
        while time.time() - t0 < timeout:
            fwd, lat_raw, _z = adp.get_position()
            yaw = adp.get_yaw_deg()
            if stop_callback is not None:
                reason = stop_callback()
                if reason:
                    adp.stop()
                    print("[line] 前向循线因%s提前停止，已前进 %.2fm" %
                          (reason, fwd - start_fwd), flush=True)
                    return True
            if fwd >= target_fwd:
                adp.stop()
                print("[line] 前向循线到达 dist=%.2fm" % (fwd - start_fwd), flush=True)
                return True

            d, ang = self.measure(side, n=1)
            now = time.time()
            reliable = (d is not None and DIST_MIN <= d <= DIST_MAX and
                        ang is not None and abs(ang) <= ANGLE_PLAUSIBLE)
            if reliable:
                lost_t0 = None
                err = d - target_dist
                vy = _clamp(sign * K_VY * err, -VY_MAX, VY_MAX)
                wz = _clamp(K_WZ_ANGLE * ang, -WZ_MAX, WZ_MAX)
                d_disp = "%.3f" % d
            else:
                if lost_t0 is None:
                    lost_t0 = now
                    print("[line] 前向循线读数不可信(dist=%s ang=%s)，按当前朝向前进 ..." % (
                        "%.3f" % d if d is not None else "n/a",
                        "%.1f" % ang if ang is not None else "n/a"), flush=True)
                elif now - lost_t0 > lost_stop:
                    adp.stop()
                    print("[line] 前向循线连续不可信 %.1fs，中止" % lost_stop, flush=True)
                    return False
                vy = 0.0
                wz = _clamp(K_WZ * wrap_deg(hold_yaw - yaw), -WZ_MAX, WZ_MAX)
                d_disp = "noline" if d is None else "%.3f" % d

            adp.walk(speed, vy, wz)
            if now - last_print >= PRINT_DT:
                last_print = now
                print("[line]   前向循线 fwd=%.3f/%.3f lat=%.3f yaw=%.1f line=%s angle=%s" %
                      (fwd, target_fwd, -lat_raw, yaw, d_disp,
                       "%.1f°" % ang if ang is not None else "--"), flush=True)
            time.sleep(LOOP_DT)

        adp.stop()
        print("[line] 前向循线超时", flush=True)
        return False

    def shutdown(self):
        try:
            if self.node is not None:
                self.node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
