#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第四赛段·走廊黄线 标定 + 验证（鱼眼测线距 -> fwd 校准）

背景几何：
  黄线 = 第四赛段走廊的边界，在 fwd=0 处，沿走廊横向（lat 方向）铺设。
  狗在走廊中心线 fwd=0.6 行走；从通道出来后面朝 +fwd（倒退着出通道），
  左转90°后面朝下一个通道（-lat 方向），此时黄线位于狗身左侧约 0.6m。
  左侧鱼眼测出的"到左侧黄线的垂直距离" ≈ 狗的 fwd 坐标，目标 0.6m。
  标定公式（与 line_monitor.py 一致）：d_real = scale * d_raw + offset

模式：
  --calib    交互式标定：把狗摆到已知离黄线距离的位置，输入实际距离(米)回车，
             自动采集若干帧原始线距；采够 2 个点后用最小二乘拟合 scale/offset，
             打印结果并写入 line_calib.json（同时也打印可粘进 line_monitor.py 的值）
  --monitor  实时打印"已标定"线距，用于验证/调试（scale/offset 用命令行或 json）

用法（在狗上，与 fisheye_line_distance.py 同目录）：
  source /etc/mi/ros2_env.conf
  # 标定（推荐两个位置：0.30m 和 0.50m，正常站立姿态）
  python3 stage4_line_calib.py --calib --side left --body-height 0.235
  # 验证（用标定出的 scale/offset）
  python3 stage4_line_calib.py --monitor --side left --scale-left 0.78 --offset-left 0.0
"""
import argparse
import json
import os
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

from fish_line import CAMS, measure_vectorized

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(os.path.dirname(HERE), "core", "line.json")


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
            print("[calib] $ %s -> rc=%d" % (" ".join(cmd), r.returncode))
        except Exception as e:
            print("[calib] $ %s -> ERR %s" % (" ".join(cmd), e))


def measure_once(img, side, body_height, scale, offset, cfg):
    dist, _ = measure_vectorized(
        img, side, body_height, cfg["roi_bottom"], cfg["thresh_k"],
        cfg["hsv"], None, cfg["step"], scale, offset, line_mode=cfg["line_mode"])
    return dist


def collect_point(side, body_height, frames, cfg, scale=1.0, offset=0.0, n=10):
    """采集 n 帧原始线距，返回非 None 的平均值；采不到返回 None"""
    vals = []
    for _ in range(n):
        msg = frames.get(side)
        if msg is None:
            time.sleep(0.1)
            continue
        try:
            img = image_to_bgr(msg)
        except Exception:
            time.sleep(0.1)
            continue
        d = measure_once(img, side, body_height, scale, offset, cfg)
        if d is not None:
            vals.append(d)
        time.sleep(0.12)
    if not vals:
        return None
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser(description="第四赛段走廊黄线 标定/验证")
    ap.add_argument("--calib", action="store_true", help="交互式标定模式（默认）")
    ap.add_argument("--monitor", action="store_true", help="实时验证模式（--monitor 优先于 --mode）")
    ap.add_argument("--mode", choices=["calib", "monitor"], default="calib")
    ap.add_argument("--side", choices=["left", "right"], default="left",
                    help="用哪个鱼眼（狗左转90°后黄线在左侧，默认 left）")
    ap.add_argument("--body-height", type=float, default=0.235,
                    help="机身中心离地高度(m)，正常站立姿态，标定和实测必须一致；0=从 /odom_out 自动读")
    ap.add_argument("--frames", type=int, default=10, help="每个标定点采集帧数")
    ap.add_argument("--scale-left", type=float, default=1.0)
    ap.add_argument("--offset-left", type=float, default=0.0)
    ap.add_argument("--scale-right", type=float, default=1.0)
    ap.add_argument("--offset-right", type=float, default=0.0)
    ap.add_argument("--roi-bottom", type=float, default=0.5)
    ap.add_argument("--thresh-k", type=float, default=1.6)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--line-mode", default="all")
    args = ap.parse_args()
    mode = "monitor" if args.monitor else ("calib" if args.calib else args.mode)

    cfg = {
        "roi_bottom": args.roi_bottom,
        "thresh_k": args.thresh_k,
        "step": args.step,
        "line_mode": args.line_mode,
        "hsv": {"h_low": 10, "h_high": 50, "s_min": 50, "v_min": 50},
    }

    # 读取已保存的标定（monitor 模式未显式给 scale/offset 时用）
    saved = {}
    if os.path.exists(JSON_PATH):
        try:
            saved = json.load(open(JSON_PATH))
        except Exception:
            saved = {}

    if mode == "monitor":
        sc = args.scale_left if args.side == "left" else args.scale_right
        off = args.offset_left if args.side == "left" else args.offset_right
        s = saved.get(args.side, {})
        if args.scale_left == 1.0 and "scale" in s:
            sc = s["scale"]
        if args.offset_left == 0.0 and "offset" in s:
            off = s["offset"]
        print("===== 黄线线距 实时监测 (side=%s, scale=%.3f, offset=%.3f) =====" % (args.side, sc, off))
        print("目标线距 = 0.60m（狗在走廊中心线 fwd=0.6 时）；Ctrl+C 退出")
    else:
        print("===== 黄线 两点标定 (side=%s) =====" % args.side)
        print("把狗放在平地、正常站立（与比赛时同一姿态）。")
        print("用卷尺从狗身体中心(正下方投影)量到黄线的实际距离，然后输入该距离(米)并回车。")
        print("推荐采两个位置：0.30 和 0.50。输 q 结束。")

    rclpy.init()
    node = rclpy.create_node("stage4_line_calib")
    ns = detect_namespace(node)
    if not ns:
        print("[calib] ERROR: 找不到 /image_left 命名空间")
        rclpy.shutdown()
        return 2
    print("[calib] namespace = %s" % ns)

    frames = Latest()
    odom_h = Latest()
    node.create_subscription(Image, ns + "/image_left",
                             lambda m: frames.set("left", m), 1)
    node.create_subscription(Image, ns + "/image_right",
                             lambda m: frames.set("right", m), 1)
    node.create_subscription(Odometry, ns + "/odom_out",
                             lambda m: odom_h.set("z", float(m.pose.pose.position.z)), 5)
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()

    t0 = time.time()
    while time.time() - t0 < 12 and not (frames.get("left") or frames.get("right")):
        time.sleep(0.2)
    if not (frames.get("left") or frames.get("right")):
        print("[calib] 没有鱼眼帧，尝试激活 stereo_camera ...")
        activate_stereo(ns)
        time.sleep(1.0)
        t0 = time.time()
        while time.time() - t0 < 12 and not (frames.get("left") or frames.get("right")):
            time.sleep(0.2)
    if not (frames.get("left") or frames.get("right")):
        print("[calib] 仍没有鱼眼帧，请先运行 activate_cameras.py")
        rclpy.shutdown()
        return 3

    body_height = args.body_height
    if body_height <= 0:
        t0 = time.time()
        while time.time() - t0 < 5 and odom_h.get("z") is None:
            time.sleep(0.2)
        body_height = odom_h.get("z")
        if body_height is None:
            body_height = 0.235
    print("[calib] body_height = %.3f m" % body_height)

    try:
        if mode == "calib":
            pts = []  # (d_real, d_raw)
            while True:
                try:
                    inp = input("输入当前离黄线的实际距离(米)，如 0.30；q 结束: ").strip().lower()
                except EOFError:
                    break
                if inp in ("q", "quit", ""):
                    break
                try:
                    d_real = float(inp)
                except ValueError:
                    print("  输入无效，重新输入")
                    continue
                print("  采集 %d 帧原始线距 ..." % args.frames)
                raw = collect_point(args.side, body_height, frames, cfg,
                                    scale=1.0, offset=0.0, n=args.frames)
                if raw is None:
                    print("  没检测到黄线！检查：狗是否在走廊上、黄线是否在 %s 鱼眼视野内、光照/HSV" % args.side)
                    continue
                pts.append((d_real, raw))
                print("  已记录: d_real=%.3f m, d_raw=%.3f m" % (d_real, raw))
                if len(pts) >= 2:
                    print("  已采 %d 个点，可以继续采更多（更准），或输 q 计算" % len(pts))
            if len(pts) < 2:
                print("[calib] 至少需要 2 个点")
                rclpy.shutdown()
                return 4
            A = np.array([[r, 1.0] for _, r in pts])
            b = np.array([d for d, _ in pts])
            scale, offset = np.linalg.lstsq(A, b, rcond=None)[0]
            pred = A @ np.array([scale, offset])
            res = b - pred
            rms = float(np.sqrt(np.mean(res ** 2)))
            print("")
            print("===== 标定结果 =====")
            for (d, r) in pts:
                print("  实际 %.3f m -> 原始 %.3f m -> 换算 %.3f m" % (d, r, scale * r + offset))
            print("  scale = %.4f" % scale)
            print("  offset = %.4f" % offset)
            print("  拟合残差 RMS = %.4f m" % rms)
            print("")
            print("写入 %s" % JSON_PATH)
            saved[args.side] = {"scale": float(scale), "offset": float(offset)}
            with open(JSON_PATH, "w") as f:
                json.dump(saved, f, indent=2)
            print("也可粘进 line_monitor.py 的 CONFIG['calibration']['%s']:" % args.side)
            print("    \"%s\": {\"scale\": %.4f, \"offset\": %.4f}" % (args.side, scale, offset))
        else:
            sc = args.scale_left if args.side == "left" else args.scale_right
            off = args.offset_left if args.side == "left" else args.offset_right
            s = saved.get(args.side, {})
            if args.scale_left == 1.0 and "scale" in s:
                sc = s["scale"]
            if args.offset_left == 0.0 and "offset" in s:
                off = s["offset"]
            print("  (side=%s, scale=%.4f, offset=%.4f)  目标 0.60m" % (args.side, sc, off))
            last = 0.0
            while rclpy.ok():
                now = time.time()
                if now - last >= 0.25:
                    last = now
                    d = collect_point(args.side, body_height, frames, cfg,
                                      scale=sc, offset=off, n=3)
                    if d is None:
                        print("  [%s] noline" % args.side, flush=True)
                    else:
                        print("  [%s] %.3f m   (fwd≈%.3f, 偏差 %.3f)" % (args.side, d, d, d - 0.60), flush=True)
                time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n[calib] stopped")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
