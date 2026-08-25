#!/usr/bin/env python3
"""
Grab live left/right fish-eye frames and measure the lateral distance to the
yellow lines on the ground, ONE DISTANCE PER SIDE:
    left fish-eye  -> distance to the LEFT  yellow line (+y in body frame)
    right fish-eye -> distance to the RIGHT yellow line (-y in body frame)

The pipeline per pixel is: MEI back-projection -> body ray -> ground-plane
intersection.  Colour fish-eye is detected via HSV yellow; grayscale via
brightness.

The namespace is found by polling the topic list for <ns>/image_left (DDS
discovery can take a few seconds after the node starts).  If the fish-eye
topics have no publishers yet, the stereo_camera lifecycle node is activated
via the ros2 CLI.

Usage (on the dog):
  cd /home/mi/cyberdog_competition
  source /etc/mi/ros2_env.conf
  python3 live_line_measure.py --side both --body-height 0.26
  python3 live_line_measure.py --side left --duration 8 --debug-dir /tmp/linedbg
"""

import os
import shlex
import sys
import time


def _ensure_ros_env():
    if os.environ.get("ROS_DISTRO") or os.name != "posix":
        return
    if os.environ.get("CYBERDOG_ROS_REEXEC"):
        return
    setup_candidates = [
        "/opt/ros2/cyberdog/setup.bash",
        "/opt/ros2/galactic/setup.bash",
        "/opt/ros/foxy/setup.bash",
        "/opt/ros/galactic/setup.bash",
        "/opt/ros/humble/setup.bash",
    ]
    setup = next((p for p in setup_candidates if os.path.exists(p)), None)
    if setup is None:
        return
    env = dict(os.environ)
    env["CYBERDOG_ROS_REEXEC"] = "1"
    script = os.path.abspath(__file__)
    args = " ".join(shlex.quote(a) for a in sys.argv[1:])
    cmd = (f"source {shlex.quote(setup)} && "
           f"cd {shlex.quote(os.path.dirname(script))} && "
           f"exec python3 {shlex.quote(script)} {args}")
    os.execve("/bin/bash", ["bash", "-lc", cmd], env)


_ensure_ros_env()

import rclpy
from sensor_msgs.msg import Image
import numpy as np
import cv2
from fish_line import CAMS, measure


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
    raise ValueError(f"unsupported encoding {msg.encoding}")


def detect_namespace(node, timeout=15.0):
    """Poll for <ns>/image_left; fall back to the most common topic prefix."""
    t0 = time.time()
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
    """Bring the fish-eye stereo_camera lifecycle node up via ros2 CLI."""
    import subprocess
    for cmd in (
        ["ros2", "lifecycle", "set", f"{ns}/camera/camera", "configure"],
        ["ros2", "lifecycle", "set", f"{ns}/camera/camera", "activate"],
        ["ros2", "lifecycle", "set", f"{ns}/camera/camera_align", "activate"],
        ["ros2", "lifecycle", "set", f"{ns}/stereo_camera", "activate"],
    ):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            print(f"[live] $ {' '.join(cmd)} -> rc={r.returncode} {r.stdout.strip()[:80]}")
        except Exception as e:
            print(f"[live] $ {' '.join(cmd)} -> ERR {e}")


def wait_frames(node, ns, duration):
    frames = {}

    def cb(side):
        def _cb(msg):
            if side not in frames:
                frames[side] = msg
                print(f"[live] got {side}: {msg.encoding} {msg.width}x{msg.height}")
        return _cb

    node.create_subscription(Image, ns + "/image_left", cb("left"), 1)
    node.create_subscription(Image, ns + "/image_right", cb("right"), 1)
    t0 = time.time()
    while time.time() - t0 < duration:
        rclpy.spin_once(node, timeout_sec=0.1)
        if len(frames) >= 2:
            break
    return frames


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="both", choices=["left", "right", "both"])
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--body-height", type=float, default=0.26)
    ap.add_argument("--debug-dir", default="/tmp/linedbg")
    ap.add_argument("--roi-bottom", type=float, default=0.5)
    ap.add_argument("--thresh-k", type=float, default=1.6)
    ap.add_argument("--h-low", type=int, default=15)
    ap.add_argument("--h-high", type=int, default=40)
    ap.add_argument("--s-min", type=int, default=70)
    ap.add_argument("--v-min", type=int, default=70)
    args = ap.parse_args()
    hsv = {"h_low": args.h_low, "h_high": args.h_high,
           "s_min": args.s_min, "v_min": args.v_min}

    rclpy.init()
    node = rclpy.create_node("live_line_measure")
    print("[live] waiting for <ns>/image_left to appear on DDS ...")
    ns = detect_namespace(node)
    if not ns:
        print("[live] ERROR: namespace not found on the topic list")
        rclpy.shutdown()
        return 2
    print(f"[live] namespace = {ns}")

    frames = wait_frames(node, ns, args.duration)
    if not frames:
        print("[live] no fish-eye frames yet, activating stereo_camera ...")
        activate_stereo(ns)
        time.sleep(1.0)
        frames = wait_frames(node, ns, args.duration)
    if not frames:
        print("[live] still no frames after activation")
        rclpy.shutdown()
        return 3

    os.makedirs(args.debug_dir, exist_ok=True)
    results = {}
    for side in ("left", "right"):
        if args.side not in ("both", side):
            continue
        if side not in frames:
            print(f"[live] no {side} frame")
            continue
        img = image_to_bgr(frames[side])
        raw_path = f"{args.debug_dir}/{side}_raw.png"
        if img.ndim == 3:
            cv2.imwrite(raw_path, img)
        else:
            cv2.imwrite(raw_path, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        dbg_path = f"{args.debug_dir}/{side}_debug.png"
        dist, _ = measure(img, side, args.body_height, dbg_path,
                          args.roi_bottom, args.thresh_k, hsv)
        results[side] = dist
        print(f"[live] saved raw {side} -> {raw_path}")

    print("=" * 46)
    for side in ("left", "right"):
        if side in results:
            d = results[side]
            print(f"[RESULT] {side} yellow-line distance = "
                  f"{d:.3f} m" if d is not None else f"[RESULT] {side}: no line")
    print("=" * 46)

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
