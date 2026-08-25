#!/usr/bin/env python3
"""
Continuous monitor: prints the dog's BODY-CENTRE distance to the LEFT and RIGHT
yellow lines, one line per update.

Per fish-eye frame it:
  - detects the yellow line (HSV on colour, brightness on grayscale)
  - vectorised MEI back-projection of the line pixels -> unit rays
  - intersects with the ground plane at z = -body_height
  - fits the ground line and prints the perpendicular distance from the body
    centre (0,0) to it (signed: + left / - right)

All tunables live in the CONFIG block below - edit them directly, no need for
command-line arguments.  Command-line flags (--side, --body-height, --scale,
--scale-left, ...) still override CONFIG if you want to tweak on the fly.

Usage (on the dog):
  cd /home/mi/cyberdog_competition
  source /etc/mi/ros2_env.conf
  python3 line_monitor.py
Ctrl-C to stop.
"""

import os
import shlex
import sys
import time
import threading
import json
from collections import deque
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

# =====================================================================
#  FIELD CONFIG - edit these, then just run `python3 line_monitor.py`
# =====================================================================
CONFIG = {
    "side": "both",             # "left", "right" or "both"
    "hz": 5.0,                  # update rate (Hz)
    # body origin height above the ground. Keep the dog in the SAME pose used
    # when the scale calibration below was measured.  None => auto from /odom_out
    "body_height": 0.235,
    "roi_bottom": 0.5,          # ground ROI: bottom half of the image
    "thresh_k": 1.6,            # grayscale threshold scale (gray fish-eye only)
    "hsv": {                    # yellow HSV range for colour fish-eye
        "h_low": 10, "h_high": 50,
        "s_min": 50, "v_min": 50,
    },
    # per-side distance calibration:  d_real = scale * d_measured + offset
    # calibrated 2026-08-10: dog centre was 0.30 m from BOTH lines.
    "calibration": {
        "left":  {"scale": 0.78, "offset": 0.0},
        "right": {"scale": 0.94, "offset": 0.0},
    },
    "debug_dir": None,          # e.g. "/tmp/linedbg" to also dump debug frames
    "debug_save_interval": 1.0, # archive one raw+annotated frame per side each N seconds
    "step": 2,                  # sample every Nth line pixel (speed)
    # "body_x" keeps the side boundary line and excludes an L-connected front
    # line after ground projection.  Keep "all" as the compatibility default.
    "line_mode": "all",         # "all" or "body_x"
}
# =====================================================================


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
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import numpy as np
import cv2
from fish_line import CAMS, measure_vectorized


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


def hue_hint(img, roi_bottom=0.5):
    """Diagnostic for 'no line': dominant hue among saturated pixels in the ROI."""
    h, w = img.shape[:2]
    if img.ndim != 3:
        return " (gray cam)"
    roi = img[int(h * roi_bottom):, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    sel = (S > 80) & (V > 80)
    if not sel.any():
        return " (no saturated px in ROI)"
    Hs = H[sel]
    hist, edges = np.histogram(Hs, bins=90, range=(0, 180))
    dom = int(edges[np.argmax(hist)])
    return f" (domH={dom}, n={len(Hs)}, S~{S[sel].mean():.0f}, V~{V[sel].mean():.0f})"


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
        ["ros2", "lifecycle", "set", f"{ns}/camera/camera", "configure"],
        ["ros2", "lifecycle", "set", f"{ns}/camera/camera", "activate"],
        ["ros2", "lifecycle", "set", f"{ns}/camera/camera_align", "activate"],
        ["ros2", "lifecycle", "set", f"{ns}/stereo_camera", "activate"],
    ):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            print(f"[monitor] $ {' '.join(cmd)} -> rc={r.returncode}")
        except Exception as e:
            print(f"[monitor] $ {' '.join(cmd)} -> ERR {e}")


def main():
    import argparse
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--side", choices=["left", "right", "both"], default=None)
    ap.add_argument("--hz", type=float, default=None)
    ap.add_argument("--body-height", type=float, default=None)
    ap.add_argument("--roi-bottom", type=float, default=None)
    ap.add_argument("--thresh-k", type=float, default=None)
    ap.add_argument("--h-low", type=int, default=None)
    ap.add_argument("--h-high", type=int, default=None)
    ap.add_argument("--s-min", type=int, default=None)
    ap.add_argument("--v-min", type=int, default=None)
    ap.add_argument("--debug-dir", default=None)
    ap.add_argument("--debug-save-interval", type=float, default=None)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--line-mode", choices=["all", "body_x"], default=None)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--offset", type=float, default=None)
    ap.add_argument("--scale-left", type=float, default=None)
    ap.add_argument("--scale-right", type=float, default=None)
    ap.add_argument("--offset-left", type=float, default=None)
    ap.add_argument("--offset-right", type=float, default=None)
    a = ap.parse_args()

    # CONFIG is the source of truth; CLI flags override it when given
    cfg = {
        "side": a.side or CONFIG["side"],
        "hz": a.hz if a.hz is not None else CONFIG["hz"],
        "body_height": (a.body_height if a.body_height is not None
                        else CONFIG["body_height"]),
        "roi_bottom": a.roi_bottom if a.roi_bottom is not None else CONFIG["roi_bottom"],
        "thresh_k": a.thresh_k if a.thresh_k is not None else CONFIG["thresh_k"],
        "hsv": dict(CONFIG["hsv"]),
        "debug_dir": a.debug_dir if a.debug_dir is not None else CONFIG["debug_dir"],
        "debug_save_interval": (
            a.debug_save_interval if a.debug_save_interval is not None
            else CONFIG["debug_save_interval"]
        ),
        "step": a.step if a.step is not None else CONFIG["step"],
        "line_mode": a.line_mode or CONFIG["line_mode"],
        "cal": {
            "left": [a.scale_left if a.scale_left is not None else
                     (a.scale if a.scale is not None else CONFIG["calibration"]["left"]["scale"]),
                     a.offset_left if a.offset_left is not None else
                     (a.offset if a.offset is not None else CONFIG["calibration"]["left"]["offset"])],
            "right": [a.scale_right if a.scale_right is not None else
                      (a.scale if a.scale is not None else CONFIG["calibration"]["right"]["scale"]),
                      a.offset_right if a.offset_right is not None else
                      (a.offset if a.offset is not None else CONFIG["calibration"]["right"]["offset"])],
        },
    }
    for k in ("h_low", "h_high", "s_min", "v_min"):
        v = getattr(a, k)
        if v is not None:
            cfg["hsv"][k] = v

    rclpy.init()
    node = rclpy.create_node("line_monitor")
    ns = detect_namespace(node)
    if not ns:
        print("[monitor] ERROR: namespace not found")
        rclpy.shutdown()
        return 2
    print(f"[monitor] namespace = {ns}")

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
        print("[monitor] no fish-eye frames, activating stereo_camera ...")
        activate_stereo(ns)
        time.sleep(1.0)
        t0 = time.time()
        while time.time() - t0 < 12 and not (frames.get("left") or frames.get("right")):
            time.sleep(0.2)
    if not (frames.get("left") or frames.get("right")):
        print("[monitor] still no fish-eye frames")
        rclpy.shutdown()
        return 3

    body_height = cfg["body_height"]
    if body_height is None:
        t0 = time.time()
        while time.time() - t0 < 5 and odom_h.get("z") is None:
            time.sleep(0.2)
        body_height = odom_h.get("z")
        if body_height is None:
            body_height = 0.26
    print(f"[monitor] body height above ground = {body_height:.3f} m "
          f"({'from /odom_out' if cfg['body_height'] is None else 'from CONFIG'})")
    print(f"[monitor] line_mode={cfg['line_mode']} Ctrl-C to stop")

    period = 1.0 / max(cfg["hz"], 0.5)
    last = 0.0
    last_debug_save = {"left": 0.0, "right": 0.0}
    try:
        while rclpy.ok():
            now = time.time()
            if now - last >= period:
                last = now
                parts = []
                for side in ("left", "right"):
                    if cfg["side"] not in ("both", side):
                        continue
                    msg = frames.get(side)
                    if msg is None:
                        parts.append(f"{side}:--")
                        continue
                    try:
                        img = image_to_bgr(msg)
                    except Exception:
                        parts.append(f"{side}:enc?")
                        continue
                    dbg = None
                    if cfg["debug_dir"]:
                        os.makedirs(cfg["debug_dir"], exist_ok=True)
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        if now - last_debug_save[side] >= max(0.1, cfg["debug_save_interval"]):
                            last_debug_save[side] = now
                            cv2.imwrite(
                                os.path.join(cfg["debug_dir"], f"{side}_raw_{timestamp}.png"),
                                img,
                            )
                            dbg = os.path.join(cfg["debug_dir"], f"{side}_dbg_{timestamp}.png")
                        else:
                            dbg = os.path.join(cfg["debug_dir"], f"{side}_dbg_latest.png")
                    sc, off = cfg["cal"][side]
                    dist, _ = measure_vectorized(img, side, body_height,
                                                 cfg["roi_bottom"], cfg["thresh_k"],
                                                 cfg["hsv"], dbg, cfg["step"],
                                                 sc, off, line_mode=cfg["line_mode"])
                    if dist is not None:
                        parts.append(f"{side}:{dist:.3f}m")
                    else:
                        parts.append(f"{side}:noline{hue_hint(img, cfg['roi_bottom'])}")
                cal_txt = "  ".join(
                    f"{s}*{cfg['cal'][s][0]:.3f}{'+' if cfg['cal'][s][1] >= 0 else '-'}{abs(cfg['cal'][s][1]):.3f}"
                    for s in ("left", "right"))
                print("[" + "  ".join(parts) +
                      f" ]  body_h={body_height:.3f}  cal[{cal_txt}]", flush=True)
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n[monitor] stopped")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


class Latest:
    def __init__(self):
        self._lock = threading.Lock()
        self._data = {}

    def set(self, key, val):
        with self._lock:
            self._data[key] = val

    def get(self, key, default=None):
        with self._lock:
            return self._data.get(key, default)


class LineDistanceMonitor:
    """Reusable yellow-line distance monitor for stage3 closed-loop control."""

    def __init__(self, target_dist=0.30, k_vy=0.18, max_vy=0.02, hz=6.0,
                 width_tol_low=0.18, width_tol_high=0.28,
                 window=7, min_valid=5, deadband=0.03, max_vy_step=0.004,
                 side="both", body_height=None, roi_bottom=None, thresh_k=None,
                 hsv=None, step=None, debug_dir=None, line_mode=None,
                 calibration=None, remote_url=None, remote_timeout=0.5,
                 valid_distance_min=None, valid_distance_max=None):
        self.target_dist = float(target_dist)
        self.k_vy = float(k_vy)
        self.max_vy = float(max_vy)
        self.width_tol_low = float(width_tol_low)
        self.width_tol_high = float(width_tol_high)
        self.window = max(1, int(window))
        self.min_valid = min(self.window, max(1, int(min_valid)))
        self.deadband = max(0.0, float(deadband))
        self.max_vy_step = max(0.001, float(max_vy_step))
        self.valid_distance_min = (
            None if valid_distance_min is None else float(valid_distance_min)
        )
        self.valid_distance_max = (
            None if valid_distance_max is None else float(valid_distance_max)
        )
        if (
            self.valid_distance_min is not None
            and self.valid_distance_max is not None
            and self.valid_distance_min > self.valid_distance_max
        ):
            raise ValueError("valid_distance_min must not exceed valid_distance_max")
        self.period = 1.0 / max(float(hz), 0.5)
        self.cfg = {
            "side": side,
            "body_height": CONFIG["body_height"] if body_height is None else body_height,
            "roi_bottom": CONFIG["roi_bottom"] if roi_bottom is None else roi_bottom,
            "thresh_k": CONFIG["thresh_k"] if thresh_k is None else thresh_k,
            "hsv": dict(CONFIG["hsv"] if hsv is None else hsv),
            "debug_dir": debug_dir,
            "step": CONFIG["step"] if step is None else step,
            "line_mode": CONFIG["line_mode"] if line_mode is None else line_mode,
            "remote_url": remote_url,
            "remote_timeout": max(0.05, float(remote_timeout)),
            "cal": {
                "left": [
                    CONFIG["calibration"]["left"]["scale"],
                    CONFIG["calibration"]["left"]["offset"],
                ],
                "right": [
                    CONFIG["calibration"]["right"]["scale"],
                    CONFIG["calibration"]["right"]["offset"],
                ],
            },
        }
        if calibration is not None:
            for camera_side, values in calibration.items():
                if camera_side not in ("left", "right"):
                    raise ValueError("calibration keys must be 'left' or 'right'")
                if len(values) != 2:
                    raise ValueError("each calibration must contain scale and offset")
                self.cfg["cal"][camera_side] = [float(values[0]), float(values[1])]
        self.node = None
        self.executor = None
        self.ns = ""
        self.frames = Latest()
        self.odom_h = Latest()
        self.spin = None
        self.body_height = None
        self.last_update = 0.0
        self.last_vy = 0.0
        self.last_info = {
            "ok": False,
            "left": None,
            "right": None,
            "err": 0.0,
            "vy": 0.0,
            "reason": "not_started",
        }
        self.samples = deque(maxlen=self.window)
        self._owns_rclpy = False

    def start(self, timeout=12.0, activate=True):
        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True

        self.node = rclpy.create_node("stage3_line_distance_monitor")
        self.ns = detect_namespace(self.node, timeout=min(timeout, 8.0))
        if not self.ns:
            self.last_info = dict(self.last_info, reason="namespace_not_found")
            return False

        self.node.create_subscription(
            Image, self.ns + "/image_left",
            lambda m: self.frames.set("left", m), 1,
        )
        self.node.create_subscription(
            Image, self.ns + "/image_right",
            lambda m: self.frames.set("right", m), 1,
        )
        self.node.create_subscription(
            Odometry, self.ns + "/odom_out",
            lambda m: self.odom_h.set("z", float(m.pose.pose.position.z)), 5,
        )

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.spin = threading.Thread(target=self._spin_loop, daemon=True)
        self.spin.start()

        t0 = time.time()
        while time.time() - t0 < timeout and not (self.frames.get("left") or self.frames.get("right")):
            time.sleep(0.1)
        if activate and not (self.frames.get("left") or self.frames.get("right")):
            print("[line_monitor] no fish-eye frames, activating stereo_camera ...")
            activate_stereo(self.ns)
            time.sleep(1.0)
            t0 = time.time()
            while time.time() - t0 < timeout and not (self.frames.get("left") or self.frames.get("right")):
                time.sleep(0.1)
        if not (self.frames.get("left") or self.frames.get("right")):
            self.last_info = dict(self.last_info, reason="no_fisheye_frames")
            return False

        self.body_height = self.cfg["body_height"]
        if self.body_height is None:
            t0 = time.time()
            while time.time() - t0 < 3.0 and self.odom_h.get("z") is None:
                time.sleep(0.1)
            self.body_height = self.odom_h.get("z") or 0.26
        print(
            f"[line_monitor] started ns={self.ns} target={self.target_dist:.2f}m "
            f"body_h={self.body_height:.3f} max_vy={self.max_vy:.3f}"
        )
        return True

    def _spin_loop(self):
        while self.node is not None and self.executor is not None and rclpy.ok():
            try:
                self.executor.spin_once(timeout_sec=0.1)
            except Exception:
                if self.node is not None:
                    time.sleep(0.05)
                else:
                    break

    def close(self):
        node = self.node
        self.node = None
        spin = self.spin
        self.spin = None
        if spin is not None and spin.is_alive():
            spin.join(timeout=1.0)
        executor = self.executor
        self.executor = None
        if executor is not None:
            if node is not None:
                try:
                    executor.remove_node(node)
                except Exception:
                    pass
            try:
                executor.shutdown()
            except Exception:
                pass
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        if self._owns_rclpy and rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass
        self._owns_rclpy = False

    def reset(self, reason="reset"):
        self.samples.clear()
        self.last_vy = 0.0
        self.last_info = {
            "ok": False,
            "left": None,
            "right": None,
            "err": 0.0,
            "vy": 0.0,
            "reason": reason,
        }

    def set_side(self, side):
        if side not in ("left", "right", "both"):
            raise ValueError("side must be 'left', 'right' or 'both'")
        if self.cfg["side"] != side:
            self.cfg["side"] = side
            self.reset("side_switch")

    def _measure_side(self, side, msg=None):
        msg = self.frames.get(side) if msg is None else msg
        if msg is None:
            return None
        try:
            img = image_to_bgr(msg)
        except Exception:
            return None
        if self.cfg["remote_url"]:
            try:
                ok, encoded = cv2.imencode(
                    ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                if not ok:
                    return None
                remote_parts = urlsplit(self.cfg["remote_url"])
                remote_query = parse_qsl(remote_parts.query, keep_blank_values=True)
                remote_query = [item for item in remote_query if item[0] != "side"]
                remote_query.append(("side", side))
                remote_url = urlunsplit(
                    (
                        remote_parts.scheme,
                        remote_parts.netloc,
                        remote_parts.path,
                        urlencode(remote_query),
                        remote_parts.fragment,
                    )
                )
                request = Request(
                    remote_url,
                    data=encoded.tobytes(),
                    headers={"Content-Type": "image/jpeg"},
                    method="POST",
                )
                with urlopen(request, timeout=self.cfg["remote_timeout"]) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                distance = payload.get("distance")
                return None if distance is None else abs(float(distance))
            except Exception:
                return None

        dbg = None
        if self.cfg["debug_dir"]:
            os.makedirs(self.cfg["debug_dir"], exist_ok=True)
            dbg = os.path.join(self.cfg["debug_dir"], f"{side}_dbg.png")
        sc, off = self.cfg["cal"][side]
        dist, _ = measure_vectorized(
            img, side, self.body_height,
            self.cfg["roi_bottom"], self.cfg["thresh_k"],
            self.cfg["hsv"], dbg, self.cfg["step"], sc, off,
            line_mode=self.cfg["line_mode"],
        )
        return None if dist is None else abs(float(dist))

    def measure_distance(self, side, msg=None):
        """Return the latest calibrated body-centre distance for one camera."""
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        return self._measure_side(side, msg=msg)

    def correction(self):
        now = time.time()
        if now - self.last_update < self.period:
            return self.last_vy, self.last_info
        self.last_update = now

        left = self._measure_side("left") if self.cfg["side"] in ("both", "left") else None
        right = self._measure_side("right") if self.cfg["side"] in ("both", "right") else None

        invalid_sides = []
        for side, distance in (("left", left), ("right", right)):
            if distance is None:
                continue
            if (
                self.valid_distance_min is not None
                and distance < self.valid_distance_min
            ):
                invalid_sides.append("{}:{:.2f}<min".format(side, distance))
            if (
                self.valid_distance_max is not None
                and distance > self.valid_distance_max
            ):
                invalid_sides.append("{}:{:.2f}>max".format(side, distance))
        if invalid_sides:
            self.samples.append(None)
            self.last_vy = 0.0
            self.last_info = {
                "ok": False,
                "left": left,
                "right": right,
                "err": 0.0,
                "vy": 0.0,
                "reason": "bad_distance:" + ",".join(invalid_sides),
            }
            return self.last_vy, self.last_info

        errs = []
        if left is not None:
            errs.append(left - self.target_dist)
        if right is not None:
            errs.append(self.target_dist - right)
        if not errs:
            self.samples.append(None)
            self.last_vy = 0.0
            self.last_info = {
                "ok": False,
                "left": left,
                "right": right,
                "err": 0.0,
                "vy": 0.0,
                "reason": "no_line",
            }
            return self.last_vy, self.last_info

        if left is not None and right is not None:
            width = left + right
            target_width = 2.0 * self.target_dist
            if (
                width < target_width - self.width_tol_low
                or width > target_width + self.width_tol_high
            ):
                self.samples.append(None)
                self.last_vy = 0.0
                self.last_info = {
                    "ok": False,
                    "left": left,
                    "right": right,
                    "err": 0.0,
                    "vy": 0.0,
                    "reason": f"bad_width:{width:.2f}",
                }
                return self.last_vy, self.last_info

        if (
            self.cfg["side"] != "right"
            and left is None
            and right is not None
            and right >= self.target_dist
        ):
            self.samples.append(None)
            self.last_vy = 0.0
            self.last_info = {
                "ok": False,
                "left": left,
                "right": right,
                "err": 0.0,
                "vy": 0.0,
                "reason": "right_not_close",
            }
            return self.last_vy, self.last_info

        if (
            self.cfg["side"] != "left"
            and right is None
            and left is not None
            and left >= self.target_dist
        ):
            self.samples.append(None)
            self.last_vy = 0.0
            self.last_info = {
                "ok": False,
                "left": left,
                "right": right,
                "err": 0.0,
                "vy": 0.0,
                "reason": "left_not_close",
            }
            return self.last_vy, self.last_info

        err = sum(errs) / float(len(errs))
        self.samples.append({
            "left": left,
            "right": right,
            "err": err,
        })
        valid = [s for s in self.samples if s is not None]
        if len(valid) < self.min_valid:
            self.last_vy = 0.0
            self.last_info = {
                "ok": False,
                "left": left,
                "right": right,
                "err": err,
                "vy": 0.0,
                "reason": f"warming:{len(valid)}/{self.min_valid}",
            }
            return self.last_vy, self.last_info

        avg_err = sum(s["err"] for s in valid) / float(len(valid))
        avg_lefts = [s["left"] for s in valid if s["left"] is not None]
        avg_rights = [s["right"] for s in valid if s["right"] is not None]
        avg_left = sum(avg_lefts) / float(len(avg_lefts)) if avg_lefts else None
        avg_right = sum(avg_rights) / float(len(avg_rights)) if avg_rights else None
        control_err = 0.0 if abs(avg_err) < self.deadband else avg_err
        target_vy = max(-self.max_vy, min(self.max_vy, self.k_vy * control_err))
        delta_vy = target_vy - self.last_vy
        if abs(delta_vy) > self.max_vy_step:
            target_vy = self.last_vy + self.max_vy_step * (1.0 if delta_vy > 0.0 else -1.0)
        vy = max(-self.max_vy, min(self.max_vy, target_vy))
        self.last_vy = vy
        self.last_info = {
            "ok": True,
            "left": avg_left,
            "right": avg_right,
            "err": avg_err,
            "vy": vy,
            "reason": f"ok:{len(valid)}/{self.window}",
        }
        return self.last_vy, self.last_info


if __name__ == "__main__":
    raise SystemExit(main())
