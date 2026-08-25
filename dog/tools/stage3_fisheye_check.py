import argparse
import math
import os
import shlex
import sys
import time

import cv2
import lcm
import numpy as np
import threading


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
    cmd = (
        f"source {shlex.quote(setup)} && "
        f"cd {shlex.quote(os.path.dirname(script))} && "
        f"exec python3 {shlex.quote(script)} {args}"
    )
    os.execve("/bin/bash", ["bash", "-lc", cmd], env)


_ensure_ros_env()

import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "motion", "utils"))
from robot_control_cmd_lcmt import robot_control_cmd_lcmt


YELLOW_LOWER = np.array([12, 55, 70], dtype=np.uint8)
YELLOW_UPPER = np.array([45, 255, 255], dtype=np.uint8)
LCM_URL = "udpm://239.255.76.67:7671?ttl=255"


def _image_to_rgb(msg):
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)
    encoding = (msg.encoding or "").lower()
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if height <= 0 or width <= 0 or step <= 0:
        return None
    try:
        rows = raw.reshape((height, step))
    except ValueError:
        return None

    if encoding in ("rgb8", "bgr8"):
        pixels = rows[:, :width * 3].reshape((height, width, 3))
        if encoding == "bgr8":
            pixels = pixels[:, :, ::-1]
        return np.ascontiguousarray(pixels.copy())
    if encoding in ("rgba8", "bgra8"):
        pixels = rows[:, :width * 4].reshape((height, width, 4))[:, :, :3]
        if encoding == "bgra8":
            pixels = pixels[:, :, ::-1]
        return np.ascontiguousarray(pixels.copy())
    return None


def _find_contours(mask):
    found = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(found) == 3:
        return found[1]
    return found[0]


class FisheyeGrabber:
    def __init__(self, left_topic=None, right_topic=None):
        self.left_topic = left_topic
        self.right_topic = right_topic
        self.left_msg = None
        self.right_msg = None
        self.seen = {}

    def choose_topics(self, node):
        topics = dict(node.get_topic_names_and_types())
        image_topics = [
            name for name, types in topics.items()
            if "sensor_msgs/msg/Image" in types
        ]
        lower = {name: name.lower() for name in image_topics}

        def pick(side):
            explicit = self.left_topic if side == "left" else self.right_topic
            if explicit:
                return explicit
            side_words = [side, "lf" if side == "left" else "rf"]
            candidates = []
            for name in image_topics:
                text = lower[name]
                if not any(word in text for word in side_words):
                    continue
                if not any(key in text for key in ("fish", "fisheye", "wide", "image")):
                    continue
                if "depth" in text or "compressed" in text:
                    continue
                score = 0
                if "fisheye" in text or "fish" in text:
                    score += 10
                if "rgb" in text or "color" in text:
                    score += 5
                if text.endswith("image_rgb") or text.endswith("image_raw"):
                    score += 3
                candidates.append((score, name))
            candidates.sort(reverse=True)
            return candidates[0][1] if candidates else None

        self.left_topic = pick("left")
        self.right_topic = pick("right")
        return image_topics

    def subscribe(self, node):
        image_topics = self.choose_topics(node)
        print(f"[fisheye_check] left_topic={self.left_topic}")
        print(f"[fisheye_check] right_topic={self.right_topic}")
        if not self.left_topic or not self.right_topic:
            print("[fisheye_check] image topics:")
            for name in sorted(image_topics):
                print(f"[fisheye_check] topic {name}")
            return False

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.VOLATILE
        node.create_subscription(Image, self.left_topic, self._left_cb, qos)
        node.create_subscription(Image, self.right_topic, self._right_cb, qos)
        return True

    def _left_cb(self, msg):
        self.left_msg = msg
        self.seen["left"] = time.time()

    def _right_cb(self, msg):
        self.right_msg = msg
        self.seen["right"] = time.time()

    def ready(self):
        return self.left_msg is not None and self.right_msg is not None


def analyze_yellow_line(rgb, side, roi_top_frac=0.45, edge_frac=0.40):
    h, w = rgb.shape[:2]
    y0 = int(h * roi_top_frac)
    roi = rgb[y0:h, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if side == "left":
        edge_x0, edge_x1 = 0, int(w * edge_frac)
        center_sign = 1.0
    else:
        edge_x0, edge_x1 = int(w * (1.0 - edge_frac)), w
        center_sign = -1.0

    edge_mask = mask[:, edge_x0:edge_x1]
    total_edge = max(edge_mask.size, 1)
    edge_ratio = float(np.count_nonzero(edge_mask)) / float(total_edge)

    best = None
    for contour in _find_contours(mask):
        area = float(cv2.contourArea(contour))
        if area < 80.0:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        aspect = float(bw) / max(float(bh), 1.0)
        if aspect < 0.25 and area < 250.0:
            continue
        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-6:
            cx = x + bw / 2.0
        else:
            cx = moments["m10"] / moments["m00"]
        if side == "left":
            edge_dist = cx / max(float(w), 1.0)
        else:
            edge_dist = (w - cx) / max(float(w), 1.0)
        score = area * (1.0 - min(edge_dist / max(edge_frac, 1e-6), 1.0) * 0.65)
        item = {
            "area": area,
            "bbox": (int(x), int(y + y0), int(bw), int(bh)),
            "cx": float(cx),
            "cy": float(y + y0 + bh / 2.0),
            "edge_dist": float(edge_dist),
            "score": float(score),
            "aspect": float(aspect),
        }
        if best is None or item["score"] > best["score"]:
            best = item

    close = False
    severity = 0.0
    if best is not None:
        edge_close = max(0.0, 1.0 - best["edge_dist"] / max(edge_frac, 1e-6))
        area_close = min(1.0, best["area"] / 2500.0)
        ratio_close = min(1.0, edge_ratio / 0.08)
        severity = max(edge_close * 0.75 + area_close * 0.25, ratio_close * 0.65)
        close = severity >= 0.35

    return {
        "side": side,
        "mask": mask,
        "roi_y0": y0,
        "edge_band": (edge_x0, edge_x1),
        "edge_ratio": edge_ratio,
        "best": best,
        "close": close,
        "severity": severity,
        "vy_sign": center_sign,
    }


def save_debug(rgb, result, path):
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    y0 = result["roi_y0"]
    edge_x0, edge_x1 = result["edge_band"]
    cv2.rectangle(img, (0, y0), (w - 1, h - 1), (255, 255, 255), 1)
    cv2.rectangle(img, (edge_x0, y0), (edge_x1, h - 1), (0, 255, 255), 2)

    best = result["best"]
    if best is not None:
        x, y, bw, bh = best["bbox"]
        color = (0, 255, 255) if result["close"] else (0, 180, 255)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 3)
        cv2.circle(img, (int(best["cx"]), int(best["cy"])), 5, color, -1)

    label = (
        f"{result['side']} close={result['close']} sev={result['severity']:.2f} "
        f"edge_ratio={result['edge_ratio']:.3f}"
    )
    cv2.putText(img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(path, img)


class PostureController:
    def __init__(self):
        self.lc = lcm.LCM(LCM_URL)
        self.msg = robot_control_cmd_lcmt()
        self.life_count = 0
        self._hold_running = False
        self._hold_thread = None

    def _publish(self, mode, gait_id, seconds, pos_z=0.0):
        self.msg.mode = int(mode)
        self.msg.gait_id = int(gait_id)
        self.msg.contact = 15
        self.msg.value = 0
        self.msg.duration = 0
        self.msg.vel_des = [0.0, 0.0, 0.0]
        self.msg.rpy_des = [0.0, 0.0, 0.0]
        self.msg.pos_des = [0.0, 0.0, float(pos_z)]
        self.msg.step_height = [0.09, 0.09]
        t0 = time.time()
        while time.time() - t0 < seconds:
            self.life_count = (self.life_count + 1) % 127
            self.msg.life_count = self.life_count
            self.lc.publish("robot_control_cmd", self.msg.encode())
            time.sleep(0.05)

    def stand(self, seconds=5.0):
        print("[fisheye_check] stand")
        self._publish(mode=12, gait_id=0, seconds=seconds)
        self._publish(mode=11, gait_id=26, seconds=0.2, pos_z=0.28)
        print("[fisheye_check] stand done")

    def lie_down(self, seconds=2.0):
        print("[fisheye_check] lie down")
        self._publish(mode=7, gait_id=0, seconds=seconds)
        print("[fisheye_check] lie down done")

    def start_stand_hold(self):
        if self._hold_running:
            return
        self.stand(seconds=5.0)
        self._hold_running = True
        self._hold_thread = threading.Thread(target=self._hold_loop, daemon=True)
        self._hold_thread.start()

    def _hold_loop(self):
        while self._hold_running:
            self._publish(mode=11, gait_id=26, seconds=0.2, pos_z=0.28)

    def stop_stand_hold(self):
        self._hold_running = False
        thread = self._hold_thread
        self._hold_thread = None
        if thread and thread.is_alive():
            thread.join(timeout=1.0)


def main():
    parser = argparse.ArgumentParser(description="Check yellow line proximity with left/right fisheye cameras.")
    parser.add_argument("--left-topic")
    parser.add_argument("--right-topic")
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--debug-dir", default="stage3_fisheye_debug")
    parser.add_argument("--roi-top-frac", type=float, default=0.45)
    parser.add_argument("--edge-frac", type=float, default=0.40)
    parser.add_argument("--vy-max", type=float, default=0.06)
    parser.add_argument("--wz-max", type=float, default=0.12)
    parser.add_argument("--no-stand", action="store_true")
    parser.add_argument("--no-lie-down", action="store_true")
    args = parser.parse_args()

    posture = PostureController()
    if not args.no_stand:
        posture.start_stand_hold()
        time.sleep(1.0)

    rclpy.init()
    node = rclpy.create_node("stage3_fisheye_check")
    grabber = FisheyeGrabber(args.left_topic, args.right_topic)
    try:
        if not grabber.subscribe(node):
            return 2

        t0 = time.time()
        while time.time() - t0 < args.duration:
            rclpy.spin_once(node, timeout_sec=0.1)
        if not grabber.ready():
            print(f"[fisheye_check] ERROR: frames not ready seen={grabber.seen}")
            return 3

        left_rgb = _image_to_rgb(grabber.left_msg)
        right_rgb = _image_to_rgb(grabber.right_msg)
        if left_rgb is None or right_rgb is None:
            print("[fisheye_check] ERROR: unsupported image encoding")
            return 4

        left = analyze_yellow_line(left_rgb, "left", args.roi_top_frac, args.edge_frac)
        right = analyze_yellow_line(right_rgb, "right", args.roi_top_frac, args.edge_frac)

        vy_repel = 0.0
        wz_repel = 0.0
        if left["close"]:
            vy_repel -= args.vy_max * left["severity"]
            wz_repel -= args.wz_max * left["severity"]
        if right["close"]:
            vy_repel += args.vy_max * right["severity"]
            wz_repel += args.wz_max * right["severity"]

        os.makedirs(args.debug_dir, exist_ok=True)
        left_path = os.path.abspath(os.path.join(args.debug_dir, "left.jpg"))
        right_path = os.path.abspath(os.path.join(args.debug_dir, "right.jpg"))
        save_debug(left_rgb, left, left_path)
        save_debug(right_rgb, right, right_path)

        for result in (left, right):
            best = result["best"]
            if best is None:
                best_text = "none"
            else:
                best_text = (
                    f"area={best['area']:.0f} edge_dist={best['edge_dist']:.2f} "
                    f"aspect={best['aspect']:.2f}"
                )
            print(
                f"[fisheye_check] {result['side']} close={result['close']} "
                f"severity={result['severity']:.2f} edge_ratio={result['edge_ratio']:.3f} "
                f"best={best_text}"
            )

        print(f"[fisheye_check] suggested body-frame repel vy={vy_repel:.3f} wz={wz_repel:.3f}")
        print(f"[fisheye_check] debug left={left_path}")
        print(f"[fisheye_check] debug right={right_path}")
        return 0

    finally:
        posture.stop_stand_hold()
        node.destroy_node()
        rclpy.shutdown()
        if not args.no_lie_down:
            posture.lie_down()


if __name__ == "__main__":
    raise SystemExit(main())
