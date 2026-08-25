#!/usr/bin/env python3
"""Save one current image from each CyberDog fish-eye camera."""

import os
import shlex
import sys
import time


def _ensure_ros_env():
    if os.environ.get("ROS_DISTRO") or os.name != "posix":
        return
    if os.environ.get("CYBERDOG_ROS_REEXEC"):
        return
    candidates = [
        "/opt/ros2/cyberdog/setup.bash",
        "/opt/ros2/galactic/setup.bash",
        "/opt/ros/foxy/setup.bash",
    ]
    setup = next((path for path in candidates if os.path.exists(path)), None)
    if setup is None:
        return
    env = dict(os.environ)
    env["CYBERDOG_ROS_REEXEC"] = "1"
    script = os.path.abspath(__file__)
    args = " ".join(shlex.quote(arg) for arg in sys.argv[1:])
    command = "source {} && cd {} && exec python3 {} {}".format(
        shlex.quote(setup),
        shlex.quote(os.path.dirname(script)),
        shlex.quote(script),
        args,
    )
    os.execve("/bin/bash", ["bash", "-lc", command], env)


_ensure_ros_env()

import rclpy
from sensor_msgs.msg import Image

from line_monitor import detect_namespace, image_to_bgr


def main():
    output_dir = "/home/mi/cyberdog_competition/fisheye_snapshots"
    os.makedirs(output_dir, exist_ok=True)
    frames = {}

    rclpy.init()
    node = rclpy.create_node("fisheye_snapshot")
    namespace = detect_namespace(node, timeout=8.0)
    if not namespace:
        print("[fisheye_snapshot] ERROR: fish-eye namespace not found")
        node.destroy_node()
        rclpy.shutdown()
        return 2

    def save_frame(side, message):
        if side in frames:
            return
        try:
            import cv2

            path = os.path.join(output_dir, "{}.jpg".format(side))
            if cv2.imwrite(path, image_to_bgr(message)):
                frames[side] = path
                print("[fisheye_snapshot] saved {}={}".format(side, path), flush=True)
        except Exception as error:
            print("[fisheye_snapshot] WARN {}={}".format(side, repr(error)), flush=True)

    node.create_subscription(
        Image,
        namespace + "/image_left",
        lambda message: save_frame("left", message),
        1,
    )
    node.create_subscription(
        Image,
        namespace + "/image_right",
        lambda message: save_frame("right", message),
        1,
    )

    deadline = time.time() + 12.0
    while time.time() < deadline and len(frames) < 2:
        rclpy.spin_once(node, timeout_sec=0.2)

    node.destroy_node()
    rclpy.shutdown()
    if len(frames) != 2:
        print("[fisheye_snapshot] ERROR: captured {} of 2 frames".format(len(frames)))
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
