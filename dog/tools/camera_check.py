import os
import shlex
import sys


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
    cmd = f"source {shlex.quote(setup)} && cd {shlex.quote(os.path.dirname(script))} && exec python3 {shlex.quote(script)} {args}"
    os.execve("/bin/bash", ["bash", "-lc", cmd], env)


_ensure_ros_env()

import rclpy
from orange import OrangeDetector


def main():
    rclpy.init()
    detector = OrangeDetector()
    detector.start()
    ready = detector.wait_ready(timeout=6.0)
    print(f"[camera_check] image_topic={detector._img_topic}")
    print(f"[camera_check] active_image_topic={detector._active_img_topic}")
    print(f"[camera_check] scan_topic={detector._scan_topic}")
    print(f"[camera_check] ready={ready}")
    if ready:
        print(f"[camera_check] orange_blobs={len(detector.get_orange_detections())}")
        print(f"[camera_check] blue_blobs={len(detector.get_blue_detections())}")
    else:
        print("[camera_check] ERROR: image and scan are not both publishing frames")
        print("[camera_check] If scan is OK but image is missing, activate/restart stereo_camera first.")
    detector.stop()
    rclpy.shutdown()
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
