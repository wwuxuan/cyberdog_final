#!/usr/bin/env python3
"""Resume at the stage-2 upper-left four-ball center, then complete stages 2-3."""
import argparse
import os
import shlex
import sys
import time


def _ensure_ros_env():
    if os.environ.get("ROS_DISTRO") or os.name != "posix":
        return
    if os.environ.get("CYBERDOG_ROS_REEXEC"):
        return
    setup_candidates = (
        "/etc/mi/ros2_env.conf",
        "/opt/ros2/cyberdog/setup.bash",
        "/opt/ros2/galactic/setup.bash",
    )
    setup = next((path for path in setup_candidates if os.path.exists(path)), None)
    if setup is None:
        return
    env = dict(os.environ)
    env["CYBERDOG_ROS_REEXEC"] = "1"
    command = "source {} && cd {} && exec python3 {} {}".format(
        shlex.quote(setup),
        shlex.quote(os.path.dirname(os.path.abspath(__file__))),
        shlex.quote(os.path.abspath(__file__)),
        " ".join(shlex.quote(argument) for argument in sys.argv[1:]),
    )
    os.execve("/bin/bash", ["bash", "-lc", command], env)


_ensure_ros_env()

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)

from adapter import RealDogAdapter
from orange import OrangeDetector
from run1to3 import (
    STAGE3_LEFT_LINE_CONFIG,
    STAGE3_LOCAL_VISION_URL,
    run_stage2_to_stage3,
    run_stage3_standalone_profile,
)
from stage2 import S2_UPPER_LEFT_CENTER, run_stage2


START_YAW = 90.0


def main():
    parser = argparse.ArgumentParser(
        description="Resume at the stage-2 upper-left four-ball center and finish stages 2-3."
    )
    parser.add_argument("--stand", action="store_true")
    parser.add_argument("--arm", action="store_true")
    parser.add_argument("--stage3-vision-url", default=STAGE3_LOCAL_VISION_URL)
    args = parser.parse_args()

    adapter = RealDogAdapter(None)
    detector = None
    left_line_monitor = None
    try:
        if not adapter.wait_odom(timeout=5.0):
            print("[s2_upper_left] ERROR: no odometry")
            return 2
        if args.stand:
            adapter.stand()

        adapter.set_mapped_pose(
            S2_UPPER_LEFT_CENTER[0], S2_UPPER_LEFT_CENTER[1], START_YAW
        )
        print(
            "[s2_upper_left] mapped current pose to upper-left four-ball center "
            "(%.3f, %.3f), yaw=+y" % S2_UPPER_LEFT_CENTER,
            flush=True,
        )
        if not args.arm:
            print("[s2_upper_left] dry run only; add --arm to move")
            return 0

        import rclpy
        if not rclpy.ok():
            rclpy.init()
        detector = OrangeDetector()
        detector.start()
        if not run_stage2(adapter, detector=detector):
            print("[s2_upper_left] ERROR: stage2 failed")
            return 1
        detector.stop()
        detector = None
        time.sleep(0.25)

        from line_monitor import LineDistanceMonitor

        line_config = dict(STAGE3_LEFT_LINE_CONFIG)
        line_config["remote_url"] = args.stage3_vision_url or None
        left_line_monitor = LineDistanceMonitor(**line_config)
        if not left_line_monitor.start(timeout=12.0, activate=True):
            print("[s2_upper_left] WARN: stage3 entry fish-eye unavailable; use odometry")
            left_line_monitor.close()
            left_line_monitor = None
        try:
            if not run_stage2_to_stage3(
                    adapter, left_line_monitor=left_line_monitor):
                print("[s2_upper_left] ERROR: stage2-to-stage3 transition failed")
                return 1
        finally:
            if left_line_monitor is not None:
                left_line_monitor.close()
                left_line_monitor = None

        if not run_stage3_standalone_profile(
                adapter, vision_url=args.stage3_vision_url or None):
            print("[s2_upper_left] ERROR: stage3 failed")
            return 1
        print("[s2_upper_left] stages 2-3 complete")
        return 0
    except KeyboardInterrupt:
        print("[s2_upper_left] interrupted")
        return 130
    finally:
        if detector is not None:
            detector.stop()
        if left_line_monitor is not None:
            left_line_monitor.close()
        try:
            import rclpy
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
        adapter.stop()
        adapter.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
