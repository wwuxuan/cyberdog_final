import argparse
import math
import os
import shlex
import sys
import threading
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
    cmd = (
        f"source {shlex.quote(setup)} && "
        f"cd {shlex.quote(os.path.dirname(script))} && "
        f"exec python3 {shlex.quote(script)} {args}"
    )
    os.execve("/bin/bash", ["bash", "-lc", cmd], env)


_ensure_ros_env()

import rclpy

from adapter import RealDogAdapter
from orange import OrangeDetector


S2_VISUAL_WINDOW_DEG = 32.0
S2_VISUAL_MIN_AREA = 180.0
S2_BLOB_MIN_AREA = 180.0
S2_VISUAL_REF_RADIUS_MIN = 70.0
S2_VISUAL_REF_RADIUS_MAX = 160.0
S2_VISUAL_REF_AREA_MIN = 20000.0
S2_VISUAL_REF_AREA_MAX = 55000.0


def _set_stand_command(adapter):
    with adapter._cmd_lock:
        adapter.cmd.mode = 12
        adapter.cmd.gait_id = 0
        adapter.cmd.vel_des = [0.0, 0.0, 0.0]


def _set_walk_command(adapter):
    with adapter._cmd_lock:
        adapter.cmd.mode = 11
        adapter.cmd.gait_id = 26
        adapter.cmd.vel_des = [0.0, 0.0, 0.0]


def _start_stand_guard(adapter, interval=0.18):
    stop_event = threading.Event()

    def _loop():
        while not stop_event.wait(interval):
            _set_stand_command(adapter)

    _set_stand_command(adapter)
    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return stop_event, thread


def _fmt_candidate(item):
    if not item:
        return "none"
    return (
        f"src={item.get('source', 'ball_candidate')} "
        f"ang={item['angle_deg']:.1f}deg area={item['area']:.0f} "
        f"r={item.get('radius', 0.0):.1f} "
        f"score={item.get('score', 0.0):.1f} "
        f"circ={item['circularity']:.2f} fill={item['fill_ratio']:.2f}"
    )


def _size_ok(item):
    if not item:
        return False
    radius = float(item.get("radius", 0.0))
    area = float(item.get("area", 0.0))
    return (
        S2_VISUAL_REF_RADIUS_MIN <= radius <= S2_VISUAL_REF_RADIUS_MAX
        and S2_VISUAL_REF_AREA_MIN <= area <= S2_VISUAL_REF_AREA_MAX
    )


def _choose_reference(detector, rgb):
    info = detector.classify_center_ball(
        expected_angle_deg=0.0,
        target_window_deg=S2_VISUAL_WINDOW_DEG,
    )

    candidates = []
    ball = info.get("orange")
    if ball and ball.get("area", 0.0) >= S2_VISUAL_MIN_AREA and _size_ok(ball):
        ball["source"] = ball.get("source", "ball_candidate")
        candidates.append(ball)

    if hasattr(detector, "find_center_color_blob"):
        blob = detector.find_center_color_blob(
            color="orange",
            target_window_deg=S2_VISUAL_WINDOW_DEG,
            min_area=S2_BLOB_MIN_AREA,
            rgb=rgb,
        )
        if blob is not None and _size_ok(blob):
            candidates.append(blob)

    selected = max(candidates, key=lambda c: c.get("score", 0.0)) if candidates else None
    debug_info = dict(info)
    debug_info["rgb"] = rgb
    debug_info["center_candidates"] = candidates
    debug_info["selected"] = selected
    return debug_info, selected


def main():
    parser = argparse.ArgumentParser(description="Snapshot stage2 RGB and run the same yellow-ball reference detection.")
    parser.add_argument("--output", default="stage2_debug/snapshot.jpg")
    parser.add_argument("--save-raw", default=None, help="optional raw RGB output path")
    parser.add_argument("--wait-ready", type=float, default=6.0)
    parser.add_argument("--settle", type=float, default=0.35)
    parser.add_argument("--no-stand", action="store_true", help="do not hold stand before capture")
    args = parser.parse_args()

    try:
        rclpy.init()
    except RuntimeError:
        pass

    adapter = None
    stand_stop = None
    stand_thread = None
    detector = OrangeDetector()
    try:
        if not args.no_stand:
            adapter = RealDogAdapter(None)
            if not adapter.wait_odom(timeout=3.0):
                print("[stage2_snapshot] ERROR: no odom")
                return 2
            stand_stop, stand_thread = _start_stand_guard(adapter)
            print("[stage2_snapshot] stand_guard=on")
        else:
            print("[stage2_snapshot] stand_guard=off")

        detector.start()
        print("[stage2_snapshot] camera topic must be /image_rgb")
        if not detector.wait_ready(timeout=args.wait_ready):
            print("[stage2_snapshot] ERROR: image_rgb not ready")
            return 2
        time.sleep(max(0.0, args.settle))

        rgb = detector.get_latest_rgb()
        if rgb is None:
            print("[stage2_snapshot] ERROR: no RGB frame")
            return 3

        debug_info, selected = _choose_reference(detector, rgb)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        ok = detector.save_debug_image(os.path.abspath(args.output), debug_info, title="stage2_snapshot")
        if args.save_raw:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_raw)), exist_ok=True)
            import cv2

            cv2.imwrite(os.path.abspath(args.save_raw), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        print(f"[stage2_snapshot] image_topic={detector._img_topic}")
        print(f"[stage2_snapshot] active_image_topic={detector._active_img_topic}")
        print(f"[stage2_snapshot] output={os.path.abspath(args.output)} saved={ok}")
        print(f"[stage2_snapshot] reference={'none' if selected is None else _fmt_candidate(selected)}")
        if selected is None:
            print("[stage2_snapshot] result=no_confirmed_yellow")
            return 1
        print("[stage2_snapshot] result=yellow_confirmed")
        return 0
    finally:
        if stand_stop is not None:
            stand_stop.set()
        if stand_thread is not None:
            stand_thread.join(timeout=1.0)
        if adapter is not None:
            _set_walk_command(adapter)
            adapter.shutdown()
        detector.stop()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
