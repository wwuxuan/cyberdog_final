#!/usr/bin/env python3
"""Run stages 1, 2, and 3 through the complete stage 3 exit."""

import argparse
import os
import shlex
import sys
import threading
import time
from collections import deque


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
    setup = next((path for path in setup_candidates if os.path.exists(path)), None)
    if setup is None:
        return
    env = dict(os.environ)
    env["CYBERDOG_ROS_REEXEC"] = "1"
    script = os.path.abspath(__file__)
    args = " ".join(shlex.quote(arg) for arg in sys.argv[1:])
    command = (
        f"source {shlex.quote(setup)} && "
        f"cd {shlex.quote(os.path.dirname(script))} && "
        f"exec python3 {shlex.quote(script)} {args}"
    )
    os.execve("/bin/bash", ["bash", "-lc", command], env)


_ensure_ros_env()

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)

from adapter import RealDogAdapter
from stage1 import (
    LOCAL_VISION_URL,
    _prewarm_cameras,
    run_stage1,
    run_transition,
)
from stage2 import S2_END, S3_START, _clamp, _yaw_err, s2_align_yaw, run_stage2
from stage3 import build_three_anchor_entry_path, run_stage3


STAGE3_START_TOL = 0.020
S2_TO_S3_LATERAL_VY = 0.10
S2_TO_S3_FORWARD_VX = 0.12
S2_TO_S3_AXIS_TIMEOUT = 12.0
S2_TO_S3_LATERAL_TIMEOUT = 60.0
STAGE3_TEST_EXIT_Y = 6.58
STAGE3_LOCAL_VISION_URL = "http://192.168.43.102:9877/measure"
STAGE3_ENTRY_LEFT_LINE_TARGET = 0.25
STAGE3_LINE_MONITOR_CONFIG = {
    "target_dist": 0.30,
    "k_vy": 0.18,
    "max_vy": 0.02,
    "hz": 6.0,
    "deadband": 0.03,
    "valid_distance_min": 0.08,
    "valid_distance_max": 0.48,
}
STAGE3_LEFT_LINE_CONFIG = {
    **STAGE3_LINE_MONITOR_CONFIG,
    "target_dist": STAGE3_ENTRY_LEFT_LINE_TARGET,
    "side": "left",
    "body_height": 0.235,
    "line_mode": "body_x",
    "calibration": {"left": (0.78, 0.0)},
}
STAGE3_LEFT_LINE_TOL = 0.025
STAGE3_LEFT_LINE_STABLE_FRAMES = 3
STAGE3_LEFT_LINE_MAX_VY = 0.08
STAGE3_LEFT_LINE_STALE_SECS = 0.30
STAGE3_LEFT_LINE_STALE_VY = 0.020
STAGE3_LINE_ACTIVE_START = 0.00
STAGE3_LINE_ACTIVE_END = 0.80
STAGE3_LINE_BIAS_MAX = 0.12
STAGE3_LINE_WZ_GATE = 0.16
STAGE3_PATH_SPACING = 0.08
STAGE3_TAIL_HEADING_START = 0.80
STAGE3_TAIL_HEADING_DEG = 90.0


def _hold_for_stage4_handoff(handoff_dir, timeout):
    if not handoff_dir:
        return True
    os.makedirs(handoff_dir, exist_ok=True)
    ready_path = os.path.join(handoff_dir, "stage3_ready")
    release_path = os.path.join(handoff_dir, "release_stage3")
    try:
        os.unlink(release_path)
    except FileNotFoundError:
        pass
    with open(ready_path, "w"):
        pass
    print("[s1_s2_s3] handoff ready: hold standing heartbeat for stage4", flush=True)
    deadline = time.time() + max(1.0, float(timeout))
    while time.time() < deadline:
        if os.path.exists(release_path):
            print("[s1_s2_s3] handoff released by stage4", flush=True)
            return True
        time.sleep(0.05)
    print("[s1_s2_s3] ERROR: stage4 handoff release timeout", flush=True)
    return False


def _pose_text(adapter):
    x, y, z = adapter.get_position()
    return f"pos=({x:.3f},{y:.3f},{z:.3f}) yaw={adapter.get_yaw_deg():.1f}"


def _align_stage3_plus_y(adapter, label, tol=0.8):
    if s2_align_yaw(adapter, 90.0, label=label, tol=tol):
        return True
    print(f"[s2_s3] ERROR: cannot align +y {_pose_text(adapter)}")
    return False


def _map_stage3_x_from_left_line(adapter):
    """Anchor the field x coordinate after the left-line distance is stable."""
    x, y, _ = adapter.get_position()
    yaw = adapter.get_yaw_deg()
    if not hasattr(adapter, "set_mapped_pose"):
        print("[s2_s3] WARN: adapter cannot map left-line x correction", flush=True)
        return False
    try:
        adapter.set_mapped_pose(S3_START[0], y, yaw, quiet=True)
    except TypeError:
        adapter.set_mapped_pose(S3_START[0], y, yaw)
    print(
        f"[s2_s3] left-line x anchor x={x:.3f}->{S3_START[0]:.3f} "
        f"y={y:.3f} yaw={yaw:.1f}",
        flush=True,
    )
    return True


def _frame_key(message):
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None:
        sec = getattr(stamp, "sec", 0)
        nanosec = getattr(stamp, "nanosec", 0)
        if sec or nanosec:
            return (sec, nanosec)
    return id(message)


class _AsyncLineMeasurement:
    """Measure one fisheye stream without blocking the motion loop."""

    def __init__(self, monitor, side):
        self.monitor = monitor
        self.side = side
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sample = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=0.10)

    def latest(self):
        with self._lock:
            return None if self._sample is None else dict(self._sample)

    def _run(self):
        last_key = None
        while not self._stop.is_set():
            message = self.monitor.frames.get(self.side)
            if message is None:
                self._stop.wait(0.02)
                continue
            frame_key = _frame_key(message)
            if frame_key == last_key:
                self._stop.wait(0.01)
                continue
            last_key = frame_key
            started = time.time()
            try:
                distance = self.monitor.measure_distance(self.side, msg=message)
            except Exception:
                distance = None
            finished = time.time()
            with self._lock:
                self._sample = {
                    "key": frame_key,
                    "distance": distance,
                    "time": finished,
                    "compute": finished - started,
                }


def _side_step_to_stage3_x(adapter, left_line_monitor=None):
    """Hold +y and use the left line to center the field x position."""
    if not _align_stage3_plus_y(adapter, "stage3_lateral_start"):
        return False
    started = time.time()
    previous_error = None
    last_log = 0.0
    left_line_stable = 0
    left_line_samples = deque(maxlen=STAGE3_LEFT_LINE_STABLE_FRAMES)
    last_left_sample_key = None
    last_left_distance = None
    last_left_time = None
    last_left_compute = None
    line_reader = None
    if left_line_monitor is not None:
        line_reader = _AsyncLineMeasurement(left_line_monitor, "left")
        line_reader.start()

    def finish(ok):
        if line_reader is not None:
            line_reader.close()
        return ok

    while time.time() - started < S2_TO_S3_LATERAL_TIMEOUT:
        x, y, _ = adapter.get_position()
        error = S3_START[0] - x
        if abs(error) <= STAGE3_START_TOL:
            adapter.stop()
            print(f"[s2_s3] lateral -x reached {_pose_text(adapter)}", flush=True)
            return finish(True)
        if previous_error is not None and previous_error * error < 0.0:
            adapter.stop()
            print(f"[s2_s3] lateral -x crossed target {_pose_text(adapter)}", flush=True)
            return finish(True)
        previous_error = error
        yaw_error = _yaw_err(90.0, adapter.get_yaw_deg())
        if abs(yaw_error) > 12.0:
            adapter.stop()
            if not _align_stage3_plus_y(adapter, "stage3_lateral_realign"):
                return finish(False)
            continue
        left_distance = last_left_distance
        body_vy = None
        control_source = "odom"
        if line_reader is not None:
            sample = line_reader.latest()
            if sample is not None and sample["key"] != last_left_sample_key:
                last_left_sample_key = sample["key"]
                last_left_time = sample["time"]
                last_left_compute = sample["compute"]
                measured = sample["distance"]
                if measured is None:
                    left_distance = None
                    last_left_distance = None
                    left_line_stable = 0
                    left_line_samples.clear()
                else:
                    left_distance = measured
                    last_left_distance = measured
                    left_line_samples.append(measured)
                    if abs(measured - left_line_monitor.target_dist) <= STAGE3_LEFT_LINE_TOL:
                        left_line_stable += 1
                    else:
                        left_line_stable = 0
                    if (
                        len(left_line_samples) == STAGE3_LEFT_LINE_STABLE_FRAMES
                        and max(left_line_samples) - min(left_line_samples) <= 0.04
                        and left_line_stable >= STAGE3_LEFT_LINE_STABLE_FRAMES
                    ):
                        adapter.stop()
                        _map_stage3_x_from_left_line(adapter)
                        print(
                            f"[s2_s3] lateral -x left-line calibrated="
                            f"{measured:.3f}m target="
                            f"{left_line_monitor.target_dist:.3f}m "
                            f"stable={left_line_stable} {_pose_text(adapter)}",
                            flush=True,
                        )
                        return finish(True)

            left_line_fresh = (
                last_left_distance is not None
                and last_left_time is not None
                and time.time() - last_left_time <= STAGE3_LEFT_LINE_STALE_SECS
            )
            if left_line_fresh:
                line_error = last_left_distance - left_line_monitor.target_dist
                body_vy = _clamp(
                    0.60 * line_error,
                    -STAGE3_LEFT_LINE_MAX_VY,
                    STAGE3_LEFT_LINE_MAX_VY,
                )
                control_source = "line"
            elif last_left_distance is not None:
                speed = _clamp(0.5 * abs(error), 0.008, STAGE3_LEFT_LINE_STALE_VY)
                body_vy = speed if error < 0.0 else -speed
                control_source = "odom_stale"
            else:
                body_vy = 0.0
                control_source = "wait_line"

        if body_vy is None:
            speed = _clamp(0.8 * abs(error), 0.04, S2_TO_S3_LATERAL_VY)
            body_vy = speed if error < 0.0 else -speed
        wz = _clamp(0.025 * yaw_error, -0.18, 0.18)
        adapter.walk(0.0, body_vy, wz)
        if time.time() - last_log >= 1.0:
            line_text = "--" if left_distance is None else f"{left_distance:.3f}"
            line_age = "--" if last_left_time is None else f"{time.time() - last_left_time:.2f}"
            line_compute = "--" if last_left_compute is None else f"{last_left_compute:.2f}"
            print(
                f"[s2_s3] lateral -x pos=({x:.3f},{y:.3f}) "
                f"err_x={error:+.3f} left-line={line_text} "
                f"target={left_line_monitor.target_dist:.3f} "
                f"vy={body_vy:+.3f} src={control_source} "
                f"stable={left_line_stable}/{STAGE3_LEFT_LINE_STABLE_FRAMES} "
                f"age={line_age}s compute={line_compute}s yaw={adapter.get_yaw_deg():.1f}",
                flush=True,
            )
            last_log = time.time()
        time.sleep(0.07)
    adapter.stop()
    print(f"[s2_s3] ERROR: lateral -x timeout {_pose_text(adapter)}")
    return finish(False)


def _walk_forward_to_stage3_y(adapter):
    """Hold +y and advance only after the lateral transition is complete."""
    if not _align_stage3_plus_y(adapter, "stage3_forward_start"):
        return False
    started = time.time()
    previous_error = None
    last_log = 0.0
    while time.time() - started < S2_TO_S3_AXIS_TIMEOUT:
        x, y, _ = adapter.get_position()
        error = S3_START[1] - y
        if abs(error) <= STAGE3_START_TOL:
            adapter.stop()
            print(f"[s2_s3] forward +y reached {_pose_text(adapter)}", flush=True)
            return True
        if previous_error is not None and previous_error * error < 0.0:
            adapter.stop()
            print(f"[s2_s3] forward +y crossed target {_pose_text(adapter)}", flush=True)
            return True
        previous_error = error
        yaw_error = _yaw_err(90.0, adapter.get_yaw_deg())
        if abs(yaw_error) > 12.0:
            adapter.stop()
            if not _align_stage3_plus_y(adapter, "stage3_forward_realign"):
                return False
            continue
        speed = _clamp(0.8 * abs(error), 0.04, S2_TO_S3_FORWARD_VX)
        wz = _clamp(0.025 * yaw_error, -0.18, 0.18)
        adapter.walk(speed if error > 0.0 else -speed, 0.0, wz)
        if time.time() - last_log >= 1.0:
            print(
                f"[s2_s3] forward +y pos=({x:.3f},{y:.3f}) "
                f"err_y={error:+.3f} yaw={adapter.get_yaw_deg():.1f}",
                flush=True,
            )
            last_log = time.time()
        time.sleep(0.07)
    adapter.stop()
    print(f"[s2_s3] ERROR: forward +y timeout {_pose_text(adapter)}")
    return False


def run_stage2_to_stage3(
        adapter, left_line_monitor=None, forward_first=True,
        transition_start=None):
    """Move from stage2 to stage3, optionally walking +y before side-stepping."""
    start_node = S2_END if transition_start is None else transition_start
    transition_order = (
        "face +y, walk +y first, then side-step -x with left-line x correction"
        if forward_first
        else "face +y, side-step -x with left-line x correction, then walk +y"
    )
    print(
        f"[s2_s3] transition from start=({start_node[0]:.3f},{start_node[1]:.3f}) "
        f"to S3_START=({S3_START[0]:.3f},{S3_START[1]:.3f}): {transition_order}",
        flush=True,
    )
    if forward_first:
        if not _walk_forward_to_stage3_y(adapter):
            return False
        if not _side_step_to_stage3_x(
                adapter, left_line_monitor=left_line_monitor):
            return False
    else:
        if not _side_step_to_stage3_x(
                adapter, left_line_monitor=left_line_monitor):
            return False
        if not _walk_forward_to_stage3_y(adapter):
            return False
    if not _align_stage3_plus_y(adapter, "stage3_start"):
        return False
    adapter.stop()
    print(
        f"[s2_s3] stopped at stage3 start {_pose_text(adapter)} "
        f"target=({S3_START[0]:.3f},{S3_START[1]:.3f}) yaw=90.0",
        flush=True,
    )
    return True


def run_stage3_standalone_profile(adapter, vision_url=None):
    """Run the exact control profile used by the standalone stage3 test."""
    from line_monitor import LineDistanceMonitor

    stage3_points, entry_join_progress = build_three_anchor_entry_path(
        STAGE3_PATH_SPACING
    )
    print(
        f"[s1_s2_s3] stage3 prewarm fish-eye at {_pose_text(adapter)} "
        f"then run to standalone exit_y={STAGE3_TEST_EXIT_Y:.2f}",
        flush=True,
    )
    print(
        "[s1_s2_s3] stage3 entry anchors "
        "(-0.203,4.280)->(-0.200,4.700)->(0.100,5.380); "
        f"left-fisheye first, dual-fisheye at prog={entry_join_progress:.2f}",
        flush=True,
    )
    stage3_config = dict(STAGE3_LINE_MONITOR_CONFIG)
    stage3_config["remote_url"] = vision_url or None
    stage3_config["side"] = "left"
    line_monitor = LineDistanceMonitor(**stage3_config)
    if not line_monitor.start(timeout=12.0, activate=True):
        print("[s1_s2_s3] WARN: stage3 fish-eye unavailable; continue without it")
        line_monitor.close()
        line_monitor = None
        stage3_config = None

    try:
        if not _align_stage3_plus_y(adapter, "stage3_curve_start", tol=2.5):
            return False
        ok = run_stage3(
            adapter,
            exit_y=STAGE3_TEST_EXIT_Y,
            control_hz=15.0,
            vy_enabled=False,
            line_monitor=line_monitor,
            line_monitor_config=(
                stage3_config if line_monitor is not None else None
            ),
            line_active_start=STAGE3_LINE_ACTIVE_START,
            line_active_end=STAGE3_LINE_ACTIVE_END,
            line_bias_max=STAGE3_LINE_BIAS_MAX,
            line_wz_gate=STAGE3_LINE_WZ_GATE,
            path_spacing=STAGE3_PATH_SPACING,
            tail_heading_start=STAGE3_TAIL_HEADING_START,
            tail_heading_deg=STAGE3_TAIL_HEADING_DEG,
            path_points=stage3_points,
            line_side_switch_progress=entry_join_progress,
            line_side_after="both",
            entry_turn_progress_end=0.36,
            entry_turn_wz_limit=0.26,
            entry_turn_slow_error_deg=6.0,
            entry_turn_min_vx=0.044,
        )
        print(f"[s1_s2_s3] stage3_ok={ok} final {_pose_text(adapter)}")
        return ok
    finally:
        if line_monitor is not None:
            line_monitor.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run stages 1, 2, and 3 through the complete stage3 exit."
    )
    parser.add_argument("--stand", action="store_true", help="stand before moving")
    parser.add_argument("--arm", action="store_true", help="enable real-dog motion")
    parser.add_argument("--stage1-timeout", type=float, default=90.0)
    parser.add_argument(
        "--vision-url", default=LOCAL_VISION_URL,
        help="local PC right-fisheye vision endpoint; empty uses dog-side vision",
    )
    parser.add_argument(
        "--stage3-vision-url", default=STAGE3_LOCAL_VISION_URL,
        help="stage3 local PC left/right vision endpoint; empty uses dog-side vision",
    )
    parser.add_argument(
        "--turn-calibration", choices=("map", "physical"), default="physical",
        help="map: reset pose from line readings; physical: side-step to both lines",
    )
    parser.add_argument("--handoff-dir", default="",
                        help="internal: keep stage3 heartbeat until stage4 is ready")
    parser.add_argument("--handoff-timeout", type=float, default=30.0,
                        help="internal: maximum stage4 handoff wait, seconds")
    args = parser.parse_args()

    print("[s1_s2_s3] place dog at the normal stage1 start, facing field +x")
    print(
        f"[s1_s2_s3] stage3 curve start is S3_START=({S3_START[0]:.3f},"
        f"{S3_START[1]:.3f}); final stop matches standalone stage3 y="
        f"{STAGE3_TEST_EXIT_Y:.2f}",
    )
    print(
        f"[s1_s2_s3] arm={args.arm} turn_calibration={args.turn_calibration} "
        f"vision={'local' if args.vision_url else 'dog'}",
    )
    print(
        f"[s1_s2_s3] stage3_vision="
        f"{'local' if args.stage3_vision_url else 'dog'}"
        + (
            f" url={args.stage3_vision_url}"
            if args.stage3_vision_url
            else ""
        ),
    )

    adapter = RealDogAdapter(None)
    line_monitor = None
    left_line_monitor = None
    detector = None
    try:
        if not adapter.wait_odom(timeout=3.0):
            print("[s1_s2_s3] ERROR: no odom")
            return 2

        # Before standing, warm the RGB path, then retain only the fish-eye
        # monitor.  It is released as soon as stage1 finishes its two line
        # calibrations; the already-warmed RGB/lidar node then runs through
        # stage2 and the final stage2-to-stage3 transition.
        line_monitor = _prewarm_cameras(
            keep_right_line_monitor=True,
            remote_vision_url=args.vision_url or None,
        )
        if line_monitor is None:
            print("[s1_s2_s3] ERROR: right fish-eye monitor is unavailable")
            return 3
        if args.stand:
            adapter.stand()
        adapter.set_origin()
        if not args.arm:
            print(f"[s1_s2_s3] dry-run only {_pose_text(adapter)}; add --arm to move")
            return 0

        stage1_ok, detector = run_stage1(
            adapter,
            line_monitor,
            args.turn_calibration,
            timeout=args.stage1_timeout,
        )
        if not stage1_ok or detector is None:
            print("[s1_s2_s3] ERROR: stage1 or RGB/lidar handoff failed")
            return 1
        line_monitor = None

        if not run_transition(adapter, detector):
            print("[s1_s2_s3] ERROR: cannot reach stage2 start")
            return 1

        print("[s1_s2_s3] start stage2 with the existing RGB/lidar node")
        if not run_stage2(adapter, detector=detector):
            print("[s1_s2_s3] ERROR: stage2 failed; do not enter stage3 transition")
            return 1

        # Release RGB/lidar before creating the left fish-eye node.  Keeping
        # both executors alive caused wait-set failures on the dog.
        detector.stop()
        detector = None
        time.sleep(0.25)

        from line_monitor import LineDistanceMonitor

        stage3_left_config = dict(STAGE3_LEFT_LINE_CONFIG)
        stage3_left_config["remote_url"] = args.stage3_vision_url or None
        left_line_monitor = LineDistanceMonitor(**stage3_left_config)
        if not left_line_monitor.start(timeout=12.0, activate=True):
            print("[s1_s2_s3] WARN: left fish-eye x calibration unavailable; use odom fallback")
            left_line_monitor.close()
            left_line_monitor = None
        try:
            if not run_stage2_to_stage3(adapter, left_line_monitor=left_line_monitor):
                return 1
        finally:
            if left_line_monitor is not None:
                left_line_monitor.close()
                left_line_monitor = None

        # Stage 3 uses a fresh fish-eye monitor after the transition node is
        # fully released.
        stage3_ok = run_stage3_standalone_profile(
            adapter,
            vision_url=args.stage3_vision_url or None,
        )
        if stage3_ok and args.handoff_dir:
            stage3_ok = _hold_for_stage4_handoff(
                args.handoff_dir,
                args.handoff_timeout,
            )
        return 0 if stage3_ok else 1
    except KeyboardInterrupt:
        print("[s1_s2_s3] interrupted")
        return 130
    finally:
        if detector is not None:
            detector.stop()
        if left_line_monitor is not None:
            left_line_monitor.close()
        if line_monitor is not None:
            line_monitor.close()
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
