#!/usr/bin/env python3
"""Run stage1, then stop at the standalone stage2 rule-scan start point."""

import argparse
from collections import deque
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

from adapter import RealDogAdapter
from s1_control import (
    EXIT_Y,
    _clamp,
    _yaw_err,
    stage1_lane_keep,
)
from s2 import (
    BOARD_BALLS,
    S2_RULE_SCAN,
    _RadarPoseCorrector,
    s2_calibrate_node_from_front_pair,
    s2_walk_segment,
)


S2_ENTRY = (2.700, 0.92)
STAGE1_EXIT_GUARD_BALL = (3.300, 1.340)
STAGE1_CALIBRATION_X = 2.95
STAGE1_WALK_VX = 0.18
STAGE1_EXIT_VX = 0.16
TRANSITION_FORWARD_VX = 0.14
TRANSITION_LATERAL_VY = -0.10
TRANSITION_YAW = 90.0
TRANSITION_AXIS_TOL = 0.04
TRANSITION_FINAL_AXIS_TOL = 0.015
TRANSITION_FINAL_YAW_TOL = 0.8
PRETURN_POSE = (2.95, 0.00)
TURN_POSE = (3.17, 0.00)
RIGHT_LINE_PRETURN_DIST = 0.50
RIGHT_LINE_POSTTURN_DIST = 0.33
RIGHT_LINE_CALIBRATION = {"right": (1.04, -0.040)}
LOCAL_VISION_URL = "http://192.168.43.102:9876/measure"
LINE_SAMPLE_COUNT = 3
LINE_MAX_SAMPLE_SPREAD = 0.040
LINE_DISTANCE_TOL = 0.015
LINE_MAX_SIDE_SPEED = 0.035
LINE_STALE_FRAME_SECS = 1.0
# Only use the right-lower ball after reaching the final part of the +y exit.
# Earlier returns can be other compact objects in the lidar scan, while the
# known ball is still far ahead.
STAGE1_EXIT_BALL_ENABLE_Y = 0.65
STAGE1_EXIT_BALL_STOP_BODY_X = 0.60
STAGE1_EXIT_BALL_SLOW_BODY_X = 0.75
STAGE1_EXIT_BALL_MIN_VX = 0.045
STAGE1_EXIT_BALL_FALLBACK_Y = 0.82


def _pose_text(adapter):
    x, y, z = adapter.get_position()
    return f"pos=({x:.3f},{y:.3f},{z:.3f}) yaw={adapter.get_yaw_deg():.1f}"


def _prewarm_cameras(keep_right_line_monitor=False, remote_vision_url=None):
    """Start RGB and fish-eye streams before the dog is asked to stand."""
    import rclpy
    from line_monitor import LineDistanceMonitor
    from orange import OrangeDetector

    print("[stage1_s2] prewarm RGB, left fish-eye, and right fish-eye before stand")
    if not rclpy.ok():
        rclpy.init()
    rgb = OrangeDetector()
    rgb_ready = False
    fish_eye = None
    fish_eye_ready = False
    try:
        # Galactic on this dog can corrupt its global wait set if two nodes
        # spin concurrently.  Warm RGB first, destroy it, then keep only the
        # fish-eye node alive for the subsequent line calibration.
        rgb.start()
        deadline = time.time() + 5.0
        while time.time() < deadline:
            rgb_ready = rgb.ready()
            if rgb_ready:
                break
            time.sleep(0.05)
        rgb.stop()

        fish_eye = LineDistanceMonitor(
            side="both",
            body_height=0.235,
            line_mode="body_x",
            calibration=RIGHT_LINE_CALIBRATION,
            remote_url=remote_vision_url,
        )
        fish_eye_ready = fish_eye.start(timeout=12.0, activate=True)
        deadline = time.time() + 3.0
        while time.time() < deadline:
            left_ready = fish_eye.frames.get("left") is not None
            right_ready = fish_eye.frames.get("right") is not None
            if fish_eye_ready and left_ready and right_ready:
                break
            time.sleep(0.05)
        left_ready = fish_eye.frames.get("left") is not None
        right_ready = fish_eye.frames.get("right") is not None
        print(
            f"[stage1_s2] camera prewarm rgb={rgb_ready} (stopped) "
            f"left={left_ready} right={right_ready}",
            flush=True,
        )
    except Exception as exc:
        print(f"[stage1_s2] WARN: camera prewarm failed: {exc!r}")
    finally:
        rgb.stop()
        if fish_eye is not None and (not keep_right_line_monitor or not fish_eye_ready):
            fish_eye.close()
    return fish_eye if keep_right_line_monitor and fish_eye_ready else None


def _start_transition_radar():
    """Start the RGB/lidar node after the fish-eye node has been released."""
    import rclpy
    from orange import OrangeDetector

    if not rclpy.ok():
        rclpy.init()

    detector = OrangeDetector()
    try:
        print("[stage1_s2] start RGB/lidar after stage1 turn")
        detector.start()
        deadline = time.time() + 5.0
        rgb_ready = False
        scan_ready = False
        while time.time() < deadline:
            rgb_ready = detector.ready()
            scan_ready = getattr(detector, "_scan_seq", 0) > 0
            if rgb_ready and scan_ready:
                break
            time.sleep(0.05)
        print(
            f"[stage1_s2] radar ready rgb={rgb_ready} "
            f"scan={scan_ready}",
            flush=True,
        )
        if scan_ready:
            return detector
        detector.stop()
    except Exception as exc:
        print(f"[stage1_s2] WARN: cannot start transition radar: {exc!r}")
        detector.stop()
    return None


def _fresh_right_line_distance(line_monitor, last_frame_key):
    """Return one measurement only when the right fish-eye frame is new."""
    msg = line_monitor.frames.get("right")
    if msg is None:
        return None, last_frame_key
    header = getattr(msg, "header", None)
    msg_stamp = getattr(header, "stamp", None)
    if msg_stamp is None:
        frame_key = id(msg)
    else:
        frame_key = (int(msg_stamp.sec), int(msg_stamp.nanosec))
    if frame_key == last_frame_key:
        return None, last_frame_key
    return line_monitor.measure_distance("right", msg=msg), frame_key


def _stable_right_line_distance(line_monitor, label):
    """Wait for a robust right-line reading without timing out."""
    samples = deque(maxlen=LINE_SAMPLE_COUNT)
    last_log = 0.0
    last_frame_key = None
    while True:
        distance, last_frame_key = _fresh_right_line_distance(
            line_monitor, last_frame_key
        )
        if distance is not None:
            samples.append(float(distance))
        if len(samples) == LINE_SAMPLE_COUNT:
            ordered = sorted(samples)
            median = ordered[len(ordered) // 2]
            spread = ordered[-1] - ordered[0]
            if spread <= LINE_MAX_SAMPLE_SPREAD:
                print(
                    f"[stage1_s2] {label} right-line stable={median:.3f}m "
                    f"spread={spread:.3f} samples={LINE_SAMPLE_COUNT}",
                    flush=True,
                )
                return median
        if time.time() - last_log >= 0.8:
            last = "waiting" if distance is None else f"{distance:.3f}"
            print(
                f"[stage1_s2] {label} right-line={last} "
                f"samples={len(samples)}/{LINE_SAMPLE_COUNT}",
                flush=True,
            )
        time.sleep(0.10)


def _move_to_right_line_distance(adapter, line_monitor, target_distance, label):
    """Continuously side-step from fresh, filtered right fish-eye readings."""
    samples = deque(maxlen=LINE_SAMPLE_COUNT)
    last_frame_key = None
    last_frame_time = time.monotonic()
    last_log = 0.0
    motion_held = True
    while True:
        distance, next_frame_key = _fresh_right_line_distance(
            line_monitor, last_frame_key
        )
        now = time.monotonic()
        if next_frame_key != last_frame_key:
            last_frame_key = next_frame_key
            last_frame_time = now
            if distance is not None:
                samples.append(float(distance))
            elif not motion_held:
                adapter.stop()
                motion_held = True
                print(f"[stage1_s2] {label} wait for detectable right line", flush=True)

        if len(samples) == LINE_SAMPLE_COUNT and distance is not None:
            ordered = sorted(samples)
            median = ordered[len(ordered) // 2]
            spread = ordered[-1] - ordered[0]
            if spread <= LINE_MAX_SAMPLE_SPREAD:
                error = target_distance - median
                if abs(error) <= LINE_DISTANCE_TOL:
                    adapter.stop()
                    print(
                        f"[stage1_s2] {label} reached right-line={median:.3f}m "
                        f"target={target_distance:.3f}m spread={spread:.3f}",
                        flush=True,
                    )
                    return True

                # At both required headings, +body vy moves away from the
                # right line: field +y while facing +x, field -x while +y.
                speed = _clamp(0.45 * abs(error), 0.012, LINE_MAX_SIDE_SPEED)
                body_vy = speed if error > 0.0 else -speed
                adapter.walk(0.0, body_vy, 0.0)
                motion_held = False
                if now - last_log >= 0.8:
                    print(
                        f"[stage1_s2] {label} right={median:.3f} target={target_distance:.3f} "
                        f"err={error:+.3f} vy={body_vy:+.3f} spread={spread:.3f}",
                        flush=True,
                    )
                    last_log = now
            else:
                adapter.stop()
                motion_held = True
                if now - last_log >= 0.8:
                    print(
                        f"[stage1_s2] {label} wait stable spread={spread:.3f} "
                        f"limit={LINE_MAX_SAMPLE_SPREAD:.3f}",
                        flush=True,
                    )
                    last_log = now

        # This is a motion watchdog, not a calibration timeout: the script
        # remains in the loop but never continues walking on a frozen frame.
        if now - last_frame_time > LINE_STALE_FRAME_SECS and not motion_held:
            adapter.stop()
            motion_held = True
            print(f"[stage1_s2] {label} wait for new right fish-eye frame", flush=True)
        elif len(samples) < LINE_SAMPLE_COUNT and now - last_log >= 0.8:
            print(
                f"[stage1_s2] {label} collecting samples={len(samples)}/{LINE_SAMPLE_COUNT}",
                flush=True,
            )
            last_log = now
        time.sleep(0.02)


def _calibrate_turn_pose(adapter, line_monitor, mode):
    """Use two perpendicular right lines to reach the final turn pose."""
    print(
        f"[stage1_s2] turn calibration mode={mode} preturn={PRETURN_POSE} "
        f"final={TURN_POSE}",
        flush=True,
    )
    if mode == "physical":
        # A line distance has no information along the line.  First use
        # odometry to physically recover the known pre-turn x=2.95.
        if not adapter.align_x(PRETURN_POSE[0]):
            print("[stage1_s2] WARN: turn_preturn cannot align x=2.95")
            return False
        if not _move_to_right_line_distance(
                adapter, line_monitor, RIGHT_LINE_PRETURN_DIST, "turn_preturn"):
            return False
        adapter.set_mapped_pose(PRETURN_POSE[0], PRETURN_POSE[1], 0.0)
    else:
        preturn = _stable_right_line_distance(line_monitor, "turn_preturn")
        if preturn is None:
            return False
        # Facing +x, the right line lies at field y=-0.50.  Its measured
        # distance therefore corrects only the field y coordinate.
        current_x, _, _ = adapter.get_position()
        adapter.set_mapped_pose(current_x, preturn - RIGHT_LINE_PRETURN_DIST, 0.0)

    print(
        f"[stage1_s2] turn before_left_turn {_pose_text(adapter)} "
        f"expected_preturn=({PRETURN_POSE[0]:.3f},{PRETURN_POSE[1]:.3f})",
        flush=True,
    )
    if not adapter.align_yaw(90.0, tol=2.5, timeout=7.0):
        print("[stage1_s2] WARN: stage1 turn timeout")
        return False
    print(
        f"[stage1_s2] turn after_left_turn_before_postturn {_pose_text(adapter)}",
        flush=True,
    )

    if mode == "physical":
        if not _move_to_right_line_distance(
                adapter, line_monitor, RIGHT_LINE_POSTTURN_DIST, "turn_postturn"):
            return False
        adapter.set_mapped_pose(TURN_POSE[0], TURN_POSE[1], 90.0)
    else:
        postturn = _stable_right_line_distance(line_monitor, "turn_postturn")
        if postturn is None:
            return False
        # Facing +y, the right line lies at field x=3.50.  Its measured
        # distance therefore corrects only the field x coordinate.
        _, current_y, _ = adapter.get_position()
        adapter.set_mapped_pose(
            TURN_POSE[0] + (RIGHT_LINE_POSTTURN_DIST - postturn), current_y, 90.0
        )
    print(
        f"[stage1_s2] turn after_postturn {_pose_text(adapter)} "
        f"expected_final=({TURN_POSE[0]:.3f},{TURN_POSE[1]:.3f})",
        flush=True,
    )
    return True


def _front_ball_measurement(adapter, detector, ball, last_scan_seq):
    """Return one fresh body-frame observation for a known front ball."""
    if detector is None:
        return None, last_scan_seq
    x, y, _ = adapter.get_position()
    observations, scan_debug = detector.get_lidar_landmark_observations(
        x,
        y,
        adapter.get_yaw_deg(),
        (ball,),
        min_distance=0.25,
        max_distance=2.20,
        max_match_error=0.45,
        return_debug=True,
    )
    scan_seq = scan_debug.get("scan_seq", 0)
    if scan_seq == last_scan_seq:
        return None, last_scan_seq
    for observation in observations:
        if observation["landmark"] == ball:
            return observation, scan_seq
    return None, scan_seq


def _apply_front_ball_guard(adapter, detector, state, label):
    """Use the right-lower blue ball only as a forward stop guard."""
    _, field_y, _ = adapter.get_position()
    if field_y < STAGE1_EXIT_BALL_ENABLE_Y:
        return False, 1.0

    observation, state["scan_seq"] = _front_ball_measurement(
        adapter,
        detector,
        STAGE1_EXIT_GUARD_BALL,
        state["scan_seq"],
    )
    if observation is not None:
        state["body_x"] = float(observation["body_x"])
        state["body_y"] = float(observation["body_y"])
        state["seen_t"] = time.monotonic()
        state["match_error"] = float(observation.get("match_error", 0.0))

    body_x = state["body_x"]
    now = time.monotonic()
    if body_x is not None and now - state["seen_t"] <= 0.50:
        if body_x <= STAGE1_EXIT_BALL_STOP_BODY_X:
            adapter.stop()
            print(
                f"[stage1_s2] {label} radar stop right-lower blue "
                f"forward={body_x:.3f}m lateral={state['body_y']:+.3f}m "
                f"limit={STAGE1_EXIT_BALL_STOP_BODY_X:.3f}m",
                flush=True,
            )
            return True, 0.0
        speed_scale = 1.0
        if body_x < STAGE1_EXIT_BALL_SLOW_BODY_X:
            span = STAGE1_EXIT_BALL_SLOW_BODY_X - STAGE1_EXIT_BALL_STOP_BODY_X
            speed_scale = _clamp(
                (body_x - STAGE1_EXIT_BALL_STOP_BODY_X) / max(span, 1e-6),
                STAGE1_EXIT_BALL_MIN_VX / max(STAGE1_EXIT_VX, 1e-6),
                1.0,
            )
        if now - state["last_log"] >= 0.8:
            print(
                f"[stage1_s2] {label} radar right-lower blue "
                f"forward={body_x:.3f}m lateral={state['body_y']:+.3f}m "
                f"score={state['match_error']:.3f} speed={speed_scale:.2f}",
                flush=True,
            )
            state["last_log"] = now
        return False, speed_scale

    return False, 1.0


def run_stage1(adapter, line_monitor, calibration_mode, timeout=90.0):
    print("[stage1_s2] stage1 align field +x")
    adapter.align_yaw(0.0, tol=2.5, timeout=6.0)
    adapter.set_origin()
    _, target_y, _ = adapter.get_position()
    phase = "forward_x"
    exit_radar = None
    front_ball_guard = {
        "scan_seq": None,
        "body_x": None,
        "body_y": 0.0,
        "seen_t": 0.0,
        "match_error": 0.0,
        "last_log": 0.0,
    }
    started = time.time()
    last_log = 0.0
    print(f"[stage1_s2] stage1 start {_pose_text(adapter)}")

    while time.time() - started < timeout:
        x, y, _ = adapter.get_position()
        yaw = adapter.get_yaw_deg()
        now = time.time()
        if now - last_log >= 2.0:
            print(
                f"[stage1_s2] stage1 phase={phase} "
                f"pos=({x:.3f},{y:.3f}) yaw={yaw:.1f}",
                flush=True,
            )
            last_log = now

        if phase == "forward_x":
            if x >= STAGE1_CALIBRATION_X:
                adapter.stop()
                print(f"[stage1_s2] stage1 reached x={x:.3f}; calibrate then turn +y")
                if not _calibrate_turn_pose(adapter, line_monitor, calibration_mode):
                    print("[stage1_s2] WARN: turn calibration failed")
                    return False, exit_radar
                # The fish-eye task is complete.  Release it before spinning
                # the lidar node, then guard the short +y approach to the
                # right-lower blue ball without adding a sideways avoidance.
                line_monitor.close()
                exit_radar = _start_transition_radar()
                if exit_radar is None:
                    print("[stage1_s2] ERROR: lidar unavailable for stage1 exit guard")
                    return False, None
                phase = "forward_y"
                continue
            stage1_lane_keep(adapter, target_y)

        else:
            stopped, speed_scale = _apply_front_ball_guard(
                adapter,
                exit_radar,
                front_ball_guard,
                "stage1_exit",
            )
            if stopped:
                print(f"[stage1_s2] stage1 exit stopped by blue-ball guard {_pose_text(adapter)}")
                return True, exit_radar
            if y >= STAGE1_EXIT_BALL_FALLBACK_Y:
                adapter.stop()
                print(
                    "[stage1_s2] stage1 exit stop at forward hard limit "
                    f"y={STAGE1_EXIT_BALL_FALLBACK_Y:.3f}",
                    flush=True,
                )
                return True, exit_radar
            if y >= EXIT_Y:
                adapter.stop()
                print(f"[stage1_s2] stage1 done {_pose_text(adapter)}")
                return True, exit_radar
            wz = _clamp(0.025 * _yaw_err(90.0, yaw), -0.35, 0.35)
            adapter.walk(STAGE1_EXIT_VX * speed_scale, 0.0, wz)

        time.sleep(0.07)

    adapter.stop()
    print(f"[stage1_s2] ERROR: stage1 timeout {_pose_text(adapter)}")
    return False, exit_radar


def _align_transition_yaw(adapter, label, tol=2.5):
    if adapter.align_yaw(TRANSITION_YAW, tol=tol, timeout=6.0):
        return True
    print(f"[stage1_s2] ERROR: {label} cannot align +y {_pose_text(adapter)}")
    return False


def _walk_forward_to_y(adapter, target_y, label, timeout=15.0,
                       front_ball_guard=None, tol=TRANSITION_AXIS_TOL,
                       stop_on_overshoot=True):
    if not _align_transition_yaw(adapter, label):
        return False
    started = time.time()
    last_log = 0.0
    while time.time() - started < timeout:
        x, y, _ = adapter.get_position()
        err = target_y - y
        if abs(err) <= tol or (stop_on_overshoot and err < 0.0 and y > target_y):
            adapter.stop()
            print(f"[stage1_s2] {label} reached {_pose_text(adapter)}")
            return True
        yaw_err = _yaw_err(TRANSITION_YAW, adapter.get_yaw_deg())
        if abs(yaw_err) > 12.0:
            adapter.stop()
            if not _align_transition_yaw(adapter, f"{label}_realign"):
                return False
            continue
        wz = _clamp(0.025 * yaw_err, -0.20, 0.20)
        speed = _clamp(0.8 * abs(err), 0.06, TRANSITION_FORWARD_VX)
        if front_ball_guard is not None and err > 0.0:
            stopped, speed_scale = _apply_front_ball_guard(
                adapter,
                front_ball_guard["detector"],
                front_ball_guard["state"],
                label,
            )
            if stopped:
                print(f"[stage1_s2] {label} reached by blue-ball guard {_pose_text(adapter)}")
                return True
            speed *= speed_scale
        adapter.walk(speed if err > 0.0 else -speed, 0.0, wz)
        if time.time() - last_log >= 1.0:
            print(
                f"[stage1_s2] {label} pos=({x:.3f},{y:.3f}) "
                f"err_y={err:.3f} yaw={adapter.get_yaw_deg():.1f}",
                flush=True,
            )
            last_log = time.time()
        time.sleep(0.07)
    adapter.stop()
    print(f"[stage1_s2] ERROR: {label} timeout {_pose_text(adapter)}")
    return False


def _walk_lateral_to_x(adapter, target_x, label, timeout=12.0,
                       detector=None, radar_landmarks=None,
                       radar_corrector=None, radar_from_node=None,
                       radar_to_node=None, tol=TRANSITION_AXIS_TOL,
                       stop_on_overshoot=True):
    if not _align_transition_yaw(adapter, label):
        return False
    started = time.time()
    last_log = 0.0
    while time.time() - started < timeout:
        # The robot keeps its +y heading while moving sideways.  Treat the
        # upcoming +y lane as a virtual radar leg, so its front ball pair can
        # continuously correct the mapped x/y/yaw during this side-step.
        if (
                radar_corrector is not None
                and radar_from_node is not None
                and radar_to_node is not None):
            radar_corrector.update(
                adapter,
                detector,
                radar_landmarks,
                radar_from_node,
                radar_to_node,
                label=label,
            )
        x, y, _ = adapter.get_position()
        err = target_x - x
        if abs(err) <= tol or (stop_on_overshoot and err < 0.0 and x < target_x):
            adapter.stop()
            print(f"[stage1_s2] {label} reached {_pose_text(adapter)}")
            return True
        yaw_err = _yaw_err(TRANSITION_YAW, adapter.get_yaw_deg())
        if abs(yaw_err) > 12.0:
            adapter.stop()
            if not _align_transition_yaw(adapter, f"{label}_realign"):
                return False
            continue
        wz = _clamp(0.025 * yaw_err, -0.20, 0.20)
        speed = min(abs(TRANSITION_LATERAL_VY), max(0.05, 0.8 * abs(err)))
        # With field yaw=+y, positive body vy moves toward field -x.
        body_vy = speed if err < 0.0 else -speed
        adapter.walk(0.0, body_vy, wz)
        if time.time() - last_log >= 1.0:
            print(
                f"[stage1_s2] {label} pos=({x:.3f},{y:.3f}) "
                f"err_x={err:.3f} yaw={adapter.get_yaw_deg():.1f}",
                flush=True,
            )
            last_log = time.time()
        time.sleep(0.07)
    adapter.stop()
    print(f"[stage1_s2] ERROR: {label} timeout {_pose_text(adapter)}")
    return False


def _calibrate_transition_front_pair(adapter, detector, landmarks, corrector,
                                     from_node, node, label):
    """Map one stable pose correction from the two balls ahead of a node."""
    print(
        f"[stage1_s2] {label}: front-pair radar calibration at "
        f"({node[0]:.3f},{node[1]:.3f})",
        flush=True,
    )
    applied = s2_calibrate_node_from_front_pair(
        adapter,
        detector,
        landmarks,
        corrector,
        from_node,
        node,
    )
    print(
        f"[stage1_s2] {label}: front-pair radar "
        f"{'applied' if applied else 'skipped'} {_pose_text(adapter)}",
        flush=True,
    )
    return applied


def _refine_transition_pose(adapter, node, label):
    """Physically settle on the radar-corrected final node in both directions."""
    if not _walk_lateral_to_x(
            adapter,
            node[0],
            label=f"{label}_x",
            tol=TRANSITION_FINAL_AXIS_TOL,
            stop_on_overshoot=False):
        return False
    if not _walk_forward_to_y(
            adapter,
            node[1],
            label=f"{label}_y",
            tol=TRANSITION_FINAL_AXIS_TOL,
            stop_on_overshoot=False):
        return False
    return _align_transition_yaw(
        adapter,
        f"{label}_final",
        tol=TRANSITION_FINAL_YAW_TOL,
    )


def run_transition(adapter, detector):
    landmarks = set(BOARD_BALLS)
    radar_corrector = _RadarPoseCorrector()
    print(
        f"[stage1_s2] transition: lateral -x to x={S2_ENTRY[0]:.3f}, "
        f"then forward to y={S2_ENTRY[1]:.3f}, then forward to "
        f"stage2_test_start=({S2_RULE_SCAN[0]:.3f},{S2_RULE_SCAN[1]:.3f}) "
        "with front-pair radar calibration"
    )
    # Stage1's front-ball guard has already stopped the +y motion.  Move
    # sideways first to create clearance from the right-lower blue ball;
    # do not undo that protection by taking another forward step here.
    if not _walk_lateral_to_x(
            adapter,
            S2_ENTRY[0],
            label="transition_lateral_entry",
            detector=detector,
            radar_landmarks=landmarks,
            radar_corrector=radar_corrector,
            radar_from_node=S2_ENTRY,
            radar_to_node=S2_RULE_SCAN):
        return False
    if not _walk_forward_to_y(adapter, S2_ENTRY[1], label="transition_forward_entry"):
        return False
    _calibrate_transition_front_pair(
        adapter,
        detector,
        landmarks,
        radar_corrector,
        TURN_POSE,
        S2_ENTRY,
        "transition_entry",
    )

    # This is the same in-leg lidar correction used by stage2.  It keeps the
    # dog centered while approaching the lower-right four-ball center.
    if not s2_walk_segment(
            adapter,
            S2_ENTRY,
            S2_RULE_SCAN,
            detector=detector,
            radar_landmarks=landmarks,
            radar_corrector=radar_corrector):
        return False
    _calibrate_transition_front_pair(
        adapter,
        detector,
        landmarks,
        radar_corrector,
        S2_ENTRY,
        S2_RULE_SCAN,
        "transition_final",
    )
    if not _refine_transition_pose(adapter, S2_RULE_SCAN, "transition_final"):
        return False
    if not _align_transition_yaw(
            adapter,
            "transition_final",
            tol=TRANSITION_FINAL_YAW_TOL):
        return False
    adapter.stop()
    print(
        f"[stage1_s2] done at stage2_test_start {_pose_text(adapter)} "
        f"target=({S2_RULE_SCAN[0]:.3f},{S2_RULE_SCAN[1]:.3f}) yaw=90.0",
        flush=True,
    )
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run stage1 and finish at the stage2 standalone test start."
    )
    parser.add_argument("--stand", action="store_true", help="stand before moving")
    parser.add_argument("--arm", action="store_true", help="enable real-dog motion")
    parser.add_argument("--stage1-timeout", type=float, default=90.0)
    parser.add_argument(
        "--vision-url", default=LOCAL_VISION_URL,
        help="local PC right-fisheye vision endpoint; empty uses dog-side vision",
    )
    parser.add_argument(
        "--turn-calibration", choices=("map", "physical"), default="physical",
        help="map: reset pose from the two line readings; physical: side-step to both lines",
    )
    args = parser.parse_args()

    print("[stage1_s2] place dog at the normal stage1 start, facing field +x")
    print(
        f"[stage1_s2] final target is S2_RULE_SCAN=({S2_RULE_SCAN[0]:.3f},"
        f"{S2_RULE_SCAN[1]:.3f}), facing field +y"
    )
    print(
        f"[stage1_s2] arm={args.arm} turn_calibration={args.turn_calibration} "
        f"vision={'local' if args.vision_url else 'dog'}",
    )

    adapter = RealDogAdapter(None)
    line_monitor = None
    transition_radar = None
    try:
        if not adapter.wait_odom(timeout=3.0):
            print("[stage1_s2] ERROR: no odom")
            return 2
        line_monitor = _prewarm_cameras(
            keep_right_line_monitor=True,
            remote_vision_url=args.vision_url or None,
        )
        if line_monitor is None:
            print("[stage1_s2] ERROR: right fish-eye monitor is unavailable")
            return 3
        if args.stand:
            adapter.stand()
        adapter.set_origin()
        if not args.arm:
            print(f"[stage1_s2] dry-run only {_pose_text(adapter)}; add --arm to move")
            return 0
        stage1_ok, transition_radar = run_stage1(
            adapter,
            line_monitor,
            args.turn_calibration,
            timeout=args.stage1_timeout,
        )
        if not stage1_ok:
            return 1
        # Galactic on this dog is unreliable with two independently spinning
        # ROS nodes.  The line monitor is no longer needed after stage1, so
        # it was released before the +y exit guard started the RGB/lidar node.
        line_monitor = None
        if transition_radar is None:
            print("[stage1_s2] ERROR: lidar is unavailable for stage1 exit guard")
            return 3
        return 0 if run_transition(adapter, transition_radar) else 1
    except KeyboardInterrupt:
        print("[stage1_s2] interrupted")
        return 130
    finally:
        if transition_radar is not None:
            transition_radar.stop()
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
