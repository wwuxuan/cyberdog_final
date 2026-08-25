import argparse
import math
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
    cmd = (
        f"source {shlex.quote(setup)} && "
        f"cd {shlex.quote(os.path.dirname(script))} && "
        f"exec python3 {shlex.quote(script)} {args}"
    )
    os.execve("/bin/bash", ["bash", "-lc", cmd], env)


_ensure_ros_env()

from adapter import RealDogAdapter
from run1to3 import (
    STAGE3_LEFT_LINE_CONFIG,
    run_stage2_to_stage3,
)
from s3 import (
    _adjust_early_turn_y,
    _load_curve_points,
    _nearest_index,
    _path_heading,
    _resample_points,
    _wrap_to_pi,
    build_three_anchor_entry_path,
    run_stage3,
)


DEFAULT_FULL_EXIT_Y = 6.58
DEFAULT_SHORT_EXIT_Y = 5.20
DEFAULT_STAGE3_VISION_URL = "http://192.168.43.102:9877/measure"
STAGE2_UPPER_LEFT_CENTER = (0.300, 3.440)
class Stage3PoseAdapter:
    """Map current dog pose to the stage3 curve start in field coordinates."""

    def __init__(self, base, field_x, field_y, field_yaw_deg):
        self.base = base
        self.field_x = float(field_x)
        self.field_y = float(field_y)
        self.field_yaw_deg = float(field_yaw_deg)

    def get_position(self):
        local_x, local_y, local_z = self.base.get_position()
        yaw0 = math.radians(self.field_yaw_deg)
        field_x = (
            self.field_x
            + local_x * math.cos(yaw0)
            - local_y * math.sin(yaw0)
        )
        field_y = (
            self.field_y
            + local_x * math.sin(yaw0)
            + local_y * math.cos(yaw0)
        )
        return field_x, field_y, local_z

    def get_yaw_deg(self):
        return ((self.base.get_yaw_deg() + self.field_yaw_deg + 180.0) % 360.0) - 180.0

    def align_yaw(self, target_deg, *args, **kwargs):
        local_target = ((float(target_deg) - self.field_yaw_deg + 180.0) % 360.0) - 180.0
        return self.base.align_yaw(local_target, *args, **kwargs)

    def walk(self, vx, vy=0.0, wz=0.0):
        return self.base.walk(vx, vy, wz)

    def stop(self):
        return self.base.stop()

    def set_mapped_pose(self, field_x, field_y, field_yaw_deg, quiet=False):
        current_x, current_y, _ = self.get_position()
        current_yaw = self.get_yaw_deg()
        yaw_delta = ((float(field_yaw_deg) - current_yaw + 180.0) % 360.0) - 180.0
        self.field_yaw_deg = (
            (self.field_yaw_deg + yaw_delta + 180.0) % 360.0
        ) - 180.0
        rotated_x, rotated_y, _ = self.get_position()
        self.field_x += float(field_x) - rotated_x
        self.field_y += float(field_y) - rotated_y
        if not quiet:
            print(
                f"[stage3_test] mapped pose set pos=({float(field_x):.3f},"
                f"{float(field_y):.3f}) yaw={float(field_yaw_deg):.1f}",
                flush=True,
            )


def _curve_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "tools", "curve_fitting", "curve.txt")


def _print_pose_check(adapter, points, seconds):
    t0 = time.time()
    hint = 0
    while time.time() - t0 < seconds:
        x, y, _ = adapter.get_position()
        yaw_deg = adapter.get_yaw_deg()
        hint = _nearest_index(points, x, y, hint)
        xr, yr = points[hint]
        psi = _path_heading(points, hint)
        dx, dy = x - xr, y - yr
        e_y = -math.sin(psi) * dx + math.cos(psi) * dy
        e_psi = _wrap_to_pi(math.radians(yaw_deg) - psi)
        print(
            f"[stage3_test] pose x={x:.3f} y={y:.3f} yaw={yaw_deg:.1f} "
            f"nearest_idx={hint} ref=({xr:.3f},{yr:.3f}) "
            f"ey={e_y:.3f} epsi={math.degrees(e_psi):.1f}"
        )
        time.sleep(1.0)


def _hold_for_stage4_handoff(handoff_dir, timeout):
    """Keep the standing command alive until stage4 has its own heartbeat."""
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
    print(
        "[stage3_test] handoff ready: hold standing heartbeat for stage4",
        flush=True,
    )
    deadline = time.time() + max(1.0, float(timeout))
    while time.time() < deadline:
        if os.path.exists(release_path):
            print("[stage3_test] handoff released by stage4", flush=True)
            return True
        time.sleep(0.05)
    print("[stage3_test] ERROR: stage4 handoff release timeout", flush=True)
    return False


def main():
    parser = argparse.ArgumentParser(description="Standalone real-dog stage3 test.")
    parser.add_argument("--stand", action="store_true", help="stand first")
    parser.add_argument("--arm", action="store_true", help="actually run stage3 movement")
    parser.add_argument("--dry-secs", type=float, default=5.0)
    parser.add_argument("--exit-y", type=float, default=DEFAULT_SHORT_EXIT_Y)
    parser.add_argument("--full-stage", action="store_true", help="run to y=6.58")
    parser.add_argument(
        "--from-stage2-end",
        action="store_true",
        help="start at the stage2 upper-left four-ball center and transition to stage3",
    )
    parser.add_argument("--start-x", type=float, default=None)
    parser.add_argument("--start-y", type=float, default=None)
    parser.add_argument("--field-yaw", type=float, default=None)
    parser.add_argument("--with-vy", action="store_true", help="enable lateral velocity")
    parser.add_argument(
        "--vision-url",
        default=DEFAULT_STAGE3_VISION_URL,
        help="stage3 laptop vision endpoint; empty uses dog-side computation",
    )
    parser.add_argument("--no-line-correction", action="store_true",
                        help="disable fisheye yellow-line centering")
    parser.add_argument("--line-target", type=float, default=0.30,
                        help="target distance from body center to each yellow line, meters")
    parser.add_argument("--line-k-vy", type=float, default=0.18,
                        help="body-frame vy gain for yellow-line centering")
    parser.add_argument("--line-max-vy", type=float, default=0.02,
                        help="max body-frame vy added by yellow-line centering")
    parser.add_argument("--line-hz", type=float, default=6.0,
                        help="yellow-line correction update rate")
    parser.add_argument("--line-deadband", type=float, default=0.03,
                        help="ignore yellow-line distance error below this value, meters")
    parser.add_argument("--line-wz-gate", type=float, default=0.16,
                        help="disable yellow-line vy while abs(wz) is above this value")
    parser.add_argument("--line-active-start", type=float, default=0.00,
                        help="stage progress where fisheye correction starts")
    parser.add_argument("--line-active-end", type=float, default=0.80,
                        help="stage progress where fisheye correction stops")
    parser.add_argument("--line-bias-max", type=float, default=0.12,
                        help="max final path bias learned from fisheye correction, meters")
    parser.add_argument("--tail-heading-start", type=float, default=0.80,
                        help="progress where final heading starts blending to +y")
    parser.add_argument("--tail-heading-deg", type=float, default=90.0,
                        help="target field heading for the final exit")
    parser.add_argument("--path-spacing", type=float, default=0.08,
                        help="resampled curve point spacing for stage3 control, meters")
    parser.add_argument("--handoff-dir", default="",
                        help="internal: keep stage3 heartbeat until stage4 is ready")
    parser.add_argument("--handoff-timeout", type=float, default=30.0,
                        help="internal: maximum stage4 handoff wait, seconds")
    args = parser.parse_args()

    entry_join_progress = None
    if args.from_stage2_end:
        points, entry_join_progress = build_three_anchor_entry_path(args.path_spacing)
    else:
        points = _resample_points(
            _adjust_early_turn_y(_load_curve_points(_curve_path())),
            spacing=args.path_spacing,
        )
    if args.from_stage2_end:
        start_x, start_y = STAGE2_UPPER_LEFT_CENTER
        start_yaw = 90.0
    else:
        start_x = points[0][0] if args.start_x is None else args.start_x
        start_y = points[0][1] if args.start_y is None else args.start_y
        start_yaw = (
            math.degrees(_path_heading(points, 0))
            if args.field_yaw is None
            else args.field_yaw
        )
    exit_y = DEFAULT_FULL_EXIT_Y if args.full_stage else args.exit_y

    if args.from_stage2_end:
        print(
            "[stage3_test] place dog at stage2 upper-left four-ball center "
            "(0.300,3.440), facing field +y"
        )
        print(
            "[stage3_test] entry anchors "
            "(-0.203,4.280)->(-0.200,4.700)->(0.100,5.380); "
            f"left-fisheye starts at prog=0.00; "
            f"dual-fisheye starts at prog={entry_join_progress:.2f}"
        )
    else:
        print("[stage3_test] place dog at stage3 curve start, facing field +y")
    print(
        f"[stage3_test] current pose will map to "
        f"({start_x:.3f},{start_y:.3f}) yaw={start_yaw:.1f}"
    )
    print(f"[stage3_test] arm={args.arm} exit_y={exit_y:.3f}")
    print(
        f"[stage3_test] vision={'local' if args.vision_url else 'dog'}"
        + (f" url={args.vision_url}" if args.vision_url else "")
    )

    base = RealDogAdapter(None)
    line_monitor_config = None
    try:
        if not base.wait_odom(timeout=3.0):
            print("[stage3_test] ERROR: no odom")
            return 2
        if args.stand:
            base.stand()
        base.set_origin()
        adapter = Stage3PoseAdapter(base, start_x, start_y, start_yaw)

        _print_pose_check(adapter, points, seconds=max(0.1, args.dry_secs))

        if not args.arm:
            print("[stage3_test] dry-run only; add --arm to move")
            return 0

        if not args.no_line_correction:
            line_monitor_config = {
                "target_dist": args.line_target,
                "k_vy": args.line_k_vy,
                "max_vy": args.line_max_vy,
                "hz": args.line_hz,
                "deadband": args.line_deadband,
                "valid_distance_min": 0.08,
                "valid_distance_max": 0.48,
                "remote_url": args.vision_url or None,
            }

        # ROS/CV initialisation can briefly monopolise the interpreter.  Warm
        # the left monitor first when the run includes the stage2-to-stage3
        # transition; stage3 starts its own monitor after the transition.
        line_monitor = None
        transition_line_monitor = None
        if args.from_stage2_end and not args.no_line_correction:
            from line_monitor import LineDistanceMonitor

            print("[stage3_test] prewarm left fisheye for stage2-to-stage3 transition")
            transition_config = dict(STAGE3_LEFT_LINE_CONFIG)
            transition_config["remote_url"] = args.vision_url or None
            transition_line_monitor = LineDistanceMonitor(**transition_config)
            if not transition_line_monitor.start(timeout=12.0, activate=True):
                print("[stage3_test] WARN: left fisheye unavailable; use odom transition")
                transition_line_monitor.close()
                transition_line_monitor = None

        elif line_monitor_config is not None:
            from line_monitor import LineDistanceMonitor

            print("[stage3_test] prewarm fisheye monitor while standing")
            line_monitor = LineDistanceMonitor(**line_monitor_config)
            if not line_monitor.start(timeout=12.0, activate=True):
                print("[stage3_test] WARN: fisheye monitor unavailable; continue without it")
                line_monitor.close()
                line_monitor = None
                line_monitor_config = None

        adapter.align_yaw(start_yaw, tol=2.5, timeout=6.0)
        if args.from_stage2_end:
            try:
                if not run_stage2_to_stage3(
                    adapter,
                    left_line_monitor=transition_line_monitor,
                    forward_first=True,
                    transition_start=STAGE2_UPPER_LEFT_CENTER,
                ):
                    return 1
            finally:
                if transition_line_monitor is not None:
                    transition_line_monitor.close()
                    transition_line_monitor = None

            if line_monitor_config is not None:
                from line_monitor import LineDistanceMonitor

                stage3_monitor_config = dict(line_monitor_config)
                if entry_join_progress is not None:
                    stage3_monitor_config["side"] = "left"
                    print(
                        "[stage3_test] prewarm left fisheye for stage3 first fifth; "
                        "switch to dual fisheye after entry anchors"
                    )
                else:
                    stage3_monitor_config["side"] = "both"
                    print("[stage3_test] prewarm dual fisheye for stage3")
                line_monitor = LineDistanceMonitor(**stage3_monitor_config)
                if not line_monitor.start(timeout=12.0, activate=True):
                    print("[stage3_test] WARN: stage3 fisheye unavailable; continue without it")
                    line_monitor.close()
                    line_monitor = None
                    line_monitor_config = None

        line_active_start = args.line_active_start
        ok = run_stage3(
            adapter,
            exit_y=exit_y,
            control_hz=15.0,
            vy_enabled=args.with_vy,
            line_monitor=line_monitor,
            line_monitor_config=line_monitor_config,
            line_active_start=line_active_start,
            line_active_end=args.line_active_end,
            line_bias_max=args.line_bias_max,
            line_wz_gate=args.line_wz_gate,
            path_spacing=args.path_spacing,
            tail_heading_start=args.tail_heading_start,
            tail_heading_deg=args.tail_heading_deg,
            path_points=points,
            line_side_switch_progress=entry_join_progress,
            line_side_after="both",
            entry_turn_progress_end=0.36 if entry_join_progress is not None else 0.0,
            entry_turn_wz_limit=0.26,
            entry_turn_slow_error_deg=6.0,
            entry_turn_min_vx=0.044,
        )
        print(f"[stage3_test] stage3_ok={ok}")
        if ok and args.handoff_dir:
            ok = _hold_for_stage4_handoff(
                args.handoff_dir,
                args.handoff_timeout,
            )
        return 0 if ok else 1

    except KeyboardInterrupt:
        print("[stage3_test] interrupted")
        return 130
    finally:
        if 'transition_line_monitor' in locals() and transition_line_monitor is not None:
            transition_line_monitor.close()
        if 'line_monitor' in locals() and line_monitor is not None:
            line_monitor.close()
        base.stop()
        base.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
