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

import rclpy

from adapter import RealDogAdapter
from orange import OrangeDetector
from s2 import (
    BOARD_BALLS,
    FIXED_BLUE,
    GRID_X,
    GRID_Y,
    S2_END,
    PROBE_CELLS,
    _RadarPoseCorrector,
    S2_RULE_SCAN,
    grid_cell,
    hit_ball,
    infer_orange_balls_from_probes,
    s2_navigate,
    s2_plan_hits,
    start_pose_logger,
    snap_to_grid,
)


FIELD_YAW_WHEN_ODOM_ZERO = 90.0
DEFAULT_SAMPLE_SECS = 1.2
DEFAULT_SAMPLE_DT = 0.08
DEFAULT_CENTER_WINDOW_DEG = 18.0
DEFAULT_COLOR_MARGIN = 1.12
DEFAULT_TARGET_WINDOW_DEG = 16.0


def _wrap_deg(angle):
    return ((angle + 180.0) % 360.0) - 180.0


def _field_yaw(adapter):
    return _wrap_deg(FIELD_YAW_WHEN_ODOM_ZERO + adapter.get_yaw_deg())


def _field_yaw_to_odom(field_yaw):
    return _wrap_deg(field_yaw - FIELD_YAW_WHEN_ODOM_ZERO)


class Stage2PoseAdapter:
    def __init__(self, base, field_x, field_y, field_yaw_deg):
        self.base = base
        self.field_x = float(field_x)
        self.field_y = float(field_y)
        self.field_yaw_deg = float(field_yaw_deg)

    def get_position(self):
        local_x, local_y, local_z = self.base.get_position()
        yaw0 = math.radians(self.field_yaw_deg)
        field_x = self.field_x + local_x * math.cos(yaw0) - local_y * math.sin(yaw0)
        field_y = self.field_y + local_x * math.sin(yaw0) + local_y * math.cos(yaw0)
        return field_x, field_y, local_z

    def get_yaw_deg(self):
        return _wrap_deg(self.base.get_yaw_deg() + self.field_yaw_deg)

    def set_mapped_pose(self, x, y, yaw_deg, quiet=False):
        """Keep the field position fixed while updating the wrapped yaw mapping."""
        yaw0 = math.radians(self.field_yaw_deg)
        dx = float(x) - self.field_x
        dy = float(y) - self.field_y
        local_x = dx * math.cos(yaw0) + dy * math.sin(yaw0)
        local_y = -dx * math.sin(yaw0) + dy * math.cos(yaw0)
        self.base.set_mapped_pose(
            local_x,
            local_y,
            _wrap_deg(float(yaw_deg) - self.field_yaw_deg),
            quiet=quiet,
        )

    def align_yaw(self, target_deg, *args, **kwargs):
        local_target = _wrap_deg(float(target_deg) - self.field_yaw_deg)
        return self.base.align_yaw(local_target, *args, **kwargs)

    def walk(self, vx, vy=0.0, wz=0.0):
        return self.base.walk(vx, vy, wz)

    def stop(self):
        return self.base.stop()


def _record_probe_color(adapter, detector, target_pos, votes, args):
    rx, ry = S2_RULE_SCAN
    yaw_rad = math.radians(_field_yaw(adapter))
    expected_angle = math.atan2(target_pos[1] - ry, target_pos[0] - rx)
    expected_off = (expected_angle - yaw_rad + math.pi) % (2.0 * math.pi) - math.pi
    info = detector.classify_center_ball(
        center_window_deg=args.center_window_deg,
        margin=args.color_margin,
        expected_angle_deg=math.degrees(expected_off),
        target_window_deg=args.target_window_deg,
    )
    color = info["color"]
    if color in ("orange", "blue"):
        votes[color] += 1
    else:
        votes["none"] += 1
    return info


def _fmt_center_debug(info):
    def fmt(item):
        if not item:
            return "none"
        return (
            f"ang={item['angle_deg']:.1f}deg area={item['area']:.0f} "
            f"circ={item['circularity']:.2f} fill={item['fill_ratio']:.2f} "
            f"asp={item['aspect']:.2f} hr={item['height_ratio']:.2f} "
            f"sat={item['mean_sat']:.0f} val={item['mean_val']:.0f} "
            f"score={item['score']:.1f}"
        )

    return f"yellow[{fmt(info.get('orange'))}] blue[{fmt(info.get('blue'))}] note={info.get('note')}"


def _info_score(info):
    selected = info.get("selected") if info else None
    return selected.get("score", 0.0) if selected else 0.0


def probe_one_cell(adapter, detector, name, row, col, args):
    target_pos = grid_cell(row, col)
    rx, ry = S2_RULE_SCAN
    field_target_yaw = math.degrees(math.atan2(target_pos[1] - ry, target_pos[0] - rx))
    odom_target_yaw = _field_yaw_to_odom(field_target_yaw)

    print(
        f"[stage2_check] probe {name} row={row} col={col} "
        f"pos=({target_pos[0]:.2f},{target_pos[1]:.2f}) "
        f"field_yaw={field_target_yaw:.1f} odom_yaw={odom_target_yaw:.1f}"
    )
    if not args.no_rotate:
        adapter.align_yaw(odom_target_yaw, tol=2.5, timeout=6.0)
    time.sleep(0.25)

    votes = {"orange": 0, "blue": 0, "none": 0}
    last_info = None
    best_info = None
    t0 = time.time()
    while time.time() - t0 < args.sample_secs:
        last_info = _record_probe_color(adapter, detector, target_pos, votes, args)
        if _info_score(last_info) >= _info_score(best_info):
            best_info = last_info
        time.sleep(args.sample_dt)

    total = votes["orange"] + votes["blue"]
    if total == 0:
        result = False
        conf = 0.0
        note = "no_votes_assume_blue"
    elif votes["orange"] == votes["blue"]:
        result = False
        conf = 0.5
        note = "tie_assume_blue"
    else:
        result = votes["orange"] > votes["blue"]
        conf = max(votes["orange"], votes["blue"]) / total
        note = "ok"

    color = "yellow" if result else "blue"
    print(
        f"[stage2_check] probe {name} result={color} conf={conf:.2f} "
        f"yellow={votes['orange']} blue={votes['blue']} none={votes['none']} {note}"
    )
    if last_info is not None:
        print(f"[stage2_check] probe {name} center {_fmt_center_debug(last_info)}")
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
        image_info = best_info or last_info
        image_path = os.path.abspath(os.path.join(args.debug_dir, f"{name}.jpg"))
        exp = image_info.get("expected_angle_deg", 0.0) if image_info else 0.0
        title = f"{name} result={color} y={votes['orange']} b={votes['blue']} exp={exp:.1f}"
        if image_info is not None and detector.save_debug_image(image_path, image_info, title=title):
            print(f"[stage2_check] probe {name} debug_image={image_path}")
    return result, votes


def print_board(orange_balls):
    orange_set = set(orange_balls)
    probe_by_pos = {grid_cell(row, col): name[0] for name, (row, col) in PROBE_CELLS.items()}

    print("[stage2_check] board legend: Y=yellow/orange B=blue *=fixed-blue A/B/C=probed")
    print("[stage2_check]        c1      c2      c3      c4")
    for row in (4, 3, 2, 1):
        cells = []
        for col in (1, 2, 3, 4):
            pos = grid_cell(row, col)
            color = "Y" if pos in orange_set else "B"
            suffix = ""
            if pos in FIXED_BLUE:
                suffix += "*"
            if pos in probe_by_pos:
                suffix += probe_by_pos[pos]
            cells.append(f"{color}{suffix}".ljust(5))
        print(f"[stage2_check] r{row}  " + "  ".join(cells))


def main():
    parser = argparse.ArgumentParser(
        description="Validate stage2 rule scan from the lower-right four-ball center."
    )
    parser.add_argument("--sample-secs", type=float, default=DEFAULT_SAMPLE_SECS)
    parser.add_argument("--sample-dt", type=float, default=DEFAULT_SAMPLE_DT)
    parser.add_argument("--center-window-deg", type=float, default=DEFAULT_CENTER_WINDOW_DEG)
    parser.add_argument("--color-margin", type=float, default=DEFAULT_COLOR_MARGIN)
    parser.add_argument("--target-window-deg", type=float, default=DEFAULT_TARGET_WINDOW_DEG)
    parser.add_argument("--debug-dir", default="stage2_debug")
    parser.add_argument("--run-stage2", action="store_true", help="run full stage2 after checking probes")
    parser.add_argument("--stand", action="store_true", help="stand first before probing")
    parser.add_argument("--no-rotate", action="store_true", help="do not rotate; only sample current view")
    args = parser.parse_args()

    print("[stage2_check] place dog at S2_RULE_SCAN=(2.700,1.76), facing field +y")
    print("[stage2_check] camera topic is image_rgb only")

    adapter = RealDogAdapter(None)
    detector = None
    try:
        if not adapter.wait_odom(timeout=3.0):
            print("[stage2_check] ERROR: no odom")
            return 2

        if args.stand:
            adapter.stand()

        adapter.set_origin()
        print(
            f"[stage2_check] origin set; odom_yaw=0 maps to field_yaw="
            f"{FIELD_YAW_WHEN_ODOM_ZERO:.1f}"
        )

        try:
            rclpy.init()
        except RuntimeError:
            pass
        detector = OrangeDetector()
        detector.start()
        if not detector.wait_ready(timeout=6.0):
            print("[stage2_check] ERROR: image_rgb not ready")
            return 3

        probe_is_orange = {}
        probe_votes = {}
        for name, (row, col) in PROBE_CELLS.items():
            probe_is_orange[name], probe_votes[name] = probe_one_cell(
                adapter, detector, name, row, col, args
            )

        print(
            "[stage2_check] probe bits "
            + " ".join(
                f"{name}={'yellow' if value else 'blue'}"
                for name, value in probe_is_orange.items()
            )
        )
        orange_balls = infer_orange_balls_from_probes(probe_is_orange)
        if not orange_balls:
            print("[stage2_check] ERROR: failed to infer orange balls")
            return 4

        print("[stage2_check] inferred yellow/orange coordinates:")
        for x, y in orange_balls:
            print(f"[stage2_check] yellow=({x:.2f},{y:.2f})")
        print_board(orange_balls)

        if args.run_stage2:
            print("[stage2_check] run full stage2")
            stage2_adapter = Stage2PoseAdapter(
                adapter,
                S2_RULE_SCAN[0],
                S2_RULE_SCAN[1],
                FIELD_YAW_WHEN_ODOM_ZERO,
            )
            pose_stop = None
            pose_thread = None
            sx, sy, _ = stage2_adapter.get_position()
            print(
                f"[stage2_check] stage2 pose mapped pos=({sx:.2f},{sy:.2f}) "
                f"yaw={stage2_adapter.get_yaw_deg():.1f}"
            )
            plan = s2_plan_hits(orange_balls, S2_RULE_SCAN)
            active_landmarks = set(BOARD_BALLS)
            radar_corrector = _RadarPoseCorrector(orange_balls)
            print(
                f"[stage2_check] radar unhit balls n={len(active_landmarks)}",
                flush=True,
            )
            print(f"[stage2_check] hit plan n={len(plan)}")
            for ball, node in plan:
                print(
                    f"[stage2_check] plan ball=({ball[0]:.2f},{ball[1]:.2f}) "
                    f"node=({node[0]:.2f},{node[1]:.2f})"
            )
            all_hit_ok = True
            for hit_index, (ball, node) in enumerate(plan, start=1):
                if not hit_ball(
                    stage2_adapter,
                    node,
                    ball,
                    detector=detector,
                    hit_index=hit_index,
                    radar_landmarks=active_landmarks,
                    radar_corrector=radar_corrector,
                ):
                    all_hit_ok = False
                    print(
                        f"[stage2_check] WARN: stop hit sequence after failed ball="
                        f"({ball[0]:.2f},{ball[1]:.2f})"
                    )
                    break
                active_landmarks.discard(ball)
                radar_corrector.reset()
            print(f"[stage2_check] go end {S2_END}")
            if all_hit_ok:
                s2_navigate(
                    stage2_adapter,
                    S2_END,
                    final_yaw=90.0,
                    detector=detector,
                    radar_landmarks=active_landmarks,
                    radar_corrector=radar_corrector,
                )
            print("[stage2_check] stage2 done")
            if pose_stop is not None:
                pose_stop.set()
            if pose_thread is not None:
                pose_thread.join(timeout=1.0)
        return 0

    except KeyboardInterrupt:
        print("[stage2_check] interrupted")
        return 130
    finally:
        adapter.stop()
        adapter.shutdown()
        if detector is not None:
            detector.stop()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
