#!/usr/bin/env python3
"""正式比赛入口：按顺序运行第一至第六赛段。"""
import argparse
import os
import subprocess
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.join(BASE_DIR, "core")
STAGES_1_TO_4_SCRIPT = os.path.join(CORE_DIR, "run1to4.py")
STAGE2_UPPER_LEFT_SCRIPT = os.path.join(CORE_DIR, "resume2.py")
STAGE4_FROM_START_SCRIPT = os.path.join(CORE_DIR, "start4.py")
DEFAULT_STAGE5_SCRIPT = os.path.join(BASE_DIR, "stage5.py")
DEFAULT_STAGE6_SCRIPT = os.path.join(BASE_DIR, "stage6.py")

START_CHOICES = (
    "stage1",
    "stage2-upper-left",
    "stage4",
    "stage5",
    "stage5-turn1",
    "stage5-turn2",
    "stage5-turn3",
    "stage5-turn4",
)
STAGE5_START_MARKERS = {
    "stage5": "stage5",
    "stage5-turn1": "turn1",
    "stage5-turn2": "turn2",
    "stage5-turn3": "turn3",
    "stage5-turn4": "turn4",
}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run stages 1-6; unrecognized arguments pass through to stages 1-4."
    )
    parser.add_argument(
        "--start-at",
        choices=START_CHOICES,
        default="stage1",
        help="stage1=full run; later values resume at the named field marker",
    )
    parser.add_argument(
        "--stage5-script",
        default=DEFAULT_STAGE5_SCRIPT,
        help="dog-side stage 5 script",
    )
    parser.add_argument(
        "--stage5-no-tilt",
        action="store_true",
        help="pass --no-tilt to stage 5",
    )
    parser.add_argument(
        "--stage5-roll",
        type=float,
        default=None,
        help="pass a roll value to stage 5",
    )
    parser.add_argument(
        "--stage6-script",
        default=DEFAULT_STAGE6_SCRIPT,
        help="dog-side stage 6 script",
    )
    parser.add_argument(
        "--stage6-pc-host",
        default=None,
        help="PC running the stage 6 YOLO service",
    )
    parser.add_argument("--stage6-det-port", type=int, default=9890)
    parser.add_argument("--stage6-stream-port", type=int, default=9892)
    parser.add_argument("--stage6-stream-fps", type=float, default=15.0)
    parser.add_argument("--stage6-depth-topic", default=None)
    parser.add_argument("--stage6-gait-dir", default=None)
    parser.add_argument("--stage6-no-depth", action="store_true")
    parser.add_argument("--stage6-no-stream", action="store_true")
    parser.add_argument(
        "--stage6-wait-odom",
        action="store_true",
        help="wait for odometry in stage 6 instead of skipping its redundant startup wait",
    )
    return parser.parse_known_args()


def _run(command, label, cwd=None):
    print("[s1_s2_s3_s4_s5_s6] %s: %s" % (label, " ".join(command)), flush=True)
    return subprocess.call(command, cwd=cwd)


def _require_script(path, label):
    if os.path.isfile(path):
        return True
    print("[s1_s2_s3_s4_s5_s6] %s missing: %s" % (label, path))
    return False


def _has_flag(arguments, flag):
    return flag in arguments


def _option_value(arguments, option, default=None):
    prefix = option + "="
    for index, value in enumerate(arguments):
        if value == option and index + 1 < len(arguments):
            return arguments[index + 1]
        if value.startswith(prefix):
            return value[len(prefix):]
    return default


def _run_stage4_from_start(arguments, start_at_corridor):
    if not _require_script(STAGE4_FROM_START_SCRIPT, "standalone stage 4 runner"):
        return 2
    command = [
        sys.executable,
        STAGE4_FROM_START_SCRIPT,
        "--detour", _option_value(arguments, "--stage4-detour", "right"),
        "--pc-host", _option_value(arguments, "--stage4-pc-host", "192.168.43.102"),
        "--stream-port", _option_value(arguments, "--stage4-stream-port", "9891"),
        "--det-port", _option_value(arguments, "--stage4-det-port", "9890"),
    ]
    gait_dir = _option_value(arguments, "--stage4-gait-dir")
    if gait_dir:
        command.extend(["--gait-dir", gait_dir])
    if start_at_corridor:
        command.append("--start-at-corridor")
    if _has_flag(arguments, "--no-stage4-speech"):
        command.append("--no-speech")
    if _has_flag(arguments, "--stage4-no-wait-detector"):
        command.append("--no-wait-detector")
    max_channels = _option_value(arguments, "--stage4-max-channels")
    if max_channels is not None:
        command.extend(["--max-channels", max_channels])
    bar_norm = _option_value(arguments, "--stage4-bar-norm")
    if bar_norm is not None:
        command.extend(["--bar-norm", bar_norm])
    return _run(command, "run stage 4 from selected entry", cwd=BASE_DIR)


def _run_selected_prefix(start_at, arguments):
    if start_at == "stage1":
        return _run(
            [sys.executable, STAGES_1_TO_4_SCRIPT] + arguments,
            "run stages 1-4",
            cwd=BASE_DIR,
        )

    if start_at == "stage2-upper-left":
        if not _require_script(STAGE2_UPPER_LEFT_SCRIPT, "stage2 upper-left runner"):
            return 2
        command = [sys.executable, STAGE2_UPPER_LEFT_SCRIPT]
        if _has_flag(arguments, "--stand"):
            command.append("--stand")
        if _has_flag(arguments, "--arm"):
            command.append("--arm")
        stage3_vision_url = _option_value(
            arguments, "--stage3-vision-url", "http://192.168.43.102:9877/measure"
        )
        command.extend(["--stage3-vision-url", stage3_vision_url])
        result = _run(command, "run stages 2-3 from upper-left center", cwd=BASE_DIR)
        if result != 0:
            return result
        return _run_stage4_from_start(arguments, start_at_corridor=False)

    if start_at == "stage4":
        return _run_stage4_from_start(arguments, start_at_corridor=True)

    return 0


def main():
    args, stage1_to_4_args = _parse_args()
    if args.start_at != "stage1" and "--arm" not in stage1_to_4_args:
        print("[s1_s2_s3_s4_s5_s6] %s requires --arm; no motion started" % args.start_at)
        return 0

    print("[s1_s2_s3_s4_s5_s6] selected start=%s" % args.start_at, flush=True)
    result = _run_selected_prefix(args.start_at, stage1_to_4_args)
    if result != 0:
        print("[s1_s2_s3_s4_s5_s6] selected prefix failed exit=%d; stop" % result)
        return result
    if "--arm" not in stage1_to_4_args:
        print("[s1_s2_s3_s4_s5_s6] dry run complete; stages 5-6 are not started without --arm")
        return 0
    if not _require_script(args.stage5_script, "stage 5 script"):
        return 2
    if not _require_script(args.stage6_script, "stage 6 script"):
        return 2

    print("[s1_s2_s3_s4_s5_s6] stage 5 handoff: world=(3.250, 7.200), facing +y, standing")
    stage5_command = [sys.executable, args.stage5_script]
    stage5_marker = STAGE5_START_MARKERS.get(args.start_at, "stage5")
    stage5_command.extend(["--start-at", stage5_marker])
    if args.stage5_no_tilt:
        stage5_command.append("--no-tilt")
    if args.stage5_roll is not None:
        stage5_command.extend(["--roll", str(args.stage5_roll)])
    result = _run(stage5_command, "run stage 5", cwd=os.path.dirname(args.stage5_script))
    if result != 0:
        print("[s1_s2_s3_s4_s5_s6] stage 5 failed exit=%d; stage 6 will not start" % result)
        return result

    print("[s1_s2_s3_s4_s5_s6] stage 5 jump handoff complete; stage 6 remaps its own start pose")
    stage6_command = [
        sys.executable,
        args.stage6_script,
        "--det-port",
        str(args.stage6_det_port),
        "--stream-port",
        str(args.stage6_stream_port),
        "--stream-fps",
        str(args.stage6_stream_fps),
    ]
    if not args.stage6_wait_odom:
        stage6_command.append("--skip-odom-wait")
    if args.stage6_pc_host:
        stage6_command.extend(["--pc-host", args.stage6_pc_host])
    if args.stage6_depth_topic:
        stage6_command.extend(["--depth-topic", args.stage6_depth_topic])
    if args.stage6_gait_dir:
        stage6_command.extend(["--gait-dir", args.stage6_gait_dir])
    if args.stage6_no_depth:
        stage6_command.append("--no-depth")
    if args.stage6_no_stream:
        stage6_command.append("--no-stream")
    return _run(stage6_command, "run stage 6", cwd=os.path.dirname(args.stage6_script))


if __name__ == "__main__":
    raise SystemExit(main())
