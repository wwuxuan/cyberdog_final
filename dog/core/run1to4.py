#!/usr/bin/env python3
"""Run stages 1-4 with a clean process handoff into stage 4."""

import argparse
import os
import shlex
import subprocess
import sys
import time


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(BASE_DIR)
STAGES_1_TO_3_SCRIPT = os.path.join(BASE_DIR, "run1to3.py")
STAGE4_SCRIPTS = {
    "right": os.path.join(PACKAGE_DIR, "stage4.py"),
    "left": os.path.join(PACKAGE_DIR, "alternatives", "stage4", "left_detour_impl.py"),
}
STAGE4_VISION_DIR = "/home/mi/stage4/vision"
STOP_MOTION_MANAGER = os.path.join(PACKAGE_DIR, "support", "stop_motion_manager.sh")
HANDOFF_DIR = "/tmp/stage1_stage2_stage3_stage4_handoff"
HANDOFF_TIMEOUT = 30.0


def _run(command, label):
    print(
        "[s1_s2_s3_s4] {}: {}".format(
            label,
            " ".join(shlex.quote(str(arg)) for arg in command),
        ),
        flush=True,
    )
    return subprocess.call(command)


def _start_stage4_helper(script_name, arguments, log_path):
    command = (
        "source /etc/mi/ros2_env.conf && "
        "cd {} && exec python3 {} {}"
    ).format(
        shlex.quote(STAGE4_VISION_DIR),
        shlex.quote(script_name),
        " ".join(shlex.quote(str(argument)) for argument in arguments),
    )
    log_file = open(log_path, "a")
    process = subprocess.Popen(
        ["/bin/bash", "-lc", command],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    log_file.close()
    return process


def _stop_helper(process, label):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=4.0)
    except subprocess.TimeoutExpired:
        print("[s1_s2_s3_s4] {} did not exit; killing it".format(label))
        process.kill()
        process.wait()


def _handoff_path(name):
    return os.path.join(HANDOFF_DIR, name)


def _prepare_handoff():
    os.makedirs(HANDOFF_DIR, exist_ok=True)
    for name in ("stage3_ready", "stage4_ready", "release_stage3"):
        try:
            os.unlink(_handoff_path(name))
        except FileNotFoundError:
            pass


def _mark_handoff(name):
    with open(_handoff_path(name), "w"):
        pass


def _wait_for_handoff(name, process, timeout=None):
    deadline = None if timeout is None else time.time() + max(1.0, float(timeout))
    path = _handoff_path(name)
    while deadline is None or time.time() < deadline:
        if os.path.exists(path):
            return True
        if process is not None and process.poll() is not None:
            print(
                "[s1_s2_s3_s4] ERROR: process exited while waiting for {} rc={}".format(
                    name,
                    process.returncode,
                ),
                flush=True,
            )
            return False
        time.sleep(0.05)
    print(
        "[s1_s2_s3_s4] ERROR: handoff timeout waiting for {}".format(name),
        flush=True,
    )
    return False


def _release_stage3(process):
    _mark_handoff("release_stage3")
    if process is None or process.poll() is not None:
        return process.returncode if process is not None else 0
    try:
        return process.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        process.terminate()
        return process.wait()


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run stages 1-4; stage4 starts at the stage3 exit automatically."
    )
    parser.add_argument("--stand", action="store_true", help="stand before stage1")
    parser.add_argument("--arm", action="store_true", help="enable real-dog motion")
    parser.add_argument("--stage1-timeout", type=float, default=90.0)
    parser.add_argument(
        "--vision-url",
        default="http://192.168.43.102:9876/measure",
        help="stage1 local PC right-fisheye endpoint; empty uses dog-side vision",
    )
    parser.add_argument(
        "--stage3-vision-url",
        default="http://192.168.43.102:9877/measure",
        help="stage3 local PC left/right-fisheye endpoint; empty uses dog-side vision",
    )
    parser.add_argument(
        "--turn-calibration",
        choices=("map", "physical"),
        default="physical",
    )
    parser.add_argument(
        "--stage4-detour",
        choices=("right", "left"),
        default="right",
        help="stage4 obstacle detour side",
    )
    parser.add_argument(
        "--stage4-pc-host",
        default="192.168.43.102",
        help="PC running stage4 live_detect_server.py",
    )
    parser.add_argument("--stage4-stream-port", type=int, default=9891)
    parser.add_argument("--no-stage4-stream", action="store_true")
    parser.add_argument("--no-stage4-speech", action="store_true")
    parser.add_argument("--stage4-no-wait-detector", action="store_true")
    parser.add_argument("--stage4-max-channels", type=int, default=None)
    parser.add_argument("--stage4-bar-norm", type=float, default=None)
    parser.add_argument("--stage4-det-port", type=int, default=9890)
    parser.add_argument(
        "--stage4-gait-dir",
        default=os.path.join(BASE_DIR, "gait"),
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    stage1_to_stage3_process = None
    stage4_process = None
    handoff_active = False
    stage1_to_stage3_command = [
        sys.executable,
        STAGES_1_TO_3_SCRIPT,
        "--stage1-timeout",
        str(args.stage1_timeout),
        "--vision-url",
        args.vision_url,
        "--stage3-vision-url",
        args.stage3_vision_url,
        "--turn-calibration",
        args.turn_calibration,
    ]
    if args.stand:
        stage1_to_stage3_command.append("--stand")
    if args.arm:
        stage1_to_stage3_command.append("--arm")
        _prepare_handoff()
        handoff_active = True
        stage1_to_stage3_command.extend([
            "--handoff-dir",
            HANDOFF_DIR,
            "--handoff-timeout",
            str(HANDOFF_TIMEOUT),
        ])

    print(
        "[s1_s2_s3_s4] stage3 exit is stage4 origin "
        "(field x=3.200, y=6.580, facing +y)",
        flush=True,
    )
    if args.arm:
        print(
            "[s1_s2_s3_s4] prepare LCM control before stage1; keep it through "
            "the stage3-to-stage4 handoff",
            flush=True,
        )
        if _run(["/bin/bash", STOP_MOTION_MANAGER], "prepare LCM control") != 0:
            print("[s1_s2_s3_s4] ERROR: cannot prepare LCM controller")
            return 2
        print(
            "[s1_s2_s3_s4] run stages 1-3 with standing handoff: {}".format(
                " ".join(shlex.quote(str(arg)) for arg in stage1_to_stage3_command)
            ),
            flush=True,
        )
        stage1_to_stage3_process = subprocess.Popen(stage1_to_stage3_command)
        if not _wait_for_handoff(
                "stage3_ready", stage1_to_stage3_process):
            _release_stage3(stage1_to_stage3_process)
            return 1
        result = None
    else:
        result = _run(stage1_to_stage3_command, "run stages 1-3")
    if result is not None and result != 0:
        print("[s1_s2_s3_s4] stages 1-3 failed exit={}".format(result))
        return result
    if not args.arm:
        print("[s1_s2_s3_s4] dry run ends after stages 1-3")
        return 0

    stage4_script = STAGE4_SCRIPTS[args.stage4_detour]
    if not os.path.exists(stage4_script):
        print("[s1_s2_s3_s4] ERROR: stage4 script missing: {}".format(stage4_script))
        if handoff_active:
            _release_stage3(stage1_to_stage3_process)
            handoff_active = False
        return 2

    stream_process = None
    speech_process = None
    try:
        if not args.no_stage4_stream:
            stream_log = "/tmp/stage4_stream_full.log"
            print(
                "[s1_s2_s3_s4] start stage4 RGB stream -> {}:{} "
                "(log {})".format(args.stage4_pc_host, args.stage4_stream_port, stream_log),
                flush=True,
            )
            stream_process = _start_stage4_helper(
                "stream_images.py",
                ["--host", args.stage4_pc_host, "--port", args.stage4_stream_port],
                stream_log,
            )
        if not args.no_stage4_speech:
            speech_log = "/tmp/stage4_speak_full.log"
            print("[s1_s2_s3_s4] start stage4 speech (log {})".format(speech_log))
            speech_process = _start_stage4_helper(
                "speak_on_detect.py",
                [],
                speech_log,
            )
        time.sleep(1.0)

        stage4_arguments = [
            "--det-port",
            str(args.stage4_det_port),
            "--gait-dir",
            args.stage4_gait_dir,
            "--no-stream",
        ]
        if args.stage4_no_wait_detector:
            stage4_arguments.append("--no-wait-detector")
        if args.stage4_max_channels is not None:
            stage4_arguments.extend(["--max-channels", str(args.stage4_max_channels)])
        if args.stage4_bar_norm is not None:
            stage4_arguments.extend(["--bar-norm", str(args.stage4_bar_norm)])
        if handoff_active:
            stage4_arguments.extend([
                "--handoff-dir",
                HANDOFF_DIR,
                "--handoff-timeout",
                str(HANDOFF_TIMEOUT),
            ])
        stage4_shell = (
            "source /etc/mi/ros2_env.conf && cd {} && exec python3 {} {}"
        ).format(
            shlex.quote(BASE_DIR),
            shlex.quote(stage4_script),
            " ".join(shlex.quote(str(argument)) for argument in stage4_arguments),
        )
        print(
            "[s1_s2_s3_s4] start stage4 detour={}; it rebases origin while "
            "stage3 keeps the dog standing, then approaches corridor fwd=0.60".format(
                args.stage4_detour
            ),
            flush=True,
        )
        stage4_process = subprocess.Popen(["/bin/bash", "-lc", stage4_shell])
        if handoff_active:
            if not _wait_for_handoff("stage4_ready", stage4_process, HANDOFF_TIMEOUT):
                return 1
            stage3_result = _release_stage3(stage1_to_stage3_process)
            if stage3_result != 0:
                print("[s1_s2_s3_s4] ERROR: stage3 handoff exited={}".format(stage3_result))
                return stage3_result
            handoff_active = False
            print("[s1_s2_s3_s4] standing handoff complete", flush=True)
        return stage4_process.wait()
    finally:
        if handoff_active:
            _release_stage3(stage1_to_stage3_process)
        _stop_helper(speech_process, "stage4 speech")
        _stop_helper(stream_process, "stage4 RGB stream")


if __name__ == "__main__":
    raise SystemExit(main())
