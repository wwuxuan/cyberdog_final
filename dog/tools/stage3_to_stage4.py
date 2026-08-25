#!/usr/bin/env python3
"""Run the standalone stage3 profile, then hand off directly to stage4."""

import argparse
import os
import shlex
import subprocess
import sys
import time

from run1to4 import (
    BASE_DIR,
    PACKAGE_DIR,
    STAGE4_SCRIPTS,
    STOP_MOTION_MANAGER,
    _run,
    _start_stage4_helper,
    _stop_helper,
)


STAGE3_SCRIPT = os.path.join(PACKAGE_DIR, "tools", "stage3_single_test.py")
HANDOFF_DIR = "/tmp/stage3_to_stage4_handoff"
HANDOFF_TIMEOUT = 30.0


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
                "[s3_s4] ERROR: process exited while waiting for {} rc={}".format(
                    name,
                    process.returncode,
                ),
                flush=True,
            )
            return False
        time.sleep(0.05)
    print("[s3_s4] ERROR: handoff timeout waiting for {}".format(name), flush=True)
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
        description="Run standalone stage3 from stage2 upper-left center, then stage4."
    )
    parser.add_argument("--stand", action="store_true", help="stand before stage3")
    parser.add_argument("--arm", action="store_true", help="enable real-dog motion")
    parser.add_argument("--dry-secs", type=float, default=0.0)
    parser.add_argument(
        "--vision-url",
        default="http://192.168.43.102:9877/measure",
        help="stage3 local PC left/right-fisheye endpoint",
    )
    parser.add_argument("--stage3-no-line-correction", action="store_true")
    parser.add_argument("--stage3-with-vy", action="store_true")
    parser.add_argument(
        "--stage4-detour",
        choices=("right", "left"),
        default="right",
    )
    parser.add_argument("--stage4-pc-host", default="192.168.43.102")
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
    stage3_process = None
    stage4_process = None
    handoff_active = False
    stage3_command = [
        sys.executable,
        STAGE3_SCRIPT,
        "--from-stage2-end",
        "--full-stage",
        "--dry-secs",
        str(args.dry_secs),
        "--vision-url",
        args.vision_url,
    ]
    if args.stand:
        stage3_command.append("--stand")
    if args.arm:
        stage3_command.append("--arm")
    if args.stage3_no_line_correction:
        stage3_command.append("--no-line-correction")
    if args.stage3_with_vy:
        stage3_command.append("--with-vy")

    if args.arm:
        _prepare_handoff()
        handoff_active = True
        stage3_command.extend([
            "--handoff-dir",
            HANDOFF_DIR,
            "--handoff-timeout",
            str(HANDOFF_TIMEOUT),
        ])

    print(
        "[s3_s4] place dog at stage2 upper-left four-ball center "
        "(0.300,3.440), facing +y",
        flush=True,
    )
    if args.arm:
        print(
            "[s3_s4] prepare LCM control before stage3; do not stop it at handoff",
            flush=True,
        )
        if _run(["/bin/bash", STOP_MOTION_MANAGER], "prepare LCM control") != 0:
            print("[s3_s4] ERROR: cannot prepare LCM controller")
            return 2
        print(
            "[s3_s4] run standalone stage3 with standing handoff: {}".format(
                " ".join(shlex.quote(str(arg)) for arg in stage3_command)
            ),
            flush=True,
        )
        stage3_process = subprocess.Popen(stage3_command)
        if not _wait_for_handoff("stage3_ready", stage3_process):
            _release_stage3(stage3_process)
            return 1
        result = None
    else:
        result = _run(stage3_command, "run standalone stage3")
    if result is not None and result != 0:
        print("[s3_s4] stage3 failed exit={}".format(result))
        return result
    if not args.arm:
        print("[s3_s4] dry run ends after stage3")
        return 0

    stage4_script = STAGE4_SCRIPTS[args.stage4_detour]
    if not os.path.exists(stage4_script):
        print("[s3_s4] ERROR: stage4 script missing: {}".format(stage4_script))
        if handoff_active:
            _release_stage3(stage3_process)
            handoff_active = False
        return 2

    stream_process = None
    speech_process = None
    try:
        if not args.no_stage4_stream:
            stream_log = "/tmp/stage4_stream_s3_s4.log"
            print(
                "[s3_s4] start stage4 RGB stream -> {}:{} (log {})".format(
                    args.stage4_pc_host,
                    args.stage4_stream_port,
                    stream_log,
                ),
                flush=True,
            )
            stream_process = _start_stage4_helper(
                "stream_images.py",
                ["--host", args.stage4_pc_host, "--port", args.stage4_stream_port],
                stream_log,
            )
        if not args.no_stage4_speech:
            speech_log = "/tmp/stage4_speak_s3_s4.log"
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
            "[s3_s4] start stage4 detour={}; rebase origin while stage3 keeps "
            "the dog standing, then approach corridor fwd=0.60".format(
                args.stage4_detour
            ),
            flush=True,
        )
        stage4_process = subprocess.Popen(["/bin/bash", "-lc", stage4_shell])
        if handoff_active:
            if not _wait_for_handoff("stage4_ready", stage4_process, HANDOFF_TIMEOUT):
                return 1
            stage3_result = _release_stage3(stage3_process)
            if stage3_result != 0:
                print("[s3_s4] ERROR: stage3 handoff exited={}".format(stage3_result))
                return stage3_result
            handoff_active = False
            print("[s3_s4] standing handoff complete", flush=True)
        return stage4_process.wait()
    finally:
        if handoff_active:
            _release_stage3(stage3_process)
        _stop_helper(speech_process, "stage4 speech")
        _stop_helper(stream_process, "stage4 RGB stream")


if __name__ == "__main__":
    raise SystemExit(main())
