#!/usr/bin/env python3
"""Run standalone stage 4 with its speech service, for later-stage handoffs."""
import argparse
import os
import shlex
import subprocess
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(BASE_DIR)
STAGE4_SCRIPTS = {
    "right": os.path.join(PACKAGE_DIR, "stage4.py"),
    "left": os.path.join(PACKAGE_DIR, "alternatives", "stage4", "left_detour_impl.py"),
}
SPEECH_SCRIPT = os.path.join(PACKAGE_DIR, "support", "stage4_vision", "speak_on_detect.py")


def _ros_command(script, arguments):
    return (
        "source /etc/mi/ros2_env.conf && cd {} && exec python3 {} {}"
    ).format(
        shlex.quote(BASE_DIR),
        shlex.quote(script),
        " ".join(shlex.quote(str(argument)) for argument in arguments),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run standalone stage 4, including dog-side speech service."
    )
    parser.add_argument("--detour", choices=tuple(STAGE4_SCRIPTS), default="right")
    parser.add_argument("--start-at-corridor", action="store_true")
    parser.add_argument("--pc-host", default="192.168.43.102")
    parser.add_argument("--stream-port", type=int, default=9891)
    parser.add_argument("--det-port", type=int, default=9890)
    parser.add_argument("--gait-dir", default=os.path.join(BASE_DIR, "gait"))
    parser.add_argument("--no-speech", action="store_true")
    parser.add_argument("--no-wait-detector", action="store_true")
    parser.add_argument("--max-channels", type=int, default=None)
    parser.add_argument("--bar-norm", type=float, default=None)
    args = parser.parse_args()

    stage4_script = STAGE4_SCRIPTS[args.detour]
    if not os.path.isfile(stage4_script):
        print("[stage4_from_start] missing stage4 script: %s" % stage4_script)
        return 2

    speech_process = None
    try:
        if not args.no_speech and os.path.isfile(SPEECH_SCRIPT):
            print("[stage4_from_start] start stage4 speech")
            speech_process = subprocess.Popen([
                "/bin/bash", "-lc", _ros_command(SPEECH_SCRIPT, [])
            ])

        command = [
            stage4_script,
            "--pc-host", args.pc_host,
            "--stream-port", str(args.stream_port),
            "--det-port", str(args.det_port),
            "--gait-dir", args.gait_dir,
        ]
        if args.start_at_corridor:
            command.append("--start-at-corridor")
        if args.no_wait_detector:
            command.append("--no-wait-detector")
        if args.max_channels is not None:
            command.extend(["--max-channels", str(args.max_channels)])
        if args.bar_norm is not None:
            command.extend(["--bar-norm", str(args.bar_norm)])
        shell_command = _ros_command(command[0], command[1:])
        print("[stage4_from_start] run: %s" % shell_command, flush=True)
        return subprocess.call(["/bin/bash", "-lc", shell_command])
    finally:
        if speech_process is not None and speech_process.poll() is None:
            speech_process.terminate()
            try:
                speech_process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                speech_process.kill()
                speech_process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
