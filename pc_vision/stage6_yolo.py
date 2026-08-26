#!/usr/bin/env python3
"""电脑端第六赛段足球 YOLO 服务，不发送语音播报。"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SUPPORT_DIR = os.path.join(ROOT_DIR, "support")
if SUPPORT_DIR not in sys.path:
    sys.path.insert(0, SUPPORT_DIR)

from yolo_server import main


def _set_default(option, value):
    if option not in sys.argv:
        sys.argv.extend([option, value])


if __name__ == "__main__":
    _set_default("--model", os.path.join(ROOT_DIR, "models", "stage6.pt"))
    _set_default("--targets", "football")
    _set_default("--dog-ip", "")
    main()
