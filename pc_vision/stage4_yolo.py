#!/usr/bin/env python3
"""电脑端第四赛段 YOLO 服务。"""
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
    _set_default("--model", os.path.join(ROOT_DIR, "models", "stage4.pt"))
    main()
