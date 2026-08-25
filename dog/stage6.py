#!/usr/bin/env python3
"""第六赛段正式逻辑：俯身前冲后使用 YOLO 推球。"""
import os
import sys

CORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "core")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from s6 import main


if __name__ == "__main__":
    main()
