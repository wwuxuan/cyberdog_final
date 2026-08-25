#!/usr/bin/env python3
"""第一赛段正式逻辑。联合运行请使用 main.py。"""
import os
import sys

CORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "core")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from s1 import *
from s1 import _prewarm_cameras


if __name__ == "__main__":
    raise SystemExit(main())
