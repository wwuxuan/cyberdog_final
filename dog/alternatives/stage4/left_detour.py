#!/usr/bin/env python3
"""第四赛段左绕备选入口，不是 main.py 的默认方案。"""
import os
import sys

CORE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core"))
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from left_detour_impl import main


if __name__ == "__main__":
    main()
