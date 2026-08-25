#!/usr/bin/env python3
"""第六赛段低姿倒推备选方案。"""
import os
import sys

CORE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core"))
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from s6_rear import main


if __name__ == "__main__":
    main()
