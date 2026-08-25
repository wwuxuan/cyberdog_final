#!/usr/bin/env python3
"""第四赛段正式逻辑：默认右绕。"""
import os
import sys

CORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "core")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from s4 import *


if __name__ == "__main__":
    main()
