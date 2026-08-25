#!/usr/bin/env python3
"""第三赛段正式逻辑模块，由 main.py 调用。"""
import os
import sys

CORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "core")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from s3 import *
