#!/usr/bin/env python3
"""电脑端第三赛段左右鱼眼黄线服务。"""
import os
import sys

SUPPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "support")
if SUPPORT_DIR not in sys.path:
    sys.path.insert(0, SUPPORT_DIR)

from stage3_fisheye_service import main


if __name__ == "__main__":
    raise SystemExit(main())
