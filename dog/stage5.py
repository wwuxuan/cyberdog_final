#!/usr/bin/env python3
"""第五赛段正式逻辑：独木桥后最远跳下。"""
import os
import sys

STAGE5_RUNTIME = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "support", "stage5"
)
if STAGE5_RUNTIME not in sys.path:
    sys.path.insert(0, STAGE5_RUNTIME)

from stage5_real_jump import main


if __name__ == "__main__":
    raise SystemExit(main())
