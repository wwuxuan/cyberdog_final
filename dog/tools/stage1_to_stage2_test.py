#!/usr/bin/env python3
"""Run stage1 and finish at the stage2 standalone test start.

This is the coordinate-remap variant: it uses two right-line readings at the
stage1 turn to correct the field map, without side-stepping the dog to the
lines.  For physical side-step calibration, run the stage-dog script directly.
"""

import sys

from s1 import main


if __name__ == "__main__":
    if "--turn-calibration" not in sys.argv:
        sys.argv.extend(("--turn-calibration", "map"))
    raise SystemExit(main())
