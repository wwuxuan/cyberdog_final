#!/usr/bin/env python3
"""Slow walking pitch comparison for the CyberDog body controller."""

import argparse
import time

from adapter import RealDogAdapter


def run_phase(adapter, label, pitch, speed, seconds):
    print(
        "[pitch_test] phase=%s pitch=%+.3frad speed=%.2fm/s duration=%.1fs" % (
            label,
            pitch,
            speed,
            seconds,
        ),
        flush=True,
    )
    deadline = time.monotonic() + seconds
    last_log_time = 0.0
    while time.monotonic() < deadline:
        adapter.walk(speed, 0.0, 0.0, pitch=pitch)
        now = time.monotonic()
        if now - last_log_time >= 0.5:
            print(
                "[pitch_test] phase=%s remaining=%.1fs pitch=%+.3frad" % (
                    label,
                    max(0.0, deadline - now),
                    pitch,
                ),
                flush=True,
            )
            last_log_time = now
        time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(
        description="Low-speed forward walk with neutral, positive, and negative pitch"
    )
    parser.add_argument("--arm", action="store_true", help="allow walking commands")
    parser.add_argument("--stand", action="store_true", help="stand before the test")
    parser.add_argument("--speed", type=float, default=0.08)
    parser.add_argument("--normal-secs", type=float, default=1.5)
    parser.add_argument("--positive-secs", type=float, default=2.0)
    parser.add_argument("--negative-secs", type=float, default=2.0)
    parser.add_argument("--pitch", type=float, default=0.20)
    args = parser.parse_args()

    if not args.arm:
        print("[pitch_test] dry run; add --arm to send walking commands", flush=True)
        return
    if args.speed <= 0.0 or args.speed > 0.20:
        parser.error("--speed must be in (0, 0.20]")
    if args.pitch <= 0.0 or args.pitch > 0.35:
        parser.error("--pitch must be in (0, 0.35]")

    phases = (
        ("neutral", 0.0, args.normal_secs),
        ("positive", args.pitch, args.positive_secs),
        ("negative", -args.pitch, args.negative_secs),
    )
    if any(seconds <= 0.0 for _label, _pitch, seconds in phases):
        parser.error("all phase durations must be positive")

    adapter = RealDogAdapter(None)
    try:
        if args.stand:
            print("[pitch_test] stand", flush=True)
            adapter.stand()
        print(
            "[pitch_test] total distance about %.2fm; Ctrl+C stops immediately" % (
                args.speed * sum(seconds for _label, _pitch, seconds in phases),
            ),
            flush=True,
        )
        for label, pitch, seconds in phases:
            run_phase(adapter, label, pitch, args.speed, seconds)
        adapter.stop()
        print("[pitch_test] done", flush=True)
    except KeyboardInterrupt:
        adapter.stop()
        print("\n[pitch_test] interrupted and stopped", flush=True)
    finally:
        try:
            adapter.stop()
        finally:
            adapter.shutdown()


if __name__ == "__main__":
    main()
