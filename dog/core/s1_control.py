import os
import shlex
import sys
import time


def _ensure_ros_env():
    if os.environ.get("ROS_DISTRO") or os.name != "posix":
        return
    if os.environ.get("CYBERDOG_ROS_REEXEC"):
        return
    setup_candidates = [
        "/opt/ros2/cyberdog/setup.bash",
        "/opt/ros2/galactic/setup.bash",
        "/opt/ros/foxy/setup.bash",
        "/opt/ros/galactic/setup.bash",
        "/opt/ros/humble/setup.bash",
    ]
    setup = next((p for p in setup_candidates if os.path.exists(p)), None)
    if setup is None:
        return
    env = dict(os.environ)
    env["CYBERDOG_ROS_REEXEC"] = "1"
    script = os.path.abspath(__file__)
    args = " ".join(shlex.quote(a) for a in sys.argv[1:])
    cmd = f"source {shlex.quote(setup)} && cd {shlex.quote(os.path.dirname(script))} && exec python3 {shlex.quote(script)} {args}"
    os.execve("/bin/bash", ["bash", "-lc", cmd], env)


_ensure_ros_env()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from adapter import RealDogAdapter


TARGET_X = 3.07
EXIT_Y = 0.85
STAGE3_EXIT_Y = 6.58
STAGE3_LINE_TARGET = 0.30
STAGE3_LINE_K_VY = 0.18
STAGE3_LINE_MAX_VY = 0.02
STAGE3_LINE_HZ = 6.0
STAGE3_LINE_DEADBAND = 0.03
STAGE3_LINE_WZ_GATE = 0.16
STAGE3_LINE_ACTIVE_START = 0.20
STAGE3_LINE_ACTIVE_END = 0.80
STAGE3_LINE_BIAS_MAX = 0.12

LANE_K_Y = 0.55
LANE_MAX_VY = 0.12
LANE_K_YAW = 0.10
LANE_MAX_WZ = 0.35
WALK_VX = 0.18
EXIT_WALK_VX = 0.16


def _clamp(value, low, high):
    return max(low, min(high, value))


def _yaw_err(target, current):
    return ((target - current) + 180.0) % 360.0 - 180.0


def stage1_lane_keep(adapter, target_y):
    _, y, _ = adapter.get_position()
    yaw = adapter.get_yaw_deg()
    vy = _clamp(LANE_K_Y * (target_y - y), -LANE_MAX_VY, LANE_MAX_VY)
    wz = _clamp(LANE_K_YAW * _yaw_err(0.0, yaw), -LANE_MAX_WZ, LANE_MAX_WZ)
    adapter.walk(WALK_VX, vy, wz)


def main():
    adapter = RealDogAdapter(None)

    try:
        print("========================================")
        print("  real dog competition - stage 1+2+3")
        print("========================================")

        if not adapter.wait_odom(timeout=3.0):
            print("[main] WARN: no odom yet, continuing with zeros")

        adapter.stand()
        adapter.set_origin()

        print("\n===== Stage 1: straight + turn =====")
        print("[stage1] align yaw 0")
        adapter.align_yaw(0.0)
        adapter.set_origin()
        x, y, _ = adapter.get_position()
        target_y = y
        phase = "forward_x"
        t0 = time.time()
        last_log = 0.0
        print(f"[stage1] start=({x:.3f},{y:.3f}) yaw={adapter.get_yaw_deg():.1f}")

        while time.time() - t0 < 90.0:
            x, y, _ = adapter.get_position()
            yaw = adapter.get_yaw_deg()
            now = time.time()
            if now - last_log >= 2.0:
                print(f"[odom] x={x:.3f} y={y:.3f} yaw={yaw:.1f} phase={phase}")
                last_log = now

            if phase == "forward_x":
                if x >= TARGET_X:
                    adapter.stop()
                    print(f"[stage1] reached x={x:.3f}, turn to +y")
                    adapter.align_yaw(90.0)
                    adapter.align_x(TARGET_X)
                    phase = "forward_y"
                    continue
                stage1_lane_keep(adapter, target_y)

            elif phase == "forward_y":
                if y >= EXIT_Y:
                    adapter.stop()
                    print(f"[stage1] done y={y:.3f}")
                    break
                wz = _clamp(0.025 * _yaw_err(90.0, yaw), -LANE_MAX_WZ, LANE_MAX_WZ)
                adapter.walk(EXIT_WALK_VX, 0.0, wz)

            time.sleep(0.07)

        print("\n===== Stage 2: ball hitting =====")
        from s2 import run_stage2
        stage2_ok = run_stage2(adapter)
        print(f"[main] stage2_ok={stage2_ok}")
        if not stage2_ok:
            print("[main] skip stage3 because stage2 failed")
            return

        print("\n===== Stage 3: curve follow =====")
        from s3 import run_stage3
        line_monitor = None
        try:
            # Match the standalone stage3 test: initialise ROS/CV before
            # movement so it cannot interrupt the locomotion heartbeat.
            from line_monitor import LineDistanceMonitor

            print("[main] prewarm stage3 fisheye monitor while standing")
            line_monitor = LineDistanceMonitor(
                target_dist=STAGE3_LINE_TARGET,
                k_vy=STAGE3_LINE_K_VY,
                max_vy=STAGE3_LINE_MAX_VY,
                hz=STAGE3_LINE_HZ,
                deadband=STAGE3_LINE_DEADBAND,
            )
            if not line_monitor.start(timeout=12.0, activate=True):
                print("[main] WARN: stage3 fisheye unavailable; continue without it")
                line_monitor.close()
                line_monitor = None

            stage3_ok = run_stage3(
                adapter,
                exit_y=STAGE3_EXIT_Y,
                vy_enabled=False,
                line_monitor=line_monitor,
                line_active_start=STAGE3_LINE_ACTIVE_START,
                line_active_end=STAGE3_LINE_ACTIVE_END,
                line_bias_max=STAGE3_LINE_BIAS_MAX,
                line_wz_gate=STAGE3_LINE_WZ_GATE,
            )
            print(f"[main] stage3_ok={stage3_ok}")
        finally:
            if line_monitor is not None:
                line_monitor.close()

        print("\n[main] stage 1-3 finished")

    except KeyboardInterrupt:
        print("\n[main] interrupted")
    finally:
        adapter.shutdown()
        print("[main] end")


if __name__ == "__main__":
    main()
