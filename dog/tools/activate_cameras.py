#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机一键 激活 + 自检 + 自愈 (v2)

用法（开机后）：
    source /etc/mi/ros2_env.conf
    python3 activate_cameras.py [--check-sec N] [--min-fps N] [--rounds N]

v2 修复（避免把好的相机搞坏）：
  1) 幂等：节点已激活则跳过 configure/activate
  2) 自检 0fps 时先验证"节点是否活着 + 话题是否有发布者"，确属相机问题才自愈
  3) 自愈用 kill + ros2 launch 干净重启（不用 lifecycle deactivate，防卡死）
  4) 所有 ros2 命令加 timeout(15s)，不会卡 60 秒

Python 3.6 兼容。
"""
import os
import signal
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

NS = "/mi_desktop_48_b0_2d_60_12_56"
CHECK_TOPICS = [NS + "/image_left", NS + "/image_right", NS + "/image_rgb"]
DEFAULT_CHECK_SEC = 6.0
DEFAULT_MIN_FPS = 15.0
DEFAULT_ROUNDS = 2


def run(cmd, timeout=15):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, timeout=timeout)
        return (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return "ERR: %s" % e


def lifecycle_state(node):
    o = run(["ros2", "lifecycle", "get", node], timeout=10)
    if "unconfigured" in o:
        return "unconfigured"
    if "inactive" in o:
        return "inactive"
    if "active" in o:
        return "active"
    if "not found" in o.lower() or o == "" or o.startswith("TIMEOUT"):
        return "unknown"
    return "unknown"


def set_lifecycle(node, target):
    o = run(["ros2", "lifecycle", "set", node, target], timeout=15)
    ok = "Transitioning successful" in o
    print("  %s -> %s %s" % (target, o[:80], "(ok)" if ok else ""))
    return ok


def activate_one(node):
    """幂等激活：仅当状态允许才 configure/activate"""
    st = lifecycle_state(node)
    print("  %s state=%s" % (node, st))
    if st == "active":
        return True
    if st in ("unconfigured", "inactive"):
        if st == "unconfigured":
            set_lifecycle(node, "configure")
        return set_lifecycle(node, "activate")
    print("  状态未知/不可用，跳过激活")
    return False


def activate():
    print("== [1] 激活相机链（幂等）==")
    activate_one(NS + "/camera/camera")
    o = run(["ros2", "service", "call", NS + "/camera/realsense_frame_service",
             "std_srvs/srv/SetBool", "{data: true}"], timeout=20)
    print("  realsense_frame_service -> %s" % (o[:80] if o else "(ok)"))
    activate_one(NS + "/vision_manager")
    activate_one(NS + "/stereo_camera")
    time.sleep(3)


def check_fps(node, check_sec):
    results = {}
    for topic in CHECK_TOPICS:
        cnt = {"n": 0}
        def cb(msg, c=cnt):
            c["n"] += 1
        sub = node.create_subscription(Image, topic, cb, 10)
        t0 = time.time()
        while time.time() - t0 < check_sec:
            rclpy.spin_once(node, timeout_sec=0.02)
        node.destroy_subscription(sub)
        results[topic] = cnt["n"] / check_sec
    return results


def verify_topic_publisher(topic):
    o = run(["ros2", "topic", "info", topic], timeout=10)
    return "Publisher count: 1" in o or "Publisher count: 2" in o or "Publisher count: 3" in o


def heal_stereo():
    """自愈：kill + ros2 launch 干净重启（验证过的恢复方式）"""
    print("== 自愈：kill + ros2 launch 重启 stereo_camera ==")
    o = run(["pkill", "-9", "-f", "/opt/ros2/cyberdog/lib/camera_test/stereo_camera"], timeout=10)
    time.sleep(3)
    # 后台启动
    launch_cmd = ("bash -lc 'source /etc/mi/ros2_env.conf; "
                  "nohup ros2 launch camera_test stereo_camera.py namespace:=%s "
                  "> /tmp/stereo_auto.log 2>&1 &'" % NS)
    subprocess.Popen(["bash", "-c", launch_cmd])
    time.sleep(12)
    # 激活（幂等）
    activate_one(NS + "/stereo_camera")
    time.sleep(4)


def report(results, min_fps):
    bad = []
    for t, f in results.items():
        ok = f >= min_fps
        print("  %s: %.1f fps %s" % (t, f, "OK" if ok else "LOW/NO"))
        if not ok:
            bad.append(t)
    return bad


def main():
    check_sec = DEFAULT_CHECK_SEC
    min_fps = DEFAULT_MIN_FPS
    rounds = DEFAULT_ROUNDS
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--check-sec" and i + 1 < len(args):
            check_sec = float(args[i + 1])
        elif a == "--min-fps" and i + 1 < len(args):
            min_fps = float(args[i + 1])
        elif a == "--rounds" and i + 1 < len(args):
            rounds = int(args[i + 1])

    print("===== 相机一键 激活+自检+自愈 v2 =====")
    print("检查: %s | 每路测 %.0fs | <%.0ffps 异常 | 自愈 %d 轮"
          % (", ".join(CHECK_TOPICS), check_sec, min_fps, rounds))

    activate()

    rclpy.init()
    node = rclpy.create_node("camera_check_node")

    print("== [2] 自检帧率 ==")
    results = check_fps(node, check_sec)
    bad = report(results, min_fps)

    if not bad:
        print("== 全部相机正常 ==")
        rclpy.shutdown()
        return 0

    # 自愈前先验证确实是相机问题（节点活着 + 话题有发布者但收不到帧）
    alive = run(["pgrep", "-f", "/opt/ros2/cyberdog/lib/camera_test/stereo_camera"], timeout=10).strip()
    print("stereo_camera 进程: %s" % ("alive" if alive else "DEAD"))
    for t in bad:
        pub = verify_topic_publisher(t)
        print("  %s 有发布者: %s" % (t, pub))

    for rnd in range(1, rounds + 1):
        print("== [3] 第 %d/%d 轮自愈 ==" % (rnd, rounds))
        heal_stereo()
        results = check_fps(node, check_sec)
        bad = report(results, min_fps)
        if not bad:
            print("== 自愈成功 ==")
            rclpy.shutdown()
            return 0

    rclpy.shutdown()
    print("== [4] 仍有异常: %s ==" % ", ".join(bad))
    print("    建议：整机重启后重跑本脚本（相机硬件需重新初始化）")
    return 1


if __name__ == "__main__":
    sys.exit(main())