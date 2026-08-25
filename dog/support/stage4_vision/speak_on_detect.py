#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
狗端：接收电脑识别结果并语音播报（配合电脑端 live_detect_server.py 使用）
原理：
  电脑端识别到目标(置信度达标)后，通过 TCP 把类别名发到本脚本(默认端口 9888)；
  本脚本收到后，向 {ns}/speech_play_extend 发布 AudioPlayExtend(is_online=True, text="识别到可乐")，
  由 cyberdog_audio 在线 TTS 播报。同类播报有冷却时间，不会一直重复。
运行（先 source ROS2 环境）：
  source /etc/mi/ros2_env.conf
  python3 speak_on_detect.py
  # 自定义端口/命名空间：
  python3 speak_on_detect.py --port 9888 --ns /mi_desktop_48_b0_2d_60_12_56
"""
import argparse
import socket
import threading
import time
import queue

import rclpy
from rclpy.node import Node
from protocol.msg import AudioPlayExtend

DEFAULT_NS = "/mi_desktop_48_b0_2d_60_12_56"
# 类别 -> 播报文案（可按需增改）
CLASS_TEXT = {
    "cola": "识别到可乐瓶",
    "football": "识别到足球",
    "orange_ball": "识别到橙色小球",
    "obstacle": "识别到无法跨越障碍",
    "limit_bar": "识别到限高杆",
}

# 下行静音：stage4_real.py 上行写 "up"、下行写 "down" 到该文件；下行时本脚本不播报
MUTE_FILE = "/tmp/stage4_speech_mute"
ONE_SHOT_CLASSES = {"football", "orange_ball", "cola"}


def main():
    ap = argparse.ArgumentParser(description="狗端：接收识别结果并语音播报")
    ap.add_argument("--port", type=int, default=9888, help="监听端口（电脑端 --dog-port 要一致）")
    ap.add_argument("--ns", default=DEFAULT_NS, help="ROS2 命名空间")
    ap.add_argument("--module", default="stage4", help="模块名")
    ap.add_argument("--cooldown", type=float, default=15.0, help="同类文案播报冷却秒数")
    args = ap.parse_args()

    rclpy.init()
    node = rclpy.create_node("speak_on_detect")
    topic = args.ns.rstrip("/") + "/speech_play_extend"
    pub = node.create_publisher(AudioPlayExtend, topic, 10)
    print("===== 狗端语音播报服务 =====")
    print("监听端口: %d | 话题: %s | 冷却: %.0fs" % (args.port, topic, args.cooldown))
    print("等待电脑端回传识别结果 (Ctrl+C 退出) ...")

    msg_q = queue.Queue()
    last_spoke = {}
    spoken_once = set()

    def muted():
        try:
            with open(MUTE_FILE) as f:
                return f.read().strip().lower() == "down"
        except Exception:
            return False

    def do_speak(cls):
        cls = (cls or "").strip()
        if not cls:
            return
        text = CLASS_TEXT.get(cls, "识别到" + cls)
        if muted():
            print("[speak] 下行静音，跳过: %s" % text, flush=True)
            return
        if cls in ONE_SHOT_CLASSES and cls in spoken_once:
            print("[speak] 本次任务已播报过，跳过: %s" % text, flush=True)
            return
        now = time.time()
        if now - last_spoke.get(text, 0) < args.cooldown:
            return
        last_spoke[text] = now
        msg = AudioPlayExtend()
        msg.module_name = args.module
        msg.is_online = True
        msg.text = text
        pub.publish(msg)
        if cls in ONE_SHOT_CLASSES:
            spoken_once.add(cls)
        print("已播报: %s" % text, flush=True)

    def server():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", args.port))
        srv.listen(1)
        print("TCP 服务已启动 0.0.0.0:%d（支持长连接持续接收）" % args.port)
        while True:
            try:
                conn, addr = srv.accept()
            except OSError:
                break
            conn.settimeout(60)
            with conn:
                buf = b""
                while True:
                    try:
                        data = conn.recv(4096)
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                    if not data:
                        break
                    buf += data
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if line:
                            msg_q.put(line.decode("utf-8", "replace"))

    th = threading.Thread(target=server, daemon=True)
    th.start()
    try:
        while True:
            try:
                cls = msg_q.get_nowait()
                do_speak(cls)
            except queue.Empty:
                pass
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        print("退出")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
