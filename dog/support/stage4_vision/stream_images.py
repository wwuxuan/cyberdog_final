#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第四赛段识别 - 狗端实时推流脚本
功能：订阅前方 /image_rgb，压缩成 JPEG 通过 TCP 实时发给电脑（只推流，不动狗）
用法：
  source /etc/mi/ros2_env.conf
  python3 stream_images.py --host 10.151.9.227 --port 9876
  # 电脑要先运行 live_detect_server.py
"""
import os
import time
import socket
import struct
import argparse

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

DEFAULT_NS = "/mi_desktop_48_b0_2d_60_12_56"


def main():
    ap = argparse.ArgumentParser(description="狗端实时推流 /image_rgb 到电脑")
    ap.add_argument("--host", required=True, help="电脑 IP，如 10.151.9.227")
    ap.add_argument("--port", type=int, default=9876)
    ap.add_argument("--fps", type=float, default=15.0, help="推流帧率（默认15，画面流畅/延迟低）")
    ap.add_argument("--quality", type=int, default=60, help="JPEG 质量 0-100")
    ap.add_argument("--ns", default=DEFAULT_NS)
    args = ap.parse_args()

    rclpy.init()
    node = rclpy.create_node("stage4_stream")

    latest = {"msg": None}
    def cb(msg):
        latest["msg"] = msg
    node.create_subscription(Image, args.ns.rstrip("/") + "/image_rgb", cb, 10)

    print("等待图像...")
    t0 = time.time()
    while latest["msg"] is None and time.time() - t0 < 8:
        rclpy.spin_once(node, timeout_sec=0.1)
    if latest["msg"] is None:
        print("没收到图像！请先激活相机：python3 /home/mi/cyberdog_competition/activate_cameras.py")
        return 1
    first = latest["msg"]
    chan = {"rgb8": 3, "bgr8": 3, "mono8": 1}.get(first.encoding, 3)
    print("图像: %s %dx%d" % (first.encoding, first.width, first.height))

    print("连接电脑 %s:%d ..." % (args.host, args.port))
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(10)
    s.connect((args.host, args.port))
    print("已连接，开始推流 (fps=%.1f, quality=%d)" % (args.fps, args.quality))

    last_send = 0.0
    try:
        while True:
            rclpy.spin_once(node, timeout_sec=0.02)
            now = time.time()
            if now - last_send < 1.0 / args.fps:
                continue
            last_send = now
            msg = latest["msg"]
            if msg is None:
                continue
            data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            if msg.step == msg.width * chan:
                img = data.reshape(msg.height, msg.width, chan)
            else:
                img = data.reshape(msg.height, msg.step)[:, : msg.width * chan].reshape(msg.height, msg.width, chan)
            if msg.encoding == "rgb8":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
            if not ok:
                continue
            payload = buf.tobytes()
            s.sendall(struct.pack(">I", len(payload)))
            s.sendall(payload)
    except KeyboardInterrupt:
        print("停止推流")
    except (BrokenPipeError, ConnectionResetError, socket.timeout) as e:
        print("连接中断: %s" % e)
    finally:
        s.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())