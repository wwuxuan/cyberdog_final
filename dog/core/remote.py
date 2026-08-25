#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
狗端：电脑 YOLO 识别结果接收器（RemoteDetector）

配合电脑端 live_detect_server.py 的 --push-ip <狗IP> 使用：
电脑每帧把全部检测结果用一行 JSON 推送到本监听端口（默认 9890）。

接口与初赛 yolo_detector.YoloDetector 保持一致，stage4_real.py 可直接替换：
  detections = detector.get_detections()
  r = detector.nearest_of(["limit_bar", "obstacle"])   # -> (class_name, bbox_area) 或 None
  d = detector.get_front_distance()                    # 无雷达，恒返回 None
  detector.alive()                                     # 最近 1.5s 内是否收到过电脑数据
"""
import json
import socket
import threading
import time

KEY_MAP = {
    "name": "class_name",
    "conf": "confidence",
    "area": "bbox_area",
    "area_norm": "bbox_area_norm",
    "cx": "bbox_cx_norm",
}


class RemoteDetector(object):
    def __init__(self, port=9890, stale_timeout=1.5):
        self._port = port
        self._timeout = stale_timeout
        self._lock = threading.Lock()
        self._detections = []
        self._last = 0.0
        self._frames = 0
        self._running = True
        self._th = threading.Thread(target=self._server, daemon=True)
        self._th.start()

    # ---------- TCP server ----------
    def _server(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind(("0.0.0.0", self._port))
            srv.listen(1)
        except OSError as e:
            print("[remote_detector] bind 0.0.0.0:%d failed: %s" % (self._port, e))
            return
        print("[remote_detector] 监听 0.0.0.0:%d（电脑端 live_detect_server 加 --push-ip 狗IP）" % self._port)
        while self._running:
            try:
                conn, addr = srv.accept()
            except OSError:
                break
            conn.settimeout(1.0)
            with conn:
                buf = b""
                while self._running:
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
                        if not line:
                            continue
                        try:
                            obj = json.loads(line.decode("utf-8", "replace"))
                            self._apply(obj)
                        except Exception as e:
                            print("[remote_detector] parse err:", e)
            if self._running:
                print("[remote_detector] 电脑端断开，等待重连 ...")

    def _apply(self, obj):
        dets = []
        for d in obj.get("dets", []):
            item = {}
            for k, v in d.items():
                item[KEY_MAP.get(k, k)] = v
            item.setdefault("bbox_area", 0.0)
            item.setdefault("bbox_area_norm", 0.0)
            dets.append(item)
        with self._lock:
            self._detections = dets
            self._last = time.time()
            self._frames += 1

    # ---------- public API ----------
    def stop(self):
        self._running = False

    def alive(self):
        return self.age() <= self._timeout

    def age(self):
        with self._lock:
            last = self._last
        return float("inf") if last <= 0.0 else time.time() - last

    def status(self):
        age = self.age()
        with self._lock:
            count = len(self._detections)
            frames = self._frames
        return {
            "alive": age <= self._timeout,
            "age": age,
            "timeout": self._timeout,
            "detections": count,
            "frames": frames,
        }

    def get_detections(self):
        with self._lock:
            return list(self._detections)

    def nearest_of(self, classes):
        cands = [d for d in self.get_detections() if d.get("class_name") in classes]
        if not cands:
            return None
        best = max(cands, key=lambda d: d.get("bbox_area", 0.0))
        return (best["class_name"], best.get("bbox_area", 0.0))

    def get_front_distance(self):
        return None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="狗端远程识别接收（调试用，打印收到的检测）")
    ap.add_argument("--port", type=int, default=9890)
    args = ap.parse_args()
    det = RemoteDetector(port=args.port)
    try:
        while True:
            dets = det.get_detections()
            if dets:
                print("[%s] dets=%d %s" % (
                    "alive" if det.alive() else "STALE",
                    len(dets),
                    [(d["class_name"], round(d.get("confidence", 0), 2)) for d in dets]))
            time.sleep(0.5)
    except KeyboardInterrupt:
        det.stop()
