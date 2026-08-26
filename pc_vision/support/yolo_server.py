#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第四赛段识别 - 电脑端实时识别服务（识别 + 全量回传 + 语音播报闭环）

功能：
  1) 接收狗推来的 JPEG 帧，用 GPU(YOLO) 实时识别并显示；
  2) 每帧把【全部检测结果】(类别/置信度/bbox面积/面积占比) 通过 TCP 推给狗端
     remote_detector.py（--push-ip <狗IP>，端口 9890），供 stage4_real 控制使用；
  3) 当目标置信度 >= --speak-conf 且连续 --stable 帧命中时，把类别回传给狗
     speak_on_detect.py(9888) 播报"识别到XX"（每类有冷却，不会一直播报）。

用法（先跑本脚本，再在狗上跑 stream_images.py 和 speak_on_detect.py / stage4_real.py）：
  python live_detect_server.py
"""
import os
import sys
import glob
import json
import socket
import struct
import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO


def default_model():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    cands = glob.glob(os.path.join(base, "stage4_*.pt"))
    if cands:
        return max(cands, key=os.path.getmtime)
    return os.path.join(base, "stage4_cola.pt")


def draw_detections(img, r, model):
    """把检测框画到图上：类别 置信度 占画面%"""
    ann = img.copy()
    h, w = img.shape[:2]
    if r.boxes is not None:
        for box in r.boxes:
            cid = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cid]
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            norm_pct = (x2 - x1) * (y2 - y1) / max(h * w, 1) * 100.0
            cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = "%s %.2f 占%.1f%%" % (name, conf, norm_pct)
            cv2.putText(ann, label, (x1, max(y1 - 8, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return ann


def main():
    ap = argparse.ArgumentParser(description="电脑端实时 YOLO 识别（可乐/足球/橙球/障碍物/限高杆...）")
    ap.add_argument("--model", default=default_model())
    ap.add_argument("--port", type=int, default=9891, help="接收狗推流端口")
    ap.add_argument("--conf", type=float, default=0.25, help="显示/判定置信度阈值")
    ap.add_argument("--targets", default="cola,football,orange_ball,obstacle,limit_bar",
                    help="关注类别，逗号分隔")
    ap.add_argument("--save-dir", default=None, help="保存命中图片的目录（可选）")
    ap.add_argument("--show", action="store_true", default=True, help="live camera window (default on)")
    ap.add_argument("--no-show", action="store_true", help="disable window")
    # ---- 回传 + 语音 ----
    ap.add_argument("--dog-ip", default="192.168.43.247", help="狗 IP，设置后识别到目标会把类别回传给狗触发语音")
    ap.add_argument("--dog-port", type=int, default=9888, help="狗端 speak_on_detect.py 监听端口")
    ap.add_argument("--speak-conf", type=float, default=0.75, help="触发语音的默认置信度阈值")
    ap.add_argument("--speak-conf-map", default="limit_bar:0.65",
                    help="按类别覆盖播报置信度阈值，如 limit_bar:0.65（逗号分隔）")
    ap.add_argument("--stable", type=int, default=3, help="连续几帧命中才播报")
    ap.add_argument("--cooldown", type=float, default=15.0, help="同类目标播报冷却秒数")
    ap.add_argument("--speak-norm-map", default="football:0.07,limit_bar:0.65,cola:0.08,orange_ball:0.05",
                    help="按类别覆盖播报面积占比阈值，如 football:0.07,limit_bar:0.65（逗号分隔）")
    ap.add_argument("--speak-min-norm", type=float, default=0.08,
                    help="触发语音的最小 bbox 面积占比(0~1)，目标占画面小于此值不播报；0=不限(仅按置信度)")
    # ---- 全量检测推送（给 stage4_real 用） ----
    ap.add_argument("--push-ip", default="192.168.43.247", help="狗 IP：每帧推送全部检测结果给狗端 remote_detector(9890)")
    ap.add_argument("--push-port", type=int, default=9890, help="狗端 remote_detector 监听端口")
    args = ap.parse_args()

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    speak_norm_map = {}
    for item in args.speak_norm_map.split(","):
        if ":" in item:
            k, v = item.split(":", 1)
            k = k.strip()
            if k:
                try:
                    speak_norm_map[k] = float(v.strip())
                except ValueError:
                    pass
    speak_conf_map = {}
    for item in args.speak_conf_map.split(","):
        if ":" in item:
            k, v = item.split(":", 1)
            k = k.strip()
            if k:
                try:
                    speak_conf_map[k] = float(v.strip())
                except ValueError:
                    pass
    if speak_norm_map:
        print("按类别播报面积阈值: %s" % speak_norm_map)
    if speak_conf_map:
        print("按类别播报置信度阈值: %s" % speak_conf_map)
    print("加载模型: %s" % args.model)
    model = YOLO(args.model)
    print("模型类别(%d): %s | 关注: %s" % (len(model.names), list(model.names.values()), targets))
    print("语音回传: %s (默认置信度>=%.2f，类别阈值优先，类别面积阈值优先于通用阈值 %.2f，连续%d帧，冷却%.0fs)" % (
        ("狗 %s:%d" % (args.dog_ip, args.dog_port)) if args.dog_ip else "关闭",
        args.speak_conf, args.speak_min_norm, args.stable, args.cooldown))
    print("全量检测推送: %s" % (("狗 %s:%d" % (args.push_ip, args.push_port)) if args.push_ip else "关闭"))
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        print("命中图保存到: %s" % args.save_dir)

    hit_cnt = {}      # class -> 连续达标帧数
    last_spoke = {}   # class -> 上次播报时间
    spoken_once = set()
    one_shot_classes = {"football", "orange_ball", "cola"}
    push_sock = None

    def send_to_dog(cls_name):
        # 每次播报都新建一条连接（狗端 speak_on_detect 收一条就关连接，
        # 复用长连接会导致后续消息发到已关闭的连接上丢失）
        if not args.dog_ip:
            return False
        try:
            with socket.create_connection((args.dog_ip, args.dog_port), timeout=3) as sock:
                sock.sendall((cls_name + "\n").encode("utf-8"))
            return True
        except Exception as e:
            print("[回传] 连接狗失败: %s" % e)
            return False

    def push_dets(dets_json):
        nonlocal push_sock
        if not args.push_ip:
            return
        try:
            if push_sock is None:
                push_sock = socket.create_connection((args.push_ip, args.push_port), timeout=3)
            push_sock.sendall((dets_json + "\n").encode("utf-8"))
        except Exception as e:
            try:
                if push_sock:
                    push_sock.close()
            except Exception:
                pass
            push_sock = None
            # 不刷屏：只在首次失败时提示
            if not hasattr(push_dets, "_warned"):
                push_dets._warned = True
                print("[推送] 连接狗 %s:%d 失败: %s（将自动重试）" % (args.push_ip, args.push_port, e))

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port))
    srv.listen(1)
    print("等待狗连接 %s:%d ... (Ctrl+C 退出)" % ("0.0.0.0", args.port))

    try:
        conn, addr = srv.accept()
    except KeyboardInterrupt:
        print("退出")
        return 0
    print("狗已连接: %s" % (addr,))
    conn.settimeout(15)

    frame_count = 0
    hit_count = 0
    last_print = 0.0
    try:
        while True:
            try:
                hdr = conn.recv(4)
                if len(hdr) < 4:
                    print("连接关闭")
                    break
                n = struct.unpack(">I", hdr)[0]
                payload = b""
                while len(payload) < n:
                    chunk = conn.recv(n - len(payload))
                    if not chunk:
                        break
                    payload += chunk
                if len(payload) < n:
                    print("数据不完整，连接关闭")
                    break
            except socket.timeout:
                continue
            except (ConnectionResetError, BrokenPipeError):
                print("连接断开")
                break

            img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            frame_count += 1
            results = model(img, conf=args.conf, verbose=False)
            r = results[0]
            h, w = img.shape[:2]

            # ---- 全量检测推送（给狗端控制用） ----
            all_dets = []
            if r.boxes is not None:
                for box in r.boxes:
                    cid = int(box.cls[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bw, bh = x2 - x1, y2 - y1
                    all_dets.append({
                        "name": model.names[cid],
                        "conf": round(float(box.conf[0]), 4),
                        "area": round(bw * bh, 1),
                        "area_norm": round((bw * bh) / max(h * w, 1), 4),
                        "cx": round(((x1 + x2) / 2.0) / max(w, 1), 4),
                        "x1_norm": round(x1 / max(w, 1), 4),
                        "y1_norm": round(y1 / max(h, 1), 4),
                        "x2_norm": round(x2 / max(w, 1), 4),
                        "y2_norm": round(y2 / max(h, 1), 4),
                    })
            if args.push_ip:
                push_dets(json.dumps({"t": time.time(), "dets": all_dets}))

            # ---- 关注类别（显示/语音），带 area_norm（距离闸） ----
            hits = []
            for d in all_dets:
                if d["name"] in targets:
                    hits.append((d["name"], d["conf"], d["area_norm"]))

            # 语音触发：置信度 + 面积占比(距离) + 连续帧数 + 每类冷却
            best_conf = {}
            best_norm = {}
            for name, c, norm in hits:
                best_conf[name] = max(best_conf.get(name, 0), c)
                best_norm[name] = max(best_norm.get(name, 0), norm)
            now = time.time()
            for name in targets:
                if name in spoken_once:
                    hit_cnt[name] = 0
                    continue
                norm_gate = speak_norm_map.get(name, args.speak_min_norm)
                conf_gate = speak_conf_map.get(name, args.speak_conf)
                if (name in best_conf and best_conf[name] >= conf_gate
                        and (norm_gate <= 0 or best_norm[name] >= norm_gate)):
                    hit_cnt[name] = hit_cnt.get(name, 0) + 1
                    if hit_cnt[name] >= args.stable and now - last_spoke.get(name, 0) >= args.cooldown:
                        if send_to_dog(name):
                            last_spoke[name] = now
                            if name in one_shot_classes:
                                spoken_once.add(name)
                            print("★触发语音: %s (conf=%.2f/%.2f, norm=%.3f/%.3f)" %
                                  (name, best_conf[name], conf_gate,
                                   best_norm[name], norm_gate))
                else:
                    hit_cnt[name] = 0

            if hits or now - last_print >= 0.5:
                line = "[帧%d] " % frame_count
                if hits:
                    line += "★识别到: " + ", ".join("%s(%.2f)" % (n, c) for n, c, _norm in hits)
                    hit_count += 1
                else:
                    line += "无目标"
                print(line, flush=True)
                last_print = now

            if args.save_dir and hits:
                ann = draw_detections(img, r, model)
                fn = os.path.join(args.save_dir, "hit_%05d.jpg" % frame_count)
                cv2.imwrite(fn, ann)

            if args.show and not args.no_show:
                ann = draw_detections(img, r, model)
                cv2.imshow("Dog Camera (Q/ESC quit)", ann)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    print("user quit")
                    break
    except KeyboardInterrupt:
        print("退出")
    finally:
        try:
            conn.close()
        except Exception:
            pass
        try:
            if push_sock:
                push_sock.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        srv.close()
    print("推流结束：共处理 %d 帧，命中目标 %d 帧" % (frame_count, hit_count))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
