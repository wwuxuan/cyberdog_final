#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
旧狗自定义低姿态（限高杆）步态控制模块

机制（照搬 low_gait_test / 旧狗 Robot_Ctrl.py）：
  1) 用 file_send_lcmt 把 Gait_Def_limit.toml 发到 user_gait_file，等 user_gait_result ACK；
  2) 再把 Gait_Params_limit.toml 生成的 Full_Params（可覆盖 vel_x）发上去等 ACK；
  3) 之后把运动指令心跳改成 mode=62/gait=110（由 stage4_real 改 adapter.cmd）即可低姿行走。

注意：低姿前进速度由 Full_Params 的 vel_x 决定；前进/倒退速度不同时需按 vel_x 重新上传。
"""
import os
import sys
import time
import threading

import lcm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

URL = "udpm://239.255.76.67:7671?ttl=255"
CH_GAIT_FILE = "user_gait_file"
CH_GAIT_RESULT = "user_gait_result"


class LowGait(object):
    def __init__(self, gait_dir=None):
        if gait_dir is None:
            gait_dir = os.path.join(BASE_DIR, "gait")
        self.dir = gait_dir
        self.def_path = os.path.join(gait_dir, "Gait_Def_limit.toml")
        self.params_path = os.path.join(gait_dir, "Gait_Params_limit.toml")
        self.list_path = os.path.join(gait_dir, "Usergait_List.toml")
        missing = [p for p in (self.def_path, self.params_path, self.list_path)
                   if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError("缺少低姿步态文件: %s" % missing)

        self.lc = lcm.LCM(URL)          # 只用于 publish（user_gait_file）
        self.lc_ack = lcm.LCM(URL)      # 独立实例，只用于订阅 user_gait_result
        self._evt = threading.Event()
        self._ok = {"v": False}
        self._lock = threading.Lock()
        self._running = [True]
        self._last_vel = None
        threading.Thread(target=self._ack_loop, daemon=True).start()

    def _ack_loop(self):
        try:
            from file_recv_lcmt import file_recv_lcmt
        except Exception as e:
            print("[lowgait] 无法导入 file_recv_lcmt:", e)
            return

        def on_ack(channel, data):
            try:
                m = file_recv_lcmt.decode(data)
                with self._lock:
                    self._ok["v"] = (m.result == 0)
            except Exception:
                with self._lock:
                    self._ok["v"] = False
            self._evt.set()

        self.lc_ack.subscribe(CH_GAIT_RESULT, on_ack)
        while self._running[0]:
            try:
                self.lc_ack.handle_timeout(100)
            except Exception:
                pass

    def _send_file(self, content, label, timeout=6.0):
        try:
            from file_send_lcmt import file_send_lcmt
        except Exception as e:
            print("[lowgait] 无法导入 file_send_lcmt:", e)
            return False
        self._evt.clear()
        with self._lock:
            self._ok["v"] = False
        msg = file_send_lcmt()
        msg.data = content
        self.lc.publish(CH_GAIT_FILE, msg.encode())
        if self._evt.wait(timeout=timeout):
            with self._lock:
                ok = self._ok["v"]
            print("[lowgait] %s ACK=%s" % (label, ok))
            return ok
        print("[lowgait] %s ACK 超时" % label)
        return False

    def _gen_full_params(self, vel_x=None):
        import copy
        import math
        try:
            import toml
        except ImportError:
            print("[lowgait] 缺少 toml 模块")
            return None
        template = {
            "mode": 0, "gait_id": 0, "contact": 0, "life_count": 0,
            "vel_des": [0.0, 0.0, 0.0], "rpy_des": [0.0, 0.0, 0.0],
            "pos_des": [0.0, 0.0, 0.0], "acc_des": [0.0] * 6,
            "ctrl_point": [0.0] * 3, "foot_pose": [0.0] * 6,
            "step_height": [0.0, 0.0], "value": 0, "duration": 0,
        }
        original = toml.load(self.params_path)
        steps = []
        for p in original.get("step", []):
            e = copy.deepcopy(template)
            e["duration"] = p.get("duration", 0)
            if p.get("type") == "usergait":
                e["mode"] = 11
                e["gait_id"] = 110
                e["vel_des"] = list(p.get("body_vel_des", [0.0, 0.0, 0.0]))
                if vel_x is not None:
                    e["vel_des"][0] = float(vel_x)
                bpd = p.get("body_pos_des", [0.0] * 6)
                e["rpy_des"] = bpd[0:3]
                e["pos_des"] = bpd[3:6]
                lpd = p.get("landing_pos_des", [0.0] * 11)
                e["foot_pose"][0:2] = lpd[0:2]
                e["foot_pose"][2:4] = lpd[3:5]
                e["foot_pose"][4:6] = lpd[6:8]
                e["ctrl_point"][0:2] = lpd[9:11]
                sh = p.get("step_height", [0.0] * 4)
                e["step_height"][0] = math.ceil(sh[0] * 1e3) + math.ceil(sh[1] * 1e3) * 1e3
                e["step_height"][1] = math.ceil(sh[2] * 1e3) + math.ceil(sh[3] * 1e3) * 1e3
                e["acc_des"] = p.get("weight", [0.0] * 6)
                e["value"] = p.get("use_mpc_traj", 0)
                e["contact"] = math.floor(p.get("landing_gain", 0.0) * 1e1)
                e["ctrl_point"][2] = p.get("mu", 0.0)
                steps.append(e)
        return "# Gait Params (Full - Generated)\n" + toml.dumps({"step": steps})

    def ensure_uploaded(self, vel_x=0.14):
        key = None if vel_x is None else round(float(vel_x), 4)
        if key == self._last_vel:
            return True
        ok1 = self._send_file(open(self.def_path, "r").read(), "Gait_Def_limit.toml")
        time.sleep(0.4)
        full = self._gen_full_params(vel_x=vel_x)
        if full is None:
            return False
        ok2 = self._send_file(full, "Full_Params(vel_x=%s)" % (vel_x,))
        time.sleep(0.3)
        if ok1 and ok2:
            self._last_vel = key
            return True
        return False

    def stop(self):
        self._running[0] = False
