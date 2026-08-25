import math
import multiprocessing
import os
import sys
import threading
import time

import lcm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "motion", "utils"))
from localization_lcmt import localization_lcmt
from robot_control_cmd_lcmt import robot_control_cmd_lcmt
from robot_control_response_lcmt import robot_control_response_lcmt


def _clamp(value, low, high):
    return max(low, min(high, value))


def _wrap_deg(angle):
    return ((angle + 180.0) % 360.0) - 180.0


def _copy_command_fields(target, source):
    for field in robot_control_cmd_lcmt.__slots__:
        if field != "life_count":
            setattr(target, field, getattr(source, field))


def _heartbeat_process(command_pipe, send_count, late_count, last_send_t):
    """Publish the last complete command independently from ROS/CV work."""
    lcm_cmd = lcm.LCM("udpm://239.255.76.67:7671?ttl=255")
    command = robot_control_cmd_lcmt()
    life_count = 0
    period = 0.05
    next_send_t = time.monotonic()
    previous_send_t = 0.0

    while True:
        now = time.monotonic()
        wait_s = max(0.0, next_send_t - now)
        if command_pipe.poll(wait_s):
            payload = command_pipe.recv()
            if payload is None:
                break
            _copy_command_fields(command, robot_control_cmd_lcmt.decode(payload))
            while command_pipe.poll():
                payload = command_pipe.recv()
                if payload is None:
                    return
                _copy_command_fields(command, robot_control_cmd_lcmt.decode(payload))

        now = time.monotonic()
        if now < next_send_t:
            continue
        if previous_send_t and now - previous_send_t > 0.10:
            with late_count.get_lock():
                late_count.value += 1
        life_count = (life_count + 1) % 127
        command.life_count = life_count
        lcm_cmd.publish("robot_control_cmd", command.encode())
        previous_send_t = now
        with send_count.get_lock():
            send_count.value += 1
        with last_send_t.get_lock():
            last_send_t.value = now
        next_send_t += period
        if now - next_send_t > period:
            next_send_t = now + period


class RealDogAdapter:
    def __init__(self, node=None, namespace=""):
        self.node = node
        self.namespace = namespace

        self.lc_cmd = lcm.LCM("udpm://239.255.76.67:7671?ttl=255")
        self.cmd = robot_control_cmd_lcmt()
        self.cmd.mode = 11
        self.cmd.gait_id = 26
        self.cmd.contact = 15
        self.cmd.value = 0
        self.cmd.duration = 0
        self.cmd.vel_des = [0.0, 0.0, 0.0]
        self.cmd.rpy_des = [0.0, 0.0, 0.0]
        self.cmd.pos_des = [0.0, 0.0, 0.28]
        self.cmd.acc_des = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.cmd.ctrl_point = [0.0, 0.0, 0.0]
        self.cmd.foot_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.cmd.step_height = [0.09, 0.09]
        self.cmd.life_count = 0
        self._cmd_lock = threading.Lock()
        self._running = True
        context = multiprocessing.get_context("spawn")
        heartbeat_receive, self._heartbeat_send = context.Pipe(duplex=False)
        self._heartbeat_send_count = context.Value("L", 0)
        self._heartbeat_late_count = context.Value("L", 0)
        self._heartbeat_last_send_t = context.Value("d", 0.0)
        self._heartbeat_process = context.Process(
            target=_heartbeat_process,
            args=(
                heartbeat_receive,
                self._heartbeat_send_count,
                self._heartbeat_late_count,
                self._heartbeat_last_send_t,
            ),
            name="cmd-heartbeat",
            daemon=True,
        )
        self._heartbeat_process.start()
        heartbeat_receive.close()

        self.lc_response = lcm.LCM("udpm://239.255.76.67:7670?ttl=255")
        self._response_lock = threading.Lock()
        self._response_count = 0
        self._last_response_t = 0.0
        self._response = None
        self.lc_response.subscribe("robot_control_response", self._response_handler)

        self.lc_odom = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
        self._odom_lock = threading.Lock()
        self._raw_x = 0.0
        self._raw_y = 0.0
        self._raw_z = 0.0
        self._raw_yaw_deg = 0.0
        self._origin_x = 0.0
        self._origin_y = 0.0
        self._origin_yaw = 0.0
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_z = 0.0
        self._odom_yaw_deg = 0.0
        self._odom_set = False
        self.lc_odom.subscribe("global_to_robot", self._odom_handler)

        self._odom_thread = threading.Thread(target=self._odom_loop, name="odom-receive", daemon=True)
        self._response_thread = threading.Thread(target=self._response_loop, name="response-receive", daemon=True)
        self._odom_thread.start()
        self._response_thread.start()
        self._sync_heartbeat_command()

    def _sync_heartbeat_command(self):
        with self._cmd_lock:
            payload = self.cmd.encode()
        try:
            self._heartbeat_send.send(payload)
        except (BrokenPipeError, EOFError, OSError) as exc:
            print(f"[adapter] ERROR heartbeat_command={exc!r}")

    def heartbeat_status(self):
        with self._heartbeat_send_count.get_lock():
            send_count = self._heartbeat_send_count.value
        with self._heartbeat_late_count.get_lock():
            late_count = self._heartbeat_late_count.value
        with self._heartbeat_last_send_t.get_lock():
            last_send_t = self._heartbeat_last_send_t.value
        with self._response_lock:
            response = self._response
            response_age = (
                time.monotonic() - self._last_response_t
                if self._last_response_t else None
            )
            response_count = self._response_count
        status = {
            "count": send_count,
            "late": late_count,
            "age": time.monotonic() - last_send_t if last_send_t else None,
            "process_alive": self._heartbeat_process.is_alive(),
            "response_count": response_count,
            "response_age": response_age,
        }
        if response is not None:
            status.update({
                "response_mode": int(response.mode),
                "response_gait_id": int(response.gait_id),
                "response_contact": int(response.contact),
                "response_switch_status": int(response.switch_status),
                "response_ori_error": int(response.ori_error),
                "response_footpos_error": int(response.footpos_error),
            })
        return status

    def _response_handler(self, _channel, data):
        try:
            response = robot_control_response_lcmt.decode(data)
            with self._response_lock:
                self._response = response
                self._response_count += 1
                self._last_response_t = time.monotonic()
        except Exception as exc:
            print(f"[adapter] WARN response_decode={exc!r}")

    def _response_loop(self):
        while self._running:
            try:
                self.lc_response.handle_timeout(100)
            except Exception as exc:
                if self._running:
                    print(f"[adapter] WARN response_receive={exc!r}")

    def _odom_handler(self, channel, data):
        try:
            msg = localization_lcmt.decode(data)
            with self._odom_lock:
                self._raw_x = float(msg.xyz[0])
                self._raw_y = float(msg.xyz[1])
                self._raw_z = float(msg.xyz[2])
                self._raw_yaw_deg = math.degrees(float(msg.rpy[2]))
                dx = self._raw_x - self._origin_x
                dy = self._raw_y - self._origin_y
                origin_yaw_rad = math.radians(self._origin_yaw)
                self._odom_x = dx * math.cos(origin_yaw_rad) + dy * math.sin(origin_yaw_rad)
                self._odom_y = -dx * math.sin(origin_yaw_rad) + dy * math.cos(origin_yaw_rad)
                self._odom_z = self._raw_z
                self._odom_yaw_deg = _wrap_deg(self._raw_yaw_deg - self._origin_yaw)
                self._odom_set = True
        except Exception:
            pass

    def _odom_loop(self):
        while self._running:
            try:
                self.lc_odom.handle_timeout(100)
            except Exception:
                pass

    def wait_odom(self, timeout=3.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.odom_ready():
                return True
            time.sleep(0.05)
        return False

    def odom_ready(self):
        with self._odom_lock:
            return self._odom_set

    def stand(self):
        print("[adapter] stand")
        with self._cmd_lock:
            self.cmd.mode = 12
            self.cmd.gait_id = 0
            self.cmd.contact = 15
            self.cmd.value = 0
            self.cmd.duration = 0
            self.cmd.vel_des = [0.0, 0.0, 0.0]
        self._sync_heartbeat_command()
        time.sleep(5.0)
        with self._cmd_lock:
            self.cmd.mode = 11
            self.cmd.gait_id = 26
            self.cmd.contact = 15
            self.cmd.value = 0
            self.cmd.duration = 0
        self._sync_heartbeat_command()
        print("[adapter] stand done")

    def stop(self):
        with self._cmd_lock:
            self.cmd.vel_des = [0.0, 0.0, 0.0]
        self._sync_heartbeat_command()

    def walk(self, vx, vy=0.0, wz=0.0, roll=0.0, gait_id=26,
             step_height=(0.09, 0.09)):
        with self._cmd_lock:
            self.cmd.mode = 11
            self.cmd.gait_id = int(gait_id)
            self.cmd.contact = 15
            self.cmd.value = 0
            self.cmd.duration = 0
            self.cmd.rpy_des = [float(roll), 0.0, 0.0]
            self.cmd.pos_des = [0.0, 0.0, 0.28]
            self.cmd.acc_des = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.cmd.ctrl_point = [0.0, 0.0, 0.0]
            self.cmd.foot_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.cmd.step_height = [float(step_height[0]), float(step_height[1])]
            self.cmd.vel_des = [float(vx), float(vy), float(wz)]
        self._sync_heartbeat_command()

    def walk_tilt(self, vx, vy=0.0, wz=0.0, roll=-0.4,
                  step_height=(0.06, 0.06)):
        self.walk(vx, vy, wz, roll=roll, gait_id=27)
        with self._cmd_lock:
            self.cmd.step_height = [float(step_height[0]), float(step_height[1])]
        self._sync_heartbeat_command()

    def set_jump(self, gait_id=1, velocity=1.5, duration=1000,
                 step_height=(0.12, 0.12)):
        """持续发送 jump3D 命令，调用方负责在动作完成后切回站立。"""
        with self._cmd_lock:
            self.cmd.mode = 16
            self.cmd.gait_id = int(gait_id)
            self.cmd.contact = 15
            self.cmd.value = 0
            self.cmd.duration = int(duration)
            self.cmd.vel_des = [float(velocity), 0.0, 0.0]
            self.cmd.rpy_des = [0.0, 0.0, 0.0]
            self.cmd.pos_des = [0.0, 0.0, 0.0]
            self.cmd.acc_des = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.cmd.ctrl_point = [0.0, 0.0, 0.0]
            self.cmd.foot_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.cmd.step_height = [float(step_height[0]), float(step_height[1])]
        self._sync_heartbeat_command()

    def lie(self):
        with self._cmd_lock:
            self.cmd.mode = 7
            self.cmd.gait_id = 0
            self.cmd.contact = 15
            self.cmd.value = 0
            self.cmd.duration = 0
            self.cmd.vel_des = [0.0, 0.0, 0.0]
            self.cmd.rpy_des = [0.0, 0.0, 0.0]
        self._sync_heartbeat_command()

    def align_yaw(self, target_deg, tol=2.0, timeout=6.0):
        target_deg = _wrap_deg(target_deg)
        t0 = time.time()
        while time.time() - t0 < timeout:
            yaw = self.get_yaw_deg()
            err = _wrap_deg(target_deg - yaw)
            if abs(err) <= tol:
                self.stop()
                return True
            wz = _clamp(0.035 * err, -0.45, 0.45)
            if 0.0 < abs(wz) < 0.12:
                wz = 0.12 if wz > 0.0 else -0.12
            self.walk(0.0, 0.0, wz)
            time.sleep(0.07)
        self.stop()
        return False

    def turn_timed(self, angle_deg):
        return self.align_yaw(self.get_yaw_deg() + angle_deg)

    def walk_to(self, target_x, target_y, target_yaw=None, tol=0.07,
                timeout=20.0, max_v=0.25, max_wz=0.35):
        t0 = time.time()
        while time.time() - t0 < timeout:
            x, y, _ = self.get_position()
            yaw_deg = self.get_yaw_deg()
            dx = float(target_x) - x
            dy = float(target_y) - y
            dist = math.hypot(dx, dy)
            if dist <= tol:
                self.stop()
                if target_yaw is not None:
                    self.align_yaw(target_yaw)
                return True

            speed = _clamp(0.75 * dist, 0.05, max_v)
            world_vx = speed * dx / max(dist, 1e-6)
            world_vy = speed * dy / max(dist, 1e-6)
            yaw_rad = math.radians(yaw_deg)
            body_vx = world_vx * math.cos(yaw_rad) + world_vy * math.sin(yaw_rad)
            body_vy = -world_vx * math.sin(yaw_rad) + world_vy * math.cos(yaw_rad)

            if target_yaw is None:
                target_heading = math.degrees(math.atan2(dy, dx))
                yaw_err = _wrap_deg(target_heading - yaw_deg)
                wz = _clamp(0.02 * yaw_err, -max_wz, max_wz)
            else:
                yaw_err = _wrap_deg(float(target_yaw) - yaw_deg)
                wz = _clamp(0.03 * yaw_err, -max_wz, max_wz)

            self.walk(body_vx, body_vy, wz)
            time.sleep(0.07)

        self.stop()
        return False

    def align_x(self, target_x):
        _, y, _ = self.get_position()
        return self.walk_to(target_x, y, self.get_yaw_deg(), tol=0.03,
                            timeout=10.0, max_v=0.10, max_wz=0.20)

    def align_y(self, target_y):
        x, _, _ = self.get_position()
        return self.walk_to(x, target_y, self.get_yaw_deg(), tol=0.03,
                            timeout=10.0, max_v=0.10, max_wz=0.20)

    def navigate_to(self, x, y, yaw):
        ok = self.walk_to(x, y, yaw, tol=0.07, timeout=25.0, max_v=0.25)
        self.align_x(x)
        self.align_y(y)
        self.align_yaw(yaw)
        return ok

    def get_position(self):
        with self._odom_lock:
            return (self._odom_x, self._odom_y, self._odom_z)

    def get_yaw_deg(self):
        with self._odom_lock:
            return self._odom_yaw_deg

    def set_origin(self):
        self.wait_odom(timeout=2.0)
        with self._odom_lock:
            self._origin_x = self._raw_x
            self._origin_y = self._raw_y
            self._origin_yaw = self._raw_yaw_deg
            self._odom_x = 0.0
            self._odom_y = 0.0
            self._odom_yaw_deg = 0.0
        print(f"[adapter] origin set raw=({self._origin_x:.3f},{self._origin_y:.3f}) yaw={self._origin_yaw:.1f}")

    def set_mapped_pose(self, x, y, yaw_deg, quiet=False):
        """Map the current raw pose to a known field pose without moving."""
        self.wait_odom(timeout=2.0)
        with self._odom_lock:
            origin_yaw = _wrap_deg(self._raw_yaw_deg - float(yaw_deg))
            origin_yaw_rad = math.radians(origin_yaw)
            cosine = math.cos(origin_yaw_rad)
            sine = math.sin(origin_yaw_rad)
            self._origin_x = self._raw_x - (cosine * float(x) - sine * float(y))
            self._origin_y = self._raw_y - (sine * float(x) + cosine * float(y))
            self._origin_yaw = origin_yaw
            self._odom_x = float(x)
            self._odom_y = float(y)
            self._odom_z = self._raw_z
            self._odom_yaw_deg = float(yaw_deg)
            self._odom_set = True
        if not quiet:
            print(
                f"[adapter] mapped pose set pos=({float(x):.3f},{float(y):.3f}) "
                f"yaw={float(yaw_deg):.1f} raw=({self._raw_x:.3f},{self._raw_y:.3f})"
            )

    def shutdown(self):
        self.stop()
        time.sleep(0.20)
        self._running = False
        try:
            self._heartbeat_send.send(None)
        except (BrokenPipeError, EOFError, OSError):
            pass
        if self._heartbeat_process.is_alive():
            self._heartbeat_process.join(timeout=1.0)
        if self._heartbeat_process.is_alive():
            self._heartbeat_process.terminate()
            self._heartbeat_process.join(timeout=1.0)
        for thread in (
            getattr(self, "_odom_thread", None),
            getattr(self, "_response_thread", None),
        ):
            if thread is not None and thread.is_alive():
                thread.join(timeout=0.5)
