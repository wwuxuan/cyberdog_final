#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第五赛段真机测试版：走到跳跃点后停车，不执行跳跃。

坐标采用初赛仿真第五赛段数据，但运行时以第五赛段起点为原点：
  fwd = 仿真世界 y - 第五赛段起点 y
  lat = 仿真世界 x - 第五赛段起点 x

第五赛段起点为第一段独木桥起点 (3.250, 7.200)，狗初始面朝仿真 +y
方向摆放。建立原点后，狗先逆时针转 180° 背对独木桥，再依次倒着走、
横移右、倒着右倾走、横移左、向前走，抵达初赛中触发左转/jump3D 的
位置后停车，便于先验证桥上路线和姿态。
"""
import argparse
import math
import os
import sys
import time


HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTER_PATH = os.path.join(HERE, "adapter.py")
if not os.path.isfile(ADAPTER_PATH):
    raise RuntimeError("找不到第五赛段适配器：%s" % ADAPTER_PATH)
sys.path = [path for path in sys.path if os.path.abspath(path or os.curdir) != HERE]
sys.path.insert(0, HERE)

from adapter import RealDogAdapter

for method_name in ("align_yaw", "walk_tilt", "lie"):
    if not hasattr(RealDogAdapter, method_name):
        raise RuntimeError(
            "第五赛段适配器缺少 %s()：%s" % (method_name, ADAPTER_PATH)
        )


# 第一段独木桥起点/第五赛段起点。
STAGE5_START_WORLD_X = 3.250
STAGE5_START_WORLD_Y = 7.200

# 初赛 stage5.py 的路线边界，全部转换到第五赛段局部坐标。
SEG1_LOCK_LAT = 3.250 - STAGE5_START_WORLD_X
SEG1_END_FWD = 12.280 - STAGE5_START_WORLD_Y
SEG2_FWD = 12.450 - STAGE5_START_WORLD_Y
SEG2_TARGET_LAT = -0.300 - STAGE5_START_WORLD_X
SEG3_END_FWD = 15.360 - STAGE5_START_WORLD_Y
SEG4_FWD = 15.369 - STAGE5_START_WORLD_Y
SEG4_TARGET_LAT = 3.030 - STAGE5_START_WORLD_X
JUMP_APPROACH_FWD = 13.650 - STAGE5_START_WORLD_Y
JUMP_APPROACH_LAT = 2.980 - STAGE5_START_WORLD_X


NORMAL_VX = 0.18
SIDE_MAX_VY = 0.16
BACK_VX = -0.16
TILT_VX = 0.05
FIRST_BRIDGE_STEP_HEIGHT = (0.11, 0.11)
MIN_TRANSLATION_V = 0.05
SIDE_FWD_CORRECT_MAX = 0.12
ROLL_TARGET = 0.40
FIFTH_ROLL_TARGET = -0.40
ROLL_RAMP_SEC = 2.0
ROLL_STEP = 0.05
TURN_CCW_TARGET_DEG = 179.0
BACKWARD_HEADING_DEG = 180.0
TURN_TOL_DEG = 2.0
TURN_TIMEOUT = 10.0
YAW_K = 0.03
YAW_MAX_WZ = 0.15
LAT_K = 0.50
FWD_K = 0.55
POSITION_TOL = 0.07
SEGMENT_TIMEOUT = 135.0
LOG_INTERVAL = 1.0


def _wrap_deg(angle):
    return ((angle + 180.0) % 360.0) - 180.0


def _clamp(value, low, high):
    return max(low, min(high, value))


class Stage5Real:
    def __init__(self, adapter, roll_target=ROLL_TARGET, use_tilt=True):
        self.adp = adapter
        self.roll_target = float(roll_target)
        self.use_tilt = bool(use_tilt)

    def _position(self):
        fwd, local_y, _z = self.adp.get_position()
        return float(fwd), float(-local_y)

    def _yaw(self):
        return float(self.adp.get_yaw_deg())

    def _print_position(self, label):
        fwd, lat = self._position()
        print("[stage5] %-12s fwd=%+.3f lat=%+.3f yaw=%+.1f" %
              (label, fwd, lat, self._yaw()), flush=True)

    def _send(self, vx, vy, wz, roll=0.0, step_height=None):
        if abs(roll) > 1e-6:
            self.adp.walk_tilt(vx, vy, wz, roll=roll)
        elif step_height is None:
            self.adp.walk(vx, vy, wz)
        else:
            self.adp.walk(vx, vy, wz, step_height=step_height)

    def _yaw_correction(self):
        yaw_err = _wrap_deg(BACKWARD_HEADING_DEG - self._yaw())
        return _clamp(YAW_K * yaw_err, -YAW_MAX_WZ, YAW_MAX_WZ)

    def _walk_fwd_to(self, target_fwd, target_lat, vx, label,
                     roll=0.0, step_height=None, timeout=SEGMENT_TIMEOUT):
        start = time.time()
        last_log = 0.0
        direction = 1.0 if vx >= 0.0 else -1.0
        while time.time() - start < timeout:
            fwd, lat = self._position()
            remaining = target_fwd - fwd
            if direction * remaining <= POSITION_TOL:
                self.adp.stop()
                self._print_position(label + " done")
                return

            speed = min(abs(vx), max(MIN_TRANSLATION_V, 0.75 * abs(remaining)))
            command_vx = -direction * speed
            lat_error = target_lat - lat
            command_vy = _clamp(LAT_K * lat_error, -SIDE_MAX_VY, SIDE_MAX_VY)
            command_wz = self._yaw_correction()
            self._send(
                command_vx, command_vy, command_wz, roll=roll,
                step_height=step_height
            )

            now = time.time()
            if now - last_log >= LOG_INTERVAL:
                self._print_position(label)
                last_log = now
            time.sleep(0.07)

        self.adp.stop()
        raise RuntimeError("[%s] 前进/倒退到目标超时" % label)

    def _walk_lat_to(self, target_fwd, target_lat, label, timeout=SEGMENT_TIMEOUT):
        start = time.time()
        last_log = 0.0
        while time.time() - start < timeout:
            fwd, lat = self._position()
            lat_error = target_lat - lat
            if abs(lat_error) <= POSITION_TOL:
                self.adp.stop()
                self._print_position(label + " done")
                return

            # 背桥横移时，机体 +vy 对应赛道右侧；因此误差同号。
            command_vy = _clamp(LAT_K * lat_error, -SIDE_MAX_VY, SIDE_MAX_VY)
            fwd_error = target_fwd - fwd
            command_vx = _clamp(-FWD_K * fwd_error,
                                -SIDE_FWD_CORRECT_MAX,
                                SIDE_FWD_CORRECT_MAX)
            self._send(command_vx, command_vy, self._yaw_correction())

            now = time.time()
            if now - last_log >= LOG_INTERVAL:
                self._print_position(label)
                last_log = now
            time.sleep(0.07)

        self.adp.stop()
        raise RuntimeError("[%s] 横移到目标超时" % label)

    def _ramp_roll(self, start_roll, end_roll):
        if not self.use_tilt:
            return
        distance = abs(end_roll - start_roll)
        steps = max(1, int(math.ceil(distance / ROLL_STEP)))
        hold = ROLL_RAMP_SEC / steps
        for index in range(steps + 1):
            ratio = index / float(steps)
            roll = start_roll + (end_roll - start_roll) * ratio
            self.adp.walk_tilt(0.0, 0.0, 0.0, roll=roll)
            time.sleep(hold)
        self.adp.stop()

    def run(self):
        self._print_position("起点")
        print("[stage5] 背桥倒行路线相对坐标：")
        print("  倒着走 -> fwd=%.3f lat=%.3f" % (SEG1_END_FWD, SEG1_LOCK_LAT))
        print("  横移右 -> fwd=%.3f lat=%.3f" % (SEG2_FWD, SEG2_TARGET_LAT))
        print("  倒着右倾走 -> fwd=%.3f lat=%.3f" % (SEG3_END_FWD, SEG2_TARGET_LAT))
        print("  横移左 -> fwd=%.3f lat=%.3f" % (SEG4_FWD, SEG4_TARGET_LAT))
        print("  向前走 -> fwd=%.3f lat=%.3f（跳点前停车）" %
              (JUMP_APPROACH_FWD, JUMP_APPROACH_LAT))

        self._walk_fwd_to(
            SEG1_END_FWD, SEG1_LOCK_LAT, NORMAL_VX, "第一段倒着走",
            step_height=FIRST_BRIDGE_STEP_HEIGHT
        )
        self._walk_lat_to(SEG2_FWD, SEG2_TARGET_LAT, "第二段横移右")

        if self.use_tilt:
            print("[stage5] 渐进施加 roll=%.3f rad" % self.roll_target)
            self._ramp_roll(0.0, self.roll_target)
            tilt_roll = self.roll_target
        else:
            print("[stage5] --no-tilt：跳过倾斜姿态")
            tilt_roll = 0.0

        self._walk_fwd_to(
            SEG3_END_FWD, SEG2_TARGET_LAT, TILT_VX, "第三段倒着右倾走", roll=tilt_roll
        )
        if self.use_tilt:
            print("[stage5] 倾斜段结束，渐进回正")
            self._ramp_roll(self.roll_target, 0.0)

        self._walk_lat_to(SEG4_FWD, SEG4_TARGET_LAT, "第四段横移左")
        if self.use_tilt:
            print("[stage5] 第五段前渐进施加左倾 roll=%.3f rad" % FIFTH_ROLL_TARGET)
            self._ramp_roll(0.0, FIFTH_ROLL_TARGET)
            fifth_roll = FIFTH_ROLL_TARGET
        else:
            fifth_roll = 0.0
        self._walk_fwd_to(
            JUMP_APPROACH_FWD, JUMP_APPROACH_LAT, BACK_VX, "第五段向前走",
            roll=fifth_roll
        )

        self.adp.stop()
        self._print_position("跳跃点")
        print("[stage5] 已到达初赛 jump3D 触发区域；按要求不跳跃，保持站立停车")


def main():
    parser = argparse.ArgumentParser(description="第五赛段真机路线测试（跳点停车）")
    parser.add_argument(
        "--no-tilt",
        action="store_true",
        help="调试用：第三段不用 roll=+0.4，只走普通步态",
    )
    parser.add_argument(
        "--roll",
        type=float,
        default=ROLL_TARGET,
        help="倾斜段 roll，默认 +0.4rad（机体右倾）",
    )
    args = parser.parse_args()

    adapter = RealDogAdapter(None)
    try:
        print("===== 第五赛段真机路线测试：跳点停车 =====")
        if not adapter.wait_odom(timeout=5.0):
            print("[stage5] 警告：未收到里程计，终止，避免无坐标行走")
            return 2

        adapter.stand()
        adapter.set_origin()
        print("[stage5] 当前姿态设为第五赛段原点，要求此时面朝桥的 +y 方向")
        print("[stage5] 原地逆时针旋转 180°，转身后背对独木桥")
        if not adapter.align_yaw(
                TURN_CCW_TARGET_DEG, tol=TURN_TOL_DEG, timeout=TURN_TIMEOUT
        ):
            raise RuntimeError("逆时针 180° 转身超时，终止路线")
        print("[stage5] 转身完成，保持背桥航向 %.1f°" % BACKWARD_HEADING_DEG)

        runner = Stage5Real(adapter, roll_target=args.roll, use_tilt=not args.no_tilt)
        runner.run()
        return 0
    except KeyboardInterrupt:
        print("\n[stage5] 中断，停止并保持站立")
        adapter.stop()
        return 130
    except Exception as exc:
        print("[stage5] 异常：%s" % exc)
        adapter.stop()
        return 1
    finally:
        adapter.stop()
        time.sleep(0.3)
        adapter.shutdown()
        print("[stage5] end")


if __name__ == "__main__":
    raise SystemExit(main())
