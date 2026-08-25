#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第五赛段跳下版本，复用 stage5_real.py 的真机路线控制。"""
import argparse
import os
import sys
import time


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import stage5_real as route
from adapter import RealDogAdapter


for method_name in ("align_yaw", "set_jump", "stand"):
    if not hasattr(RealDogAdapter, method_name):
        raise RuntimeError("第五赛段适配器缺少 %s()" % method_name)


# 保持 stage5_real.py 的起点定义，并按本次真机路线覆盖各段终点。
route.SEG1_END_FWD = 12.250 - route.STAGE5_START_WORLD_Y
route.SEG2_FWD = 12.250 - route.STAGE5_START_WORLD_Y
route.SEG3_END_FWD = 15.320 - route.STAGE5_START_WORLD_Y
route.SEG4_FWD = 15.320 - route.STAGE5_START_WORLD_Y
route.SEG4_TARGET_LAT = 3.300 - route.STAGE5_START_WORLD_X
route.JUMP_APPROACH_FWD = 13.250 - route.STAGE5_START_WORLD_Y
route.JUMP_APPROACH_LAT = 3.300 - route.STAGE5_START_WORLD_X

JUMP_FACING_DEG = 90.0
JUMP_GAIT_ID = 13
JUMP_VELOCITY = 1.5
JUMP_DURATION_MS = 1000
JUMP_STEP_HEIGHT = (0.12, 0.12)
JUMP_HOLD_SEC = 3.5

START_MARKERS = {
    "stage5": (0.0, 0.0),
    "turn1": (route.SEG1_END_FWD, route.SEG1_LOCK_LAT),
    "turn2": (route.SEG2_FWD, route.SEG2_TARGET_LAT),
    "turn3": (route.SEG3_END_FWD, route.SEG2_TARGET_LAT),
    "turn4": (route.SEG4_FWD, route.SEG4_TARGET_LAT),
}


class Stage5RealJump(route.Stage5Real):
    def run(self, start_at="stage5"):
        if start_at not in START_MARKERS:
            raise ValueError("unknown stage5 start marker: %s" % start_at)
        self._print_position("起点")
        print("[stage5] 背桥倒行路线相对坐标：")
        print("  倒着走 -> fwd=%.3f lat=%.3f" % (
            route.SEG1_END_FWD, route.SEG1_LOCK_LAT
        ))
        print("  横移右 -> fwd=%.3f lat=%.3f" % (
            route.SEG2_FWD, route.SEG2_TARGET_LAT
        ))
        print("  倒着右倾走 -> fwd=%.3f lat=%.3f" % (
            route.SEG3_END_FWD, route.SEG2_TARGET_LAT
        ))
        print("  横移左 -> fwd=%.3f lat=%.3f" % (
            route.SEG4_FWD, route.SEG4_TARGET_LAT
        ))
        print("  向前走 -> fwd=%.3f lat=%.3f（跳下点）" % (
            route.JUMP_APPROACH_FWD, route.JUMP_APPROACH_LAT
        ))

        if start_at == "stage5":
            self._walk_fwd_to(
                route.SEG1_END_FWD,
                route.SEG1_LOCK_LAT,
                route.NORMAL_VX,
                "第一段倒着走",
                step_height=route.FIRST_BRIDGE_STEP_HEIGHT,
            )

        if start_at in ("stage5", "turn1"):
            self._walk_lat_to(route.SEG2_FWD, route.SEG2_TARGET_LAT, "第二段横移右")

        if start_at in ("stage5", "turn1", "turn2"):
            if self.use_tilt:
                print("[stage5] 渐进施加 roll=%.3f rad" % self.roll_target)
                self._ramp_roll(0.0, self.roll_target)
                tilt_roll = self.roll_target
            else:
                print("[stage5] --no-tilt：跳过倾斜姿态")
                tilt_roll = 0.0

            self._walk_fwd_to(
                route.SEG3_END_FWD,
                route.SEG2_TARGET_LAT,
                route.TILT_VX,
                "第三段倒着右倾走",
                roll=tilt_roll,
            )
            if self.use_tilt:
                print("[stage5] 倾斜段结束，渐进回正")
                self._ramp_roll(self.roll_target, 0.0)

        if start_at in ("stage5", "turn1", "turn2", "turn3"):
            self._walk_lat_to(route.SEG4_FWD, route.SEG4_TARGET_LAT, "第四段横移左")

        if self.use_tilt:
            print("[stage5] 第五段前渐进施加左倾 roll=%.3f rad" % route.FIFTH_ROLL_TARGET)
            self._ramp_roll(0.0, route.FIFTH_ROLL_TARGET)
            fifth_roll = route.FIFTH_ROLL_TARGET
        else:
            fifth_roll = 0.0

        self._walk_fwd_to(
            route.JUMP_APPROACH_FWD,
            route.JUMP_APPROACH_LAT,
            route.BACK_VX,
            "第五段向前走",
            roll=fifth_roll,
        )

        self.adp.stop()
        if self.use_tilt:
            print("[stage5] 跳下前渐进回正")
            self._ramp_roll(fifth_roll, 0.0)
        self._print_position("跳下点")
        print("[stage5] 原地右转 90°，面向赛道 -x（相对 yaw=+90°）")
        if not self.adp.align_yaw(
                JUMP_FACING_DEG,
                tol=route.TURN_TOL_DEG,
                timeout=route.TURN_TIMEOUT,
        ):
            raise RuntimeError("跳下前右转 90° 超时")

        print("[stage5] 执行最远跳下：gait=%d velocity=%.1f" % (
            JUMP_GAIT_ID, JUMP_VELOCITY
        ))
        self.adp.set_jump(
            gait_id=JUMP_GAIT_ID,
            velocity=JUMP_VELOCITY,
            duration=JUMP_DURATION_MS,
            step_height=JUMP_STEP_HEIGHT,
        )
        time.sleep(JUMP_HOLD_SEC)
        self.adp.stop()
        print("[stage5] 跳下命令结束，切回站立")
        self.adp.stand()


def main():
    parser = argparse.ArgumentParser(description="第五赛段真机路线（跳下版本）")
    parser.add_argument(
        "--start-at",
        choices=tuple(START_MARKERS),
        default="stage5",
        help="stage5: 起点；turn1-turn4: 对应转向点后立即执行下一段",
    )
    parser.add_argument("--no-tilt", action="store_true")
    parser.add_argument("--roll", type=float, default=route.ROLL_TARGET)
    args = parser.parse_args()

    adapter = RealDogAdapter(None)
    try:
        print("===== 第五赛段真机路线：跳下版本 =====")
        if not adapter.wait_odom(timeout=5.0):
            print("[stage5] 警告：未收到里程计，终止，避免无坐标行走")
            return 2

        adapter.stand()
        if args.start_at == "stage5":
            adapter.set_origin()
            print("[stage5] 当前姿态设为第五赛段原点，要求此时面朝桥的 +y 方向")
            print("[stage5] 原地逆时针旋转 180°，转身后背对独木桥")
            if not adapter.align_yaw(
                    route.TURN_CCW_TARGET_DEG,
                    tol=route.TURN_TOL_DEG,
                    timeout=route.TURN_TIMEOUT,
            ):
                raise RuntimeError("逆时针 180° 转身超时，终止路线")
        else:
            start_fwd, start_lat = START_MARKERS[args.start_at]
            adapter.set_mapped_pose(start_fwd, -start_lat, route.BACKWARD_HEADING_DEG)
            world_x = route.STAGE5_START_WORLD_X + start_lat
            world_y = route.STAGE5_START_WORLD_Y + start_fwd
            print(
                "[stage5] 从 %s 继续：当前点映射为 world=(%.3f, %.3f)，"
                "直接执行转向后的下一段" % (args.start_at, world_x, world_y)
            )

        runner = Stage5RealJump(
            adapter,
            roll_target=args.roll,
            use_tilt=not args.no_tilt,
        )
        runner.run(start_at=args.start_at)
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
