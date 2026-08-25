#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第六赛段固定路线：低姿倒退，用屁股将足球推向缺口。"""
import argparse
import math
import os
import time

import s6_base as base

try:
    from low_gait import LowGait
except ImportError:
    from stage4_package.dog.stage4_lowgait import LowGait


PUSH_GAP_WORLD_X = 3.000
PUSH_GAP_WORLD_Y = 12.600
PUSH_GAP_FWD = PUSH_GAP_WORLD_Y - base._START_Y
PUSH_GAP_LAT = PUSH_GAP_WORLD_X - base._START_X

END_WORLD_X = 3.250
END_WORLD_Y = 12.750
END_FWD = END_WORLD_Y - base._START_Y
END_LAT = END_WORLD_X - base._START_X

FACE_WORLD_MINUS_X_YAW = 90.0
BACK_PUSH_LOW_VX = 0.07
BACK_PUSH_TOL = 0.06
BACK_PUSH_TURN_WZ = 0.25
END_TURN_WZ = 0.16
END_REVERSE_MAX_V = 0.12
END_REVERSE_MAX_VY = 0.06


def _clamp(value, low, high):
    return max(low, min(high, value))


class Stage6BackPush(base.Stage6Real):
    def __init__(self, adapter, low):
        super(Stage6BackPush, self).__init__(adapter, det=None, depth_tracker=None)
        self.low = low
        self.low_active = False
        self._back_push_yaw = None

    def _push_yaw(self, ball_fwd=base.BALL_INITIAL_FWD,
                  ball_lat=base.BALL_INITIAL_LAT):
        return math.degrees(math.atan2(
            -(PUSH_GAP_LAT - ball_lat),
            PUSH_GAP_FWD - ball_fwd,
        ))

    def _behind_point(self, ball_fwd, ball_lat):
        delta_fwd = ball_fwd - PUSH_GAP_FWD
        delta_lat = ball_lat - PUSH_GAP_LAT
        norm = math.hypot(delta_fwd, delta_lat)
        if norm < 1e-3:
            raise RuntimeError("[stage6_backpush] invalid ball-to-gap geometry")
        unit_fwd = delta_fwd / norm
        unit_lat = delta_lat / norm
        distance = base.BALL_BEHIND_DIST
        low_fwd = base.FIELD_FWD_MIN + 0.1
        high_fwd = base.FIELD_FWD_MAX - 0.1
        low_lat = base.FIELD_LAT_MIN + 0.1
        high_lat = base.FIELD_LAT_MAX - 0.1
        for coordinate, direction, lower, upper in (
                (ball_fwd, unit_fwd, low_fwd, high_fwd),
                (ball_lat, unit_lat, low_lat, high_lat)):
            if direction > 1e-6:
                distance = min(distance, max(0.0, (upper - coordinate) / direction))
            elif direction < -1e-6:
                distance = min(distance, max(0.0, (lower - coordinate) / direction))
        return ball_fwd + unit_fwd * distance, ball_lat + unit_lat * distance

    def _face_yaw(self, target_yaw, tol=2.0):
        target_yaw = base._wrap_deg(target_yaw)
        while True:
            yaw = self._yaw()
            error = base._wrap_deg(target_yaw - yaw)
            if abs(error) <= tol:
                self._stop()
                return True
            wz = _clamp(0.035 * error, -0.45, 0.45)
            if 0.0 < abs(wz) < 0.12:
                wz = 0.12 if wz > 0.0 else -0.12
            self.adp.walk(0.0, 0.0, wz)
            time.sleep(0.07)

    def _walk_fixed_to(self, target_fwd, target_lat, max_v, label, tol=0.06):
        print("[stage6_backpush] fixed %s target=(fwd=%.2f, lat=%+.2f)" % (
            label, target_fwd, target_lat))
        last_log_t = 0.0
        while True:
            fwd, adapter_y, _z = self._pos()
            lat = -adapter_y
            df = target_fwd - fwd
            dlat = target_lat - lat
            distance = math.hypot(df, dlat)
            if distance <= tol:
                self._stop()
                print("[stage6_backpush] fixed %s reached fwd=%.2f lat=%+.2f" % (
                    label, fwd, lat))
                return True
            yaw_rad = math.radians(self._yaw())
            target_adapter_y = -target_lat
            dy = target_adapter_y - adapter_y
            bvx = df * math.cos(yaw_rad) + dy * math.sin(yaw_rad)
            bvy = -df * math.sin(yaw_rad) + dy * math.cos(yaw_rad)
            speed = min(max_v, max(0.05, 0.8 * distance))
            vx = bvx / max(distance, 0.01) * speed
            vy = bvy / max(distance, 0.01) * speed
            self.adp.walk(vx, vy, 0.0)
            now = time.monotonic()
            if now - last_log_t >= 0.5:
                print("[stage6_backpush] fixed %s fwd=%.2f lat=%+.2f err=(%+.2f,%+.2f) cmd=(%.2f,%+.2f)" % (
                    label, fwd, lat, df, dlat, vx, vy))
                last_log_t = now
            time.sleep(0.05)

    def _turn_counterclockwise_to(self, target_yaw, max_wz, label, tol=2.0):
        target_yaw = base._wrap_deg(target_yaw)
        print("[stage6_backpush] %s: counterclockwise turn to yaw=%.1f" % (
            label, target_yaw))
        while True:
            yaw = self._yaw()
            remaining = (target_yaw - yaw) % 360.0
            if remaining <= tol:
                self._stop()
                print("[stage6_backpush] %s done yaw=%.1f" % (label, yaw))
                return True
            wz = _clamp(0.025 * remaining, 0.10, max_wz)
            self.adp.walk(0.0, 0.0, wz)
            time.sleep(0.05)

    def _low_on(self, velocity_x):
        with self.adp._cmd_lock:
            self.adp.cmd.mode = 12
            self.adp.cmd.gait_id = 0
            self.adp.cmd.contact = 15
            self.adp.cmd.value = 0
            self.adp.cmd.duration = 0
            self.adp.cmd.vel_des = [0.0, 0.0, 0.0]
            self.adp.cmd.pos_des = [0.0, 0.0, 0.28]
            self.adp.cmd.step_height = [0.09, 0.09]
        self.adp._sync_heartbeat_command()
        print("[stage6_backpush] low gait prepare")
        time.sleep(1.5)
        if not self.low.ensure_uploaded(vel_x=velocity_x):
            raise RuntimeError("[stage6_backpush] low gait upload failed")
        with self.adp._cmd_lock:
            self.adp.cmd.mode = 62
            self.adp.cmd.gait_id = 110
            self.adp.cmd.vel_des = [0.0, 0.0, 0.0]
            self.adp.cmd.pos_des = [0.0, 0.0, 0.0]
            self.adp.cmd.step_height = [0.0, 0.0]
            self.adp.cmd.contact = 15
            self.adp.cmd.value = 0
            self.adp.cmd.duration = 9600
        self.adp._sync_heartbeat_command()
        self.low_active = True
        print("[stage6_backpush] low gait active vel_x=%.2f" % velocity_x)

    def _low_off(self):
        if self.low_active:
            self.adp.walk(0.0, 0.0, 0.0)
            self.low_active = False
            print("[stage6_backpush] low gait off")

    def _recover_stand(self):
        self.adp.stand()
        time.sleep(0.5)

    def run_preparation_route(self):
        print("[stage6_backpush] fixed prep to left corner, then ball-behind point; no RGB/D430/fish-eye")
        self._face_yaw(base.PREP_FACE_FWD_YAW)
        lateral_ok = self._walk_fixed_to(
            self._fwd(),
            base.FIXED_CORNER_LAT,
            max_v=base.PREP_CRAB_MAX_V,
            label="left-corner lateral",
        )
        self._face_yaw(base.PREP_FACE_FWD_YAW)
        corner_ok = self._walk_fixed_to(
            base.FIXED_CORNER_FWD,
            base.FIXED_CORNER_LAT,
            max_v=base.PREP_FORWARD_MAX_V,
            label="left-corner forward",
        )
        behind_fwd, behind_lat = self._behind_point(
            base.BALL_INITIAL_FWD,
            base.BALL_INITIAL_LAT,
        )
        behind_ok = self._walk_fixed_to(
            behind_fwd,
            behind_lat,
            max_v=base.PREP_FORWARD_MAX_V,
            label="ball-behind point",
        )
        push_yaw = self._push_yaw()
        self._back_push_yaw = base._wrap_deg(push_yaw + 180.0)
        print("[stage6_backpush] ball->gap yaw=%.1f; rear faces gap at yaw=%.1f" % (
            push_yaw, self._back_push_yaw))
        turn_ok = self._turn_counterclockwise_to(
            self._back_push_yaw,
            BACK_PUSH_TURN_WZ,
            label="rear-align",
        )
        self._stop()
        print("[stage6_backpush] prep complete pos=(world x=%.2f, y=%.2f) yaw=%.1f" % (
            base._START_X + self._lat(), base._START_Y + self._fwd(), self._yaw()))
        return lateral_ok and corner_ok and behind_ok and turn_ok

    def push_backwards_low_to_gap(self):
        if self._back_push_yaw is None:
            raise RuntimeError("[stage6_backpush] rear push yaw is unset")
        heading_rad = math.radians(self._back_push_yaw)
        last_log_t = 0.0
        self._low_on(-BACK_PUSH_LOW_VX)
        try:
            while True:
                delta_fwd = PUSH_GAP_FWD - self._fwd()
                delta_lat = PUSH_GAP_LAT - self._lat()
                forward_error = (
                    delta_fwd * math.cos(heading_rad)
                    - delta_lat * math.sin(heading_rad)
                )
                backward_remaining = -forward_error
                cross = (
                    delta_fwd * math.sin(heading_rad)
                    + delta_lat * math.cos(heading_rad)
                )
                if backward_remaining <= BACK_PUSH_TOL:
                    print("[stage6_backpush] rear push reached gap world=(%.2f, %.2f) cross=%+.2f" % (
                        base._START_X + self._lat(), base._START_Y + self._fwd(), cross))
                    return True
                now = time.monotonic()
                if now - last_log_t >= 0.5:
                    print("[stage6_backpush] rear push remaining=%.2f cross=%+.2f low_vx=-%.2f yaw=%.1f" % (
                        backward_remaining, cross, BACK_PUSH_LOW_VX, self._yaw()))
                    last_log_t = now
                time.sleep(0.05)
        finally:
            self._low_off()
            self._recover_stand()

    def _reverse_slowly_to_end(self):
        if not self._turn_counterclockwise_to(
                FACE_WORLD_MINUS_X_YAW,
                END_TURN_WZ,
                label="end-face-minus-x"):
            return False
        last_log_t = 0.0
        while True:
            delta_fwd = END_FWD - self._fwd()
            delta_lat = END_LAT - self._lat()
            distance = math.hypot(delta_fwd, delta_lat)
            if distance <= BACK_PUSH_TOL:
                self._stop()
                print("[stage6_backpush] end reached world=(%.2f, %.2f)" % (
                    base._START_X + self._lat(), base._START_Y + self._fwd()))
                return True
            yaw_rad = math.radians(self._yaw())
            desired_vx = (
                delta_fwd * math.cos(yaw_rad)
                - delta_lat * math.sin(yaw_rad)
            )
            desired_vy = (
                -delta_fwd * math.sin(yaw_rad)
                - delta_lat * math.cos(yaw_rad)
            )
            if desired_vx >= -0.01:
                self._stop()
                print("[stage6_backpush] end is no longer behind body; stop at world=(%.2f, %.2f)" % (
                    base._START_X + self._lat(), base._START_Y + self._fwd()))
                return False
            speed = min(END_REVERSE_MAX_V, max(0.05, 0.8 * distance))
            vx = max(-speed, desired_vx / max(distance, 0.01) * speed)
            vy = _clamp(desired_vy / max(distance, 0.01) * speed,
                        -END_REVERSE_MAX_VY, END_REVERSE_MAX_VY)
            yaw_error = base._wrap_deg(FACE_WORLD_MINUS_X_YAW - self._yaw())
            wz = _clamp(0.025 * yaw_error, -0.12, 0.12)
            self._walk(vx, vy, wz)
            now = time.monotonic()
            if now - last_log_t >= 0.5:
                print("[stage6_backpush] end reverse err=(%+.2f,%+.2f) cmd=(%.2f,%+.2f) yaw=%.1f" % (
                    delta_fwd, delta_lat, vx, vy, self._yaw()))
                last_log_t = now
            time.sleep(0.05)

    def run(self):
        print("[stage6_backpush] ==== stage 6 rear low-gait push ====")
        if not self.run_preparation_route():
            print("[stage6_backpush] prep incomplete; do not start rear push")
            return False
        if not self.push_backwards_low_to_gap():
            print("[stage6_backpush] rear push incomplete; do not enter endpoint")
            return False
        if not self._reverse_slowly_to_end():
            print("[stage6_backpush] endpoint move incomplete; no lie-down")
            return False
        self.adp.lie_down()
        time.sleep(1.0)
        print("[stage6_backpush] complete")
        return True


def main():
    parser = argparse.ArgumentParser(description="stage6 rear low-gait football push")
    parser.add_argument("--skip-odom-wait", action="store_true")
    parser.add_argument("--gait-dir", default=os.path.join(base.BASE_DIR, "gait"))
    args = parser.parse_args()

    adapter = base.RealDogAdapter(None)
    low = LowGait(gait_dir=args.gait_dir)
    runner = Stage6BackPush(adapter, low)
    try:
        if not args.skip_odom_wait and not adapter.wait_odom(timeout=5.0):
            print("[stage6_backpush] warning: odometry not ready")
        adapter.stand()
        adapter.set_mapped_pose(base.START_FWD, base.START_ADAPTER_Y,
                                base.START_WORLD_YAW)
        print("[stage6_backpush] start world=(%.3f, %.3f) yaw=%.1f; gap=(%.2f, %.2f); end=(%.2f, %.2f)" % (
            base.START_WORLD_X, base.START_WORLD_Y, base.START_WORLD_YAW,
            PUSH_GAP_WORLD_X, PUSH_GAP_WORLD_Y, END_WORLD_X, END_WORLD_Y))
        ok = runner.run()
        print("[stage6_backpush] done ok=%s" % ok)
    except KeyboardInterrupt:
        print("[stage6_backpush] interrupted")
    except Exception as exc:
        print("[stage6_backpush] error: %s" % exc)
        import traceback
        traceback.print_exc()
    finally:
        if runner.low_active:
            runner._low_off()
        runner._stop()
        time.sleep(0.3)
        runner.close()
        low.stop()
        adapter.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
