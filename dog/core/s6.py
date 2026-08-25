#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第六赛段：对准缺口后俯身前冲，短倒退，再以 YOLO 完成后续推球。"""
import argparse
import math
import os
import time

import s6_forward as forward
import s6_mid as mid
import s6_base as base


INITIAL_CHARGE_RATIO = 2.0 / 3.0
POST_CHARGE_BACKWARD = mid.POST_MID_FORWARD / 3.0
INITIAL_CHARGE_SPEED = base.PUSH_VX
INITIAL_CHARGE_LEFT_OFFSET = 0.08
CHARGE_SCAN_ROUNDS = 4
CHARGE_END_WORLD_X = 3.350
CHARGE_END_WORLD_Y = 12.750
CHARGE_END_FWD = CHARGE_END_WORLD_Y - base._START_Y
CHARGE_END_LAT = CHARGE_END_WORLD_X - base._START_X


class Stage6MidPushChargeYolo(forward.Stage6MidPushForwardYolo):
    def _initial_charge_left_offset(self, heading):
        heading_rad = math.radians(heading)
        start_fwd, start_lat = self._fwd(), self._lat()
        last_log_t = 0.0
        print("[stage6_midpush_charge] initial left offset %.2fm before pitch charge" %
              INITIAL_CHARGE_LEFT_OFFSET)
        while True:
            delta_fwd = self._fwd() - start_fwd
            delta_lat = self._lat() - start_lat
            left_progress = (
                delta_fwd * -math.sin(heading_rad)
                + delta_lat * -math.cos(heading_rad)
            )
            remaining = INITIAL_CHARGE_LEFT_OFFSET - left_progress
            if remaining <= 0.015:
                self._stop()
                print("[stage6_midpush_charge] initial left offset complete=%.3fm" % left_progress)
                return True
            vy = min(mid.LIVE_LEFT_MAX_VY, max(0.03, 0.8 * remaining))
            yaw_error = base._wrap_deg(heading - self._yaw())
            wz = mid._clamp(0.035 * yaw_error, -0.25, 0.25)
            self.adp.walk(0.0, vy, wz)
            now = time.monotonic()
            if now - last_log_t >= 0.5:
                print("[stage6_midpush_charge] initial left offset=%.3f/%.3fm vy=%.2f yaw=%.1f" % (
                    left_progress, INITIAL_CHARGE_LEFT_OFFSET, vy, self._yaw()))
                last_log_t = now
            time.sleep(0.05)

    def _charge_forward_to_reduced_target(self):
        push_yaw = self._push_yaw()
        self._face_yaw(push_yaw)
        self._initial_charge_left_offset(push_yaw)
        heading_rad = math.radians(push_yaw)
        start_fwd, start_lat = self._fwd(), self._lat()
        travel_ratio = mid.MID_PUSH_TRAVEL_RATIO * INITIAL_CHARGE_RATIO
        target_fwd = start_fwd + travel_ratio * (mid.MID_PUSH_FWD - start_fwd)
        target_lat = start_lat + travel_ratio * (mid.MID_PUSH_LAT - start_lat)
        last_log_t = 0.0
        print("[stage6_midpush_charge] pitch charge %.0f%% of current forward-push target world=(%.3f, %.3f) speed=%.2f pitch=%.2f yaw=%.1f" % (
            INITIAL_CHARGE_RATIO * 100.0,
            base._START_X + target_lat,
            base._START_Y + target_fwd,
            INITIAL_CHARGE_SPEED,
            base.PUSH_PITCH,
            push_yaw))
        while True:
            delta_fwd = target_fwd - self._fwd()
            delta_lat = target_lat - self._lat()
            remaining = (
                delta_fwd * math.cos(heading_rad)
                - delta_lat * math.sin(heading_rad)
            )
            cross = (
                delta_fwd * math.sin(heading_rad)
                + delta_lat * math.cos(heading_rad)
            )
            if remaining <= mid.MID_PUSH_TOL:
                self._stop()
                print("[stage6_midpush_charge] pitch charge target reached world=(%.2f, %.2f) cross=%+.2f" % (
                    base._START_X + self._lat(), base._START_Y + self._fwd(), cross))
                return True
            yaw_error = base._wrap_deg(push_yaw - self._yaw())
            wz = mid._clamp(0.03 * yaw_error, -base.PUSH_WZ_MAX, base.PUSH_WZ_MAX)
            self._walk(INITIAL_CHARGE_SPEED, 0.0, wz, pitch=base.PUSH_PITCH)
            now = time.monotonic()
            if now - last_log_t >= 0.5:
                print("[stage6_midpush_charge] pitch charge remaining=%.2f cross=%+.2f vx=+%.2f vy=0.00 yaw=%.1f" % (
                    remaining, cross, INITIAL_CHARGE_SPEED, self._yaw()))
                last_log_t = now
            time.sleep(0.05)

    def _go_to_end_and_lie_down(self, reason):
        print("[stage6_midpush_charge] %s; go directly to end world=(%.2f, %.2f)" % (
            reason, CHARGE_END_WORLD_X, CHARGE_END_WORLD_Y))
        self._walk_fixed_to(
            CHARGE_END_FWD,
            CHARGE_END_LAT,
            max_v=mid.END_MAX_V,
            label="end-point",
        )
        self._face_yaw(base.GAP_EDGE_YAW, tol=3.0)
        self._stop()
        self.adp.lie_down()
        time.sleep(1.0)
        print("[stage6_midpush_charge] complete")
        return True

    def _scan_stable_ball(self):
        if self._front_push_yaw is None:
            raise RuntimeError("[stage6_midpush_charge] front push yaw is unset")
        for scan_round in range(1, CHARGE_SCAN_ROUNDS + 1):
            print("[stage6_midpush_charge] YOLO scan round %d/%d: +/-%.0fdeg" % (
                scan_round, CHARGE_SCAN_ROUNDS, mid.SCAN_HALF_ANGLE))
            self._face_yaw(self._front_push_yaw, tol=mid.SCAN_TOL)
            samples = []
            last_frame = None
            for angle_deg in (mid.SCAN_HALF_ANGLE, -2.0 * mid.SCAN_HALF_ANGLE, mid.SCAN_HALF_ANGLE):
                last_frame, stable = self._turn_scan_segment(angle_deg, samples, last_frame)
                if stable is not None:
                    return stable
            stable = self._stable_ball_from_samples(samples)
            if stable is not None:
                return stable
            print("[stage6_midpush_charge] scan round %d did not obtain a stable football" % scan_round)
        self._stop()
        return None

    def run(self):
        print("[stage6_midpush_charge] ==== stage 6 pitch charge then YOLO front push ====")
        self.run_preparation_route()
        self._charge_forward_to_reduced_target()
        self._move_body_backward(POST_CHARGE_BACKWARD)
        self._front_push_yaw = self._push_yaw()
        print("[stage6_midpush_charge] face push direction for YOLO scan yaw=%.1f" % self._front_push_yaw)
        self._face_yaw(self._front_push_yaw)
        ball_state = self._scan_stable_ball()
        if ball_state is None:
            return self._go_to_end_and_lie_down("four YOLO scan rounds found no stable football")
        ball_state = self._walk_to_live_ball_behind(ball_state)
        self._front_push_one_third(ball_state)
        return self._go_to_end_and_lie_down("front push finished")


def main():
    parser = argparse.ArgumentParser(description="stage6 pitch charge then YOLO front push")
    parser.add_argument("--det-port", type=int, default=9890)
    parser.add_argument("--depth-topic", default=base.DEFAULT_DEPTH_TOPIC)
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--pc-host", default=base.DEFAULT_STREAM_PC_HOST)
    parser.add_argument("--stream-port", type=int, default=base.DEFAULT_STREAM_PORT)
    parser.add_argument("--stream-fps", type=float, default=base.DEFAULT_STREAM_FPS)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--skip-odom-wait", action="store_true")
    parser.add_argument("--gait-dir", default=os.path.join(base.BASE_DIR, "gait"))
    args = parser.parse_args()

    adapter = base.RealDogAdapter(None)
    detector = base.RemoteDetector(port=args.det_port, stale_timeout=base.DETECTOR_STALE_TIMEOUT)
    depth = None
    if not args.no_depth:
        base.ensure_depth_camera(args.depth_topic)
        depth = base.DepthBallTracker(args.depth_topic)
    low = mid.LowGait(gait_dir=args.gait_dir)
    runner = Stage6MidPushChargeYolo(adapter, detector, depth, low)
    stream_process = None
    try:
        if not args.no_stream:
            stream_process = base._start_rgb_stream(args.pc_host, args.stream_port, args.stream_fps)
        if not args.skip_odom_wait and not adapter.wait_odom(timeout=5.0):
            print("[stage6_midpush_charge] warning: odometry not ready")
        adapter.stand()
        adapter.set_mapped_pose(base.START_FWD, base.START_ADAPTER_Y, base.START_WORLD_YAW)
        print("[stage6_midpush_charge] start world=(%.3f, %.3f) gap=(%.2f, %.2f)" % (
            base.START_WORLD_X,
            base.START_WORLD_Y,
            mid.PUSH_GAP_WORLD_X,
            mid.PUSH_GAP_WORLD_Y,
        ))
        ok = runner.run()
        print("[stage6_midpush_charge] done ok=%s" % ok)
    except KeyboardInterrupt:
        print("[stage6_midpush_charge] interrupted")
    except Exception as exc:
        print("[stage6_midpush_charge] error: %s" % exc)
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
        detector.stop()
        if depth is not None:
            depth.stop()
        if stream_process is not None:
            stream_process.terminate()
            try:
                stream_process.wait(timeout=2.0)
            except Exception:
                stream_process.kill()
        print("[stage6_midpush_charge] end")


if __name__ == "__main__":
    main()
