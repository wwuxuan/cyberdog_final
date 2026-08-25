#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第六赛段：低姿正推至缩短目标，站立后倒退，再以 YOLO 完成后续推球。"""
import argparse
import math
import os
import time

import s6_mid as mid
import s6_base as base


class Stage6MidPushForwardYolo(mid.Stage6MidPushYolo):
    def run_preparation_route(self):
        print("[stage6_midpush_forward] fixed prep to left corner, then ball-behind point; front faces gap")
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
        print("[stage6_midpush_forward] ball->gap yaw=%.1f; front faces gap" % push_yaw)
        turn_ok = self._face_yaw(push_yaw)
        self._stop()
        print("[stage6_midpush_forward] prep complete pos=(world x=%.2f, y=%.2f) yaw=%.1f" % (
            base._START_X + self._lat(), base._START_Y + self._fwd(), self._yaw()))
        return lateral_ok and corner_ok and behind_ok and turn_ok

    def _push_forwards_low_to_reduced_target(self):
        push_yaw = self._push_yaw()
        heading_rad = math.radians(push_yaw)
        start_fwd, start_lat = self._fwd(), self._lat()
        target_fwd = start_fwd + mid.MID_PUSH_TRAVEL_RATIO * (mid.MID_PUSH_FWD - start_fwd)
        target_lat = start_lat + mid.MID_PUSH_TRAVEL_RATIO * (mid.MID_PUSH_LAT - start_lat)
        last_log_t = 0.0
        print("[stage6_midpush_forward] low forward push %.0f%% target world=(%.3f, %.3f) speed=%.2f yaw=%.1f" % (
            mid.MID_PUSH_TRAVEL_RATIO * 100.0,
            base._START_X + target_lat,
            base._START_Y + target_fwd,
            mid.MID_PUSH_LOW_VX,
            push_yaw))
        self._low_on(mid.MID_PUSH_LOW_VX)
        try:
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
                    print("[stage6_midpush_forward] reduced low-push target reached world=(%.2f, %.2f) cross=%+.2f" % (
                        base._START_X + self._lat(), base._START_Y + self._fwd(), cross))
                    return True
                now = time.monotonic()
                if now - last_log_t >= 0.5:
                    print("[stage6_midpush_forward] low forward remaining=%.2f cross=%+.2f vx=+%.2f yaw=%.1f" % (
                        remaining, cross, mid.MID_PUSH_LOW_VX, self._yaw()))
                    last_log_t = now
                time.sleep(0.05)
        finally:
            self._low_off()
            self._recover_stand()

    def _move_body_backward(self, distance):
        heading = self._yaw()
        heading_rad = math.radians(heading)
        start_fwd, start_lat = self._fwd(), self._lat()
        last_log_t = 0.0
        print("[stage6_midpush_forward] stand then direct backward %.2fm (vx<0, vy=0)" % distance)
        while True:
            delta_fwd = self._fwd() - start_fwd
            delta_lat = self._lat() - start_lat
            forward_progress = (
                delta_fwd * math.cos(heading_rad)
                - delta_lat * math.sin(heading_rad)
            )
            backward_progress = -forward_progress
            if backward_progress >= distance - 0.03:
                self._stop()
                print("[stage6_midpush_forward] direct backward complete=%.2fm" % backward_progress)
                return True
            remaining = distance - backward_progress
            vx = -min(mid.POST_MID_FORWARD_MAX_V, max(0.05, 0.8 * remaining))
            yaw_error = base._wrap_deg(heading - self._yaw())
            wz = mid._clamp(0.035 * yaw_error, -0.25, 0.25)
            self.adp.walk(vx, 0.0, wz)
            now = time.monotonic()
            if now - last_log_t >= 0.5:
                print("[stage6_midpush_forward] direct backward=%.2f/%.2fm vx=%.2f vy=0.00 yaw=%.1f" % (
                    backward_progress, distance, vx, self._yaw()))
                last_log_t = now
            time.sleep(0.05)

    def run(self):
        print("[stage6_midpush_forward] ==== stage 6 midpoint forward low push then YOLO front push ====")
        self.run_preparation_route()
        self._push_forwards_low_to_reduced_target()
        self._move_body_backward(mid.POST_MID_FORWARD)
        self._front_push_yaw = self._push_yaw()
        print("[stage6_midpush_forward] face push direction for YOLO scan yaw=%.1f" % self._front_push_yaw)
        self._face_yaw(self._front_push_yaw)
        ball_state = self._scan_stable_ball()
        if ball_state is None:
            return self._go_to_end_and_lie_down("three YOLO scan rounds found no stable football")
        ball_state = self._walk_to_live_ball_behind(ball_state)
        self._front_push_one_third(ball_state)
        return self._go_to_end_and_lie_down("front push finished")


def main():
    parser = argparse.ArgumentParser(description="stage6 midpoint forward low push then YOLO front push")
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
    runner = Stage6MidPushForwardYolo(adapter, detector, depth, low)
    stream_process = None
    try:
        if not args.no_stream:
            stream_process = base._start_rgb_stream(args.pc_host, args.stream_port, args.stream_fps)
        if not args.skip_odom_wait and not adapter.wait_odom(timeout=5.0):
            print("[stage6_midpush_forward] warning: odometry not ready")
        adapter.stand()
        adapter.set_mapped_pose(base.START_FWD, base.START_ADAPTER_Y, base.START_WORLD_YAW)
        print("[stage6_midpush_forward] start world=(%.3f, %.3f) gap=(%.2f, %.2f)" % (
            base.START_WORLD_X,
            base.START_WORLD_Y,
            mid.PUSH_GAP_WORLD_X,
            mid.PUSH_GAP_WORLD_Y,
        ))
        ok = runner.run()
        print("[stage6_midpush_forward] done ok=%s" % ok)
    except KeyboardInterrupt:
        print("[stage6_midpush_forward] interrupted")
    except Exception as exc:
        print("[stage6_midpush_forward] error: %s" % exc)
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
        print("[stage6_midpush_forward] end")


if __name__ == "__main__":
    main()
