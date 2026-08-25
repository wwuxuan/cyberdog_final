#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第六赛段：倒推至球线中点后，以 YOLO 球位完成一次俯身前冲。"""
import argparse
import math
import os
import time

import s6_rear as rear
import s6_base as base

try:
    from low_gait import LowGait
except ImportError:
    from stage4_package.dog.stage4_lowgait import LowGait


PUSH_GAP_WORLD_X = rear.PUSH_GAP_WORLD_X
PUSH_GAP_WORLD_Y = rear.PUSH_GAP_WORLD_Y
PUSH_GAP_FWD = rear.PUSH_GAP_FWD
PUSH_GAP_LAT = rear.PUSH_GAP_LAT

MID_PUSH_WORLD_X = (base.BALL_INITIAL_WORLD_X + PUSH_GAP_WORLD_X) / 2.0
MID_PUSH_WORLD_Y = (base.BALL_INITIAL_WORLD_Y + PUSH_GAP_WORLD_Y) / 2.0
MID_PUSH_FWD = MID_PUSH_WORLD_Y - base._START_Y
MID_PUSH_LAT = MID_PUSH_WORLD_X - base._START_X
MID_PUSH_TRAVEL_RATIO = 2.0 / 3.0

MID_PUSH_LOW_VX = 0.04
MID_PUSH_TOL = 0.06
POST_MID_FORWARD = 0.70
POST_MID_FORWARD_MAX_V = 0.12
SCAN_HALF_ANGLE = 70.0
SCAN_WZ = 0.22
SCAN_TOL = 2.0
SCAN_ROUNDS = 3
SCAN_STABLE_SAMPLES = 3
SCAN_STABLE_SPREAD = 0.35
SCAN_MAX_DEPTH_SPREAD = 0.40
LIVE_BEHIND_MAX_V = 0.10
LIVE_BEHIND_TOL = 0.07
LIVE_BEHIND_CENTER_K = 1.0
LIVE_BEHIND_CENTER_WZ_MAX = 0.30
LIVE_LEFT_OFFSET = 0.08
LIVE_LEFT_MAX_VY = 0.08
FRONT_PUSH_DISTANCE = 0.56
FRONT_PUSH_SPEED = base.PUSH_VX
FRONT_PUSH_TOL = 0.05
END_MAX_V = 0.20


def _clamp(value, low, high):
    return max(low, min(high, value))


class Stage6MidPushYolo(rear.Stage6BackPush):
    def __init__(self, adapter, detector, depth_tracker, low):
        base.Stage6Real.__init__(self, adapter, detector, depth_tracker=depth_tracker)
        self.low = low
        self.low_active = False
        self._back_push_yaw = None
        self._front_push_yaw = None

    def _wait_detector(self):
        waited = False
        while not self.det.alive():
            now = time.monotonic()
            if now - self._last_detector_wait_log_t >= 1.0:
                status = self.det.status() if hasattr(self.det, "status") else {}
                age = status.get("age")
                age_text = "never" if age is None or not math.isfinite(age) else "%.2fs" % age
                print("[stage6_midpush] waiting YOLO link frame_age=%s" % age_text)
                self._last_detector_wait_log_t = now
            self._stop()
            waited = True
            time.sleep(0.10)
        if waited:
            print("[stage6_midpush] YOLO link restored; continue current pose")

    def _push_backwards_low_to_midpoint(self):
        if self._back_push_yaw is None:
            raise RuntimeError("[stage6_midpush] rear push yaw is unset")
        heading_rad = math.radians(self._back_push_yaw)
        start_fwd, start_lat = self._fwd(), self._lat()
        target_fwd = start_fwd + MID_PUSH_TRAVEL_RATIO * (MID_PUSH_FWD - start_fwd)
        target_lat = start_lat + MID_PUSH_TRAVEL_RATIO * (MID_PUSH_LAT - start_lat)
        last_log_t = 0.0
        print("[stage6_midpush] low rear push %.0f%% target world=(%.3f, %.3f) speed=%.2f" % (
            MID_PUSH_TRAVEL_RATIO * 100.0,
            base._START_X + target_lat,
            base._START_Y + target_fwd,
            MID_PUSH_LOW_VX))
        self._low_on(-MID_PUSH_LOW_VX)
        try:
            while True:
                delta_fwd = target_fwd - self._fwd()
                delta_lat = target_lat - self._lat()
                forward_error = (
                    delta_fwd * math.cos(heading_rad)
                    - delta_lat * math.sin(heading_rad)
                )
                remaining = -forward_error
                cross = (
                    delta_fwd * math.sin(heading_rad)
                    + delta_lat * math.cos(heading_rad)
                )
                if remaining <= MID_PUSH_TOL:
                    print("[stage6_midpush] reduced low-push target reached world=(%.2f, %.2f) cross=%+.2f" % (
                        base._START_X + self._lat(), base._START_Y + self._fwd(), cross))
                    return True
                now = time.monotonic()
                if now - last_log_t >= 0.5:
                    print("[stage6_midpush] low rear push remaining=%.2f cross=%+.2f vx=-%.2f yaw=%.1f" % (
                        remaining, cross, MID_PUSH_LOW_VX, self._yaw()))
                    last_log_t = now
                time.sleep(0.05)
        finally:
            self._low_off()
            self._recover_stand()

    def _move_body_forward(self, distance):
        yaw_rad = math.radians(self._yaw())
        target_fwd = self._fwd() + distance * math.cos(yaw_rad)
        target_lat = self._lat() - distance * math.sin(yaw_rad)
        print("[stage6_midpush] normal forward %.2fm before turn-around" % distance)
        return self._walk_fixed_to(
            target_fwd,
            target_lat,
            max_v=POST_MID_FORWARD_MAX_V,
            label="post-mid forward",
            tol=0.03,
        )

    def _capture_ball_sample(self, samples, last_frame):
        status = self.det.status() if hasattr(self.det, "status") else {}
        frame = status.get("frames")
        if frame is None or frame == last_frame:
            return last_frame
        observation = self._ball_observation()
        if observation is not None:
            detail = observation.get("detail") or {}
            depth_spread = detail.get("spread") if observation["source"] == "depth" else None
            if (depth_spread is not None and math.isfinite(depth_spread)
                    and depth_spread > SCAN_MAX_DEPTH_SPREAD):
                print("[stage6_midpush] ignore scan sample: D430 depth spread=%.2fm exceeds %.2fm" % (
                    depth_spread, SCAN_MAX_DEPTH_SPREAD))
                return frame
            samples.append(observation)
            print("[stage6_midpush] scan ball sample=%d world=(%.2f, %.2f) range=%.2f source=%s" % (
                len(samples),
                base._START_X + observation["lat"],
                base._START_Y + observation["fwd"],
                observation["distance"],
                observation["source"],
            ))
        return frame

    def _turn_scan_segment(self, angle_deg, samples, last_frame):
        start_yaw = self._yaw()
        target_yaw = base._wrap_deg(start_yaw + angle_deg)
        direction = 1.0 if angle_deg >= 0.0 else -1.0
        while True:
            self._wait_detector()
            last_frame = self._capture_ball_sample(samples, last_frame)
            stable = self._stable_ball_from_samples(samples)
            if stable is not None:
                self._stop()
                return last_frame, stable
            yaw = self._yaw()
            if direction > 0.0:
                remaining = (target_yaw - yaw) % 360.0
            else:
                remaining = (yaw - target_yaw) % 360.0
            if remaining <= SCAN_TOL:
                self._stop()
                return last_frame, None
            self.adp.walk(0.0, 0.0, direction * SCAN_WZ)
            time.sleep(0.05)

    def _stable_ball_from_samples(self, samples):
        if len(samples) < SCAN_STABLE_SAMPLES:
            return None
        recent = samples[-SCAN_STABLE_SAMPLES:]
        fwd = sum(item["fwd"] for item in recent) / float(len(recent))
        lat = sum(item["lat"] for item in recent) / float(len(recent))
        spread = max(math.hypot(item["fwd"] - fwd, item["lat"] - lat) for item in recent)
        if spread > SCAN_STABLE_SPREAD:
            return None
        latest = dict(recent[-1])
        latest["fwd"] = fwd
        latest["lat"] = lat
        latest["timestamp"] = time.monotonic()
        print("[stage6_midpush] stable ball world=(%.2f, %.2f) recent=%d total=%d spread=%.2fm" % (
            base._START_X + lat,
            base._START_Y + fwd,
            len(recent),
            len(samples),
            spread,
        ))
        return latest

    def _scan_stable_ball(self):
        if self._front_push_yaw is None:
            raise RuntimeError("[stage6_midpush] front push yaw is unset")
        for scan_round in range(1, SCAN_ROUNDS + 1):
            print("[stage6_midpush] YOLO scan round %d/%d: +/-%.0fdeg" % (
                scan_round, SCAN_ROUNDS, SCAN_HALF_ANGLE))
            self._face_yaw(self._front_push_yaw, tol=SCAN_TOL)
            samples = []
            last_frame = None
            for angle_deg in (SCAN_HALF_ANGLE, -2.0 * SCAN_HALF_ANGLE, SCAN_HALF_ANGLE):
                last_frame, stable = self._turn_scan_segment(angle_deg, samples, last_frame)
                if stable is not None:
                    return stable
            stable = self._stable_ball_from_samples(samples)
            if stable is not None:
                return stable
            print("[stage6_midpush] scan round %d did not obtain a stable football" % scan_round)
        self._stop()
        return None

    def _walk_to_live_ball_behind(self, ball_state):
        active_ball = dict(ball_state)
        last_log_t = 0.0
        print("[stage6_midpush] move behind the YOLO ball while continuously updating position")
        while True:
            self._wait_detector()
            tracked = self._update_ball_track()
            if tracked is not None:
                active_ball = dict(tracked)
            target_fwd, target_lat = self._behind_point(
                active_ball["fwd"],
                active_ball["lat"],
            )
            df = target_fwd - self._fwd()
            dlat = target_lat - self._lat()
            distance = math.hypot(df, dlat)
            if distance <= LIVE_BEHIND_TOL:
                self._stop()
                print("[stage6_midpush] ball-behind reached world=(%.2f, %.2f)" % (
                    base._START_X + self._lat(), base._START_Y + self._fwd()))
                return active_ball
            yaw_rad = math.radians(self._yaw())
            adapter_y = self._pos()[1]
            target_adapter_y = -target_lat
            dy = target_adapter_y - adapter_y
            body_vx = df * math.cos(yaw_rad) + dy * math.sin(yaw_rad)
            body_vy = -df * math.sin(yaw_rad) + dy * math.cos(yaw_rad)
            speed = min(LIVE_BEHIND_MAX_V, max(0.03, 0.8 * distance))
            vx = body_vx / max(distance, 0.01) * speed
            vy = body_vy / max(distance, 0.01) * speed
            ball = self._ball()
            wz = 0.0
            if ball is not None:
                wz = _clamp(-LIVE_BEHIND_CENTER_K * (ball[2] - 0.5),
                            -LIVE_BEHIND_CENTER_WZ_MAX,
                            LIVE_BEHIND_CENTER_WZ_MAX)
            self.adp.walk(vx, vy, wz)
            now = time.monotonic()
            if now - last_log_t >= 0.5:
                print("[stage6_midpush] live-behind target=(%.2f, %.2f) err=(%+.2f,%+.2f) cmd=(%.2f,%+.2f,%.2f) ball=%s" % (
                    base._START_X + target_lat,
                    base._START_Y + target_fwd,
                    df,
                    dlat,
                    vx,
                    vy,
                    wz,
                    "cx=%.2f" % ball[2] if ball is not None else "lost",
                ))
                last_log_t = now
            time.sleep(0.05)

    def _side_step_left(self, heading):
        heading_rad = math.radians(heading)
        start_fwd, start_lat = self._fwd(), self._lat()
        last_log_t = 0.0
        while True:
            delta_fwd = self._fwd() - start_fwd
            delta_lat = self._lat() - start_lat
            left_progress = (
                delta_fwd * -math.sin(heading_rad)
                + delta_lat * -math.cos(heading_rad)
            )
            remaining = LIVE_LEFT_OFFSET - left_progress
            if remaining <= 0.015:
                self._stop()
                print("[stage6_midpush] left offset complete=%.3fm" % left_progress)
                return True
            vy = min(LIVE_LEFT_MAX_VY, max(0.03, 0.8 * remaining))
            yaw_error = base._wrap_deg(heading - self._yaw())
            wz = _clamp(0.035 * yaw_error, -0.25, 0.25)
            self.adp.walk(0.0, vy, wz)
            now = time.monotonic()
            if now - last_log_t >= 0.5:
                print("[stage6_midpush] left offset=%.3f/%.3fm vy=%.2f yaw=%.1f" % (
                    left_progress, LIVE_LEFT_OFFSET, vy, self._yaw()))
                last_log_t = now
            time.sleep(0.05)

    def _front_push_one_third(self, ball_state):
        push_yaw = math.degrees(math.atan2(
            -(PUSH_GAP_LAT - ball_state["lat"]),
            PUSH_GAP_FWD - ball_state["fwd"],
        ))
        self._face_yaw(push_yaw, tol=2.5)
        self._side_step_left(push_yaw)
        heading_rad = math.radians(push_yaw)
        start_fwd, start_lat = self._fwd(), self._lat()
        last_log_t = 0.0
        print("[stage6_midpush] pitch front push distance=%.2fm speed=%.2f yaw=%.1f" % (
            FRONT_PUSH_DISTANCE, FRONT_PUSH_SPEED, push_yaw))
        while True:
            delta_fwd = self._fwd() - start_fwd
            delta_lat = self._lat() - start_lat
            progress = (
                delta_fwd * math.cos(heading_rad)
                - delta_lat * math.sin(heading_rad)
            )
            if progress >= FRONT_PUSH_DISTANCE - FRONT_PUSH_TOL:
                self._stop()
                print("[stage6_midpush] pitch front push complete=%.2fm" % progress)
                return True
            ball = self._ball()
            wz = 0.0
            if ball is not None:
                wz = _clamp(-base.PUSH_K_CX * (ball[2] - 0.5),
                            -base.PUSH_WZ_MAX, base.PUSH_WZ_MAX)
            self._walk(FRONT_PUSH_SPEED, 0.0, wz, pitch=base.PUSH_PITCH)
            now = time.monotonic()
            if now - last_log_t >= 0.5:
                print("[stage6_midpush] pitch push progress=%.2f/%.2fm yaw=%.1f ball=%s" % (
                    progress,
                    FRONT_PUSH_DISTANCE,
                    self._yaw(),
                    "seen" if ball is not None else "held-heading",
                ))
                last_log_t = now
            time.sleep(0.05)

    def _go_to_end_and_lie_down(self, reason):
        print("[stage6_midpush] %s; go directly to end world=(%.2f, %.2f)" % (
            reason, base.END_WORLD_X, base.END_WORLD_Y))
        self._walk_fixed_to(
            base.END_FWD,
            base.END_LAT,
            max_v=END_MAX_V,
            label="end-point",
        )
        self._face_yaw(base.GAP_EDGE_YAW, tol=3.0)
        self._stop()
        self.adp.lie_down()
        time.sleep(1.0)
        print("[stage6_midpush] complete")
        return True

    def run(self):
        print("[stage6_midpush] ==== stage 6 midpoint rear push then YOLO front push ====")
        self.run_preparation_route()
        self._push_backwards_low_to_midpoint()
        self._move_body_forward(POST_MID_FORWARD)
        self._front_push_yaw = self._push_yaw()
        print("[stage6_midpush] turn around to front push yaw=%.1f" % self._front_push_yaw)
        self._face_yaw(self._front_push_yaw)
        ball_state = self._scan_stable_ball()
        if ball_state is None:
            return self._go_to_end_and_lie_down("three YOLO scan rounds found no stable football")
        ball_state = self._walk_to_live_ball_behind(ball_state)
        self._front_push_one_third(ball_state)
        return self._go_to_end_and_lie_down("front push finished")


def main():
    parser = argparse.ArgumentParser(description="stage6 midpoint rear push then YOLO front push")
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
    low = LowGait(gait_dir=args.gait_dir)
    runner = Stage6MidPushYolo(adapter, detector, depth, low)
    stream_process = None
    try:
        if not args.no_stream:
            stream_process = base._start_rgb_stream(args.pc_host, args.stream_port, args.stream_fps)
        if not args.skip_odom_wait and not adapter.wait_odom(timeout=5.0):
            print("[stage6_midpush] warning: odometry not ready")
        adapter.stand()
        adapter.set_mapped_pose(base.START_FWD, base.START_ADAPTER_Y, base.START_WORLD_YAW)
        print("[stage6_midpush] start world=(%.3f, %.3f) gap=(%.2f, %.2f) midpoint=(%.2f, %.3f)" % (
            base.START_WORLD_X,
            base.START_WORLD_Y,
            PUSH_GAP_WORLD_X,
            PUSH_GAP_WORLD_Y,
            MID_PUSH_WORLD_X,
            MID_PUSH_WORLD_Y,
        ))
        ok = runner.run()
        print("[stage6_midpush] done ok=%s" % ok)
    except KeyboardInterrupt:
        print("[stage6_midpush] interrupted")
    except Exception as exc:
        print("[stage6_midpush] error: %s" % exc)
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
        print("[stage6_midpush] end")


if __name__ == "__main__":
    main()
