"""Stage 3 real-dog MPC curve following."""

import math
import os
import threading
import time

try:
    import cvxpy as cp
except Exception:
    cp = None


def _wrap_to_pi(rad):
    return (rad + math.pi) % (2.0 * math.pi) - math.pi


def _clamp(v, v_min, v_max):
    return max(v_min, min(v_max, v))


def _load_curve_points(curve_path):
    pts = []
    with open(curve_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) != 2:
                continue
            pts.append((float(parts[0]) / 100.0, float(parts[1]) / 100.0))
    if len(pts) < 3:
        raise ValueError("not enough curve points")
    return pts


def _adjust_early_turn_y(points):
    """Lower the entry turn smoothly while preserving its endpoints."""
    start_y = 4.60
    end_y = 5.60
    max_drop = 0.06
    adjusted = []
    for x, y in points:
        progress = (y - start_y) / (end_y - start_y)
        if 0.0 < progress < 1.0:
            y -= max_drop * math.sin(math.pi * progress)
        adjusted.append((x, y))
    return adjusted


def _resample_points(points, spacing=0.08):
    if len(points) < 3:
        return points
    out = [points[0]]
    carry = 0.0
    last = points[0]
    for p in points[1:]:
        seg = math.hypot(p[0] - last[0], p[1] - last[1])
        if seg <= 1e-9:
            last = p
            continue
        while carry + seg >= spacing:
            ratio = (spacing - carry) / seg
            nx = last[0] + ratio * (p[0] - last[0])
            ny = last[1] + ratio * (p[1] - last[1])
            new_p = (nx, ny)
            out.append(new_p)
            last = new_p
            seg = math.hypot(p[0] - last[0], p[1] - last[1])
            carry = 0.0
            if seg <= 1e-9:
                break
        carry += seg
        last = p
    if out[-1] != points[-1]:
        out.append(points[-1])
    return out


STAGE3_ENTRY_ANCHORS = (
    (-0.203, 4.280),
    (-0.200, 4.700),
    (0.100, 5.380),
)
STAGE3_ENTRY_CORNER_CUT = 0.10


def _unit_vector(start, end):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max(math.hypot(dx, dy), 1e-6)
    return dx / length, dy / length


def _quadratic_bezier(start, corner, end, count):
    points = []
    for index in range(max(2, int(count)) + 1):
        t = index / float(max(2, int(count)))
        one_minus_t = 1.0 - t
        points.append((
            one_minus_t * one_minus_t * start[0]
            + 2.0 * one_minus_t * t * corner[0]
            + t * t * end[0],
            one_minus_t * one_minus_t * start[1]
            + 2.0 * one_minus_t * t * corner[1]
            + t * t * end[1],
        ))
    return points


def build_three_anchor_entry_path(spacing=0.08, curve_path=None):
    """Build the shared stage3 entry and return its dual-fisheye progress."""
    if curve_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        curve_path = os.path.join(base_dir, "tools", "curve_fitting", "curve.txt")
    original = _load_curve_points(curve_path)
    entry, turn, join = STAGE3_ENTRY_ANCHORS
    join_index = min(
        range(len(original)),
        key=lambda index: math.hypot(
            original[index][0] - join[0],
            original[index][1] - join[1],
        ),
    )
    previous = original[max(0, join_index - 1)]
    next_point = original[min(len(original) - 1, join_index + 1)]
    incoming_turn = _unit_vector(entry, turn)
    outgoing_turn = _unit_vector(turn, join)
    outgoing_join = _unit_vector(previous, next_point)
    cut = STAGE3_ENTRY_CORNER_CUT

    turn_before = (
        turn[0] - cut * incoming_turn[0],
        turn[1] - cut * incoming_turn[1],
    )
    turn_after = (
        turn[0] + cut * outgoing_turn[0],
        turn[1] + cut * outgoing_turn[1],
    )
    join_before = (
        join[0] - cut * outgoing_turn[0],
        join[1] - cut * outgoing_turn[1],
    )
    join_after = (
        join[0] + cut * outgoing_join[0],
        join[1] + cut * outgoing_join[1],
    )
    first_corner = _quadratic_bezier(turn_before, turn, turn_after, count=8)
    second_corner = _quadratic_bezier(join_before, join, join_after, count=8)

    tail_index = join_index + 1
    while (
        tail_index < len(original) - 1
        and (
            (original[tail_index][0] - join[0]) * outgoing_join[0]
            + (original[tail_index][1] - join[1]) * outgoing_join[1]
        ) < cut
    ):
        tail_index += 1
    raw_points = (
        [entry, turn_before]
        + first_corner[1:]
        + [join_before]
        + second_corner[1:]
        + original[tail_index:]
    )
    points = _resample_points(raw_points, spacing=spacing)
    join_control_index = min(
        range(len(points)),
        key=lambda index: math.hypot(
            points[index][0] - join[0],
            points[index][1] - join[1],
        ),
    )
    join_progress = join_control_index / max(float(len(points) - 1), 1.0)
    return points, join_progress


def _nearest_index(points, x, y, hint_idx=0, window=10, backtrack=0):
    n = len(points)
    i0 = max(0, hint_idx - backtrack)
    i1 = min(n - 1, hint_idx + window)
    best_i, best_d2 = hint_idx, float("inf")
    for i in range(i0, i1 + 1):
        d2 = (x - points[i][0]) ** 2 + (y - points[i][1]) ** 2
        if d2 < best_d2:
            best_d2, best_i = d2, i
    return best_i


def _vx_ref_for_progress(progress):
    if progress >= 0.78:
        return 0.078
    if progress >= 0.55:
        return 0.091
    return 0.104


def _wz_limit_for_progress(progress):
    if progress >= 0.78:
        return 0.12
    if progress >= 0.55:
        return 0.15
    return 0.18


def _path_heading(points, i):
    n = len(points)
    j, i0 = min(i + 1, n - 1), max(0, i - 1)
    return math.atan2(points[j][1] - points[i0][1], points[j][0] - points[i0][0])


def _lookahead_index(points, index, distance):
    """Return the first path index at least distance ahead of index."""
    target_distance = max(0.0, float(distance))
    traveled = 0.0
    last_index = len(points) - 1
    for next_index in range(index + 1, len(points)):
        previous = points[next_index - 1]
        current = points[next_index]
        traveled += math.hypot(current[0] - previous[0], current[1] - previous[1])
        if traveled >= target_distance:
            return next_index
    return last_index


def _smoothstep(t):
    t = _clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _tail_heading(points, index, start_progress=0.80, target_deg=90.0):
    """Blend only the final path heading toward a straight +y exit."""
    raw_heading = _path_heading(points, index)
    progress = index / max(float(len(points) - 1), 1.0)
    if progress <= start_progress:
        return raw_heading
    blend = _smoothstep((progress - start_progress) / max(1.0 - start_progress, 1e-6))
    target_heading = math.radians(target_deg)
    return raw_heading + blend * _wrap_to_pi(target_heading - raw_heading)


def _solve_mpc_once(e_y, e_psi, vx_ref, dt, bounds, q, r):
    if cp is None:
        return None
    n_horizon = 12
    ey = cp.Variable(n_horizon + 1)
    eps = cp.Variable(n_horizon + 1)
    u = cp.Variable((3, n_horizon))
    vx_min, vx_max = bounds["vx"]
    vy_min, vy_max = bounds["vy"]
    wz_min, wz_max = bounds["wz"]
    cost = 0
    cons = [ey[0] == e_y, eps[0] == e_psi]
    for k in range(n_horizon):
        vxk, vyk, wzk = u[0, k], u[1, k], u[2, k]
        cons += [ey[k + 1] == ey[k] + dt * (vyk + vxk * eps[k])]
        cons += [eps[k + 1] == eps[k] + dt * wzk]
        cons += [
            vxk >= vx_min,
            vxk <= vx_max,
            vyk >= vy_min,
            vyk <= vy_max,
            wzk >= wz_min,
            wzk <= wz_max,
        ]
        cost += q["ey"] * cp.square(ey[k]) + q["eps"] * cp.square(eps[k])
        cost += (
            r["vx"] * cp.square(vxk - vx_ref)
            + r["vy"] * cp.square(vyk)
            + r["wz"] * cp.square(wzk)
        )
    cost += q["ey_terminal"] * cp.square(ey[n_horizon])
    cost += q["eps_terminal"] * cp.square(eps[n_horizon])
    prob = cp.Problem(cp.Minimize(cost), cons)
    try:
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
    except Exception:
        return None
    if u.value is None:
        return None
    return float(u.value[0, 0]), float(u.value[1, 0]), float(u.value[2, 0])


def _fallback_control(e_y, e_psi, vx_ref, bounds, vy_enabled):
    vx = _clamp(vx_ref * (1.0 - 0.25 * abs(e_y)), bounds["vx"][0], bounds["vx"][1])
    vy = _clamp(-0.8 * e_y, bounds["vy"][0], bounds["vy"][1]) if vy_enabled else 0.0
    wz = _clamp(-1.8 * e_psi - 0.35 * e_y, bounds["wz"][0], bounds["wz"][1])
    return vx, vy, wz


def _format_line_info(info):
    if not info:
        return "line=off"
    left = info.get("left")
    right = info.get("right")
    left_s = "--" if left is None else f"{left:.2f}"
    right_s = "--" if right is None else f"{right:.2f}"
    return (
        f"line L={left_s} R={right_s} "
        f"err={info.get('err', 0.0):+.3f} vy={info.get('vy', 0.0):+.3f} "
        f"{info.get('reason', '')}"
    )


def run_stage3(adapter, exit_y=6.58, control_hz=15.0, vy_enabled=False,
               line_monitor=None, line_active_start=0.00,
               line_active_end=0.80, line_bias_max=0.12,
               line_monitor_config=None, line_wz_gate=0.16,
               path_spacing=0.08, tail_heading_start=0.80,
               tail_heading_deg=90.0, path_points=None,
               line_side_switch_progress=None, line_side_after="both",
               entry_turn_progress_end=0.0, entry_turn_wz_limit=0.18,
               entry_turn_slow_error_deg=6.0, entry_turn_min_vx=0.044,
               turn_lookahead_m=0.20, turn_heading_gain=0.60,
               turn_trigger_deg=4.0, turn_strong_deg=20.0,
               turn_wz_limit=0.24, turn_vx_max=0.0605,
               turn_vx_min=0.044, line_path_error_max=0.22,
               line_path_conflict_error=0.10,
               line_bias_learn_error_max=0.10):
    """Run stage3 path following, optionally aided by fisheye line centering."""
    if path_points is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        curve_path = os.path.join(base_dir, "tools", "curve_fitting", "curve.txt")
        raw_points = _adjust_early_turn_y(_load_curve_points(curve_path))
        points = _resample_points(raw_points, spacing=path_spacing)
    else:
        points = [(float(x), float(y)) for x, y in path_points]
        if len(points) < 3:
            raise ValueError("custom stage3 path must contain at least three points")
        raw_points = points

    base_bounds = {"vx": (0.02, 0.20), "vy": (-0.20, 0.20), "wz": (-0.28, 0.28)}
    q = {"ey": 60.0, "eps": 26.0, "ey_terminal": 90.0, "eps_terminal": 38.0}
    r = {"vx": 4.0, "vy": 8.0, "wz": 1.5}
    dt = 1.0 / control_hz
    hint_idx = 0
    last_log_t = 0.0
    line_was_active = False
    line_path_bias = 0.0
    # The caller prewarms the monitor; this threshold controls when its
    # correction begins to affect the path.
    line_start_progress = line_active_start
    line_starting = False
    line_start_failed = False
    line_side_switched = False
    own_line_monitor = False
    line_state = {"monitor": line_monitor, "ready": line_monitor is not None, "error": None}

    def _start_line_monitor_bg():
        nonlocal own_line_monitor
        try:
            from line_monitor import LineDistanceMonitor

            cfg = dict(line_monitor_config or {})
            monitor = LineDistanceMonitor(**cfg)
            if monitor.start(timeout=8.0, activate=True):
                line_state["monitor"] = monitor
                line_state["ready"] = True
                own_line_monitor = True
            else:
                monitor.close()
                line_state["error"] = "not_ready"
        except Exception as exc:
            line_state["error"] = repr(exc)

    print(
        f"[stage3] start MPC curve follow raw_points={len(raw_points)} "
        f"points={len(points)} spacing={path_spacing:.2f}m "
        f"custom_path={'yes' if path_points is not None else 'no'}"
    )
    try:
        while True:
            x, y, _ = adapter.get_position()
            yaw_deg = adapter.get_yaw_deg()
            yaw = math.radians(yaw_deg)

            if y >= exit_y:
                final_yaw = float(tail_heading_deg)
                print(
                    f"[stage3] exit y={y:.3f}; align final_yaw={final_yaw:.1f}"
                )
                adapter.stop()
                final_ok = adapter.align_yaw(final_yaw, tol=2.0, timeout=5.0)
                print(
                    f"[stage3] done y={y:.3f} yaw={adapter.get_yaw_deg():.1f} "
                    f"final_align={final_ok}"
                )
                break

            hint_idx = _nearest_index(points, x, y, hint_idx)
            progress = hint_idx / max(float(len(points) - 1), 1.0)
            vx_ref = _vx_ref_for_progress(progress)
            wz_limit = _wz_limit_for_progress(progress)

            if (
                line_monitor_config is not None
                and line_state["monitor"] is None
                and not line_starting
                and not line_start_failed
                and progress >= line_start_progress
            ):
                line_starting = True
                print(f"[stage3] line monitor background start at prog={progress:.2f}")
                thread = threading.Thread(target=_start_line_monitor_bg, daemon=True)
                thread.start()

            if line_starting and line_state["error"] is not None:
                line_starting = False
                line_start_failed = True
                print(f"[stage3] WARN: line monitor failed: {line_state['error']}")
            if line_state["ready"]:
                line_starting = False

            active_monitor = line_state["monitor"]
            if (
                active_monitor is not None
                and line_side_switch_progress is not None
                and not line_side_switched
                and progress >= line_side_switch_progress
            ):
                if hasattr(active_monitor, "set_side"):
                    active_monitor.set_side(line_side_after)
                else:
                    active_monitor.cfg["side"] = line_side_after
                    active_monitor.reset("side_switch")
                line_side_switched = True
                print(
                    f"[stage3] fisheye side switch at prog={progress:.2f}: "
                    f"{line_side_after}",
                    flush=True,
                )
            xr, yr = points[hint_idx]
            psi_raw = _path_heading(points, hint_idx)
            psi_path = _tail_heading(
                points,
                hint_idx,
                start_progress=tail_heading_start,
                target_deg=tail_heading_deg,
            )
            lookahead_idx = _lookahead_index(points, hint_idx, turn_lookahead_m)
            psi_ahead = _tail_heading(
                points,
                lookahead_idx,
                start_progress=tail_heading_start,
                target_deg=tail_heading_deg,
            )
            turn_ahead_deg = math.degrees(_wrap_to_pi(psi_ahead - psi_path))
            turn_abs_deg = abs(turn_ahead_deg)
            turn_strength = _smoothstep(
                (turn_abs_deg - turn_trigger_deg)
                / max(turn_strong_deg - turn_trigger_deg, 1e-6)
            )
            if turn_abs_deg > turn_trigger_deg:
                heading_blend = turn_heading_gain * _clamp(
                    turn_abs_deg / max(turn_strong_deg, 1e-6), 0.20, 1.0
                )
                psi_r = psi_path + heading_blend * _wrap_to_pi(psi_ahead - psi_path)
                vx_ref = min(
                    vx_ref,
                    float(turn_vx_max)
                    + (float(turn_vx_min) - float(turn_vx_max)) * turn_strength,
                )
                wz_limit = max(wz_limit, float(turn_wz_limit))
            else:
                psi_r = psi_path
            if progress < entry_turn_progress_end:
                wz_limit = max(wz_limit, float(entry_turn_wz_limit))
            bounds = dict(base_bounds)
            bounds["wz"] = (-wz_limit, wz_limit)
            dx, dy = x - xr, y - yr
            sinp, cosp = math.sin(psi_r), math.cos(psi_r)
            e_y = -sinp * dx + cosp * dy
            e_y_ctrl = e_y
            if active_monitor is not None and progress >= line_active_end:
                e_y_ctrl = e_y - line_path_bias
            e_psi = _wrap_to_pi(yaw - psi_r)
            if progress < entry_turn_progress_end:
                turn_error_deg = abs(math.degrees(e_psi))
                if turn_error_deg > entry_turn_slow_error_deg:
                    turn_scale = _clamp(
                        1.0 - 0.55 * (turn_error_deg - entry_turn_slow_error_deg) / 20.0,
                        entry_turn_min_vx / max(vx_ref, 1e-6),
                        1.0,
                    )
                    vx_ref *= turn_scale

            cmd = _solve_mpc_once(e_y_ctrl, e_psi, vx_ref, dt, bounds, q, r)
            if cmd is None:
                vx, vy, wz = _fallback_control(e_y_ctrl, e_psi, vx_ref, bounds, vy_enabled)
            else:
                vx, vy, wz = cmd
                if not vy_enabled:
                    vy = 0.0

            line_info = None
            if active_monitor is not None:
                line_active = line_active_start <= progress < line_active_end
                if line_active and not line_was_active:
                    active_monitor.reset("active_start")
                if not line_active and line_was_active:
                    active_monitor.reset("active_end")
                line_was_active = line_active

                if line_active:
                    line_vy, line_info = active_monitor.correction()
                    if line_info and line_info.get("ok"):
                        reject_reason = None
                        if abs(e_y) > float(line_path_error_max):
                            reject_reason = "path_far:{:+.2f}".format(e_y)
                        elif (
                            abs(e_y) > float(line_path_conflict_error)
                            and e_y * line_vy > 0.0
                        ):
                            reject_reason = "path_conflict:{:+.2f}".format(e_y)
                        if reject_reason is not None:
                            if hasattr(active_monitor, "reset"):
                                active_monitor.reset(reject_reason)
                            line_info = dict(
                                line_info,
                                ok=False,
                                vy=0.0,
                                reason="reject_" + reject_reason,
                            )
                            line_vy = 0.0
                        elif abs(e_y) <= float(line_bias_learn_error_max):
                            # Only learn a curve bias while odometry and the
                            # accepted corridor observation already agree.
                            bias_target = _clamp(e_y, -line_bias_max, line_bias_max)
                            line_path_bias = 0.96 * line_path_bias + 0.04 * bias_target
                    if line_info is not None:
                        vy = _clamp(vy + line_vy, bounds["vy"][0], bounds["vy"][1])
                elif progress < line_active_start:
                    line_info = {
                        "ok": False,
                        "left": None,
                        "right": None,
                        "err": 0.0,
                        "vy": 0.0,
                        "reason": "pre_line_zone",
                    }
                else:
                    line_info = {
                        "ok": False,
                        "left": None,
                        "right": None,
                        "err": line_path_bias,
                        "vy": 0.0,
                        "reason": "path_bias_only",
                    }
            elif line_monitor_config is not None:
                if progress < line_start_progress:
                    reason = "pre_line_start"
                elif line_starting:
                    reason = "line_starting"
                elif line_start_failed:
                    reason = "line_failed"
                else:
                    reason = "line_not_ready"
                line_info = {
                    "ok": False,
                    "left": None,
                    "right": None,
                    "err": 0.0,
                    "vy": 0.0,
                    "reason": reason,
                }

            adapter.walk(vx, vy, wz)

            now = time.time()
            if now - last_log_t > 1.0:
                heartbeat = getattr(adapter, "heartbeat_status", lambda: None)()
                heartbeat_s = ""
                if heartbeat is not None:
                    age = heartbeat.get("age")
                    age_s = "--" if age is None else f"{age:.3f}"
                    response_age = heartbeat.get("response_age")
                    response_age_s = "--" if response_age is None else f"{response_age:.3f}"
                    response_s = ""
                    if heartbeat.get("response_count", 0):
                        response_s = (
                            f" rsp={heartbeat.get('response_mode', '--')}/"
                            f"{heartbeat.get('response_gait_id', '--')}"
                            f" c={heartbeat.get('response_contact', '--')}"
                            f" sw={heartbeat.get('response_switch_status', '--')}"
                            f" oe={heartbeat.get('response_ori_error', '--')}"
                            f" fe={heartbeat.get('response_footpos_error', '--')}"
                            f" ra={response_age_s}"
                        )
                    heartbeat_s = (
                        f" hb={'ok' if heartbeat.get('process_alive', True) else 'dead'}"
                        f" n{heartbeat.get('count', 0)}"
                        f" late={heartbeat.get('late', 0)} age={age_s}"
                        f"{response_s}"
                    )
                print(
                    f"[stage3] prog={progress:.2f} y={y:.3f} ey={e_y:.3f} "
                    f"ey_ctrl={e_y_ctrl:.3f} bias={line_path_bias:+.3f} "
                    f"eps={math.degrees(e_psi):.1f} psi={math.degrees(psi_r):.1f} "
                    f"rawpsi={math.degrees(psi_raw):.1f} ahead={turn_ahead_deg:+.1f}deg "
                    f"vx_ref={vx_ref:.2f} wz_lim={wz_limit:.2f} "
                    f"v=({vx:.3f},{vy:.3f},{wz:.3f}) "
                    f"{_format_line_info(line_info)}{heartbeat_s}"
                )
                last_log_t = now

            time.sleep(dt)
    finally:
        if own_line_monitor and line_state["monitor"] is not None:
            line_state["monitor"].close()

    return True
