import math
import os
import threading
import time
from collections import deque
from itertools import permutations

try:
    import rclpy
    from orange import OrangeDetector
    HAS_ROS = True
except ImportError:
    HAS_ROS = False


GRID_X = [-0.30, 0.90, 2.10, 3.30]
GRID_Y = [1.34, 2.18, 3.02, 3.86]
PATH_X = [0.300, 1.500, 2.700]
PATH_Y = [0.92, 1.76, 2.60, 3.44, 4.28]

S2_SCAN = (1.500, 2.60)
S2_RULE_SCAN = (2.700, 1.76)
S2_UPPER_LEFT_CENTER = (0.300, 3.44)
S2_UPPER_LEFT_FRONT_FROM = (0.300, 2.60)
S2_PROBE_LEFT_TRAVERSE = (0.300, 1.76)
S2_END = (0.300, 4.28)
S3_START = (-0.203, 4.280)
FIXED_BLUE = {(2.10, 1.34), (3.30, 1.34), (3.30, 2.18)}
BOARD_BALLS = {(x, y) for x in GRID_X for y in GRID_Y}
FIXED_BLUE_PRIOR = 999
N_BALLS = 4

SCAN_DEG = 360.0
SCAN_STEP_DEG = 30.0
WARMUP_SECS = 1.5
PROBE_SAMPLE_SECS = 1.2
PROBE_SAMPLE_DT = 0.08
PROBE_ANGLE_WINDOW_DEG = 14.0
PROBE_DIST_WINDOW = 0.45
PROBE_CENTER_WINDOW_DEG = 18.0
PROBE_COLOR_MARGIN = 1.12
PROBE_TARGET_WINDOW_DEG = 16.0
PROBE_MIN_COLOR_VOTES = 3
PROBE_MIN_CONFIDENCE = 0.65
PROBE_RETRY_SAMPLE_SECS = 0.65
PROBE_RETRY_YAW_OFFSETS_DEG = (-3.5, 3.5)
PROBE_LEFT_PAIR_NAME = "C_r1c2"
PROBE_LEFT_PAIR_MIN_AREA = 180.0
PROBE_LEFT_PAIR_MIN_RADIUS = 8.0
PROBE_LEFT_PAIR_MIN_HORIZONTAL_GAP_PX = 24.0
PROBE_LEFT_PAIR_MAX_VERTICAL_GAP_RATIO = 0.16
PROBE_LEFT_PAIR_CENTER_BAND_RATIO = 0.32
PROBE_LEFT_PAIR_MAX_MIDPOINT_ERR_DEG = 14.0
PROBE_LEFT_PAIR_MIN_SAMPLES = 2

PROBE_CELLS = {
    "A_r3c4": (3, 4),
    "B_r2c3": (2, 3),
    "C_r1c2": (1, 2),
}

S2_APPROACH_VX = 0.28
S2_RETURN_VX = 0.22
S2_WALK_VX = 0.22
S2_HIT_DRIVE_DIST = 0.80
S2_HIT_DRIVE_RATIO = 0.50
S2_WP_TOL = 0.08
S2_WP_TIMEOUT = 20.0
S2_LANE_K = 0.50
S2_LANE_MAX_VY = 0.10
S2_K_YAW = 0.03
S2_MAX_WZ = 0.18
S2_RADAR_MIN_DISTANCE = 0.25
S2_RADAR_MAX_DISTANCE = 2.20
S2_RADAR_MAX_MATCH_ERROR = 0.45
S2_RADAR_CORRIDOR_MARGIN = 0.08
S2_RADAR_CORRIDOR_ALONG_MARGIN = 0.48
S2_RADAR_MAX_SIDE_ALONG_GAP = 0.95
S2_RADAR_FILTER_ALPHA = 0.55
S2_RADAR_APPLY_GAIN = 0.30
S2_RADAR_MAX_STEP = 0.035
S2_RADAR_MAX_YAW_STEP_DEG = 1.5
S2_RADAR_MIN_STEP = 0.006
S2_RADAR_MIN_YAW_STEP_DEG = 0.35
S2_RADAR_MAX_POSE_ERROR = 0.60
S2_RADAR_MAX_YAW_ERROR_DEG = 35.0
S2_RADAR_APPLY_COOLDOWN = 0.18
S2_RADAR_PAIR_SPACING_TOL = 0.12
S2_RADAR_REACQUIRE_CONFIRM_FRAMES = 3
S2_RADAR_REACQUIRE_MAX_POSE_DELTA = 0.08
S2_RADAR_REACQUIRE_MAX_YAW_DELTA_DEG = 5.0
S2_RADAR_CROSS_TRACK_MAX_STEP = 0.012
S2_RADAR_REACQUIRE_SPEED_SCALE = 0.55
S2_RADAR_NODE_CALIBRATION_TIMEOUT = 2.8
S2_RADAR_NODE_STABLE_FRAMES = 3
S2_RADAR_NODE_STABLE_POS_DELTA = 0.05
S2_RADAR_NODE_STABLE_YAW_DELTA_DEG = 4.0
S2_RADAR_NODE_MAX_POSE_ERROR = 0.20
S2_RADAR_NODE_MAX_YAW_ERROR_DEG = 12.0
S2_RADAR_SINGLE_LANE_RANGE = 1.60
S2_RADAR_SINGLE_LANE_K = 0.45
S2_RADAR_SINGLE_LANE_MAX_VY = 0.04
S2_RADAR_SINGLE_GUARD_STALE_SECS = 0.45
S2_LEG_ALIGN_TOL = 0.8
S2_ALIGN_TOL = 2.5
S2_MOVE_MAX_YAW_ERR = 14.0
S2_OVERSHOOT_MARGIN = 0.03
S2_RETURN_TOL = 0.12
S2_RETURN_OVERSHOOT_MARGIN = 0.04
S2_RETURN_EXTRA_TIME = 1.2
S2_VISUAL_REFINE_START_INDEX = 1
S2_VISUAL_REFINE_STEPS = 10
S2_VISUAL_REFINE_WINDOW_DEG = 32.0
S2_VISUAL_REFINE_TOL_DEG = 3.0
S2_VISUAL_REFINE_MAX_STEP_DEG = 5.0
S2_VISUAL_REFINE_MIN_AREA = 180.0
S2_VISUAL_BLOB_MIN_AREA = 180.0
S2_VISUAL_REF_RADIUS_MIN = 70.0
S2_VISUAL_REF_RADIUS_MAX = 160.0
S2_VISUAL_REF_AREA_MIN = 20000.0
S2_VISUAL_REF_AREA_MAX = 55000.0
S2_VISUAL_HIT_TOL_DEG = 12.0
S2_VISUAL_HIT_STABLE_FRAMES = 2
S2_VISUAL_LOST_FRAMES = 6
S2_VISUAL_FALLBACK_ANGLE_DEG = 16.0
S2_VISUAL_FALLBACK_MAX_DELTA_DEG = 12.0
S2_VISUAL_YAW_ANCHOR_MAX_NODE_ERROR = 0.18
S2_VISUAL_YAW_ANCHOR_MAX_CORRECTION_DEG = 40.0
S2_VISUAL_SEARCH_STEP_DEG = 5.0
S2_VISUAL_SEARCH_MAX_DEG = 20.0
S2_VISUAL_SEARCH_SETTLE = 0.08
S2_VISUAL_SEARCH_PRE_DEG = 28.0
S2_VISUAL_SEARCH_DT = 0.07
S2_VISUAL_SEARCH_WZ_MIN = 0.10
S2_VISUAL_SEARCH_WZ_MAX = 0.22
S2_DEBUG_DIR = "stage2_debug"
PROBE_DEBUG_SAVE_INTERVAL_SECS = 0.25
PROBE_DEBUG_LAST_WRITE = {}
S2_FIRST_HIT_DIRECT_APPROACH_TOL = 0.16


def _clamp(value, low, high):
    return max(low, min(high, value))


def _yaw_err(target, current):
    return ((target - current) + 180.0) % 360.0 - 180.0


def _wrap_deg(angle):
    return ((angle + 180.0) % 360.0) - 180.0


def snap_to_grid(bx, by):
    return (
        min(GRID_X, key=lambda gx: abs(gx - bx)),
        min(GRID_Y, key=lambda gy: abs(gy - by)),
    )


def grid_cell(row, col):
    return (GRID_X[col - 1], GRID_Y[row - 1])


def orange_confidence(votes):
    orange = votes.get("orange", 0)
    blue = votes.get("blue", 0)
    return orange / (orange + blue) if orange + blue > 0 else 0.0


def select_orange_balls(grid_votes):
    scored = []
    for sx in GRID_X:
        for sy in GRID_Y:
            votes = grid_votes.get((sx, sy), {})
            if (sx, sy) in FIXED_BLUE:
                continue
            orange = votes.get("orange", 0)
            blue = votes.get("blue", 0)
            if orange + blue == 0:
                continue
            scored.append((orange_confidence(votes), orange, -blue, sx, sy))

    scored.sort(reverse=True)
    selected = []
    used_x = set()
    used_y = set()
    for conf, orange, _neg_blue, sx, sy in scored:
        if sx in used_x or sy in used_y:
            continue
        if orange <= 0:
            continue
        selected.append((conf, (sx, sy)))
        used_x.add(sx)
        used_y.add(sy)
        if len(selected) >= N_BALLS:
            break
    return selected


def infer_orange_balls_from_probes(probe_is_orange):
    matches = []
    for perm in permutations((1, 2, 3, 4)):
        row_to_col = {row: perm[row - 1] for row in (1, 2, 3, 4)}
        oranges = {grid_cell(row, col) for row, col in row_to_col.items()}
        if any(pos in oranges for pos in FIXED_BLUE):
            continue
        if (row_to_col[3] == 4) != probe_is_orange["A_r3c4"]:
            continue
        if (row_to_col[2] == 3) != probe_is_orange["B_r2c3"]:
            continue
        if (row_to_col[1] == 2) != probe_is_orange["C_r1c2"]:
            continue
        matches.append(row_to_col)

    if len(matches) != 1:
        print(f"[stage2] ERROR: probe rule matched {len(matches)} layouts")
        return []

    row_to_col = matches[0]
    balls = [grid_cell(row, row_to_col[row]) for row in (1, 2, 3, 4)]
    labels = " ".join(
        f"r{row}c{row_to_col[row]}=({balls[row - 1][0]:.2f},{balls[row - 1][1]:.2f})"
        for row in (1, 2, 3, 4)
    )
    print(f"[stage2] inferred orange layout {labels}")
    return balls


def infer_orange_balls_from_probe_probabilities(probe_results):
    matches = []
    for perm in permutations((1, 2, 3, 4)):
        row_to_col = {row: perm[row - 1] for row in (1, 2, 3, 4)}
        oranges = {grid_cell(row, col) for row, col in row_to_col.items()}
        if any(pos in oranges for pos in FIXED_BLUE):
            continue

        log_prob = 0.0
        for name, (row, col) in PROBE_CELLS.items():
            orange_prob = _clamp(
                float(probe_results[name]["orange_prob"]), 0.01, 0.99
            )
            expected_orange = row_to_col[row] == col
            log_prob += math.log(orange_prob if expected_orange else 1.0 - orange_prob)
        matches.append((log_prob, row_to_col))

    if not matches:
        print("[stage2] ERROR: no valid board layout; use fixed fallback")
        return [grid_cell(row, row) for row in (1, 2, 3, 4)]

    matches.sort(key=lambda item: item[0], reverse=True)
    log_prob, row_to_col = matches[0]
    balls = [grid_cell(row, row_to_col[row]) for row in (1, 2, 3, 4)]
    labels = " ".join(
        f"r{row}c{row_to_col[row]}=({balls[row - 1][0]:.2f},{balls[row - 1][1]:.2f})"
        for row in (1, 2, 3, 4)
    )
    print(
        f"[stage2] inferred orange layout max_probability logp={log_prob:.2f} {labels}",
        flush=True,
    )
    return balls


def print_inferred_board(orange_balls):
    """Print the rule-inferred board in the same compact form as the check tool."""
    orange_set = set(orange_balls)
    probe_by_pos = {
        grid_cell(row, col): name[0]
        for name, (row, col) in PROBE_CELLS.items()
    }
    print("[stage2] board legend: Y=yellow/orange B=blue *=fixed-blue A/B/C=probed")
    print("[stage2]        c1      c2      c3      c4")
    for row in (4, 3, 2, 1):
        cells = []
        for col in (1, 2, 3, 4):
            pos = grid_cell(row, col)
            color = "Y" if pos in orange_set else "B"
            suffix = "*" if pos in FIXED_BLUE else ""
            suffix += probe_by_pos.get(pos, "")
            cells.append((color + suffix).ljust(5))
        print(f"[stage2] r{row}  " + "  ".join(cells))


def snap_to_path(x, y):
    return (
        min(PATH_X, key=lambda px: abs(px - x)),
        min(PATH_Y, key=lambda py: abs(py - y)),
    )


def path_neighbors(node):
    x, y = node
    xi = PATH_X.index(x)
    yi = PATH_Y.index(y)
    neighbors = []
    if xi > 0:
        neighbors.append((PATH_X[xi - 1], y))
    if xi < len(PATH_X) - 1:
        neighbors.append((PATH_X[xi + 1], y))
    if yi > 0:
        neighbors.append((x, PATH_Y[yi - 1]))
    if yi < len(PATH_Y) - 1:
        neighbors.append((x, PATH_Y[yi + 1]))
    return neighbors


def path_bfs(start, end):
    if start == end:
        return [start]
    queue = deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        for neighbor in path_neighbors(path[-1]):
            if neighbor == end:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return [start]


def path_cost(start, end):
    route = path_bfs(start, end)
    return sum(
        math.hypot(route[i + 1][0] - route[i][0], route[i + 1][1] - route[i][1])
        for i in range(len(route) - 1)
    )


def ball_approach_nodes(bx, by):
    left_x = max((px for px in PATH_X if px <= bx), default=PATH_X[0])
    right_x = min((px for px in PATH_X if px >= bx), default=PATH_X[-1])
    below_y = max((py for py in PATH_Y if py <= by), default=PATH_Y[0])
    above_y = min((py for py in PATH_Y if py >= by), default=PATH_Y[-1])
    px_list = [left_x] if left_x == right_x else [left_x, right_x]
    py_list = [below_y] if below_y == above_y else [below_y, above_y]
    return [(px, py) for px in px_list for py in py_list]


def start_pose_logger(adapter, interval=3.0, prefix="[stage2]"):
    stop_event = threading.Event()

    def _loop():
        while not stop_event.wait(interval):
            try:
                x, y, z = adapter.get_position()
                yaw = adapter.get_yaw_deg()
                path_node = snap_to_path(x, y)
                grid_node = snap_to_grid(x, y)
                dist_end = math.hypot(x - S2_END[0], y - S2_END[1])
                dist_stage3 = math.hypot(x - S3_START[0], y - S3_START[1])
                print(
                    f"{prefix} dbg pos=({x:.2f},{y:.2f},{z:.2f}) yaw={yaw:.1f} "
                    f"path=({path_node[0]:.2f},{path_node[1]:.2f}) "
                    f"grid=({grid_node[0]:.2f},{grid_node[1]:.2f}) "
                    f"d_end={dist_end:.2f} d_s3={dist_stage3:.2f}",
                    flush=True,
                )
            except Exception as exc:
                print(f"{prefix} dbg error={exc}", flush=True)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return stop_event, thread


def _pose_text(adapter):
    x, y, z = adapter.get_position()
    yaw = adapter.get_yaw_deg()
    path_node = snap_to_path(x, y)
    grid_node = snap_to_grid(x, y)
    return (
        f"pos=({x:.2f},{y:.2f},{z:.2f}) yaw={yaw:.1f} "
        f"path=({path_node[0]:.2f},{path_node[1]:.2f}) "
        f"grid=({grid_node[0]:.2f},{grid_node[1]:.2f})"
    )


def s2_align_yaw(adapter, target_deg, label="yaw", tol=S2_ALIGN_TOL, timeout=None):
    start_yaw = adapter.get_yaw_deg()
    start_err = _yaw_err(target_deg, start_yaw)
    if timeout is None:
        timeout = max(6.0, min(13.0, abs(start_err) / 18.0 + 3.0))
    print(
        f"[stage2] align {label} target={target_deg:.1f} "
        f"yaw={start_yaw:.1f} err={start_err:.1f} timeout={timeout:.1f}",
        flush=True,
    )
    ok = adapter.align_yaw(target_deg, tol=tol, timeout=timeout)
    end_yaw = adapter.get_yaw_deg()
    end_err = _yaw_err(target_deg, end_yaw)
    print(
        f"[stage2] align {label} done ok={ok} "
        f"yaw={end_yaw:.1f} err={end_err:.1f}",
        flush=True,
    )
    return ok or abs(end_err) <= tol


def _fmt_candidate(item):
    if not item:
        return "none"
    return (
        f"ang={item['angle_deg']:.1f}deg area={item['area']:.0f} "
        f"r={item.get('radius', 0.0):.1f} "
        f"score={item.get('score', 0.0):.1f} "
        f"circ={item['circularity']:.2f} fill={item['fill_ratio']:.2f} "
        f"src={item.get('source', 'ball_candidate')}"
    )


def _s2_visual_ref_size_ok(item):
    if not item:
        return False
    radius = float(item.get("radius", 0.0))
    area = float(item.get("area", 0.0))
    return (
        S2_VISUAL_REF_RADIUS_MIN <= radius <= S2_VISUAL_REF_RADIUS_MAX
        and S2_VISUAL_REF_AREA_MIN <= area <= S2_VISUAL_REF_AREA_MAX
    )


def s2_visual_refine_yaw(adapter, detector, expected_yaw, label="hit_visual"):
    if detector is None:
        print(f"[stage2] visual_refine {label} skipped: no detector", flush=True)
        return None

    last_info = None
    fallback_yaw = None
    fallback_angle_deg = None

    def remember_fallback(item):
        nonlocal fallback_yaw, fallback_angle_deg
        if item is None:
            return
        angle_deg = float(item["angle_deg"])
        if abs(angle_deg) > S2_VISUAL_FALLBACK_ANGLE_DEG:
            return
        if fallback_angle_deg is not None and abs(angle_deg) >= abs(fallback_angle_deg):
            return
        raw_yaw = _wrap_deg(adapter.get_yaw_deg() + angle_deg)
        delta = _clamp(
            _yaw_err(raw_yaw, expected_yaw),
            -S2_VISUAL_FALLBACK_MAX_DELTA_DEG,
            S2_VISUAL_FALLBACK_MAX_DELTA_DEG,
        )
        fallback_yaw = _wrap_deg(expected_yaw + delta)
        fallback_angle_deg = angle_deg

    def save_visual_debug(info, item, suffix):
        if info is None or not hasattr(detector, "save_debug_image"):
            return
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            debug_dir = os.path.join(base_dir, S2_DEBUG_DIR)
            os.makedirs(debug_dir, exist_ok=True)
            info = dict(info)
            if item is not None:
                info["selected"] = item
                items = list(info.get("center_candidates") or [])
                if item not in items:
                    items.append(item)
                info["center_candidates"] = items
            path = os.path.join(debug_dir, f"{label}_{suffix}.jpg")
            title = f"{label} {suffix} yaw={adapter.get_yaw_deg():.1f} {_fmt_candidate(item)}"
            if detector.save_debug_image(path, info, title=title):
                print(f"[stage2] visual_refine {label} debug_image={path}", flush=True)
        except Exception as exc:
            print(f"[stage2] visual_refine {label} debug_image_error={exc}", flush=True)

    def find_yellow(tag):
        info = detector.classify_center_ball(
            expected_angle_deg=0.0,
            target_window_deg=S2_VISUAL_REFINE_WINDOW_DEG,
        )
        candidates = []
        item = info.get("orange")
        if (
            item
            and item.get("area", 0.0) >= S2_VISUAL_REFINE_MIN_AREA
            and _s2_visual_ref_size_ok(item)
        ):
            item["source"] = item.get("source", "ball_candidate")
            candidates.append(item)
        if hasattr(detector, "find_center_color_blob"):
            blob = detector.find_center_color_blob(
                color="orange",
                target_window_deg=S2_VISUAL_REFINE_WINDOW_DEG,
                min_area=S2_VISUAL_BLOB_MIN_AREA,
                rgb=info.get("rgb"),
            )
            if blob is not None and _s2_visual_ref_size_ok(blob):
                candidates.append(blob)
        item = max(candidates, key=lambda c: c.get("score", 0.0)) if candidates else None
        ok = item is not None
        print(
            f"[stage2] visual_refine {label} {tag} "
            f"{'yellow' if ok else 'no_yellow'} candidate={_fmt_candidate(item)} "
            f"yaw={adapter.get_yaw_deg():.1f}",
            flush=True,
        )
        return ok, item, info

    def fine_step_for(angle_deg):
        abs_angle = abs(angle_deg)
        if abs_angle <= 8.0:
            return _clamp(angle_deg, -2.0, 2.0)
        if abs_angle <= 15.0:
            return _clamp(angle_deg, -3.0, 3.0)
        if abs_angle <= 25.0:
            return _clamp(angle_deg, -4.0, 4.0)
        return _clamp(angle_deg, -S2_VISUAL_REFINE_MAX_STEP_DEG, S2_VISUAL_REFINE_MAX_STEP_DEG)

    def turn_until_seen():
        nonlocal last_info
        start_yaw = adapter.get_yaw_deg()
        initial_err = _yaw_err(expected_yaw, start_yaw)
        search_sign = 1.0 if initial_err >= 0.0 else -1.0
        extra_turned = 0.0
        passed_expected = False
        print(
            f"[stage2] visual_refine {label} turn_search "
            f"start_yaw={start_yaw:.1f} expected={expected_yaw:.1f} "
            f"initial_err={initial_err:.1f} pre_deg={S2_VISUAL_SEARCH_PRE_DEG:.1f}",
            flush=True,
        )
        t0 = time.time()
        while time.time() - t0 < 12.0:
            yaw = adapter.get_yaw_deg()
            err = _yaw_err(expected_yaw, yaw)
            if abs(err) <= S2_VISUAL_SEARCH_PRE_DEG or passed_expected:
                ok, item, last_info = find_yellow(
                    f"turn err={err:.1f} extra={extra_turned:.1f}"
                )
                if ok:
                    adapter.stop()
                    return item

            if not passed_expected and abs(err) <= S2_ALIGN_TOL:
                passed_expected = True

            if passed_expected:
                if extra_turned >= S2_VISUAL_SEARCH_MAX_DEG:
                    break
                wz = search_sign * S2_VISUAL_SEARCH_WZ_MIN
                extra_turned += abs(math.degrees(wz) * S2_VISUAL_SEARCH_DT)
            else:
                wz_mag = _clamp(
                    abs(err) * 0.025,
                    S2_VISUAL_SEARCH_WZ_MIN,
                    S2_VISUAL_SEARCH_WZ_MAX,
                )
                wz = search_sign * wz_mag

            adapter.walk(0.0, 0.0, wz)
            time.sleep(S2_VISUAL_SEARCH_DT)

        adapter.stop()
        return None

    item = turn_until_seen()

    if item is None:
        save_visual_debug(last_info, None, "failed")
        print(
            f"[stage2] visual_refine {label} unavailable: yellow not found; "
            "continue with coordinate heading",
            flush=True,
        )
        return None

    stable_frames = 0
    lost_streak = 0
    last_angle_deg = None
    for step in range(S2_VISUAL_REFINE_STEPS):
        if item is None:
            ok, item, last_info = find_yellow(f"center_step={step}")
            if not ok:
                if fallback_yaw is not None:
                    print(
                        f"[stage2] visual_refine {label} lost yellow; "
                        f"use last visual heading={fallback_yaw:.1f} "
                        f"last_angle={fallback_angle_deg:+.1f}deg",
                        flush=True,
                    )
                    return fallback_yaw, None
                lost_streak += 1
                stable_frames = 0
                if lost_streak < S2_VISUAL_LOST_FRAMES:
                    if last_angle_deg is not None:
                        step_deg = fine_step_for(last_angle_deg)
                        s2_align_yaw(
                            adapter,
                            adapter.get_yaw_deg() + step_deg,
                            label=f"{label}_track",
                            tol=2.0,
                            timeout=2.0,
                        )
                    time.sleep(0.08)
                    continue
                print(
                    f"[stage2] visual_refine {label} lost yellow while centering",
                    flush=True,
                )
                save_visual_debug(last_info, item, "lost")
                return None
            lost_streak = 0

        angle_deg = float(item["angle_deg"])
        last_angle_deg = angle_deg
        remember_fallback(item)
        if abs(angle_deg) <= S2_VISUAL_HIT_TOL_DEG:
            stable_frames += 1
        else:
            stable_frames = 0
            lost_streak = 0
        print(
            f"[stage2] visual_refine {label} step={step} "
            f"yellow={_fmt_candidate(item)} yaw={adapter.get_yaw_deg():.1f} "
            f"stable={stable_frames}/{S2_VISUAL_HIT_STABLE_FRAMES}",
            flush=True,
        )
        if stable_frames >= S2_VISUAL_HIT_STABLE_FRAMES:
            yaw = adapter.get_yaw_deg()
            save_visual_debug(last_info, item, "done")
            print(f"[stage2] visual_refine {label} done yaw={yaw:.1f}", flush=True)
            return yaw, item
        step_deg = fine_step_for(angle_deg)
        s2_align_yaw(
            adapter,
            adapter.get_yaw_deg() + step_deg,
            label=f"{label}_step",
            tol=2.0,
            timeout=4.0,
        )
        time.sleep(0.18)
        ok, item, last_info = find_yellow(f"center_step={step + 1}")
        if not ok:
            item = None

    ok, item, last_info = find_yellow("final_check")
    if ok:
        final_angle = abs(float(item["angle_deg"]))
    else:
        final_angle = float("inf")
    if ok and final_angle <= S2_VISUAL_HIT_TOL_DEG:
        yaw = adapter.get_yaw_deg()
        save_visual_debug(last_info, item, "done")
        print(f"[stage2] visual_refine {label} done yaw={yaw:.1f}", flush=True)
        return yaw, item
    if fallback_yaw is not None:
        print(
            f"[stage2] visual_refine {label} final fallback "
            f"heading={fallback_yaw:.1f} last_angle={fallback_angle_deg:+.1f}deg",
            flush=True,
        )
        return fallback_yaw, None
    save_visual_debug(last_info, item, "not_centered")
    print(
        f"[stage2] visual_refine {label} failed: yellow not centered "
        f"last={_fmt_candidate(item)}",
        flush=True,
    )
    return None


def s2_anchor_yaw_from_centered_ball(adapter, ball_pos, approach_node, visual_item, label):
    """Anchor the field yaw to a visually centered, known yellow ball."""
    if not hasattr(adapter, "set_mapped_pose"):
        print(
            f"[stage2] visual_yaw_anchor {label} skipped: adapter cannot remap pose",
            flush=True,
        )
        return None

    x, y, _ = adapter.get_position()
    node_error = math.hypot(x - approach_node[0], y - approach_node[1])
    if node_error > S2_VISUAL_YAW_ANCHOR_MAX_NODE_ERROR:
        print(
            f"[stage2] visual_yaw_anchor {label} skipped: node_error={node_error:.2f} "
            f"limit={S2_VISUAL_YAW_ANCHOR_MAX_NODE_ERROR:.2f}",
            flush=True,
        )
        return None

    expected_yaw = math.degrees(math.atan2(ball_pos[1] - y, ball_pos[0] - x))
    residual_angle = float(visual_item.get("angle_deg", 0.0))
    anchored_yaw = _wrap_deg(expected_yaw - residual_angle)
    reported_yaw = adapter.get_yaw_deg()
    correction = _yaw_err(anchored_yaw, reported_yaw)
    if abs(correction) > S2_VISUAL_YAW_ANCHOR_MAX_CORRECTION_DEG:
        print(
            f"[stage2] visual_yaw_anchor {label} rejected: correction={correction:+.1f} "
            f"limit={S2_VISUAL_YAW_ANCHOR_MAX_CORRECTION_DEG:.1f} "
            f"expected={expected_yaw:.1f} residual={residual_angle:+.1f}",
            flush=True,
        )
        return None

    adapter.set_mapped_pose(x, y, anchored_yaw)
    print(
        f"[stage2] visual_yaw_anchor {label} applied correction={correction:+.1f} "
        f"expected={expected_yaw:.1f} residual={residual_angle:+.1f} "
        f"yaw={anchored_yaw:.1f} node_error={node_error:.2f}",
        flush=True,
    )
    return anchored_yaw


def s2_plan_hits(balls, start_pos):
    remaining = list(balls)
    sequence = []
    current = snap_to_path(*start_pos)
    while remaining:
        best_ball = None
        best_node = None
        best_cost = float("inf")
        for ball in remaining:
            for node in ball_approach_nodes(*ball):
                cost = path_cost(current, node)
                if cost < best_cost:
                    best_cost = cost
                    best_ball = ball
                    best_node = node
        sequence.append((best_ball, best_node))
        remaining.remove(best_ball)
        current = best_node
    return sequence


def _direction_key(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return (
        1 if dx > 1e-6 else (-1 if dx < -1e-6 else 0),
        1 if dy > 1e-6 else (-1 if dy < -1e-6 else 0),
    )


def _radar_corridor_pair(active_balls, from_node, to_node, robot_x, robot_y):
    """Return the two unhit balls bordering the current path corridor."""
    debug = {"reason": "", "axis": None, "candidates": 0}
    if not active_balls:
        debug["reason"] = "no_unhit_balls"
        return None, debug

    dx = to_node[0] - from_node[0]
    dy = to_node[1] - from_node[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        debug["reason"] = "zero_length_leg"
        return None, debug

    balls = {(float(x), float(y)) for x, y in active_balls}
    vertical = abs(dy) >= abs(dx)
    if vertical:
        lane = from_node[0]
        left_columns = sorted({x for x, _ in balls if x < lane - S2_RADAR_CORRIDOR_MARGIN})
        right_columns = sorted({x for x, _ in balls if x > lane + S2_RADAR_CORRIDOR_MARGIN})
        if not left_columns or not right_columns:
            debug.update({
                "reason": "missing_left_or_right",
                "axis": "y",
                "lane": lane,
                "first_values": left_columns,
                "second_values": right_columns,
            })
            return None, debug
        left_x = left_columns[-1]
        right_x = right_columns[0]
        first_side = [ball for ball in balls if abs(ball[0] - left_x) < 1e-6]
        second_side = [ball for ball in balls if abs(ball[0] - right_x) < 1e-6]
        current = robot_y
        direction = 1.0 if dy >= 0.0 else -1.0
        along = lambda ball: ball[1]
        side_names = ("left", "right")
    else:
        lane = from_node[1]
        lower_rows = sorted({y for _, y in balls if y < lane - S2_RADAR_CORRIDOR_MARGIN})
        upper_rows = sorted({y for _, y in balls if y > lane + S2_RADAR_CORRIDOR_MARGIN})
        if not lower_rows or not upper_rows:
            debug.update({
                "reason": "missing_lower_or_upper",
                "axis": "x",
                "lane": lane,
                "first_values": lower_rows,
                "second_values": upper_rows,
            })
            return None, debug
        lower_y = lower_rows[-1]
        upper_y = upper_rows[0]
        first_side = [ball for ball in balls if abs(ball[1] - lower_y) < 1e-6]
        second_side = [ball for ball in balls if abs(ball[1] - upper_y) < 1e-6]
        current = robot_x
        direction = 1.0 if dx >= 0.0 else -1.0
        along = lambda ball: ball[0]
        side_names = ("lower", "upper")

    preferred = current + direction * S2_RADAR_CORRIDOR_ALONG_MARGIN
    candidate_pairs = []
    for first in first_side:
        for second in second_side:
            first_along = along(first)
            second_along = along(second)
            first_forward = direction * (first_along - current)
            second_forward = direction * (second_along - current)
            if first_forward < -S2_RADAR_CORRIDOR_ALONG_MARGIN:
                continue
            if second_forward < -S2_RADAR_CORRIDOR_ALONG_MARGIN:
                continue
            along_gap = abs(first_along - second_along)
            if along_gap > S2_RADAR_MAX_SIDE_ALONG_GAP:
                continue
            score = (
                abs(first_along - preferred)
                + abs(second_along - preferred)
                + 0.35 * along_gap
            )
            candidate_pairs.append((score, first, second))
    if not candidate_pairs:
        debug.update({
            "reason": "no_pair_near_leg",
            "axis": "y" if vertical else "x",
            "first_side": sorted(first_side),
            "second_side": sorted(second_side),
            "current": current,
            "direction": direction,
        })
        return None, debug

    _score, first, second = min(candidate_pairs, key=lambda item: item[0])
    debug.update({
        "reason": "selected",
        "axis": "y" if vertical else "x",
        "candidates": len(candidate_pairs),
    })
    return {
        "balls": (first, second),
        "axis": "y" if vertical else "x",
        "sides": side_names,
    }, debug


def _radar_front_pair_at_node(active_balls, from_node, node):
    """Return the unhit pair immediately ahead of a node along its arrival leg."""
    dx = node[0] - from_node[0]
    dy = node[1] - from_node[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None

    balls = {(float(x), float(y)) for x, y in active_balls}
    vertical = abs(dy) >= abs(dx)
    if vertical:
        direction = 1.0 if dy >= 0.0 else -1.0
        lane = node[0]
        left_columns = sorted({x for x, _ in balls if x < lane - S2_RADAR_CORRIDOR_MARGIN})
        right_columns = sorted({x for x, _ in balls if x > lane + S2_RADAR_CORRIDOR_MARGIN})
        if not left_columns or not right_columns:
            return None
        left_x = left_columns[-1]
        right_x = right_columns[0]
        rows = sorted(
            {
                y for x, y in balls
                if x in (left_x, right_x)
                and direction * (y - node[1]) > S2_RADAR_CORRIDOR_MARGIN
            },
            key=lambda y: direction * (y - node[1]),
        )
        for row_y in rows:
            pair = ((left_x, row_y), (right_x, row_y))
            if pair[0] in balls and pair[1] in balls:
                return {"balls": pair, "axis": "y"}
        return None

    direction = 1.0 if dx >= 0.0 else -1.0
    lane = node[1]
    lower_rows = sorted({y for _, y in balls if y < lane - S2_RADAR_CORRIDOR_MARGIN})
    upper_rows = sorted({y for _, y in balls if y > lane + S2_RADAR_CORRIDOR_MARGIN})
    if not lower_rows or not upper_rows:
        return None
    lower_y = lower_rows[-1]
    upper_y = upper_rows[0]
    columns = sorted(
        {
            x for x, y in balls
            if y in (lower_y, upper_y)
            and direction * (x - node[0]) > S2_RADAR_CORRIDOR_MARGIN
        },
        key=lambda x: direction * (x - node[0]),
    )
    for column_x in columns:
        pair = ((column_x, lower_y), (column_x, upper_y))
        if pair[0] in balls and pair[1] in balls:
            return {"balls": pair, "axis": "x"}
    return None


class _RadarPoseCorrector:
    def __init__(self, yellow_balls=None):
        self._yellow_balls = {
            (float(x), float(y)) for x, y in (yellow_balls or ())
        }
        self._last_scan_seq = None
        self._last_apply_t = 0.0
        self._pair_id = None
        self._filtered_error = None
        self._last_status_key = None
        self._last_status_printed_key = None
        self._last_status_print_t = 0.0
        self._last_correction_pair = None
        self._last_correction_print_t = 0.0
        self._partial_match_pairs = set()
        self._reacquire_samples = {}
        self._reacquire_announced_pairs = set()
        self._reacquire_confirming_pair = None
        self._single_guard = None
        self._last_guard_key = None

    def reset(self):
        self._last_scan_seq = None
        self._pair_id = None
        self._filtered_error = None
        self._last_status_key = None
        self._last_status_printed_key = None
        self._last_status_print_t = 0.0
        self._last_correction_pair = None
        self._last_correction_print_t = 0.0
        self._partial_match_pairs = set()
        self._reacquire_samples = {}
        self._reacquire_announced_pairs = set()
        self._reacquire_confirming_pair = None
        self._single_guard = None
        self._last_guard_key = None

    def walk_speed_scale(self):
        if self._reacquire_confirming_pair is not None:
            return S2_RADAR_REACQUIRE_SPEED_SCALE
        return 1.0

    def single_ball_guard(self):
        guard = self._single_guard
        if guard is None or time.time() - guard["time"] > S2_RADAR_SINGLE_GUARD_STALE_SECS:
            return 1.0, 0.0
        return guard["speed_scale"], guard["vy"]

    def _ball_label(self, ball):
        ball = (float(ball[0]), float(ball[1]))
        column = min(range(len(GRID_X)), key=lambda index: abs(GRID_X[index] - ball[0])) + 1
        row = min(range(len(GRID_Y)), key=lambda index: abs(GRID_Y[index] - ball[1])) + 1
        color = "黄球" if ball in self._yellow_balls else "蓝球"
        return f"第{row}排第{column}列{color}"

    @staticmethod
    def _route_label(from_node, to_node, axis):
        if axis == "y":
            direction = "+y" if to_node[1] >= from_node[1] else "-y"
            return f"沿 x={from_node[0]:.2f} 的 {direction} 路径"
        direction = "+x" if to_node[0] >= from_node[0] else "-x"
        return f"沿 y={from_node[1]:.2f} 的 {direction} 路径"

    def _report_status(self, label, pair_info, from_node, to_node, observations):
        pair = pair_info["balls"]
        pair_id = tuple(sorted(pair))
        matched = {item["landmark"] for item in observations}
        matched_balls = [ball for ball in pair if ball in matched]
        if len(matched_balls) == 2:
            message = (
                f"[stage2] radar {label}: {self._route_label(from_node, to_node, pair_info['axis'])}，"
                f"匹配 {self._ball_label(pair[0])}、{self._ball_label(pair[1])}"
            )
            state = (pair_id, "two")
        elif len(matched_balls) == 1:
            missing = pair[1] if matched_balls[0] == pair[0] else pair[0]
            message = (
                f"[stage2] radar {label}: 只匹配到 {self._ball_label(matched_balls[0])}，"
                f"{self._ball_label(missing)} 未匹配"
            )
            state = (pair_id, "one", matched_balls[0])
        else:
            message = (
                f"[stage2] radar {label}: 未匹配到 "
                f"{self._ball_label(pair[0])}、{self._ball_label(pair[1])}"
            )
            state = (pair_id, "none")

        self._last_status_key = state
        now = time.time()
        if state == self._last_status_printed_key:
            return
        if now - self._last_status_print_t < 0.8:
            return
        self._last_status_printed_key = state
        self._last_status_print_t = now
        print(message, flush=True)

    def _report_correction(self, label, pair_info, from_node, to_node, step, note=None):
        now = time.time()
        pair_id = tuple(sorted(pair_info["balls"]))
        if (
            pair_id == self._last_correction_pair
            and now - self._last_correction_print_t < 1.0
        ):
            return
        self._last_correction_pair = pair_id
        self._last_correction_print_t = now
        route = self._route_label(from_node, to_node, pair_info["axis"])
        if note is not None:
            print(f"[stage2] radar {label}: {route}，{note}", flush=True)
            return
        print(
            f"[stage2] radar {label}: {route}；位姿回写 "
            f"dx={step[0]:+.3f}m dy={step[1]:+.3f}m dyaw={step[2]:+.1f}deg",
            flush=True,
        )

    def _update_single_lane_assist(self, label, ball, observation, adapter,
                                   from_node, to_node):
        forward = float(observation["body_x"])
        if forward <= 0.0 or forward > S2_RADAR_SINGLE_LANE_RANGE:
            self._single_guard = None
            self._last_guard_key = None
            return

        dx = to_node[0] - from_node[0]
        dy = to_node[1] - from_node[1]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return
        vertical = abs(dy) >= abs(dx)
        target_cross = from_node[0] if vertical else from_node[1]
        yaw_rad = math.radians(adapter.get_yaw_deg())
        body_x = float(observation["body_x"])
        body_y = float(observation["body_y"])
        world_offset_x = math.cos(yaw_rad) * body_x - math.sin(yaw_rad) * body_y
        world_offset_y = math.sin(yaw_rad) * body_x + math.cos(yaw_rad) * body_y
        estimated_cross = (
            ball[0] - world_offset_x if vertical
            else ball[1] - world_offset_y
        )
        cross_error = target_cross - estimated_cross
        world_cross_speed = _clamp(
            S2_RADAR_SINGLE_LANE_K * cross_error,
            -S2_RADAR_SINGLE_LANE_MAX_VY,
            S2_RADAR_SINGLE_LANE_MAX_VY,
        )
        if vertical:
            guard_vy = -math.sin(yaw_rad) * world_cross_speed
        else:
            guard_vy = math.cos(yaw_rad) * world_cross_speed
        self._single_guard = {
            "time": time.time(),
            "speed_scale": 1.0,
            "vy": guard_vy,
        }
        key = (ball, round(cross_error, 2))
        if key != self._last_guard_key:
            self._last_guard_key = key
            print(
                f"[stage2] radar single: 仅见 {self._ball_label(ball)}，"
                f"按走廊中线辅助横移 err={cross_error:+.2f}m vy={guard_vy:+.2f}",
                flush=True,
            )

    @staticmethod
    def _pair_pose_error(adapter, pair, observations):
        by_landmark = {item["landmark"]: item for item in observations}
        first_obs = by_landmark[pair[0]]
        second_obs = by_landmark[pair[1]]
        body_dx = second_obs["body_x"] - first_obs["body_x"]
        body_dy = second_obs["body_y"] - first_obs["body_y"]
        field_dx = pair[1][0] - pair[0][0]
        field_dy = pair[1][1] - pair[0][1]
        measured_pair_spacing = math.hypot(body_dx, body_dy)
        expected_pair_spacing = math.hypot(field_dx, field_dy)
        if math.hypot(body_dx, body_dy) < 0.10:
            return None, measured_pair_spacing, expected_pair_spacing

        measured_yaw = _wrap_deg(
            math.degrees(math.atan2(field_dy, field_dx)
                         - math.atan2(body_dy, body_dx))
        )
        measured_yaw_rad = math.radians(measured_yaw)
        cos_yaw = math.cos(measured_yaw_rad)
        sin_yaw = math.sin(measured_yaw_rad)
        robot_estimates = []
        for ball, observation in ((pair[0], first_obs), (pair[1], second_obs)):
            world_body_x = (
                cos_yaw * observation["body_x"]
                - sin_yaw * observation["body_y"]
            )
            world_body_y = (
                sin_yaw * observation["body_x"]
                + cos_yaw * observation["body_y"]
            )
            robot_estimates.append((ball[0] - world_body_x, ball[1] - world_body_y))
        measured_x = sum(item[0] for item in robot_estimates) / len(robot_estimates)
        measured_y = sum(item[1] for item in robot_estimates) / len(robot_estimates)
        robot_x, robot_y, _ = adapter.get_position()
        raw_error = (
            measured_x - robot_x,
            measured_y - robot_y,
            _yaw_err(measured_yaw, adapter.get_yaw_deg()),
        )
        return raw_error, measured_pair_spacing, expected_pair_spacing

    def _needs_reacquire_confirmation(self, pair_id, raw_error):
        if pair_id not in self._partial_match_pairs:
            if self._reacquire_confirming_pair == pair_id:
                self._reacquire_confirming_pair = None
            return False

        self._reacquire_confirming_pair = pair_id
        samples = self._reacquire_samples.setdefault(pair_id, deque())
        samples.append(raw_error)
        while len(samples) > S2_RADAR_REACQUIRE_CONFIRM_FRAMES:
            samples.popleft()
        if pair_id not in self._reacquire_announced_pairs:
            self._reacquire_announced_pairs.add(pair_id)
            print(
                "[stage2] radar: 单球后恢复双球匹配，"
                f"等待 {S2_RADAR_REACQUIRE_CONFIRM_FRAMES} 帧一致结果再校正，"
                f"行走降至 {S2_RADAR_REACQUIRE_SPEED_SCALE:.0%}",
                flush=True,
            )
        if len(samples) < S2_RADAR_REACQUIRE_CONFIRM_FRAMES:
            return True

        reference = samples[0]
        consistent = all(
            math.hypot(sample[0] - reference[0], sample[1] - reference[1])
            <= S2_RADAR_REACQUIRE_MAX_POSE_DELTA
            and abs(_yaw_err(sample[2], reference[2]))
            <= S2_RADAR_REACQUIRE_MAX_YAW_DELTA_DEG
            for sample in samples
        )
        if not consistent:
            newest = samples[-1]
            self._reacquire_samples[pair_id] = deque([newest])
            return True

        self._partial_match_pairs.discard(pair_id)
        self._reacquire_samples.pop(pair_id, None)
        self._reacquire_announced_pairs.discard(pair_id)
        self._reacquire_confirming_pair = None
        self._filtered_error = None
        return False

    def update(self, adapter, detector, active_balls, from_node, to_node, label="walk"):
        if detector is None or not active_balls:
            return None
        robot_x, robot_y, _ = adapter.get_position()
        pair_info, pair_debug = _radar_corridor_pair(
            active_balls, from_node, to_node, robot_x, robot_y
        )
        if pair_info is None:
            return None
        pair = pair_info["balls"]
        pair_id = tuple(sorted(pair))
        if pair_id != self._pair_id:
            self._pair_id = pair_id
            self._filtered_error = None
        observations, scan_debug = detector.get_lidar_landmark_observations(
            robot_x,
            robot_y,
            adapter.get_yaw_deg(),
            pair,
            min_distance=S2_RADAR_MIN_DISTANCE,
            max_distance=S2_RADAR_MAX_DISTANCE,
            max_match_error=S2_RADAR_MAX_MATCH_ERROR,
            return_debug=True,
        )
        scan_seq = scan_debug.get("scan_seq", 0)
        if scan_seq == self._last_scan_seq:
            return None
        self._last_scan_seq = scan_seq
        by_landmark = {item["landmark"]: item for item in observations}
        self._report_status(label, pair_info, from_node, to_node, observations)
        matched_count = sum(ball in by_landmark for ball in pair)
        if matched_count == 1:
            matched_ball = next(ball for ball in pair if ball in by_landmark)
            self._update_single_lane_assist(
                label,
                matched_ball,
                by_landmark[matched_ball],
                adapter,
                from_node,
                to_node,
            )
            self._partial_match_pairs.add(pair_id)
            self._reacquire_samples.pop(pair_id, None)
            self._reacquire_announced_pairs.discard(pair_id)
            if self._reacquire_confirming_pair == pair_id:
                self._reacquire_confirming_pair = None
        else:
            self._single_guard = None
            self._last_guard_key = None
        if matched_count != 2:
            return None
        raw_error, measured_pair_spacing, expected_pair_spacing = self._pair_pose_error(
            adapter, pair, observations
        )
        if raw_error is None:
            return None
        if abs(measured_pair_spacing - expected_pair_spacing) > S2_RADAR_PAIR_SPACING_TOL:
            self._report_correction(
                label, pair_info, from_node, to_node, None,
                note=(
                    f"两球间距 {measured_pair_spacing:.2f}m，"
                    f"应为 {expected_pair_spacing:.2f}m，本帧不回写"
                ),
            )
            return None
        if (
            math.hypot(raw_error[0], raw_error[1]) > S2_RADAR_MAX_POSE_ERROR
            or abs(raw_error[2]) > S2_RADAR_MAX_YAW_ERROR_DEG
        ):
            self._report_correction(
                label, pair_info, from_node, to_node, None,
                note="两球定位结果异常，本帧不回写",
            )
            return None

        if self._needs_reacquire_confirmation(pair_id, raw_error):
            return None

        if self._filtered_error is None:
            filtered_error = raw_error
        else:
            previous = self._filtered_error
            alpha = S2_RADAR_FILTER_ALPHA
            filtered_error = tuple(
                alpha * current + (1.0 - alpha) * prior
                for current, prior in zip(raw_error, previous)
            )
        self._filtered_error = filtered_error

        now = time.time()
        if now - self._last_apply_t < S2_RADAR_APPLY_COOLDOWN:
            return None
        step_x = _clamp(
            S2_RADAR_APPLY_GAIN * filtered_error[0],
            -S2_RADAR_MAX_STEP,
            S2_RADAR_MAX_STEP,
        )
        step_y = _clamp(
            S2_RADAR_APPLY_GAIN * filtered_error[1],
            -S2_RADAR_MAX_STEP,
            S2_RADAR_MAX_STEP,
        )
        step_yaw = _clamp(
            S2_RADAR_APPLY_GAIN * filtered_error[2],
            -S2_RADAR_MAX_YAW_STEP_DEG,
            S2_RADAR_MAX_YAW_STEP_DEG,
        )
        if pair_info["axis"] == "x":
            step_y = _clamp(
                step_y,
                -S2_RADAR_CROSS_TRACK_MAX_STEP,
                S2_RADAR_CROSS_TRACK_MAX_STEP,
            )
        else:
            step_x = _clamp(
                step_x,
                -S2_RADAR_CROSS_TRACK_MAX_STEP,
                S2_RADAR_CROSS_TRACK_MAX_STEP,
            )
        if (
            math.hypot(step_x, step_y) < S2_RADAR_MIN_STEP
            and abs(step_yaw) < S2_RADAR_MIN_YAW_STEP_DEG
        ):
            return None

        try:
            adapter.set_mapped_pose(
                robot_x + step_x,
                robot_y + step_y,
                _wrap_deg(adapter.get_yaw_deg() + step_yaw),
                quiet=True,
            )
        except TypeError:
            adapter.set_mapped_pose(
                robot_x + step_x,
                robot_y + step_y,
                _wrap_deg(adapter.get_yaw_deg() + step_yaw),
            )
        self._last_apply_t = now
        self._report_correction(
            label, pair_info, from_node, to_node, (step_x, step_y, step_yaw),
        )
        return step_x, step_y, step_yaw


def s2_calibrate_node_from_front_pair(adapter, detector, active_balls,
                                      radar_corrector, from_node, node):
    """Use the first unhit pair beyond an arrival node before turning or hitting."""
    if detector is None or radar_corrector is None or not active_balls:
        return False
    pair_info = _radar_front_pair_at_node(active_balls, from_node, node)
    if pair_info is None:
        return False

    pair = pair_info["balls"]
    expected_spacing = math.hypot(
        pair[1][0] - pair[0][0], pair[1][1] - pair[0][1]
    )
    print(
        f"[stage2] node_cal at=({node[0]:.2f},{node[1]:.2f}) front="
        f"{radar_corrector._ball_label(pair[0])}、{radar_corrector._ball_label(pair[1])}",
        flush=True,
    )
    samples = deque()
    last_scan_seq = None
    t0 = time.time()
    while time.time() - t0 < S2_RADAR_NODE_CALIBRATION_TIMEOUT:
        x, y, _ = adapter.get_position()
        observations, scan_debug = detector.get_lidar_landmark_observations(
            x,
            y,
            adapter.get_yaw_deg(),
            pair,
            min_distance=S2_RADAR_MIN_DISTANCE,
            max_distance=S2_RADAR_MAX_DISTANCE,
            max_match_error=S2_RADAR_MAX_MATCH_ERROR,
            return_debug=True,
        )
        scan_seq = scan_debug.get("scan_seq", 0)
        if scan_seq == last_scan_seq:
            time.sleep(0.03)
            continue
        last_scan_seq = scan_seq
        by_landmark = {item["landmark"]: item for item in observations}
        if any(ball not in by_landmark for ball in pair):
            samples.clear()
            time.sleep(0.03)
            continue
        raw_error, measured_spacing, _ = radar_corrector._pair_pose_error(
            adapter, pair, observations
        )
        if (
            raw_error is None
            or abs(measured_spacing - expected_spacing) > S2_RADAR_PAIR_SPACING_TOL
            or math.hypot(raw_error[0], raw_error[1]) > S2_RADAR_NODE_MAX_POSE_ERROR
            or abs(raw_error[2]) > S2_RADAR_NODE_MAX_YAW_ERROR_DEG
        ):
            samples.clear()
            time.sleep(0.03)
            continue
        samples.append(raw_error)
        while len(samples) > S2_RADAR_NODE_STABLE_FRAMES:
            samples.popleft()
        if len(samples) < S2_RADAR_NODE_STABLE_FRAMES:
            time.sleep(0.03)
            continue

        reference = samples[0]
        stable = all(
            math.hypot(sample[0] - reference[0], sample[1] - reference[1])
            <= S2_RADAR_NODE_STABLE_POS_DELTA
            and abs(_yaw_err(sample[2], reference[2]))
            <= S2_RADAR_NODE_STABLE_YAW_DELTA_DEG
            for sample in samples
        )
        if not stable:
            newest = samples[-1]
            samples.clear()
            samples.append(newest)
            time.sleep(0.03)
            continue

        correction = tuple(
            sum(sample[index] for sample in samples) / len(samples)
            for index in range(3)
        )
        x, y, _ = adapter.get_position()
        yaw = adapter.get_yaw_deg()
        adapter.set_mapped_pose(
            x + correction[0],
            y + correction[1],
            _wrap_deg(yaw + correction[2]),
            quiet=True,
        )
        radar_corrector.reset()
        print(
            f"[stage2] node_cal applied at=({node[0]:.2f},{node[1]:.2f}) "
            f"dx={correction[0]:+.3f}m dy={correction[1]:+.3f}m "
            f"dyaw={correction[2]:+.1f}deg",
            flush=True,
        )
        return True

    print(
        f"[stage2] node_cal skipped at=({node[0]:.2f},{node[1]:.2f}) "
        "front pair was not stable",
        flush=True,
    )
    return False


def s2_calibrate_upper_left_front_pair(adapter, detector, active_balls,
                                       radar_corrector):
    """Calibrate at the upper-left four-ball center while facing field +y."""
    if detector is None or radar_corrector is None or not active_balls:
        return False

    pair_info = _radar_front_pair_at_node(
        active_balls,
        S2_UPPER_LEFT_FRONT_FROM,
        S2_UPPER_LEFT_CENTER,
    )
    if pair_info is None:
        print(
            "[stage2] upper_left_node_cal skipped: front pair unavailable",
            flush=True,
        )
        return False

    pair = pair_info["balls"]
    print(
        f"[stage2] upper_left_node_cal at=({S2_UPPER_LEFT_CENTER[0]:.2f},"
        f"{S2_UPPER_LEFT_CENTER[1]:.2f}) facing +y front="
        f"{radar_corrector._ball_label(pair[0])}, "
        f"{radar_corrector._ball_label(pair[1])}",
        flush=True,
    )
    if not s2_align_yaw(
        adapter,
        90.0,
        label="upper_left_front_pair",
        tol=S2_LEG_ALIGN_TOL,
    ):
        print("[stage2] upper_left_node_cal skipped: cannot face +y", flush=True)
        return False

    return s2_calibrate_node_from_front_pair(
        adapter,
        detector,
        active_balls,
        radar_corrector,
        S2_UPPER_LEFT_FRONT_FROM,
        S2_UPPER_LEFT_CENTER,
    )


def _lane_keep(adapter, target_yaw, target_lat):
    x, y, _ = adapter.get_position()
    yaw = adapter.get_yaw_deg()

    if abs(abs(target_yaw) - 90.0) < 1.0:
        err = target_lat - x
        vy_sign = -1.0 if target_yaw > 0.0 else 1.0
    else:
        err = target_lat - y
        vy_sign = 1.0 if abs(target_yaw) < 90.0 else -1.0

    vy = _clamp(vy_sign * S2_LANE_K * err, -S2_LANE_MAX_VY, S2_LANE_MAX_VY)
    wz = _clamp(S2_K_YAW * _yaw_err(target_yaw, yaw), -S2_MAX_WZ, S2_MAX_WZ)
    return vy, wz


def s2_walk_segment(adapter, from_node, to_node, detector=None,
                    radar_landmarks=None, radar_corrector=None):
    dx = to_node[0] - from_node[0]
    dy = to_node[1] - from_node[1]
    seg_len = math.hypot(dx, dy)
    if seg_len < 1e-6:
        return True
    target_yaw = math.degrees(math.atan2(dy, dx))
    target_lat = from_node[0] if abs(dy) > abs(dx) else from_node[1]
    ux = dx / seg_len
    uy = dy / seg_len

    print(
        f"[stage2] leg ({from_node[0]:.2f},{from_node[1]:.2f})"
        f" -> ({to_node[0]:.2f},{to_node[1]:.2f}) yaw={target_yaw:.1f}",
        flush=True,
    )
    if not s2_align_yaw(
        adapter, target_yaw, label="leg_start", tol=S2_LEG_ALIGN_TOL
    ):
        adapter.stop()
        print(
            f"[stage2] WARN: leg yaw not ready; skip walking to "
            f"({to_node[0]:.2f},{to_node[1]:.2f})",
            flush=True,
        )
        return False

    t0 = time.time()
    while time.time() - t0 < S2_WP_TIMEOUT:
        walk_speed_scale = 1.0
        guard_speed_scale = 1.0
        guard_vy = 0.0
        if radar_corrector is not None:
            radar_corrector.update(
                adapter, detector, radar_landmarks, from_node, to_node,
                label=f"leg_{from_node[0]:.2f}_{from_node[1]:.2f}",
            )
            walk_speed_scale = radar_corrector.walk_speed_scale()
            guard_speed_scale, guard_vy = radar_corrector.single_ball_guard()
        x, y, _ = adapter.get_position()
        progress = (x - from_node[0]) * ux + (y - from_node[1]) * uy
        dist = math.hypot(to_node[0] - x, to_node[1] - y)
        if dist < S2_WP_TOL or progress >= seg_len - S2_OVERSHOOT_MARGIN:
            adapter.stop()
            print(
                f"[stage2] leg reached target=({to_node[0]:.2f},{to_node[1]:.2f}) "
                f"{_pose_text(adapter)} dist={dist:.2f} progress={progress:.2f}/{seg_len:.2f}",
                flush=True,
            )
            return True
        yaw_err = _yaw_err(target_yaw, adapter.get_yaw_deg())
        if abs(yaw_err) > S2_MOVE_MAX_YAW_ERR:
            adapter.stop()
            print(
                f"[stage2] WARN: leg yaw drift err={yaw_err:.1f}; realign before moving",
                flush=True,
            )
            if not s2_align_yaw(
                adapter, target_yaw, label="leg_realign", tol=S2_LEG_ALIGN_TOL
            ):
                return False
        vy, wz = _lane_keep(adapter, target_yaw, target_lat)
        speed_scale = min(walk_speed_scale, guard_speed_scale)
        vy = _clamp(
            vy * speed_scale + guard_vy,
            -S2_LANE_MAX_VY,
            S2_LANE_MAX_VY,
        )
        adapter.walk(S2_WALK_VX * speed_scale, vy, wz)
        time.sleep(0.07)

    adapter.stop()
    print(f"[stage2] walk_segment timeout target=({to_node[0]:.2f},{to_node[1]:.2f})", flush=True)
    return False


def s2_axis_adjust(adapter, target_x=None, target_y=None, timeout=8.0, tol=0.035,
                   detector=None, radar_landmarks=None, radar_corrector=None):
    x, y, _ = adapter.get_position()
    if target_x is not None:
        target_yaw = 0.0 if target_x >= x else 180.0
        target_lat = y
        axis = "x"
    elif target_y is not None:
        target_yaw = 90.0 if target_y >= y else -90.0
        target_lat = x
        axis = "y"
    else:
        return True

    err = (target_x - x) if axis == "x" else (target_y - y)
    if abs(err) <= tol:
        print(
            f"[stage2] axis_adjust skip axis={axis} err={err:.3f} tol={tol:.3f} "
            f"{_pose_text(adapter)}",
            flush=True,
        )
        return True

    print(
        f"[stage2] axis_adjust axis={axis} err={err:.3f} yaw={target_yaw:.1f}",
        flush=True,
    )
    if not s2_align_yaw(adapter, target_yaw, label=f"axis_{axis}"):
        return False
    prev_err = err
    t0 = time.time()
    while time.time() - t0 < timeout:
        x, y, _ = adapter.get_position()
        err = (target_x - x) if axis == "x" else (target_y - y)
        if abs(err) <= tol:
            adapter.stop()
            return True
        if prev_err * err < 0.0:
            adapter.stop()
            print(
                f"[stage2] axis_adjust crossed axis={axis} err={err:.3f} "
                f"{_pose_text(adapter)}",
                flush=True,
            )
            return True
        prev_err = err
        yaw_err = _yaw_err(target_yaw, adapter.get_yaw_deg())
        if abs(yaw_err) > S2_MOVE_MAX_YAW_ERR:
            adapter.stop()
            print(
                f"[stage2] WARN: axis_adjust yaw drift axis={axis} err={yaw_err:.1f}",
                flush=True,
            )
            if not s2_align_yaw(adapter, target_yaw, label=f"axis_{axis}_realign"):
                return False
        speed = _clamp(abs(err) * 0.8, 0.05, 0.10)
        vy, wz = _lane_keep(adapter, target_yaw, target_lat)
        adapter.walk(speed, vy, wz)
        time.sleep(0.07)

    adapter.stop()
    print(f"[stage2] axis_adjust timeout axis={axis}", flush=True)
    return False


def s2_refine_path_node(adapter, node, detector=None, radar_landmarks=None,
                        radar_corrector=None):
    return (
        s2_axis_adjust(
            adapter, target_x=node[0], detector=detector,
            radar_landmarks=radar_landmarks, radar_corrector=radar_corrector,
        )
        and s2_axis_adjust(
            adapter, target_y=node[1], detector=detector,
            radar_landmarks=radar_landmarks, radar_corrector=radar_corrector,
        )
    )


def s2_navigate(adapter, end_node, final_yaw=None, detector=None,
                radar_landmarks=None, radar_corrector=None):
    x, y, _ = adapter.get_position()
    cur_node = snap_to_path(x, y)
    end_node = snap_to_path(*end_node)
    route = path_bfs(cur_node, end_node)
    labels = "->".join(f"({node[0]:.2f},{node[1]:.2f})" for node in route)
    print(f"[stage2] nav {labels}", flush=True)

    if len(route) == 1:
        if route[0] == S2_UPPER_LEFT_CENTER:
            s2_calibrate_upper_left_front_pair(
                adapter,
                detector,
                radar_landmarks,
                radar_corrector,
            )
        dist = math.hypot(end_node[0] - x, end_node[1] - y)
        if dist > S2_WP_TOL:
            return s2_refine_path_node(
                adapter, end_node, detector=detector,
                radar_landmarks=radar_landmarks, radar_corrector=radar_corrector,
            )
        if final_yaw is not None:
            return s2_align_yaw(adapter, final_yaw, label="final")
        return True

    legs = []
    i = 0
    while i < len(route) - 1:
        j = i
        while (
            j + 1 < len(route) - 1
            and _direction_key(route[j], route[j + 1])
            == _direction_key(route[j + 1], route[j + 2])
            and route[j + 1] != S2_UPPER_LEFT_CENTER
        ):
            j += 1
        legs.append((route[i], route[j + 1]))
        i = j + 1

    if route[0] == S2_UPPER_LEFT_CENTER:
        s2_calibrate_upper_left_front_pair(
            adapter,
            detector,
            radar_landmarks,
            radar_corrector,
        )

    ok = True
    for leg_from, leg_to in legs:
        if not s2_walk_segment(
            adapter, leg_from, leg_to, detector=detector,
            radar_landmarks=radar_landmarks, radar_corrector=radar_corrector,
        ):
            ok = False
            break
        if leg_to == S2_UPPER_LEFT_CENTER:
            s2_calibrate_upper_left_front_pair(
                adapter,
                detector,
                radar_landmarks,
                radar_corrector,
            )
        else:
            s2_calibrate_node_from_front_pair(
                adapter,
                detector,
                radar_landmarks,
                radar_corrector,
                leg_from,
                leg_to,
            )

    # The next leg's lateral controller removes small intermediate errors while
    # moving. Refining every corner would add two in-place turns per corner.
    if ok and not s2_refine_path_node(
        adapter, end_node, detector=detector,
        radar_landmarks=radar_landmarks, radar_corrector=radar_corrector,
    ):
        ok = False

    if final_yaw is not None:
        ok = s2_align_yaw(adapter, final_yaw, label="final") and ok
    return ok


def full_scan_and_locate(adapter, detector):
    grid_votes = {pos: {"orange": 0, "blue": FIXED_BLUE_PRIOR} for pos in FIXED_BLUE}
    start_yaw = adapter.get_yaw_deg()
    n_steps = int(SCAN_DEG / SCAN_STEP_DEG) + 1
    print(f"[stage2] scan steps={n_steps} step_deg={SCAN_STEP_DEG}")

    for i in range(n_steps):
        target_yaw = ((start_yaw + i * SCAN_STEP_DEG) + 180.0) % 360.0 - 180.0
        adapter.align_yaw(target_yaw, tol=3.0, timeout=5.0)
        time.sleep(0.20)

        rx, ry, _ = adapter.get_position()
        yaw_rad = math.radians(adapter.get_yaw_deg())

        def record(angle_off, tag):
            dist = detector.get_distance_at_angle(angle_off)
            if dist is None:
                return
            wx = rx + dist * math.cos(yaw_rad + angle_off)
            wy = ry + dist * math.sin(yaw_rad + angle_off)
            sx, sy = snap_to_grid(wx, wy)
            votes = grid_votes.setdefault((sx, sy), {"orange": 0, "blue": 0})
            votes[tag] += 1

        for angle_off, _area in detector.get_orange_detections():
            record(angle_off, "orange")
        for angle_off, _area in detector.get_blue_detections():
            record(angle_off, "blue")

    print("[stage2] grid votes:")
    for sy in GRID_Y:
        for sx in GRID_X:
            votes = grid_votes.get((sx, sy), {"orange": 0, "blue": 0})
            suffix = " fixed_blue=True" if (sx, sy) in FIXED_BLUE else ""
            print(
                f"[stage2] grid=({sx:.2f},{sy:.2f}) "
                f"orange={votes.get('orange', 0)} blue={votes.get('blue', 0)}{suffix}"
            )

    selected = select_orange_balls(grid_votes)
    print(f"[stage2] selected balls n={len(selected)}")
    for conf, (sx, sy) in selected:
        votes = grid_votes.get((sx, sy), {})
        print(
            f"[stage2] ball=({sx:.2f},{sy:.2f}) conf={conf:.2f} "
            f"orange={votes.get('orange', 0)} blue={votes.get('blue', 0)}"
        )
    return [pos for _conf, pos in selected]


def _save_probe_debug(detector, tag, info, final=False):
    if info is None or not hasattr(detector, "save_debug_image"):
        return
    now = time.time()
    if not final:
        last_write = PROBE_DEBUG_LAST_WRITE.get(tag, 0.0)
        if now - last_write < PROBE_DEBUG_SAVE_INTERVAL_SECS:
            return
        PROBE_DEBUG_LAST_WRITE[tag] = now
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        debug_dir = os.path.join(base_dir, S2_DEBUG_DIR)
        os.makedirs(debug_dir, exist_ok=True)
        suffix = "" if final else "_live"
        path = os.path.join(debug_dir, f"{tag}{suffix}.jpg")
        expected = float(info.get("expected_angle_deg", 0.0))
        title = (
            f"{tag} {'final' if final else 'live'} "
            f"result={info.get('color') or 'none'} exp={expected:+.1f}"
        )
        if detector.save_debug_image(path, info, title=title) and final:
            print(f"[stage2] probe {tag} debug_image={path}", flush=True)
    except Exception as exc:
        print(f"[stage2] probe {tag} debug_image_error={exc}", flush=True)


def _probe_unique_center_candidates(info):
    """Deduplicate contour/Hough detections of the same physical ball."""
    candidates = []
    for item in sorted(
            info.get("center_candidates") or (),
            key=lambda candidate: candidate.get("score", 0.0),
            reverse=True,
    ):
        if item.get("color") not in ("orange", "blue"):
            continue
        if (
                item.get("area", 0.0) < PROBE_LEFT_PAIR_MIN_AREA
                or item.get("radius", 0.0) < PROBE_LEFT_PAIR_MIN_RADIUS):
            continue
        duplicate = False
        for chosen in candidates:
            distance = math.hypot(
                float(item["cx"]) - float(chosen["cx"]),
                float(item["cy"]) - float(chosen["cy"]),
            )
            merge_radius = max(
                8.0,
                0.55 * min(float(item["radius"]), float(chosen["radius"])),
            )
            if distance <= merge_radius:
                duplicate = True
                break
        if not duplicate:
            candidates.append(item)
    return candidates


def _probe_left_ball_of_adjacent_pair(info):
    """Return the left ball of the r1c1/r1c2 image pair when it is visible."""
    rgb = info.get("rgb")
    if rgb is None:
        return None
    height = float(rgb.shape[0])
    image_center_y = 0.5 * height
    center_band = PROBE_LEFT_PAIR_CENTER_BAND_RATIO * height
    max_vertical_gap = PROBE_LEFT_PAIR_MAX_VERTICAL_GAP_RATIO * height
    expected_angle = float(info.get("expected_angle_deg", 0.0))
    candidates = [
        item for item in _probe_unique_center_candidates(info)
        if abs(float(item["cy"]) - image_center_y) <= center_band
    ]

    best_pair = None
    for first_index, first in enumerate(candidates):
        for second in candidates[first_index + 1:]:
            if abs(float(first["cy"]) - float(second["cy"])) > max_vertical_gap:
                continue
            left, right = sorted((first, second), key=lambda item: float(item["cx"]))
            if float(right["cx"]) - float(left["cx"]) < PROBE_LEFT_PAIR_MIN_HORIZONTAL_GAP_PX:
                continue
            midpoint_error = abs(
                0.5 * (float(left["angle_deg"]) + float(right["angle_deg"]))
                - expected_angle
            )
            if midpoint_error > PROBE_LEFT_PAIR_MAX_MIDPOINT_ERR_DEG:
                continue
            pair_score = (
                midpoint_error,
                abs(float(left["cy"]) - float(right["cy"])),
                -min(float(left["area"]), float(right["area"])),
            )
            if best_pair is None or pair_score < best_pair[0]:
                best_pair = (pair_score, left, right, midpoint_error)

    if best_pair is None:
        return None
    _score, left, right, midpoint_error = best_pair
    return {
        "left": left,
        "right": right,
        "midpoint_error_deg": midpoint_error,
    }


def _probe_record_color(adapter, detector, target_pos, tag, votes,
                        prefer_left_pair=False):
    rx, ry, _ = adapter.get_position()
    yaw_rad = math.radians(adapter.get_yaw_deg())
    expected_angle = math.atan2(target_pos[1] - ry, target_pos[0] - rx)
    expected_off = (expected_angle - yaw_rad + math.pi) % (2.0 * math.pi) - math.pi
    info = detector.classify_center_ball(
        center_window_deg=PROBE_CENTER_WINDOW_DEG,
        margin=PROBE_COLOR_MARGIN,
        expected_angle_deg=math.degrees(expected_off),
        target_window_deg=PROBE_TARGET_WINDOW_DEG,
    )
    pair = _probe_left_ball_of_adjacent_pair(info) if prefer_left_pair else None
    if pair is not None:
        info = dict(info)
        info["selected"] = pair["left"]
        info["color"] = pair["left"]["color"]
        info["note"] = "left_ball_of_adjacent_pair"
        info["probe_left_pair"] = pair
    color = info["color"]
    if color in ("orange", "blue"):
        votes[color] += 1
    else:
        votes["none"] += 1
    _save_probe_debug(detector, tag, info)
    return info


def _probe_color_probability(votes):
    total = votes["orange"] + votes["blue"]
    if total <= 0:
        return 0.5, 0.0
    orange_prob = float(votes["orange"]) / float(total)
    confidence = max(orange_prob, 1.0 - orange_prob)
    return orange_prob, confidence


def _sample_probe_color(adapter, detector, target_pos, name, votes, seconds,
                        prefer_left_pair=False):
    last_info = None
    left_pair_samples = 0
    deadline = time.time() + seconds
    while time.time() < deadline:
        last_info = _probe_record_color(
            adapter,
            detector,
            target_pos,
            name,
            votes,
            prefer_left_pair=prefer_left_pair,
        )
        if last_info.get("probe_left_pair") is not None:
            left_pair_samples += 1
        time.sleep(PROBE_SAMPLE_DT)
    return last_info, left_pair_samples


def _fmt_center_debug(info):
    def fmt(item):
        if not item:
            return "none"
        return (
            f"ang={item['angle_deg']:.1f}deg area={item['area']:.0f} "
            f"circ={item['circularity']:.2f} fill={item['fill_ratio']:.2f} "
            f"asp={item['aspect']:.2f} hr={item['height_ratio']:.2f} "
            f"sat={item['mean_sat']:.0f} val={item['mean_val']:.0f} "
            f"score={item['score']:.1f}"
        )

    pair = info.get("probe_left_pair")
    pair_text = ""
    if pair is not None:
        pair_text = (
            f" left_pair={pair['left']['color']}@{pair['left']['angle_deg']:.1f}deg"
            f" right={pair['right']['color']}@{pair['right']['angle_deg']:.1f}deg"
        )
    return (
        f"yellow[{fmt(info.get('orange'))}] blue[{fmt(info.get('blue'))}] "
        f"note={info.get('note')}{pair_text}"
    )


def _probe_from_upper_left_center(adapter, detector, target_pos, name):
    """Take the C probe from the calibrated upper-left four-ball center."""
    radar_corrector = _RadarPoseCorrector()
    radar_landmarks = set(BOARD_BALLS)
    print(
        f"[stage2] probe {name} fallback: face -x and walk to "
        f"({S2_PROBE_LEFT_TRAVERSE[0]:.2f},{S2_PROBE_LEFT_TRAVERSE[1]:.2f}); "
        "radar keeps the row1/row2 column2 ball pair available",
        flush=True,
    )
    if not s2_navigate(
            adapter,
            S2_PROBE_LEFT_TRAVERSE,
            detector=detector,
            radar_landmarks=radar_landmarks,
            radar_corrector=radar_corrector):
        return False
    print(
        f"[stage2] probe {name} fallback: face +y and walk to upper-left "
        f"center ({S2_UPPER_LEFT_CENTER[0]:.2f},{S2_UPPER_LEFT_CENTER[1]:.2f})",
        flush=True,
    )
    if not s2_navigate(
            adapter,
            S2_UPPER_LEFT_CENTER,
            final_yaw=90.0,
            detector=detector,
            radar_landmarks=radar_landmarks,
            radar_corrector=radar_corrector):
        return False

    x, y, _ = adapter.get_position()
    target_yaw = math.degrees(math.atan2(target_pos[1] - y, target_pos[0] - x))
    print(
        f"[stage2] probe {name} fallback observe "
        f"target=({target_pos[0]:.2f},{target_pos[1]:.2f}) yaw={target_yaw:.1f}",
        flush=True,
    )
    return s2_align_yaw(adapter, target_yaw, label=f"probe_{name}_fallback")


def probe_one_cell(adapter, detector, name, row, col):
    target_pos = grid_cell(row, col)
    x, y, _ = adapter.get_position()
    target_yaw = math.degrees(math.atan2(target_pos[1] - y, target_pos[0] - x))
    print(
        f"[stage2] probe {name} row={row} col={col} "
        f"pos=({target_pos[0]:.2f},{target_pos[1]:.2f}) yaw={target_yaw:.1f}"
    )
    adapter.align_yaw(target_yaw, tol=2.5, timeout=6.0)
    time.sleep(0.25)

    votes = {"orange": 0, "blue": 0, "none": 0}
    prefer_left_pair = name == PROBE_LEFT_PAIR_NAME
    last_info, left_pair_samples = _sample_probe_color(
        adapter,
        detector,
        target_pos,
        name,
        votes,
        PROBE_SAMPLE_SECS,
        prefer_left_pair=prefer_left_pair,
    )
    used_fallback = False
    if prefer_left_pair and left_pair_samples < PROBE_LEFT_PAIR_MIN_SAMPLES:
        print(
            f"[stage2] probe {name} adjacent pair unavailable "
            f"samples={left_pair_samples}/{PROBE_LEFT_PAIR_MIN_SAMPLES}; "
            "move to upper-left calibrated observation point",
            flush=True,
        )
        if _probe_from_upper_left_center(adapter, detector, target_pos, name):
            used_fallback = True
            votes = {"orange": 0, "blue": 0, "none": 0}
            time.sleep(0.25)
            last_info, _ignored_pair_samples = _sample_probe_color(
                adapter,
                detector,
                target_pos,
                name,
                votes,
                PROBE_SAMPLE_SECS,
            )
        else:
            print(
                f"[stage2] WARN: probe {name} fallback navigation failed; "
                "retain primary observation votes",
                flush=True,
            )
    orange_prob, conf = _probe_color_probability(votes)
    total = votes["orange"] + votes["blue"]
    if total < PROBE_MIN_COLOR_VOTES or conf < PROBE_MIN_CONFIDENCE:
        print(
            f"[stage2] probe {name} uncertain; retry votes="
            f"yellow={votes['orange']} blue={votes['blue']} none={votes['none']} "
            f"p_yellow={orange_prob:.2f}",
            flush=True,
        )
        for yaw_offset in PROBE_RETRY_YAW_OFFSETS_DEG:
            adapter.align_yaw(target_yaw + yaw_offset, tol=2.5, timeout=3.0)
            time.sleep(0.12)
            retry_info, _ignored_pair_samples = _sample_probe_color(
                adapter,
                detector,
                target_pos,
                name,
                votes,
                PROBE_RETRY_SAMPLE_SECS,
                prefer_left_pair=prefer_left_pair and not used_fallback,
            )
            if retry_info is not None:
                last_info = retry_info
            orange_prob, conf = _probe_color_probability(votes)
            total = votes["orange"] + votes["blue"]
            if total >= PROBE_MIN_COLOR_VOTES and conf >= PROBE_MIN_CONFIDENCE:
                break

    color = "yellow" if orange_prob >= 0.5 else "blue"
    print(
        f"[stage2] probe {name} result={color} p_yellow={orange_prob:.2f} "
        f"conf={conf:.2f} yellow={votes['orange']} blue={votes['blue']} "
        f"none={votes['none']}",
        flush=True,
    )
    if last_info is not None:
        _save_probe_debug(detector, name, last_info, final=True)
        print(f"[stage2] probe {name} center {_fmt_center_debug(last_info)}")
    return {
        "orange_prob": orange_prob,
        "confidence": conf,
        "votes": dict(votes),
    }


def rule_scan_and_locate(adapter, detector):
    print(f"[stage2] go rule observe point {S2_RULE_SCAN}")
    # The stage1 transition already leaves the dog at this node facing +y.
    # Let the required probes make their own turns, without an extra 90-degree
    # turn before the first probe.
    if not s2_navigate(adapter, S2_RULE_SCAN):
        print("[stage2] WARN: rule observe point navigation reported timeout")

    probe_results = {}
    for name, (row, col) in PROBE_CELLS.items():
        probe_results[name] = probe_one_cell(adapter, detector, name, row, col)

    print(
        "[stage2] probe bits "
        + " ".join(
            f"{name}={'yellow' if result['orange_prob'] >= 0.5 else 'blue'}"
            for name, result in probe_results.items()
        )
    )
    balls = infer_orange_balls_from_probe_probabilities(probe_results)
    if balls:
        print_inferred_board(balls)
    return balls


def hit_ball(adapter, approach_node, ball_pos, detector=None, hit_index=None,
             radar_landmarks=None, radar_corrector=None,
             direct_visual_approach_tol=None):
    bx, by = ball_pos
    px, py = approach_node
    print(f"[stage2] hit ball=({bx:.2f},{by:.2f}) via node=({px:.2f},{py:.2f})", flush=True)
    print(f"[stage2] hit begin {_pose_text(adapter)}", flush=True)
    x, y, _ = adapter.get_position()
    node_error = math.hypot(px - x, py - y)
    if (
            direct_visual_approach_tol is not None
            and node_error <= direct_visual_approach_tol):
        print(
            f"[stage2] hit direct_visual_approach node_error={node_error:.2f}m "
            "skip axis settle; visual ball alignment will set the heading",
            flush=True,
        )
    elif not s2_navigate(
            adapter, approach_node, detector=detector,
            radar_landmarks=radar_landmarks, radar_corrector=radar_corrector):
        print(
            f"[stage2] WARN: approach navigation failed for "
            f"ball=({bx:.2f},{by:.2f}) node=({px:.2f},{py:.2f})",
            flush=True,
        )
        return False
    x, y, _ = adapter.get_position()
    hit_yaw = math.degrees(math.atan2(by - y, bx - x))
    print(
        f"[stage2] hit at_node {_pose_text(adapter)} "
        f"target_yaw={hit_yaw:.1f} ball_dist={math.hypot(bx - x, by - y):.2f}",
        flush=True,
    )
    use_visual = (
        detector is not None
        and hit_index is not None
        and hit_index >= S2_VISUAL_REFINE_START_INDEX
    )
    if use_visual:
        refined = s2_visual_refine_yaw(
            adapter,
            detector,
            hit_yaw,
            label=f"hit{hit_index}",
        )
        if refined is None:
            print(
                f"[stage2] WARN: visual confirmation unavailable for hit{hit_index}; "
                f"use coordinate heading={hit_yaw:.1f} to hit ball=({bx:.2f},{by:.2f})",
                flush=True,
            )
            s2_align_yaw(
                adapter,
                hit_yaw,
                label=f"hit{hit_index}_coordinate_fallback",
                tol=2.0,
                timeout=4.0,
            )
        else:
            refined_yaw, visual_item = refined
            hit_yaw = refined_yaw
            if visual_item is None:
                print(
                    f"[stage2] visual hit{hit_index} use last-seen heading={hit_yaw:.1f}",
                    flush=True,
                )
                s2_align_yaw(
                    adapter,
                    hit_yaw,
                    label=f"hit{hit_index}_last_seen",
                    tol=2.0,
                    timeout=4.0,
                )
            else:
                anchored_yaw = s2_anchor_yaw_from_centered_ball(
                    adapter,
                    ball_pos,
                    approach_node,
                    visual_item,
                    label=f"hit{hit_index}",
                )
                if anchored_yaw is not None:
                    hit_yaw = anchored_yaw
                print(
                    f"[stage2] visual hit heading area={visual_item.get('area', 0.0):.0f} "
                    f"r={visual_item.get('radius', 0.0):.1f}",
                    flush=True,
                )
    else:
        if not s2_align_yaw(adapter, hit_yaw, label="hit_target"):
            print(
                f"[stage2] WARN: hit yaw not ready for ball=({bx:.2f},{by:.2f})",
                flush=True,
            )
            return False

    drive_time = (S2_HIT_DRIVE_DIST / S2_APPROACH_VX) * S2_HIT_DRIVE_RATIO
    est_dist = S2_APPROACH_VX * drive_time
    print(
        f"[stage2] hit drive vx={S2_APPROACH_VX:.2f} time={drive_time:.2f}s "
        f"est_dist={est_dist:.2f} ratio={S2_HIT_DRIVE_RATIO:.2f}",
        flush=True,
    )
    t0 = time.time()
    while time.time() - t0 < drive_time:
        err = _yaw_err(hit_yaw, adapter.get_yaw_deg())
        wz = _clamp(S2_K_YAW * err, -S2_MAX_WZ, S2_MAX_WZ)
        adapter.walk(S2_APPROACH_VX, 0.0, wz)
        time.sleep(0.07)
    adapter.stop()
    print(f"[stage2] hit after_drive {_pose_text(adapter)}", flush=True)

    start_back_x, start_back_y, _ = adapter.get_position()
    back_target_x = px
    back_target_y = py
    back_dx = back_target_x - start_back_x
    back_dy = back_target_y - start_back_y
    back_dist = math.hypot(back_dx, back_dy)
    if back_dist > 1e-6:
        back_ux = back_dx / back_dist
        back_uy = back_dy / back_dist
    else:
        back_ux = 0.0
        back_uy = 0.0

    max_back_time = max(1.0, back_dist / max(S2_RETURN_VX, 1e-6) + S2_RETURN_EXTRA_TIME)
    print(
        f"[stage2] hit back target_node=({px:.2f},{py:.2f}) "
        f"target=({back_target_x:.2f},{back_target_y:.2f}) "
        f"start_dist={back_dist:.2f} max_time={max_back_time:.2f}s",
        flush=True,
    )

    prev_dist = back_dist
    t0 = time.time()
    while time.time() - t0 < max_back_time:
        x, y, _ = adapter.get_position()
        dist = math.hypot(back_target_x - x, back_target_y - y)
        progress = (x - start_back_x) * back_ux + (y - start_back_y) * back_uy
        if dist <= S2_RETURN_TOL:
            print(
                f"[stage2] hit back reached dist={dist:.2f} "
                f"progress={progress:.2f}/{back_dist:.2f}",
                flush=True,
            )
            break
        if progress >= back_dist - S2_RETURN_OVERSHOOT_MARGIN:
            print(
                f"[stage2] hit back passed target dist={dist:.2f} "
                f"progress={progress:.2f}/{back_dist:.2f}",
                flush=True,
            )
            break
        if progress > 0.5 * back_dist and dist > prev_dist + 0.05:
            print(
                f"[stage2] hit back distance increasing dist={dist:.2f} "
                f"prev={prev_dist:.2f} progress={progress:.2f}/{back_dist:.2f}",
                flush=True,
            )
            break
        prev_dist = dist
        err = _yaw_err(hit_yaw, adapter.get_yaw_deg())
        wz = _clamp(S2_K_YAW * err, -S2_MAX_WZ, S2_MAX_WZ)
        adapter.walk(-S2_RETURN_VX, 0.0, wz)
        time.sleep(0.07)
    adapter.stop()
    print(f"[stage2] hit after_back {_pose_text(adapter)} target_node=({px:.2f},{py:.2f})", flush=True)
    print(
        f"[stage2] hit return ready {_pose_text(adapter)} "
        f"target_node=({px:.2f},{py:.2f}); defer node refinement to next leg",
        flush=True,
    )
    return True


def run_stage2(adapter, detector=None):
    """Run stage 2 using a new or already-running RGB/lidar detector.

    Supplying ``detector`` lets an enclosing route keep the RGB/lidar ROS
    subscriptions alive across stage boundaries.  This avoids creating a
    second concurrently spinning node on the dog.
    """
    if not HAS_ROS:
        print("[stage2] ERROR: rclpy/OrangeDetector unavailable; source ROS first")
        return False

    print("[stage2] start")
    own_ros = False
    own_detector = detector is None
    if own_detector:
        if not rclpy.ok():
            rclpy.init()
            own_ros = True
        detector = OrangeDetector()
        detector.start()
    pose_stop = None
    pose_thread = None

    try:
        if not detector.wait_ready(timeout=5.0):
            print("[stage2] ERROR: image_rgb not ready")
            return False
        lidar_deadline = time.time() + 5.0
        while time.time() < lidar_deadline and getattr(detector, "_scan_seq", 0) <= 0:
            time.sleep(0.05)
        if getattr(detector, "_scan_seq", 0) <= 0:
            print("[stage2] ERROR: lidar scan not ready")
            return False

        time.sleep(WARMUP_SECS)
        x, y, _ = adapter.get_position()
        print(f"[stage2] pos=({x:.2f},{y:.2f}) yaw={adapter.get_yaw_deg():.1f}")
        pose_stop, pose_thread = start_pose_logger(adapter, interval=3.0, prefix="[stage2]")

        balls = rule_scan_and_locate(adapter, detector)
        if not balls:
            print("[stage2] no ball selected")
            return False

        # Every grid cell contains a ball.  Radar may use either colour while
        # walking, but a yellow ball is removed immediately after it is hit.
        active_landmarks = set(BOARD_BALLS)
        radar_corrector = _RadarPoseCorrector()
        print(
            f"[stage2] radar unhit balls n={len(active_landmarks)} "
            f"range={S2_RADAR_MIN_DISTANCE:.2f}-{S2_RADAR_MAX_DISTANCE:.2f}m",
            flush=True,
        )

        plan_start_x, plan_start_y, _ = adapter.get_position()
        print(
            f"[stage2] plan from current probe pose "
            f"({plan_start_x:.2f},{plan_start_y:.2f})",
            flush=True,
        )
        plan = s2_plan_hits(balls, (plan_start_x, plan_start_y))
        print(f"[stage2] hit plan n={len(plan)}")
        for ball, node in plan:
            print(f"[stage2] plan ball=({ball[0]:.2f},{ball[1]:.2f}) node=({node[0]:.2f},{node[1]:.2f})")

        all_hit_ok = True
        for hit_index, (ball, node) in enumerate(plan, start=1):
            if not hit_ball(
                adapter, node, ball, detector=detector, hit_index=hit_index,
                radar_landmarks=active_landmarks, radar_corrector=radar_corrector,
                direct_visual_approach_tol=(
                    S2_FIRST_HIT_DIRECT_APPROACH_TOL if hit_index == 1 else None
                ),
            ):
                all_hit_ok = False
                print(
                    f"[stage2] WARN: stop hit sequence after failed ball="
                    f"({ball[0]:.2f},{ball[1]:.2f})",
                    flush=True,
                )
                break
            active_landmarks.discard(ball)
            radar_corrector.reset()

        print(f"[stage2] go end {S2_END}")
        end_ok = True
        if all_hit_ok:
            end_ok = s2_navigate(
                adapter, S2_END, final_yaw=90.0, detector=detector,
                radar_landmarks=active_landmarks, radar_corrector=radar_corrector,
            )
        if not end_ok:
            print(f"[stage2] WARN: cannot reach stage2 exit {_pose_text(adapter)}")
        print("[stage2] done")
        return all_hit_ok and end_ok
    except Exception as exc:
        import traceback
        print(f"[stage2] ERROR: {exc}")
        traceback.print_exc()
        return False
    finally:
        if pose_stop is not None:
            pose_stop.set()
        if pose_thread is not None:
            pose_thread.join(timeout=1.0)
        if own_detector:
            detector.stop()
        if own_ros:
            try:
                rclpy.shutdown()
            except Exception:
                pass
