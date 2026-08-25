#!/usr/bin/env python3
"""
Measure the lateral distance from the dog to a line on the ground, using the
left/right fisheye cameras via MEI back-projection + ground-plane intersection.

Calibration is loaded at runtime from the dog's factory calibration:
    /params/camera/calibration/
      params_intrinsic.yaml         cam4=left, cam5=right (omni/MEI)
      params_extrinsic.yaml         T_cn_c0
      params_bodyimu_extrinsic.yaml cam0 T_cam_imu
The camera pose in body frame is  T_camN_body = inv(T_cam_imu) @ inv(T_cn_c0).
If loading fails, a built-in fallback (a past factory unit) is used.

Line detection is dual:
  - colour fish-eye (e.g. 247, format=1 RGB): HSV yellow in the lower ROI
  - grayscale fish-eye (e.g. 98, format=2): adaptive brightness threshold

Usage:
  python3 fisheye_line_distance.py left  left.jpg  --body-height 0.26 --debug dbg.png
  python3 fisheye_line_distance.py right right.jpg --body-height 0.26 \
          --calib-dir /params/camera/calibration --roi-bottom 0.5
  python3 fisheye_line_distance.py right right.jpg --body-height 0.235 \
          --line-mode body_x --debug filtered.png
"""

import os
import sys
import numpy as np
import cv2

# ---------------- built-in fallback (a past factory unit, dog 98) ----------
CAMS_FALLBACK = {
    "left": {
        "xi": 1.090313559782818,
        "fx": 452.7006749176067, "fy": 452.96923793226307,
        "cx": 247.58816096382975, "cy": 203.35060671610714,
        "D": [-0.0924754784733309, 0.07514047254439235,
              0.0004139572090225899, -0.0008044889417039669],
        "R": np.array([[0.9995911766617918, -0.007776810227915737, 0.027513646842537355],
                       [-0.02767737282612739, -0.021794105250699625, 0.9993792973690054],
                       [-0.007172347826229599, -0.9997322332499851, -0.02200043698791384]]),
        "t": np.array([-0.056491036274741946, 0.09046013368948155, -0.015134488358812759]),
    },
    "right": {
        "xi": 1.1168230601834885,
        "fx": 458.33957317479457, "fy": 458.92329881799,
        "cx": 250.38622748112894, "cy": 203.15301832794748,
        "D": [-0.08183965210987922, 0.0762966583460183,
              0.0006543674014488295, 0.0006162770066776136],
        "R": np.array([[-0.9994420003894251, 0.0019527924026576983, -0.033344781592005726],
                       [0.03339552186097854, 0.038890345609724944, -0.9986852758191598],
                       [-0.0006534349388719238, -0.9992415762067701, -0.03893385933339242]]),
        "t": np.array([-0.06303590287800859, -0.09453467949670537, -0.014527939295042452]),
    },
}


def _invT(m):
    out = np.eye(4)
    out[:3, :3] = m[:3, :3].T
    out[:3, 3] = -m[:3, :3].T @ m[:3, 3]
    return out


def load_calibration(base_dir="/params/camera/calibration"):
    """Read the dog's factory calibration and build {left,right} camera models."""
    try:
        import yaml
        with open(os.path.join(base_dir, "params_intrinsic.yaml")) as f:
            intr = yaml.safe_load(f)
        with open(os.path.join(base_dir, "params_extrinsic.yaml")) as f:
            extr = yaml.safe_load(f)
        with open(os.path.join(base_dir, "params_bodyimu_extrinsic.yaml")) as f:
            body = yaml.safe_load(f)

        out = {}
        for name, cam in (("left", "cam4"), ("right", "cam5")):
            ii = intr[cam]
            xi, fx, fy, cx, cy = ii["intrinsics"]
            D = ii["distortion_coeffs"]
            D = [D[0], D[1], D[3], D[4]]                 # drop index-2 placeholder
            Tb = _invT(np.array(body["cam0"]["T_cam_imu"])) @ _invT(
                np.array(extr[cam]["T_cn_c0"]))
            out[name] = {
                "xi": xi, "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                "D": D,
                "R": Tb[:3, :3], "t": Tb[:3, 3],
            }
        print(f"[load_calibration] loaded from {base_dir}")
        for k, c in out.items():
            print(f"  {k}: fx={c['fx']:.1f} fy={c['fy']:.1f} "
                  f"cx={c['cx']:.1f} cy={c['cy']:.1f} xi={c['xi']:.3f}")
        return out
    except Exception as e:
        print(f"[load_calibration] FAILED ({e}), using built-in fallback")
        return {k: dict(v, R=v["R"].copy(), t=v["t"].copy()) for k, v in CAMS_FALLBACK.items()}


CAMS = load_calibration()


def _radtan_distort(p, pu):
    k1, k2, pa, pb = p["D"]
    r2 = pu[0] * pu[0] + pu[1] * pu[1]
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    xd = pu[0] * radial + 2 * pa * pu[0] * pu[1] + pb * (r2 + 2 * pu[0] * pu[0])
    yd = pu[1] * radial + pa * (r2 + 2 * pu[1] * pu[1]) + 2 * pb * pu[0] * pu[1]
    return np.array([xd, yd])


def _radtan_jacobian(p, pu):
    k1, k2, pa, pb = p["D"]
    x, y = pu
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    J11 = radial + x * x * (2 * k1 + 4 * k2 * r2) + 2 * pa * y + 6 * pb * x
    J12 = 2 * k1 * x * y + 4 * k2 * x * y * r2 + 2 * pa * x + 2 * pb * y
    J21 = 2 * k1 * x * y + 4 * k2 * x * y * r2 + 2 * pa * y + 2 * pb * x
    J22 = radial + y * y * (2 * k1 + 4 * k2 * r2) + 2 * pb * x + 6 * pa * y
    return np.array([[J11, J12], [J21, J22]])


def mei_backproject(p, u, v):
    """MEI/omni pixel -> unit ray in the camera frame (incl. radtan undistort)."""
    mx = (u - p["cx"]) / p["fx"]
    my = (v - p["cy"]) / p["fy"]
    m2 = mx * mx + my * my
    xi = p["xi"]
    disc = xi * xi - (m2 + 1.0) * (xi * xi - 1.0)
    t = (xi + np.sqrt(max(disc, 0.0))) / (m2 + 1.0)
    ps = np.array([mx * t, my * t, t - xi])
    pd = ps[:2] / ps[2]
    pu = pd.copy()
    for _ in range(8):
        res = _radtan_distort(p, pu) - pd
        if np.hypot(*res) < 1e-7:
            break
        J = _radtan_jacobian(p, pu)
        det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(det) < 1e-12:
            break
        dx = (res[0] * J[1, 1] - res[1] * J[0, 1]) / det
        dy = (res[1] * J[0, 0] - res[0] * J[1, 0]) / det
        pu = pu - 0.5 * np.array([dx, dy])
    ray = np.array([pu[0], pu[1], 1.0])
    return ray / np.linalg.norm(ray)


def ray_to_ground(p, ray_cam, body_height):
    """Camera-frame ray -> ground intersection point in body frame (x, y, 0)."""
    ray_b = p["R"] @ ray_cam
    if ray_b[2] >= -1e-6:
        return None
    lam = (-body_height - p["t"][2]) / ray_b[2]
    return np.array([p["t"][0] + lam * ray_b[0],
                     p["t"][1] + lam * ray_b[1],
                     -body_height])


# default HSV yellow range (BGR fish-eye, dog 247 is colour)
# generous defaults; tune with --h-low/--h-high/--s-min/--v-min at the field
DEFAULT_HSV = {"h_low": 10, "h_high": 50, "s_min": 50, "v_min": 50}


def detect_line_mask(img, roi_bottom=0.5, k=1.6, hsv=None):
    """Detect the yellow line. RGB: HSV; grayscale: adaptive brightness.

    For colour fish-eye (best method): HSV yellow threshold + morphology +
    merge fragments (dilate) + keep the largest merged blob.  Falls back to
    the raw yellow mask if the blob filter would drop a legitimate line.
    """
    h, w = img.shape[:2]
    y0 = int(h * roi_bottom)
    if img.ndim == 3:                                    # colour fish-eye
        hsv = hsv or DEFAULT_HSV
        lower = np.array([hsv["h_low"], hsv["s_min"], hsv["v_min"]], dtype=np.uint8)
        upper = np.array([hsv["h_high"], 255, 255], dtype=np.uint8)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, lower, upper)
        mask[:y0, :] = 0
    else:                                                # grayscale fish-eye
        roi = img[y0:, :]
        thr = float(roi.mean()) + k * float(roi.std())
        mask = cv2.inRange(img, int(thr), 255)
        mask[:y0, :] = 0
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # merge fragments so a glare-broken line still counts as one blob
    dil = cv2.dilate(mask, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dil, 8)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        merged = np.where(lab == big, 255, 0).astype(np.uint8)
        out = cv2.bitwise_and(mask, merged)               # true yellow inside blob
        if out.sum() >= max(20, 0.05 * mask.sum()):
            mask = out
    return mask


def measure(img, side, body_height=0.26, debug_path=None, roi_bottom=0.5,
            thresh_k=1.6, hsv=None, scale=1.0, offset=0.0):
    p = CAMS[side]
    h, w = img.shape[:2]
    mask = detect_line_mask(img, roi_bottom=roi_bottom, k=thresh_k, hsv=hsv)
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        print("[measure] no line found in image")
        return None, mask

    pts = []
    for x, y in zip(xs[::4], ys[::4]):
        ray = mei_backproject(p, float(x), float(y))
        g = ray_to_ground(p, ray, body_height)
        if g is not None:
            pts.append(g)
    if not pts:
        print("[measure] no ground intersection (rays not hitting ground)")
        return None, mask
    G = np.array(pts)

    A = np.column_stack([G[:, 0], np.ones(len(G))])
    a, b = np.linalg.lstsq(A, G[:, 1], rcond=None)[0]
    signed = b / np.sqrt(a * a + 1.0)
    dist = abs(signed)
    dist = max(scale * dist + offset, 0.0)

    print(f"[measure] side={side} body_height={body_height} "
          f"line_pixels={len(xs)} ground_points={len(G)}")
    print(f"[measure] ground line: y = {a:.3f}*x + {b:.3f} "
          f"(x range {G[:,0].min():.2f}..{G[:,0].max():.2f} m)")
    print(f"[measure] lateral distance = {dist:.3f} m "
          f"(signed {signed:+.3f} m; + left / - right)")

    if debug_path:
        if img.ndim == 3:
            dbg = img.copy()
        else:
            dbg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        dbg[mask > 0] = (0, 200, 255)
        cv2.putText(dbg, f"dist={dist:.2f} m ({side})",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(debug_path, dbg)
        print("[measure] debug written to", debug_path)

    return dist, mask


def _mei_backproject_vec(p, u, v):
    """Vectorised MEI back-projection for many pixels (N,)->(N,3) unit rays."""
    mx = (u - p["cx"]) / p["fx"]
    my = (v - p["cy"]) / p["fy"]
    m2 = mx * mx + my * my
    xi = p["xi"]
    disc = xi * xi - (m2 + 1.0) * (xi * xi - 1.0)
    t = (xi + np.sqrt(np.maximum(disc, 0.0))) / (m2 + 1.0)
    psx, psy, psz = mx * t, my * t, t - xi
    pdx, pdy = psx / psz, psy / psz
    k1, k2, pa, pb = p["D"]
    px_, py_ = pdx.copy(), pdy.copy()
    for _ in range(8):
        r2 = px_ * px_ + py_ * py_
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        fxx = px_ * radial + 2 * pa * px_ * py_ + pb * (r2 + 2 * px_ * px_)
        fyy = py_ * radial + pa * (r2 + 2 * py_ * py_) + 2 * pb * px_ * py_
        J11 = radial + px_ * px_ * (2 * k1 + 4 * k2 * r2) + 2 * pa * py_ + 6 * pb * px_
        J12 = 2 * k1 * px_ * py_ + 4 * k2 * px_ * py_ * r2 + 2 * pa * px_ + 2 * pb * py_
        J21 = 2 * k1 * px_ * py_ + 4 * k2 * px_ * py_ * r2 + 2 * pa * py_ + 2 * pb * px_
        J22 = radial + py_ * py_ * (2 * k1 + 4 * k2 * r2) + 2 * pb * px_ + 6 * pa * py_
        det = J11 * J22 - J12 * J21
        rx = (fxx - pdx) * J22 - (fyy - pdy) * J12
        ry = (fyy - pdy) * J11 - (fxx - pdx) * J21
        px_ = px_ - 0.5 * rx / det
        py_ = py_ - 0.5 * ry / det
    n = np.sqrt(px_ * px_ + py_ * py_ + 1.0)
    return np.stack([px_ / n, py_ / n, 1.0 / n], axis=-1)


def _fit_ground_line(G):
    """Fit ``y = a*x + b`` and return its body-centre signed distance."""
    A = np.column_stack([G[:, 0], np.ones(len(G))])
    a, b = np.linalg.lstsq(A, G[:, 1], rcond=None)[0]
    signed = b / np.sqrt(a * a + 1.0)
    return float(a), float(b), float(signed)


def _find_front_branch_corner(mask):
    """Find the image-space corner where a front line joins the right line.

    In the right fish-eye view, a front yellow line arrives from the lower-left
    edge and makes a sharp upward-facing corner with the right boundary line.
    The boundary then continues as the broad arc across the bottom of the
    image.  Detect that corner from the yellow band's column centreline rather
    than using a fixed left-image crop.
    """
    height, width = mask.shape[:2]
    columns = np.flatnonzero((mask > 0).sum(axis=0) >= 3)
    if len(columns) < max(30, width // 5):
        return None

    centres = np.full(width, np.nan, dtype=np.float64)
    for column in columns:
        centres[column] = np.median(np.flatnonzero(mask[:, column] > 0))
    valid = np.flatnonzero(np.isfinite(centres))
    centres = np.interp(np.arange(width), valid, centres[valid])
    smooth = cv2.GaussianBlur(
        centres.reshape(1, -1), (0, 0), sigmaX=5.0,
    ).reshape(-1)

    left_span = max(18, int(width * 0.045))
    right_span = max(45, int(width * 0.16))
    start = left_span
    stop = min(int(width * 0.30), width - right_span)
    best = None
    for corner in range(start, stop):
        # Image y grows downward.  A genuine L corner is visibly above both
        # the incoming left branch and the outgoing right boundary arc.
        rise_from_left = smooth[corner - left_span] - smooth[corner]
        fall_to_right = smooth[corner + right_span] - smooth[corner]
        # The second debug frame has only a short, shallow front branch.  The
        # relevant signal is a local slope reversal, not a large pixel height.
        left_slope = rise_from_left / left_span
        right_slope = fall_to_right / right_span
        if (rise_from_left < 3.0 or fall_to_right < 8.0 or
                left_slope < 0.12 or right_slope < 0.08):
            continue
        score = 2.0 * min(left_slope, right_slope) + 0.015 * (
            rise_from_left + fall_to_right
        )
        if best is None or score > best[0]:
            best = (score, corner)
    return None if best is None else int(best[1])


def _fit_equal_x_bins(G, x_min=-0.25, x_max=0.15, bins=16):
    """Fit from equal-width body-x bins inside a fixed local ground window.

    A yellow line can be visible for a long or short range depending on its
    connection to a front boundary.  Fitting every projected pixel gives the
    longer visible portion more influence.  One robust median per fixed x-bin
    makes the lateral estimate invariant to dropping a remote line segment.
    """
    edges = np.linspace(float(x_min), float(x_max), int(bins) + 1)
    representatives = []
    for index in range(len(edges) - 1):
        include_right = index == len(edges) - 2
        inside = ((G[:, 0] >= edges[index]) &
                  (G[:, 0] <= edges[index + 1] if include_right
                   else G[:, 0] < edges[index + 1]))
        if int(inside.sum()) < 4:
            continue
        representatives.append(np.median(G[inside], axis=0))
    if len(representatives) < max(5, int(bins) // 2):
        return None
    fit_points = np.asarray(representatives)
    a, b, signed = _fit_ground_line(fit_points)
    return {
        "a": a,
        "b": b,
        "signed": signed,
        "points": fit_points,
        "bins": len(fit_points),
        "x_min": float(x_min),
        "x_max": float(x_max),
    }


def _select_body_y_line(G, candidates, max_angle_deg, inlier_threshold,
                        min_span=0.045, max_points=700, trials=700):
    """Fit a near-vertical ground line ``x = a*y + b`` with RANSAC."""
    indices = np.flatnonzero(candidates)
    if len(indices) < 20:
        return None

    max_slope = float(np.tan(np.deg2rad(max(1.0, max_angle_deg))))
    threshold = max(0.005, float(inlier_threshold))
    rng = np.random.RandomState(20260813)
    sample = rng.choice(indices, size=min(len(indices), int(max_points)), replace=False)
    best = None
    for _ in range(min(int(trials), max(80, len(sample) * 2))):
        first, second = rng.choice(sample, size=2, replace=False)
        dy = G[second, 1] - G[first, 1]
        if abs(dy) < 0.04:
            continue
        a = (G[second, 0] - G[first, 0]) / dy
        if abs(a) > max_slope:
            continue
        b = G[first, 0] - a * G[first, 1]
        residual = np.abs(G[:, 0] - (a * G[:, 1] + b)) / np.sqrt(a * a + 1.0)
        inliers = candidates & (residual <= threshold)
        count = int(inliers.sum())
        span = float(np.ptp(G[inliers, 1])) if count else 0.0
        if count < 20 or span < min_span:
            continue
        score = count * min(span / 0.10, 1.0)
        if best is None or score > best[0]:
            best = (score, inliers)

    if best is None:
        return None
    inliers = best[1]
    for _ in range(3):
        A = np.column_stack([G[inliers, 1], np.ones(int(inliers.sum()))])
        a, b = np.linalg.lstsq(A, G[inliers, 0], rcond=None)[0]
        if abs(a) > max_slope:
            return None
        residual = np.abs(G[:, 0] - (a * G[:, 1] + b)) / np.sqrt(a * a + 1.0)
        refined = candidates & (residual <= threshold)
        if int(refined.sum()) < 20 or float(np.ptp(G[refined, 1])) < min_span:
            return None
        if np.array_equal(refined, inliers):
            break
        inliers = refined
    return {"inliers": inliers, "a": float(a), "b": float(b),
            "span": float(np.ptp(G[inliers, 1]))}


def _select_body_x_line(G, max_angle_deg=28.0, inlier_threshold=0.025,
                        band_threshold=None, max_points=900, trials=900,
                        check_front=True):
    """Select the full ground-line band that runs along the body x axis.

    The right boundary line is approximately parallel to the dog's forward
    direction, so in body-ground coordinates it has the form ``y = a*x + b``
    with a small slope.  A front boundary line is approximately perpendicular
    to it and is rejected by that direction constraint.  RANSAC first finds
    a narrow centre core, then expands it to the complete physical paint band.
    This keeps the fit robust without treating the edge of the same right
    boundary line as a rejected line.
    """
    if len(G) < 20:
        return None

    max_slope = float(np.tan(np.deg2rad(max(1.0, max_angle_deg))))
    threshold = max(0.005, float(inlier_threshold))
    band_threshold = (max(0.060, 3.0 * threshold) if band_threshold is None
                      else max(threshold, float(band_threshold)))
    rng = np.random.RandomState(20260812)
    sample_count = min(len(G), int(max_points))
    sample = rng.choice(len(G), size=sample_count, replace=False)
    trials = min(int(trials), max(80, sample_count * 2))

    best = None
    for _ in range(trials):
        first, second = rng.choice(sample, size=2, replace=False)
        dx = G[second, 0] - G[first, 0]
        if abs(dx) < 0.04:
            continue
        a = (G[second, 1] - G[first, 1]) / dx
        if abs(a) > max_slope:
            continue
        b = G[first, 1] - a * G[first, 0]
        residual = np.abs(G[:, 1] - (a * G[:, 0] + b)) / np.sqrt(a * a + 1.0)
        inliers = residual <= threshold
        count = int(inliers.sum())
        if count < 20:
            continue
        span = float(np.ptp(G[inliers, 0]))
        if span < 0.18:
            continue
        score = count * min(span / 0.55, 1.0)
        if best is None or score > best[0]:
            best = (score, inliers)

    if best is None:
        return None

    inliers = best[1]
    for _ in range(3):
        a, b, _ = _fit_ground_line(G[inliers])
        if abs(a) > max_slope:
            return None
        residual = np.abs(G[:, 1] - (a * G[:, 0] + b)) / np.sqrt(a * a + 1.0)
        refined = residual <= threshold
        if int(refined.sum()) < 20 or float(np.ptp(G[refined, 0])) < 0.18:
            return None
        if np.array_equal(refined, inliers):
            break
        inliers = refined

    a, b, signed = _fit_ground_line(G[inliers])
    residual = np.abs(G[:, 1] - (a * G[:, 0] + b)) / np.sqrt(a * a + 1.0)
    support = residual <= band_threshold
    if int(support.sum()) < 20 or float(np.ptp(G[support, 0])) < 0.18:
        return None

    # A front boundary joins the side boundary at an L-shaped corner.  It lies
    # on the inner side of the selected side line and is near-vertical after
    # ground projection.  Detect that branch separately, then discard it while
    # retaining overlap pixels for the side line.
    side_sign = 1.0 if signed >= 0.0 else -1.0
    inward = side_sign * (G[:, 1] - (a * G[:, 0] + b)) < -threshold
    front = (_select_body_y_line(
        G, inward, max_angle_deg, threshold,
        max_points=max_points, trials=trials,
    ) if check_front else None)
    rejected_front = np.zeros(len(G), dtype=bool)
    if front is not None:
        front_residual = np.abs(
            G[:, 0] - (front["a"] * G[:, 1] + front["b"])
        ) / np.sqrt(front["a"] * front["a"] + 1.0)
        rejected_front = support & inward & (front_residual <= threshold)
        support &= ~rejected_front

    if int(support.sum()) < 20 or float(np.ptp(G[support, 0])) < 0.18:
        return None
    return {
        "core_inliers": inliers,
        "inliers": support,
        "a": a,
        "b": b,
        "signed": signed,
        "span": float(np.ptp(G[inliers, 0])),
        "support_span": float(np.ptp(G[support, 0])),
        "band_threshold": band_threshold,
        "front": front,
        "rejected_front": rejected_front,
    }


def measure_vectorized(img, side, body_height=0.26, roi_bottom=0.5,
                       thresh_k=1.6, hsv=None, debug_path=None, step=2,
                       scale=1.0, offset=0.0, line_mode="all",
                       max_line_angle_deg=28.0, line_inlier_threshold=0.025,
                       line_band_threshold=None, fit_x_min=-0.25,
                       fit_x_max=0.15, fit_x_bins=16,
                       max_fit_points=900, ransac_trials=900,
                       check_front_branch=True,
                       return_angle=False):
    """Fast numpy version of :func:`measure` for continuous monitoring.

    ``line_mode="all"`` preserves the original behaviour.  ``"body_x"``
    selects only the line parallel to the body forward axis after ground-plane
    projection, excluding a crossing front line from the distance fit.
    """
    def _ret(d, m, ang=None):
        return (d, m, ang) if return_angle else (d, m)
    p = CAMS[side]
    mask = detect_line_mask(img, roi_bottom=roi_bottom, k=thresh_k, hsv=hsv)
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        if debug_path:
            _save_debug(img, mask, None, side, debug_path)
        return _ret(None, mask)
    xs, ys = xs[::step].astype(np.float64), ys[::step].astype(np.float64)
    rays = _mei_backproject_vec(p, xs, ys)
    ray_b = rays @ p["R"].T
    valid = ray_b[:, 2] < -1e-6
    if not valid.any():
        if debug_path:
            _save_debug(img, mask, None, side, debug_path)
        return _ret(None, mask)
    lam = (-body_height - p["t"][2]) / ray_b[valid, 2]
    gx = p["t"][0] + lam * ray_b[valid, 0]
    gy = p["t"][1] + lam * ray_b[valid, 1]
    G = np.column_stack([gx, gy])
    px = np.column_stack([xs[valid].astype(np.int32), ys[valid].astype(np.int32)])
    selected = None
    if line_mode == "body_x":
        all_px = px
        front_corner = _find_front_branch_corner(mask)
        keep = np.ones(len(G), dtype=bool)
        if front_corner is not None:
            keep = px[:, 0] >= front_corner
        fit_G = G[keep]
        fit_px = px[keep]
        selected = _select_body_x_line(
            fit_G,
            max_angle_deg=max_line_angle_deg,
            inlier_threshold=line_inlier_threshold,
            band_threshold=line_band_threshold,
            max_points=max_fit_points,
            trials=ransac_trials,
            check_front=check_front_branch,
        )
        if selected is None:
            if debug_path:
                _save_debug(img, mask, None, side, debug_path,
                            candidate_pixels=all_px,
                            note="no body-x line")
            return _ret(None, mask)
        equal_fit = _fit_equal_x_bins(
            fit_G[selected["inliers"]],
            x_min=fit_x_min,
            x_max=fit_x_max,
            bins=fit_x_bins,
        )
        if equal_fit is None:
            if debug_path:
                _save_debug(img, mask, None, side, debug_path,
                            candidate_pixels=all_px,
                            note="insufficient fixed-window line")
            return _ret(None, mask)
        a, b, signed = equal_fit["a"], equal_fit["b"], equal_fit["signed"]
    elif line_mode == "all":
        a, b, signed = _fit_ground_line(G)
    else:
        raise ValueError("line_mode must be 'all' or 'body_x'")
    dist = abs(signed)
    dist = max(scale * dist + offset, 0.0)
    angle_deg = float(np.rad2deg(np.arctan(a)))
    if debug_path:
        _save_debug(
            img, mask, dist, side, debug_path,
            selected_pixels=(fit_px[selected["inliers"]]
                             if selected is not None else px),
            candidate_pixels=all_px if selected is not None else None,
            note=("body-x %.1fdeg bins=%d x=[%.2f,%.2f]%s%s" %
                  (np.rad2deg(np.arctan(a)), equal_fit["bins"],
                   equal_fit["x_min"], equal_fit["x_max"],
                   " corner-x=%d" % front_corner if front_corner is not None else "",
                   " front-branch rejected" if selected["front"] is not None else ""))
                  if selected is not None else "all line pixels",
        )
    return _ret(dist, mask, angle_deg)


def _save_debug(img, mask, dist, side, debug_path, selected_pixels=None,
                candidate_pixels=None, note=None):
    """Overlay accepted and rejected projected pixels onto the source image."""
    if img.ndim == 3:
        dbg = img.copy()
    else:
        dbg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    dbg[mask > 0] = (0, 150, 255)
    if candidate_pixels is not None and len(candidate_pixels):
        rejected = np.zeros(mask.shape, dtype=np.uint8)
        rejected[candidate_pixels[:, 1], candidate_pixels[:, 0]] = 255
        rejected = cv2.dilate(rejected, np.ones((3, 3), np.uint8))
        dbg[rejected > 0] = (0, 0, 255)
    if selected_pixels is not None and len(selected_pixels):
        accepted = np.zeros(mask.shape, dtype=np.uint8)
        accepted[selected_pixels[:, 1], selected_pixels[:, 0]] = 255
        accepted = cv2.dilate(accepted, np.ones((5, 5), np.uint8))
        dbg[accepted > 0] = (0, 255, 0)
    txt = f"dist={dist:.2f} m ({side})" if dist is not None else f"noline ({side})"
    cv2.putText(dbg, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if note:
        cv2.putText(dbg, note, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (255, 255, 255), 1, cv2.LINE_AA)
    try:
        cv2.imwrite(debug_path, dbg)
    except Exception:
        pass
        if img.ndim == 3:
            dbg = img.copy()
        else:
            dbg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        dbg[mask > 0] = (0, 200, 255)
        cv2.putText(dbg, f"dist={dist:.2f} m ({side})",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(debug_path, dbg)
    return dist, mask


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        return 1
    side, src = args[0].lower(), args[1]
    body_height, debug, roi_bottom, thresh_k, calib_dir = 0.26, None, 0.5, 1.6, None
    scale, offset = 1.0, 0.0
    line_mode, max_line_angle_deg, line_inlier_threshold = "all", 28.0, 0.025
    line_band_threshold = None
    hsv = dict(DEFAULT_HSV)
    for i, a in enumerate(args[2:], start=2):
        if a == "--body-height":
            body_height = float(args[i + 1])
        if a == "--debug":
            debug = args[i + 1]
        if a == "--roi-bottom":
            roi_bottom = float(args[i + 1])
        if a == "--thresh-k":
            thresh_k = float(args[i + 1])
        if a == "--calib-dir":
            calib_dir = args[i + 1]
        if a == "--h-low":
            hsv["h_low"] = int(args[i + 1])
        if a == "--h-high":
            hsv["h_high"] = int(args[i + 1])
        if a == "--s-min":
            hsv["s_min"] = int(args[i + 1])
        if a == "--v-min":
            hsv["v_min"] = int(args[i + 1])
        if a == "--scale":
            scale = float(args[i + 1])
        if a == "--offset":
            offset = float(args[i + 1])
        if a == "--line-mode":
            line_mode = args[i + 1]
        if a == "--max-line-angle-deg":
            max_line_angle_deg = float(args[i + 1])
        if a == "--line-inlier-threshold":
            line_inlier_threshold = float(args[i + 1])
        if a == "--line-band-threshold":
            line_band_threshold = float(args[i + 1])
    if calib_dir:
        CAMS.update(load_calibration(calib_dir))
    if side not in CAMS:
        print("side must be 'left' or 'right'")
        return 1
    img = cv2.imread(src)
    if img is None:
        print("cannot read", src)
        return 1
    if line_mode == "all":
        dist, _ = measure(img, side, body_height, debug, roi_bottom, thresh_k,
                          hsv, scale, offset)
    else:
        dist, _ = measure_vectorized(
            img, side, body_height, roi_bottom, thresh_k, hsv, debug,
            scale=scale, offset=offset, line_mode=line_mode,
            max_line_angle_deg=max_line_angle_deg,
            line_inlier_threshold=line_inlier_threshold,
            line_band_threshold=line_band_threshold,
        )
        print(f"[measure] side={side} line_mode={line_mode} "
              f"lateral distance={dist if dist is not None else 'none'}")
    return 0 if dist is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
