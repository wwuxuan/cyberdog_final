#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read the RealSense depth image and estimate the depth of an RGB detection."""
import threading
import time
import math
import shutil
import subprocess

try:
    import numpy as np
except ImportError:
    np = None

try:
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image
except ImportError:
    rclpy = None
    QoSProfile = None
    ReliabilityPolicy = None
    DurabilityPolicy = None
    Image = None
    CameraInfo = None


DEFAULT_DEPTH_TOPIC = (
    "/mi_desktop_48_b0_2d_60_12_56/camera/depth/image_rect_raw"
)


def _run_ros_command(command, timeout):
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=timeout,
        )
        return result.returncode, (result.stdout or "").strip()
    except Exception as exc:
        return -1, "%s: %s" % (type(exc).__name__, exc)


def ensure_depth_camera(topic=DEFAULT_DEPTH_TOPIC):
    """Idempotently configure and activate the D430 ROS driver."""
    if shutil.which("ros2") is None:
        print("[stage6_depth] ros2 command unavailable; cannot activate D430")
        return False
    marker = "/depth/"
    if marker not in topic:
        print("[stage6_depth] cannot infer camera node from topic=%s" % topic)
        return False
    camera_root = topic.split(marker, 1)[0]
    camera_node = camera_root + "/camera"
    frame_service = camera_root + "/realsense_frame_service"

    _code, state_output = _run_ros_command(
        ["ros2", "lifecycle", "get", camera_node], timeout=5.0)
    state = state_output.lower()
    if state_output.strip().lower().startswith("active"):
        print("[stage6_depth] D430 lifecycle=active node=%s" % camera_node)
        return True
    _run_ros_command([
        "ros2", "service", "call", frame_service,
        "std_srvs/srv/SetBool", "{data: true}",
    ], timeout=8.0)
    if "unconfigured" in state:
        code, output = _run_ros_command(
            ["ros2", "lifecycle", "set", camera_node, "configure"],
            timeout=12.0,
        )
        if code != 0 or "successful" not in output.lower():
            print("[stage6_depth] D430 configure failed: %s" % output)
            return False
        state = "inactive"
    if "inactive" in state:
        code, output = _run_ros_command(
            ["ros2", "lifecycle", "set", camera_node, "activate"],
            timeout=12.0,
        )
        if code != 0 or "successful" not in output.lower():
            print("[stage6_depth] D430 activate failed: %s" % output)
            return False
    _code, state_output = _run_ros_command(
        ["ros2", "lifecycle", "get", camera_node], timeout=5.0)
    active = state_output.strip().lower().startswith("active")
    print("[stage6_depth] D430 lifecycle=%s node=%s" % (
        "active" if active else state_output or "unknown", camera_node))
    return active


class DepthBallTracker(object):
    """Keep the newest D430 depth frame and search near an RGB ball box."""

    def __init__(self, topic=DEFAULT_DEPTH_TOPIC, stale_timeout=0.75):
        self.topic = topic
        self.stale_timeout = stale_timeout
        self._lock = threading.Lock()
        self._latest = None
        self._camera = None
        self._ground_reference = None
        self._last_ramp_eval = (0.0, None)
        self._frames = 0
        self._encoding = None
        self._frame_id = None
        self._valid_ratio = 0.0
        self._median_depth = None
        self._last_measure = {"reason": "not_requested"}
        self._running = True
        self._owns_rclpy = False
        self._warned = False
        self._thread = threading.Thread(target=self._spin, name="stage6-depth")
        self._thread.daemon = True
        self._thread.start()

    def _spin(self):
        if np is None or rclpy is None or Image is None:
            print("[stage6_depth] numpy/rclpy unavailable; use RGB distance fallback")
            return
        node = None
        try:
            if not rclpy.ok():
                rclpy.init(args=None)
                self._owns_rclpy = True
            node = rclpy.create_node("stage6_depth_ball")
            qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )
            node.create_subscription(Image, self.topic, self._on_image, qos)
            node.create_subscription(
                CameraInfo,
                self.topic.rsplit("/", 1)[0] + "/camera_info",
                self._on_camera_info,
                qos,
            )
            print("[stage6_depth] subscribing %s" % self.topic)
            while self._running and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.10)
        except Exception as exc:
            print("[stage6_depth] subscriber unavailable: %s" % exc)
        finally:
            if node is not None:
                try:
                    node.destroy_node()
                except Exception:
                    pass
            try:
                if self._owns_rclpy and rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass

    def _on_image(self, message):
        try:
            encoding = (message.encoding or "").lower()
            if encoding in ("16uc1", "mono16"):
                values = np.frombuffer(message.data, dtype=np.uint16)
                if message.is_bigendian:
                    values = values.byteswap()
                depth = values.reshape((message.height, message.width)).astype(
                    np.float32
                ) * 0.001
            elif encoding in ("32fc1", "32fc"):
                depth = np.frombuffer(message.data, dtype=np.float32).reshape(
                    (message.height, message.width)
                ).copy()
            else:
                if not self._warned:
                    print("[stage6_depth] unsupported encoding=%s" % message.encoding)
                    self._warned = True
                return
            sample = depth[::8, ::8]
            valid = np.isfinite(sample) & (sample >= 0.20) & (sample <= 4.0)
            valid_count = int(np.count_nonzero(valid))
            valid_ratio = float(valid_count) / max(float(valid.size), 1.0)
            median_depth = (
                float(np.median(sample[valid])) if valid_count else None
            )
            with self._lock:
                self._latest = (time.monotonic(), depth, message.width, message.height)
                self._frames += 1
                first_frame = self._frames == 1
                self._encoding = message.encoding
                self._frame_id = message.header.frame_id
                self._valid_ratio = valid_ratio
                self._median_depth = median_depth
            if first_frame:
                print("[stage6_depth] first frame topic=%s frame_id=%s encoding=%s size=%dx%d valid=%.0f%% median=%s" % (
                    self.topic,
                    message.header.frame_id or "(empty)",
                    message.encoding,
                    message.width,
                    message.height,
                    valid_ratio * 100.0,
                    "n/a" if median_depth is None else "%.2fm" % median_depth,
                ))
        except Exception as exc:
            if not self._warned:
                print("[stage6_depth] decode failed: %s" % exc)
                self._warned = True

    def _on_camera_info(self, message):
        try:
            fx, fy = float(message.k[0]), float(message.k[4])
            cx, cy = float(message.k[2]), float(message.k[5])
            if fx > 1.0 and fy > 1.0:
                with self._lock:
                    first_camera = self._camera is None
                    self._camera = (fx, fy, cx, cy)
                if first_camera:
                    print("[stage6_depth] camera_info frame_id=%s fx=%.1f fy=%.1f cx=%.1f cy=%.1f" % (
                        message.header.frame_id or "(empty)", fx, fy, cx, cy))
        except Exception:
            pass

    def status(self):
        with self._lock:
            latest = self._latest
            camera = self._camera
            status = {
                "topic": self.topic,
                "frames": self._frames,
                "encoding": self._encoding,
                "frame_id": self._frame_id,
                "valid_ratio": self._valid_ratio,
                "median_depth": self._median_depth,
                "camera_info": camera is not None,
                "last_measure": dict(self._last_measure),
            }
        status["frame_age"] = (
            None if latest is None else time.monotonic() - latest[0]
        )
        status["ready"] = (
            latest is not None and status["frame_age"] <= self.stale_timeout
        )
        if latest is not None:
            status["width"] = latest[2]
            status["height"] = latest[3]
        else:
            status["width"] = None
            status["height"] = None
        return status

    def wait_ready(self, timeout=4.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.status()["ready"]:
                return True
            time.sleep(0.05)
        return self.status()["ready"]

    def _set_measure_status(self, reason, **details):
        result = {"reason": reason}
        result.update(details)
        with self._lock:
            self._last_measure = result

    @staticmethod
    def _fit_plane(depth, camera, row_start, row_end):
        height, width = depth.shape
        fx, fy, cx, cy = camera
        y0 = max(0, int(height * row_start))
        y1 = min(height, int(height * row_end))
        x0 = int(width * 0.22)
        x1 = int(width * 0.78)
        if y1 <= y0 or x1 <= x0:
            return None
        image = depth[y0:y1:4, x0:x1:4]
        valid = np.isfinite(image) & (image >= 0.25) & (image <= 2.0)
        if np.count_nonzero(valid) < 80:
            return None
        rows, cols = np.indices(image.shape)
        pixel_x = x0 + cols * 4
        pixel_y = y0 + rows * 4
        z = image[valid]
        points = np.column_stack((
            (pixel_x[valid] - cx) * z / fx,
            (pixel_y[valid] - cy) * z / fy,
            z,
        ))
        center = np.mean(points, axis=0)
        _unused, _singular, vectors = np.linalg.svd(points - center, full_matrices=False)
        normal = vectors[-1]
        residual = np.abs(np.dot(points - center, normal))
        inliers = residual <= 0.035
        if np.count_nonzero(inliers) >= 60:
            points = points[inliers]
            center = np.mean(points, axis=0)
            _unused, _singular, vectors = np.linalg.svd(points - center, full_matrices=False)
            normal = vectors[-1]
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-6:
            return None
        return {
            "normal": normal / norm,
            "distance": float(np.median(points[:, 2])),
            "samples": int(points.shape[0]),
        }

    def _snapshot(self):
        with self._lock:
            latest = self._latest
            camera = self._camera
        if latest is None:
            return None
        stamp, depth, _width, _height = latest
        if time.monotonic() - stamp > self.stale_timeout:
            return None
        if camera is None:
            height, width = depth.shape
            focal = 0.60 * width
            camera = (focal, focal, width * 0.5, height * 0.5)
        return stamp, depth, camera

    def capture_ground_reference(self):
        """Capture the normal of the flat Stage 6 floor before moving."""
        snapshot = self._snapshot()
        if snapshot is None:
            return None
        _stamp, depth, camera = snapshot
        reference = self._fit_plane(depth, camera, 0.58, 0.92)
        if reference is not None:
            with self._lock:
                self._ground_reference = reference
            print("[stage6_depth] ground reference samples=%d" % reference["samples"])
        return reference

    def ramp_ahead(self):
        """Return a rising-plane observation ahead of the robot, or None."""
        now = time.monotonic()
        cached_t, cached = self._last_ramp_eval
        if now - cached_t < 0.15:
            return cached
        snapshot = self._snapshot()
        with self._lock:
            reference = self._ground_reference
        result = None
        if snapshot is not None and reference is not None:
            _stamp, depth, camera = snapshot
            ahead = self._fit_plane(depth, camera, 0.38, 0.70)
            if ahead is not None:
                dot = abs(float(np.dot(ahead["normal"], reference["normal"])))
                dot = max(-1.0, min(1.0, dot))
                tilt_deg = math.degrees(math.acos(dot))
                if tilt_deg >= 7.0:
                    result = {
                        "distance": ahead["distance"],
                        "tilt_deg": tilt_deg,
                        "samples": ahead["samples"],
                    }
        self._last_ramp_eval = (now, result)
        return result

    def stop(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def measure(self, detection, expected_distance=None):
        """Return a depth estimate for a normalized RGB box, or None.

        The cameras are only approximately registered, so the search region is
        deliberately expanded around the RGB box. The inner ellipse suppresses
        most background and floor pixels before selecting the near depth cluster.
        """
        if np is None:
            self._set_measure_status("numpy_unavailable")
            return None
        if detection is None:
            self._set_measure_status("no_detection")
            return None
        try:
            x1 = float(detection.get("x1_norm"))
            y1 = float(detection.get("y1_norm"))
            x2 = float(detection.get("x2_norm"))
            y2 = float(detection.get("y2_norm"))
        except (TypeError, ValueError):
            self._set_measure_status("missing_bbox")
            return None
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            self._set_measure_status(
                "invalid_bbox", bbox=(x1, y1, x2, y2))
            return None

        with self._lock:
            latest = self._latest
        if latest is None:
            self._set_measure_status("no_frame")
            return None
        stamp, depth, width, height = latest
        age = time.monotonic() - stamp
        if age > self.stale_timeout:
            self._set_measure_status("stale_frame", age=age)
            return None

        box_w = max(x2 - x1, 0.01)
        box_h = max(y2 - y1, 0.01)
        pad_x = max(0.06, box_w * 0.40)
        pad_y = max(0.06, box_h * 0.40)
        sx1 = max(0, int((x1 - pad_x) * width))
        sy1 = max(0, int((y1 - pad_y) * height))
        sx2 = min(width, int((x2 + pad_x) * width) + 1)
        sy2 = min(height, int((y2 + pad_y) * height) + 1)
        if sx2 <= sx1 or sy2 <= sy1:
            self._set_measure_status("empty_roi")
            return None

        roi = depth[sy1:sy2, sx1:sx2]
        grid_y, grid_x = np.indices(roi.shape)
        center_x = ((x1 + x2) * 0.5 * width) - sx1
        center_y = ((y1 + y2) * 0.5 * height) - sy1
        radius_x = max(box_w * width * 0.65, 4.0)
        radius_y = max(box_h * height * 0.65, 4.0)
        ellipse = (
            ((grid_x - center_x) / radius_x) ** 2
            + ((grid_y - center_y) / radius_y) ** 2
            <= 1.0
        )
        valid = ellipse & np.isfinite(roi) & (roi >= 0.25) & (roi <= 4.0)
        values = roi[valid]
        if values.size < 12:
            self._set_measure_status(
                "too_few_pixels", pixels=int(values.size), age=age)
            return None

        gated_pixels = None
        if expected_distance is not None and expected_distance > 0.0:
            gate = max(0.45, expected_distance * 0.55)
            gated = values[np.abs(values - expected_distance) <= gate]
            gated_pixels = int(gated.size)
            if gated.size >= 8:
                values = gated
        if values.size < 8:
            self._set_measure_status(
                "gate_empty", pixels=int(values.size), age=age)
            return None

        p35 = float(np.percentile(values, 35))
        near_window = max(0.10, p35 * 0.12)
        near = values[values <= p35 + near_window]
        if near.size < 6:
            near = values
        distance = float(np.median(near))
        inlier_window = max(0.12, distance * 0.18)
        support = float(np.count_nonzero(np.abs(values - distance) <= inlier_window)) / max(
            float(values.size), 1.0
        )
        spread = float(np.percentile(near, 85) - np.percentile(near, 15))
        if support < 0.08:
            self._set_measure_status(
                "low_support", support=support, spread=spread,
                pixels=int(values.size), age=age)
            return None
        if spread > max(0.45, distance * 0.55):
            self._set_measure_status(
                "large_spread", support=support, spread=spread,
                pixels=int(values.size), age=age)
            return None
        self._set_measure_status(
            "ok", distance=distance, support=support, spread=spread,
            pixels=int(values.size), gated_pixels=gated_pixels, age=age)
        return {
            "distance": distance,
            "support": support,
            "spread": spread,
            "age": age,
        }

    def front_clearance(self, max_distance=2.5):
        """Measure the central forward sector for boundary/gap following."""
        if np is None:
            return None
        with self._lock:
            latest = self._latest
        if latest is None:
            return None
        stamp, depth, width, height = latest
        age = time.monotonic() - stamp
        if age > self.stale_timeout:
            return None
        x0, x1 = int(width * 0.30), int(width * 0.70)
        y0, y1 = int(height * 0.24), int(height * 0.66)
        roi = depth[y0:y1:2, x0:x1:2]
        valid = np.isfinite(roi) & (roi >= 0.20) & (roi <= max_distance)
        valid_count = int(np.count_nonzero(valid))
        valid_ratio = float(valid_count) / max(float(valid.size), 1.0)
        if valid_count < 20:
            return {
                "distance": None,
                "median": None,
                "valid_ratio": valid_ratio,
                "near_support": 0.0,
                "age": age,
                "samples": valid_count,
            }
        values = roi[valid]
        distance = float(np.percentile(values, 25))
        median = float(np.median(values))
        near_support = float(np.count_nonzero(values <= distance + 0.18)) / max(
            float(values.size), 1.0)
        return {
            "distance": distance,
            "median": median,
            "valid_ratio": valid_ratio,
            "near_support": near_support,
            "age": age,
            "samples": valid_count,
        }
