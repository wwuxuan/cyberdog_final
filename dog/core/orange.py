import math
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, LaserScan


# Stage-2 real balls: orange-red/orange and sky blue.  These deliberately
# exclude the former simulated pale-yellow and dark-blue ball colors.
BALL_HSV_RANGES = {
    "orange": [
        (
            np.array([0, 80, 65], dtype=np.uint8),
            np.array([25, 255, 255], dtype=np.uint8),
        ),
        (
            np.array([165, 55, 45], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        ),
    ],
    "blue": [
        (
            np.array([80, 40, 35], dtype=np.uint8),
            np.array([112, 255, 255], dtype=np.uint8),
        )
    ],
}


class OrangeDetector:
    FOV_H = 1.46608

    LIDAR_MIN_ANG = -1.5700
    LIDAR_MAX_ANG = 1.5700
    LIDAR_BODY_X = 0.21425
    LIDAR_BODY_Y = 0.0
    BALL_RADIUS_M = 0.10

    MIN_BLOB_PX = 60

    def __init__(self, img_topic=None, scan_topic=None):
        self._img_topic = img_topic
        self._active_img_topic = None
        self._scan_topic = scan_topic
        self._img_lock = threading.Lock()
        self._scan_lock = threading.Lock()
        self._img_msg = None
        self._scan_msg = None
        self._scan_seq = 0
        self._node = None
        self._executor = None
        self._thread = None
        self._running = False

    def _choose_topics(self):
        topics = {}
        for _ in range(20):
            topics = dict(self._node.get_topic_names_and_types())
            if any(t.endswith("/image_rgb") for t in topics):
                break
            time.sleep(0.1)
        topic_names = list(topics)

        if self._img_topic is None:
            image_topics = [t for t, types in topics.items()
                            if "sensor_msgs/msg/Image" in types]
            if not image_topics:
                image_topics = [t for t in topic_names if "image" in t]
            preferred = [t for t in image_topics if t.endswith("/image_rgb")]
            self._img_topic = preferred[0] if preferred else "/image_rgb"
        elif not self._img_topic.endswith("/image_rgb"):
            raise ValueError(f"OrangeDetector only supports image_rgb, got {self._img_topic}")

        if self._scan_topic is None:
            scan_topics = [t for t, types in topics.items()
                           if "sensor_msgs/msg/LaserScan" in types]
            if not scan_topics:
                scan_topics = [t for t in topic_names if t.endswith("/scan")]
            prefix = self._img_topic.rsplit("/", 1)[0]
            same_ns_scan = prefix + "/scan"
            if prefix:
                self._scan_topic = same_ns_scan
                return
            preferred = [t for t in scan_topics if t.endswith("/scan")]
            self._scan_topic = preferred[0] if preferred else "/scan"

    def start(self) -> None:
        self._node = rclpy.create_node("orange_detector")
        self._choose_topics()
        print(f"[orange] image_topic={self._img_topic} scan_topic={self._scan_topic}")

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.VOLATILE

        self._node.create_subscription(Image, self._img_topic,
                                       lambda msg: self._img_cb(msg, self._img_topic), qos)
        self._node.create_subscription(LaserScan, self._scan_topic, self._scan_cb, qos)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._running = True
        self._thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._thread.start()

    def _spin_loop(self):
        while self._running and self._node is not None:
            try:
                if not rclpy.ok():
                    break
                self._executor.spin_once(timeout_sec=0.1)
            except Exception:
                if self._running:
                    time.sleep(0.05)
                else:
                    break

    def stop(self) -> None:
        self._running = False
        thread = self._thread
        self._thread = None
        if thread and thread.is_alive():
            thread.join(timeout=1.0)
        node = self._node
        self._node = None
        executor = self._executor
        self._executor = None
        if executor is not None:
            if node is not None:
                try:
                    executor.remove_node(node)
                except Exception:
                    pass
            try:
                executor.shutdown()
            except Exception:
                pass
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass

    def ready(self):
        with self._img_lock:
            image_ready = self._img_msg is not None
        return image_ready

    def wait_ready(self, timeout=4.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.ready():
                return True
            time.sleep(0.05)
        return False

    def _img_cb(self, msg: Image, topic=None) -> None:
        with self._img_lock:
            self._img_msg = msg
            self._active_img_topic = topic or self._img_topic

    def _scan_cb(self, msg: LaserScan) -> None:
        with self._scan_lock:
            self._scan_msg = msg
            self._scan_seq += 1

    def _image_to_rgb(self, msg):
        height = int(msg.height)
        width = int(msg.width)
        step = int(msg.step)
        encoding = (msg.encoding or "").lower()
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        if height <= 0 or width <= 0 or step <= 0:
            return None
        try:
            rows = raw.reshape((height, step))
        except ValueError:
            return None

        if encoding in ("rgb8", "bgr8"):
            pixels = rows[:, :width * 3].reshape((height, width, 3))
            if encoding == "bgr8":
                pixels = pixels[:, :, ::-1]
            return pixels
        if encoding in ("rgba8", "bgra8"):
            pixels = rows[:, :width * 4].reshape((height, width, 4))[:, :, :3]
            if encoding == "bgra8":
                pixels = pixels[:, :, ::-1]
            return pixels
        if encoding in ("mono8", "8uc1"):
            mono = rows[:, :width]
            return np.repeat(mono[:, :, None], 3, axis=2)
        return None

    def get_latest_rgb(self):
        with self._img_lock:
            msg = self._img_msg
        if msg is None:
            return None

        rgb = self._image_to_rgb(msg)
        if rgb is None:
            return None
        return np.ascontiguousarray(rgb.copy())

    def _find_contours(self, mask):
        found = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(found) == 3:
            return found[1]
        return found[0]

    def _mask_hsv_ranges(self, hsv, ranges):
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _color_ranges(self, color):
        return BALL_HSV_RANGES.get(color)

    def _hough_color_candidates(self, rgb, color, target_window_deg=32.0, min_support_ratio=0.35):
        if rgb is None:
            return []

        ranges = self._color_ranges(color)
        if ranges is None:
            return []

        height, width = rgb.shape[:2]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = self._mask_hsv_ranges(hsv, ranges)
        if cv2.countNonZero(mask) == 0:
            return []

        blur = cv2.GaussianBlur(mask, (9, 9), 2.0)
        min_dim = min(height, width)
        min_radius = max(8, int(round(min_dim * 0.08)))
        max_radius = max(min_radius + 2, int(round(min_dim * 0.48)))
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(24, int(round(min_dim * 0.12))),
            param1=80,
            param2=18,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is None:
            return []

        fx = (width / 2.0) / math.tan(self.FOV_H / 2.0)
        window = max(math.radians(target_window_deg), 1e-6)
        candidates = []
        for cx, cy, radius in np.round(circles[0]).astype(int):
            if radius < 1:
                continue
            angle = math.atan2(width / 2.0 - cx, fx)
            if abs(angle) > window:
                continue

            circle_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.circle(circle_mask, (int(cx), int(cy)), int(radius), 255, -1)
            support = cv2.countNonZero(cv2.bitwise_and(mask, mask, mask=circle_mask))
            circle_area = math.pi * radius * radius
            support_ratio = support / max(circle_area, 1.0)
            if support_ratio < min_support_ratio:
                continue

            mean_hsv = cv2.mean(hsv, mask=circle_mask)
            mean_sat = float(mean_hsv[1])
            mean_val = float(mean_hsv[2])
            if mean_sat < 55.0 or mean_val < 45.0:
                continue

            x = max(0, int(cx - radius))
            y = max(0, int(cy - radius))
            bw = min(width - x, int(radius * 2))
            bh = min(height - y, int(radius * 2))
            center_score = 1.0 / (1.0 + (abs(angle) / window) ** 2)
            lower_score = 0.80 + 0.35 * (cy / max(float(height), 1.0))
            score = circle_area * (support_ratio ** 2.0) * center_score * lower_score
            candidates.append(
                {
                    "color": color,
                    "cx": float(cx),
                    "cy": float(cy),
                    "radius": float(radius),
                    "bbox": (int(x), int(y), int(bw), int(bh)),
                    "area": float(support),
                    "angle_rad": float(angle),
                    "angle_deg": float(math.degrees(angle)),
                    "distance": None,
                    "circularity": float(support_ratio),
                    "fill_ratio": float(support_ratio),
                    "aspect": 1.0,
                    "height_ratio": 1.0,
                    "width_ratio": 1.0,
                    "mean_sat": mean_sat,
                    "mean_val": mean_val,
                    "score": float(score),
                    "source": "hough_circle",
                    "support_ratio": float(support_ratio),
                }
            )

        return candidates

    def get_ball_candidates(self, rgb=None):
        if rgb is None:
            rgb = self.get_latest_rgb()
        if rgb is None:
            return []

        height, width = rgb.shape[:2]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        fx = (width / 2.0) / math.tan(self.FOV_H / 2.0)

        candidates = []
        for color in ("orange", "blue"):
            ranges = self._color_ranges(color)
            if ranges is None:
                continue
            mask = self._mask_hsv_ranges(hsv, ranges)
            for contour in self._find_contours(mask):
                contour_area = float(cv2.contourArea(contour))
                if contour_area < float(self.MIN_BLOB_PX):
                    continue

                perimeter = float(cv2.arcLength(contour, True))
                if perimeter <= 1e-6:
                    continue

                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                x, y, bw, bh = cv2.boundingRect(contour)
                if radius < 4.0 or bw <= 0 or bh <= 0:
                    continue

                circle_area = math.pi * radius * radius
                contour_mask = np.zeros(mask.shape, dtype=np.uint8)
                cv2.drawContours(contour_mask, [contour], -1, 255, -1)
                mean_hsv = cv2.mean(hsv, mask=contour_mask)
                mean_sat = float(mean_hsv[1])
                mean_val = float(mean_hsv[2])
                if mean_sat < 72.0 or mean_val < 45.0:
                    continue
                circularity = 4.0 * math.pi * contour_area / (perimeter * perimeter)
                fill_ratio = contour_area / max(circle_area, 1.0)
                aspect = float(bw) / float(bh)
                height_ratio = float(bh) / max(2.0 * radius, 1.0)
                width_ratio = float(bw) / max(2.0 * radius, 1.0)
                large_candidate = contour_area >= 600.0
                min_circularity = 0.18 if large_candidate else 0.38
                min_fill = 0.18 if large_candidate else 0.30
                if (
                    circularity < min_circularity
                    or fill_ratio < min_fill
                    or aspect < 0.40
                    or aspect > 1.85
                    or height_ratio < 0.45
                    or width_ratio < 0.45
                ):
                    continue

                angle = math.atan2(width / 2.0 - cx, fx)
                candidates.append(
                    {
                        "color": color,
                        "cx": float(cx),
                        "cy": float(cy),
                        "radius": float(radius),
                        "bbox": (int(x), int(y), int(bw), int(bh)),
                        "area": contour_area,
                        "angle_rad": float(angle),
                        "angle_deg": float(math.degrees(angle)),
                        "distance": None,
                        "circularity": float(circularity),
                        "fill_ratio": float(fill_ratio),
                        "aspect": float(aspect),
                        "height_ratio": float(height_ratio),
                        "width_ratio": float(width_ratio),
                        "mean_sat": mean_sat,
                        "mean_val": mean_val,
                        "score": 0.0,
                        "source": "contour",
                    }
                )

        return candidates

    def _detections_for_color(self, color):
        return [
            (c["angle_rad"], c["area"])
            for c in self.get_ball_candidates()
            if c["color"] == color
        ]


    def get_orange_detections(self) -> List[Tuple[float, float]]:
        return self._detections_for_color("orange")

    def get_blue_detections(self) -> List[Tuple[float, float]]:
        return self._detections_for_color("blue")

    def classify_center_ball(
        self,
        center_window_deg: float = 18.0,
        margin: float = 1.12,
        half_dist_deg: float = 3.0,
        expected_angle_deg: float = 0.0,
        target_window_deg: float = 34.0,
    ):
        """Classify the visually nearest ball-shaped candidate near the image center."""
        del margin, half_dist_deg
        expected_angle_rad = math.radians(expected_angle_deg)
        del center_window_deg
        window = max(math.radians(target_window_deg), 1e-6)
        rgb = self.get_latest_rgb()
        candidates = self.get_ball_candidates(rgb)
        candidates.extend(self._hough_color_candidates(
            rgb,
            "orange",
            target_window_deg=target_window_deg,
            min_support_ratio=0.35,
        ))
        candidates.extend(self._hough_color_candidates(
            rgb,
            "blue",
            target_window_deg=target_window_deg,
            min_support_ratio=0.35,
        ))
        center_candidates = []

        for item in candidates:
            angle_err = abs(item["angle_rad"] - expected_angle_rad)
            if angle_err > window:
                continue
            center_score = 1.0 / (1.0 + (angle_err / window) ** 2)
            ballness = (
                max(0.25, min(item["circularity"], 1.25))
                * max(0.25, min(item["fill_ratio"], 1.00))
                * max(0.25, min(item["height_ratio"], 1.10))
            )
            lower_score = 0.80 + 0.35 * (item["cy"] / max(float(rgb.shape[0]), 1.0))
            item["score"] = (
                item["area"]
                * center_score
                * lower_score
                * ballness
            )
            center_candidates.append(item)

        if center_candidates:
            selected = max(center_candidates, key=lambda c: c["score"])
            color = selected["color"]
            note = "visual_center_ball"
        else:
            selected = None
            color = None
            note = "no_center_ball"

        best_by_color = {}
        for item in center_candidates:
            current = best_by_color.get(item["color"])
            if current is None or item["score"] > current["score"]:
                best_by_color[item["color"]] = item

        return {
            "color": color,
            "note": note,
            "selected": selected,
            "orange": best_by_color.get("orange"),
            "blue": best_by_color.get("blue"),
            "orange_score": best_by_color.get("orange", {}).get("score", 0.0),
            "blue_score": best_by_color.get("blue", {}).get("score", 0.0),
            "candidates": candidates,
            "center_candidates": center_candidates,
            "rgb": rgb,
            "expected_angle_deg": float(expected_angle_deg),
            "target_window_deg": float(target_window_deg),
        }

    def find_center_color_blob(
        self,
        color: str = "orange",
        target_window_deg: float = 32.0,
        min_area: float = 80.0,
        rgb=None,
    ):
        if rgb is None:
            rgb = self.get_latest_rgb()
        if rgb is None:
            return None

        ranges = self._color_ranges(color)
        if ranges is None:
            return None

        height, width = rgb.shape[:2]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = self._mask_hsv_ranges(hsv, ranges)
        fx = (width / 2.0) / math.tan(self.FOV_H / 2.0)
        window = max(math.radians(target_window_deg), 1e-6)

        best = None
        for contour in self._find_contours(mask):
            area = float(cv2.contourArea(contour))
            if area < float(min_area):
                continue
            moments = cv2.moments(contour)
            if abs(moments["m00"]) < 1e-6:
                continue

            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
            angle = math.atan2(width / 2.0 - cx, fx)
            if abs(angle) > window:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            if bw <= 0 or bh <= 0:
                continue
            aspect = float(bw) / float(bh)
            if bw < 8 or bh < 8 or aspect < 0.35 or aspect > 2.8:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            circularity = (
                4.0 * math.pi * area / (perimeter * perimeter)
                if perimeter > 1e-6 else 0.0
            )
            radius = 0.5 * max(float(bw), float(bh))
            circle_area = math.pi * radius * radius
            fill_ratio = area / max(circle_area, 1.0)
            if circularity < 0.24 or fill_ratio < 0.28:
                continue
            contour_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            mean_hsv = cv2.mean(hsv, mask=contour_mask)
            mean_sat = float(mean_hsv[1])
            mean_val = float(mean_hsv[2])
            if mean_sat < 55.0 or mean_val < 45.0:
                continue

            angle_err = abs(angle)
            center_score = 1.0 / (1.0 + (angle_err / window) ** 2)
            lower_score = 0.80 + 0.35 * (cy / max(float(height), 1.0))
            shape_score = (
                max(0.30, min(circularity, 1.0))
                * max(0.35, min(fill_ratio, 1.0))
                * max(0.35, 1.0 - min(abs(aspect - 1.0), 1.0) * 0.6)
            )
            score = math.sqrt(area) * center_score * lower_score * shape_score
            item = {
                "color": color,
                "cx": cx,
                "cy": cy,
                "radius": radius,
                "bbox": (int(x), int(y), int(bw), int(bh)),
                "area": area,
                "angle_rad": float(angle),
                "angle_deg": float(math.degrees(angle)),
                "distance": None,
                "circularity": float(circularity),
                "fill_ratio": float(fill_ratio),
                "aspect": aspect,
                "height_ratio": float(bh) / max(2.0 * radius, 1.0),
                "width_ratio": float(bw) / max(2.0 * radius, 1.0),
                "mean_sat": mean_sat,
                "mean_val": mean_val,
                "score": float(score),
                "source": "color_blob",
            }
            if best is None or item["score"] > best["score"]:
                best = item

        hough_candidates = self._hough_color_candidates(
            rgb,
            color,
            target_window_deg=target_window_deg,
            min_support_ratio=0.35,
        )
        for item in hough_candidates:
            if best is None or item["score"] > best["score"]:
                best = item

        return best

    def save_debug_image(self, path, info, title=None):
        rgb = info.get("rgb")
        if rgb is None:
            return False
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        selected = info.get("selected")

        draw_items = info.get("center_candidates") or []
        if selected is not None and selected not in draw_items:
            draw_items.append(selected)

        for item in draw_items:
            color = (0, 255, 255) if item["color"] == "orange" else (255, 0, 0)
            thickness = 3 if item is selected else 1
            cx = int(round(item["cx"]))
            cy = int(round(item["cy"]))
            radius = int(round(item["radius"]))
            x, y, bw, bh = item["bbox"]
            cv2.rectangle(image, (x, y), (x + bw, y + bh), color, thickness)
            cv2.circle(image, (cx, cy), radius, color, thickness)
            label = "yellow" if item["color"] == "orange" else "blue"
            label += " a=%.1f area=%.0f r=%.0f" % (
                item["angle_deg"],
                item["area"],
                item.get("radius", 0.0),
            )
            source = item.get("source")
            if source:
                label += f" src={source}"
            cv2.putText(
                image,
                label,
                (x, max(15, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        height, width = image.shape[:2]
        cv2.line(image, (width // 2, 0), (width // 2, height), (255, 255, 255), 1)
        expected_angle = math.radians(info.get("expected_angle_deg", 0.0))
        fx = (width / 2.0) / math.tan(self.FOV_H / 2.0)
        expected_x = int(round(width / 2.0 - fx * math.tan(expected_angle)))
        if -width <= expected_x <= 2 * width:
            cv2.line(
                image,
                (expected_x, 0),
                (expected_x, height),
                (255, 0, 255),
                2,
            )
        if title:
            cv2.putText(
                image,
                title,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(path, image)
        return True

    def get_distance_at_angle(self, angle_rad: float, half_deg: float = 2.0) -> Optional[float]:
        with self._scan_lock:
            scan = self._scan_msg
        if scan is None:
            return None

        ranges = list(scan.ranges)
        n = len(ranges)
        if n == 0:
            return None

        angle_min = getattr(scan, "angle_min", self.LIDAR_MIN_ANG)
        angle_max = getattr(scan, "angle_max", self.LIDAR_MAX_ANG)
        angle_increment = getattr(scan, "angle_increment", 0.0)
        if angle_increment == 0.0:
            angle_increment = (angle_max - angle_min) / max(n - 1, 1)
        if angle_rad < angle_min or angle_rad > angle_max:
            return None

        center_idx = int(round((angle_rad - angle_min) / angle_increment))
        half = max(1, int(round(math.radians(half_deg) / abs(angle_increment))))
        lo = max(0, center_idx - half)
        hi = min(n, center_idx + half + 1)
        valid = [r for r in ranges[lo:hi]
                 if math.isfinite(r) and scan.range_min < r < scan.range_max]
        return float(min(valid)) if valid else None

    def get_lidar_landmark_observations(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw_deg: float,
        landmarks,
        min_distance: float = 0.25,
        max_distance: float = 2.20,
        max_match_error: float = 0.65,
        lidar_body_x: float = None,
        lidar_body_y: float = None,
        ball_radius: float = None,
        return_debug: bool = False,
    ):
        """Match compact lidar returns to known field landmarks.

        The returned offsets are field-frame corrections for the current
        mapped pose.  Matching is intentionally geometric; camera detections
        remain responsible for ball color and impact heading.
        """
        with self._scan_lock:
            scan = self._scan_msg
            scan_seq = self._scan_seq
        debug = {
            "scan_seq": scan_seq,
            "reason": "",
            "compact_count": 0,
            "candidate_count": 0,
            "expected": [],
        }

        def finish(observations):
            if return_debug:
                return observations, debug
            return observations

        if scan is None:
            debug["reason"] = "no_scan"
            return finish([])

        ranges = np.asarray(list(scan.ranges), dtype=np.float32)
        if ranges.size == 0:
            debug["reason"] = "empty_scan"
            return finish([])
        angle_min = float(getattr(scan, "angle_min", self.LIDAR_MIN_ANG))
        angle_max = float(getattr(scan, "angle_max", self.LIDAR_MAX_ANG))
        angle_increment = float(getattr(scan, "angle_increment", 0.0))
        if abs(angle_increment) < 1e-9:
            angle_increment = (angle_max - angle_min) / max(ranges.size - 1, 1)
        if abs(angle_increment) < 1e-9:
            debug["reason"] = "invalid_angle_increment"
            return finish([])

        scan_min = float(getattr(scan, "range_min", 0.05))
        scan_max = float(getattr(scan, "range_max", max_distance))
        lower = max(min_distance, scan_min + 0.02)
        upper = min(max_distance, scan_max - 0.02)
        if lower >= upper:
            debug["reason"] = "invalid_range_window"
            return finish([])

        angles = angle_min + np.arange(ranges.size, dtype=np.float32) * angle_increment
        valid = np.isfinite(ranges) & (ranges >= lower) & (ranges <= upper)
        points = np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles)))

        clusters = []
        current = []
        for index in range(ranges.size):
            if not valid[index]:
                if current:
                    clusters.append(current)
                    current = []
                continue
            if current:
                previous = current[-1]
                gap = float(np.linalg.norm(points[index] - points[previous]))
                if gap > 0.09:
                    clusters.append(current)
                    current = []
            current.append(index)
        if current:
            clusters.append(current)

        compact = []
        for indices in clusters:
            if len(indices) < 2:
                continue
            local = points[indices]
            center = np.median(local, axis=0)
            radial = np.linalg.norm(local, axis=1)
            width = float(np.max(np.linalg.norm(local - center, axis=1)))
            if width > 0.22 or float(np.ptp(radial)) > 0.18:
                continue
            compact.append({
                "indices": indices,
                "x": float(center[0]),
                "y": float(center[1]),
                "range": float(np.linalg.norm(center)),
                "angle_rad": float(math.atan2(center[1], center[0])),
            })
        debug["compact_count"] = len(compact)

        if not compact:
            debug["reason"] = "no_compact_clusters"
            return finish([])

        lidar_body_x = self.LIDAR_BODY_X if lidar_body_x is None else float(lidar_body_x)
        lidar_body_y = self.LIDAR_BODY_Y if lidar_body_y is None else float(lidar_body_y)
        ball_radius = self.BALL_RADIUS_M if ball_radius is None else max(0.0, float(ball_radius))
        yaw_rad = math.radians(float(robot_yaw_deg))
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        sensor_x = (
            float(robot_x)
            + cos_yaw * lidar_body_x
            - sin_yaw * lidar_body_y
        )
        sensor_y = (
            float(robot_y)
            + sin_yaw * lidar_body_x
            + cos_yaw * lidar_body_y
        )
        candidates = []
        for landmark in dict.fromkeys((float(p[0]), float(p[1])) for p in landmarks):
            dx = landmark[0] - sensor_x
            dy = landmark[1] - sensor_y
            expected_x = dx * math.cos(yaw_rad) + dy * math.sin(yaw_rad)
            expected_y = -dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)
            expected_center_range = math.hypot(expected_x, expected_y)
            expected_range = expected_center_range - ball_radius
            if expected_range < lower or expected_range > upper:
                debug["expected"].append({
                    "landmark": landmark,
                    "reason": "range",
                    "range": expected_range,
                })
                continue
            surface_scale = expected_range / max(expected_center_range, 1e-6)
            expected_surface_x = expected_x * surface_scale
            expected_surface_y = expected_y * surface_scale
            expected_angle = math.atan2(expected_y, expected_x)
            if expected_angle < angle_min or expected_angle > angle_max:
                debug["expected"].append({
                    "landmark": landmark,
                    "reason": "fov",
                    "range": expected_range,
                    "angle_deg": math.degrees(expected_angle),
                })
                continue
            debug["expected"].append({
                "landmark": landmark,
                "reason": "search",
                "range": expected_range,
                "angle_deg": math.degrees(expected_angle),
            })
            for cluster_index, cluster in enumerate(compact):
                angle_error = abs(math.atan2(
                    math.sin(cluster["angle_rad"] - expected_angle),
                    math.cos(cluster["angle_rad"] - expected_angle),
                ))
                range_error = abs(cluster["range"] - expected_range)
                local_error = math.hypot(
                    cluster["x"] - expected_surface_x,
                    cluster["y"] - expected_surface_y,
                )
                if angle_error > math.radians(35.0):
                    continue
                if range_error > max_match_error or local_error > max_match_error:
                    continue
                candidates.append((
                    local_error + 0.35 * range_error,
                    landmark,
                    cluster_index,
                    cluster,
                ))
        debug["candidate_count"] = len(candidates)

        candidates.sort(key=lambda item: item[0])
        used_landmarks = set()
        used_clusters = set()
        observations = []
        for score, landmark, cluster_index, cluster in candidates:
            if landmark in used_landmarks or cluster_index in used_clusters:
                continue
            used_landmarks.add(landmark)
            used_clusters.add(cluster_index)
            center_range = cluster["range"] + ball_radius
            body_center_x = center_range * math.cos(cluster["angle_rad"])
            body_center_y = center_range * math.sin(cluster["angle_rad"])
            body_x = lidar_body_x + body_center_x
            body_y = lidar_body_y + body_center_y
            world_x = sensor_x + cos_yaw * body_center_x - sin_yaw * body_center_y
            world_y = sensor_y + sin_yaw * body_center_x + cos_yaw * body_center_y
            observations.append({
                "landmark": landmark,
                "range": cluster["range"],
                "angle_deg": math.degrees(cluster["angle_rad"]),
                "body_x": body_x,
                "body_y": body_y,
                "world_x": world_x,
                "world_y": world_y,
                "correction_x": landmark[0] - world_x,
                "correction_y": landmark[1] - world_y,
                "match_error": score,
                "points": len(cluster["indices"]),
                "scan_seq": scan_seq,
            })
        debug["reason"] = "matched" if observations else "no_landmark_match"
        debug["matched"] = [item["landmark"] for item in observations]
        return finish(observations)
