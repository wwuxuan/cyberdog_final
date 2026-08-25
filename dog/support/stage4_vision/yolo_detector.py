import os
import threading
from typing import Dict, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import Image, LaserScan
except ImportError:
    rclpy = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class YoloDetector:
    """
    Thread-safe YOLO obstacle detector with LIDAR-based proximity support.

    Subscribes to:
      - /rgb_camera/image_raw  (RGB, 320×180, 15 Hz) — YOLO detection
      - /scan                  (LaserScan, 180 samples [-90°,+90°], 5 Hz)

    Caller must call rclpy.init() before start(), and rclpy.shutdown() after stop().
    """

    def __init__(self, model_path: str, conf: float = 0.5,
                 topic: str = "/rgb_camera/image_raw"):
        self._model_path = model_path
        self._conf = conf
        self._topic = topic

        self._lock = threading.Lock()
        self._detections: List[Dict] = []

        self._scan_lock = threading.Lock()
        self._scan_msg = None  # latest LaserScan

        self._node = None
        self._bridge = None
        self._model = None
        self._spin_thread = None

    def start(self) -> None:
        self._model = YOLO(self._model_path)
        print(f"[yolo] classes={len(self._model.names)}")

        self._bridge = CvBridge()
        self._node = rclpy.create_node("yolo_detector_stage4")

        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=1)
        self._node.create_subscription(Image, self._topic, self._image_callback, qos)
        self._node.create_subscription(LaserScan, "/scan", self._scan_callback, qos)

        self._spin_thread = threading.Thread(
            target=rclpy.spin, args=(self._node,), daemon=True
        )
        self._spin_thread.start()
        print("[yolo] started")

    def stop(self) -> None:
        if self._node:
            self._node.destroy_node()
            self._node = None
        print("[yolo] stopped")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _image_callback(self, msg) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            results = self._model(frame, conf=self._conf, verbose=False)
            h, w = frame.shape[:2]
            new_dets = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = self._model.names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bw, bh = x2 - x1, y2 - y1
                new_dets.append({
                    "class_name": cls_name,
                    "confidence": float(box.conf[0]),
                    "bbox_area": bw * bh,
                    "bbox_area_norm": (bw * bh) / max(h * w, 1),
                    "bbox_cx_norm": ((x1 + x2) / 2.0) / max(w, 1),
                })
            with self._lock:
                self._detections = new_dets
        except Exception as e:
            print(f"[yolo] rgb error: {e}")

    def _scan_callback(self, msg) -> None:
        with self._scan_lock:
            self._scan_msg = msg

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def get_detections(self) -> List[Dict]:
        with self._lock:
            return list(self._detections)

    def nearest_of(self, classes: List[str]) -> Optional[Tuple[str, float]]:
        """Return (class_name, bbox_area) of the largest-bbox detection among given classes."""
        candidates = [d for d in self.get_detections() if d["class_name"] in classes]
        if not candidates:
            return None
        best = max(candidates, key=lambda d: d["bbox_area"])
        return (best["class_name"], best["bbox_area"])

    def get_front_distance(self, half_angle_deg: float = 40.0) -> Optional[float]:
        """Min LIDAR range in a forward cone (±half_angle_deg around 0°), in metres.

        Returns None if no scan received yet or no valid samples in the window.
        180-sample scan covers [-90°, +90°], so forward = center index ~89/90.

        Filters out:
          - inf / NaN  (no return — empty space)
          - r >= range_max
          - r < LIDAR_NEAR_FILTER (0.10 m) — robot self-occlusion / sensor noise spikes
        """
        LIDAR_NEAR_FILTER = 0.10  # m — anything closer is robot body / spurious noise

        with self._scan_lock:
            if self._scan_msg is None:
                return None
            scan = self._scan_msg

        ranges = list(scan.ranges)
        n = len(ranges)
        if n == 0:
            return None

        center = n // 2
        samples_per_deg = (n - 1) / 180.0  # ~1 sample / deg
        half_samples = max(1, int(round(half_angle_deg * samples_per_deg)))
        lo = max(0, center - half_samples)
        hi = min(n, center + half_samples + 1)

        valid = [r for r in ranges[lo:hi]
                 if LIDAR_NEAR_FILTER <= r < scan.range_max]
        if not valid:
            return None
        return float(min(valid))
