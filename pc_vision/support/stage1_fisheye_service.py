#!/usr/bin/env python3
"""Run on the laptop: receive a right fish-eye JPEG and return its line distance."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import time
from urllib.parse import urlsplit

import cv2
import numpy as np

import fisheye_line as line_distance


BODY_HEIGHT = 0.235
RIGHT_SCALE = 1.04
RIGHT_OFFSET = -0.040


class VisionHandler(BaseHTTPRequestHandler):
    server_version = "CyberdogStage1Vision/1.0"

    def do_POST(self):
        if urlsplit(self.path).path != "/measure":
            self.send_error(404)
            return
        allowed_ips = getattr(self.server, "allowed_ips", set())
        if allowed_ips and self.client_address[0] not in allowed_ips:
            self.send_error(403, "client IP is not allowed")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 2_000_000:
                raise ValueError("invalid image size")
            raw = self.rfile.read(length)
            image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("cannot decode JPEG")
            started = time.perf_counter()
            distance, _ = line_distance.measure_vectorized(
                image, "right", BODY_HEIGHT,
                scale=RIGHT_SCALE,
                offset=RIGHT_OFFSET,
                line_mode="body_x",
                step=4,
                max_fit_points=240,
                ransac_trials=180,
                check_front_branch=True,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            body = json.dumps({
                "distance": distance,
                "elapsed_ms": round(elapsed_ms, 1),
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            body = json.dumps({"distance": None, "error": str(exc)}).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, fmt, *args):
        print("[local_vision] " + (fmt % args), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Stage 1 local right fish-eye line service.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument(
        "--dog-ip",
        default="",
        help="optional dog IP allowlist for incoming HTTP requests",
    )
    parser.add_argument(
        "--push-ip",
        default="",
        help="optional compatibility peer IP; also accepted as an allowed client",
    )
    parser.add_argument(
        "--calib-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration"),
        help="directory containing the dog's factory camera calibration YAML files",
    )
    args = parser.parse_args()
    line_distance.CAMS = line_distance.load_calibration(args.calib_dir)
    server = ThreadingHTTPServer((args.host, args.port), VisionHandler)
    server.allowed_ips = {
        ip.strip()
        for ip in (args.dog_ip, args.push_ip)
        if ip and ip.strip()
    }
    print(
        "[local_vision] listening on http://{}:{}/measure body_h={:.3f} "
        "scale={:.3f} offset={:+.3f}".format(
            args.host, args.port, BODY_HEIGHT, RIGHT_SCALE, RIGHT_OFFSET
        ),
        flush=True,
    )
    if server.allowed_ips:
        print(
            "[stage1_vision] allowed client IPs: {}".format(
                ", ".join(sorted(server.allowed_ips))
            ),
            flush=True,
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
