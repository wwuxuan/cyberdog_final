#!/usr/bin/env python3
"""Laptop service for the stage3 left and right fish-eye line distances."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import time
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

import fisheye_line as line_distance


BODY_HEIGHT = 0.235
STAGE3_CALIBRATION = {
    "left": (0.78, 0.0),
    "right": (0.94, 0.0),
}


class VisionHandler(BaseHTTPRequestHandler):
    server_version = "CyberdogStage3LineVision/1.0"

    def do_POST(self):
        request = urlsplit(self.path)
        if request.path != "/measure":
            self.send_error(404)
            return
        allowed_ips = getattr(self.server, "allowed_ips", set())
        if allowed_ips and self.client_address[0] not in allowed_ips:
            self.send_error(403, "client IP is not allowed")
            return

        side = parse_qs(request.query).get("side", ["right"])[0].lower()
        if side not in STAGE3_CALIBRATION:
            self._send_json(
                400,
                {"distance": None, "error": "side must be left or right"},
            )
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 2_000_000:
                raise ValueError("invalid image size")
            raw = self.rfile.read(length)
            image = cv2.imdecode(
                np.frombuffer(raw, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if image is None:
                raise ValueError("cannot decode JPEG")

            started = time.perf_counter()
            scale, offset = STAGE3_CALIBRATION[side]
            distance, _ = line_distance.measure_vectorized(
                image,
                side,
                BODY_HEIGHT,
                scale=scale,
                offset=offset,
                line_mode="body_x",
                step=4,
                max_fit_points=240,
                ransac_trials=180,
                check_front_branch=True,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._send_json(
                200,
                {
                    "distance": distance,
                    "side": side,
                    "elapsed_ms": round(elapsed_ms, 1),
                },
            )
        except Exception as exc:
            self._send_json(400, {"distance": None, "error": str(exc)})

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print("[stage3_vision] " + (fmt % args), flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Laptop service for stage3 left/right fish-eye line distance."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9877)
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
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "calibration",
        ),
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
        "[stage3_vision] listening on http://{}:{}/measure ".format(
            args.host,
            args.port,
        )
        + "left=0.780/+0.000 right=0.940/+0.000 body_h=0.235",
        flush=True,
    )
    if server.allowed_ips:
        print(
            "[stage3_vision] allowed client IPs: {}".format(
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
    raise SystemExit(main())
