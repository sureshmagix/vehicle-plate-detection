#!/usr/bin/env python3
"""
ui_server.py
============
Interactive Web UI & Local Server for Vehicle and License Plate Detection.

Features:
- Web-based image uploader / browser with drag-and-drop.
- Instant sample image testing.
- Two-stage detection (Vehicle + Plate) and Vehicle-only mode.
- Interactive bounding box coordinate viewer:
  * Pixel coordinates [x1, y1, x2, y2]
  * Width, Height, Area, Aspect Ratio
  * YOLO normalized coordinates [cx, cy, w, h] for AI training
- High-res cropped vehicle & plate image previews.
- One-click copy JSON / YOLO .txt annotations.
- Completely self-contained using Python's built-in http.server (no external web framework needed).
"""

import os
import sys
import json
import base64
import cgi
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# Add current directory to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Safeguard environment variables
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/yolo")

import cv2
import numpy as np

from detect_vehicle import VehicleDetector
from detect_plate import TwoStagePlateDetector

HOST = "0.0.0.0"
PORT = 8080

# Initialize detectors globally
print("[UI Server] Initializing detector models...")
CAR_MODEL_PATH = str(BASE_DIR / "yolov8n.pt")
PLATE_MODEL_PATH = str(BASE_DIR / "plate_model.pt")

vehicle_detector = VehicleDetector(model_path=CAR_MODEL_PATH)
two_stage_detector = TwoStagePlateDetector(
    car_model_path=CAR_MODEL_PATH,
    plate_model_path=PLATE_MODEL_PATH,
)
print("[UI Server] Models ready!")


def bgr_to_base64_data_uri(bgr_img: np.ndarray, ext: str = ".jpg") -> str:
    """Encode OpenCV BGR image to base64 data URI."""
    success, buffer = cv2.imencode(ext, bgr_img)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    mime = "image/jpeg" if ext.lower() in [".jpg", ".jpeg"] else "image/png"
    return f"data:{mime};base64,{b64_str}"


class DetectionRequestHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        url_path = urllib.parse.urlparse(self.path).path

        if url_path in ["/", "/index.html"]:
            self.serve_ui()
        elif url_path == "/sample.jpg":
            sample_path = BASE_DIR / "sample.jpg"
            if sample_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                with open(sample_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "Sample image not found")
        elif url_path.startswith("/docs/"):
            # Serve files from docs directory
            rel_file = url_path.replace("/docs/", "")
            file_path = BASE_DIR / "docs" / rel_file
            if file_path.exists() and file_path.is_file():
                self.serve_file(file_path)
            else:
                self.send_error(404, "File not found")
        else:
            self.send_error(404, "Endpoint not found")

    def do_POST(self):
        url_path = urllib.parse.urlparse(self.path).path

        if url_path == "/api/detect":
            self.handle_detect_api()
        else:
            self.send_error(404, "Endpoint not found")

    def serve_file(self, file_path: Path):
        ext = file_path.suffix.lower()
        content_types = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css",
            ".js": "application/javascript",
            ".json": "application/json",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml",
        }
        ctype = content_types.get(ext, "application/octet-stream")
        with open(file_path, "rb") as f:
            content = f.read()

        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def serve_ui(self):
        ui_file = BASE_DIR / "docs" / "index.html"
        if ui_file.exists():
            self.serve_file(ui_file)
        else:
            self.send_error(500, "docs/index.html UI file missing")

    def handle_detect_api(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            req_data = json.loads(body.decode("utf-8"))

            mode = req_data.get("mode", "two_stage")  # 'two_stage' or 'vehicle_only'
            car_conf = float(req_data.get("car_conf", 0.35))
            plate_conf = float(req_data.get("plate_conf", 0.25))

            # Decode image
            image_b64 = req_data.get("image_base64", "")
            if image_b64:
                if "," in image_b64:
                    image_b64 = image_b64.split(",", 1)[1]
                img_bytes = base64.b64decode(image_b64)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                # Use default sample.jpg
                sample_path = BASE_DIR / "sample.jpg"
                img = cv2.imread(str(sample_path))

            if img is None:
                self.send_json_error(400, "Could not decode image")
                return

            h, w = img.shape[:2]

            antiblur_vehicle = bool(req_data.get("antiblur_vehicle", False))
            antiblur_plate = bool(req_data.get("antiblur_plate", False))
            antiblur_char = bool(req_data.get("antiblur_char", False))

            if mode == "vehicle_only":
                vehicle_detector.conf_threshold = car_conf
                results = vehicle_detector.detect(img, antiblur=antiblur_vehicle)
                annotated = vehicle_detector.draw_detections(
                    results["raw_image"], results["vehicles"]
                )

                response_payload = {
                    "status": "success",
                    "mode": "vehicle_only",
                    "dimensions": {"width": w, "height": h},
                    "vehicle_count": results["vehicle_count"],
                    "plate_count": 0,
                    "vehicles": results["vehicles"],
                    "annotated_image_uri": bgr_to_base64_data_uri(annotated),
                    "antiblur": {
                        "vehicle": antiblur_vehicle,
                        "plate": antiblur_plate,
                        "char": antiblur_char,
                    },
                }
            else:
                # Two-Stage
                two_stage_detector.car_conf = car_conf
                two_stage_detector.plate_conf = plate_conf
                results = two_stage_detector.detect(
                    img,
                    extract_text=True,
                    extract_characters=True,
                    antiblur_vehicle=antiblur_vehicle,
                    antiblur_plate=antiblur_plate,
                    antiblur_char=antiblur_char,
                )
                annotated = two_stage_detector.draw_annotations(
                    results["raw_image"], results["vehicles"]
                )

                # Format results with base64 crops for UI display
                formatted_vehicles = []
                for v in results["vehicles"]:
                    v_crop_uri = bgr_to_base64_data_uri(v["vehicle_crop_bgr"])
                    formatted_plates = []
                    for idx, p in enumerate(v["plates_detected"]):
                        patch = v["_plate_patches"][idx]
                        p_crop_uri = bgr_to_base64_data_uri(patch)

                        # Plate with character bounding boxes overlay
                        ann_patch = None
                        if idx < len(v.get("_plate_annotated_patches", [])):
                            ann_patch = v["_plate_annotated_patches"][idx]
                        ann_uri = bgr_to_base64_data_uri(ann_patch) if ann_patch is not None and ann_patch.size > 0 else p_crop_uri

                        # Encode individual character crops
                        char_patches = []
                        if idx < len(v.get("_char_patches_per_plate", [])):
                            char_patches = v["_char_patches_per_plate"][idx]

                        formatted_chars = []
                        for c_idx, ch in enumerate(p.get("characters", [])):
                            ch_item = dict(ch)
                            if c_idx < len(char_patches) and char_patches[c_idx] is not None:
                                ch_item["crop_uri"] = bgr_to_base64_data_uri(char_patches[c_idx], ext=".png")
                            else:
                                ch_item["crop_uri"] = ""
                            formatted_chars.append(ch_item)

                        p_item = dict(p)
                        p_item["crop_uri"] = p_crop_uri
                        p_item["plate_annotated_crop_uri"] = ann_uri
                        p_item["characters"] = formatted_chars
                        formatted_plates.append(p_item)

                    v_data = {
                        "vehicle_id": v["vehicle_id"],
                        "class_name": v["class_name"],
                        "confidence": v["confidence"],
                        "bbox_xyxy": v["bbox_xyxy"],
                        "bbox_xywh": v["bbox_xywh"],
                        "bbox_yolo_norm": v["bbox_yolo_norm"],
                        "vehicle_crop_uri": v_crop_uri,
                        "plates_detected": formatted_plates,
                    }
                    formatted_vehicles.append(v_data)

                response_payload = {
                    "status": "success",
                    "mode": "two_stage",
                    "dimensions": {"width": w, "height": h},
                    "vehicle_count": results["vehicle_count"],
                    "plate_count": results["plate_count"],
                    "vehicles": formatted_vehicles,
                    "annotated_image_uri": bgr_to_base64_data_uri(annotated),
                    "antiblur": {
                        "vehicle": antiblur_vehicle,
                        "plate": antiblur_plate,
                        "char": antiblur_char,
                    },
                }

            self.send_json_response(response_payload)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send_json_error(500, str(e))

    def send_json_response(self, data: dict, status_code: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_json_error(self, status_code: int, message: str):
        self.send_json_response({"status": "error", "message": message}, status_code)


def run_server(port: int = PORT):
    server_address = (HOST, port)
    httpd = HTTPServer(server_address, DetectionRequestHandler)
    print("\n" + "=" * 60)
    print(f"🚀 Plate & Vehicle Detection UI Server running!")
    print(f"👉 Local Web UI URL: http://localhost:{port}")
    print(f"👉 To stop server: Press Ctrl + C")
    print("=" * 60 + "\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[UI Server] Shutting down...")
        httpd.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start the Detection Web UI Server.")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port (default: {PORT})")
    args = parser.parse_args()
    run_server(args.port)
