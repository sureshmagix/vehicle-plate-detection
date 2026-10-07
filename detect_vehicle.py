#!/usr/bin/env python3
"""
detect_vehicle.py
=================
Stage 1: Vehicle Detection Module

Identifies vehicles (cars, trucks, buses, motorcycles) using a pre-trained
YOLO model. Designed both as a standalone CLI tool and as an importable module
for upstream AI training pipelines and annotation systems.

Key Features:
- Outputs pixel bounding boxes (xyxy, xywh) and normalized YOLO training format.
- Generates annotated visualization overlays.
- Exports structured detection metadata in JSON for training pipelines.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Union

# Safeguard cache directories for sandboxed/restricted environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/yolo")

try:
    import cv2
    import numpy as np
except ImportError:
    print("Error: OpenCV is required. Install via: pip install opencv-python")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: Ultralytics is required. Install via: pip install ultralytics")
    sys.exit(1)


# Standard COCO vehicle class names
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}


class VehicleDetector:
    """
    Modular Vehicle Detector utilizing YOLO architecture.
    """

    def __init__(
        self,
        model_path: str = "plate_identify/yolov8n.pt",
        conf_threshold: float = 0.35,
        target_classes: set = None,
    ):
        """
        Initialize the Vehicle Detector.

        :param model_path: Path to YOLO weights (.pt)
        :param conf_threshold: Minimum detection confidence threshold
        :param target_classes: Set of class names to detect (defaults to VEHICLE_CLASSES)
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            # Fallback to local directory if relative path differs
            fallback = Path(__file__).parent / self.model_path.name
            if fallback.exists():
                self.model_path = fallback

        self.conf_threshold = conf_threshold
        self.target_classes = target_classes or VEHICLE_CLASSES

        print(f"[VehicleDetector] Loading model from: {self.model_path}")
        self.model = YOLO(str(self.model_path))

    def detect(
        self, image_input: Union[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Detect vehicles in an image.

        :param image_input: File path (str) or loaded BGR image (np.ndarray)
        :return: Dictionary containing image metadata, vehicle detections, and coordinates.
        """
        if isinstance(image_input, (str, Path)):
            img_path = str(image_input)
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Unable to read image at: {img_path}")
        elif isinstance(image_input, np.ndarray):
            img_path = "in_memory_image"
            image = image_input.copy()
        else:
            raise TypeError("image_input must be a file path string or np.ndarray")

        orig_h, orig_w = image.shape[:2]

        # Run YOLO inference
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        detections: List[Dict[str, Any]] = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names.get(cls_id, str(cls_id))

                # Filter strictly for vehicle classes
                if cls_name.lower() in self.target_classes:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    # Clamp coordinates to image boundaries
                    x1 = max(0, min(orig_w - 1, x1))
                    y1 = max(0, min(orig_h - 1, y1))
                    x2 = max(0, min(orig_w - 1, x2))
                    y2 = max(0, min(orig_h - 1, y2))

                    w = x2 - x1
                    h = y2 - y1

                    # Normalized coordinates (YOLO training format: class x_c y_c w h)
                    x_center_norm = ((x1 + x2) / 2.0) / orig_w
                    y_center_norm = ((y1 + y2) / 2.0) / orig_h
                    w_norm = w / float(orig_w)
                    h_norm = h / float(orig_h)

                    detection_item = {
                        "id": len(detections) + 1,
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": round(conf, 4),
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "bbox_xywh": [x1, y1, w, h],
                        "bbox_yolo_norm": [
                            round(x_center_norm, 6),
                            round(y_center_norm, 6),
                            round(w_norm, 6),
                            round(h_norm, 6),
                        ],
                        "area_pixels": w * h,
                    }
                    detections.append(detection_item)

        return {
            "source_image": str(img_path),
            "image_dimensions": {"width": orig_w, "height": orig_h},
            "vehicle_count": len(detections),
            "vehicles": detections,
            "raw_image": image,
        }

    def draw_detections(
        self,
        image: np.ndarray,
        vehicles: List[Dict[str, Any]],
        color: tuple = (0, 220, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw clean bounding boxes with labels and confidence tags.
        """
        annotated = image.copy()
        for v in vehicles:
            x1, y1, x2, y2 = v["bbox_xyxy"]
            label = f"{v['class_name'].upper()} {v['confidence']*100:.1f}%"

            # Draw outer rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Draw text label banner
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            font_thick = 1
            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_scale, font_thick
            )
            banner_y1 = max(0, y1 - text_h - baseline - 6)
            banner_y2 = y1

            cv2.rectangle(
                annotated,
                (x1, banner_y1),
                (x1 + text_w + 8, banner_y2),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 4, banner_y2 - baseline - 2),
                font,
                font_scale,
                (0, 0, 0),
                font_thick,
                lineType=cv2.LINE_AA,
            )
        return annotated


def main():
    parser = argparse.ArgumentParser(
        description="Detect vehicles in an image and output precise coordinates."
    )
    parser.add_argument(
        "--image",
        "-i",
        default="plate_identify/sample.jpg",
        help="Path to the input image file.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="plate_identify/yolov8n.pt",
        help="Path to vehicle YOLO model weights.",
    )
    parser.add_argument(
        "--conf",
        "-c",
        type=float,
        default=0.35,
        help="Confidence threshold for vehicle detection (0.0 - 1.0).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="plate_identify/outputs",
        help="Directory to save output annotated images and JSON coordinates.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save visual outputs to disk.",
    )

    args = parser.parse_args()

    detector = VehicleDetector(model_path=args.model, conf_threshold=args.conf)
    results = detector.detect(args.image)

    print("\n" + "=" * 60)
    print(" VEHICLE DETECTION SUMMARY")
    print("=" * 60)
    print(f"Image: {results['source_image']}")
    print(
        f"Resolution: {results['image_dimensions']['width']}x{results['image_dimensions']['height']}"
    )
    print(f"Vehicles Detected: {results['vehicle_count']}")
    print("-" * 60)

    for v in results["vehicles"]:
        print(f"[{v['id']}] Class: {v['class_name']} | Confidence: {v['confidence']*100:.1f}%")
        print(f"    Bounding Box (xyxy)  : {v['bbox_xyxy']}")
        print(f"    Bounding Box (xywh)  : {v['bbox_xywh']}")
        print(f"    YOLO Normalized (cx,cy,w,h): {v['bbox_yolo_norm']}")
        print(f"    Area (px^2)          : {v['area_pixels']}")

    if not args.no_save:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save annotated image
        annotated_img = detector.draw_detections(
            results["raw_image"], results["vehicles"]
        )
        base_name = Path(args.image).stem
        out_img_path = out_dir / f"{base_name}_vehicle_detected.jpg"
        cv2.imwrite(str(out_img_path), annotated_img)

        # Save JSON metadata (excluding raw_image)
        export_data = {k: v for k, v in results.items() if k != "raw_image"}
        out_json_path = out_dir / f"{base_name}_vehicles.json"
        with open(out_json_path, "w") as f:
            json.dump(export_data, f, indent=2)

        print("-" * 60)
        print(f"Annotated image saved : {out_img_path}")
        print(f"Coordinates JSON saved: {out_json_path}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
