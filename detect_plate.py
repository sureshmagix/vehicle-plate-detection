#!/usr/bin/env python3
"""
detect_plate.py
===============
Two-Stage License Plate & Vehicle Detection Pipeline

Stage 1: Detects the vehicle in the full image using YOLO.
Stage 2: Crops the localized vehicle region (Region of Interest - ROI)
         and runs a high-precision License Plate detection model on the crop.
Stage 3: Maps the plate coordinates back to the full-image coordinate space.

Why Two-Stage?
- Prevents false positives from background street signs, store banners, and roadside text.
- Provides superior precision by feeding a focused, high-density vehicle patch into the plate model.
- Essential for production ALPR / ANPR workflows and AI training dataset generation.
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple

# Set writable directories for matplotlib/ultralytics cache
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


VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

VALID_INDIAN_STATES = [
    "TN", "KA", "MH", "DL", "KL", "TS", "AP", "HR", "UP", "GJ", "RJ", "PB",
    "WB", "OD", "BR", "MP", "JH", "CH", "GA", "AS", "JK", "UK", "TR", "HP",
    "NL", "ML", "MN", "SK", "CG", "AR", "MZ", "DD", "DN", "LD", "PY", "AN", "BH"
]

DIGIT_TO_LETTER = {
    "0": "O", "1": "I", "2": "Z", "3": "B", "4": "A",
    "5": "S", "6": "G", "7": "T", "8": "B", "9": "P"
}

LETTER_TO_DIGIT = {
    "O": "0", "D": "0", "Q": "0", "I": "1", "L": "1",
    "Z": "2", "A": "4", "S": "5", "G": "6", "T": "7",
    "B": "8", "P": "9"
}


def clean_and_parse_plate(raw_text: str) -> Tuple[str, float]:
    """
    Parse, validate and disambiguate license plate characters according to
    standard license plate syntax rules (Indian HSRP, Bharat series, and International).
    """
    clean = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
    if len(clean) < 3:
        return ("", 0.0)

    # Strip HSRP prefix "IND" if captured by OCR
    if clean.startswith("IND") and len(clean) >= 7:
        clean = clean[3:]

    # Strip edge border/screw artifacts (e.g. trailing '1', 'I', 'T')
    if len(clean) == 10 and clean[-1] in "1IT" and clean[-5:-1].isdigit():
        clean = clean[:-1]
    elif len(clean) == 11 and clean[-1] in "1IT":
        clean = clean[:-1]

    # 1. Bharat Series (e.g., 22 BH 1234 AA)
    bh_match = re.match(r"^(\d{2})BH(\d{4})([A-Z]{1,2})$", clean)
    if bh_match:
        return (f"{bh_match.group(1)} BH {bh_match.group(2)} {bh_match.group(3)}", 0.98)

    # 2. Standard Indian License Plate Format:
    # State (2 letters) + RTO (1-2 digits) + Series (0-3 letters) + Number (1-4 digits)
    if 7 <= len(clean) <= 10:
        st_cand = clean[:2]
        st_mapped = "".join(DIGIT_TO_LETTER.get(c, c) for c in st_cand)
        best_st = None

        if st_mapped in VALID_INDIAN_STATES:
            best_st = st_mapped
        else:
            # Common OCR letter confusion for state codes
            if st_cand[0] == "T" and st_cand[1] in "04KHON":
                best_st = "TN"
            elif st_cand[0] == "K" and st_cand[1] in "4A0":
                best_st = "KA"
            elif st_cand[0] == "D" and st_cand[1] in "1LI":
                best_st = "DL"
            elif st_cand[0] == "M" and st_cand[1] in "4HP":
                best_st = "MH"
            else:
                for s in VALID_INDIAN_STATES:
                    if st_mapped[0] == s[0] or st_mapped[1] == s[1]:
                        best_st = s
                        break

        if best_st:
            rem = clean[2:]
            rto_cand = rem[:2]
            rto_digits = "".join(LETTER_TO_DIGIT.get(c, c) for c in rto_cand)
            if rto_digits.isdigit():
                rem2 = rem[2:]
                if len(rem2) >= 4:
                    num_cand = rem2[-4:]
                    num_digits = "".join(LETTER_TO_DIGIT.get(c, c) for c in num_cand)
                    series_cand = rem2[:-4]
                    series_letters = "".join(DIGIT_TO_LETTER.get(c, c) for c in series_cand)
                    if num_digits.isdigit():
                        formatted = f"{best_st} {rto_digits} {series_letters} {num_digits}".strip()
                        formatted = re.sub(r"\s+", " ", formatted)
                        return (formatted, 0.96)

    # 3. Generic formatted alphanumeric plate
    if len(clean) >= 4:
        # Group into readable chunks (e.g., 2-3 letters, 2-4 digits)
        formatted = re.sub(r"([A-Z]+)(\d+)", r"\1 \2", clean)
        return (formatted, 0.60)

    return (clean, 0.30)


class TwoStagePlateDetector:
    """
    Precision Two-Stage Vehicle & License Plate Detector.
    """

    def __init__(
        self,
        car_model_path: str = "plate_identify/yolov8n.pt",
        plate_model_path: str = "plate_identify/plate_model.pt",
        car_conf: float = 0.35,
        plate_conf: float = 0.25,
    ):
        """
        Initialize the detector models and OCR engine.

        :param car_model_path: Path to vehicle YOLO weights.
        :param plate_model_path: Path to fine-tuned license plate YOLO weights.
        :param car_conf: Confidence threshold for vehicle detection.
        :param plate_conf: Confidence threshold for license plate detection.
        """
        self.car_model_path = Path(car_model_path)
        if not self.car_model_path.exists():
            fallback = Path(__file__).parent / self.car_model_path.name
            if fallback.exists():
                self.car_model_path = fallback

        self.plate_model_path = Path(plate_model_path)
        if not self.plate_model_path.exists():
            fallback = Path(__file__).parent / self.plate_model_path.name
            if fallback.exists():
                self.plate_model_path = fallback

        self.car_conf = car_conf
        self.plate_conf = plate_conf

        print(f"[TwoStagePlateDetector] Loading vehicle model: {self.car_model_path}")
        self.car_model = YOLO(str(self.car_model_path))

        print(f"[TwoStagePlateDetector] Loading plate model  : {self.plate_model_path}")
        self.plate_model = YOLO(str(self.plate_model_path))

        self.ocr_reader = None
        try:
            import easyocr
            print("[TwoStagePlateDetector] Initializing EasyOCR engine...")
            self.ocr_reader = easyocr.Reader(["en"], gpu=False)
            print("[TwoStagePlateDetector] EasyOCR engine ready!")
        except Exception as e:
            print(f"[TwoStagePlateDetector] EasyOCR unavailable ({e}); will try pytesseract fallback.")

    def extract_plate_number(
        self,
        plate_bgr: np.ndarray,
        global_xyxy: List[int],
        orig_w: int,
        orig_h: int,
    ) -> str:
        """
        Extract registration number from plate crop using high-precision multi-pass
        optical character recognition (EasyOCR / Pytesseract) and syntax disambiguation.
        """
        if plate_bgr is None or plate_bgr.size == 0:
            return ""

        h, w = plate_bgr.shape[:2]
        if h < 8 or w < 16:
            return ""

        # High-resolution rescaling: Target height 90-120px for clear character strokes
        target_h = max(90, min(140, int(h * 2.5)))
        scale = target_h / float(h)
        nw, nh = int(w * scale), target_h
        resized = cv2.resize(plate_bgr, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

        # Edge padding to prevent character cutoff
        pad = int(nh * 0.12)
        padded = cv2.copyMakeBorder(resized, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)

        # Multi-pass preprocessing variants
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
        clahe_high = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(6, 6)).apply(gray)
        unsharp = cv2.addWeighted(clahe_high, 1.5, cv2.GaussianBlur(clahe_high, (0, 0), 2), -0.5, 0)

        # Gamma enhancement for darker characters
        gamma_table = np.array([((i / 255.0) ** (1.0 / 1.3)) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_enhanced = cv2.LUT(clahe_high, gamma_table)

        # Otsu thresholding
        _, otsu = cv2.threshold(clahe_high, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        passes = [
            ("clahe_high", clahe_high),
            ("unsharp", unsharp),
            ("clahe", clahe),
            ("gamma", gamma_enhanced),
            ("gray", gray),
            ("otsu", otsu),
        ]

        candidates = []

        # 1. EasyOCR Primary Multi-Pass
        if self.ocr_reader is not None:
            for pass_name, p_img in passes:
                try:
                    res = self.ocr_reader.readtext(
                        p_img,
                        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                        detail=1,
                    )
                    if not res:
                        continue

                    # Sort text boxes top-to-bottom and left-to-right
                    sorted_res = sorted(
                        res,
                        key=lambda x: (int(x[0][0][1] // 30), x[0][0][0])
                    )
                    raw_str = " ".join([x[1] for x in sorted_res])
                    parsed_str, parse_score = clean_and_parse_plate(raw_str)
                    avg_ocr_conf = sum(x[2] for x in res) / len(res)
                    total_score = parse_score * 0.65 + avg_ocr_conf * 0.35

                    if parsed_str:
                        candidates.append((total_score, parsed_str, raw_str))
                except Exception:
                    continue

        # 2. Pytesseract Secondary Fallback if EasyOCR found no text
        if not candidates:
            try:
                import pytesseract
                for pass_name, p_img in passes:
                    try:
                        text = pytesseract.image_to_string(
                            p_img,
                            config="--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                        ).strip()
                        parsed_str, score = clean_and_parse_plate(text)
                        if parsed_str:
                            candidates.append((score, parsed_str, text))
                    except Exception:
                        continue
            except ImportError:
                pass

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        # If OCR could not confidently read the plate, return empty string
        # (DO NOT hallucinate or synthesize fake registration numbers)
        return ""

    def detect(
        self, image_input: Union[str, np.ndarray], padding_pct: float = 0.02
    ) -> Dict[str, Any]:
        """
        Execute two-stage detection on an image.

        :param image_input: File path or cv2 BGR image array.
        :param padding_pct: Percentage of bounding box to pad vehicle crop (ensures plate edges aren't clipped).
        :return: Comprehensive structured detection results including all coordinates.
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

        # Stage 1: Detect Vehicles
        car_results = self.car_model(image, conf=self.car_conf, verbose=False)
        detected_vehicles: List[Dict[str, Any]] = []
        total_plates_found = 0

        for r in car_results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.car_model.names.get(cls_id, str(cls_id))

                if cls_name.lower() in VEHICLE_CLASSES:
                    v_conf = float(box.conf[0])
                    vx1, vy1, vx2, vy2 = map(int, box.xyxy[0].tolist())

                    # Apply optional padding to avoid cutting off bumper/edges
                    pad_w = int((vx2 - vx1) * padding_pct)
                    pad_h = int((vy2 - vy1) * padding_pct)
                    crop_x1 = max(0, vx1 - pad_w)
                    crop_y1 = max(0, vy1 - pad_h)
                    crop_x2 = min(orig_w, vx2 + pad_w)
                    crop_y2 = min(orig_h, vy2 + pad_h)

                    # Crop vehicle Region of Interest
                    vehicle_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
                    crop_h, crop_w = vehicle_crop.shape[:2]

                    # Stage 2: Detect Plate inside the cropped vehicle ROI
                    plates_for_vehicle: List[Dict[str, Any]] = []
                    if crop_w > 10 and crop_h > 10:
                        plate_results = self.plate_model(
                            vehicle_crop, conf=self.plate_conf, verbose=False
                        )
                        for pr in plate_results:
                            for pbox in pr.boxes:
                                p_conf = float(pbox.conf[0])
                                px1, py1, px2, py2 = map(int, pbox.xyxy[0].tolist())

                                # Clamp crop coordinates
                                px1 = max(0, min(crop_w - 1, px1))
                                py1 = max(0, min(crop_h - 1, py1))
                                px2 = max(0, min(crop_w - 1, px2))
                                py2 = max(0, min(crop_h - 1, py2))

                                p_w = px2 - px1
                                p_h = py2 - py1

                                # Map back to global full-image coordinates
                                global_px1 = crop_x1 + px1
                                global_py1 = crop_y1 + py1
                                global_px2 = crop_x1 + px2
                                global_py2 = crop_y1 + py2

                                # Normalized YOLO format on full image
                                g_xc_norm = ((global_px1 + global_px2) / 2.0) / orig_w
                                g_yc_norm = ((global_py1 + global_py2) / 2.0) / orig_h
                                g_w_norm = p_w / float(orig_w)
                                g_h_norm = p_h / float(orig_h)

                                # Extract cropped plate patch
                                plate_crop_patch = vehicle_crop[py1:py2, px1:px2]

                                aspect_ratio = (
                                    round(p_w / float(p_h), 2) if p_h > 0 else 0.0
                                )

                                # Generous padding for OCR extraction to prevent character clipping at edges
                                pad_ocr_x = int(p_w * 0.08)
                                pad_ocr_y = int(p_h * 0.12)
                                ocr_px1 = max(0, px1 - pad_ocr_x)
                                ocr_py1 = max(0, py1 - pad_ocr_y)
                                ocr_px2 = min(crop_w, px2 + pad_ocr_x)
                                ocr_py2 = min(crop_h, py2 + pad_ocr_y)
                                ocr_patch = vehicle_crop[ocr_py1:ocr_py2, ocr_px1:ocr_px2]
                                if ocr_patch.size == 0:
                                    ocr_patch = plate_crop_patch

                                p_text = self.extract_plate_number(
                                    ocr_patch,
                                    [global_px1, global_py1, global_px2, global_py2],
                                    orig_w,
                                    orig_h,
                                )

                                plate_item = {
                                    "plate_id": len(plates_for_vehicle) + 1,
                                    "plate_text": p_text,
                                    "confidence": round(p_conf, 4),
                                    "bbox_crop_xyxy": [px1, py1, px2, py2],
                                    "bbox_global_xyxy": [
                                        global_px1,
                                        global_py1,
                                        global_px2,
                                        global_py2,
                                    ],
                                    "bbox_global_xywh": [
                                        global_px1,
                                        global_py1,
                                        p_w,
                                        p_h,
                                    ],
                                    "bbox_yolo_norm": [
                                        round(g_xc_norm, 6),
                                        round(g_yc_norm, 6),
                                        round(g_w_norm, 6),
                                        round(g_h_norm, 6),
                                    ],
                                    "aspect_ratio": aspect_ratio,
                                    "plate_crop_bgr": plate_crop_patch,
                                }
                                plates_for_vehicle.append(plate_item)
                                total_plates_found += 1

                    # Normalized vehicle box
                    v_w = vx2 - vx1
                    v_h = vy2 - vy1
                    v_xc_norm = ((vx1 + vx2) / 2.0) / orig_w
                    v_yc_norm = ((vy1 + vy2) / 2.0) / orig_h
                    v_w_norm = v_w / float(orig_w)
                    v_h_norm = v_h / float(orig_h)

                    vehicle_data = {
                        "vehicle_id": len(detected_vehicles) + 1,
                        "class_name": cls_name,
                        "confidence": round(v_conf, 4),
                        "bbox_xyxy": [vx1, vy1, vx2, vy2],
                        "bbox_xywh": [vx1, vy1, v_w, v_h],
                        "bbox_yolo_norm": [
                            round(v_xc_norm, 6),
                            round(v_yc_norm, 6),
                            round(v_w_norm, 6),
                            round(v_h_norm, 6),
                        ],
                        "vehicle_crop_bgr": vehicle_crop,
                        "plates_detected": [
                            {k: v for k, v in p.items() if k != "plate_crop_bgr"}
                            for p in plates_for_vehicle
                        ],
                        "_plate_patches": [
                            p["plate_crop_bgr"] for p in plates_for_vehicle
                        ],
                    }
                    detected_vehicles.append(vehicle_data)

        return {
            "source_image": str(img_path),
            "image_dimensions": {"width": orig_w, "height": orig_h},
            "vehicle_count": len(detected_vehicles),
            "plate_count": total_plates_found,
            "vehicles": detected_vehicles,
            "raw_image": image,
        }

    def draw_annotations(
        self,
        image: np.ndarray,
        vehicles: List[Dict[str, Any]],
        show_vehicle: bool = True,
        show_plate: bool = True,
    ) -> np.ndarray:
        """
        Draw color-coded annotations for both vehicle and license plate.
        - Vehicle: Emerald Green
        - Plate: Vibrant Amber / Red with coordinates tag
        """
        canvas = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        for v in vehicles:
            if show_vehicle:
                vx1, vy1, vx2, vy2 = v["bbox_xyxy"]
                v_label = f"{v['class_name'].upper()} {v['confidence']*100:.1f}%"
                cv2.rectangle(canvas, (vx1, vy1), (vx2, vy2), (0, 204, 102), 2)

                # Vehicle label
                (tw, th), bl = cv2.getTextSize(v_label, font, 0.55, 1)
                by1 = max(0, vy1 - th - bl - 4)
                cv2.rectangle(
                    canvas, (vx1, by1), (vx1 + tw + 6, vy1), (0, 204, 102), -1
                )
                cv2.putText(
                    canvas,
                    v_label,
                    (vx1 + 3, vy1 - bl - 2),
                    font,
                    0.55,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            if show_plate:
                for p in v["plates_detected"]:
                    px1, py1, px2, py2 = p["bbox_global_xyxy"]
                    p_str = p.get("plate_text", "")
                    p_label = f"{p_str} ({p['confidence']*100:.0f}%)" if p_str else f"PLATE {p['confidence']*100:.1f}%"

                    # Plate box in sharp red/orange
                    cv2.rectangle(canvas, (px1, py1), (px2, py2), (0, 69, 255), 2)

                    # Tag
                    (tw, th), bl = cv2.getTextSize(p_label, font, 0.45, 1)
                    py_tag = max(0, py1 - th - bl - 4)
                    cv2.rectangle(
                        canvas, (px1, py_tag), (px1 + tw + 6, py1), (0, 69, 255), -1
                    )
                    cv2.putText(
                        canvas,
                        p_label,
                        (px1 + 3, py1 - bl - 2),
                        font,
                        0.45,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

        return canvas


def main():
    parser = argparse.ArgumentParser(
        description="Two-Stage Pipeline: Detect Vehicle and License Plate with High Precision."
    )
    parser.add_argument(
        "--image",
        "-i",
        default="plate_identify/sample.jpg",
        help="Input image path.",
    )
    parser.add_argument(
        "--car-model",
        default="plate_identify/yolov8n.pt",
        help="YOLO model path for vehicles.",
    )
    parser.add_argument(
        "--plate-model",
        default="plate_identify/plate_model.pt",
        help="YOLO model path for license plates.",
    )
    parser.add_argument(
        "--car-conf",
        type=float,
        default=0.35,
        help="Vehicle confidence threshold.",
    )
    parser.add_argument(
        "--plate-conf",
        type=float,
        default=0.25,
        help="License plate confidence threshold.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="plate_identify/outputs",
        help="Output directory for visual results and crops.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save output images to disk.",
    )

    args = parser.parse_args()

    detector = TwoStagePlateDetector(
        car_model_path=args.car_model,
        plate_model_path=args.plate_model,
        car_conf=args.car_conf,
        plate_conf=args.plate_conf,
    )

    results = detector.detect(args.image)

    print("\n" + "=" * 65)
    print(" TWO-STAGE VEHICLE & LICENSE PLATE DETECTION REPORT")
    print("=" * 65)
    print(f"Source Image    : {results['source_image']}")
    print(
        f"Resolution      : {results['image_dimensions']['width']}x{results['image_dimensions']['height']}"
    )
    print(f"Vehicles Found  : {results['vehicle_count']}")
    print(f"Plates Found    : {results['plate_count']}")
    print("-" * 65)

    for v in results["vehicles"]:
        print(f"\n[Vehicle #{v['vehicle_id']}] {v['class_name']} ({v['confidence']*100:.1f}%)")
        print(f"  BBox Global (xyxy)  : {v['bbox_xyxy']}")
        print(f"  BBox Global (xywh)  : {v['bbox_xywh']}")
        print(f"  YOLO Norm (cx,cy,w,h): {v['bbox_yolo_norm']}")

        if not v["plates_detected"]:
            print("  --> No license plate localized on this vehicle.")
        else:
            for p in v["plates_detected"]:
                print(f"  --> [Plate #{p['plate_id']}] Conf: {p['confidence']*100:.1f}%")
                print(f"      Global BBox (xyxy) : {p['bbox_global_xyxy']}")
                print(f"      Global BBox (xywh) : {p['bbox_global_xywh']}")
                print(f"      YOLO Norm (cx,cy,w,h): {p['bbox_yolo_norm']}")
                print(f"      Crop Relative BBox : {p['bbox_crop_xyxy']}")
                print(f"      Aspect Ratio (w/h) : {p['aspect_ratio']}")

    if not args.no_save:
        out_dir = Path(args.output_dir)
        crops_dir = out_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        base_name = Path(args.image).stem

        # Draw master annotated image
        annotated = detector.draw_annotations(
            results["raw_image"], results["vehicles"]
        )
        out_master_path = out_dir / f"{base_name}_plate_detected.jpg"
        cv2.imwrite(str(out_master_path), annotated)

        # Save individual vehicle and plate crops
        for v in results["vehicles"]:
            v_id = v["vehicle_id"]
            v_crop_path = crops_dir / f"{base_name}_veh_{v_id}.jpg"
            cv2.imwrite(str(v_crop_path), v["vehicle_crop_bgr"])

            for idx, p_crop in enumerate(v["_plate_patches"]):
                p_crop_path = crops_dir / f"{base_name}_veh_{v_id}_plate_{idx+1}.jpg"
                cv2.imwrite(str(p_crop_path), p_crop)

        # Export JSON metadata
        export_data = {
            "source_image": results["source_image"],
            "image_dimensions": results["image_dimensions"],
            "vehicle_count": results["vehicle_count"],
            "plate_count": results["plate_count"],
            "vehicles": [
                {
                    k: val
                    for k, val in v.items()
                    if k not in ("vehicle_crop_bgr", "_plate_patches")
                }
                for v in results["vehicles"]
            ],
        }
        out_json_path = out_dir / f"{base_name}_two_stage.json"
        with open(out_json_path, "w") as f:
            json.dump(export_data, f, indent=2)

        print("-" * 65)
        print(f"Annotated Master Image : {out_master_path}")
        print(f"Cropped Vehicle & Plates: {crops_dir}")
        print(f"Detection Metadata JSON: {out_json_path}")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
