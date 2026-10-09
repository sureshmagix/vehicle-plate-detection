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


def _clean_plate_text(raw_text: str) -> str:
    """Return OCR text in a comparison-friendly form without guessing characters."""
    clean = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
    # The blue HSRP band occasionally enters the OCR result.  It is metadata,
    # not part of the registration number.
    return clean[3:] if clean.startswith("IND") and len(clean) >= 7 else clean


def _map_characters(value: str, replacements: Dict[str, str]) -> Tuple[str, int]:
    """Map OCR look-alikes only when the plate grammar identifies the field."""
    mapped = "".join(replacements.get(char, char) for char in value)
    substitutions = sum(left != right for left, right in zip(value, mapped))
    return mapped, substitutions


def clean_and_parse_plate(raw_text: str) -> Tuple[str, float]:
    """Parse an Indian registration only when the OCR text supports it.

    Character look-alikes are corrected *only* in a known letter or number
    field.  In particular, this function never changes an unknown state code
    to the first superficially similar state; that behaviour can turn a weak
    OCR read into a plausible but incorrect registration number.
    """
    clean = _clean_plate_text(raw_text)
    if len(clean) < 3:
        return ("", 0.0)

    # Bharat Series, e.g. 22 BH 1234 AA.  The fixed "BH" is intentionally
    # not inferred from a similar-looking OCR sequence.
    bh_match = re.fullmatch(r"(\d{2})BH(\d{4})([A-Z]{1,2})", clean)
    if bh_match:
        return (f"{bh_match.group(1)} BH {bh_match.group(2)} {bh_match.group(3)}", 0.99)

    # Standard Indian plate: state (2 letters), RTO (1-2 digits), optional
    # series (0-3 letters), and a 1-4 digit running number.  Try the possible
    # field lengths rather than assuming every RTO and running number is two
    # and four characters respectively.
    if 6 <= len(clean) <= 11:
        state, state_changes = _map_characters(clean[:2], DIGIT_TO_LETTER)
        if state in VALID_INDIAN_STATES:
            body = clean[2:]
            parsed_options = []
            for rto_length in (2, 1):
                for number_length in range(4, 0, -1):
                    series_length = len(body) - rto_length - number_length
                    if not 0 <= series_length <= 3:
                        continue

                    rto_raw = body[:rto_length]
                    series_raw = body[rto_length:rto_length + series_length]
                    number_raw = body[-number_length:]
                    rto, rto_changes = _map_characters(rto_raw, LETTER_TO_DIGIT)
                    series, series_changes = _map_characters(series_raw, DIGIT_TO_LETTER)
                    number, number_changes = _map_characters(number_raw, LETTER_TO_DIGIT)
                    if not (rto.isdigit() and (not series or series.isalpha()) and number.isdigit()):
                        continue

                    changes = state_changes + rto_changes + series_changes + number_changes
                    # Prefer normal two-digit RTO / four-digit registration
                    # fields, but retain valid shorter plate forms.
                    preference = (rto_length == 2) * 2 + (number_length == 4)
                    parsed_options.append((preference, -changes, rto, series, number, changes))

            if parsed_options:
                _, _, rto, series, number, changes = max(parsed_options)
                formatted = " ".join(part for part in (state, rto, series, number) if part)
                # Syntax contributes confidence, but substitutions still lower
                # it so OCR consensus remains the deciding signal.
                return (formatted, max(0.82, 0.99 - 0.03 * changes))

    # Preserve a non-destructive suggestion for manual review.  This value is
    # never treated as a verified registration by the OCR pipeline.
    if len(clean) >= 4:
        return (re.sub(r"([A-Z]+)(\d+)", r"\1 \2", clean), 0.25)
    return (clean, 0.0)


def deblur_vehicle_stage(image: np.ndarray) -> np.ndarray:
    """
    Stage 1 Anti-Blur: Sharpening and unsharp masking on the full image
    prior to vehicle localization.
    """
    if image is None or image.size == 0:
        return image
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def deblur_plate_stage(vehicle_crop: np.ndarray) -> np.ndarray:
    """
    Stage 2 Anti-Blur: Deblurring on the vehicle crop prior to plate localization.
    Uses edge-boost unsharp masking to enhance plate boundary and border contours.
    """
    if vehicle_crop is None or vehicle_crop.size == 0:
        return vehicle_crop
    blurred = cv2.GaussianBlur(vehicle_crop, (0, 0), sigmaX=2.5)
    sharpened = cv2.addWeighted(vehicle_crop, 1.7, blurred, -0.7, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def deblur_char_stage(plate_crop: np.ndarray) -> np.ndarray:
    """
    Stage 3 Anti-Blur: Wiener deconvolution, unsharp masking, and contrast enhancement
    (CLAHE) on the plate crop before character segmentation and OCR.
    """
    if plate_crop is None or plate_crop.size == 0:
        return plate_crop
    h, w = plate_crop.shape[:2]
    if h < 8 or w < 8:
        return plate_crop

    # Work in LAB color space to preserve chromaticity while enhancing luminance
    lab = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    # 1. Frequency-Domain Wiener Deconvolution
    kernel_size = 5
    psf = cv2.getGaussianKernel(kernel_size, 1.2)
    psf = psf @ psf.T
    kh, kw = psf.shape
    pad_psf = np.zeros((h, w), dtype=np.float32)
    pad_psf[:kh, :kw] = psf
    pad_psf = np.roll(pad_psf, -kh // 2, axis=0)
    pad_psf = np.roll(pad_psf, -kw // 2, axis=1)

    l_fft = np.fft.fft2(l_chan.astype(np.float32))
    psf_fft = np.fft.fft2(pad_psf)
    psf_conj = np.conj(psf_fft)
    psf_pow = np.abs(psf_fft) ** 2
    noise_var = 0.015
    wiener_filter = psf_conj / (psf_pow + noise_var)
    deconv_l = np.real(np.fft.ifft2(l_fft * wiener_filter))
    deconv_l = np.clip(deconv_l, 0, 255).astype(np.uint8)

    # 2. Laplacian / Gaussian Unsharp Masking
    blurred_l = cv2.GaussianBlur(deconv_l, (0, 0), sigmaX=1.5)
    sharp_l = cv2.addWeighted(deconv_l, 1.5, blurred_l, -0.5, 0)

    # 3. CLAHE Contrast Enhancement
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
    enh_l = clahe.apply(sharp_l)

    merged_lab = cv2.merge([enh_l, a_chan, b_chan])
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)


def draw_character_annotations(
    plate_bgr: np.ndarray,
    characters: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Visual Overlay: Draw colored bounding boxes and index tags around each detected
    character/digit directly on the plate preview.
    """
    if plate_bgr is None or plate_bgr.size == 0 or not characters:
        return plate_bgr if plate_bgr is not None else np.zeros((30, 80, 3), dtype=np.uint8)

    orig_h, orig_w = plate_bgr.shape[:2]
    # Upscale for crisp tag and text rendering
    target_h = max(90, min(140, int(orig_h * 2.5)))
    scale = target_h / float(orig_h)
    canvas = cv2.resize(plate_bgr, (int(orig_w * scale), target_h), interpolation=cv2.INTER_LANCZOS4)

    font = cv2.FONT_HERSHEY_SIMPLEX
    palette = [
        (255, 200, 0),   # Vibrant Cyan-Blue
        (0, 235, 255),   # Bright Amber
        (0, 255, 128),   # Emerald Green
        (255, 100, 220), # Magenta
        (0, 180, 255),   # Gold Orange
    ]

    for idx, ch in enumerate(characters):
        ox1, oy1, ox2, oy2 = ch["bbox_plate_xyxy"]
        sx1 = int(round(ox1 * scale))
        sy1 = int(round(oy1 * scale))
        sx2 = int(round(ox2 * scale))
        sy2 = int(round(oy2 * scale))

        color = palette[idx % len(palette)]
        # Character box
        cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), color, 2)

        # Index Tag banner above/inside the box
        label = f"#{ch['char_id']}:{ch['char_text']}" if ch.get("char_text") else f"#{ch['char_id']}"
        (tw, th), bl = cv2.getTextSize(label, font, 0.40, 1)
        tag_y1 = max(0, sy1 - th - bl - 4)
        tag_y2 = sy1
        tag_x2 = min(canvas.shape[1] - 1, sx1 + tw + 6)
        cv2.rectangle(canvas, (sx1, tag_y1), (tag_x2, tag_y2), color, -1)
        cv2.putText(
            canvas,
            label,
            (sx1 + 3, tag_y2 - bl - 2),
            font,
            0.40,
            (0, 0, 0),
            1,
            lineType=cv2.LINE_AA,
        )

    return canvas


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
        """Backward-compatible convenience wrapper for plate text only."""
        return self.extract_plate_recognition(
            plate_bgr, global_xyxy, orig_w, orig_h
        )["plate_text"]

    def extract_plate_recognition(
        self,
        plate_bgr: np.ndarray,
        global_xyxy: List[int],
        orig_w: int,
        orig_h: int,
    ) -> Dict[str, Any]:
        """
        Extract a registration number from a plate crop.

        A plate number is returned only if multiple OCR passes agree on a valid
        Indian plate grammar.  This is deliberately conservative: a wrong
        number is substantially worse than an empty result that asks for human
        review.  The raw OCR suggestion is returned separately for that review.
        """
        empty_result = {
            "plate_text": "",
            "plate_text_raw": "",
            "ocr_confidence": 0.0,
            "ocr_status": "unreadable",
        }
        if plate_bgr is None or plate_bgr.size == 0:
            return empty_result

        h, w = plate_bgr.shape[:2]
        if h < 8 or w < 16:
            return empty_result

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

        # Valid candidates include the OCR confidence as well as the grammar
        # score.  `raw_candidates` is retained only as a useful manual-review
        # hint; it must never become an automatic plate value.
        candidates: List[Dict[str, Any]] = []
        raw_candidates: List[Dict[str, Any]] = []

        def record_candidate(raw_text: str, ocr_confidence: float, source: str) -> None:
            clean_raw = _clean_plate_text(raw_text)
            if not clean_raw:
                return

            parsed_text, syntax_score = clean_and_parse_plate(clean_raw)
            candidate = {
                "raw": clean_raw,
                "text": parsed_text,
                "ocr_confidence": max(0.0, min(1.0, float(ocr_confidence))),
                "syntax_score": syntax_score,
                "source": source,
            }
            raw_candidates.append(candidate)
            # Scores under 0.8 are generic formatting suggestions, not a
            # validated plate grammar.
            if syntax_score >= 0.8:
                candidates.append(candidate)

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
                    avg_ocr_conf = sum(x[2] for x in res) / len(res)
                    record_candidate(raw_str, avg_ocr_conf, pass_name)
                except Exception:
                    continue

        # 2. Pytesseract secondary fallback if EasyOCR did not provide a
        # structurally valid plate.  It may still rescue an otherwise unreadable
        # crop, but is held to the same evidence threshold.
        if not candidates:
            try:
                import pytesseract
                for pass_name, p_img in passes:
                    try:
                        text = pytesseract.image_to_string(
                            p_img,
                            config="--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                        ).strip()
                        # pytesseract's string API does not expose a reliable
                        # aggregate confidence, so use a deliberately modest
                        # score and require repeatability below.
                        record_candidate(text, 0.45, f"tesseract:{pass_name}")
                    except Exception:
                        continue
            except ImportError:
                pass

        if candidates:
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for candidate in candidates:
                grouped.setdefault(candidate["text"], []).append(candidate)

            total_valid = len(candidates)
            ranked = []
            for text, group in grouped.items():
                mean_ocr = sum(item["ocr_confidence"] for item in group) / len(group)
                mean_syntax = sum(item["syntax_score"] for item in group) / len(group)
                support = len(group) / total_valid
                # Consensus prevents one preprocessing artifact from winning
                # merely because the grammar can make it look plausible.
                combined = mean_ocr * 0.60 + mean_syntax * 0.20 + support * 0.20
                ranked.append((combined, support, mean_ocr, text))

            combined, support, mean_ocr, text = max(ranked, key=lambda item: item[0])
            if combined >= 0.58 and (support >= 0.34 or mean_ocr >= 0.82):
                return {
                    "plate_text": text,
                    "plate_text_raw": max(grouped[text], key=lambda item: item["ocr_confidence"])["raw"],
                    "ocr_confidence": round(combined, 4),
                    "ocr_status": "recognized",
                }

        if raw_candidates:
            best_raw = max(raw_candidates, key=lambda item: item["ocr_confidence"])
            return {
                "plate_text": "",
                "plate_text_raw": best_raw["text"] or best_raw["raw"],
                "ocr_confidence": round(best_raw["ocr_confidence"], 4),
                "ocr_status": "manual_review",
            }

        return empty_result

    def extract_character_segments(
        self,
        plate_bgr: np.ndarray,
        recognized_text: str = "",
        ocr_confidence: float = 0.0,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Extract character bounding boxes from the localized plate using contour / connected
        component analysis (filtering by aspect ratio, height, area, and sorting left-to-right)
        combined with OCR character-level coordinates (EasyOCR / PyTesseract).

        :param plate_bgr: BGR license plate image patch.
        :param recognized_text: Known OCR string if already recognized.
        :param ocr_confidence: OCR confidence score.
        :return: Tuple of (characters_list, annotated_plate_bgr)
        """
        if plate_bgr is None or plate_bgr.size == 0:
            return [], plate_bgr

        orig_h, orig_w = plate_bgr.shape[:2]
        if orig_h < 8 or orig_w < 16:
            return [], plate_bgr

        # Target height scaling for reliable character stroke analysis
        target_h = max(90, min(140, int(orig_h * 2.5)))
        scale = target_h / float(orig_h)
        scaled_w = int(orig_w * scale)
        resized = cv2.resize(plate_bgr, (scaled_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        pad = 12
        padded = cv2.copyMakeBorder(resized, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        pad_h, pad_w = padded.shape[:2]
        gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)

        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)

        # Multi-pass binarization (Otsu & Adaptive Gaussian)
        _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        adapt = cv2.adaptiveThreshold(
            clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
        )

        raw_candidates = []
        for bin_mask in [otsu, adapt]:
            contours, _ = cv2.findContours(bin_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                aspect = w / float(h)
                area_pct = (w * h) / float(pad_w * pad_h)
                h_pct = h / float(target_h)

                # Filter by aspect ratio, height ratio, and area
                if 0.10 <= aspect <= 1.25 and 0.22 <= h_pct <= 0.95 and 0.003 <= area_pct <= 0.32:
                    # Filter out outer border artifacts
                    if x > 2 and (x + w) < (pad_w - 2) and y > 2 and (y + h) < (pad_h - 2):
                        raw_candidates.append((x, y, w, h))

        # Include OCR character-level coordinates if PyTesseract is available
        try:
            import pytesseract
            tess_boxes_str = pytesseract.image_to_boxes(
                padded,
                config="--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            )
            for line in tess_boxes_str.strip().split("\n"):
                parts = line.split()
                if len(parts) >= 6:
                    tx1, ty1, tx2, ty2 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                    top_y = pad_h - ty2
                    th = ty2 - ty1
                    tw = tx2 - tx1
                    if 0.10 <= tw / float(max(1, th)) <= 1.25 and 0.20 <= th / float(target_h) <= 0.95:
                        raw_candidates.append((tx1, top_y, tw, th))
        except Exception:
            pass

        # EasyOCR character coordinate interpolation fallback if single-line
        if not raw_candidates and self.ocr_reader is not None:
            try:
                ocr_details = self.ocr_reader.readtext(padded, detail=1)
                for bbox, txt, c_conf in ocr_details:
                    clean_t = _clean_plate_text(txt)
                    if clean_t:
                        bx1 = int(min(pt[0] for pt in bbox))
                        by1 = int(min(pt[1] for pt in bbox))
                        bx2 = int(max(pt[0] for pt in bbox))
                        by2 = int(max(pt[1] for pt in bbox))
                        span_w = bx2 - bx1
                        span_h = by2 - by1
                        char_step = span_w / float(len(clean_t))
                        for i_c in range(len(clean_t)):
                            cx = int(bx1 + i_c * char_step)
                            cw = int(char_step * 0.85)
                            raw_candidates.append((cx, by1, cw, span_h))
            except Exception:
                pass

        # Deduplicate and suppress overlapping boxes (NMS)
        raw_candidates.sort(key=lambda b: (b[0], b[1]))
        clean_boxes = []
        for b in raw_candidates:
            bx, by, bw, bh = b
            overlap = False
            for idx, cb in enumerate(clean_boxes):
                cx, cy, cw, ch = cb
                ix1 = max(bx, cx)
                iy1 = max(by, cy)
                ix2 = min(bx + bw, cx + cw)
                iy2 = min(by + bh, cy + ch)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    min_area = min(bw * bh, cw * ch)
                    if inter / float(min_area) > 0.45:
                        overlap = True
                        if bh * bw > ch * cw:
                            clean_boxes[idx] = b
                        break
            if not overlap:
                clean_boxes.append(b)

        # Sort: check for 2-line license plates, otherwise sort left-to-right
        if clean_boxes:
            y_centers = [b[1] + b[3] / 2.0 for b in clean_boxes]
            mean_y = sum(y_centers) / len(y_centers)
            top_line = [b for b in clean_boxes if (b[1] + b[3] / 2.0) < mean_y - target_h * 0.12]
            bottom_line = [b for b in clean_boxes if (b[1] + b[3] / 2.0) >= mean_y - target_h * 0.12]
            if len(top_line) >= 2 and len(bottom_line) >= 2:
                top_line.sort(key=lambda b: b[0])
                bottom_line.sort(key=lambda b: b[0])
                sorted_boxes = top_line + bottom_line
            else:
                sorted_boxes = sorted(clean_boxes, key=lambda b: b[0])
        else:
            sorted_boxes = []

        # Map to original coordinates, match OCR characters, and extract individual crops
        clean_ocr_str = _clean_plate_text(recognized_text) if recognized_text else ""
        num_clean_chars = len(clean_ocr_str)
        num_boxes = len(sorted_boxes)

        characters: List[Dict[str, Any]] = []
        for idx, (bx, by, bw, bh) in enumerate(sorted_boxes):
            rx1 = max(0, bx - pad)
            ry1 = max(0, by - pad)
            rx2 = min(scaled_w, rx1 + bw)
            ry2 = min(target_h, ry1 + bh)

            ox1 = max(0, min(orig_w - 1, int(round(rx1 / scale))))
            oy1 = max(0, min(orig_h - 1, int(round(ry1 / scale))))
            ox2 = max(ox1 + 1, min(orig_w, int(round(rx2 / scale))))
            oy2 = max(oy1 + 1, min(orig_h, int(round(ry2 / scale))))
            ow = ox2 - ox1
            oh = oy2 - oy1

            char_text = ""
            char_conf = ocr_confidence if ocr_confidence > 0 else 0.85
            if clean_ocr_str:
                if num_boxes == num_clean_chars:
                    char_text = clean_ocr_str[idx]
                elif idx < num_clean_chars:
                    char_text = clean_ocr_str[idx]

            # Individual character crop from resized high-res patch with padding
            crop_pad_x = max(1, int(bw * 0.08))
            crop_pad_y = max(1, int(bh * 0.08))
            cx1 = max(0, rx1 - crop_pad_x)
            cy1 = max(0, ry1 - crop_pad_y)
            cx2 = min(scaled_w, rx2 + crop_pad_x)
            cy2 = min(target_h, ry2 + crop_pad_y)
            char_patch = resized[cy1:cy2, cx1:cx2]
            if char_patch.size == 0 or char_patch.shape[0] < 4 or char_patch.shape[1] < 4:
                char_patch = plate_bgr[oy1:oy2, ox1:ox2]

            # Ensure high-res preview height (at least 48px)
            c_h, c_w = char_patch.shape[:2]
            if c_h < 48:
                c_scale = 48.0 / float(c_h)
                char_patch = cv2.resize(
                    char_patch,
                    (max(16, int(c_w * c_scale)), 48),
                    interpolation=cv2.INTER_LANCZOS4,
                )

            char_item = {
                "char_id": idx + 1,
                "char_text": char_text,
                "confidence": round(char_conf, 4),
                "bbox_plate_xyxy": [ox1, oy1, ox2, oy2],
                "bbox_plate_xywh": [ox1, oy1, ow, oh],
                "bbox_scaled_xyxy": [rx1, ry1, rx2, ry2],
                "aspect_ratio": round(ow / float(oh), 2) if oh > 0 else 1.0,
                "char_crop_bgr": char_patch,
            }
            characters.append(char_item)

        annotated_plate = draw_character_annotations(plate_bgr, characters)
        return characters, annotated_plate

    def detect(
        self,
        image_input: Union[str, np.ndarray],
        padding_pct: float = 0.02,
        extract_text: bool = False,
        extract_characters: bool = True,
        antiblur_vehicle: bool = False,
        antiblur_plate: bool = False,
        antiblur_char: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute two-stage detection on an image.

        :param image_input: File path or cv2 BGR image array.
        :param padding_pct: Percentage of bounding box to pad vehicle crop (ensures plate edges aren't clipped).
        :param extract_text: If True, run OCR recognition passes on cropped plates.
        :param extract_characters: If True, segment character bounding boxes and crops.
        :param antiblur_vehicle: Stage 1 sharpening/deblur on full image before vehicle detection.
        :param antiblur_plate: Stage 2 deblur on vehicle crop before plate detection.
        :param antiblur_char: Stage 3 Wiener deconvolution & contrast boost on plate crop before character segmentation.
        :return: Comprehensive structured detection results including all coordinates and crops.
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

        # Stage 1: Vehicle Detection (Optional anti-blur sharpening)
        vehicle_infer_image = deblur_vehicle_stage(image) if antiblur_vehicle else image
        car_results = self.car_model(vehicle_infer_image, conf=self.car_conf, verbose=False)
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

                    # Crop vehicle Region of Interest from raw image
                    vehicle_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
                    crop_h, crop_w = vehicle_crop.shape[:2]

                    # Stage 2: License Plate Detection (Optional anti-blur on vehicle crop)
                    plates_for_vehicle: List[Dict[str, Any]] = []
                    if crop_w > 10 and crop_h > 10:
                        plate_infer_crop = deblur_plate_stage(vehicle_crop) if antiblur_plate else vehicle_crop
                        plate_results = self.plate_model(
                            plate_infer_crop, conf=self.plate_conf, verbose=False
                        )
                        for pr in plate_results:
                            for pbox in pr.boxes:
                                p_conf = float(pbox.conf[0])
                                px1, py1, px2, py2 = map(int, pbox.xyxy[0].tolist())

                                # Clamp crop coordinates
                                px1 = max(0, min(crop_w - 1, px1))
                                py1 = max(0, min(crop_h - 1, py1))
                                py2 = max(0, min(crop_h - 1, py2))
                                px2 = max(0, min(crop_w - 1, px2))

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

                                # Extract cropped plate patch from raw vehicle crop
                                plate_crop_patch = vehicle_crop[py1:py2, px1:px2]

                                # Stage 3: Character & OCR Processing (Optional anti-blur deconvolution)
                                processed_plate_patch = deblur_char_stage(plate_crop_patch) if antiblur_char else plate_crop_patch

                                aspect_ratio = (
                                    round(p_w / float(p_h), 2) if p_h > 0 else 0.0
                                )

                                if extract_text:
                                    pad_ocr_x = int(p_w * 0.08)
                                    pad_ocr_y = int(p_h * 0.12)
                                    ocr_px1 = max(0, px1 - pad_ocr_x)
                                    ocr_py1 = max(0, py1 - pad_ocr_y)
                                    ocr_px2 = min(crop_w, px2 + pad_ocr_x)
                                    ocr_py2 = min(crop_h, py2 + pad_ocr_y)
                                    ocr_source = deblur_char_stage(vehicle_crop) if antiblur_char else vehicle_crop
                                    ocr_patch = ocr_source[ocr_py1:ocr_py2, ocr_px1:ocr_px2]
                                    if ocr_patch.size == 0:
                                        ocr_patch = processed_plate_patch

                                    recognition = self.extract_plate_recognition(
                                        ocr_patch,
                                        [global_px1, global_py1, global_px2, global_py2],
                                        orig_w,
                                        orig_h,
                                    )
                                else:
                                    recognition = {
                                        "plate_text": "",
                                        "plate_text_raw": "",
                                        "ocr_confidence": 0.0,
                                        "ocr_status": "crop_extracted",
                                    }

                                characters: List[Dict[str, Any]] = []
                                plate_annotated_img = None
                                if extract_characters:
                                    characters, plate_annotated_img = self.extract_character_segments(
                                        processed_plate_patch,
                                        recognition["plate_text"] or recognition["plate_text_raw"],
                                        recognition["ocr_confidence"],
                                    )

                                plate_item = {
                                    "plate_id": len(plates_for_vehicle) + 1,
                                    "plate_text": recognition["plate_text"],
                                    "plate_text_raw": recognition["plate_text_raw"],
                                    "ocr_confidence": recognition["ocr_confidence"],
                                    "ocr_status": recognition["ocr_status"],
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
                                    "plate_annotated_bgr": plate_annotated_img,
                                    "character_count": len(characters),
                                    "characters": [
                                        {k: val for k, val in ch.items() if k != "char_crop_bgr"}
                                        for ch in characters
                                    ],
                                    "_char_patches": [
                                        ch["char_crop_bgr"] for ch in characters
                                    ],
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
                            {
                                k: v
                                for k, v in p.items()
                                if k not in ("plate_crop_bgr", "plate_annotated_bgr", "_char_patches")
                            }
                            for p in plates_for_vehicle
                        ],
                        "_plate_patches": [
                            p["plate_crop_bgr"] for p in plates_for_vehicle
                        ],
                        "_plate_annotated_patches": [
                            p.get("plate_annotated_bgr") for p in plates_for_vehicle
                        ],
                        "_char_patches_per_plate": [
                            p.get("_char_patches", []) for p in plates_for_vehicle
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
                    p_label = f"PLATE {p['confidence']*100:.1f}%"

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
        "--extract-text",
        action="store_true",
        help="Run OCR text extraction on license plate crops.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save output images to disk.",
    )
    parser.add_argument(
        "--antiblur-vehicle",
        action="store_true",
        help="Stage 1: Apply sharpening and unsharp mask before vehicle localization.",
    )
    parser.add_argument(
        "--antiblur-plate",
        action="store_true",
        help="Stage 2: Apply deblurring on the vehicle crop before plate localization.",
    )
    parser.add_argument(
        "--antiblur-char",
        action="store_true",
        help="Stage 3: Apply Wiener deconvolution, unsharp mask, and CLAHE before character OCR and segmentation.",
    )
    parser.add_argument(
        "--no-characters",
        action="store_true",
        help="Disable character-level bounding box segmentation and cropping.",
    )

    args = parser.parse_args()

    detector = TwoStagePlateDetector(
        car_model_path=args.car_model,
        plate_model_path=args.plate_model,
        car_conf=args.car_conf,
        plate_conf=args.plate_conf,
    )

    results = detector.detect(
        args.image,
        extract_text=args.extract_text,
        extract_characters=not args.no_characters,
        antiblur_vehicle=args.antiblur_vehicle,
        antiblur_plate=args.antiblur_plate,
        antiblur_char=args.antiblur_char,
    )

    print("\n" + "=" * 65)
    print(" TWO-STAGE VEHICLE & LICENSE PLATE DETECTION REPORT")
    print("=" * 65)
    print(f"Source Image    : {results['source_image']}")
    print(
        f"Resolution      : {results['image_dimensions']['width']}x{results['image_dimensions']['height']}"
    )
    print(f"Vehicles Found  : {results['vehicle_count']}")
    print(f"Plates Found    : {results['plate_count']}")
    print(f"Anti-Blur Modes : Vehicle={args.antiblur_vehicle}, Plate={args.antiblur_plate}, Char={args.antiblur_char}")
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
                if p["plate_text"]:
                    print(f"      Registration Text : {p['plate_text']} (OCR {p['ocr_confidence']*100:.1f}%)")
                else:
                    print("      Extracted Crop    : Ready (Image Crop Saved)")

                if p.get("characters"):
                    print(f"      Characters Found  : {len(p['characters'])} character(s)")
                    for ch in p["characters"]:
                        label = f"'{ch['char_text']}'" if ch.get("char_text") else "glyph"
                        print(f"        [Char #{ch['char_id']}] {label} (Conf {ch['confidence']*100:.1f}%) | BBox Plate: {ch['bbox_plate_xyxy']}")

    if not args.no_save:
        out_dir = Path(args.output_dir)
        crops_dir = out_dir / "crops"
        char_crops_dir = crops_dir / "characters"
        char_crops_dir.mkdir(parents=True, exist_ok=True)
        # Also ensure local crops/characters folder exists
        local_char_crops_dir = Path("crops/characters")
        local_char_crops_dir.mkdir(parents=True, exist_ok=True)
        base_name = Path(args.image).stem

        # Draw master annotated image
        annotated = detector.draw_annotations(
            results["raw_image"], results["vehicles"]
        )
        out_master_path = out_dir / f"{base_name}_plate_detected.jpg"
        cv2.imwrite(str(out_master_path), annotated)

        # Save individual vehicle, plate, and character crops
        for v in results["vehicles"]:
            v_id = v["vehicle_id"]
            v_crop_path = crops_dir / f"{base_name}_veh_{v_id}.jpg"
            cv2.imwrite(str(v_crop_path), v["vehicle_crop_bgr"])

            for idx, p_crop in enumerate(v["_plate_patches"]):
                p_crop_path = crops_dir / f"{base_name}_veh_{v_id}_plate_{idx+1}.jpg"
                cv2.imwrite(str(p_crop_path), p_crop)

                # Save annotated plate preview with character bounding boxes
                if idx < len(v.get("_plate_annotated_patches", [])):
                    ann_patch = v["_plate_annotated_patches"][idx]
                    if ann_patch is not None and ann_patch.size > 0:
                        p_chars_ann_path = crops_dir / f"{base_name}_veh_{v_id}_plate_{idx+1}_chars.jpg"
                        cv2.imwrite(str(p_chars_ann_path), ann_patch)

                # Save individual character crops
                if idx < len(v.get("_char_patches_per_plate", [])):
                    char_patches = v["_char_patches_per_plate"][idx]
                    for c_idx, c_patch in enumerate(char_patches):
                        c_id = c_idx + 1
                        c_crop_path = char_crops_dir / f"{base_name}_veh_{v_id}_plate_{idx+1}_char_{c_id:02d}.png"
                        cv2.imwrite(str(c_crop_path), c_patch)
                        # Also save char_01.png etc. in crops/characters/
                        alias_path = local_char_crops_dir / f"char_{c_id:02d}.png"
                        cv2.imwrite(str(alias_path), c_patch)

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
                    if k not in (
                        "vehicle_crop_bgr",
                        "_plate_patches",
                        "_plate_annotated_patches",
                        "_char_patches_per_plate",
                    )
                }
                for v in results["vehicles"]
            ],
        }
        out_json_path = out_dir / f"{base_name}_two_stage.json"
        with open(out_json_path, "w") as f:
            json.dump(export_data, f, indent=2)

        print("-" * 65)
        print(f"Annotated Master Image   : {out_master_path}")
        print(f"Cropped Vehicle & Plates  : {crops_dir}")
        print(f"Individual Character Crops: {char_crops_dir}")
        print(f"Detection Metadata JSON   : {out_json_path}")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
