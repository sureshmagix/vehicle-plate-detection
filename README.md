# ALPR: Two-Stage Vehicle & License Plate Detection with Character Segmentation

[![GitHub Pages](https://img.shields.io/badge/Demo-Live%20on%20GitHub%20Pages-brightgreen)](https://sureshmagix.github.io/alpr-character-segmentation/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-Ultralytics%20v8-blue)](https://github.com/ultralytics/ultralytics)
[![TensorFlow.js](https://img.shields.io/badge/Browser%20AI-TensorFlow.js-orange)](https://www.tensorflow.org/js)
[![License Plate Character OCR](https://img.shields.io/badge/OCR-EasyOCR%20%7C%20PyTesseract-purple)](https://github.com/JaidedAI/EasyOCR)

A high-performance ALPR (Automated License Plate Recognition) pipeline, character-level bounding box segmentation engine, and interactive dataset annotation suite built for dense, real-world traffic scenes.

👉 **Live Interactive Demo**: **[https://sureshmagix.github.io/alpr-character-segmentation/](https://sureshmagix.github.io/alpr-character-segmentation/)**

---

## 🌟 Key Features

### 1. Dedicated Zoomed License Plate Studio & Professional Number Bounding Boxes
* **Separately Magnified Plate Inspection**: Dedicated studio canvas rendering license plates at selectable zoom levels (2.5x, 4.0x, 6.0x, or fit width) with crisp nearest-neighbor / pixelated rendering.
* **Professional HUD Bounding Boxes**: Precision bounding boxes drawn against each individual number with corner bracket reticles (`┌ ┐ └ ┘`), semi-transparent character zone highlights, and floating monospace index pills (`[#1: K]`, `[#2: A]`, `[#3: 0]`...).
* **Interactive HUD Controls**: Instant toggles for number bounding boxes, index labels, and live raw vs. deblurred visual comparison.

### 2. Character Cutting & 1-Click Batch Exporter (Separate Studio)
* **Isolated Glyph Card Gallery**: Clean, high-density studio grid displaying each segmented character separately on velvet black background with dimensions ($w \times h$), aspect ratio, and OCR confidence.
* **1-Click ZIP Batch Download**: Prominent single-click button (`📦 Download All Characters in 1 Click (.ZIP)`) that packages all segmented character crops (`char_01_K.png`, `char_02_A.png`, etc.) along with JSON metadata into a zip archive and initiates immediate download.
* **One-Click Quick Actions**: Single-character `.png` download buttons on every glyph tile, plus single-click clipboard copying for formatted registration numbers and JSON bounding boxes.

### 3. Distinct Anti-Blur Preprocessing (Manual-Only · Default OFF)
* **Default-Off Safety**: All anti-blur preprocessing options are **disabled by default**, preventing unwanted distortion and executing only when manually enabled by the user.
* **Distinct Visual Sharpening**:
  * **Stage 1 (Full Frame / Vehicles)**: High-boost unsharp masking ($\alpha=2.2, \beta=-1.2$) + LAB lightness contrast expansion for vehicle silhouette clarity.
  * **Stage 2 (Vehicle Crop / Plates)**: Edge-boost sharpening + bilateral edge smoothing for plate boundary enhancement.
  * **Stage 3 (Plate Crop / Characters)**: Frequency-domain **Wiener Deconvolution** ($G(u,v) = \frac{H^*}{|H|^2 + K} F(u,v)$ via 2D FFT) + Laplacian high-boost filter + high-contrast CLAHE ($3.8$ clip limit) to make character strokes razor-sharp.
* **Multi-Intensity Control**: 1.0x (Standard), 1.8x (High), and 2.5x (Ultra) selectable intensities with active status indicator banners.

### 4. Strict Two-Stage Localization Hierarchy
Prevents false positives from background billboards, road signs, and roadside text by enforcing strict spatial localization:
1. **Stage 1**: Detects vehicle boundary (`car`, `truck`, `bus`, `motorcycle`).
2. **Stage 2**: Searches for license plate strictly within localized vehicle patches.
3. **Stage 3**: Segments individual characters, extracts glyph crops, and performs multi-pass OCR.

---

## 🚀 Quick Start (Python CLI)

### 1. Standalone Vehicle Detection
```bash
python3 detect_vehicle.py --image sample.jpg --conf 0.35 --antiblur-vehicle
```

### 2. Two-Stage Detection with Anti-Blur & Character Crops
```bash
python3 detect_plate.py \
  --image sample.jpg \
  --car-conf 0.35 \
  --plate-conf 0.25 \
  --extract-text \
  --antiblur-vehicle \
  --antiblur-plate \
  --antiblur-char \
  --output-dir outputs
```

### 3. Launch Local Web UI & API Server
```bash
python3 ui_server.py --port 8080
```
Then visit `http://localhost:8080` in your browser.

---

## 💻 Python API Example

```python
from detect_plate import TwoStagePlateDetector

detector = TwoStagePlateDetector(
    car_model_path="yolov8n.pt",
    plate_model_path="plate_model.pt",
    car_conf=0.35,
    plate_conf=0.25,
)

results = detector.detect(
    "sample.jpg",
    extract_text=True,
    extract_characters=True,
    antiblur_vehicle=True,
    antiblur_plate=True,
    antiblur_char=True,
)

for vehicle in results["vehicles"]:
    print(f"Vehicle #{vehicle['vehicle_id']} ({vehicle['class_name']}) at {vehicle['bbox_xyxy']}")
    for plate in vehicle["plates_detected"]:
        print(f"  Plate #{plate['plate_id']} -> Text: {plate['plate_text']} (Conf: {plate['confidence']:.2f})")
        print(f"  Found {len(plate['characters'])} character(s):")
        for char in plate["characters"]:
            print(f"    Char #{char['char_id']}: '{char['char_text']}' BBox: {char['bbox_plate_xyxy']}")
```

---

## 🧪 Testing

Run the full unit test suite:
```bash
python3 -m unittest discover tests
```

---

## 🌐 Deploying to GitHub Pages (`github.io`)

1. Repository Settings &rarr; **Pages**.
2. **Build and deployment** &rarr; Source: **Deploy from a branch**.
3. Branch: **`main`**, Folder: **`/docs`**.
4. Click **Save**.
Live site: `https://sureshmagix.github.io/alpr-character-segmentation/`
