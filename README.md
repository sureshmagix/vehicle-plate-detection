# ALPR: Two-Stage Vehicle & License Plate Detection with Character Segmentation

[![GitHub Pages](https://img.shields.io/badge/Demo-Live%20on%20GitHub%20Pages-brightgreen)](https://sureshmagix.github.io/alpr-character-segmentation/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-Ultralytics%20v8-blue)](https://github.com/ultralytics/ultralytics)
[![TensorFlow.js](https://img.shields.io/badge/Browser%20AI-TensorFlow.js-orange)](https://www.tensorflow.org/js)
[![License Plate Character OCR](https://img.shields.io/badge/OCR-EasyOCR%20%7C%20PyTesseract-purple)](https://github.com/JaidedAI/EasyOCR)

A high-performance ALPR (Automated License Plate Recognition) pipeline, character-level bounding box segmentation engine, and interactive dataset annotation suite built for dense, real-world traffic scenes.

👉 **Live Interactive Demo**: **[https://sureshmagix.github.io/alpr-character-segmentation/](https://sureshmagix.github.io/alpr-character-segmentation/)**

---

## 🌟 Key Features

### 1. Character-Level Bounding Boxes & Cropping
* **Contour & Connected Component Analysis**: Normalizes license plates, applies CLAHE contrast equalization, adaptive/Otsu multi-thresholding, and extracts individual character contours filtered by aspect ratio ($0.10 \le w/h \le 1.25$), relative height, and area.
* **OCR Character-Level Coordinates Fusion**: Correlates contour positions with EasyOCR token boundaries and PyTesseract character boxes, matching characters to their semantic labels (`K`, `A`, `0`, `9`, etc.).
* **Visual Overlay**: Renders color-coded bounding boxes and index tags (`#1:K`, `#2:A`...) directly on the plate preview.
* **Individual Character Crops**: Automatically isolates, upscales ($\ge 48\text{ px}$), and saves each character glyph to disk (`crops/characters/char_01.png`, etc.) and returns base64 PNG previews in the Web UI.

### 2. Per-Stage Anti-Blur Checkbox Controls
* **Stage 1 (Vehicle Detection)**: Optional sharpening / unsharp masking on the full image prior to vehicle localization (`--antiblur-vehicle`).
* **Stage 2 (Plate Detection)**: Optional edge-boost unsharp filtering on the vehicle crop prior to license plate localization (`--antiblur-plate`).
* **Stage 3 (Character OCR & Segmentation)**: Optional frequency-domain **Wiener Deconvolution** ($G(u,v) = \frac{H^*}{|H|^2 + K} F(u,v)$ via 2D FFT) + unsharp mask + CLAHE contrast enhancement on the plate crop before character segmentation (`--antiblur-char`).

### 3. Strict Two-Stage Hierarchy
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
