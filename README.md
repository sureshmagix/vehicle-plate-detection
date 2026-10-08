# Deep ALPR: Two-Stage Vehicle & License Plate Detection

[![GitHub Pages](https://img.shields.io/badge/Demo-Live%20on%20GitHub%20Pages-brightgreen)](https://sureshmagix.github.io/vehicle-plate-detection/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-Ultralytics%20v8-blue)](https://github.com/ultralytics/ultralytics)
[![TensorFlow.js](https://img.shields.io/badge/Browser%20AI-TensorFlow.js-orange)](https://www.tensorflow.org/js)

A high-performance Two-Stage ALPR (Automated License Plate Recognition) system and dataset annotation tool designed for dense, real-world traffic scenes.

👉 **Live Interactive Demo**: **[https://sureshmagix.github.io/vehicle-plate-detection/](https://sureshmagix.github.io/vehicle-plate-detection/)**

---

## What Makes This Two-Stage Architecture Superior?

In high-density traffic (such as multi-lane Indian highways and urban corridors), direct one-stage plate spotters fail due to severe false positives caused by advertising boards, truck decals, and roadside hoardings.

This pipeline enforces strict spatial hierarchy:
1. **Stage 1 (Vehicle Detection)**: Identifies the vehicle bounding box (`car`, `truck`, `bus`, `motorcycle`).
2. **Stage 2 (Plate Localization)**: Searches for the number plate **strictly within the lower 65% bumper/grille region of each localized vehicle**.
3. **Coordinate Inversion**: Maps local crop coordinates back to full image space.

---

## Default Sample Detections (Dense Traffic Frame, 960 &times; 540 px)

In the default highway traffic scene (`sample.jpg`), the system successfully resolves multiple vehicles:

| Target Vehicle | Class & Conf | Bounding Box $[X_1, Y_1, X_2, Y_2]$ | License Plate Text | Plate Global BBox |
| :--- | :--- | :--- | :--- | :--- |
| **Hyundai Creta (White SUV)** | Car (89.2%) | `[424, 279, 742, 534]` | **`KA 09 MB 7389`** | `[506, 445, 593, 476]` (72.7%) |
| **Toyota Innova (Silver MPV)** | Car (74.3%) | `[63, 240, 433, 528]` | **`KA 05 AB 8135`** | `[155, 455, 217, 479]` (92.0%) |

---

## Interactive UI Features (`https://sureshmagix.github.io/vehicle-plate-detection/`)

* **📚 Methods & Algorithms Learning Hub**: Dedicated educational reference tab explaining all classical and deep learning methods for identifying vehicles (GMM, HOG+SVM, Faster R-CNN, YOLO, RT-DETR, Vehicle ReID) and license plates (Vertical Edge Dominance, OBB, 4-corner STN Homography unwarping, CRNN+CTC, Vision Transformers) with a complete comparison matrix.
* **Browse Any Image**: Drag and drop or browse any traffic or car image from your device.
* **Separated Car & Number Plate Inspector**: Displays each detected car crop and its corresponding high-resolution number plate crop side-by-side with exact dimensions and coordinates.
* **🧹 Clear Boxes Option**: Click "Clear Boxes" at any time to inspect the clean, unannotated image.
* **✏️ Draw Custom Box**: Click and drag anywhere on the canvas to manually define, verify, or adjust bounding boxes.
* **Focus on Canvas**: Click "Focus" on any vehicle card to highlight that vehicle and plate alone on the main viewport.
* **One-Click AI Training Export**:
  * Copy normalized YOLO training annotations: `0 cx cy w h` (Vehicle) and `1 cx cy w h` (Plate).
  * Copy complete JSON metadata.

---

## Quick Start (Python CLI)

### 1. Detect Vehicle Alone
```bash
python3 detect_vehicle.py --image sample.jpg --conf 0.35
```

### 2. Detect Both Vehicle & License Plate (Two-Stage)
```bash
python3 detect_plate.py --image sample.jpg --car-conf 0.35 --plate-conf 0.20
```

### 3. Launch Local Web Server & API
```bash
python3 ui_server.py --port 8080
```
Then visit `http://localhost:8080`.

---

## Python API Integration

```python
from detect_plate import TwoStagePlateDetector

detector = TwoStagePlateDetector(
    car_model_path="yolov8n.pt",
    plate_model_path="plate_model.pt",
    car_conf=0.35,
    plate_conf=0.20
)

results = detector.detect("sample.jpg")

for vehicle in results["vehicles"]:
    print(f"Vehicle: {vehicle['class_name']} -> {vehicle['bbox_xyxy']}")
    for plate in vehicle["plates_detected"]:
        print(f"  Plate: {plate['bbox_global_xyxy']} (Conf: {plate['confidence']:.2f})")
```
