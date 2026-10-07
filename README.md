# Precision Two-Stage Vehicle & License Plate Detection (ALPR)

A high-performance, modular Python system designed for traffic analysis and AI training pipelines.

This repository features:
1. **`detect_vehicle.py`**: Standalone vehicle detection (`car`, `truck`, `bus`, `motorcycle`) with pixel and normalized YOLO bounding boxes.
2. **`detect_plate.py`**: Precision Two-Stage ALPR pipeline (detects vehicle &rarr; crops Region of Interest &rarr; detects license plate &rarr; inverts coordinates to global image space).
3. **`ui_server.py`**: Lightweight interactive Web UI to browse/upload images, adjust confidence thresholds, inspect coordinates, and export annotations.
4. **`docs/index.html`**: Ready for **GitHub Pages (`username.github.io/repo`)** hosting with complete mathematical documentation, architecture diagrams, and interactive visualization.

---

## Folder Structure

```
plate_identify/
├── detect_vehicle.py       # Stage 1: Vehicle identification CLI & module
├── detect_plate.py         # Two-Stage: Vehicle + License Plate detector
├── ui_server.py            # Local interactive Web UI & REST API server
├── yolov8n.pt              # Pretrained vehicle detection model weights
├── plate_model.pt          # Fine-tuned license plate detection weights
├── sample.jpg              # Provided highway traffic test image
├── requirements.txt        # Python dependency list
├── outputs/                # Automatically generated results, crops & JSON
│   ├── sample_vehicle_detected.jpg
│   ├── sample_plate_detected.jpg
│   ├── sample_two_stage.json
│   └── crops/
│       ├── sample_veh_1.jpg
│       └── sample_veh_1_plate_1.jpg
└── docs/                   # GitHub Pages static documentation & web app
    ├── index.html
    ├── sample.jpg
    ├── sample_plate_detected.jpg
    └── crops/
```

---

## 1. Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### 1. Detect Vehicle Alone
Run `detect_vehicle.py` on any image:
```bash
python3 detect_vehicle.py --image sample.jpg --conf 0.35
```
**Outputs:**
- Terminal prints: Pixel bounding box `[x1, y1, x2, y2]`, `[x, y, w, h]`, area, and normalized YOLO format `(cx, cy, w, h)`.
- Saves annotated image to `outputs/sample_vehicle_detected.jpg`.
- Saves JSON coordinates to `outputs/sample_vehicles.json`.

---

### 2. Detect Both Vehicle & License Plate (Two-Stage)
Run `detect_plate.py` for high-precision detection:
```bash
python3 detect_plate.py --image sample.jpg --car-conf 0.35 --plate-conf 0.25
```
**Sample Output:**
```
=================================================================
 TWO-STAGE VEHICLE & LICENSE PLATE DETECTION REPORT
=================================================================
Source Image    : plate_identify/sample.jpg
Resolution      : 1024x576
Vehicles Found  : 1
Plates Found    : 1
-----------------------------------------------------------------
[Vehicle #1] car (93.3%)
  BBox Global (xyxy)  : [122, 149, 482, 360]
  BBox Global (xywh)  : [122, 149, 360, 211]
  YOLO Norm (cx,cy,w,h): [0.294922, 0.44184, 0.351562, 0.366319]
  --> [Plate #1] Conf: 83.6%
      Global BBox (xyxy) : [161, 305, 223, 330]
      Global BBox (xywh) : [161, 305, 62, 25]
      YOLO Norm (cx,cy,w,h): [0.1875, 0.551215, 0.060547, 0.043403]
      Crop Relative BBox : [46, 160, 108, 185]
      Aspect Ratio (w/h) : 2.48
-----------------------------------------------------------------
Annotated Master Image : outputs/sample_plate_detected.jpg
Cropped Vehicle & Plates: outputs/crops/
Detection Metadata JSON: outputs/sample_two_stage.json
=================================================================
```

---

## 2. Interactive Web UI

Launch the built-in server:
```bash
python3 ui_server.py --port 8080
```
Open **`http://localhost:8080`** in any web browser.

**Features:**
- **Browse & Drag-and-Drop**: Upload and test any highway or traffic image.
- **Interactive Canvas**: View color-coded bounding boxes (Emerald for Vehicle, Vibrant Red/Amber for Plate).
- **Coordinate Inspector**: Real-time table displaying exact pixel coordinates and normalized YOLO training formats.
- **Cropped Patches**: High-resolution zoom views of the extracted vehicle and license plate.
- **One-Click Export**: Copy YOLO training labels or full JSON metadata.

---

## 3. Integration into an AI Training Pipeline

Both scripts can be imported directly into your training or annotation scripts:

```python
# Integration Example
from detect_plate import TwoStagePlateDetector

detector = TwoStagePlateDetector(
    car_model_path="yolov8n.pt",
    plate_model_path="plate_model.pt",
    car_conf=0.35,
    plate_conf=0.25
)

# Run detection
results = detector.detect("sample.jpg")

for vehicle in results["vehicles"]:
    # Vehicle normalized coordinates for training:
    v_cx, v_cy, v_w, v_h = vehicle["bbox_yolo_norm"]
    
    # Save training label: class_id 0 = vehicle
    with open("dataset/labels/frame_001.txt", "a") as f:
        f.write(f"0 {v_cx} {v_cy} {v_w} {v_h}\n")
        
        # Save plate label: class_id 1 = license_plate
        for plate in vehicle["plates_detected"]:
            p_cx, p_cy, p_w, p_h = plate["bbox_yolo_norm"]
            f.write(f"1 {p_cx} {p_cy} {p_w} {p_h}\n")
```

---

## 4. Publishing to GitHub Pages (`username.github.io`)

The `docs/` folder contains the static documentation and interactive web application ready for hosting:

1. Create a repository on GitHub (e.g. `vehicle-plate-detection`).
2. Push your project:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of two-stage ALPR system"
   git remote add origin https://github.com/<your-username>/vehicle-plate-detection.git
   git branch -M main
   git push -u origin main
   ```
3. Enable GitHub Pages:
   - Go to **Settings** &rarr; **Pages** in your GitHub repository.
   - Under **Build and deployment** &gt; **Source**, select **Deploy from a branch**.
   - Under **Branch**, select `main` and folder `/docs`.
   - Click **Save**.
4. Your study documentation will be live at:
   `https://<your-username>.github.io/vehicle-plate-detection/`
# vehicle-plate-detection
