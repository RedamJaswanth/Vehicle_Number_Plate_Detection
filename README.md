# 🚗 Intelligent Vehicle & Automated Number Plate Recognition (ANPR) System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![EasyOCR](https://img.shields.io/badge/EasyOCR-Deep_Learning_OCR-FF6F00?style=for-the-badge)](https://github.com/JaidedAI/EasyOCR)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **Real-time multi-class vehicle detection, automated license plate localization, and high-accuracy alphanumeric character recognition using YOLOv8, OpenCV, and EasyOCR.**

---

## 📌 Executive Summary

Automatic Number Plate Recognition (ANPR / ALPR) is a critical pillar of intelligent transportation systems (ITS), automated tolling plazas, urban traffic surveillance, and parking security.

This repository implements an **end-to-end Computer Vision pipeline** that ingests high-definition video feeds or CCTV streams, tracks multi-category vehicles (cars, motorcycles, buses, trucks), isolates license plate bounding boxes, and performs character recognition to extract clean registration strings with output video rendering and structured text logs.

---

## 🏗️ System Architecture & Workflow

```text
[Input Video Stream / CCTV Feed]
               │
               ▼
   [Frame Preprocessing & Resizing]
               │
   ┌───────────┴────────────────────────────────┐
   │                                            │
   ▼                                            ▼
[YOLOv8 Vehicle Detection]           [YOLO Plate Detection]
(Cars, Trucks, Buses, Bikes)         (Localizes Plate ROI)
   │                                            │
   └───────────┬────────────────────────────────┘
               │
               ▼
     [Plate Region Cropping]
               │
               ▼
 [Binarization & Grayscale Scaling]
               │
               ▼
 [EasyOCR Alphanumeric Recognition]
               │
               ▼
   ┌───────────┴────────────────────┐
   │                                │
   ▼                                ▼
[Annotated Video Output]     [Structured Plate Text Log]
(Rendered Bounding Boxes)     (result.txt with Unique Tags)
```

---

## 🎯 Key Capabilities

- **Multi-Class Vehicle Detection:** Pretrained YOLOv8 detector recognizing cars, motorcycles, buses, and freight trucks.
- **Dedicated Plate Localization:** Specialized deep learning model trained specifically on vehicular license plate geometries.
- **Intelligent Frequency Scheduling:** Decouples heavy OCR inference from video frame rates (`VEHICLE_DETECT_EVERY=4`, `PLATE_DETECT_EVERY=5`, `OCR_EVERY=15`) to achieve smooth real-time FPS throughput.
- **Image Preprocessing for Enhanced OCR:** Applies cubic interpolation, resizing, and grayscale contrast enhancement before feeding cropped plates into EasyOCR.
- **Dual Output Delivery:** Generates annotated MP4 video (`result.mp4`) with color-coded bounding boxes alongside an exportable text registry of recognized vehicle license plates (`result.txt`).

---

## 📁 Repository Directory Structure

```text
Vehicle_Number_Plate_Detection/
│
├── requirements.txt            # Core framework dependencies (Ultralytics, OpenCV, EasyOCR)
├── README.md                   # Enterprise system documentation
├── .gitignore                  # Git exclusions for binary weights and output video dumps
│
├── Data/                       # Video Asset Storage
│   ├── Input/                  # Source video feeds (Sample.mp4)
│   └── Output/                 # Processed video (result.mp4) & plate log (result.txt)
│
├── Models/                     # Deep Learning Model Weights
│   ├── yolov8n.pt              # Nano YOLOv8 vehicle detector
│   └── license_plate_detector.pt # Fine-tuned plate detection weights
│
├── Src/                        # Core Processing Modules
│   ├── __init__.py
│   ├── Detector.py             # YOLOv8 Vehicle Detector class wrapper
│   ├── Plate_Detector.py       # YOLOv8 License Plate Detector class wrapper
│   ├── Ocr.py                  # EasyOCR text reader & alphanumeric cleaner
│   └── Main.py                 # CLI orchestration pipeline & stream processor
│
└── Utils/                      # Utility Subroutines
    ├── __init__.py
    └── Helpers.py              # Box drawing, cropping & HUD title overlay routines
```

---

## 🚀 Quickstart & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/RedamJaswanth/Vehicle_Number_Plate_Detection.git
cd Vehicle_Number_Plate_Detection
```

### 2. Set Up Python Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Model Weights Setup
Download or place your weights inside the `Models/` directory:
- `yolov8n.pt` (automatically downloaded by Ultralytics on first run)
- `license_plate_detector.pt`

### 4. Execute the Detection Pipeline
```bash
# Standard run with video window
python -m Src.Main --input Data/Input/Sample.mp4 --output Data/Output/result.mp4

# Headless server execution (ideal for cloud/Docker deployments)
python -m Src.Main --input Data/Input/Sample.mp4 --output Data/Output/result.mp4 --headless
```

---

## 📈 Detection Visual Styling Standards

| Detection Entity | Bounding Box Color | Label Format | Placement |
| :--- | :---: | :--- | :--- |
| **Vehicle** | `Green (0, 255, 0)` | `CLASS_NAME CONFIDENCE` (e.g., `CAR 0.85`) | Top-Left Anchor |
| **License Plate** | `Red (0, 0, 255)` | `RECOGNIZED_ALPHANUMERIC` (e.g., `KA05MH1234`) | Bottom/Top Auto-dock |
| **HUD Surveillance Bar** | `White Header` | `AI Traffic ANPR Surveillance \| Frame X/N` | Top Banner |

---

## 👨‍💻 Author & Maintainer

* **Author:** Redam Jaswanth
* **Email:** [jaanujaswanth4@gmail.com](mailto:jaanujaswanth4@gmail.com)
* **GitHub:** [@RedamJaswanth](https://github.com/RedamJaswanth)