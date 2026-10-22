---
title: Vehicle Number Plate Detection
emoji: 🚗
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
---

# 🚗 Vehicle Number Plate Detection & OCR

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow?style=for-the-badge)](https://huggingface.co/spaces/Jaswanth1023/Vehicle_Number_Plate_Detection)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/RedamJaswanth/Vehicle_Number_Plate_Detection)

A computer vision pipeline that detects vehicles and license plates using fine-tuned **Ultralytics YOLOv8** and reads alphanumeric registration numbers using **EasyOCR**, deployed as an interactive **Gradio** web application on CPU.

---

### 🌐 Live Demo
👉 **[Open Live App on Hugging Face Spaces](https://huggingface.co/spaces/Jaswanth1023/Vehicle_Number_Plate_Detection)**

---

### 📸 Application Interface

```
+-----------------------------------------------------------------------------------+
|  🚗 Automatic Vehicle Number Plate Detection & OCR                                 |
|  -------------------------------------------------------------------------------- |
|  [ Upload Image or Select Example ]  |  [ Annotated Detection Image ]             |
|  Confidence Threshold: [----O----]   |  - Bounding Box: [x1, y1, x2, y2]          |
|  [🔍 Detect & Read License Plate]    |  - Text: [ KUS1VSU ]                       |
|                                      |  - Confidence: 91.4%                       |
+-----------------------------------------------------------------------------------+
```
*(Screenshot placeholder: place screenshot at `assets/screenshot.png`)*

---

### ✨ Key Features

- **Automated Plate Localization**: Trained YOLOv8 deep learning model specialized in identifying license plates under varied lighting and angles.
- **Optical Character Recognition (OCR)**: Integrated EasyOCR pipeline with preprocessing (adaptive scaling and grayscale enhancement) for character extraction.
- **Interactive Confidence Tuning**: Dynamically adjust detection thresholds to balance precision and recall.
- **One-Click Sample Showcase**: Preloaded vehicle images for instant demonstration.
- **Optimized for CPU Inference**: Lightweight `opencv-python-headless` and CPU-optimized PyTorch execution for free-tier cloud hosting.

---

### 🛠️ Technology Stack

| Component | Technology |
|---|---|
| **Object Detection** | Ultralytics YOLOv8 (`best.pt`) |
| **Character Recognition** | EasyOCR |
| **Image Processing** | OpenCV (`opencv-python-headless`) |
| **Frontend Framework** | Gradio |
| **Deployment Platform** | Hugging Face Spaces (CPU basic) |

---

### ⚡ How to Run Locally

One-line run command:
```bash
pip install -r requirements.txt && python app.py
```

Opens at **http://127.0.0.1:7860**
