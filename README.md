# Vehicle Number Plate Detection

A real-time vehicle and license plate detection system using YOLOv8, OpenCV, and EasyOCR.

## Features

- Vehicle detection using YOLOv8
- Car, motorcycle, bus, and truck detection
- License plate detection
- License plate text extraction using EasyOCR
- Real-time video processing
- Output video generation
- OpenCV-based visualization

## Technologies Used

- Python
- YOLOv8
- Ultralytics
- OpenCV
- EasyOCR
- NumPy

## Project Structure

```text
Vehicle_Number_Plate_Detection_New/
│
├── Data/
│   ├── Input/
│   └── Output/
│
├── Models/
│   ├── yolov8n.pt
│   └── license_plate_detector.pt
│
├── Src/
│   ├── __init__.py
│   ├── detector.py
│   ├── plate_detector.py
│   ├── ocr.py
│   └── main.py
│
├── Utils/
│   ├── __init__.py
│   └── helpers.py
│
├── requirements.txt
├── README.md
└── .gitignore