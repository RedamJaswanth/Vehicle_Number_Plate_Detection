"""
=============================================================================
VEHICLE & LICENSE PLATE DETECTION PIPELINE
=============================================================================
Architecture: YOLOv8 Object Detection + Custom Plate Detector + EasyOCR Engine
Author: Redam Jaswanth
=============================================================================
"""

import cv2
import os
import sys
import argparse
from typing import List, Dict, Any

from Src.Detector import VehicleDetector
from Src.Plate_Detector import PlateDetector
from Src.Ocr import PlateOCR
from Utils.Helpers import (
    crop_image,
    draw_vehicle,
    draw_plate,
    put_title
)

def run_pipeline(
    input_video: str = "Data/Input/Sample.mp4",
    output_video: str = "Data/Output/result.mp4",
    output_text: str = "Data/Output/result.txt",
    vehicle_detect_every: int = 4,
    plate_detect_every: int = 5,
    ocr_every: int = 15,
    display_width: int = 1280,
    display_height: int = 720,
    headless: bool = False
):
    # Ensure Output Directory Exists
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    os.makedirs(os.path.dirname(output_text), exist_ok=True)

    print("=" * 60)
    print("🚀 INITIALIZING AUTOMATED NUMBER PLATE RECOGNITION (ANPR)")
    print("=" * 60)

    # 1. Initialize Detectors
    print("[1/3] Loading YOLOv8 Vehicle Detector...")
    try:
        vehicle_detector = VehicleDetector()
        print("      ✓ Vehicle detector online.")
    except Exception as e:
        print(f"      ✗ Vehicle detector initialization warning: {e}")
        vehicle_detector = None

    print("[2/3] Loading License Plate Detector...")
    try:
        plate_detector = PlateDetector()
        print("      ✓ License plate detector online.")
    except Exception as e:
        print(f"      ✗ Plate detector initialization warning: {e}")
        plate_detector = None

    print("[3/3] Loading EasyOCR Text Recognition Engine...")
    try:
        ocr_engine = PlateOCR()
        print("      ✓ EasyOCR engine online.")
    except Exception as e:
        print(f"      ✗ OCR engine initialization warning: {e}")
        ocr_engine = None

    # Check Input Video Existence
    if not os.path.exists(input_video):
        print(f"\n[!] Input video '{input_video}' not found.")
        print(f"[*] Creating demo simulated feed placeholder in {input_video}...")
        os.makedirs(os.path.dirname(input_video), exist_ok=True)
        # Create a mock 3-second black/color test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        mock_writer = cv2.VideoWriter(input_video, fourcc, 20.0, (display_width, display_height))
        for frame_idx in range(60):
            mock_frame = (cv2.imread(None) if False else None) or (
                cv2.rectangle(
                    cv2.putText(
                        (cv2.Mat.zeros((display_height, display_width, 3), cv2.CV_8UC3) if hasattr(cv2, 'Mat') else None) or 
                        __import__('numpy').zeros((display_height, display_width, 3), dtype=__import__('numpy').uint8),
                        f"ANPR Video Stream Feed - Frame {frame_idx}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
                    ),
                    (400, 300), (880, 500), (0, 255, 0), 3
                )
            )
            mock_writer.write(mock_frame)
        mock_writer.release()
        print(f"      ✓ Sample video stream created at: {input_video}")

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video file: {input_video}")
        return

    # Video Properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (display_width, display_height))

    detected_plates_log = set()
    frame_count = 0
    cached_vehicles = []
    cached_plates = []

    print(f"\n[*] Processing Video Stream ({total_frames} frames @ {fps} FPS)...")
    print("-" * 60)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            resized_frame = cv2.resize(frame, (display_width, display_height))

            # Step 1: Detect Vehicles periodically
            if vehicle_detector and (frame_count % vehicle_detect_every == 0 or not cached_vehicles):
                cached_vehicles = vehicle_detector.detect(resized_frame)

            # Draw Vehicles
            for v in cached_vehicles:
                draw_vehicle(resized_frame, v["bbox"], v["class_name"], v["confidence"])

            # Step 2: Detect License Plates periodically
            if plate_detector and (frame_count % plate_detect_every == 0 or not cached_plates):
                cached_plates = plate_detector.detect(resized_frame)

                # Step 3: Run OCR periodically on detected plates
                if ocr_engine and (frame_count % ocr_every == 0):
                    for p in cached_plates:
                        plate_crop = crop_image(resized_frame, p["bbox"])
                        if plate_crop is not None:
                            text = ocr_engine.read_plate(plate_crop)
                            if text:
                                p["text"] = text
                                detected_plates_log.add(text)

            # Draw Plates
            for p in cached_plates:
                draw_plate(resized_frame, p["bbox"], p.get("text", "PLATE"), p["confidence"])

            # Overlay Status Banner
            put_title(resized_frame, f"AI Traffic ANPR Surveillance | Frame: {frame_count}/{total_frames}")

            out.write(resized_frame)

            if not headless:
                cv2.imshow("Real-Time Vehicle Plate Detection", resized_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[!] Execution interrupted by user.")
                    break

            if frame_count % 30 == 0:
                print(f"      Progress: Frame {frame_count}/{total_frames} processed...")

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    # Save Detected Plates Log
    with open(output_text, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("ANPR SYSTEM - EXTRACTED LICENSE PLATES LOG\n")
        f.write("=" * 50 + "\n")
        for plate in sorted(detected_plates_log):
            f.write(f"{plate}\n")

    print("=" * 60)
    print("✅ PIPELINE EXECUTION COMPLETE")
    print(f"   • Annotated Video Saved: {output_video}")
    print(f"   • Recognized Plates Log: {output_text} ({len(detected_plates_log)} unique plates)")
    print("=" * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Real-Time Vehicle & License Plate Detection")
    parser.add_argument("--input", default="Data/Input/Sample.mp4", help="Path to input video")
    parser.add_argument("--output", default="Data/Output/result.mp4", help="Path to output video")
    parser.add_argument("--headless", action="store_true", help="Run in headless server mode (no cv2.imshow)")
    args = parser.parse_args()

    run_pipeline(
        input_video=args.input,
        output_video=args.output,
        headless=args.headless
    )