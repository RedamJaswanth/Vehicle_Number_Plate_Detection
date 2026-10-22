"""
app.py
------
Vehicle Number Plate Detection and OCR Recognition App
Built with Ultralytics YOLOv8, EasyOCR, and Gradio.
"""

import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import easyocr
from ultralytics import YOLO
import gradio as gr

# ── Load Models ─────────────────────────────────────────────────────────────────
# Locate weights in root (best.pt) or Models directory
WEIGHT_PATH = "best.pt" if os.path.exists("best.pt") else "Models/license_plate_detector.pt"

print(f"Loading YOLOv8 plate detector from: {WEIGHT_PATH}")
detector = YOLO(WEIGHT_PATH)

print("Initializing EasyOCR reader (CPU mode)...")
ocr_reader = easyocr.Reader(["en"], gpu=False)


def clean_plate_text(raw_text: str) -> str:
    """Clean and standardize recognized license plate text."""
    # Retain alphanumeric characters
    cleaned = "".join(c for c in raw_text if c.isalnum()).upper()
    return cleaned


def detect_and_recognize(image: np.ndarray, conf_threshold: float = 0.25):
    """
    Run YOLOv8 plate detection and EasyOCR recognition on an input image.

    Returns:
        annotated_image (np.ndarray): Image with drawn bounding boxes and text.
        results_markdown (str): Formatted summary of detected plates.
    """
    if image is None:
        return None, "Please upload or select an image."

    # Convert RGB to BGR for OpenCV processing
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w = frame.shape[:2]

    # Run inference
    results = detector(frame, conf=conf_threshold, device="cpu", verbose=False)

    detected_plates = []
    annotated_frame = frame.copy()

    for result in results:
        if result.boxes is None:
            continue

        for i, box in enumerate(result.boxes):
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Clamp coordinates to image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # Crop license plate region
            plate_crop = frame[y1:y2, x1:x2]

            # Run OCR on cropped plate
            ocr_text = ""
            if plate_crop.size > 0:
                # Preprocess for OCR: grayscale & upscale
                gray_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                gray_crop = cv2.resize(gray_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                raw_ocr = ocr_reader.readtext(gray_crop, detail=0, paragraph=False)
                if raw_ocr:
                    combined = "".join(raw_ocr)
                    ocr_text = clean_plate_text(combined)

            display_label = ocr_text if ocr_text else f"Plate ({conf:.2f})"
            detected_plates.append({
                "id": len(detected_plates) + 1,
                "text": ocr_text if ocr_text else "Unrecognized",
                "confidence": f"{conf * 100:.1f}%",
                "bbox": f"[{x1}, {y1}, {x2}, {y2}]",
            })

            # Draw bounding box (Vibrant Cyan-Green)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 128), 3)

            # Draw background tag for text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(display_label, font, font_scale, thickness)

            tag_y1 = max(0, y1 - text_h - 10)
            tag_y2 = y1
            tag_x2 = min(w, x1 + text_w + 10)

            cv2.rectangle(annotated_frame, (x1, tag_y1), (tag_x2, tag_y2), (0, 255, 128), -1)
            cv2.putText(
                annotated_frame,
                display_label,
                (x1 + 5, tag_y2 - 5),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                lineType=cv2.LINE_AA,
            )

    # Convert annotated image back to RGB for Gradio display
    annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    # Build Markdown table
    if detected_plates:
        rows = [
            f"| **#{p['id']}** | **`{p['text']}`** | {p['confidence']} | `{p['bbox']}` |"
            for p in detected_plates
        ]
        table_md = "\n".join(rows)
        summary_md = f"""### ✅ Detected {len(detected_plates)} License Plate(s)

| Plate # | Recognized Number | Detector Confidence | Bounding Box |
|:---:|:---:|:---:|:---:|
{table_md}
"""
    else:
        summary_md = """### ⚠️ No License Plates Detected
No plate region was detected with sufficient confidence. Try lowering the confidence slider or uploading a clearer, closer image.
"""

    return annotated_rgb, summary_md


# ── Gradio Interface ────────────────────────────────────────────────────────────
SAMPLE_DIR = Path("sample_images")
example_files = [
    str(SAMPLE_DIR / f"sample_{i}.jpg")
    for i in (1, 2, 3)
    if (SAMPLE_DIR / f"sample_{i}.jpg").exists()
]

with gr.Blocks(title="Vehicle Number Plate Detection & OCR") as demo:
    gr.HTML(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="font-size: 2.2rem; color: #1e3a8a;">🚗 Automatic Vehicle Number Plate Detection & OCR</h1>
            <p style="font-size: 1.05rem; color: #475569;">Upload a vehicle image to detect license plates with <strong>YOLOv8</strong> and extract text using <strong>EasyOCR</strong> on CPU.</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Upload Vehicle Image", type="numpy")
            conf_slider = gr.Slider(
                minimum=0.05,
                maximum=0.95,
                value=0.25,
                step=0.05,
                label="Detection Confidence Threshold",
            )
            submit_btn = gr.Button("🔍 Detect & Read License Plate", variant="primary")

            if example_files:
                gr.Examples(
                    examples=example_files,
                    inputs=input_image,
                    label="Click to Test Sample Images",
                )

        with gr.Column(scale=1):
            output_image = gr.Image(label="Annotated Detection", type="numpy")
            output_text = gr.Markdown(label="Recognition Results")

    submit_btn.click(
        fn=detect_and_recognize,
        inputs=[input_image, conf_slider],
        outputs=[output_image, output_text],
    )

if __name__ == "__main__":
    demo.launch()
