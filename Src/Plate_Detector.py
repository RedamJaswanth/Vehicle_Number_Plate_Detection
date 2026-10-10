from ultralytics import YOLO


class PlateDetector:

    def __init__(
        self,
        model_path="Models/license_plate_detector.pt",
        confidence=0.10
    ):

        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame):

        results = self.model(
            frame,
            conf=self.confidence,
            imgsz=640,
            device="cpu",
            verbose=False
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                confidence = float(
                    box.conf[0]
                )

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                detections.append({
                    "confidence": confidence,
                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),
                    "text": ""
                })

        return detections