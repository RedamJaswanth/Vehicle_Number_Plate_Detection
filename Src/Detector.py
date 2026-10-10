from ultralytics import YOLO


class VehicleDetector:

    def __init__(
        self,
        model_path="Models/yolov8n.pt",
        confidence=0.40
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence

        # Vehicle classes from COCO
        self.vehicle_classes = {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
        }

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

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if class_id not in self.vehicle_classes:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                detections.append({
                    "class_id": class_id,
                    "class_name": self.vehicle_classes[class_id],
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2)
                })

        return detections