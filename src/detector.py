from ultralytics import YOLO


class FaceDetector:

    def __init__(
        self,
        model_path,
        device="mps",
        confidence=0.5
    ):
        self.model = YOLO(model_path)
        self.device = device
        self.confidence = confidence

    def detect(self, frame):

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            device=self.device,
            verbose=False
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                x1, y1, x2, y2 = (
                    box.xyxy[0].tolist()
                )

                confidence = float(
                    box.conf[0]
                )

                detections.append({
                    "bbox": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ],
                    "confidence": confidence
                })

        return detections