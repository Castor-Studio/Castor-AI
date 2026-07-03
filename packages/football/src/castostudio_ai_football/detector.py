# modules/foot_ai/detector.py
import cv2
import torch
from ultralytics import YOLO

from .constants import (
    BALL_CONFIDENCE,
    DETECTION_IMAGE_SIZE,
    FOOT_BALL_CLASS_ID,
    MODEL_PATH,
    PREDICTION_CONFIDENCE,
)


class BallDetector:
    def __init__(self):
        self.stream = torch.cuda.Stream()

        self.model = YOLO(str(MODEL_PATH))

        self.model.to("cuda")

    def detect(self, frame, draw=True):
        try:
            img = cv2.resize(frame, (DETECTION_IMAGE_SIZE, DETECTION_IMAGE_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            with torch.cuda.stream(self.stream):
                results = self.model.predict(
                    img,
                    conf=PREDICTION_CONFIDENCE,
                    imgsz=DETECTION_IMAGE_SIZE,
                    device=0,
                    half=True,
                    verbose=False,
                    show=False,
                )[0]

            scale_x = frame.shape[1] / DETECTION_IMAGE_SIZE
            scale_y = frame.shape[0] / DETECTION_IMAGE_SIZE


            ball_found = False

            for i, box in enumerate(results.boxes):
                cls = int(box.cls)
                conf = float(box.conf)

                if cls == FOOT_BALL_CLASS_ID and conf >= BALL_CONFIDENCE:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)

                    if draw:
                        self._draw_detection(frame, x1, y1, x2, y2, conf)

                    ball_found = True

            return frame, ball_found

        except Exception as e:
            import traceback

            print("\n========== DETECTOR ERROR ==========")
            traceback.print_exc()
            print("====================================\n")
            raise

    @staticmethod
    def _draw_detection(frame, x1, y1, x2, y2, conf):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Ball {conf:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )