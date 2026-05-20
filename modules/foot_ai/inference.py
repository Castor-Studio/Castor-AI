# modules/foot_ai/inference.py
import os
import cv2
import time
import torch
import numpy as np
import threading
from ultralytics import YOLO
from pathlib import Path

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

FOCUS_WINDOW_WIDTH = 800
FOCUS_WINDOW_HEIGHT = 600

BALL_CONFIDENCE = 0.60
FOOT_BALL_CLASS_ID = 32
SWITCH_DELAY_SEC = 0.05

MODEL_PATH = (Path(__file__).resolve().parents[2] / "models" / "checkpoint_5.pt")


class LatestFrameCapture:
    def __init__(self, source: str, name: str):
        self.source = source
        self.name = name
        self.cap = None
        self.thread = None
        self.running = False

        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_ts = 0.0
        self.read_failures = 0

    def open(self):
        # Options FFmpeg pour réduire le buffering côté lecture OpenCV/FFmpeg
        # Sur certains builds Windows, cela aide ; sur d'autres, l'effet est limité.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay"

        self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la source {self.name}: {self.source}")

        # Peut être ignoré par certains backends, mais ne coûte rien.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def start(self):
        self.open()
        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _reader_loop(self):
        while self.running:
            ret, frame = self.cap.read()

            if not ret or frame is None or frame.size == 0:
                self.read_failures += 1
                time.sleep(0.002)
                continue

            self.read_failures = 0

            with self.lock:
                self.latest_frame = frame
                self.latest_ts = time.monotonic()

    def get_latest(self):
        with self.lock:
            if self.latest_frame is None:
                return None, 0.0
            return self.latest_frame.copy(), self.latest_ts

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()




# Couper ici apres


def run(video_left: str, video_right: str | None = None, frameskip: int = 0):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponible : PyTorch est installé en version CPU-only")

    cv2.destroyAllWindows()
    cv2.startWindowThread()
    torch.backends.cudnn.benchmark = True

    stream = torch.cuda.Stream()
    model = YOLO(str(MODEL_PATH))
    model.to("cuda")

    def detect_ball_fast(frame, draw=True):
        img = cv2.resize(frame, (416, 416))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with torch.cuda.stream(stream):
            results = model.predict(
                img,
                conf=0.25,
                imgsz=416,
                device=0,
                half=True,
                verbose=False,
                show=False
            )[0]

        scale_x = frame.shape[1] / 416
        scale_y = frame.shape[0] / 416

        ball_found = False

        for box in results.boxes:
            cls = int(box.cls)
            conf = float(box.conf)

            if cls == FOOT_BALL_CLASS_ID and conf >= BALL_CONFIDENCE:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                if draw:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"Ball {conf:.2f}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                ball_found = True

        return frame, ball_found

    dual_mode = video_right is not None and str(video_right).strip() != ""

    left_reader = LatestFrameCapture(video_left, "LEFT")
    right_reader = LatestFrameCapture(video_right, "RIGHT") if dual_mode else None

    left_reader.start()
    if dual_mode:
        right_reader.start()

    window_name = "Videos" if dual_mode else "Single"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WINDOW_WIDTH, WINDOW_HEIGHT)

    focus_window_name = "Ball Focus"
    cv2.namedWindow(focus_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(focus_window_name, FOCUS_WINDOW_WIDTH, FOCUS_WINDOW_HEIGHT)

    frame_count = 0
    focus_cam = "LEFT"
    pending_cam = None
    pending_since = 0.0
    debug_enabled = True

    last_left_display = None
    last_right_display = None

    try:
        while True:
            raw_f1, ts1 = left_reader.get_latest()
            raw_f2, ts2 = (None, 0.0)

            if dual_mode:
                raw_f2, ts2 = right_reader.get_latest()

            if raw_f1 is None:
                time.sleep(0.002)
                continue

            if dual_mode and raw_f2 is None:
                time.sleep(0.002)
                continue

            if raw_f1 is not None:
                last_left_display = raw_f1
            if dual_mode and raw_f2 is not None:
                last_right_display = raw_f2

            f1 = last_left_display.copy()
            f2 = last_right_display.copy() if dual_mode else None

            found1 = False
            found2 = False

            if frame_count % (frameskip + 1) == 0:
                now = time.monotonic()

                f1, found1 = detect_ball_fast(f1, draw=debug_enabled)

                if dual_mode:
                    f2, found2 = detect_ball_fast(f2, draw=debug_enabled)

                candidate_cam = None
                if dual_mode:
                    if found1 and found2:
                        candidate_cam = "LEFT" if ts1 >= ts2 else "RIGHT"
                    elif found1:
                        candidate_cam = "LEFT"
                    elif found2:
                        candidate_cam = "RIGHT"
                else:
                    if found1:
                        candidate_cam = "LEFT"

                if candidate_cam is None:
                    pending_cam = None
                    pending_since = 0.0
                else:
                    if SWITCH_DELAY_SEC <= 0:
                        focus_cam = candidate_cam
                        pending_cam = None
                        pending_since = 0.0
                    else:
                        if candidate_cam == focus_cam:
                            pending_cam = None
                            pending_since = 0.0
                        else:
                            if pending_cam != candidate_cam:
                                pending_cam = candidate_cam
                                pending_since = now
                            else:
                                if (now - pending_since) >= SWITCH_DELAY_SEC:
                                    focus_cam = candidate_cam
                                    pending_cam = None
                                    pending_since = 0.0

            frame_count += 1

            if not dual_mode:
                view = cv2.resize(f1, (WINDOW_WIDTH, WINDOW_HEIGHT))
                cv2.imshow(window_name, view)
            else:
                left = cv2.resize(f1, (WINDOW_WIDTH // 2 - 5, WINDOW_HEIGHT))
                right = cv2.resize(f2, (WINDOW_WIDTH // 2 - 5, WINDOW_HEIGHT))

                spacer = np.zeros((WINDOW_HEIGHT, 10, 3), dtype=np.uint8)
                combined = np.hstack((left, spacer, right))

                age_left_ms = int((time.monotonic() - ts1) * 1000) if ts1 > 0 else -1
                age_right_ms = int((time.monotonic() - ts2) * 1000) if ts2 > 0 else -1

                overlay_text = (
                    f"Q quit | D debug | "
                    f"L age: {age_left_ms} ms | R age: {age_right_ms} ms"
                )

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_w, text_h), _ = cv2.getTextSize(
                    overlay_text, font, font_scale, thickness
                )

                x = combined.shape[1] - text_w - 10
                y = text_h + 10

                cv2.putText(
                    combined,
                    overlay_text,
                    (x, y),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness,
                    cv2.LINE_AA
                )

                cv2.imshow(window_name, combined)

            focus_src = f1 if focus_cam == "LEFT" or not dual_mode else f2
            focus_view = cv2.resize(focus_src, (FOCUS_WINDOW_WIDTH, FOCUS_WINDOW_HEIGHT))
            cv2.imshow(focus_window_name, focus_view)

            print(
                f"\rFocus: {focus_cam} | Pending: {pending_cam or '-'} | "
                f"L_fail: {left_reader.read_failures} | "
                f"R_fail: {right_reader.read_failures if dual_mode else 0}      ",
                end=""
            )

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                debug_enabled = not debug_enabled

    finally:
        left_reader.stop()
        if right_reader is not None:
            right_reader.stop()
        cv2.destroyAllWindows()