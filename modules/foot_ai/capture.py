# modules/foot_ai/capture.py
import os
import threading
import time

import cv2

from .constants import READ_RETRY_DELAY_SEC


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
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay"

        self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la source {self.name}: {self.source}")

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
                time.sleep(READ_RETRY_DELAY_SEC)
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
