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
        self.total_frames = 0

    def open(self):
        print(f"[Capture:{self.name}] Opening source:")
        print(f"[Capture:{self.name}] {self.source}")

        self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la source {self.name}: {self.source}")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print(f"[Capture:{self.name}] Opened OK")

    def start(self):
        self.open()
        self.running = True
        self.thread = threading.Thread(
            target=self._reader_loop,
            name=f"CaptureThread-{self.name}",
            daemon=True,
        )
        self.thread.start()
        print(f"[Capture:{self.name}] Thread started")

    def _reader_loop(self):
        print(f"[Capture:{self.name}] Reader loop started")

        last_log = time.time()

        while self.running:
            ret, frame = self.cap.read()

            if not ret or frame is None or frame.size == 0:
                self.read_failures += 1

                now = time.time()
                if now - last_log >= 1:
                    print(f"[Capture:{self.name}] No frame")
                    print(f"[Capture:{self.name}] ret={ret}")
                    print(f"[Capture:{self.name}] failures={self.read_failures}")
                    last_log = now

                time.sleep(READ_RETRY_DELAY_SEC)
                continue

            self.read_failures = 0
            self.total_frames += 1

            with self.lock:
                self.latest_frame = frame
                self.latest_ts = time.monotonic()

            now = time.time()
            if now - last_log >= 2:
                last_log = now

        print(f"[Capture:{self.name}] Reader loop stopped")

    def get_latest(self):
        with self.lock:
            if self.latest_frame is None:
                return None, 0.0
            return self.latest_frame.copy(), self.latest_ts

    def stop(self):
        print(f"[Capture:{self.name}] Stopping")

        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=1.0)

        if self.cap is not None:
            self.cap.release()

        print(f"[Capture:{self.name}] Stopped")