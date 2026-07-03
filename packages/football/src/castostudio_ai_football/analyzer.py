import time

from .capture import LatestFrameCapture
from .constants import READ_RETRY_DELAY_SEC
from .detector import BallDetector


class FootballAnalyzer:
    def __init__(self, stream_1_url: str, stream_2_url: str, frameskip: int = 0):
        self.detector = BallDetector()

        self.stream_1_reader = LatestFrameCapture(stream_1_url, "STREAM_1")
        self.stream_2_reader = LatestFrameCapture(stream_2_url, "STREAM_2")

        self.frameskip = frameskip
        self.frame_count = 0
        self.last_focus = None
        
    def wait_until_ready(self, timeout_sec: float = 5.0) -> bool:
        start = time.time()

        while time.time() - start < timeout_sec:
            raw_1, _ = self.stream_1_reader.get_latest()
            raw_2, _ = self.stream_2_reader.get_latest()

            if raw_1 is not None and raw_2 is not None:
                print("[FootballAnalyzer] Both streams ready")
                return True

            time.sleep(READ_RETRY_DELAY_SEC)

        print("[FootballAnalyzer] Streams not ready after timeout")
        return False

    def start(self):
        self.stream_1_reader.start()
        self.stream_2_reader.start()

    def stop(self):
        self.stream_1_reader.stop()
        self.stream_2_reader.stop()

    def analyze_once(self) -> str | None:
        raw_1, ts_1 = self.stream_1_reader.get_latest()
        raw_2, ts_2 = self.stream_2_reader.get_latest()

        if raw_1 is None or raw_2 is None:
            print("[FootballAnalyzer] Waiting frames")
            print("  raw_1 is None:", raw_1 is None)
            print("  raw_2 is None:", raw_2 is None)
            print("  stream_1 failures:", self.stream_1_reader.read_failures)
            print("  stream_2 failures:", self.stream_2_reader.read_failures)
            time.sleep(READ_RETRY_DELAY_SEC)
            return None

        if self.frame_count % (self.frameskip + 1) != 0:
            self.frame_count += 1
            return self.last_focus

        _, found_1 = self.detector.detect(raw_1, draw=False)
        _, found_2 = self.detector.detect(raw_2, draw=False)

        print(
            f"[FootballAnalyzer] frame={self.frame_count} "
            f"found_1={found_1} found_2={found_2} "
            f"last_focus={self.last_focus}"
        )

        previous_focus = self.last_focus

        if found_1 and not found_2:
            self.last_focus = "STREAM_1"
        elif found_2 and not found_1:
            self.last_focus = "STREAM_2"
        elif found_1 and found_2:
            print("[FootballAnalyzer] Ball detected on both streams, keeping previous focus")
        else:
            print("[FootballAnalyzer] Ball detected on no stream, keeping previous focus")

        if self.last_focus != previous_focus:
            print(
                "[FootballAnalyzer] DETECTED SCENE CHANGE:",
                previous_focus,
                "->",
                self.last_focus,
            )

        self.frame_count += 1
        return self.last_focus