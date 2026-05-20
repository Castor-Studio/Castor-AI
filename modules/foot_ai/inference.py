# modules/foot_ai/inference.py
import time

import cv2
import torch

from .capture import LatestFrameCapture
from .constants import READ_RETRY_DELAY_SEC
from .detector import BallDetector
from .display import (
    create_windows,
    print_status,
    show_focus_window,
    show_main_window,
)
from .focus import FocusSelector


def run(video_left: str, video_right: str | None = None, frameskip: int = 0):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponible : PyTorch est installe en version CPU-only")

    cv2.destroyAllWindows()
    cv2.startWindowThread()
    torch.backends.cudnn.benchmark = True

    detector = BallDetector()
    focus_selector = FocusSelector()

    dual_mode = video_right is not None and str(video_right).strip() != ""
    left_reader = LatestFrameCapture(video_left, "LEFT")
    right_reader = LatestFrameCapture(video_right, "RIGHT") if dual_mode else None

    frame_count = 0
    debug_enabled = True
    last_left_display = None
    last_right_display = None

    try:
        left_reader.start()
        if right_reader is not None:
            right_reader.start()

        window_name, focus_window_name = create_windows(dual_mode)

        while True:
            raw_left, ts_left = left_reader.get_latest()
            raw_right, ts_right = (None, 0.0)

            if right_reader is not None:
                raw_right, ts_right = right_reader.get_latest()

            if raw_left is None or (dual_mode and raw_right is None):
                time.sleep(READ_RETRY_DELAY_SEC)
                continue

            last_left_display = raw_left
            if dual_mode:
                last_right_display = raw_right

            left_frame = last_left_display.copy()
            right_frame = last_right_display.copy() if dual_mode else None

            found_left = False
            found_right = False

            if frame_count % (frameskip + 1) == 0:
                left_frame, found_left = detector.detect(left_frame, draw=debug_enabled)

                if dual_mode:
                    right_frame, found_right = detector.detect(right_frame, draw=debug_enabled)

                focus_selector.update(
                    found_left=found_left,
                    found_right=found_right,
                    ts_left=ts_left,
                    ts_right=ts_right,
                    dual_mode=dual_mode,
                )

            frame_count += 1

            show_main_window(
                window_name=window_name,
                left_frame=left_frame,
                right_frame=right_frame,
                ts_left=ts_left,
                ts_right=ts_right,
                dual_mode=dual_mode,
            )

            focus_frame = (
                left_frame
                if focus_selector.focus_cam == "LEFT" or not dual_mode
                else right_frame
            )
            show_focus_window(focus_window_name, focus_frame)

            print_status(
                focus_cam=focus_selector.focus_cam,
                pending_cam=focus_selector.pending_cam,
                left_failures=left_reader.read_failures,
                right_failures=right_reader.read_failures if right_reader is not None else 0,
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
