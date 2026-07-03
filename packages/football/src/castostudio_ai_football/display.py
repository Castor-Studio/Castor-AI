# modules/foot_ai/display.py
import time

import cv2
import numpy as np

from .constants import (
    FOCUS_WINDOW_HEIGHT,
    FOCUS_WINDOW_WIDTH,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)


def create_windows(dual_mode: bool):
    window_name = "Videos" if dual_mode else "Single"
    focus_window_name = "Ball Focus"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WINDOW_WIDTH, WINDOW_HEIGHT)

    cv2.namedWindow(focus_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(focus_window_name, FOCUS_WINDOW_WIDTH, FOCUS_WINDOW_HEIGHT)

    return window_name, focus_window_name


def show_main_window(window_name: str, left_frame, right_frame, ts_left: float, ts_right: float, dual_mode: bool):
    if not dual_mode:
        view = cv2.resize(left_frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        cv2.imshow(window_name, view)
        return

    left = cv2.resize(left_frame, (WINDOW_WIDTH // 2 - 5, WINDOW_HEIGHT))
    right = cv2.resize(right_frame, (WINDOW_WIDTH // 2 - 5, WINDOW_HEIGHT))
    spacer = np.zeros((WINDOW_HEIGHT, 10, 3), dtype=np.uint8)
    combined = np.hstack((left, spacer, right))

    _draw_overlay(combined, ts_left, ts_right)
    cv2.imshow(window_name, combined)


def show_focus_window(focus_window_name: str, frame):
    focus_view = cv2.resize(frame, (FOCUS_WINDOW_WIDTH, FOCUS_WINDOW_HEIGHT))
    cv2.imshow(focus_window_name, focus_view)


def print_status(focus_cam: str, pending_cam: str | None, left_failures: int, right_failures: int):
    print(
        f"\rFocus: {focus_cam} | Pending: {pending_cam or '-'} | "
        f"L_fail: {left_failures} | R_fail: {right_failures}      ",
        end="",
    )


def _draw_overlay(combined, ts_left: float, ts_right: float):
    age_left_ms = int((time.monotonic() - ts_left) * 1000) if ts_left > 0 else -1
    age_right_ms = int((time.monotonic() - ts_right) * 1000) if ts_right > 0 else -1
    overlay_text = f"Q quit | D debug | L age: {age_left_ms} ms | R age: {age_right_ms} ms"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(overlay_text, font, font_scale, thickness)

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
        cv2.LINE_AA,
    )