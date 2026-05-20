# modules/foot_ai/focus.py
import time

from .constants import SWITCH_DELAY_SEC


class FocusSelector:
    def __init__(self):
        self.focus_cam = "LEFT"
        self.pending_cam = None
        self.pending_since = 0.0

    def update(
        self,
        found_left: bool,
        found_right: bool,
        ts_left: float,
        ts_right: float,
        dual_mode: bool,
    ):
        candidate_cam = self._choose_candidate(found_left, found_right, ts_left, ts_right, dual_mode)
        now = time.monotonic()

        if candidate_cam is None:
            self._clear_pending()
            return

        if SWITCH_DELAY_SEC <= 0:
            self.focus_cam = candidate_cam
            self._clear_pending()
            return

        if candidate_cam == self.focus_cam:
            self._clear_pending()
            return

        if self.pending_cam != candidate_cam:
            self.pending_cam = candidate_cam
            self.pending_since = now
            return

        if (now - self.pending_since) >= SWITCH_DELAY_SEC:
            self.focus_cam = candidate_cam
            self._clear_pending()

    @staticmethod
    def _choose_candidate(
        found_left: bool,
        found_right: bool,
        ts_left: float,
        ts_right: float,
        dual_mode: bool,
    ):
        if not dual_mode:
            return "LEFT" if found_left else None

        if found_left and found_right:
            return "LEFT" if ts_left >= ts_right else "RIGHT"
        if found_left:
            return "LEFT"
        if found_right:
            return "RIGHT"
        return None

    def _clear_pending(self):
        self.pending_cam = None
        self.pending_since = 0.0
