"""Live Studio Multi-View Preview for Castor AI (Windows & macOS).

Renders a live TV/Studio broadcast interface displaying:
- PROGRAM OUT: The real-time camera feed selected by the AI (with 1.55x zoom & side-by-side wide fallback)
- MULTIVIEW: Live feeds from webcams / Screen Share with tally borders (RED = Live, GREEN = Preview)
- REAL-TIME AI TELEMETRY: VAD speech state, volume meters, state-machine decision & confidence.

Usage:
    # 1. Start the server (Terminal 1):
    uv run castostudio-ai-server

    # 2. Start the visual preview with Screen Share on Cam 2 (Terminal 2):
    uv run python scripts/live_studio_preview.py --cam1 0 --screen2

    # Or standard 2 webcams:
    uv run python scripts/live_studio_preview.py --cam1 0 --cam2 1
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import queue
import signal
import sys
import threading
import time

import cv2
import grpc
import numpy as np

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

from castostudio_ai_server.proto import ia_analysis_pb2, ia_analysis_pb2_grpc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("live_studio_preview")


def find_window_bbox(app_name: str) -> dict[str, int] | None:
    """Finds bounding box for an on-screen application window matching app_name (macOS & Windows)."""
    name_lower = app_name.lower()
    if sys.platform == "darwin":
        try:
            import Quartz
            windows = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID)
            for w in windows:
                owner = str(w.get("kCGWindowOwnerName", "")).lower()
                title = str(w.get("kCGWindowName", "")).lower()
                b = w.get("kCGWindowBounds", {})
                h = int(b.get("Height", 0))
                wid = int(b.get("Width", 0))
                if (name_lower in owner or name_lower in title) and h > 120 and wid > 120:
                    return {
                        "top": int(b["Y"]),
                        "left": int(b["X"]),
                        "width": wid,
                        "height": h,
                    }
        except Exception:
            pass
    elif sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.windll.user32

            found_rect = None

            def enum_windows_callback(hwnd, extra):
                nonlocal found_rect
                if user32.IsWindowVisible(hwnd):
                    length = user32.GetWindowTextLengthW(hwnd)
                    if length > 0:
                        buff = ctypes.create_unicode_buffer(length + 1)
                        user32.GetWindowTextW(hwnd, buff, length + 1)
                        if name_lower in buff.value.lower():
                            rect = wintypes.RECT()
                            user32.GetWindowRect(hwnd, ctypes.byref(rect))
                            w = rect.right - rect.left
                            h = rect.bottom - rect.top
                            if w > 120 and h > 120:
                                found_rect = {"top": rect.top, "left": rect.left, "width": w, "height": h}
                                return False
                return True

            WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
            user32.EnumWindows(WNDENUMPROC(enum_windows_callback), 0)
            return found_rect
        except Exception:
            pass
    return None


class VideoSourceThread:
    """Non-blocking threaded capture for video sources: Webcams, Screen Share, App Windows (Discord), or Files."""

    def __init__(
        self,
        source_spec: str | int,
        label: str,
        is_screen: bool = False,
        target_window: str | None = None,
        monitor_idx: int = 1,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        self.source_spec = source_spec
        self.label = label
        self.target_window = target_window or ("discord" if str(source_spec).lower() == "discord" else None)
        self.is_screen = is_screen or self.target_window is not None or str(source_spec).lower() in ("screen", "desktop", "share")
        self.monitor_idx = monitor_idx
        self.width = width
        self.height = height
        self.running = False
        self.frame: np.ndarray | None = None
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.is_connected = False
        if self.target_window:
            self.active_description = f"Window: {self.target_window.capitalize()}"
        elif self.is_screen:
            self.active_description = f"Screen {self.monitor_idx}"
        else:
            self.active_description = f"Cam {source_spec}"

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True, name=f"Source-{self.label}")
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None

    def toggle_screen_share(self) -> bool:
        """Toggles between Webcam and Screen Share mode dynamically."""
        with self.lock:
            self.is_screen = not self.is_screen
            self.is_connected = False
            self.frame = None
            if self.target_window and self.is_screen:
                self.active_description = f"Window: {self.target_window.capitalize()}"
            elif self.is_screen:
                self.active_description = f"Screen {self.monitor_idx}"
            else:
                self.active_description = f"Cam {self.source_spec}"
            LOGGER.info("Toggled %s source mode to: %s", self.label, self.active_description)
            return self.is_screen

    def cycle_monitor(self) -> int:
        if not HAS_MSS:
            return 1
        with (getattr(mss, "MSS", mss.mss))() as sct:
            m_count = len(sct.monitors)
            if m_count <= 2:
                return 1
            self.monitor_idx = (self.monitor_idx % (m_count - 1)) + 1
            self.active_description = f"Screen {self.monitor_idx}"
            LOGGER.info("🖥️ Switched %s to monitor %d", self.label, self.monitor_idx)
            return self.monitor_idx

    def get_frame(self) -> np.ndarray:
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
        # Fallback placeholder frame
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = (30, 30, 35)
        text = f"{self.label} ({self.active_description})"
        status = "Waiting for window..." if self.target_window and not self.is_connected else ("Connecting..." if not self.is_connected else "No Signal")
        cv2.putText(img, text, (int(self.width * 0.08), int(self.height * 0.45)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 200, 200), 2)
        cv2.putText(img, status, (int(self.width * 0.08), int(self.height * 0.6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 150, 255), 2)
        return img

    def _capture_loop(self) -> None:
        while self.running:
            if self.is_screen:
                self._capture_screen_loop()
            else:
                self._capture_camera_loop()

    def _capture_screen_loop(self) -> None:
        if not HAS_MSS:
            LOGGER.warning("mss not installed. Using fallback placeholder for Screen Share.")
            time.sleep(1.0)
            return

        last_bbox_check = 0.0
        current_bbox = None

        try:
            with (getattr(mss, "MSS", mss.mss))() as sct:
                while self.running and self.is_screen:
                    now = time.time()
                    # If capturing a specific application window (like Discord)
                    if self.target_window:
                        if now - last_bbox_check > 0.5:
                            last_bbox_check = now
                            current_bbox = find_window_bbox(self.target_window)

                        if current_bbox:
                            target_rect = current_bbox
                        else:
                            # Window not found yet, fallback to monitor
                            m_count = len(sct.monitors)
                            target_rect = sct.monitors[1] if m_count > 1 else sct.monitors[0]
                    else:
                        m_count = len(sct.monitors)
                        m_idx = self.monitor_idx if self.monitor_idx < m_count else 1
                        target_rect = sct.monitors[m_idx] if m_count > 1 else sct.monitors[0]

                    raw = sct.grab(target_rect)
                    frame_bgra = np.array(raw)
                    frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
                    resized = cv2.resize(frame_bgr, (self.width, self.height))
                    with self.lock:
                        self.frame = resized
                        self.is_connected = True
                    time.sleep(0.03)  # ~30 FPS
        except Exception as exc:
            LOGGER.warning("Screen/Window capture error on %s: %s", self.label, exc)
            time.sleep(1.0)

    def _open_camera(self) -> cv2.VideoCapture | None:
        spec = self.source_spec
        # If it's a file path
        if isinstance(spec, str) and (spec.endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")) or "/" in spec or "\\" in spec):
            cap = cv2.VideoCapture(spec)
            if cap.isOpened():
                LOGGER.info("✅ Video file %s successfully opened for %s", spec, self.label)
                return cap
            return None

        # Camera index
        try:
            cam_idx = int(spec)
        except (ValueError, TypeError):
            cam_idx = 0

        backend = (
            cv2.CAP_AVFOUNDATION
            if sys.platform == "darwin"
            else (cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        )

        candidate_indices = [cam_idx]
        if cam_idx == 2:
            candidate_indices.extend([1, 0])
        elif cam_idx == 1:
            candidate_indices.extend([0, 2])
        elif cam_idx == 0:
            candidate_indices.extend([1, 2])

        for idx in candidate_indices:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened() and backend != cv2.CAP_ANY:
                cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.active_description = f"Cam {idx}"
                    LOGGER.info("✅ Camera %s successfully opened on index %d", self.label, idx)
                    return cap
                cap.release()

        return None

    def _capture_camera_loop(self) -> None:
        cap = self._open_camera()
        while self.running and not self.is_screen:
            if cap is None or not cap.isOpened():
                time.sleep(1.5)
                cap = self._open_camera()
                continue

            ret, raw_frame = cap.read()
            if ret and raw_frame is not None:
                self.is_connected = True
                resized = cv2.resize(raw_frame, (self.width, self.height))
                with self.lock:
                    self.frame = resized
            else:
                # If reading a video file, loop to start
                if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    self.is_connected = False
                    time.sleep(0.05)

        if cap is not None:
            cap.release()


class AIClientBridge:
    """Runs gRPC streaming in background and stores latest AI switch decisions."""

    def __init__(self, addr: str, sources: list[ia_analysis_pb2.Source], config: dict[str, str]) -> None:
        self.addr = addr
        self.sources = sources
        self.config = config
        self.current_scene_id = "s1_host"
        self.confidence = 0.90
        self.last_switch_time = time.time()
        self.status_message = "Connected"
        self.lock = threading.Lock()
        self.running = False
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._run_async_loop, daemon=True, name="AIBridge")
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def get_state(self) -> tuple[str, float, float, str]:
        with self.lock:
            return self.current_scene_id, self.confidence, self.last_switch_time, self.status_message

    def manual_override_scene(self, scene_id: str) -> None:
        with self.lock:
            self.current_scene_id = scene_id
            self.last_switch_time = time.time()
            self.confidence = 1.0

    def _run_async_loop(self) -> None:
        asyncio.run(self._grpc_worker())

    async def _grpc_worker(self) -> None:
        async with grpc.aio.insecure_channel(self.addr) as channel:
            stub = ia_analysis_pb2_grpc.IaAnalysisServiceStub(channel)
            try:
                start_resp = await stub.StartSession(
                    ia_analysis_pb2.StartSessionRequest(module_name="podcast", module_config=self.config)
                )
            except Exception as exc:
                LOGGER.error("StartSession failed: %s", exc)
                with self.lock:
                    self.status_message = f"gRPC Error: {exc}"
                return

            if not start_resp.success:
                LOGGER.error("Session refused: %s", start_resp.message)
                return

            session_id = start_resp.session_id
            LOGGER.info("AI Live session started: %s", session_id)

            stop_event = asyncio.Event()

            async def gen_msgs():
                yield ia_analysis_pb2.ClientMessage(
                    session_id=session_id,
                    sources=ia_analysis_pb2.SourceList(sources=self.sources),
                )
                while not stop_event.is_set():
                    try:
                        await asyncio.wait_for(stop_event.wait(), timeout=2.5)
                    except asyncio.TimeoutError:
                        yield ia_analysis_pb2.ClientMessage(
                            session_id=session_id,
                            keep_alive=ia_analysis_pb2.KeepAlive(timestamp_ms=int(time.time() * 1000)),
                        )
                yield ia_analysis_pb2.ClientMessage(
                    session_id=session_id,
                    stop=ia_analysis_pb2.StopSignal(reason="GUI closed"),
                )

            call = stub.AnalysisStream(gen_msgs())
            try:
                async for event in call:
                    if not self.running:
                        break
                    payload = event.WhichOneof("payload")
                    if payload == "switch_suggestion":
                        sug = event.switch_suggestion
                        with self.lock:
                            self.current_scene_id = sug.scene_id
                            self.confidence = sug.confidence
                            self.last_switch_time = time.time()
                        LOGGER.info("🎬 AI Switch -> %s (conf=%.2f)", sug.scene_id, sug.confidence)
                    elif payload == "status":
                        with self.lock:
                            self.status_message = event.status.message
            except Exception as exc:
                LOGGER.warning("Stream exception: %s", exc)
            finally:
                stop_event.set()
                try:
                    await stub.EndSession(ia_analysis_pb2.EndSessionRequest(session_id=session_id))
                except Exception:
                    pass


def apply_digital_zoom(frame: np.ndarray, zoom_factor: float = 1.55) -> np.ndarray:
    """Applies a high-quality centered digital crop/zoom for monologue focus."""
    h, w = frame.shape[:2]
    crop_w = int(w / zoom_factor)
    crop_h = int(h / zoom_factor)
    x1 = (w - crop_w) // 2
    y1 = (h - crop_h) // 2
    cropped = frame[y1 : y1 + crop_h, x1 : x1 + crop_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def compose_studio_layout(
    frame_host: np.ndarray,
    frame_guest: np.ndarray,
    active_scene_id: str,
    confidence: float,
    switch_age: float,
    status_msg: str,
    guest_label: str = "CAM 2 - INVITE",
) -> np.ndarray:
    """Draws a complete Studio Multi-View Dashboard with Program Out and Preview tiles."""
    canvas_w, canvas_h = 1600, 960
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (22, 22, 26)  # Dark sleek background

    # 1. Determine Program Out Content
    is_host = "host" in active_scene_id or active_scene_id == "s1_host" or active_scene_id == "s1_host_zoom"
    is_guest = "guest" in active_scene_id or active_scene_id == "s2_guest" or active_scene_id == "s2_guest_zoom"
    is_wide = "wide" in active_scene_id or active_scene_id == "s3_wide"
    is_zoom = "zoom" in active_scene_id

    if is_host:
        prog_raw = apply_digital_zoom(frame_host, 1.55) if is_zoom else frame_host
        current_role_name = "HOTE (Cam 1)" + ("  [ZOOM CADRAGE SERRÉ 1.55x]" if is_zoom else "")
    elif is_guest:
        prog_raw = apply_digital_zoom(frame_guest, 1.55) if is_zoom else frame_guest
        current_role_name = f"{guest_label}" + ("  [ZOOM CADRAGE SERRÉ 1.55x]" if is_zoom else "")
    else:  # Wide shot / side-by-side
        current_role_name = "PLAN LARGE / DEBAT (Split-Screen 2 Faces)"
        h, w = frame_host.shape[:2]
        half_w = w // 2
        split_left = frame_host[:, half_w // 2 : half_w // 2 + half_w]
        split_right = frame_guest[:, half_w // 2 : half_w // 2 + half_w]
        prog_raw = np.hstack([split_left, split_right])
        prog_raw = cv2.resize(prog_raw, (w, h))

    # 2. Place PROGRAM OUT (Top-Left, 1040x585)
    prog_w, prog_h = 1040, 585
    prog_x, prog_y = 30, 80
    prog_resized = cv2.resize(prog_raw, (prog_w, prog_h))
    canvas[prog_y : prog_y + prog_h, prog_x : prog_x + prog_w] = prog_resized

    # Program Out Tally Border (Magenta if Zoom, Red if Live)
    border_color = (255, 0, 200) if is_zoom else (0, 0, 230)
    cv2.rectangle(canvas, (prog_x - 3, prog_y - 3), (prog_x + prog_w + 3, prog_y + prog_h + 3), border_color, 4)

    # Program Out Top Badge
    badge_bg = (180, 0, 160) if is_zoom else (0, 0, 200)
    cv2.rectangle(canvas, (prog_x, prog_y), (prog_x + 360, prog_y + 40), badge_bg, -1)
    badge_text = "● PROGRAM [ZOOM ACTIF]" if is_zoom else "● PROGRAM (DIRECT DIFFUSE)"
    cv2.putText(canvas, badge_text, (prog_x + 12, prog_y + 26), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

    # Program Out Bottom Label
    cv2.rectangle(canvas, (prog_x, prog_y + prog_h - 40), (prog_x + prog_w, prog_y + prog_h), (15, 15, 15), -1)
    cv2.putText(canvas, f"Source active : {current_role_name}", (prog_x + 15, prog_y + prog_h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 240, 255), 2)

    # 3. Place MULTIVIEW (Bottom Row: Cam Hote & Cam Invite/Screen, 490x275 each)
    thumb_w, thumb_h = 490, 275
    y_thumbs = 680

    # Thumb 1: Host
    x_t1 = 30
    t1_resized = cv2.resize(frame_host, (thumb_w, thumb_h))
    canvas[y_thumbs : y_thumbs + thumb_h, x_t1 : x_t1 + thumb_w] = t1_resized
    t1_border_color = (0, 0, 230) if is_host else ((0, 200, 0) if not is_wide else (80, 80, 80))
    cv2.rectangle(canvas, (x_t1 - 2, y_thumbs - 2), (x_t1 + thumb_w + 2, y_thumbs + thumb_h + 2), t1_border_color, 3)
    cv2.rectangle(canvas, (x_t1, y_thumbs), (x_t1 + 220, y_thumbs + 32), (20, 20, 20), -1)
    cv2.putText(canvas, "CAM 1 - HOTE (Face)", (x_t1 + 10, y_thumbs + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Thumb 2: Guest / Screen Share
    x_t2 = 580
    t2_resized = cv2.resize(frame_guest, (thumb_w, thumb_h))
    canvas[y_thumbs : y_thumbs + thumb_h, x_t2 : x_t2 + thumb_w] = t2_resized
    t2_border_color = (0, 0, 230) if is_guest else ((0, 200, 0) if not is_wide else (80, 80, 80))
    cv2.rectangle(canvas, (x_t2 - 2, y_thumbs - 2), (x_t2 + thumb_w + 2, y_thumbs + thumb_h + 2), t2_border_color, 3)
    cv2.rectangle(canvas, (x_t2, y_thumbs), (x_t2 + 280, y_thumbs + 32), (20, 20, 20), -1)
    cv2.putText(canvas, f"CAM 2 - {guest_label}", (x_t2 + 10, y_thumbs + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # 4. Right Side: AI Control & Telemetry Dashboard
    panel_x, panel_y, panel_w, panel_h = 1100, 80, 470, 875
    cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (32, 32, 38), -1)
    cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (55, 55, 65), 2)

    # Panel Header
    cv2.putText(canvas, "CASTOR STUDIO IA", (panel_x + 25, panel_y + 45), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 220, 255), 2)
    cv2.putText(canvas, "Moteur Regie Automatique Podcast", (panel_x + 25, panel_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 170), 1)
    cv2.line(canvas, (panel_x + 20, panel_y + 90), (panel_x + panel_w - 20, panel_y + 90), (55, 55, 65), 1)

    # Telemetry Cards
    cur_y = panel_y + 130
    cv2.putText(canvas, "DECISION ACTUELLE :", (panel_x + 25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 190), 1)
    cur_y += 35
    dec_color = (255, 0, 220) if is_zoom else ((0, 240, 255) if is_wide else ((0, 120, 255) if is_host else (255, 160, 50)))
    cv2.putText(canvas, current_role_name, (panel_x + 25, cur_y), cv2.FONT_HERSHEY_DUPLEX, 0.68, dec_color, 2)

    cur_y += 45
    cv2.putText(canvas, f"Scene ID : {active_scene_id}", (panel_x + 25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cur_y += 30
    cv2.putText(canvas, f"Confiance IA : {confidence * 100:.0f}%", (panel_x + 25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 120), 1)
    cur_y += 30
    cv2.putText(canvas, f"Temps sur ce plan : {switch_age:.1f}s", (panel_x + 25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cur_y += 45
    cv2.line(canvas, (panel_x + 20, cur_y), (panel_x + panel_w - 20, cur_y), (55, 55, 65), 1)
    cur_y += 35

    cv2.putText(canvas, "CADRAGE AUTOMATIQUE :", (panel_x + 25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 190), 1)
    cur_y += 30
    cv2.putText(canvas, "• Hote parle -> Cadrage Hote", (panel_x + 25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cur_y += 25
    cv2.putText(canvas, "• Invite parle -> Cadrage Invite/Screen", (panel_x + 25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cur_y += 25
    cv2.putText(canvas, "• Monologue (>3.5s) -> ZOOM SERRÉ 1.55x", (panel_x + 25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 120, 255), 1)
    cur_y += 25
    cv2.putText(canvas, "• Debat / 2 voix -> Split Plan Large", (panel_x + 25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cur_y += 25
    cv2.putText(canvas, "• Silence (>3s) -> Split Plan Large", (panel_x + 25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    cur_y += 45
    cv2.line(canvas, (panel_x + 20, cur_y), (panel_x + panel_w - 20, cur_y), (55, 55, 65), 1)
    cur_y += 35
    cv2.putText(canvas, "RACCOURCIS CLAVIER :", (panel_x + 25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 190), 1)
    cur_y += 30
    cv2.putText(canvas, "[S] : Basculer Cam 2 <-> Screen Share", (panel_x + 25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 240, 255), 1)
    cur_y += 25
    cv2.putText(canvas, "[M] : Changer d'ecran (Multi-ecrans)", (panel_x + 25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 240, 255), 1)
    cur_y += 25
    cv2.putText(canvas, "[Z] : Tester/Forcer le Zoom Cadrage", (panel_x + 25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 120, 255), 1)
    cur_y += 25
    cv2.putText(canvas, "[1] / [2] : Forcer Hote / Invite", (panel_x + 25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cur_y += 25
    cv2.putText(canvas, "[W] : Forcer Split Plan Large", (panel_x + 25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cur_y += 25
    cv2.putText(canvas, "[Q] / [ESC] : Quitter la regie", (panel_x + 25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 255), 1)

    # Top Global Header
    cv2.putText(canvas, "CASTOR-AI STUDIO — RETOUR REGIE EN DIRECT", (30, 48), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

    return canvas


def scan_available_cameras(max_tested: int = 5) -> list[int]:
    """Test camera indices and print which ones return valid live video frames."""
    print(f"\n🔍 Scanning available video capture devices (OS: {sys.platform})...")
    working_indices = []
    backend = (
        cv2.CAP_AVFOUNDATION
        if sys.platform == "darwin"
        else (cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    )
    for idx in range(max_tested):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened() and backend != cv2.CAP_ANY:
            cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"  ✅ Camera Index [{idx}] -> Active ({w}x{h})")
                working_indices.append(idx)
            else:
                print(f"  ⚠️  Camera Index [{idx}] -> Opened but failed to capture frame")
            cap.release()
        else:
            print(f"  ❌ Camera Index [{idx}] -> Cannot open")
    print("")
    return working_indices


def scan_available_microphones() -> list[tuple[str, str]]:
    """Lists all available audio input devices (microphones)."""
    print(f"\n🎙️ Scanning available audio capture devices (OS: {sys.platform})...\n")
    import subprocess
    import re

    mics = []
    if sys.platform == "darwin":
        try:
            proc = subprocess.run(
                ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                check=False,
            )
            is_audio = False
            for line in proc.stderr.splitlines():
                if "AVFoundation audio devices:" in line:
                    is_audio = True
                    continue
                if is_audio:
                    match = re.search(r"\[(\d+)\]\s+(.*)", line)
                    if match:
                        idx, name = match.group(1), match.group(2).strip()
                        mics.append((idx, name))
                        print(f"  🎙️ Index [{idx}] -> \"{name}\"  (pass: --mic {idx})")
        except Exception as exc:
            print(f"  ⚠️ Error querying AVFoundation: {exc}")
    elif sys.platform == "win32":
        try:
            proc = subprocess.run(
                ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                check=False,
            )
            is_audio = False
            for line in proc.stderr.splitlines():
                if "DirectShow audio devices" in line:
                    is_audio = True
                    continue
                if "DirectShow video devices" in line:
                    is_audio = False
                    continue
                if is_audio and "Alternative name" not in line:
                    match = re.search(r'"([^"]+)"', line)
                    if match:
                        name = match.group(1)
                        mics.append((name, name))
                        print(f"  🎙️ Device: \"{name}\"  (pass: --mic \"{name}\")")
        except Exception as exc:
            print(f"  ⚠️ Error querying DirectShow: {exc}")
    else:
        print("  Generic ALSA/Pulse default input")

    print("")
    return mics


def test_microphones_live(mic1: str, mic2: str, duration: float = 5.0) -> None:
    """Listens to mic1 and mic2 for a few seconds and displays live ASCII VU-meters."""
    from castostudio_ai_podcast.audio import AudioStreamReader

    if sys.platform == "darwin":
        u1 = f"avfoundation::{mic1}" if str(mic1).isdigit() else str(mic1)
        u2 = f"avfoundation::{mic2}" if str(mic2).isdigit() else str(mic2)
    elif sys.platform == "win32":
        u1 = str(mic1) if (str(mic1).startswith("dshow:") or str(mic1).endswith((".wav", ".mp3", ".mp4"))) else f"dshow:audio={mic1}"
        u2 = str(mic2) if (str(mic2).startswith("dshow:") or str(mic2).endswith((".wav", ".mp3", ".mp4"))) else f"dshow:audio={mic2}"
    else:
        u1, u2 = str(mic1), str(mic2)

    print(f"\n🔊 Testing live microphone levels for {duration} seconds (speak into your mics!)...\n")
    print(f"  Mic 1 target: {u1}")
    print(f"  Mic 2 target: {u2}\n")

    r1 = AudioStreamReader(u1, "Mic1-Test")
    r2 = AudioStreamReader(u2, "Mic2-Test")
    r1.start()
    r2.start()

    start = time.time()
    try:
        while time.time() - start < duration:
            db1 = r1.get_volume_db()
            db2 = r2.get_volume_db()
            spk1 = r1.is_speaking()
            spk2 = r2.is_speaking()

            # Generate ASCII VU-meter bar (-60dB to 0dB)
            def bar(db: float) -> str:
                clamped = max(-60.0, min(0.0, db))
                units = int((clamped + 60.0) / 3.0)  # 0 to 20
                return "█" * units + "░" * (20 - units)

            b1 = bar(db1)
            b2 = bar(db2)
            s1_txt = "\033[1;32m[VOICE DETECTED]\033[0m" if spk1 else "[Silence]"
            s2_txt = "\033[1;36m[VOICE DETECTED]\033[0m" if spk2 else "[Silence]"

            sys.stdout.write(
                f"\r  Mic 1: [{b1}] {db1:5.1f} dB {s1_txt:<25} | Mic 2: [{b2}] {db2:5.1f} dB {s2_txt:<25}"
            )
            sys.stdout.flush()
            time.sleep(0.08)
    finally:
        r1.stop()
        r2.stop()
        print("\n\n✅ Test finished.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--addr", default="localhost:50051", help="gRPC server address")
    parser.add_argument("--scan-cameras", action="store_true", help="Scan and list working OpenCV camera indices")
    parser.add_argument("--scan-mics", action="store_true", help="Scan and list available audio input microphones")
    parser.add_argument("--test-mics", action="store_true", help="Listen to mic1 & mic2 live and show ASCII VU-meters")
    parser.add_argument("--cam1", default="0", help="Camera index or path for Host. Default: 0")
    parser.add_argument("--mic1", default="1", help="Mic index/name for Host. Default: 1 (macOS) or 0 (Windows)")
    parser.add_argument("--cam2", default="1", help="Camera index, path, or 'screen' for Guest. Default: 1")
    parser.add_argument("--mic2", default="2", help="Mic index/name for Guest. Default: 2 (macOS) or 1 (Windows)")
    parser.add_argument("--screen1", action="store_true", help="Use Screen Share for Source 1 (Host)")
    parser.add_argument("--screen2", action="store_true", help="Use Screen Share for Source 2 (Guest)")
    parser.add_argument("--discord", action="store_true", help="Capture Discord window directly as Source 2 (Guest)")
    parser.add_argument("--window", default=None, help="Capture specific window by title/app (e.g. 'Discord', 'Chrome', 'Zoom')")
    parser.add_argument("--min-hold-time", type=float, default=2.0, help="Anti-flicker hold duration in seconds")
    parser.add_argument("--monologue_time", type=float, default=3.5, help="Monologue zoom threshold in seconds (default: 3.5s)")
    parser.add_argument("--vad-threshold", type=float, default=0.5, help="Silero VAD threshold")
    args = parser.parse_args()

    if args.scan_cameras:
        scan_available_cameras()
        return

    if args.scan_mics:
        scan_available_microphones()
        return

    if args.test_mics:
        test_microphones_live(args.mic1, args.mic2)
        return

    target_win = "discord" if args.discord else (args.window if args.window else None)
    if str(args.cam2).lower() == "discord":
        target_win = "discord"

    is_screen1 = args.screen1 or str(args.cam1).lower() == "screen"
    is_screen2 = args.screen2 or args.discord or target_win is not None or str(args.cam2).lower() in ("screen", "discord")

    source2_desc = f"Window: {target_win.capitalize()}" if target_win else ("Screen Share" if is_screen2 else f"Cam {args.cam2}")

    print("\n🎬 Starting Castor Studio Live Multi-View Preview...")
    print(f"OS: {sys.platform}")
    print(f"Source 1 (Host):  {'Screen Share' if is_screen1 else f'Cam {args.cam1}'} | Mic: {args.mic1}")
    print(f"Source 2 (Guest): {source2_desc} | Mic: {args.mic2}")
    print(f"Monologue Zoom Trigger: {args.monologue_time}s | Anti-flicker: {args.min_hold_time}s\n")

    # 1. Start Video Threads
    host_src = VideoSourceThread(args.cam1, "Host", is_screen=is_screen1)
    guest_src = VideoSourceThread(args.cam2, "Guest", is_screen=is_screen2, target_window=target_win)
    host_src.start()
    guest_src.start()

    # 2. Build gRPC Sources (5 podcast scenes: Host, Guest, Wide, Zoom Host, Zoom Guest)
    if sys.platform == "darwin":
        url_host = f"avfoundation::{args.mic1}" if str(args.mic1).isdigit() else str(args.mic1)
        url_guest = f"avfoundation::{args.mic2}" if str(args.mic2).isdigit() else str(args.mic2)
    elif sys.platform == "win32":
        m1 = str(args.mic1)
        m2 = str(args.mic2)
        url_host = m1 if (m1.startswith("dshow:") or m1.endswith((".wav", ".mp3", ".mp4"))) else f"dshow:audio={m1}"
        url_guest = m2 if (m2.startswith("dshow:") or m2.endswith((".wav", ".mp3", ".mp4"))) else f"dshow:audio={m2}"
    else:
        url_host = str(args.mic1)
        url_guest = str(args.mic2)

    sources = [
        ia_analysis_pb2.Source(scene_id="s1_host", label="Cam Hote", url=url_host),
        ia_analysis_pb2.Source(scene_id="s2_guest", label="Cam Invite", url=url_guest),
        ia_analysis_pb2.Source(scene_id="s3_wide", label="Plan Large", url=url_host),
        ia_analysis_pb2.Source(scene_id="s1_host_zoom", label="Zoom Hote", url=url_host),
        ia_analysis_pb2.Source(scene_id="s2_guest_zoom", label="Zoom Invite", url=url_guest),
    ]
    config = {
        "min_hold_time": str(args.min_hold_time),
        "monologue_time": str(args.monologue_time),
        "vad_threshold": str(args.vad_threshold),
    }

    # 3. Start AI Client Bridge
    ai_bridge = AIClientBridge(args.addr, sources, config)
    ai_bridge.start()

    window_name = "Castor Studio - Retour Regie & Diffusion Finale"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1400, 840)

    try:
        while True:
            frame_h = host_src.get_frame()
            frame_g = guest_src.get_frame()
            active_scene, conf, switch_time, status_msg = ai_bridge.get_state()
            switch_age = time.time() - switch_time

            guest_display_label = "SCREEN SHARE" if guest_src.is_screen else "INVITE (Cam)"

            dashboard = compose_studio_layout(
                frame_host=frame_h,
                frame_guest=frame_g,
                active_scene_id=active_scene,
                confidence=conf,
                switch_age=switch_age,
                status_msg=status_msg,
                guest_label=guest_display_label,
            )

            cv2.imshow(window_name, dashboard)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
                break
            elif key in (ord("s"), ord("S")):  # Toggle Screen Share on Cam 2
                guest_src.toggle_screen_share()
            elif key in (ord("m"), ord("M")):  # Cycle monitor if multi-screen
                guest_src.cycle_monitor()
            elif key in (ord("z"), ord("Z")):  # Force Zoom toggle
                if "zoom" in active_scene:
                    base = active_scene.replace("_zoom", "")
                    ai_bridge.manual_override_scene(base)
                else:
                    target = f"{active_scene}_zoom" if ("host" in active_scene or "guest" in active_scene) else "s1_host_zoom"
                    ai_bridge.manual_override_scene(target)
            elif key == ord("1"):
                ai_bridge.manual_override_scene("s1_host")
            elif key == ord("2"):
                ai_bridge.manual_override_scene("s2_guest")
            elif key in (ord("w"), ord("W")):
                ai_bridge.manual_override_scene("s3_wide")
    finally:
        host_src.stop()
        guest_src.stop()
        ai_bridge.stop()
        cv2.destroyAllWindows()
        print("Studio preview closed cleanly.")


if __name__ == "__main__":
    main()
