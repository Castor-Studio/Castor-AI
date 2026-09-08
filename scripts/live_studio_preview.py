"""Live Studio Multi-View Preview for Castor AI on macOS.

Renders a live TV/Studio broadcast interface displaying:
- PROGRAM OUT: The real-time camera feed selected by the AI (with zoom & side-by-side wide fallback)
- MULTIVIEW: Live feeds from both webcams (FaceTime + iPhone) with tally borders (RED = Live, GREEN = Preview)
- REAL-TIME AI TELEMETRY: VAD speech state, volume meters, state-machine decision & confidence.

Usage:
    # 1. Start the server (Terminal 1):
    uv run castostudio-ai-server

    # 2. Start the visual preview (Terminal 2):
    uv run python scripts/live_studio_preview.py --cam1 0 --mic1 1 --cam2 2 --mic2 2
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

from castostudio_ai_server.proto import ia_analysis_pb2, ia_analysis_pb2_grpc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("live_studio_preview")


class CameraCaptureThread:
    """Non-blocking threaded capture for one video camera with auto-fallback."""

    def __init__(self, camera_index: int, label: str, width: int = 1280, height: int = 720) -> None:
        self.camera_index = camera_index
        self.label = label
        self.width = width
        self.height = height
        self.running = False
        self.frame: np.ndarray | None = None
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.is_connected = False
        self.active_index = camera_index

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True, name=f"Cam-{self.label}")
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None

    def get_frame(self) -> np.ndarray:
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
        # Return fallback placeholder frame
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = (30, 30, 35)
        text = f"{self.label} (Cam {self.active_index})"
        status = "Connecting / Tap to Wake iPhone..." if not self.is_connected else "No Signal"
        cv2.putText(img, text, (int(self.width * 0.08), int(self.height * 0.45)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 200, 200), 2)
        cv2.putText(img, status, (int(self.width * 0.08), int(self.height * 0.6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 150, 255), 2)
        return img

    def _open_camera(self) -> cv2.VideoCapture | None:
        backend = (
            cv2.CAP_AVFOUNDATION
            if sys.platform == "darwin"
            else (cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        )

        candidate_indices = [self.camera_index]
        if self.camera_index == 2:
            candidate_indices.extend([1, 0])
        elif self.camera_index == 1:
            candidate_indices.extend([0, 2])
        elif self.camera_index == 0:
            candidate_indices.extend([1, 2])

        for idx in candidate_indices:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened() and backend != cv2.CAP_ANY:
                cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.active_index = idx
                    LOGGER.info("✅ Camera %s successfully opened on index %d", self.label, idx)
                    return cap
                cap.release()

        return None

    def _capture_loop(self) -> None:
        cap = self._open_camera()
        while self.running:
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
        self.active_speakers: list[str] = []
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
                        LOGGER.info("AI Switch -> %s (conf=%.2f)", sug.scene_id, sug.confidence)
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


def apply_digital_zoom(frame: np.ndarray, zoom_factor: float = 1.35) -> np.ndarray:
    """Applies a smooth centered digital crop/zoom."""
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
) -> np.ndarray:
    """Draws a complete Studio Multi-View Dashboard with Program Out and Preview tiles."""
    canvas_w, canvas_h = 1600, 960
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (22, 22, 26)  # Dark sleek background

    # 1. Determine Program Out Content
    is_host = "host" in active_scene_id or active_scene_id == "s1_host"
    is_guest = "guest" in active_scene_id or active_scene_id == "s2_guest"
    is_wide = "wide" in active_scene_id or active_scene_id == "s3_wide"
    is_zoom = "zoom" in active_scene_id

    if is_host:
        prog_raw = apply_digital_zoom(frame_host, 1.3) if is_zoom else frame_host
        current_role_name = "CAM HOTE (FaceTime)" + (" [ZOOM SERRE]" if is_zoom else "")
    elif is_guest:
        prog_raw = apply_digital_zoom(frame_guest, 1.3) if is_zoom else frame_guest
        current_role_name = "CAM INVITE (iPhone)" + (" [ZOOM SERRE]" if is_zoom else "")
    else:  # Wide shot / side-by-side
        current_role_name = "PLAN LARGE / DEBAT (Split-Screen)"
        h, w = frame_host.shape[:2]
        half_w = w // 2
        split_left = frame_host[:, half_w // 2 : half_w // 2 + half_w]
        split_right = frame_guest[:, half_w // 2 : half_w // 2 + half_w]
        prog_raw = np.hstack([split_left, split_right])
        prog_raw = cv2.resize(prog_raw, (w, h))

    # 2. Place PROGRAM OUT (Top-Left, 1024x576)
    prog_w, prog_h = 1040, 585
    prog_x, prog_y = 30, 80
    prog_resized = cv2.resize(prog_raw, (prog_w, prog_h))
    canvas[prog_y : prog_y + prog_h, prog_x : prog_x + prog_w] = prog_resized

    # Red Tally Box around Program Out
    cv2.rectangle(canvas, (prog_x - 3, prog_y - 3), (prog_x + prog_w + 3, prog_y + prog_h + 3), (0, 0, 230), 4)

    # Program Out Badges
    cv2.rectangle(canvas, (prog_x, prog_y), (prog_x + 320, prog_y + 40), (0, 0, 200), -1)
    cv2.putText(canvas, "● PROGRAM (DIRECT DIFFUSE)", (prog_x + 12, prog_y + 26), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

    cv2.rectangle(canvas, (prog_x, prog_y + prog_h - 40), (prog_x + prog_w, prog_y + prog_h), (15, 15, 15), -1)
    cv2.putText(canvas, f"Source active : {current_role_name}", (prog_x + 15, prog_y + prog_h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 240, 255), 2)

    # 3. Place MULTIVIEW (Bottom Row: Cam Hote & Cam Invite, 480x270 each)
    thumb_w, thumb_h = 490, 275
    y_thumbs = 680

    # Thumb 1: Host (FaceTime)
    x_t1 = 30
    t1_resized = cv2.resize(frame_host, (thumb_w, thumb_h))
    canvas[y_thumbs : y_thumbs + thumb_h, x_t1 : x_t1 + thumb_w] = t1_resized
    t1_border_color = (0, 0, 230) if is_host else ((0, 200, 0) if not is_wide else (80, 80, 80))
    cv2.rectangle(canvas, (x_t1 - 2, y_thumbs - 2), (x_t1 + thumb_w + 2, y_thumbs + thumb_h + 2), t1_border_color, 3)
    cv2.rectangle(canvas, (x_t1, y_thumbs), (x_t1 + 220, y_thumbs + 32), (20, 20, 20), -1)
    cv2.putText(canvas, "CAM 1 - HOTE (FaceTime)", (x_t1 + 10, y_thumbs + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Thumb 2: Guest (iPhone)
    x_t2 = 580
    t2_resized = cv2.resize(frame_guest, (thumb_w, thumb_h))
    canvas[y_thumbs : y_thumbs + thumb_h, x_t2 : x_t2 + thumb_w] = t2_resized
    t2_border_color = (0, 0, 230) if is_guest else ((0, 200, 0) if not is_wide else (80, 80, 80))
    cv2.rectangle(canvas, (x_t2 - 2, y_thumbs - 2), (x_t2 + thumb_w + 2, y_thumbs + thumb_h + 2), t2_border_color, 3)
    cv2.rectangle(canvas, (x_t2, y_thumbs), (x_t2 + 220, y_thumbs + 32), (20, 20, 20), -1)
    cv2.putText(canvas, "CAM 2 - INVITE (iPhone)", (x_t2 + 10, y_thumbs + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # 4. Right Side: AI Control & Telemetry Dashboard
    panel_x, panel_y, panel_w, panel_h = 1100, 80, 470, 875
    cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (32, 32, 38), -1)
    cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (55, 55, 65), 2)

    # Panel Header
    cv2.putText(canvas, "CASTOR STUDIO IA", (panel_x + 25, panel_y + 45), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 220, 255), 2)
    cv2.putText(canvas, "Backend IA Decision Engine", (panel_x + 25, panel_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 170), 1)
    cv2.line(canvas, (panel_x + 20, panel_y + 90), (panel_x + panel_w - 20, panel_y + 90), (55, 55, 65), 1)

    # Telemetry Cards
    cur_y = panel_y + 130
    cv2.putText(canvas, "DECISION ACTUELLE :", (panel_x + 25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 190), 1)
    cur_y += 35
    dec_color = (0, 240, 255) if is_wide else ((0, 120, 255) if is_host else (255, 160, 50))
    cv2.putText(canvas, current_role_name, (panel_x + 25, cur_y), cv2.FONT_HERSHEY_DUPLEX, 0.75, dec_color, 2)

    cur_y += 45
    cv2.putText(canvas, f"Scene ID : {active_scene_id}", (panel_x + 25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cur_y += 30
    cv2.putText(canvas, f"Confiance IA : {confidence * 100:.0f}%", (panel_x + 25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 120), 1)
    cur_y += 30
    cv2.putText(canvas, f"Temps sur ce plan : {switch_age:.1f}s", (panel_x + 25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cur_y += 45
    cv2.line(canvas, (panel_x + 20, cur_y), (panel_x + panel_w - 20, cur_y), (55, 55, 65), 1)
    cur_y += 35

    cv2.putText(canvas, "REGLES DE REGIE AUTO :", (panel_x + 25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 190), 1)
    cur_y += 30
    cv2.putText(canvas, "• Hote parle seul -> Cadrage Hote", (panel_x + 25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cur_y += 25
    cv2.putText(canvas, "• Invite parle seul -> Cadrage Invite", (panel_x + 25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cur_y += 25
    cv2.putText(canvas, "• Debat / 2 micros -> Plan Large", (panel_x + 25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cur_y += 25
    cv2.putText(canvas, "• Silence (>3s) -> Plan Large", (panel_x + 25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cur_y += 25
    cv2.putText(canvas, "• Monologue (>5s) -> Zoom numerique", (panel_x + 25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    cur_y += 50
    cv2.line(canvas, (panel_x + 20, cur_y), (panel_x + panel_w - 20, cur_y), (55, 55, 65), 1)
    cur_y += 35
    cv2.putText(canvas, "CONTROLES :", (panel_x + 25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 190), 1)
    cur_y += 30
    cv2.putText(canvas, "[Q] / [ESC] : Quitter la regie", (panel_x + 25, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 255), 1)

    # Top Global Header
    cv2.putText(canvas, "CASTOR-AI STUDIO — RETOUR REGIE EN DIRECT (macOS)", (30, 48), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

    return canvas


def scan_available_cameras(max_tested: int = 5) -> list[int]:
    """Test camera indices and print which ones return valid live video frames."""
    print("\n🔍 Scanning available video capture devices...")
    working_indices = []
    for idx in range(max_tested):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"  ✅ Camera Index [{idx}] -> Active ({w}x{h})")
                working_indices.append(idx)
            else:
                print(f"  ⚠️  Camera Index [{idx}] -> Opened but failed to capture frame (Continuity Cam asleep?)")
            cap.release()
        else:
            print(f"  ❌ Camera Index [{idx}] -> Cannot open")
    print("")
    return working_indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--addr", default="localhost:50051", help="gRPC server address")
    parser.add_argument("--scan-cameras", action="store_true", help="Scan and list working OpenCV camera indices")
    parser.add_argument("--cam1", type=int, default=0, help="Camera index for Host. Default: 0")
    parser.add_argument("--mic1", default="1", help="Mic index/name for Host. Default: 1 (macOS) or 0 (Windows)")
    parser.add_argument("--cam2", type=int, default=1, help="Camera index for Guest. Default: 1")
    parser.add_argument("--mic2", default="2", help="Mic index/name for Guest. Default: 2 (macOS) or 1 (Windows)")
    parser.add_argument("--min-hold-time", type=float, default=2.5, help="Anti-flicker hold duration in seconds")
    parser.add_argument("--monologue_time", type=float, default=5.0, help="Monologue zoom threshold in seconds")
    parser.add_argument("--vad-threshold", type=float, default=0.5, help="Silero VAD threshold")
    args = parser.parse_args()

    if args.scan_cameras:
        scan_available_cameras()
        return

    print("\n🎬 Starting Castor Studio Live Multi-View Preview...")
    print(f"OS: {sys.platform}")
    print(f"Cam 1 (Host): {args.cam1} | Mic 1: {args.mic1}")
    print(f"Cam 2 (Guest): {args.cam2} | Mic 2: {args.mic2}")

    # 1. Start Camera Threads
    host_cam = CameraCaptureThread(args.cam1, "Host Cam")
    guest_cam = CameraCaptureThread(args.cam2, "Guest Cam")
    host_cam.start()
    guest_cam.start()

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
            frame_h = host_cam.get_frame()
            frame_g = guest_cam.get_frame()
            active_scene, conf, switch_time, status_msg = ai_bridge.get_state()
            switch_age = time.time() - switch_time

            dashboard = compose_studio_layout(
                frame_host=frame_h,
                frame_guest=frame_g,
                active_scene_id=active_scene,
                confidence=conf,
                switch_age=switch_age,
                status_msg=status_msg,
            )

            cv2.imshow(window_name, dashboard)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
                break
    finally:
        host_cam.stop()
        guest_cam.stop()
        ai_bridge.stop()
        cv2.destroyAllWindows()
        print("Studio preview closed cleanly.")


if __name__ == "__main__":
    main()
