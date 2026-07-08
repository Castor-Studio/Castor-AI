from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Sequence

import cv2
import torch
from castostudio_ai_core import AiModule, SceneDecision, SessionContext, Source
from ultralytics import YOLO

from .audio import AudioVolumeReader

LOGGER = logging.getLogger(__name__)

class WideCameraViewer:
    def __init__(self, url: str, model: YOLO, device: str):
        self.url = url
        self.model = model
        self.device = device
        self.running = False
        self.thread = None
        self.people_count = 0
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, name="WideViewer", daemon=True)
        self.thread.start()
        LOGGER.info("Wide camera viewer thread started for %s", self.url)

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
        LOGGER.info("Wide camera viewer thread stopped")

    def get_people_count(self) -> int:
        with self.lock:
            return self.people_count

    def _run(self):
        cap = cv2.VideoCapture(self.url)
        if not cap.isOpened():
            LOGGER.warning("Could not open wide camera stream: %s", self.url)
            return
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        while self.running:
            try:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.5)
                    continue
                
                # Predict persons (class 0)
                results = self.model.predict(
                    frame,
                    conf=0.3,
                    classes=[0],
                    device=self.device,
                    verbose=False
                )[0]
                
                count = len(results.boxes)
                with self.lock:
                    self.people_count = count
            except Exception as exc:
                LOGGER.warning("YOLO detection error: %s", exc)
                time.sleep(1)
            time.sleep(1.0)
        cap.release()


class PodcastModule(AiModule):
    def __init__(self) -> None:
        self._audio_readers: dict[str, AudioVolumeReader] = {}
        self._wide_viewer: WideCameraViewer | None = None
        self._yolo_model: YOLO | None = None
        self._device = "cpu"
        
        # Configuration
        self._db_threshold = -40.0
        self._min_hold_time = 3.0
        self._monologue_time = 5.0
        
        # State machine
        self._current_scene_id: str | None = None
        self._last_switch_time = 0.0
        self._speaker_active_since: dict[str, float] = {}
        self._last_active_speaker: str | None = None
        self._silence_start_time: float | None = None

    async def start(self, context: SessionContext) -> None:
        # Load configuration
        self._db_threshold = float(context.config.get("db_threshold", -40.0))
        self._min_hold_time = float(context.config.get("min_hold_time", 3.0))
        self._monologue_time = float(context.config.get("monologue_time", 5.0))
        
        LOGGER.info("[PodcastModule] config: db_threshold=%.1f min_hold_time=%.1f monologue_time=%.1f",
                    self._db_threshold, self._min_hold_time, self._monologue_time)

        # Load YOLO26 model with GPU acceleration in a separate thread to prevent blocking the server
        await asyncio.to_thread(self._initialize_yolo)

    def _initialize_yolo(self):
        try:
            if torch.backends.mps.is_available():
                self._device = "mps"
            elif torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"
            
            LOGGER.info("[PodcastModule] Loading YOLO model on device: %s", self._device)
            # Load the yolov8n.pt model (Ultralytics auto-downloads it if not present)
            self._yolo_model = YOLO("yolov8n.pt")
            LOGGER.info("[PodcastModule] YOLO loaded successfully")
        except Exception as exc:
            LOGGER.warning("[PodcastModule] Could not load YOLO model, falling back to audio-only. Error: %s", exc)
            self._yolo_model = None

    async def analyze_sources(self, sources: Sequence[Source]) -> SceneDecision | None:
        if not sources:
            return None

        # Delegate the parsing and state machine to thread for safe execution
        return await asyncio.to_thread(self._analyze_sync, sources)

    def _analyze_sync(self, sources: Sequence[Source]) -> SceneDecision | None:
        now = time.monotonic()
        
        # 1. Parse camera roles
        roles = self._parse_roles(sources)
        
        # 2. Manage audio reader threads
        self._update_audio_readers(sources, roles)
        
        # 3. Manage wide camera YOLO viewer
        self._update_wide_viewer_with_sources(sources, roles)

        # 4. Get active speakers based on volume or metadata
        active_speakers = []
        volumes = {}
        
        for role, reader in self._audio_readers.items():
            # Find the corresponding source for this role to check metadata
            scene_id = roles.get(role)
            source = next((s for s in sources if s.scene_id == scene_id), None)
            
            # Check metadata first (bypass if client sends speaking status)
            is_speaking_meta = False
            if source and source.metadata:
                is_speaking_str = source.metadata.get("is_speaking", "").lower()
                active_speaker_str = source.metadata.get("active_speaker", "").lower()
                if is_speaking_str == "true" or active_speaker_str == "true":
                    is_speaking_meta = True
            
            if is_speaking_meta:
                active_speakers.append(role)
                volumes[role] = 0.0 # Virtual active volume
            else:
                # Fallback to audio reader RMS
                db = reader.get_volume_db()
                volumes[role] = db
                if db >= self._db_threshold:
                    active_speakers.append(role)

        LOGGER.debug("[PodcastModule] Active speakers: %s, Volumes: %s", active_speakers, volumes)

        # 5. Anti-flicker guard
        if self._current_scene_id is not None and (now - self._last_switch_time < self._min_hold_time):
            return None

        # 6. State Machine logic to select the target role
        target_role = self._run_state_machine(active_speakers, roles, now)
        if target_role is None:
            return None

        # Get scene ID for target role
        target_scene_id = roles.get(target_role)
        if target_scene_id is None:
            # Fallback to the first available source
            target_scene_id = sources[0].scene_id

        if target_scene_id == self._current_scene_id:
            return None

        self._current_scene_id = target_scene_id
        self._last_switch_time = now
        LOGGER.info("[PodcastModule] Decision: Switch to role '%s' (scene_id='%s')", target_role, target_scene_id)
        
        return SceneDecision(scene_id=target_scene_id, confidence=0.9)

    def _parse_roles(self, sources: Sequence[Source]) -> dict[str, str]:
        """Map roles (host, guest, wide, host_zoom, guest_zoom) to scene_ids."""
        roles = {}
        matched_sources = set()

        # Helper to check keywords
        def matches_any(text: str, keywords: list[str]) -> bool:
            text = text.lower()
            return any(k in text for k in keywords)

        # 1. Look for Wide Shot
        for source in sources:
            if matches_any(source.label, ["large", "wide", "plan", "studio"]):
                roles["wide"] = source.scene_id
                matched_sources.add(source.scene_id)
                break

        # 2. Look for Zooms
        for source in sources:
            if source.scene_id in matched_sources:
                continue
            if matches_any(source.label, ["zoom", "serre", "serré", "tight", "face"]):
                if matches_any(source.label, ["hote", "host", "cam 1", "cam1"]):
                    roles["host_zoom"] = source.scene_id
                    matched_sources.add(source.scene_id)
                elif matches_any(source.label, ["invite", "invité", "guest", "cam 2", "cam2"]):
                    roles["guest_zoom"] = source.scene_id
                    matched_sources.add(source.scene_id)

        # 3. Look for regular Host and Guest
        for source in sources:
            if source.scene_id in matched_sources:
                continue
            if matches_any(source.label, ["hote", "host", "cam 1", "cam1"]):
                roles["host"] = source.scene_id
                matched_sources.add(source.scene_id)
            elif matches_any(source.label, ["invite", "invité", "guest", "cam 2", "cam2"]):
                roles["guest"] = source.scene_id
                matched_sources.add(source.scene_id)

        # 4. Fallback based on indices for remaining roles
        unmatched = [s for s in sources if s.scene_id not in matched_sources]
        
        if "wide" not in roles and unmatched:
            if len(sources) >= 3:
                roles["wide"] = sources[2].scene_id
                if sources[2].scene_id in unmatched:
                    unmatched.remove(sources[2].scene_id)

        if "host" not in roles and len(sources) >= 1:
            roles["host"] = sources[0].scene_id
            if sources[0].scene_id in unmatched:
                unmatched.remove(sources[0].scene_id)

        if "guest" not in roles and len(sources) >= 2:
            roles["guest"] = sources[1].scene_id
            if sources[1].scene_id in unmatched:
                unmatched.remove(sources[1].scene_id)

        if unmatched:
            if "host_zoom" not in roles:
                roles["host_zoom"] = unmatched.pop(0).scene_id
            elif "guest_zoom" not in roles and unmatched:
                roles["guest_zoom"] = unmatched.pop(0).scene_id

        # Make sure "wide" is at least mapped to something
        if "wide" not in roles and len(sources) >= 1:
            roles["wide"] = sources[-1].scene_id

        return roles

    def _update_audio_readers(self, sources: Sequence[Source], roles: dict[str, str]):
        active_role_urls = {}
        for role, scene_id in roles.items():
            if role in ["host", "guest", "host_zoom", "guest_zoom"]:
                source = next((s for s in sources if s.scene_id == scene_id), None)
                if source:
                    active_role_urls[role] = source.url

        # Remove readers that are no longer active
        for role in list(self._audio_readers.keys()):
            if role not in active_role_urls:
                self._audio_readers[role].stop()
                del self._audio_readers[role]

        # Start new readers
        for role, url in active_role_urls.items():
            if role not in self._audio_readers:
                reader = AudioVolumeReader(url, role)
                reader.start()
                self._audio_readers[role] = reader

    def _update_wide_viewer_with_sources(self, sources: Sequence[Source], roles: dict[str, str]):
        wide_scene_id = roles.get("wide")
        if wide_scene_id is None or self._yolo_model is None:
            if self._wide_viewer:
                self._wide_viewer.stop()
                self._wide_viewer = None
            return

        source = next((s for s in sources if s.scene_id == wide_scene_id), None)
        if not source:
            if self._wide_viewer:
                self._wide_viewer.stop()
                self._wide_viewer = None
            return

        if self._wide_viewer is None or self._wide_viewer.url != source.url:
            if self._wide_viewer:
                self._wide_viewer.stop()
            self._wide_viewer = WideCameraViewer(source.url, self._yolo_model, self._device)
            self._wide_viewer.start()

    def _run_state_machine(self, active_speakers: list[str], roles: dict[str, str], now: float) -> str | None:
        normalized_speakers = []
        for speaker in active_speakers:
            if "host" in speaker:
                normalized_speakers.append("host")
            elif "guest" in speaker:
                normalized_speakers.append("guest")
        normalized_speakers = list(set(normalized_speakers))

        # Case 1: Silence
        if not normalized_speakers:
            if self._silence_start_time is None:
                self._silence_start_time = now
            
            if now - self._silence_start_time >= 3.0:
                self._speaker_active_since.clear()
                self._last_active_speaker = None
                return "wide"
            
            return None

        self._silence_start_time = None

        # Case 2: Debate / Multiple speakers
        if len(normalized_speakers) > 1:
            self._speaker_active_since.clear()
            self._last_active_speaker = None
            return "wide"

        # Case 3: Single active speaker
        speaker = normalized_speakers[0]
        
        if speaker != self._last_active_speaker:
            self._speaker_active_since.clear()
            self._speaker_active_since[speaker] = now
            self._last_active_speaker = speaker

        active_duration = now - self._speaker_active_since.get(speaker, now)
        
        has_zoom = f"{speaker}_zoom" in roles
        
        if has_zoom and active_duration >= self._monologue_time:
            return f"{speaker}_zoom"
        else:
            return speaker

    async def stop(self) -> None:
        # Stop all threads
        for reader in list(self._audio_readers.values()):
            await asyncio.to_thread(reader.stop)
        self._audio_readers.clear()

        if self._wide_viewer:
            await asyncio.to_thread(self._wide_viewer.stop)
            self._wide_viewer = None
        
        self._yolo_model = None
        LOGGER.info("[PodcastModule] Stopped and cleaned resources")
