from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence

from castostudio_ai_core import AiModule, SceneDecision, SessionContext, Source

from .audio import AudioStreamReader
from .vad import SpeechModel

LOGGER = logging.getLogger(__name__)

AUDIO_ROLES = ("host", "guest")


class PodcastModule(AiModule):
    def __init__(self) -> None:
        self._audio_readers: dict[str, AudioStreamReader] = {}

        # Configuration
        self._min_hold_time = 3.0
        self._monologue_time = 5.0
        self._vad_threshold = 0.5
        self._min_silence_duration_ms = 400
        self._speech_pad_ms = 100

        # Role cache: recomputed only when the (scene_id, label) set changes,
        # so source-list reordering between cycles can't flip role assignment
        # and force audio readers to reconnect for no reason. Label is part
        # of the fingerprint so a scene_id that gets relabeled (camera swap
        # behind a stable id) still re-resolves its role.
        self._role_cache: dict[str, str] = {}
        self._roles_fingerprint: frozenset[tuple[str, str]] = frozenset()

        # State machine
        self._current_scene_id: str | None = None
        self._last_switch_time = 0.0
        self._speaker_active_since: dict[str, float] = {}
        self._last_active_speaker: str | None = None
        self._silence_start_time: float | None = None
        self._last_diag_log_time = 0.0

    async def start(self, context: SessionContext) -> None:
        # Configuration
        self._min_hold_time = float(context.config.get("min_hold_time", 3.0))
        self._monologue_time = float(context.config.get("monologue_time", 5.0))
        self._vad_threshold = float(context.config.get("vad_threshold", 0.5))
        self._min_silence_duration_ms = int(
            context.config.get("min_silence_duration_ms", 400)
        )
        self._speech_pad_ms = int(context.config.get("speech_pad_ms", 100))

        LOGGER.info(
            "[PodcastModule] config: min_hold_time=%.1f monologue_time=%.1f "
            "vad_threshold=%.2f min_silence_duration_ms=%d speech_pad_ms=%d",
            self._min_hold_time,
            self._monologue_time,
            self._vad_threshold,
            self._min_silence_duration_ms,
            self._speech_pad_ms,
        )

        # Load the VAD model now, off the event loop, so the first analysis
        # cycle isn't blocked by a cold model load.
        await asyncio.to_thread(SpeechModel.get)

    async def analyze_sources(self, sources: Sequence[Source]) -> SceneDecision | None:
        if not sources:
            return None

        # Delegate the parsing and state machine to thread for safe execution
        return await asyncio.to_thread(self._analyze_sync, sources)

    def _analyze_sync(self, sources: Sequence[Source]) -> SceneDecision | None:
        now = time.monotonic()

        # 1. Resolve camera roles (cached; see _get_roles)
        roles = self._get_roles(sources)

        # 2. Manage audio reader threads
        self._update_audio_readers(sources, roles)

        # 3. Get active speakers: client-sent metadata first, VAD fallback
        active_speakers = []
        volumes = {}

        for role, reader in self._audio_readers.items():
            scene_id = roles.get(role)
            source = next((s for s in sources if s.scene_id == scene_id), None)

            is_speaking_meta = False
            if source and source.metadata:
                is_speaking_str = source.metadata.get("is_speaking", "").lower()
                active_speaker_str = source.metadata.get("active_speaker", "").lower()
                if is_speaking_str == "true" or active_speaker_str == "true":
                    is_speaking_meta = True

            volumes[role] = reader.get_volume_db()

            if is_speaking_meta or reader.is_speaking():
                active_speakers.append(role)

        # 4. State machine ticks every cycle, even while a switch is being
        # held back below. It must see every cycle to keep its internal
        # timers (silence duration, monologue duration) accurate — skipping
        # ticks during the hold window used to freeze that clock, so the
        # decision made right as the hold expired was driven by whatever
        # single noisy sample happened to land at that instant instead of
        # the real sustained state.
        target_role = self._run_state_machine(active_speakers, roles, now)

        if now - self._last_diag_log_time >= 1.0:
            self._last_diag_log_time = now
            LOGGER.info(
                "[PodcastModule] diag: active_speakers=%s volumes=%s state_machine_target=%s "
                "current_scene=%s hold_remaining=%.1f",
                active_speakers,
                {role: round(db, 1) for role, db in volumes.items()},
                target_role,
                self._current_scene_id,
                max(0.0, self._min_hold_time - (now - self._last_switch_time)),
            )

        if target_role is None:
            return None

        # Get scene ID for target role
        target_scene_id = roles.get(target_role)
        if target_scene_id is None:
            # Fallback to the first available source
            target_scene_id = sources[0].scene_id

        if target_scene_id == self._current_scene_id:
            return None

        # 5. Anti-flicker guard: gate emitting the switch, not tracking it.
        if self._current_scene_id is not None and (now - self._last_switch_time < self._min_hold_time):
            return None

        self._current_scene_id = target_scene_id
        self._last_switch_time = now
        LOGGER.info(
            "[PodcastModule] Decision: Switch to role '%s' (scene_id='%s')",
            target_role,
            target_scene_id,
        )

        return SceneDecision(scene_id=target_scene_id, confidence=0.9)

    def _get_roles(self, sources: Sequence[Source]) -> dict[str, str]:
        fingerprint = frozenset((source.scene_id, source.label) for source in sources)
        if fingerprint != self._roles_fingerprint:
            self._role_cache = self._parse_roles(sources)
            self._roles_fingerprint = fingerprint
        return self._role_cache

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
            if role in AUDIO_ROLES:
                source = next((s for s in sources if s.scene_id == scene_id), None)
                if source:
                    active_role_urls[role] = source.url

        # Remove readers that are no longer active, or whose URL changed
        for role in list(self._audio_readers.keys()):
            target_url = active_role_urls.get(role)
            if target_url is None or self._audio_readers[role].url != target_url:
                self._audio_readers[role].stop()
                del self._audio_readers[role]

        # Start new readers
        for role, url in active_role_urls.items():
            if role not in self._audio_readers:
                reader = AudioStreamReader(
                    url,
                    role,
                    vad_threshold=self._vad_threshold,
                    min_silence_duration_ms=self._min_silence_duration_ms,
                    speech_pad_ms=self._speech_pad_ms,
                )
                reader.start()
                self._audio_readers[role] = reader

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

        LOGGER.info("[PodcastModule] Stopped and cleaned resources")
