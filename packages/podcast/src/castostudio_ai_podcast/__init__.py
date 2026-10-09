from __future__ import annotations

import asyncio
from functools import lru_cache
import logging
import re
import time
from collections.abc import Sequence

from castostudio_ai_core import AiModule, SceneDecision, SessionContext, Source

from .audio import AudioStreamReader
from .vad import SpeechModel

LOGGER = logging.getLogger(__name__)

_HOST_KEYWORDS = ["hote", "host", "presentateur", "presentatrice", "cam 1", "cam1"]
_GUEST_KEYWORDS = [
    "invite", "invité", "guest", "intervenant", "panelist", "panéliste",
]
# Numbered guest camera, e.g. "Invité 2", "Guest 3", "Cam 4" (cam1=host,
# cam2=1st guest, cam3=2nd guest, ...).
_NUMBERED_GUEST_RE = re.compile(
    r"(?:invite|invité|guest|intervenant|panelist|pan[ée]liste)\D*(\d+)|cam\s*(\d+)"
)


@lru_cache(maxsize=128)
def _classify_speaker_label(label: str) -> str | None:
    """Maps a camera label to a canonical speaker role: "host", "guest"
    (first/only guest), or "guest2", "guest3", ... for additional panelists.
    Returns None if the label doesn't look like a speaker camera at all
    (e.g. a wide shot). Generalizes the old hard-coded host/guest binary so
    panels with 3+ participants resolve to distinct roles instead of every
    non-host camera collapsing into "guest".
    """
    text = label.lower()

    if any(k in text for k in _HOST_KEYWORDS):
        return "host"

    match = _NUMBERED_GUEST_RE.search(text)
    if match:
        n = int(match.group(1) or match.group(2))
        guest_index = n - 1 if match.group(2) else n  # bare "camN" is offset by the host's cam1
        return "guest" if guest_index <= 1 else f"guest{guest_index}"

    if any(k in text for k in _GUEST_KEYWORDS):
        return "guest"

    return None


def _is_speaker_role(role: str) -> bool:
    """True for roles that carry their own audio feed (host, guest,
    guest2, ...) as opposed to "wide" or a "*_zoom" camera reusing another
    role's audio."""
    return role != "wide" and not role.endswith("_zoom")


class PodcastModule(AiModule):
    def __init__(self) -> None:
        self._audio_readers: dict[str, AudioStreamReader] = {}

        # Configuration
        self._min_hold_time = 2.5
        self._monologue_time = 5.0
        self._vad_threshold = 0.5
        self._min_silence_duration_ms = 400
        self._speech_pad_ms = 100
        self._min_speech_confirm_ms = 250
        self._cross_gating_threshold_db = 5.0
        self._debate_confirm_ms = 0
        self._silence_hold_time = 3.0
        self._ping_pong_threshold = 0
        self._ping_pong_window_sec = 6.0

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
        self._speaker_switches: list[tuple[float, str]] = []
        self._silence_start_time: float | None = None
        self._debate_start_time: float | None = None
        self._last_diag_log_time = 0.0

        # Backchannel filter: tracks how long each role has been continuously
        # speaking (raw VAD signal), so a brief "mm-hmm"/laugh doesn't count
        # as a real turn-taking change (see _confirm_speakers).
        self._speaking_since: dict[str, float] = {}

    async def start(self, context: SessionContext) -> None:
        # Configuration
        self._min_hold_time = float(context.config.get("min_hold_time", 2.5))
        self._monologue_time = float(context.config.get("monologue_time", 5.0))
        self._vad_threshold = float(context.config.get("vad_threshold", 0.5))
        self._min_silence_duration_ms = int(
            context.config.get("min_silence_duration_ms", 400)
        )
        self._speech_pad_ms = int(context.config.get("speech_pad_ms", 100))
        self._min_speech_confirm_ms = int(
            context.config.get("min_speech_confirm_ms", 250)
        )
        self._cross_gating_threshold_db = float(
            context.config.get("cross_gating_threshold_db", 5.0)
        )
        self._debate_confirm_ms = int(
            context.config.get("debate_confirm_ms", 0)
        )
        self._silence_hold_time = float(
            context.config.get("silence_hold_time", 3.0)
        )
        self._ping_pong_threshold = int(
            context.config.get("ping_pong_threshold", 0)
        )
        self._ping_pong_window_sec = float(
            context.config.get("ping_pong_window_sec", 6.0)
        )

        LOGGER.info(
            "[PodcastModule] config: min_hold_time=%.1f monologue_time=%.1f "
            "vad_threshold=%.2f min_silence_duration_ms=%d speech_pad_ms=%d "
            "min_speech_confirm_ms=%d cross_gating_threshold_db=%.1f "
            "debate_confirm_ms=%d silence_hold_time=%.1f ping_pong_threshold=%d",
            self._min_hold_time,
            self._monologue_time,
            self._vad_threshold,
            self._min_silence_duration_ms,
            self._speech_pad_ms,
            self._min_speech_confirm_ms,
            self._cross_gating_threshold_db,
            self._debate_confirm_ms,
            self._silence_hold_time,
            self._ping_pong_threshold,
        )

        # Load the VAD model now, off the event loop, so the first analysis
        # cycle isn't blocked by a cold model load.
        await asyncio.to_thread(SpeechModel.get)

    async def analyze_sources(self, sources: Sequence[Source]) -> SceneDecision | None:
        if not sources:
            return None

        # Execute high-performance synchronous state machine directly without thread pool overhead
        return self._analyze_sync(sources)

    def _analyze_sync(self, sources: Sequence[Source]) -> SceneDecision | None:
        now = time.monotonic()

        # 1. Resolve camera roles (cached; see _get_roles)
        roles = self._get_roles(sources)

        # 2. Manage audio reader threads
        self._update_audio_readers(sources, roles)

        # 3. Get active speakers: client-sent metadata first, VAD fallback
        active_speakers = []
        volumes = {}

        for role, scene_id in roles.items():
            if not _is_speaker_role(role):
                continue

            source = next((s for s in sources if s.scene_id == scene_id), None)
            reader = self._audio_readers.get(role)

            vol = reader.get_volume_db() if reader else -100.0
            is_speaking_meta = False

            if source and source.metadata:
                is_speaking_str = source.metadata.get("is_speaking", "").lower()
                active_speaker_str = source.metadata.get("active_speaker", "").lower()
                if is_speaking_str == "true" or active_speaker_str == "true":
                    is_speaking_meta = True
                if "volume_db" in source.metadata:
                    try:
                        vol = float(source.metadata["volume_db"])
                    except (ValueError, TypeError):
                        pass

            # Visual presence check: if camera metadata explicitly indicates an empty chair,
            # veto the audio trigger to prevent cutting to an empty room.
            # If person_detected is explicitly true, the speaker is present even if face is in profile.
            is_face_absent = False
            if source and source.metadata:
                face_meta = source.metadata.get("face_detected", "").lower()
                person_meta = source.metadata.get("person_detected", "").lower()
                if person_meta in ("true", "1"):
                    is_face_absent = False
                elif person_meta in ("false", "0"):
                    is_face_absent = True
                elif face_meta in ("false", "0"):
                    is_face_absent = True

            volumes[role] = vol
            is_speaking_audio = reader.is_speaking() if reader else False

            if (is_speaking_meta or is_speaking_audio) and not is_face_absent:
                active_speakers.append(role)

        # 4. Cross-gating: filter out mic bleed between adjacent microphones
        gated_speakers = self._apply_cross_gating(active_speakers, volumes)

        # 5. Backchannel filter: confirm sustained speech
        confirmed_speakers = self._confirm_speakers(gated_speakers, now)

        # 6. State machine ticks every cycle
        target_role = self._run_state_machine(confirmed_speakers, roles, now)

        if now - self._last_diag_log_time >= 1.0:
            self._last_diag_log_time = now
            LOGGER.info(
                "[PodcastModule] diag: active_speakers=%s gated=%s confirmed=%s volumes=%s "
                "state_machine_target=%s current_scene=%s hold_remaining=%.1f",
                active_speakers,
                gated_speakers,
                confirmed_speakers,
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

        # 7. Anti-flicker guard: gate emitting the switch, not tracking it.
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
        """Map roles (host, guest, guest2, ..., wide, host_zoom, guest_zoom,
        guest2_zoom, ...) to scene_ids. Generalized to N speakers: any
        camera whose label doesn't match a known speaker pattern falls back
        to sequential guestN assignment, instead of everyone but "host"
        collapsing into a single "guest" role.
        """
        roles: dict[str, str] = {}
        matched_sources = set()

        def matches_any(text: str, keywords: list[str]) -> bool:
            text = text.lower()
            return any(k in text for k in keywords)

        # 1. Look for Wide Shot
        for source in sources:
            if matches_any(source.label, ["large", "wide", "plan", "studio"]):
                roles["wide"] = source.scene_id
                matched_sources.add(source.scene_id)
                break

        # 2. Look for Zooms (per speaker)
        for source in sources:
            if source.scene_id in matched_sources:
                continue
            if matches_any(source.label, ["zoom", "serre", "serré", "tight", "face"]):
                base_role = _classify_speaker_label(source.label)
                if base_role and f"{base_role}_zoom" not in roles:
                    roles[f"{base_role}_zoom"] = source.scene_id
                    matched_sources.add(source.scene_id)

        # 3. Look for regular speaker cameras (host, guest, guest2, ...)
        for source in sources:
            if source.scene_id in matched_sources:
                continue
            base_role = _classify_speaker_label(source.label)
            if base_role and base_role not in roles:
                roles[base_role] = source.scene_id
                matched_sources.add(source.scene_id)

        # 4. Fallback based on indices for remaining roles
        unmatched = [s for s in sources if s.scene_id not in matched_sources]

        # The "3rd unlabeled camera is the wide shot" assumption only fits
        # the classic 3-camera rig (host, guest, wide). With 4+ cameras
        # (panels) it would wrongly steal a panelist's camera, so it's
        # scoped to exactly 3 sources; larger setups need an explicit
        # "large"/"wide"/"studio" label instead.
        if "wide" not in roles and len(sources) == 3 and unmatched:
            candidate = sources[2]
            if candidate.scene_id in {s.scene_id for s in unmatched}:
                roles["wide"] = candidate.scene_id
                unmatched = [s for s in unmatched if s.scene_id != candidate.scene_id]

        if "host" not in roles and unmatched:
            roles["host"] = unmatched.pop(0).scene_id

        if "guest" not in roles and unmatched:
            roles["guest"] = unmatched.pop(0).scene_id

        # Any remaining unlabeled cameras become extra panelists (guest2,
        # guest3, ...) instead of being dropped or misassigned to zooms.
        extra_guest_index = 2
        while unmatched:
            role_name = f"guest{extra_guest_index}"
            if role_name not in roles:
                roles[role_name] = unmatched.pop(0).scene_id
            extra_guest_index += 1

        # Make sure "wide" is at least mapped to something
        if "wide" not in roles and len(sources) >= 1:
            roles["wide"] = sources[-1].scene_id

        return roles

    def _update_audio_readers(self, sources: Sequence[Source], roles: dict[str, str]):
        active_role_urls = {}
        for role, scene_id in roles.items():
            if _is_speaker_role(role):
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

    def _apply_cross_gating(self, active_speakers: list[str], volumes: dict[str, float]) -> list[str]:
        """Filters out acoustic mic bleed (diaphonie) in real time.
        When multiple microphones trigger voice activity, compares relative volume levels.
        Any speaker whose volume is at least `cross_gating_threshold_db` lower than the
        loudest active speaker is discarded as room bleed from adjacent capsules.
        """
        if len(active_speakers) <= 1:
            return active_speakers

        speaker_vols = {r: volumes.get(r, -100.0) for r in active_speakers}
        max_role = max(speaker_vols, key=speaker_vols.get)
        max_vol = speaker_vols[max_role]

        gated = []
        for role in active_speakers:
            vol = speaker_vols[role]
            if max_vol - vol < self._cross_gating_threshold_db:
                gated.append(role)
            else:
                LOGGER.debug(
                    "[PodcastModule] Cross-gated bleed: %s (%.1f dB) discarded vs %s (%.1f dB)",
                    role,
                    vol,
                    max_role,
                    max_vol,
                )
        return gated

    def _confirm_speakers(self, raw_active_speakers: list[str], now: float) -> list[str]:
        """Filters `raw_active_speakers` down to roles that have been continuously
        speaking for at least `_min_speech_confirm_ms`. Without this, a single
        "mm-hmm" or a brief laugh above the VAD threshold steals focus.
        """
        raw_set = set(raw_active_speakers)

        for role in list(self._speaking_since.keys()):
            if role not in raw_set:
                del self._speaking_since[role]

        for role in raw_active_speakers:
            self._speaking_since.setdefault(role, now)

        threshold_seconds = self._min_speech_confirm_ms / 1000.0
        return [
            role
            for role in raw_active_speakers
            if now - self._speaking_since[role] >= threshold_seconds
        ]

    def _run_state_machine(self, active_speakers: list[str], roles: dict[str, str], now: float) -> str | None:
        normalized_speakers = sorted(set(active_speakers))

        # Case 1: Silence / Natural Pause
        if not normalized_speakers:
            self._debate_start_time = None
            if self._silence_start_time is None:
                self._silence_start_time = now

            # Generous silence before falling back to wide establishing shot
            if now - self._silence_start_time >= self._silence_hold_time:
                self._speaker_active_since.clear()
                self._last_active_speaker = None
                return "wide"

            # Natural pause: retain current speaker (or monologue zoom) without twitching
            if self._last_active_speaker is not None:
                has_zoom = f"{self._last_active_speaker}_zoom" in roles
                active_duration = now - self._speaker_active_since.get(self._last_active_speaker, now)
                if has_zoom and active_duration >= self._monologue_time:
                    return f"{self._last_active_speaker}_zoom"
                return self._last_active_speaker

            return None

        self._silence_start_time = None

        # Case 2: Debate / Multiple sustained speakers
        if len(normalized_speakers) > 1:
            if self._debate_start_time is None:
                self._debate_start_time = now

            # Require debate to be sustained before cutting to wide shot
            if now - self._debate_start_time >= (self._debate_confirm_ms / 1000.0):
                self._speaker_active_since.clear()
                self._last_active_speaker = None
                return "wide"

            # While debate is not yet confirmed, hold on the previous speaker if part of debate
            if self._last_active_speaker in normalized_speakers:
                return self._last_active_speaker
            return normalized_speakers[0]

        # Case 3: Single active speaker
        self._debate_start_time = None
        speaker = normalized_speakers[0]

        if speaker != self._last_active_speaker:
            if self._last_active_speaker is not None:
                self._speaker_switches.append((now, speaker))
                cutoff = now - self._ping_pong_window_sec
                self._speaker_switches = [s for s in self._speaker_switches if s[0] >= cutoff]

            self._speaker_active_since.clear()
            self._speaker_active_since[speaker] = now
            self._last_active_speaker = speaker

        # Ping-pong governor: if speakers alternate too rapidly in a short window,
        # cut to wide shot to keep broadcast steady
        if self._ping_pong_threshold > 0 and len(self._speaker_switches) >= self._ping_pong_threshold:
            return "wide"

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
