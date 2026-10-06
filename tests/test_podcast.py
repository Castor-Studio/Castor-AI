from __future__ import annotations

import time

import numpy as np
import pytest

from castostudio_ai_core import Source
from castostudio_ai_podcast import PodcastModule
from castostudio_ai_podcast.audio import ChunkBuffer, pcm16_to_float32, rms_to_db
from castostudio_ai_podcast.vad import VAD_CHUNK_SAMPLES, SpeechActivityTracker


def test_chunk_buffer_yields_fixed_size_chunks():
    buffer = ChunkBuffer(chunk_size=512)

    chunks = buffer.push(np.zeros(1000, dtype=np.float32))

    assert [c.size for c in chunks] == [512]


def test_chunk_buffer_carries_remainder_across_pushes():
    buffer = ChunkBuffer(chunk_size=512)

    first = buffer.push(np.zeros(300, dtype=np.float32))
    second = buffer.push(np.zeros(300, dtype=np.float32))

    assert first == []
    assert len(second) == 1
    assert second[0].size == 512


def test_pcm16_to_float32_normalizes_to_unit_range():
    samples = np.array([32767, -32768, 0], dtype=np.int16)

    floats = pcm16_to_float32(samples)

    assert floats[0] == pytest.approx(1.0, abs=1e-3)
    assert floats[1] == -1.0
    assert floats[2] == 0.0


def test_rms_to_db_silence_and_full_scale():
    assert rms_to_db(0.0) == -100.0
    assert rms_to_db(1.0) == 0.0


def test_vad_tracker_reports_not_speaking_on_digital_silence():
    tracker = SpeechActivityTracker()
    silence = np.zeros(VAD_CHUNK_SAMPLES, dtype=np.float32)

    for _ in range(10):
        speaking = tracker.process_chunk(silence)

    assert speaking is False
    assert tracker.is_speaking() is False


def test_vad_trackers_are_independent_across_streams():
    host_tracker = SpeechActivityTracker()
    guest_tracker = SpeechActivityTracker()
    silence = np.zeros(VAD_CHUNK_SAMPLES, dtype=np.float32)

    host_tracker.process_chunk(silence)
    guest_tracker.process_chunk(silence)

    assert host_tracker is not guest_tracker
    assert host_tracker.is_speaking() is False
    assert guest_tracker.is_speaking() is False


def test_role_cache_stable_when_scene_ids_reappear_in_different_order():
    module = PodcastModule()
    first_pass = [
        Source(scene_id="s1", url="rtmp://a", label="Cam Hote"),
        Source(scene_id="s2", url="rtmp://b", label="Cam Invite"),
    ]

    roles_first = module._get_roles(first_pass)

    reordered = [first_pass[1], first_pass[0]]
    roles_second = module._get_roles(reordered)

    assert roles_first == roles_second


def test_role_cache_recomputes_when_scene_set_changes():
    module = PodcastModule()
    initial = [
        Source(scene_id="s1", url="rtmp://a", label="Cam Hote"),
        Source(scene_id="s2", url="rtmp://b", label="Cam Invite"),
    ]
    module._get_roles(initial)

    with_new_source = initial + [Source(scene_id="s3", url="rtmp://c", label="Plan Large")]
    roles = module._get_roles(with_new_source)

    assert roles["wide"] == "s3"


def test_state_machine_ticks_through_hold_window(monkeypatch):
    """Regression test: a speaker change during the min_hold_time window must
    still be timed from when it actually started, not from whenever the hold
    happens to expire. Uses a hold window (6s) longer than the monologue
    threshold (2s) so a frozen internal clock and a live one disagree on
    whether "guest" has been talking long enough to zoom in.
    """
    clock = {"t": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])

    module = PodcastModule()
    module._min_hold_time = 6.0
    module._monologue_time = 2.0
    # This test is about hold/monologue timing, not the backchannel filter -
    # disable the latter so speech confirms instantly, as before its addition.
    module._min_speech_confirm_ms = 0

    host = Source(scene_id="s-host", url="rtmp://invalid/host", label="Cam Hote")
    guest = Source(scene_id="s-guest", url="rtmp://invalid/guest", label="Cam Invite")
    guest_zoom = Source(
        scene_id="s-guest-zoom", url="rtmp://invalid/guest-zoom", label="Cam Invite Zoom"
    )
    host_speaking = Source(
        scene_id=host.scene_id, url=host.url, label=host.label,
        metadata={"is_speaking": "true"},
    )
    guest_speaking = Source(
        scene_id=guest.scene_id, url=guest.url, label=guest.label,
        metadata={"is_speaking": "true"},
    )

    try:
        # t=0: host speaks, first decision is never gated -> immediate switch.
        decision = module._analyze_sync([host_speaking, guest, guest_zoom])
        assert decision is not None and decision.scene_id == "s-host"

        # t=0.5: guest starts talking while still inside the 6s hold window.
        clock["t"] = 0.5
        decision = module._analyze_sync([host, guest_speaking, guest_zoom])
        assert decision is None

        # t=3.0: guest still talking, 2.5s in - past monologue_time already,
        # but still gated. The internal timer must reflect this now.
        clock["t"] = 3.0
        decision = module._analyze_sync([host, guest_speaking, guest_zoom])
        assert decision is None

        # t=6.5: hold expires. Guest has been talking since t=0.5 (6s), well
        # past monologue_time, so the switch must land on guest_zoom
        # directly - not reset to a fresh 0s-old "guest" decision.
        clock["t"] = 6.5
        decision = module._analyze_sync([host, guest_speaking, guest_zoom])
        assert decision is not None
        assert decision.scene_id == "s-guest-zoom"
    finally:
        for reader in module._audio_readers.values():
            reader.stop()
