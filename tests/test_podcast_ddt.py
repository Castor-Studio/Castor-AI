from __future__ import annotations

import time
import pytest
from castostudio_ai_core import Source
from castostudio_ai_podcast import PodcastModule


@pytest.fixture
def make_module():
    def _factory(**kwargs):
        module = PodcastModule()
        for key, value in kwargs.items():
            setattr(module, f"_{key}", value)
        return module
    return _factory


# ---------------------------------------------------------------------------
# DATA-DRIVEN TEST 1: Backward Compatibility & Standard Audio Flow
# Ensures zero regression on standard multi-cam audio switches
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "host_meta, guest_meta, expected_scene",
    [
        # Only host speaks -> host scene
        ({"is_speaking": "true", "volume_db": "-15.0"}, {"is_speaking": "false", "volume_db": "-60.0"}, "s1_host"),
        # Only guest speaks -> guest scene
        ({"is_speaking": "false", "volume_db": "-60.0"}, {"is_speaking": "true", "volume_db": "-15.0"}, "s2_guest"),
        # Mic bleed: Host talks (-16dB), guest picks bleed (-28dB) -> Host only
        ({"is_speaking": "true", "volume_db": "-16.0"}, {"is_speaking": "true", "volume_db": "-28.0"}, "s1_host"),
        # Mic bleed inverse: Guest talks (-14dB), host picks bleed (-26dB) -> Guest only
        ({"is_speaking": "true", "volume_db": "-26.0"}, {"is_speaking": "true", "volume_db": "-14.0"}, "s2_guest"),
    ],
)
def test_ddt_audio_switching_backward_compatibility(host_meta, guest_meta, expected_scene, make_module, monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])

    module = make_module(min_hold_time=1.0, min_speech_confirm_ms=50, cross_gating_threshold_db=5.0)

    host = Source(scene_id="s1_host", url="rtmp://dummy/host", label="Cam Hote", metadata=host_meta)
    guest = Source(scene_id="s2_guest", url="rtmp://dummy/guest", label="Cam Invite", metadata=guest_meta)
    wide = Source(scene_id="s3_wide", url="rtmp://dummy/wide", label="Plan Large")

    try:
        # Tick 1: sound starts
        clock["t"] = 0.0
        module._analyze_sync([host, guest, wide])

        # Tick 2: speech confirmed
        clock["t"] = 0.1
        decision = module._analyze_sync([host, guest, wide])
        assert decision is not None
        assert decision.scene_id == expected_scene
    finally:
        for r in module._audio_readers.values():
            r.stop()


# ---------------------------------------------------------------------------
# DATA-DRIVEN TEST 2: Visual Presence Gating (Empty Chair / Veto Protection)
# If a mic picks up noise while a camera has NO face detected (empty chair),
# the director must veto the switch and avoid cutting to an empty room!
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "guest_face_detected, guest_person_detected, guest_vol_db, expected_switch_allowed",
    [
        # Face is present and speaking -> switch allowed
        ("true", None, "-16.0", True),
        # No face/person metadata provided (audio-only fallback) -> switch allowed (never breaks legacy)
        (None, None, "-16.0", True),
        # Camera is empty (chair empty, noise like door slam or mic bump) -> VETOED!
        ("false", None, "-16.0", False),
        ("0", None, "-14.0", False),
        (None, "false", "-15.0", False),
        # Profile view / looking away: face not detected but person detected -> switch allowed!
        ("false", "true", "-16.0", True),
    ],
)
def test_ddt_visual_presence_gating(guest_face_detected, guest_person_detected, guest_vol_db, expected_switch_allowed, make_module, monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])

    module = make_module(min_hold_time=1.0, min_speech_confirm_ms=50)

    host_talking = Source(scene_id="s1_host", url="rtmp://dummy/host", label="Cam Hote", metadata={"is_speaking": "true", "volume_db": "-15.0"})
    host_silent = Source(scene_id="s1_host", url="rtmp://dummy/host", label="Cam Hote", metadata={"is_speaking": "false", "volume_db": "-55.0"})

    guest_silent = Source(scene_id="s2_guest", url="rtmp://dummy/guest", label="Cam Invite", metadata={"is_speaking": "false", "volume_db": "-60.0"})

    guest_talking_meta = {"is_speaking": "true", "volume_db": guest_vol_db}
    if guest_face_detected is not None:
        guest_talking_meta["face_detected"] = guest_face_detected
    if guest_person_detected is not None:
        guest_talking_meta["person_detected"] = guest_person_detected
    guest_talking = Source(scene_id="s2_guest", url="rtmp://dummy/guest", label="Cam Invite", metadata=guest_talking_meta)

    wide = Source(scene_id="s3_wide", url="rtmp://dummy/wide", label="Plan Large")

    try:
        # Step 1: Put host on program cleanly
        clock["t"] = 0.0
        module._analyze_sync([host_talking, guest_silent, wide])
        clock["t"] = 0.1
        decision = module._analyze_sync([host_talking, guest_silent, wide])
        assert decision is not None and decision.scene_id == "s1_host"

        # Step 2: Host stops, Guest mic triggers at t=1.5s (> min_hold_time)
        clock["t"] = 1.5
        module._analyze_sync([host_silent, guest_talking, wide])
        clock["t"] = 1.6
        decision = module._analyze_sync([host_silent, guest_talking, wide])

        if expected_switch_allowed:
            assert decision is not None
            assert decision.scene_id == "s2_guest"
        else:
            # Switch vetoed: must NOT switch to guest!
            assert decision is None or decision.scene_id != "s2_guest"
    finally:
        for r in module._audio_readers.values():
            r.stop()


# ---------------------------------------------------------------------------
# DATA-DRIVEN TEST 3: Anti-Ping-Pong Governor
# When rapid back-and-forth banter occurs (>= 3 switches in < 6 seconds),
# director cleanly falls back to Wide shot to prevent viewer dizziness.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "ping_pong_enabled, expected_wide_trigger",
    [
        (True, True),
        (False, False),
    ],
)
def test_ddt_anti_ping_pong_governor(ping_pong_enabled, expected_wide_trigger, make_module, monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])

    threshold = 3 if ping_pong_enabled else 0
    module = make_module(
        min_hold_time=0.8,
        min_speech_confirm_ms=50,
        ping_pong_threshold=threshold,
        ping_pong_window_sec=6.0,
    )

    host_spk = Source(scene_id="s1_host", url="rtmp://dummy/host", label="Cam Hote", metadata={"is_speaking": "true", "volume_db": "-16.0"})
    host_sil = Source(scene_id="s1_host", url="rtmp://dummy/host", label="Cam Hote", metadata={"is_speaking": "false", "volume_db": "-55.0"})
    guest_spk = Source(scene_id="s2_guest", url="rtmp://dummy/guest", label="Cam Invite", metadata={"is_speaking": "true", "volume_db": "-16.0"})
    guest_sil = Source(scene_id="s2_guest", url="rtmp://dummy/guest", label="Cam Invite", metadata={"is_speaking": "false", "volume_db": "-55.0"})
    wide = Source(scene_id="s3_wide", url="rtmp://dummy/wide", label="Plan Large")

    try:
        # Turn 1: Host speaks (t=0.0 -> confirmed at t=0.1)
        clock["t"] = 0.0
        module._analyze_sync([host_spk, guest_sil, wide])
        clock["t"] = 0.1
        d1 = module._analyze_sync([host_spk, guest_sil, wide])
        assert d1 is not None and d1.scene_id == "s1_host"

        # Turn 2: Guest speaks quickly at t=1.0 (> min_hold 0.8s)
        clock["t"] = 1.0
        module._analyze_sync([host_sil, guest_spk, wide])
        clock["t"] = 1.1
        d2 = module._analyze_sync([host_sil, guest_spk, wide])
        assert d2 is not None and d2.scene_id == "s2_guest"

        # Turn 3: Host cuts back immediately at t=2.0
        clock["t"] = 2.0
        module._analyze_sync([host_spk, guest_sil, wide])
        clock["t"] = 2.1
        d3 = module._analyze_sync([host_spk, guest_sil, wide])
        assert d3 is not None and d3.scene_id == "s1_host"

        # Turn 4: Guest replies immediately at t=3.0 (rapid banter!)
        clock["t"] = 3.0
        module._analyze_sync([host_sil, guest_spk, wide])
        clock["t"] = 3.1
        d4 = module._analyze_sync([host_sil, guest_spk, wide])

        if expected_wide_trigger:
            # Ping-pong governor catches rapid barrage and cuts to Wide shot!
            assert d4 is not None
            assert d4.scene_id == "s3_wide"
        else:
            # Without governor, standard switch to guest
            assert d4 is not None
            assert d4.scene_id == "s2_guest"
    finally:
        for r in module._audio_readers.values():
            r.stop()


# ---------------------------------------------------------------------------
# DATA-DRIVEN TEST 4: Preallocated ChunkBuffer Throughput & Invariance
# Tests across diverse audio frame push sizes to guarantee sub-millisecond execution
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "push_size, num_pushes, expected_chunks",
    [
        (128, 16, 4),    # 128 * 16 = 2048 samples -> 4 chunks of 512
        (256, 10, 5),    # 256 * 10 = 2560 samples -> 5 chunks of 512
        (512, 8, 8),     # 512 * 8 = 4096 samples -> 8 chunks of 512
        (1024, 5, 10),   # 1024 * 5 = 5120 samples -> 10 chunks of 512
        (2048, 3, 12),   # 2048 * 3 = 6144 samples -> 12 chunks of 512
        (700, 5, 6),     # 700 * 5 = 3500 samples -> 6 chunks of 512 (3072 consumed, 428 remainder)
    ],
)
def test_ddt_chunk_buffer_throughput_matrix(push_size, num_pushes, expected_chunks):
    from castostudio_ai_podcast.audio import ChunkBuffer
    import numpy as np

    buf = ChunkBuffer(chunk_size=512)
    raw_data = np.linspace(-0.9, 0.9, push_size, dtype=np.float32)

    total_chunks = 0
    t0 = time.perf_counter()
    for _ in range(num_pushes):
        extracted = buf.push(raw_data)
        total_chunks += len(extracted)
    duration = time.perf_counter() - t0

    assert total_chunks == expected_chunks
    # Microsecond throughput assertion: all pushes combined must take < 5ms
    assert duration < 0.005, f"ChunkBuffer too slow for push_size={push_size}: {duration*1000:.3f}ms"

