from __future__ import annotations

import time
import pytest
from castostudio_ai_core import Source
from castostudio_ai_podcast import PodcastModule


@pytest.fixture
def base_sources():
    host = Source(scene_id="s1_host", url="rtmp://dummy/host", label="Cam Hote")
    guest = Source(scene_id="s2_guest", url="rtmp://dummy/guest", label="Cam Invite")
    wide = Source(scene_id="s3_wide", url="rtmp://dummy/wide", label="Plan Large")
    host_zoom = Source(scene_id="s1_host_zoom", url="rtmp://dummy/host", label="Zoom Hote")
    guest_zoom = Source(scene_id="s2_guest_zoom", url="rtmp://dummy/guest", label="Zoom Invite")
    return host, guest, wide, host_zoom, guest_zoom


def test_cross_gating_eliminates_mic_bleed(monkeypatch, base_sources):
    """When both mics detect speech (VAD=True) due to acoustic bleed,
    the cross-gating filter must compare relative dB and keep ONLY the loudest speaker.
    Host: -18 dB (dominant), Guest: -28 dB (bleed).
    Result: Host is chosen, NO false switch to Wide (debate) shot.
    """
    clock = {"t": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])

    module = PodcastModule()
    module._min_hold_time = 2.0
    module._min_speech_confirm_ms = 100
    module._cross_gating_threshold_db = 6.0

    host, guest, wide, host_zoom, guest_zoom = base_sources

    # Host speaks loud into their mic, guest mic picks up faint room bleed
    host_src = Source(
        scene_id=host.scene_id, url=host.url, label=host.label,
        metadata={"is_speaking": "true", "volume_db": "-18.0"},
    )
    guest_src = Source(
        scene_id=guest.scene_id, url=guest.url, label=guest.label,
        metadata={"is_speaking": "true", "volume_db": "-28.0"},
    )

    try:
        # At t=0.0: sound arrives, recorded in tracker
        clock["t"] = 0.0
        module._analyze_sync([host_src, guest_src, wide, host_zoom, guest_zoom])

        # At t=0.15s: speech is confirmed (>100ms)
        clock["t"] = 0.15
        decision = module._analyze_sync([host_src, guest_src, wide, host_zoom, guest_zoom])
        assert decision is not None
        # Must pick host, NOT wide!
        assert decision.scene_id == "s1_host"
    finally:
        for reader in module._audio_readers.values():
            reader.stop()


def test_cross_gating_triggers_wide_on_sustained_equal_volume_debate(monkeypatch, base_sources):
    """When both speakers actually speak simultaneously at similar volumes (delta < 5dB)
    for longer than the debate confirmation threshold, the director switches to Wide.
    """
    clock = {"t": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])

    module = PodcastModule()
    module._min_hold_time = 1.0
    module._min_speech_confirm_ms = 100
    module._debate_confirm_ms = 500
    module._cross_gating_threshold_db = 6.0

    host, guest, wide, host_zoom, guest_zoom = base_sources

    # Host starts talking first
    host_src = Source(
        scene_id=host.scene_id, url=host.url, label=host.label,
        metadata={"is_speaking": "true", "volume_db": "-18.0"},
    )
    guest_silent = Source(
        scene_id=guest.scene_id, url=guest.url, label=guest.label,
        metadata={"is_speaking": "false", "volume_db": "-60.0"},
    )

    try:
        # Host begins at t=0.0
        clock["t"] = 0.0
        module._analyze_sync([host_src, guest_silent, wide, host_zoom, guest_zoom])

        # Host confirmed at t=0.15s
        clock["t"] = 0.15
        decision = module._analyze_sync([host_src, guest_silent, wide, host_zoom, guest_zoom])
        assert decision is not None and decision.scene_id == "s1_host"

        # At t=1.2s (> min_hold_time=1.0s), guest starts arguing at similar volume (-19.0 dB vs -18.0 dB)
        clock["t"] = 1.2
        guest_src = Source(
            scene_id=guest.scene_id, url=guest.url, label=guest.label,
            metadata={"is_speaking": "true", "volume_db": "-19.0"},
        )
        module._analyze_sync([host_src, guest_src, wide, host_zoom, guest_zoom])

        # At t=1.35s, guest speech is confirmed, but debate has only lasted 150ms (< debate_confirm_ms=500ms)
        clock["t"] = 1.35
        decision = module._analyze_sync([host_src, guest_src, wide, host_zoom, guest_zoom])
        assert decision is None  # still held on host

        # Debate started being confirmed at t=1.35s.
        # After sustained debate (> 500ms of simultaneous talk, t=1.9s), switch to wide!
        clock["t"] = 1.9
        decision = module._analyze_sync([host_src, guest_src, wide, host_zoom, guest_zoom])
        assert decision is not None
        assert decision.scene_id == "s3_wide"
    finally:
        for reader in module._audio_readers.values():
            reader.stop()


def test_natural_pause_does_not_cut_to_wide_too_soon(monkeypatch, base_sources):
    """A natural conversational pause (e.g. 2-3 seconds) should NOT trigger a harsh cut
    to the wide shot. The wide shot should only trigger after a generous silence (e.g. 5s).
    """
    clock = {"t": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])

    module = PodcastModule()
    module._min_hold_time = 1.0
    module._min_speech_confirm_ms = 100
    module._silence_hold_time = 5.0

    host, guest, wide, host_zoom, guest_zoom = base_sources

    # Host speaks
    host_src = Source(
        scene_id=host.scene_id, url=host.url, label=host.label,
        metadata={"is_speaking": "true", "volume_db": "-18.0"},
    )
    guest_src = Source(
        scene_id=guest.scene_id, url=guest.url, label=guest.label,
        metadata={"is_speaking": "false", "volume_db": "-60.0"},
    )

    try:
        clock["t"] = 0.0
        module._analyze_sync([host_src, guest_src, wide, host_zoom, guest_zoom])

        clock["t"] = 0.15
        decision = module._analyze_sync([host_src, guest_src, wide, host_zoom, guest_zoom])
        assert decision is not None and decision.scene_id == "s1_host"

        # Host takes a thoughtful pause starting at t=2.0s
        host_silent = Source(
            scene_id=host.scene_id, url=host.url, label=host.label,
            metadata={"is_speaking": "false", "volume_db": "-55.0"},
        )
        clock["t"] = 2.0
        module._analyze_sync([host_silent, guest_src, wide, host_zoom, guest_zoom])

        # At t=4.5s (2.5s into the silence), must NOT cut to wide
        clock["t"] = 4.5
        decision = module._analyze_sync([host_silent, guest_src, wide, host_zoom, guest_zoom])
        assert decision is None

        # At t=7.2s (> 5.0s of silence since t=2.0s), it cuts to wide cleanly
        clock["t"] = 7.2
        decision = module._analyze_sync([host_silent, guest_src, wide, host_zoom, guest_zoom])
        assert decision is not None
        assert decision.scene_id == "s3_wide"
    finally:
        for reader in module._audio_readers.values():
            reader.stop()


def test_fast_turn_taking_responsive_switch(monkeypatch, base_sources):
    """When a new speaker legitimately starts talking (sustained speech, dominant volume),
    the switch should happen immediately once min_hold_time is satisfied, with zero lag.
    """
    clock = {"t": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])

    module = PodcastModule()
    module._min_hold_time = 1.5
    module._min_speech_confirm_ms = 200

    host, guest, wide, host_zoom, guest_zoom = base_sources

    try:
        # Host speaks
        host_src = Source(
            scene_id=host.scene_id, url=host.url, label=host.label,
            metadata={"is_speaking": "true", "volume_db": "-16.0"},
        )
        guest_silent = Source(
            scene_id=guest.scene_id, url=guest.url, label=guest.label,
            metadata={"is_speaking": "false", "volume_db": "-60.0"},
        )
        clock["t"] = 0.0
        module._analyze_sync([host_src, guest_silent, wide, host_zoom, guest_zoom])

        clock["t"] = 0.25
        decision = module._analyze_sync([host_src, guest_silent, wide, host_zoom, guest_zoom])
        assert decision is not None and decision.scene_id == "s1_host"

        # At t=2.0s (> min_hold_time=1.5s), Guest begins speaking clearly
        clock["t"] = 2.0
        host_silent = Source(
            scene_id=host.scene_id, url=host.url, label=host.label,
            metadata={"is_speaking": "false", "volume_db": "-55.0"},
        )
        guest_src = Source(
            scene_id=guest.scene_id, url=guest.url, label=guest.label,
            metadata={"is_speaking": "true", "volume_db": "-17.0"},
        )
        # First tick of guest speech
        module._analyze_sync([host_silent, guest_src, wide, host_zoom, guest_zoom])

        # Not yet confirmed at 2.10s (only 100ms < 200ms)
        clock["t"] = 2.10
        decision = module._analyze_sync([host_silent, guest_src, wide, host_zoom, guest_zoom])
        assert decision is None

        # At 2.25s (250ms of speech > 200ms confirm threshold), switch immediately to guest!
        clock["t"] = 2.25
        decision = module._analyze_sync([host_silent, guest_src, wide, host_zoom, guest_zoom])
        assert decision is not None
        assert decision.scene_id == "s2_guest"
    finally:
        for reader in module._audio_readers.values():
            reader.stop()
