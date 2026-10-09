from __future__ import annotations

import time
import pytest
from castostudio_ai_core import Source, SessionContext
from castostudio_ai_podcast import PodcastModule


# ---------------------------------------------------------------------------
# SCENARIO-DRIVEN TESTS (SDD / Specification-Driven Development)
# Validates complete realistic multi-step broadcast scenarios end-to-end
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_audio_reader_network(monkeypatch):
    """Disable background socket network connections during pure scenario tests."""
    monkeypatch.setattr("castostudio_ai_podcast.audio.AudioStreamReader.start", lambda self: None)


@pytest.mark.asyncio
async def test_sdd_scenario_1_live_presidential_debate(monkeypatch):
    """Scenario 1: High-stakes broadcast debate with monologue, interruption, and anti-ping-pong wide recovery."""
    clock = {"t": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])

    ctx = SessionContext(
        session_id="sdd_debate_session",
        module_name="podcast",
        config={
            "min_hold_time": "1.0",
            "monologue_time": "4.0",
            "min_speech_confirm_ms": "50",
            "debate_confirm_ms": "100",
            "ping_pong_threshold": "4",
            "ping_pong_window_sec": "6.0",
            "cross_gating_threshold_db": "5.0",
        },
    )
    module = PodcastModule()
    await module.start(ctx)

    host_cam = Source(scene_id="s1_host", url="rtmp://cam1", label="Cam Hote")
    host_zoom = Source(scene_id="s1_host_zoom", url="rtmp://cam1", label="Zoom Hote")
    guest_cam = Source(scene_id="s2_guest", url="rtmp://cam2", label="Cam Invite")
    wide_cam = Source(scene_id="s3_wide", url="rtmp://cam3", label="Plan Large")

    # --- Phase A: Host opens the debate and speaks continuously for 5 seconds ---
    clock["t"] = 0.0
    sources_host_talk = [
        Source(scene_id="s1_host", url="rtmp://cam1", label="Cam Hote", metadata={"is_speaking": "true", "volume_db": "-14.0", "person_detected": "true"}),
        host_zoom,
        Source(scene_id="s2_guest", url="rtmp://cam2", label="Cam Invite", metadata={"is_speaking": "false", "volume_db": "-55.0", "person_detected": "true"}),
        wide_cam,
    ]
    await module.analyze_sources(sources_host_talk)

    # After 100ms (> 50ms min_speech_confirm_ms), host switch is confirmed
    clock["t"] = 0.1
    decision = await module.analyze_sources(sources_host_talk)
    assert decision is not None and decision.scene_id == "s1_host"
    assert module._current_scene_id == "s1_host"

    # Advance to t=4.5s (monologue duration threshold reached)
    clock["t"] = 4.5
    decision = await module.analyze_sources(sources_host_talk)
    assert decision is not None and decision.scene_id == "s1_host_zoom", "Expected dynamic monologue zoom after 4.5s of uninterrupted speech"

    # --- Phase B: Candidate responds (Guest takes the floor) ---
    clock["t"] = 6.0  # past min_hold_time
    sources_guest_talk = [
        Source(scene_id="s1_host", url="rtmp://cam1", label="Cam Hote", metadata={"is_speaking": "false", "volume_db": "-50.0", "person_detected": "true"}),
        host_zoom,
        Source(scene_id="s2_guest", url="rtmp://cam2", label="Cam Invite", metadata={"is_speaking": "true", "volume_db": "-15.0", "person_detected": "true"}),
        wide_cam,
    ]
    await module.analyze_sources(sources_guest_talk)
    clock["t"] = 6.1  # > min_speech_confirm_ms
    decision = await module.analyze_sources(sources_guest_talk)
    assert decision is not None and decision.scene_id == "s2_guest", "Expected clean turn-take cut to guest"

    # --- Phase C: Heated interruption / rapid turn-taking (Ping-Pong) ---
    # Switch back to host at t=7.3s
    clock["t"] = 7.3
    sources_host_quick = [
        Source(scene_id="s1_host", url="rtmp://cam1", label="Cam Hote", metadata={"is_speaking": "true", "volume_db": "-14.0", "person_detected": "true"}),
        host_zoom,
        Source(scene_id="s2_guest", url="rtmp://cam2", label="Cam Invite", metadata={"is_speaking": "false", "volume_db": "-55.0", "person_detected": "true"}),
        wide_cam,
    ]
    await module.analyze_sources(sources_host_quick)
    clock["t"] = 7.4
    d1 = await module.analyze_sources(sources_host_quick)
    assert d1 is not None and d1.scene_id == "s1_host"

    # Switch back to guest at t=8.5s
    clock["t"] = 8.5
    await module.analyze_sources(sources_guest_talk)
    clock["t"] = 8.6
    d2 = await module.analyze_sources(sources_guest_talk)
    assert d2 is not None and d2.scene_id == "s2_guest"

    # Host interrupts again at t=9.7s -> Hits ping-pong threshold (3 switches within 6s window)
    clock["t"] = 9.7
    await module.analyze_sources(sources_host_quick)
    clock["t"] = 9.8
    d3 = await module.analyze_sources(sources_host_quick)
    # Ping-Pong governor must step in and stabilize to wide shot!
    assert d3 is not None and d3.scene_id == "s3_wide", "Anti-ping-pong governor must cut to wide shot during rapid cross-fire!"


@pytest.mark.asyncio
async def test_sdd_scenario_2_empty_chair_and_noise_accident(monkeypatch):
    """Scenario 2: Guest leaves set mid-show, mic knock occurs, AI visual presence vetoes switch."""
    clock = {"t": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])

    ctx = SessionContext(
        session_id="sdd_empty_chair_session",
        module_name="podcast",
        config={
            "min_hold_time": "1.0",
            "min_speech_confirm_ms": "50",
            "silence_hold_time": "3.0",
        },
    )
    module = PodcastModule()
    await module.start(ctx)

    # 1. Show starts with Host speaking
    clock["t"] = 0.0
    host_active = [
        Source(scene_id="s1_host", url="rtmp://cam1", label="Cam Hote", metadata={"is_speaking": "true", "volume_db": "-15.0", "person_detected": "true"}),
        Source(scene_id="s2_guest", url="rtmp://cam2", label="Cam Invite", metadata={"is_speaking": "false", "volume_db": "-60.0", "person_detected": "true"}),
        Source(scene_id="s3_wide", url="rtmp://cam3", label="Plan Large"),
    ]
    await module.analyze_sources(host_active)
    clock["t"] = 0.1
    d = await module.analyze_sources(host_active)
    assert d is not None and d.scene_id == "s1_host"
    assert module._current_scene_id == "s1_host"

    # 2. Guest leaves the studio chair (chair confirmed empty by vision tracker)
    # Host is silent for a moment at t=1.5s
    # A stagehand accidentally knocks the guest mic or drops a prop (-12 dB, is_speaking="true")
    clock["t"] = 1.5
    guest_mic_spike_empty_chair = [
        Source(scene_id="s1_host", url="rtmp://cam1", label="Cam Hote", metadata={"is_speaking": "false", "volume_db": "-55.0", "person_detected": "true"}),
        Source(scene_id="s2_guest", url="rtmp://cam2", label="Cam Invite", metadata={"is_speaking": "true", "volume_db": "-12.0", "person_detected": "false", "face_detected": "false"}),
        Source(scene_id="s3_wide", url="rtmp://cam3", label="Plan Large"),
    ]
    await module.analyze_sources(guest_mic_spike_empty_chair)
    clock["t"] = 1.6
    d_mishap = await module.analyze_sources(guest_mic_spike_empty_chair)
    
    # Must NOT switch to s2_guest because the chair is empty!
    assert module._current_scene_id != "s2_guest", "Director must never cut to an empty chair despite loud audio spikes!"


@pytest.mark.asyncio
async def test_sdd_scenario_3_natural_breathing_hesitations_and_recovery(monkeypatch):
    """Scenario 3: Normal speaking pauses are preserved without stutter, and long silence safely resets to wide."""
    clock = {"t": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])

    ctx = SessionContext(
        session_id="sdd_pacing_session",
        module_name="podcast",
        config={
            "min_hold_time": "1.0",
            "min_speech_confirm_ms": "50",
            "silence_hold_time": "3.0",
        },
    )
    module = PodcastModule()
    await module.start(ctx)

    host_speaking = Source(scene_id="s1_host", url="rtmp://cam1", label="Cam Hote", metadata={"is_speaking": "true", "volume_db": "-14.0"})
    guest_silent = Source(scene_id="s2_guest", url="rtmp://cam2", label="Cam Invite", metadata={"is_speaking": "false", "volume_db": "-60.0"})
    wide = Source(scene_id="s3_wide", url="rtmp://cam3", label="Plan Large")

    # Step 1: Host starts talking at t=0.0
    clock["t"] = 0.0
    await module.analyze_sources([host_speaking, guest_silent, wide])
    clock["t"] = 0.1
    d_init = await module.analyze_sources([host_speaking, guest_silent, wide])
    assert d_init is not None and d_init.scene_id == "s1_host"
    assert module._current_scene_id == "s1_host"

    # Step 2: Host takes a 1.2s breathing hesitation (silence)
    host_breathing = Source(scene_id="s1_host", url="rtmp://cam1", label="Cam Hote", metadata={"is_speaking": "false", "volume_db": "-52.0"})
    clock["t"] = 1.5
    d_pause = await module.analyze_sources([host_breathing, guest_silent, wide])
    # Camera must remain held on host! No premature cut to wide.
    assert d_pause is None
    assert module._current_scene_id == "s1_host"

    # Step 3: Host continues talking at t=2.0
    clock["t"] = 2.0
    await module.analyze_sources([host_speaking, guest_silent, wide])
    assert module._current_scene_id == "s1_host"

    # Step 4: Full extended studio silence for 4.5 seconds (both parties done talking)
    clock["t"] = 7.0  # elapsed silence > silence_hold_time (3.0s)
    d_reset = await module.analyze_sources([host_breathing, guest_silent, wide])
    assert d_reset is not None and d_reset.scene_id == "s3_wide", "Extended silence must smoothly reset framing to wide studio shot"


@pytest.mark.asyncio
async def test_sdd_scenario_4_multi_panel_roundtable_scalability(monkeypatch):
    """Scenario 4: 5-participant roundtable scales smoothly with correct dynamic speaker assignment."""
    clock = {"t": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])

    ctx = SessionContext(
        session_id="sdd_roundtable_session",
        module_name="podcast",
        config={
            "min_hold_time": "0.5",
            "min_speech_confirm_ms": "50",
        },
    )
    module = PodcastModule()
    await module.start(ctx)

    sources = [
        Source(scene_id="cam_h", url="rtmp://1", label="Présentateur"),
        Source(scene_id="cam_p1", url="rtmp://2", label="Intervenant 1"),
        Source(scene_id="cam_p2", url="rtmp://3", label="Intervenant 2"),
        Source(scene_id="cam_p3", url="rtmp://4", label="Intervenant 3"),
        Source(scene_id="cam_w", url="rtmp://5", label="Studio Large"),
    ]

    roles = module._get_roles(sources)
    assert roles["host"] == "cam_h"
    assert roles["guest"] == "cam_p1"
    assert roles["guest2"] == "cam_p2"
    assert roles["guest3"] == "cam_p3"
    assert roles["wide"] == "cam_w"

    # Panelist 3 answers question
    clock["t"] = 1.0
    sources_p3_talk = [
        Source(scene_id="cam_h", url="rtmp://1", label="Présentateur", metadata={"is_speaking": "false", "volume_db": "-55.0"}),
        Source(scene_id="cam_p1", url="rtmp://2", label="Intervenant 1", metadata={"is_speaking": "false", "volume_db": "-55.0"}),
        Source(scene_id="cam_p2", url="rtmp://3", label="Intervenant 2", metadata={"is_speaking": "false", "volume_db": "-55.0"}),
        Source(scene_id="cam_p3", url="rtmp://4", label="Intervenant 3", metadata={"is_speaking": "true", "volume_db": "-14.0"}),
        Source(scene_id="cam_w", url="rtmp://5", label="Studio Large"),
    ]

    await module.analyze_sources(sources_p3_talk)
    clock["t"] = 1.1  # > 50ms confirm
    d = await module.analyze_sources(sources_p3_talk)
    assert d is not None and d.scene_id == "cam_p3", "Roundtable must seamlessly switch to 3rd panelist"
