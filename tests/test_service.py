from __future__ import annotations

import asyncio
import logging

import pytest

from castostudio_ai_core import AiModule, ModuleRegistry, SceneDecision
from castostudio_ai_server.proto import ia_analysis_pb2
from castostudio_ai_server.service import IaAnalysisService


class FixedModule(AiModule):
    def __init__(self) -> None:
        self.stopped = False

    async def start(self, context):
        self.context = context

    async def analyze_sources(self, sources):
        if not sources:
            return None
        return SceneDecision(scene_id=sources[0].scene_id, confidence=0.9)

    async def stop(self):
        self.stopped = True


@pytest.mark.asyncio
async def test_start_session_missing_module():
    service = IaAnalysisService(ModuleRegistry())

    response = await service.StartSession(
        ia_analysis_pb2.StartSessionRequest(module_name="missing"),
        None,
    )

    assert response.success is False
    assert response.error_code == "MODULE_NOT_FOUND"


@pytest.mark.asyncio
async def test_start_session_logs_success_and_masks_config(caplog):
    service = IaAnalysisService(ModuleRegistry({"fixed": FixedModule}))

    with caplog.at_level(logging.INFO, logger="castostudio_ai_server.service"):
        response = await service.StartSession(
            ia_analysis_pb2.StartSessionRequest(
                module_name="fixed",
                module_config={"api_key": "secret-token"},
            ),
            None,
        )

    logs = caplog.text
    assert response.success is True
    assert "StartSession received module_name=fixed module_config_keys=1" in logs
    assert f"StartSession succeeded session_id={response.session_id} module_name=fixed" in logs
    assert "secret-token" not in logs


@pytest.mark.asyncio
async def test_start_session_logs_missing_module(caplog):
    service = IaAnalysisService(ModuleRegistry())

    with caplog.at_level(logging.INFO, logger="castostudio_ai_server.service"):
        response = await service.StartSession(
            ia_analysis_pb2.StartSessionRequest(module_name="missing"),
            None,
        )

    assert response.success is False
    assert response.error_code == "MODULE_NOT_FOUND"
    assert "error_code=MODULE_NOT_FOUND" in caplog.text


@pytest.mark.asyncio
async def test_stream_sources_returns_scene_switch():
    service = IaAnalysisService(ModuleRegistry({"fixed": FixedModule}))
    start = await service.StartSession(
        ia_analysis_pb2.StartSessionRequest(module_name="fixed"),
        None,
    )

    # FixedModule always returns a decision, so the analysis loop runs
    # forever until told to stop - a StopSignal is required to end the
    # stream, exactly like a real client would send one when done. The
    # delay before the stop message gives the concurrent analysis loop a
    # chance to run at least once, proving the server keeps reading the
    # request stream (and can act on a later message) while analysis is
    # in flight - the exact case that used to be blocked forever.
    async def _messages():
        yield ia_analysis_pb2.ClientMessage(
            session_id=start.session_id,
            sources=ia_analysis_pb2.SourceList(
                sources=[
                    ia_analysis_pb2.Source(
                        scene_id="scene-1",
                        url="rtmp://example/source",
                    )
                ]
            ),
        )
        await asyncio.sleep(0.05)
        yield ia_analysis_pb2.ClientMessage(
            session_id=start.session_id,
            stop=ia_analysis_pb2.StopSignal(reason="test done"),
        )

    events = [event async for event in service.AnalysisStream(_messages(), None)]

    switch_events = [
        e for e in events if e.event_type == ia_analysis_pb2.SERVER_EVENT_SWITCH_SUGGESTED
    ]
    assert switch_events, "expected at least one scene switch before the stop was processed"
    assert switch_events[0].switch_suggestion.scene_id == "scene-1"
    assert events[-1].event_type == ia_analysis_pb2.SERVER_EVENT_STATUS_CHANGED
    assert events[-1].status.state == ia_analysis_pb2.SESSION_STATE_STOPPED


@pytest.mark.asyncio
async def test_stream_sources_logs_flow_and_masks_source_values(caplog):
    service = IaAnalysisService(ModuleRegistry({"fixed": FixedModule}))
    start = await service.StartSession(
        ia_analysis_pb2.StartSessionRequest(module_name="fixed"),
        None,
    )

    async def _messages():
        yield ia_analysis_pb2.ClientMessage(
            session_id=start.session_id,
            sources=ia_analysis_pb2.SourceList(
                sources=[
                    ia_analysis_pb2.Source(
                        scene_id="scene-1",
                        url="rtmp://example/private-source",
                        label="Camera 1",
                        metadata={"token": "stream-secret"},
                    )
                ]
            ),
        )
        await asyncio.sleep(0.05)
        yield ia_analysis_pb2.ClientMessage(
            session_id=start.session_id,
            stop=ia_analysis_pb2.StopSignal(reason="test done"),
        )

    with caplog.at_level(logging.INFO, logger="castostudio_ai_server.service"):
        events = [event async for event in service.AnalysisStream(_messages(), None)]

    logs = caplog.text
    switch_events = [
        e for e in events if e.event_type == ia_analysis_pb2.SERVER_EVENT_SWITCH_SUGGESTED
    ]
    assert switch_events
    assert "AnalysisStream received" in logs
    assert "payload=sources" in logs
    assert "source_count=1" in logs
    assert "scene-1" in logs
    assert "decision_scene_id=scene-1" in logs
    assert "ServerEvent sent" in logs
    assert "SWITCH_SUGGESTED" in logs
    assert "rtmp://example/private-source" not in logs
    assert "stream-secret" not in logs


@pytest.mark.asyncio
async def test_end_session_cleans_session():
    service = IaAnalysisService(ModuleRegistry({"fixed": FixedModule}))
    start = await service.StartSession(
        ia_analysis_pb2.StartSessionRequest(module_name="fixed"),
        None,
    )

    end = await service.EndSession(
        ia_analysis_pb2.EndSessionRequest(session_id=start.session_id),
        None,
    )
    second_end = await service.EndSession(
        ia_analysis_pb2.EndSessionRequest(session_id=start.session_id),
        None,
    )

    assert end.success is True
    assert second_end.success is False


@pytest.mark.asyncio
async def test_end_session_logs_success_and_unknown_session(caplog):
    service = IaAnalysisService(ModuleRegistry({"fixed": FixedModule}))
    start = await service.StartSession(
        ia_analysis_pb2.StartSessionRequest(module_name="fixed"),
        None,
    )

    with caplog.at_level(logging.INFO, logger="castostudio_ai_server.service"):
        end = await service.EndSession(
            ia_analysis_pb2.EndSessionRequest(session_id=start.session_id),
            None,
        )
        second_end = await service.EndSession(
            ia_analysis_pb2.EndSessionRequest(session_id=start.session_id),
            None,
        )

    logs = caplog.text
    assert end.success is True
    assert second_end.success is False
    assert f"EndSession succeeded session_id={start.session_id}" in logs
    assert f"EndSession failed session_id={start.session_id} reason=unknown" in logs


def test_podcast_module_parse_roles():
    from castostudio_ai_podcast import PodcastModule
    from castostudio_ai_core import Source

    module = PodcastModule()
    sources = [
        Source(scene_id="s1", url="rtmp://...", label="Cam Hote"),
        Source(scene_id="s2", url="rtmp://...", label="Cam Invite"),
        Source(scene_id="s3", url="rtmp://...", label="Plan Large"),
        Source(scene_id="s4", url="rtmp://...", label="Cam Hote Zoom"),
    ]
    roles = module._parse_roles(sources)
    assert roles["host"] == "s1"
    assert roles["guest"] == "s2"
    assert roles["wide"] == "s3"
    assert roles["host_zoom"] == "s4"


def test_podcast_module_state_machine():
    from castostudio_ai_podcast import PodcastModule

    module = PodcastModule()
    roles = {
        "host": "s1",
        "guest": "s2",
        "wide": "s3",
        "host_zoom": "s4"
    }

    # 1. Silence state - initial
    decision = module._run_state_machine(active_speakers=[], roles=roles, now=100.0)
    assert decision is None

    # After 3 seconds of silence, should switch to wide
    decision = module._run_state_machine(active_speakers=[], roles=roles, now=104.0)
    assert decision == "wide"

    # 2. Single speaker (host)
    decision = module._run_state_machine(active_speakers=["host"], roles=roles, now=105.0)
    assert decision == "host"

    # 3. Monologue duration exceeded (5 seconds)
    decision = module._run_state_machine(active_speakers=["host"], roles=roles, now=111.0)
    assert decision == "host_zoom"

    # 4. Multiple speakers (debate) -> Wide shot
    decision = module._run_state_machine(active_speakers=["host", "guest"], roles=roles, now=112.0)
    assert decision == "wide"


def test_podcast_module_parse_roles_panel_with_extra_guests():
    """3+ participant panels must resolve to distinct guestN roles instead
    of every non-host camera collapsing into a single "guest"."""
    from castostudio_ai_podcast import PodcastModule
    from castostudio_ai_core import Source

    module = PodcastModule()
    sources = [
        Source(scene_id="s1", url="rtmp://...", label="Cam Hote"),
        Source(scene_id="s2", url="rtmp://...", label="Cam Invite 1"),
        Source(scene_id="s3", url="rtmp://...", label="Cam Invite 2"),
        Source(scene_id="s4", url="rtmp://...", label="Cam Invite 3"),
        Source(scene_id="s5", url="rtmp://...", label="Plan Large"),
    ]
    roles = module._parse_roles(sources)
    assert roles["host"] == "s1"
    assert roles["guest"] == "s2"
    assert roles["guest2"] == "s3"
    assert roles["guest3"] == "s4"
    assert roles["wide"] == "s5"


def test_podcast_module_parse_roles_unlabeled_extra_cameras_become_guests():
    """Cameras that don't match any known naming pattern still get a
    dedicated speaker role (guestN) instead of being silently dropped."""
    from castostudio_ai_podcast import PodcastModule
    from castostudio_ai_core import Source

    module = PodcastModule()
    sources = [
        Source(scene_id="s1", url="rtmp://...", label="Cam Hote"),
        Source(scene_id="s2", url="rtmp://...", label="Cam Invite"),
        Source(scene_id="s3", url="rtmp://...", label="Camera 4"),
        Source(scene_id="s4", url="rtmp://...", label="Camera 5"),
    ]
    roles = module._parse_roles(sources)
    assert roles["host"] == "s1"
    assert roles["guest"] == "s2"
    assert {roles.get("guest2"), roles.get("guest3")} == {"s3", "s4"}


def test_podcast_module_state_machine_three_way_panel_debate():
    """A 3-participant panel must still cut to wide when 2+ speak at once,
    generalizing the old host/guest-only debate detection."""
    from castostudio_ai_podcast import PodcastModule

    module = PodcastModule()
    roles = {"host": "s1", "guest": "s2", "guest2": "s3", "wide": "s4"}

    decision = module._run_state_machine(active_speakers=["guest2"], roles=roles, now=100.0)
    assert decision == "guest2"

    decision = module._run_state_machine(
        active_speakers=["host", "guest2"], roles=roles, now=101.0
    )
    assert decision == "wide"


def test_podcast_module_confirm_speakers_filters_short_backchannel():
    """A brief "mm-hmm"/laugh below min_speech_confirm_ms must not count as
    a real speaker turn, so it can't flip focus or trigger debate mode."""
    from castostudio_ai_podcast import PodcastModule

    module = PodcastModule()
    module._min_speech_confirm_ms = 300

    # t=0.0: host and guest both start speaking at the same instant -
    # neither has been sustained long enough yet.
    confirmed = module._confirm_speakers(["host", "guest"], now=0.0)
    assert confirmed == []

    # t=0.1: guest's blip is already over (backchannel), host keeps going -
    # guest never crosses the threshold, so it's filtered out entirely.
    confirmed = module._confirm_speakers(["host"], now=0.1)
    assert confirmed == []

    # t=0.35: host has now been continuously speaking for 350ms (>=300ms) -
    # confirmed as a real turn.
    confirmed = module._confirm_speakers(["host"], now=0.35)
    assert confirmed == ["host"]

