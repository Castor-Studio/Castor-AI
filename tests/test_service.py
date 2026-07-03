from __future__ import annotations

import logging
from collections.abc import AsyncIterator

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


async def _stream(*messages) -> AsyncIterator:
    for message in messages:
        yield message


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

    messages = _stream(
        ia_analysis_pb2.ClientMessage(
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
    )

    events = [event async for event in service.AnalysisStream(messages, None)]

    assert len(events) == 1
    assert events[0].event_type == ia_analysis_pb2.SERVER_EVENT_SWITCH_SUGGESTED
    assert events[0].switch_suggestion.scene_id == "scene-1"


@pytest.mark.asyncio
async def test_stream_sources_logs_flow_and_masks_source_values(caplog):
    service = IaAnalysisService(ModuleRegistry({"fixed": FixedModule}))
    start = await service.StartSession(
        ia_analysis_pb2.StartSessionRequest(module_name="fixed"),
        None,
    )

    messages = _stream(
        ia_analysis_pb2.ClientMessage(
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
    )

    with caplog.at_level(logging.INFO, logger="castostudio_ai_server.service"):
        events = [event async for event in service.AnalysisStream(messages, None)]

    logs = caplog.text
    assert len(events) == 1
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
