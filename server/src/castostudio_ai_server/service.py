from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

from castostudio_ai_core import (
    AiModule,
    ModuleLoadError,
    ModuleNotFoundError,
    ModuleRegistry,
    SessionContext,
    Source,
)

from .proto import ia_analysis_pb2, ia_analysis_pb2_grpc

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _Session:
    context: SessionContext
    module: AiModule
    sources: list[Source] | None = None
    client_ip: str = "127.0.0.1"


class IaAnalysisService(ia_analysis_pb2_grpc.IaAnalysisServiceServicer):
    def __init__(self, registry: ModuleRegistry | None = None) -> None:
        self._registry = registry or ModuleRegistry()
        self._sessions: dict[str, _Session] = {}

    async def StartSession(self, request, context):
        module_name = request.module_name.strip()
        config = dict(request.module_config)
        LOGGER.info(
            "StartSession received module_name=%s module_config_keys=%d",
            module_name or "<empty>",
            len(config),
        )
        LOGGER.debug(
            "StartSession module_config key_names=%s",
            sorted(config),
        )
        if not module_name:
            LOGGER.warning("StartSession rejected error_code=INVALID_ARGUMENT")
            return ia_analysis_pb2.StartSessionResponse(
                success=False,
                message="module_name is required.",
                error_code="INVALID_ARGUMENT",
            )

        try:
            module = self._registry.create(module_name)
            session_id = str(uuid.uuid4())
            session_context = SessionContext(
                session_id=session_id,
                module_name=module_name,
                config=config,
            )
            await module.start(session_context)
        except ModuleNotFoundError as exc:
            LOGGER.warning(
                "StartSession failed module_name=%s error_code=MODULE_NOT_FOUND",
                module_name,
            )
            return ia_analysis_pb2.StartSessionResponse(
                success=False,
                message=str(exc),
                error_code="MODULE_NOT_FOUND",
            )
        except ModuleLoadError as exc:
            LOGGER.exception(
                "StartSession failed module_name=%s error_code=MODULE_LOAD_FAILED",
                module_name,
            )
            return ia_analysis_pb2.StartSessionResponse(
                success=False,
                message=str(exc),
                error_code="MODULE_LOAD_FAILED",
            )
        except Exception as exc:
            LOGGER.exception(
                "StartSession failed module_name=%s error_code=MODULE_START_FAILED",
                module_name,
            )
            return ia_analysis_pb2.StartSessionResponse(
                success=False,
                message=f"Module '{module_name}' failed to start: {exc}",
                error_code="MODULE_START_FAILED",
            )

        self._sessions[session_id] = _Session(session_context, module)
        LOGGER.info(
            "StartSession succeeded session_id=%s module_name=%s",
            session_id,
            module_name,
        )
        return ia_analysis_pb2.StartSessionResponse(
            success=True,
            session_id=session_id,
            message="Session started.",
        )

    async def AnalysisStream(self, request_iterator, context) -> AsyncIterator:
        event_queue = asyncio.Queue()
        session_id = None

        peer = context.peer() if context is not None else None
        client_ip = "127.0.0.1"
        if peer and peer.startswith("ipv4:"):
            parts = peer.split(":")
            if len(parts) >= 2:
                client_ip = parts[1]
        elif peer and peer.startswith("dns:"):
            parts = peer.split(":")
            if len(parts) >= 2:
                client_ip = parts[1]

        async def read_client_messages():
            nonlocal session_id
            try:
                async for request in request_iterator:
                    session_id = request.session_id
                    payload = request.WhichOneof("payload")
                    LOGGER.info(
                        "AnalysisStream received session_id=%s payload=%s client_ip=%s",
                        session_id or "<empty>",
                        payload or "<empty>",
                        client_ip,
                    )
                    session = self._sessions.get(session_id)
                    if session is None:
                        event = self._error_event(
                            session_id,
                            "SESSION_NOT_FOUND",
                            "Session is unknown or already closed.",
                            is_fatal=True,
                        )
                        await event_queue.put(event)
                        await event_queue.put(None)
                        return

                    session.client_ip = client_ip

                    if payload == "sources":
                        session.sources = self._convert_sources(session, request.sources.sources)
                        event = self._status_event(
                            session.context.session_id,
                            ia_analysis_pb2.SESSION_STATE_READY,
                            "Sources received. Continuous analysis started.",
                        )
                        self._log_server_event(event)
                    elif payload == "keep_alive":
                        event = self._status_event(
                            session.context.session_id,
                            ia_analysis_pb2.SESSION_STATE_READY,
                            "Session alive.",
                        )
                        self._log_server_event(event)
                    elif payload == "stop":
                        LOGGER.info(
                            "AnalysisStream stop requested session_id=%s reason=%s",
                            session.context.session_id,
                            request.stop.reason or "<empty>",
                        )
                        await self._stop_session(session.context.session_id)
                        event = self._status_event(
                            session.context.session_id,
                            ia_analysis_pb2.SESSION_STATE_STOPPED,
                            request.stop.reason or "Session stopped.",
                        )
                        await event_queue.put(event)
                        break
                    else:
                        event = self._error_event(
                            session_id,
                            "INVALID_ARGUMENT",
                            "ClientMessage payload is required.",
                            is_fatal=False,
                        )
                        await event_queue.put(event)
            except asyncio.CancelledError:
                LOGGER.info("Client message reader cancelled for session %s", session_id)
            except Exception as exc:
                LOGGER.exception("Error in client message reader for session %s", session_id)
                event = self._error_event(
                    session_id or "",
                    "SERVER_ERROR",
                    f"Internal server error: {exc}",
                    is_fatal=True,
                )
                await event_queue.put(event)
                await event_queue.put(None)

        reader_task = asyncio.create_task(read_client_messages())

        async def run_analysis():
            nonlocal session_id
            while session_id is None:
                await asyncio.sleep(0.1)
                if reader_task.done():
                    return

            try:
                # Wait for sources to be populated
                while session_id in self._sessions:
                    session = self._sessions.get(session_id)
                    if session is None:
                        break
                    if session.sources is not None:
                        break
                    await asyncio.sleep(0.1)

                while session_id in self._sessions:
                    session = self._sessions.get(session_id)
                    if session is None:
                        break

                    if session.sources is not None:
                        try:
                            decision = await session.module.analyze_sources(session.sources)
                            if decision is not None:
                                LOGGER.info(
                                    "Analysis completed session_id=%s decision_scene_id=%s confidence=%.3f",
                                    session_id,
                                    decision.scene_id,
                                    decision.confidence,
                                )
                                event = ia_analysis_pb2.ServerEvent(
                                    session_id=session_id,
                                    timestamp_ms=self._now_ms(),
                                    event_type=ia_analysis_pb2.SERVER_EVENT_SWITCH_SUGGESTED,
                                    switch_suggestion=ia_analysis_pb2.SceneSwitch(
                                        scene_id=decision.scene_id,
                                        confidence=decision.confidence,
                                    ),
                                )
                                await event_queue.put(event)
                        except Exception as exc:
                            LOGGER.exception(
                                "Analysis failed session_id=%s error_code=ANALYSIS_FAILED",
                                session_id,
                            )
                            event = self._error_event(
                                session_id,
                                "ANALYSIS_FAILED",
                                f"AI module failed during analysis: {exc}",
                                is_fatal=False,
                            )
                            await event_queue.put(event)
                            return
                    await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                LOGGER.info("Analysis loop cancelled for session %s", session_id)
            except Exception as exc:
                LOGGER.exception("Error in analysis loop for session %s", session_id)
            finally:
                await event_queue.put(None)

        analysis_task = asyncio.create_task(run_analysis())

        try:
            while True:
                event = await event_queue.get()
                if event is None:
                    break
                self._log_server_event(event)
                yield event
        finally:
            reader_task.cancel()
            analysis_task.cancel()
            await asyncio.gather(reader_task, analysis_task, return_exceptions=True)
            if session_id and session_id in self._sessions:
                LOGGER.info("Connection closed, stopping session %s", session_id)
                await self._stop_session(session_id)

    async def EndSession(self, request, context):
        LOGGER.info("EndSession received session_id=%s", request.session_id or "<empty>")
        if not request.session_id:
            LOGGER.warning("EndSession rejected error=session_id_required")
            return ia_analysis_pb2.EndSessionResponse(
                success=False,
                message="session_id is required.",
            )

        session = self._sessions.get(request.session_id)
        if session is None:
            LOGGER.warning("EndSession failed session_id=%s reason=unknown", request.session_id)
            return ia_analysis_pb2.EndSessionResponse(
                success=False,
                message="Session is unknown or already closed.",
            )

        await self._stop_session(request.session_id)
        LOGGER.info("EndSession succeeded session_id=%s", request.session_id)
        return ia_analysis_pb2.EndSessionResponse(
            success=True,
            message="Session ended.",
        )

    async def _convert_sources(self, session: _Session, source_messages) -> list[Source]:
        source_messages = tuple(source_messages)

        LOGGER.info(
            "SourceList received session_id=%s source_count=%d scene_ids=%s metadata_counts=%s",
            session.context.session_id,
            len(source_messages),
            [source.scene_id for source in source_messages],
            [len(source.metadata) for source in source_messages],
        )

        converted = []
        for source in source_messages:
            url = source.url
            if session.client_ip != "127.0.0.1" and ("127.0.0.1" in url or "localhost" in url):
                url = url.replace("127.0.0.1", session.client_ip).replace("localhost", session.client_ip)
                LOGGER.info("Translated source URL for remote client: %s -> %s", source.url, url)

            converted.append(
                Source(
                    scene_id=source.scene_id,
                    url=url,
                    label=source.label,
                    metadata=dict(source.metadata),
                )
            )

            yield ia_analysis_pb2.ServerEvent(
                session_id=session.context.session_id,
                timestamp_ms=self._now_ms(),
                event_type=ia_analysis_pb2.SERVER_EVENT_SWITCH_SUGGESTED,
                switch_suggestion=ia_analysis_pb2.SceneSwitch(
                    scene_id=decision.scene_id,
                    confidence=decision.confidence,
                ),
            )

        await asyncio.sleep(0.1)

    async def _stop_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            await session.module.stop()
            LOGGER.info("Session stopped session_id=%s", session_id)

    @staticmethod
    def _status_event(session_id: str, state: int, message: str):
        return ia_analysis_pb2.ServerEvent(
            session_id=session_id,
            timestamp_ms=IaAnalysisService._now_ms(),
            event_type=ia_analysis_pb2.SERVER_EVENT_STATUS_CHANGED,
            status=ia_analysis_pb2.SessionStatus(state=state, message=message),
        )

    @staticmethod
    def _error_event(session_id: str, code: str, message: str, is_fatal: bool):
        return ia_analysis_pb2.ServerEvent(
            session_id=session_id,
            timestamp_ms=IaAnalysisService._now_ms(),
            event_type=ia_analysis_pb2.SERVER_EVENT_ERROR,
            error=ia_analysis_pb2.ServerError(
                error_code=code,
                error_message=message,
                is_fatal=is_fatal,
            ),
        )

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _log_server_event(event) -> None:
        payload = event.WhichOneof("payload")
        if payload == "switch_suggestion":
            LOGGER.info(
                "ServerEvent sent session_id=%s event_type=SWITCH_SUGGESTED scene_id=%s confidence=%.3f",
                event.session_id,
                event.switch_suggestion.scene_id,
                event.switch_suggestion.confidence,
            )
        elif payload == "status":
            LOGGER.info(
                "ServerEvent sent session_id=%s event_type=STATUS_CHANGED state=%s message=%s",
                event.session_id,
                event.status.state,
                event.status.message,
            )
        elif payload == "error":
            LOGGER.warning(
                "ServerEvent sent session_id=%s event_type=ERROR error_code=%s fatal=%s",
                event.session_id,
                event.error.error_code,
                event.error.is_fatal,
            )
        else:
            LOGGER.warning(
                "ServerEvent sent session_id=%s event_type=%s payload=<empty>",
                event.session_id,
                event.event_type,
            )
