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
        async for request in request_iterator:
            payload = request.WhichOneof("payload")
            LOGGER.info(
                "AnalysisStream received session_id=%s payload=%s",
                request.session_id or "<empty>",
                payload or "<empty>",
            )
            session = self._sessions.get(request.session_id)
            if session is None:
                event = self._error_event(
                    request.session_id,
                    "SESSION_NOT_FOUND",
                    "Session is unknown or already closed.",
                    is_fatal=True,
                )
                self._log_server_event(event)
                yield event
                continue

            if payload == "sources":
                session.sources = self._convert_sources(session, request.sources.sources)

                event = self._status_event(
                    session.context.session_id,
                    ia_analysis_pb2.SESSION_STATE_READY,
                    "Sources received. Continuous analysis started.",
                )
                self._log_server_event(event)
                yield event

                async for event in self._analysis_loop(session):
                    self._log_server_event(event)
                    yield event
            elif payload == "keep_alive":
                event = self._status_event(
                    session.context.session_id,
                    ia_analysis_pb2.SESSION_STATE_READY,
                    "Session alive.",
                )
                self._log_server_event(event)
                yield event
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
                self._log_server_event(event)
                yield event
                break
            else:
                event = self._error_event(
                    request.session_id,
                    "INVALID_ARGUMENT",
                    "ClientMessage payload is required.",
                    is_fatal=False,
                )
                self._log_server_event(event)
                yield event

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

    def _convert_sources(self, session: _Session, source_messages) -> list[Source]:
        source_messages = tuple(source_messages)

        LOGGER.info(
            "SourceList received session_id=%s source_count=%d scene_ids=%s metadata_counts=%s",
            session.context.session_id,
            len(source_messages),
            [source.scene_id for source in source_messages],
            [len(source.metadata) for source in source_messages],
        )

        return [
            Source(
                scene_id=source.scene_id,
                url=source.url,
                label=source.label,
                metadata=dict(source.metadata),
            )
            for source in source_messages
        ]


    async def _analysis_loop(self, session: _Session) -> AsyncIterator:
        if session.sources is None:
            return

        while session.context.session_id in self._sessions:
            try:
                decision = await session.module.analyze_sources(session.sources)

            except Exception as exc:
                LOGGER.exception(
                    "Analysis failed session_id=%s error_code=ANALYSIS_FAILED",
                    session.context.session_id,
                )
                yield self._error_event(
                    session.context.session_id,
                    "ANALYSIS_FAILED",
                    f"AI module failed during analysis: {exc}",
                    is_fatal=False,
                )
                return

            if decision is not None:
                LOGGER.info(
                    "Analysis completed session_id=%s decision_scene_id=%s confidence=%.3f",
                    session.context.session_id,
                    decision.scene_id,
                    decision.confidence,
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
