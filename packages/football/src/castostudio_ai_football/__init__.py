from __future__ import annotations

from collections.abc import Sequence

from castostudio_ai_core import AiModule, SceneDecision, SessionContext, Source


class FootballModule(AiModule):
    def __init__(self) -> None:
        self._default_confidence = 0.65

    async def start(self, context: SessionContext) -> None:
        confidence = context.config.get("default_confidence")
        if confidence:
            self._default_confidence = float(confidence)

    async def analyze_sources(self, sources: Sequence[Source]) -> SceneDecision | None:
        if not sources:
            return None

        priority_source = max(
            sources,
            key=lambda source: int(source.metadata.get("score_priority", "0")),
        )
        return SceneDecision(
            scene_id=priority_source.scene_id,
            confidence=self._default_confidence,
        )

    async def stop(self) -> None:
        return None
