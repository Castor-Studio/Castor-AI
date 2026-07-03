from __future__ import annotations

from collections.abc import Sequence

from castostudio_ai_core import AiModule, SceneDecision, SessionContext, Source


class PodcastModule(AiModule):
    async def start(self, context: SessionContext) -> None:
        return None

    async def analyze_sources(self, sources: Sequence[Source]) -> SceneDecision | None:
        if not sources:
            return None

        speaker_source = next(
            (source for source in sources if source.metadata.get("active_speaker") == "true"),
            sources[0],
        )
        return SceneDecision(scene_id=speaker_source.scene_id, confidence=0.7)

    async def stop(self) -> None:
        return None
