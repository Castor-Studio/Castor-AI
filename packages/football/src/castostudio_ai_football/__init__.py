from __future__ import annotations

from collections.abc import Sequence

from castostudio_ai_core import AiModule, SceneDecision, SessionContext, Source

from .analyzer import FootballAnalyzer


class FootballModule(AiModule):
    def __init__(self) -> None:
        self._analyzer: FootballAnalyzer | None = None
        self._confidence = 1.0
        self._frameskip = 0

        self._source_1_scene_id: str | None = None
        self._source_2_scene_id: str | None = None
        self._last_sent_scene_id: str | None = None
        self._last_detected_focus: str | None = None

    async def start(self, context: SessionContext) -> None:
        self._confidence = float(context.config.get("default_confidence", 1.0))
        self._frameskip = int(context.config.get("frameskip", 0))

        print("[FootballModule] Started")
        print("[FootballModule] confidence:", self._confidence)
        print("[FootballModule] frameskip:", self._frameskip)

    async def analyze_sources(self, sources: Sequence[Source]) -> SceneDecision | None:
        print("[FootballModule] SourceList received in module")
        print("[FootballModule] source_count:", len(sources))

        for index, source in enumerate(sources):
            print(f"[FootballModule] SOURCE {index + 1}")
            print("  scene_id:", source.scene_id)
            print("  url:", source.url)
            print("  label:", source.label)
            print("  metadata:", dict(source.metadata or {}))

        if len(sources) < 2:
            return None

        source_1 = sources[0]
        source_2 = sources[1]

        if self._analyzer is None:
            self._source_1_scene_id = source_1.scene_id
            self._source_2_scene_id = source_2.scene_id

            self._analyzer = FootballAnalyzer(
                stream_1_url=source_1.url,
                stream_2_url=source_2.url,
                frameskip=self._frameskip,
            )

            self._analyzer.start()
            self._analyzer.wait_until_ready(timeout_sec=5.0)

        focus = None

        for i in range(30):
            focus = self._analyzer.analyze_once()
            print(f"[FootballModule] analyze attempt {i} focus={focus}")

            if focus is not None:
                break

        if focus == "STREAM_1":
            scene_id = self._source_1_scene_id
        elif focus == "STREAM_2":
            scene_id = self._source_2_scene_id
        else:
            return None

        print("[FootballModule] SWITCH scene_id:", scene_id)

        if scene_id == self._last_sent_scene_id:
            return None

        self._last_sent_scene_id = scene_id

        return SceneDecision(
            scene_id=scene_id,
            confidence=self._confidence,
        )


    async def stop(self) -> None:
        if self._analyzer is not None:
            self._analyzer.stop()
            self._analyzer = None