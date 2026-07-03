from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Source:
    scene_id: str
    url: str
    label: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SceneDecision:
    scene_id: str
    confidence: float


@dataclass(frozen=True, slots=True)
class SessionContext:
    session_id: str
    module_name: str
    config: dict[str, str] = field(default_factory=dict)
