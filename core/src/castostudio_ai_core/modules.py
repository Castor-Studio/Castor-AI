from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from .models import SceneDecision, SessionContext, Source


class AiModule(ABC):
    """Base contract implemented by optional CastoStudio AI packages."""

    @abstractmethod
    async def start(self, context: SessionContext) -> None:
        """Initialize module state for one analysis session."""

    @abstractmethod
    async def analyze_sources(self, sources: Sequence[Source]) -> SceneDecision | None:
        """Return scene decision for current source list, or None to keep current scene."""

    @abstractmethod
    async def stop(self) -> None:
        """Release resources owned by this session module."""
