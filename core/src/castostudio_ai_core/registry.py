from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from importlib import metadata
from typing import Any

from .exceptions import ModuleLoadError, ModuleNotFoundError
from .modules import AiModule

ENTRY_POINT_GROUP = "castostudio_ai.modules"
ModuleFactory = Callable[[], AiModule] | type[AiModule]


class ModuleRegistry:
    def __init__(self, factories: Mapping[str, ModuleFactory] | None = None) -> None:
        self._factories = dict(factories or {})

    def available_modules(self) -> tuple[str, ...]:
        names = set(self._factories)
        names.update(entry_point.name for entry_point in self._entry_points())
        return tuple(sorted(names))

    def create(self, name: str) -> AiModule:
        factory = self._factories.get(name) or self._load_entry_point(name)
        try:
            module = factory()
        except Exception as exc:  # pragma: no cover - depends on external packages
            raise ModuleLoadError(f"Cannot construct AI module '{name}'.") from exc

        if not isinstance(module, AiModule):
            raise ModuleLoadError(
                f"AI module '{name}' must inherit castostudio_ai_core.AiModule."
            )
        return module

    def _load_entry_point(self, name: str) -> ModuleFactory:
        for entry_point in self._entry_points():
            if entry_point.name != name:
                continue
            loaded: Any = entry_point.load()
            if inspect.isclass(loaded):
                return loaded
            if callable(loaded):
                return loaded
            raise ModuleLoadError(f"Entry point '{name}' is not callable.")
        raise ModuleNotFoundError(f"AI module '{name}' is not installed.")

    @staticmethod
    def _entry_points() -> tuple[metadata.EntryPoint, ...]:
        entry_points = metadata.entry_points()
        if hasattr(entry_points, "select"):
            return tuple(entry_points.select(group=ENTRY_POINT_GROUP))
        return tuple(entry_points.get(ENTRY_POINT_GROUP, ()))
