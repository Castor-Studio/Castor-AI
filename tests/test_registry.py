from __future__ import annotations

import pytest

from castostudio_ai_core import AiModule, ModuleLoadError, ModuleRegistry, SceneDecision


class GoodModule(AiModule):
    async def start(self, context):
        return None

    async def analyze_sources(self, sources):
        return SceneDecision(scene_id="scene-a", confidence=0.8)

    async def stop(self):
        return None


class BadModule:
    pass


def test_registry_creates_injected_module():
    registry = ModuleRegistry({"good": GoodModule})

    assert isinstance(registry.create("good"), GoodModule)
    assert "good" in registry.available_modules()


def test_registry_rejects_non_ai_module():
    registry = ModuleRegistry({"bad": BadModule})

    with pytest.raises(ModuleLoadError):
        registry.create("bad")
