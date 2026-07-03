# CastoStudio AI Worker

Python gRPC worker for CastoStudio AI modules. The server receives available stream or recording sources, delegates scene selection to an installed AI module, and returns scene switch suggestions over a bidirectional gRPC stream.

The project is organized as a `uv` workspace so production deployments can install only the AI packages they need.

## Architecture

```text
.
├── ia_analysis.proto              # gRPC contract shared with clients
├── core/                          # AI module interfaces and shared models
├── server/                        # Async gRPC server
├── packages/
│   ├── football/                  # Optional football AI module
│   └── podcast/                   # Optional podcast AI module
├── scripts/generate_proto.py      # Python gRPC code generation
└── tests/                         # Unit tests
```

### Packages

- `castostudio-ai-core`
  - Defines the `AiModule` interface.
  - Defines shared models: `Source`, `SceneDecision`, `SessionContext`.
  - Provides `ModuleRegistry`, which discovers installed AI modules through Python entry points.

- `castostudio-ai-server`
  - Runs the `grpc.aio` server.
  - Implements `IaAnalysisService` from `ia_analysis.proto`.
  - Creates and owns analysis sessions.
  - Loads only installed AI modules.

- `castostudio-ai-football`
  - Optional AI module registered as `football`.

- `castostudio-ai-podcast`
  - Optional AI module registered as `podcast`.

## Requirements

- Python `>=3.11`
- `uv`

Install `uv` from: <https://docs.astral.sh/uv/>

## Install for Development

Install all workspace packages:

```powershell
uv sync --all-packages
```

This installs:

- core
- server
- football module
- podcast module
- test and proto generation dependencies

## Generate gRPC Python Files

The `.proto` file is the source of truth:

```powershell
uv run python scripts/generate_proto.py
```

Generated files are written to:

```text
server/src/castostudio_ai_server/proto/
```

Run this command after editing `ia_analysis.proto`.

## Run the Server

```powershell
uv run castostudio-ai-server --host 0.0.0.0 --port 50051
```

Equivalent:

```powershell
uv run python -m castostudio_ai_server --host 0.0.0.0 --port 50051
```

Useful options:

```powershell
uv run castostudio-ai-server --help
```

## Run Tests

```powershell
uv run pytest
```

## Build Packages

Build all workspace packages:

```powershell
uv build --all-packages
```

Build one package:

```powershell
uv build --package castostudio-ai-core
uv build --package castostudio-ai-server
uv build --package castostudio-ai-football
uv build --package castostudio-ai-podcast
```

Build artifacts are written to `dist/` by default.

## Production Installation

The server discovers AI modules through the `castostudio_ai.modules` entry point group. This means production can install only the modules needed by that deployment.

Minimal server-only deployment:

```powershell
uv pip install castostudio-ai-core castostudio-ai-server
```

Deployment with football only:

```powershell
uv pip install castostudio-ai-core castostudio-ai-server castostudio-ai-football
```

Deployment with podcast only:

```powershell
uv pip install castostudio-ai-core castostudio-ai-server castostudio-ai-podcast
```

Deployment with both modules:

```powershell
uv pip install castostudio-ai-core castostudio-ai-server castostudio-ai-football castostudio-ai-podcast
```

Only installed modules are available to `StartSession`.

## gRPC Flow

The gRPC contract is defined in `ia_analysis.proto`.

### StartSession

Client sends:

- `module_name`
- `module_config`

Server behavior:

- Loads the installed AI module by name.
- Creates a session UUID.
- Calls `module.start(context)`.
- Returns `session_id`.

Common error codes:

- `INVALID_ARGUMENT`
- `MODULE_NOT_FOUND`
- `MODULE_LOAD_FAILED`
- `MODULE_START_FAILED`

### AnalysisStream

Bidirectional stream.

Client sends:

- `SourceList`
- `KeepAlive`
- `StopSignal`

Server returns:

- `SceneSwitch`
- `SessionStatus`
- `ServerError`

For each `SourceList`, the server calls:

```python
await module.analyze_sources(sources)
```

If the module returns a `SceneDecision`, the server emits:

```text
SERVER_EVENT_SWITCH_SUGGESTED
```

### EndSession

Client sends `session_id`.

Server behavior:

- Calls `module.stop()`.
- Removes the session from memory.

Sessions are in-memory only. Restarting the process clears all sessions.

## Add a New AI Module

Create a new package under `packages/`, for example:

```text
packages/basketball/
├── pyproject.toml
└── src/
    └── castostudio_ai_basketball/
        └── __init__.py
```

### 1. Add `pyproject.toml`

```toml
[project]
name = "castostudio-ai-basketball"
version = "0.1.0"
description = "Basketball AI module for CastoStudio."
requires-python = ">=3.11"
dependencies = ["castostudio-ai-core>=0.1.0"]

[project.entry-points."castostudio_ai.modules"]
basketball = "castostudio_ai_basketball:BasketballModule"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/castostudio_ai_basketball"]
```

The entry point name is the value clients pass as `module_name`.

For this example:

```text
module_name = "basketball"
```

### 2. Implement the Module

```python
from __future__ import annotations

from collections.abc import Sequence

from castostudio_ai_core import AiModule, SceneDecision, SessionContext, Source


class BasketballModule(AiModule):
    async def start(self, context: SessionContext) -> None:
        self.context = context

    async def analyze_sources(self, sources: Sequence[Source]) -> SceneDecision | None:
        if not sources:
            return None

        selected = sources[0]
        return SceneDecision(scene_id=selected.scene_id, confidence=0.75)

    async def stop(self) -> None:
        return None
```

### 3. Register the Package in the Workspace

Add it to the root `pyproject.toml`:

```toml
[tool.uv.workspace]
members = [
  "core",
  "server",
  "packages/football",
  "packages/podcast",
  "packages/basketball",
]

[tool.uv.sources]
castostudio-ai-basketball = { workspace = true }
```

### 4. Install and Verify

```powershell
uv sync --all-packages
uv run python -c "from castostudio_ai_core import ModuleRegistry; print(ModuleRegistry().available_modules())"
```

Expected output includes:

```text
basketball
```

## AI Module Contract

Every AI module must inherit `AiModule`:

```python
class AiModule(ABC):
    async def start(self, context: SessionContext) -> None:
        ...

    async def analyze_sources(self, sources: Sequence[Source]) -> SceneDecision | None:
        ...

    async def stop(self) -> None:
        ...
```

### `start`

Called once after `StartSession`.

Use it to:

- Read `module_config`.
- Load model files.
- Initialize session state.

### `analyze_sources`

Called when the client sends a `SourceList`.

Return:

- `SceneDecision(scene_id, confidence)` to suggest a scene switch.
- `None` to keep the current scene.

### `stop`

Called when the session ends or receives a stop signal.

Use it to:

- Release model resources.
- Flush metrics.
- Clean session state.

## Current Built-in Modules

### football

Selects the source with the highest `metadata["score_priority"]`.

Optional config:

```text
default_confidence=0.8
```

### podcast

Selects the first source where:

```text
metadata["active_speaker"] == "true"
```

Falls back to the first source.

## Development Notes

- Keep `ia_analysis.proto` backward compatible when possible.
- Regenerate Python stubs after proto changes.
- Add tests for each new AI module.
- Do not import optional AI packages directly from the server. Use entry points only.
- Keep heavy model dependencies inside optional module packages, not in `core` or `server`.
