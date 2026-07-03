from __future__ import annotations

from pathlib import Path

from grpc_tools import protoc

ROOT = Path(__file__).resolve().parents[1]
PROTO = ROOT / "ia_analysis.proto"
OUT = ROOT / "server" / "src" / "castostudio_ai_server" / "proto"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result = protoc.main(
        [
            "grpc_tools.protoc",
            f"--proto_path={ROOT}",
            f"--python_out={OUT}",
            f"--grpc_python_out={OUT}",
            str(PROTO),
        ]
    )
    if result != 0:
        raise SystemExit(result)

    grpc_file = OUT / "ia_analysis_pb2_grpc.py"
    content = grpc_file.read_text(encoding="utf-8")
    content = content.replace(
        "import ia_analysis_pb2 as ia__analysis__pb2",
        "from . import ia_analysis_pb2 as ia__analysis__pb2",
    )
    grpc_file.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
