from __future__ import annotations

import argparse
import asyncio
import logging

from .server import serve

LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CastoStudio AI gRPC server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=50051, type=int)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(log_level, int):
        parser.error(f"invalid --log-level: {args.log_level}")

    logging.basicConfig(level=log_level, format=LOG_FORMAT)
    asyncio.run(serve(host=args.host, port=args.port))
