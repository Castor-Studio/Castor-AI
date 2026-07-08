from __future__ import annotations

import asyncio
import logging

import grpc

from .proto import ia_analysis_pb2_grpc
from .service import IaAnalysisService

LOGGER = logging.getLogger(__name__)


async def create_server(service: IaAnalysisService | None = None) -> grpc.aio.Server:
    server = grpc.aio.server()
    ia_analysis_pb2_grpc.add_IaAnalysisServiceServicer_to_server(
        service or IaAnalysisService(),
        server,
    )
    return server


async def serve(host: str = "0.0.0.0", port: int = 50051) -> None:
    server = await create_server()
    address = f"{host}:{port}"
    server.add_insecure_port(address)
    await server.start()
    LOGGER.info("CastoStudio AI server listening on %s", address)
    try:
        await server.wait_for_termination()
    except asyncio.CancelledError:
        LOGGER.info("Stopping gRPC server...")
        await server.stop(grace=1)
        LOGGER.info("gRPC server stopped.")
        raise

