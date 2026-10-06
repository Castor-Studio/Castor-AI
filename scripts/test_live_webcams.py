"""Interactive live test client for macOS webcams and microphones with Castor AI.

Discovers connected AVFoundation devices (webcams, mics), connects to the gRPC
AI backend server, and runs real-time switching analysis without needing the C# frontend.

Usage:
    # 1. Start the server (in one terminal or let this script connect to localhost:50051):
    uv run castostudio-ai-server

    # 2. Run the live test (in another terminal):
    uv run python scripts/test_live_webcams.py --list-devices
    uv run python scripts/test_live_webcams.py --mic1 1 --mic2 2
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import signal
import subprocess
import sys
import time

import grpc

from castostudio_ai_server.proto import ia_analysis_pb2, ia_analysis_pb2_grpc

LOGGER = logging.getLogger("test_live_webcams")


def get_avfoundation_devices() -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Parse video and audio devices from ffmpeg AVFoundation output."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            check=False,
        )
        output = proc.stderr
    except FileNotFoundError:
        return [], []

    video_devices = []
    audio_devices = []
    is_audio = False

    for line in output.splitlines():
        if "AVFoundation video devices:" in line:
            is_audio = False
            continue
        if "AVFoundation audio devices:" in line:
            is_audio = True
            continue

        match = re.search(r"\[(\d+)\]\s+(.*)", line)
        if match:
            idx = int(match.group(1))
            name = match.group(2).strip()
            if is_audio:
                audio_devices.append((idx, name))
            else:
                video_devices.append((idx, name))

    return video_devices, audio_devices


def print_devices(video_devices: list[tuple[int, str]], audio_devices: list[tuple[int, str]]) -> None:
    print("\n🎥 Available Video Devices (Webcams/Screens):")
    if not video_devices:
        print("  (None found)")
    for idx, name in video_devices:
        print(f"  [{idx}] {name}")

    print("\n🎙️ Available Audio Devices (Microphones):")
    if not audio_devices:
        print("  (None found)")
    for idx, name in audio_devices:
        print(f"  [{idx}] {name}")
    print("")


async def outgoing_messages(
    session_id: str,
    sources: list[ia_analysis_pb2.Source],
    stop_event: asyncio.Event,
):
    # Initial SourceList
    yield ia_analysis_pb2.ClientMessage(
        session_id=session_id,
        sources=ia_analysis_pb2.SourceList(sources=sources),
    )
    # Periodic KeepAlive
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            yield ia_analysis_pb2.ClientMessage(
                session_id=session_id,
                keep_alive=ia_analysis_pb2.KeepAlive(timestamp_ms=int(time.time() * 1000)),
            )
    # Stop Signal
    yield ia_analysis_pb2.ClientMessage(
        session_id=session_id,
        stop=ia_analysis_pb2.StopSignal(reason="User stopped live session"),
    )


def resolve_device_index(query: str | int | None, devices: list[tuple[int, str]]) -> int | None:
    if query is None:
        return None
    if isinstance(query, int) or (isinstance(query, str) and query.isdigit()):
        return int(query)
    q = str(query).lower()
    for idx, name in devices:
        if q in name.lower():
            return idx
    return None


async def run_live(args: argparse.Namespace) -> None:
    video_devices, audio_devices = get_avfoundation_devices()

    if args.list_devices:
        print_devices(video_devices, audio_devices)
        return

    # Determine mic and cam selections (support name substrings or index)
    mic1 = resolve_device_index(args.mic1, audio_devices) if args.mic1 is not None else 1
    mic2 = resolve_device_index(args.mic2, audio_devices) if args.mic2 is not None else 2
    cam1 = resolve_device_index(args.cam1, video_devices)
    cam2 = resolve_device_index(args.cam2, video_devices)
    cam_wide = resolve_device_index(args.cam_wide, video_devices)

    mic1_name = next((name for idx, name in audio_devices if idx == mic1), f"Index {mic1}")
    mic2_name = next((name for idx, name in audio_devices if idx == mic2), f"Index {mic2}")
    cam1_name = next((name for idx, name in video_devices if idx == cam1), f"Index {cam1}") if cam1 is not None else "Audio only"
    cam2_name = next((name for idx, name in video_devices if idx == cam2), f"Index {cam2}") if cam2 is not None else "Audio only"

    print("\n=== Castor AI Podcast Live Test (macOS) ===")
    print(f"Server address: {args.addr}")
    print(f"Host Source (Cam Hote):    Mic [{mic1}] {mic1_name} | Cam [{cam1}] {cam1_name}")
    print(f"Guest Source (Cam Invite):  Mic [{mic2}] {mic2_name} | Cam [{cam2}] {cam2_name}")
    if cam_wide is not None:
        cam_wide_name = next((name for idx, name in video_devices if idx == cam_wide), f"Index {cam_wide}")
        print(f"Wide Source (Plan Large):   Cam [{cam_wide}] {cam_wide_name}")
    print("===========================================\n")

    # Construct device URLs for PyAV / AVFoundation
    # PyAV format for avfoundation audio only is "avfoundation::<mic_index>"
    # or with video "avfoundation:<cam_index>:<mic_index>"
    url_host = f"avfoundation::{mic1}" if cam1 is None else f"avfoundation:{cam1}:{mic1}"
    url_guest = f"avfoundation::{mic2}" if cam2 is None else f"avfoundation:{cam2}:{mic2}"

    sources = [
        ia_analysis_pb2.Source(scene_id="s1_host", label="Cam Hote", url=url_host),
        ia_analysis_pb2.Source(scene_id="s2_guest", label="Cam Invite", url=url_guest),
        ia_analysis_pb2.Source(scene_id="s3_wide", label="Plan Large", url=url_host),
        ia_analysis_pb2.Source(scene_id="s1_host_zoom", label="Zoom Hote", url=url_host),
        ia_analysis_pb2.Source(scene_id="s2_guest_zoom", label="Zoom Invite", url=url_guest),
    ]

    config = {
        "min_hold_time": str(args.min_hold_time),
        "monologue_time": str(args.monologue_time),
        "vad_threshold": str(args.vad_threshold),
    }

    print(f"Connecting to gRPC server at {args.addr}...")
    async with grpc.aio.insecure_channel(args.addr) as channel:
        stub = ia_analysis_pb2_grpc.IaAnalysisServiceStub(channel)

        try:
            start_resp = await stub.StartSession(
                ia_analysis_pb2.StartSessionRequest(module_name="podcast", module_config=config)
            )
        except grpc.RpcError as exc:
            print(f"❌ Could not connect to gRPC server ({exc.code()}): {exc.details()}")
            print("👉 Make sure 'uv run castostudio-ai-server' is running in another terminal.")
            return

        if not start_resp.success:
            print(f"❌ StartSession failed: {start_resp.message}")
            return

        session_id = start_resp.session_id
        print(f"✅ AI Session started: {session_id}")
        print("🎙️ Listening to microphones and computing real-time decisions...")
        print("💡 Try speaking into Mic 1 (Host) or Mic 2 (Guest) or both (Debate) or staying silent.\n")

        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except NotImplementedError:
                pass

        stream_call = stub.AnalysisStream(outgoing_messages(session_id, sources, stop_event))

        try:
            async for event in stream_call:
                payload_type = event.WhichOneof("payload")
                if payload_type == "switch_suggestion":
                    sug = event.switch_suggestion
                    target_label = "Unknown"
                    for s in sources:
                        if s.scene_id == sug.scene_id:
                            target_label = s.label
                            break
                    print(
                        f"🎬 \033[1;32m[SWITCH DECISION]\033[0m -> Switch to \033[1;36m{target_label}\033[0m "
                        f"(scene_id={sug.scene_id}, confidence={sug.confidence:.2f})"
                    )
                elif payload_type == "status":
                    print(f"ℹ️  [STATUS] {event.status.message}")
                elif payload_type == "error":
                    print(f"⚠️  [ERROR] {event.error.error_message}")
        except asyncio.CancelledError:
            pass
        finally:
            stop_event.set()
            await stub.EndSession(ia_analysis_pb2.EndSessionRequest(session_id=session_id))
            print(f"\n🛑 Session {session_id} ended.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--addr", default="localhost:50051", help="gRPC server address (default: localhost:50051)")
    parser.add_argument("--list-devices", action="store_true", help="List all AVFoundation video/audio devices")
    parser.add_argument("--mic1", default="1", help="AVFoundation index or name for Mic 1 (Host). Default: 1 (MacBook Mic)")
    parser.add_argument("--mic2", default="2", help="AVFoundation index or name for Mic 2 (Guest). Default: 2 (iPhone/External Mic)")
    parser.add_argument("--cam1", default=None, help="AVFoundation index or name for Camera 1 (Host), e.g. 0 or 'facetime'")
    parser.add_argument("--cam2", default=None, help="AVFoundation index or name for Camera 2 (Guest), e.g. 2 or 'iphone'")
    parser.add_argument("--cam-wide", default=None, help="AVFoundation index or name for Wide Camera")
    parser.add_argument("--min-hold-time", type=float, default=2.5, help="Anti-flicker hold duration in seconds")
    parser.add_argument("--monologue_time", type=float, default=5.0, help="Seconds before zoom trigger")
    parser.add_argument("--vad-threshold", type=float, default=0.5, help="Silero VAD speech sensitivity (0.0 to 1.0)")

    args = parser.parse_args()

    try:
        asyncio.run(run_live(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
