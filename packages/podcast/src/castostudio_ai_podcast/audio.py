from __future__ import annotations

import logging
import threading
import time

import av
import numpy as np

from .vad import VAD_CHUNK_SAMPLES, VAD_SAMPLE_RATE, SpeechActivityTracker

LOGGER = logging.getLogger(__name__)

STREAM_OPEN_OPTIONS = {
    "rtsp_transport": "tcp",
    "stimeout": "5000000",  # 5 seconds
    "rw_timeout": "5000000",
}
INITIAL_BACKOFF_SEC = 0.5
MAX_BACKOFF_SEC = 4.0


def frame_to_mono_pcm16(frame: "av.AudioFrame", resampler: "av.AudioResampler") -> np.ndarray:
    """Resample a decoded frame to 16kHz mono and flatten to int16 samples."""
    resampled_frames = resampler.resample(frame)
    if not resampled_frames:
        return np.empty(0, dtype=np.int16)
    parts = [f.to_ndarray().reshape(-1) for f in resampled_frames]
    return parts[0] if len(parts) == 1 else np.concatenate(parts)


def pcm16_to_float32(samples: np.ndarray) -> np.ndarray:
    return samples.astype(np.float32) / 32768.0


def rms_to_db(rms: float) -> float:
    if rms > 1e-5:
        return float(20 * np.log10(rms))
    return -100.0


class ChunkBuffer:
    """Accumulates variable-length PCM pushes and yields fixed-size VAD chunks."""

    def __init__(self, chunk_size: int = VAD_CHUNK_SAMPLES) -> None:
        self._chunk_size = chunk_size
        self._pending = np.empty(0, dtype=np.float32)

    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        if samples.size == 0:
            return []
        buffer = np.concatenate([self._pending, samples])
        chunks = []
        offset = 0
        while offset + self._chunk_size <= buffer.size:
            chunks.append(buffer[offset : offset + self._chunk_size])
            offset += self._chunk_size
        self._pending = buffer[offset:]
        return chunks


def is_device_url(url: str) -> tuple[bool, str, str]:
    """Check if URL targets a local hardware capture device (AVFoundation, DShow, ALSA, V4L2)."""
    device_prefixes = ("avfoundation", "dshow", "v4l2", "alsa")
    for prefix in device_prefixes:
        if url.startswith(f"{prefix}:"):
            fmt, _, dev = url.partition(":")
            return True, fmt, dev

    if url.startswith(":") or (
        len(url.split(":")) == 2 and url.split(":")[0].isdigit() and url.split(":")[1].isdigit()
    ):
        import sys
        if sys.platform == "darwin":
            return True, "avfoundation", url

    return False, "", ""


class AudioStreamReader:
    """Reads one audio source, feeding a VAD tracker and exposing is_speaking/volume."""

    def __init__(
        self,
        url: str,
        label: str,
        vad_threshold: float = 0.5,
        min_silence_duration_ms: int = 400,
        speech_pad_ms: int = 100,
    ):
        self.url = url
        self.label = label
        self.running = False
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.latest_db = -100.0
        self.failures = 0

        self._tracker = SpeechActivityTracker(
            threshold=vad_threshold,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self._buffer = ChunkBuffer()
        self._is_speaking = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(
            target=self._run, name=f"AudioReader-{self.label}", daemon=True
        )
        self.thread.start()
        LOGGER.info("Audio reader thread started for %s (%s)", self.label, self.url)

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
        LOGGER.info("Audio reader thread stopped for %s", self.label)

    def get_volume_db(self) -> float:
        """Diagnostic only. Speaking decisions come from is_speaking()."""
        with self.lock:
            return self.latest_db

    def is_speaking(self) -> bool:
        with self.lock:
            return self._is_speaking

    def _run(self):
        is_dev, fmt, dev = is_device_url(self.url)
        if is_dev:
            self._run_device_pipe(fmt, dev)
        else:
            self._run_pyav()

    def _run_device_pipe(self, fmt: str, dev: str) -> None:
        """Stream raw 16kHz PCM audio directly from hardware device via ffmpeg pipe."""
        import subprocess

        bytes_per_chunk = VAD_CHUNK_SAMPLES * 2  # 512 samples * 2 bytes = 1024 bytes

        while self.running:
            proc = None
            try:
                cmd = [
                    "ffmpeg",
                    "-loglevel", "error",
                    "-f", fmt,
                    "-i", dev,
                    "-f", "s16le",
                    "-ar", str(VAD_SAMPLE_RATE),
                    "-ac", "1",
                    "pipe:1",
                ]
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                self.failures = 0
                self._tracker.reset()
                self._buffer = ChunkBuffer()

                while self.running:
                    raw = proc.stdout.read(bytes_per_chunk)
                    if not raw or len(raw) < bytes_per_chunk:
                        break

                    pcm16 = np.frombuffer(raw, dtype=np.int16)
                    samples = pcm16_to_float32(pcm16)
                    for chunk in self._buffer.push(samples):
                        rms = float(np.sqrt(np.mean(chunk**2)))
                        speaking = self._tracker.process_chunk(chunk)
                        with self.lock:
                            self.latest_db = rms_to_db(rms)
                            self._is_speaking = speaking

                if proc:
                    proc.terminate()
                    proc.wait(timeout=0.5)

            except Exception as exc:
                self.failures += 1
                LOGGER.warning(
                    "Error reading device audio for %s (fail count: %d): %s",
                    self.label,
                    self.failures,
                    exc,
                )
                if proc:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                with self.lock:
                    self._is_speaking = False
                backoff = min(INITIAL_BACKOFF_SEC * (2 ** (self.failures - 1)), MAX_BACKOFF_SEC)
                time.sleep(backoff)

    def _run_pyav(self) -> None:
        """Stream audio using PyAV for network streams and local files."""
        while self.running:
            container = None
            try:
                container = av.open(self.url, options=STREAM_OPEN_OPTIONS)
                audio_streams = [s for s in container.streams if s.type == "audio"]

                if not audio_streams:
                    LOGGER.warning("No audio stream found for %s (%s)", self.label, self.url)
                    time.sleep(2)
                    continue

                audio_stream = audio_streams[0]
                self.failures = 0
                self._tracker.reset()
                self._buffer = ChunkBuffer()
                resampler = av.AudioResampler(format="s16", layout="mono", rate=VAD_SAMPLE_RATE)

                for frame in container.decode(audio_stream):
                    if not self.running:
                        break

                    pcm16 = frame_to_mono_pcm16(frame, resampler)
                    if pcm16.size == 0:
                        continue

                    samples = pcm16_to_float32(pcm16)
                    for chunk in self._buffer.push(samples):
                        rms = float(np.sqrt(np.mean(chunk**2)))
                        speaking = self._tracker.process_chunk(chunk)
                        with self.lock:
                            self.latest_db = rms_to_db(rms)
                            self._is_speaking = speaking

                if container:
                    container.close()

            except Exception as exc:
                self.failures += 1
                LOGGER.warning(
                    "Error reading audio for %s (fail count: %d): %s",
                    self.label,
                    self.failures,
                    exc,
                )
                if container:
                    try:
                        container.close()
                    except Exception:
                        pass
                with self.lock:
                    self._is_speaking = False
                backoff = min(INITIAL_BACKOFF_SEC * (2 ** (self.failures - 1)), MAX_BACKOFF_SEC)
                time.sleep(backoff)
