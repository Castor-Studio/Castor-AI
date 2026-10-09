from __future__ import annotations

import copy
import logging
import threading

import numpy as np
import torch
from silero_vad import VADIterator, load_silero_vad

LOGGER = logging.getLogger(__name__)

VAD_SAMPLE_RATE = 16000
VAD_CHUNK_SAMPLES = 512  # required chunk size for Silero streaming at 16kHz


def select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SpeechModel:
    """Process-wide Silero VAD weights, loaded once on the best available device."""

    _lock = threading.Lock()
    _instance: "SpeechModel | None" = None

    def __init__(self, device: str) -> None:
        self.device = device
        self.model = load_silero_vad(onnx=False)
        self.model.to(device)

    @classmethod
    def get(cls) -> "SpeechModel":
        with cls._lock:
            if cls._instance is None:
                device = select_device()
                LOGGER.info("Loading Silero VAD model on device: %s", device)
                cls._instance = cls(device)
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None


class SpeechActivityTracker:
    """Per-stream VAD state. One instance per audio source, never shared."""

    def __init__(
        self,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 400,
        speech_pad_ms: int = 100,
    ) -> None:
        speech_model = SpeechModel.get()
        self._device = speech_model.device
        # Silero's model keeps recurrent state as an internal attribute, not as a
        # call argument, so concurrent streams must never share one model object.
        # deepcopy avoids re-reading model weights from disk per stream.
        model = copy.deepcopy(speech_model.model)
        self._iterator = VADIterator(
            model,
            threshold=threshold,
            sampling_rate=VAD_SAMPLE_RATE,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self._is_speaking = False

    def process_chunk(self, chunk: np.ndarray) -> bool:
        """chunk: float32 numpy array, exactly VAD_CHUNK_SAMPLES mono samples in [-1, 1]."""
        with torch.inference_mode():
            tensor = torch.from_numpy(chunk).to(self._device)
            event = self._iterator(tensor, return_seconds=False)
            if event:
                if "start" in event:
                    self._is_speaking = True
                elif "end" in event:
                    self._is_speaking = False
            return self._is_speaking

    def is_speaking(self) -> bool:
        return self._is_speaking

    def reset(self) -> None:
        self._iterator.reset_states()
        self._is_speaking = False
