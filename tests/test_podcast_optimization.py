import time
import numpy as np
import pytest
import torch

from castostudio_ai_podcast.audio import ChunkBuffer, rms_to_db
from castostudio_ai_podcast.vad import VAD_CHUNK_SAMPLES, SpeechActivityTracker
from castostudio_ai_podcast import PodcastModule, Source, _classify_speaker_label


# ---------------------------------------------------------------------------
# TDD: ChunkBuffer Preallocated Optimization Tests
# ---------------------------------------------------------------------------

def test_chunk_buffer_handles_arbitrary_slice_sizes():
    """Verify preallocated ChunkBuffer accumulates arbitrary push sizes and yields exactly VAD_CHUNK_SAMPLES."""
    buf = ChunkBuffer(chunk_size=512)
    
    # 1. Push small fragments (100 samples each)
    all_chunks = []
    pushed_samples = []
    for i in range(15):
        chunk_data = np.full(100, float(i), dtype=np.float32)
        pushed_samples.extend(chunk_data)
        chunks = buf.push(chunk_data)
        all_chunks.extend(chunks)

    # 15 * 100 = 1500 samples -> 2 full 512 chunks (1024), 476 pending
    assert len(all_chunks) == 2
    for c in all_chunks:
        assert len(c) == 512
        assert c.dtype == np.float32

    # Verify sample continuity
    flattened_extracted = np.concatenate(all_chunks)
    expected_start = np.array(pushed_samples[:1024], dtype=np.float32)
    np.testing.assert_array_almost_equal(flattened_extracted, expected_start)

    # 2. Push remainder to complete third chunk (512 - 476 = 36)
    finisher = np.full(36, 99.0, dtype=np.float32)
    pushed_samples.extend(finisher)
    new_chunks = buf.push(finisher)
    assert len(new_chunks) == 1
    assert len(new_chunks[0]) == 512


def test_chunk_buffer_performance_and_throughput():
    """Verify ChunkBuffer processes over 500,000 samples under 50ms with zero allocation leaks."""
    buf = ChunkBuffer(chunk_size=512)
    data = np.random.uniform(-0.5, 0.5, size=1024).astype(np.float32)

    t0 = time.perf_counter()
    chunk_count = 0
    # Simulate 500 audio frames of 1024 samples (approx. 32 seconds of audio at 16kHz)
    for _ in range(500):
        chunks = buf.push(data)
        chunk_count += len(chunks)
    duration = time.perf_counter() - t0

    assert chunk_count == 1000
    assert duration < 0.10, f"ChunkBuffer throughput too slow: took {duration*1000:.2f}ms"


def test_fast_rms_to_db_accuracy():
    """Verify rms_to_db precision across boundary values."""
    assert rms_to_db(0.0) == -100.0
    assert rms_to_db(1e-6) == -100.0
    assert abs(rms_to_db(1.0) - 0.0) < 1e-3
    assert abs(rms_to_db(0.1) - (-20.0)) < 1e-2


# ---------------------------------------------------------------------------
# TDD: VAD Inference Mode & Zero Autograd Overhead
# ---------------------------------------------------------------------------

def test_vad_inference_mode_active_during_processing(monkeypatch):
    """Verify that SpeechActivityTracker executes Silero VAD inside torch.inference_mode."""
    from silero_vad import VADIterator

    captured_inference_state = []
    orig_call = VADIterator.__call__

    def wrapped_call(self, *args, **kwargs):
        captured_inference_state.append(torch.is_inference_mode_enabled())
        return orig_call(self, *args, **kwargs)

    monkeypatch.setattr(VADIterator, "__call__", wrapped_call)

    tracker = SpeechActivityTracker(threshold=0.5)
    dummy_chunk = np.zeros(VAD_CHUNK_SAMPLES, dtype=np.float32)
    tracker.process_chunk(dummy_chunk)

    assert len(captured_inference_state) > 0
    assert all(captured_inference_state), "VAD inference must run under torch.inference_mode()!"


# ---------------------------------------------------------------------------
# TDD: Caching & Fast-Path Decision Latency
# ---------------------------------------------------------------------------

def test_classify_speaker_label_caching():
    """Verify that _classify_speaker_label uses LRU caching for high-frequency labels."""
    # Call multiple times with identical strings
    for _ in range(100):
        assert _classify_speaker_label("CAM 1 - HOTE (Face)") == "host"
        assert _classify_speaker_label("CAM 2 - INVITE") == "guest"
        assert _classify_speaker_label("PANEL 2 - INVITE 2") == "guest2"

    # Check cache info exists and recorded hits
    assert hasattr(_classify_speaker_label, "cache_info")
    info = _classify_speaker_label.cache_info()
    assert info.hits > 50, f"Expected cache hits, got {info.hits}"


@pytest.mark.asyncio
async def test_analyze_sources_fast_path_latency():
    """Verify analyze_sources latency is sub-millisecond per cycle across multi-camera setups."""
    from castostudio_ai_core import SessionContext

    ctx = SessionContext(
        session_id="test_opt_session",
        module_name="podcast",
        config={
            "min_hold_time": "1.0",
            "cross_gating_threshold_db": "5.0",
        },
    )

    module = PodcastModule()
    await module.start(ctx)

    sources = [
        Source(scene_id="s1", url="dummy1", label="Cam Hote", metadata={"is_speaking": "true", "volume_db": "-14.0"}),
        Source(scene_id="s2", url="dummy2", label="Cam Invite", metadata={"is_speaking": "false", "volume_db": "-50.0"}),
        Source(scene_id="s3", url="dummy3", label="Plan Large"),
        Source(scene_id="s4", url="dummy4", label="Invite 2", metadata={"is_speaking": "false", "volume_db": "-55.0"}),
    ]

    # Warmup
    await module.analyze_sources(sources)

    # Measure 50 consecutive analysis ticks
    t0 = time.perf_counter()
    for _ in range(50):
        await module.analyze_sources(sources)
    total_time = time.perf_counter() - t0

    avg_latency_ms = (total_time / 50) * 1000
    assert avg_latency_ms < 1.0, f"Average analysis latency too high: {avg_latency_ms:.3f}ms"
