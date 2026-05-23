import time
import numpy as np
from audio.audio_pipeline import AudioPipeline
from tests.utils.fake_queue_manager import FakeQueueManager


def test_audio_pipeline_emits_event(monkeypatch):
    qm = FakeQueueManager()
    pipeline = AudioPipeline(queue_manager=qm, vad_model_path=None, device='cpu')

    # Monkeypatch VAD to trigger on any non-zero chunk
    def fake_process_chunk(chunk, state):
        energy = float(np.mean(np.abs(chunk)))
        return (0.9 if energy > 0.0 else 0.0), state

    monkeypatch.setattr(pipeline.vad, 'process_chunk', fake_process_chunk)

    # Monkeypatch transcriber to return predictable transcript
    def fake_transcribe(buf):
        return "test speech", 0.95

    monkeypatch.setattr(pipeline.transcriber, 'transcribe', fake_transcribe)

    # Start pipeline thread
    pipeline.start()

    try:
        # Submit a chunk with energy
        chunk = np.ones(pipeline.CHUNK_SIZE, dtype=np.float32) * 0.1
        pipeline.submit_audio_chunk(chunk)

        # Wait briefly for pipeline to process
        deadline = time.time() + 2.0
        while time.time() < deadline and len(qm.audio_events) == 0:
            time.sleep(0.05)

        assert len(qm.audio_events) >= 1
        evt = qm.audio_events[0]
        assert hasattr(evt, 'transcript')
    finally:
        pipeline.stop()


def test_audio_pipeline_fallback_distress_when_transcript_missing(monkeypatch):
    qm = FakeQueueManager()
    pipeline = AudioPipeline(queue_manager=qm, vad_model_path=None, device='cpu')

    def fake_process_chunk(chunk, state):
        return 0.95, state

    def fake_transcribe(buf):
        return "", 0.0

    monkeypatch.setattr(pipeline.vad, 'process_chunk', fake_process_chunk)
    monkeypatch.setattr(pipeline.transcriber, 'transcribe', fake_transcribe)

    pipeline.start()
    try:
        loud_chunk = np.ones(pipeline.CHUNK_SIZE * 2, dtype=np.float32) * 0.8
        pipeline.submit_audio_chunk(loud_chunk)

        deadline = time.time() + 2.0
        while time.time() < deadline and len(qm.audio_events) == 0:
            time.sleep(0.05)

        assert len(qm.audio_events) >= 1
        evt = qm.audio_events[-1]
        assert evt.intent.value in ("distress", "threat", "neutral")
        assert evt.intent.value == "distress"
        assert evt.intent_confidence >= 0.7
    finally:
        pipeline.stop()
