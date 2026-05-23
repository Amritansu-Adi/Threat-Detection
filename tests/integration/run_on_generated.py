"""Simple runner to exercise audio pipeline on generated assets.

Usage:
  python tests/integration/run_on_generated.py --wav tests/fixtures/generated/yourfile.wav

This script will load the WAV, feed it in CHUNK_SIZE slices to `AudioPipeline.submit_audio_chunk`
and print pipeline status. It expects ffmpeg-produced 16kHz WAVs created by
`tools/ingest_generated_assets.py`.
"""
import argparse
import os
import soundfile as sf
import time
import numpy as np

from audio.audio_pipeline import AudioPipeline
from core.queue_manager import QueueManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav', required=True, help='Path to 16kHz mono WAV')
    args = parser.parse_args()

    if not os.path.exists(args.wav):
        print('WAV not found:', args.wav)
        return

    data, sr = sf.read(args.wav)
    if sr != AudioPipeline.SAMPLE_RATE:
        print('Warning: WAV sample rate != 16kHz:', sr)

    qm = QueueManager()
    pipeline = AudioPipeline(queue_manager=qm, vad_model_path=None, device='cpu')

    pipeline.start()
    try:
        # Feed audio in CHUNK_SIZE windows
        i = 0
        while i < len(data):
            chunk = data[i:i + pipeline.CHUNK_SIZE]
            pipeline.submit_audio_chunk(np.asarray(chunk, dtype=np.float32))
            i += pipeline.CHUNK_SIZE
            time.sleep(0.01)

        # Wait for processing
        time.sleep(2.0)
        print('Pipeline status:', pipeline.get_status())
        print('Queue audio events:', len(qm.get_audio_events()))
    finally:
        pipeline.stop()


if __name__ == '__main__':
    main()
