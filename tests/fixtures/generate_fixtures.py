"""Generate audio and video fixtures for tests.

Creates:
 - sample_silence.wav    (3s, 16kHz)
 - sample_tone.wav       (3s, 16kHz, 440Hz sine)
 - sample_speech_like.wav (3s, 16kHz, amplitude-modulated tone)
 - sample_person.mp4     (3s, 1280x720, 30fps, moving rectangle)

Run:
    python tests/fixtures/generate_fixtures.py

Requires: numpy, soundfile, opencv-python
"""
import os
import numpy as np
import soundfile as sf
import cv2

OUT_DIR = os.path.dirname(__file__)
SAMPLE_RATE = 16000
DURATION = 3.0


def write_wav(filename, data, sr=SAMPLE_RATE):
    path = os.path.join(OUT_DIR, filename)
    sf.write(path, data, sr, subtype='PCM_16')
    print(f"Wrote {path}")


def generate_silence():
    samples = int(SAMPLE_RATE * DURATION)
    data = np.zeros(samples, dtype=np.float32)
    write_wav('sample_silence.wav', data)


def generate_tone(freq=440.0):
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    data = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    write_wav('sample_tone.wav', data)


def generate_speech_like():
    # Amplitude-modulated tone to mimic speech envelope
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    carrier = np.sin(2 * np.pi * 220.0 * t)

    # Create a random smooth envelope in [0,1]
    rng = np.random.default_rng(1234)
    steps = int(DURATION * 30)  # 30 envelope points
    env_points = rng.uniform(0.2, 1.0, size=steps)
    env = np.interp(np.linspace(0, steps, len(t)), np.arange(steps), env_points)
    data = (0.6 * carrier * env).astype(np.float32)
    write_wav('sample_speech_like.wav', data)


def generate_video():
    width, height = 1280, 720
    fps = 30
    frames = int(DURATION * fps)
    out_path = os.path.join(OUT_DIR, 'sample_person.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    for i in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # moving rectangle
        x = int((width - 200) * (i / max(1, frames - 1)))
        y = height // 3
        cv2.rectangle(frame, (x, y), (x + 200, y + 300), (50, 200, 50), -1)
        cv2.putText(frame, f"Frame {i}", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
        writer.write(frame)

    writer.release()
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    generate_silence()
    generate_tone()
    generate_speech_like()
    generate_video()
    print('All fixtures generated in', OUT_DIR)
