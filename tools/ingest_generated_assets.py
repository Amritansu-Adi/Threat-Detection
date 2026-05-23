"""Ingest externally-generated videos (e.g., from an LLM service) for tests.

Usage:
  python tools/ingest_generated_assets.py --src <video_dir> --out tests/fixtures/generated

This script:
 - Finds video files (mp4/mkv/avi) under --src
 - Extracts and resamples audio to 16kHz mono WAV via `ffmpeg`
 - Writes audio files into the output directory keeping base names

Requires: ffmpeg in PATH
"""
import argparse
import os
import subprocess
from pathlib import Path


VIDEO_EXTS = {'.mp4', '.mkv', '.avi', '.mov'}


def find_videos(src_dir):
    for root, _, files in os.walk(src_dir):
        for f in files:
            if Path(f).suffix.lower() in VIDEO_EXTS:
                yield os.path.join(root, f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def extract_audio(video_path, out_wav_path):
    # Use ffmpeg to extract audio and resample to 16kHz mono
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-ar', '16000', '-ac', '1', '-vn', out_wav_path
    ]
    try:
        subprocess.check_call(cmd)
        print(f'Extracted: {out_wav_path}')
        return True
    except subprocess.CalledProcessError as e:
        print(f'ffmpeg failed for {video_path}: {e}')
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='Source directory with videos')
    parser.add_argument('--out', default='tests/fixtures/generated', help='Output directory for WAVs')
    args = parser.parse_args()

    src = args.src
    out = args.out
    ensure_dir(out)

    videos = list(find_videos(src))
    if not videos:
        print('No video files found in', src)
        return

    for v in videos:
        base = Path(v).stem
        out_wav = os.path.join(out, f'{base}.wav')
        extract_audio(v, out_wav)


if __name__ == '__main__':
    main()
