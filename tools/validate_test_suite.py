"""Run TEST videos through local pipelines and summarize observed outputs.

This script will:
 - Discover TEST/test*.mp4 videos
 - Extract audio to a temp WAV (system ffmpeg or imageio-ffmpeg)
 - Run VisualPipeline and AudioPipeline on the file input
 - Collect the latest VisualData and AudioEvent

Note: This runner tolerates missing models by reporting which checks were skipped.
"""
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

import cv2
import soundfile as sf
import numpy as np
try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

import sys
# Ensure repo root is on sys.path so `import core` works when running this script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".ultralytics"))

from core.queue_manager import QueueManager
from core.config_loader import ConfigLoader
from visual.visual_pipeline import VisualPipeline
from audio.audio_pipeline import AudioPipeline


ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / 'TEST'


def normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s]', ' ', text.lower())).strip()


def transcript_matches(expected: str, actual: str) -> bool:
    expected_norm = normalize_text(expected)
    actual_norm = normalize_text(actual)
    if not expected_norm:
        return True
    if expected_norm in actual_norm:
        return True

    expected_tokens = expected_norm.split()
    actual_tokens = set(actual_norm.split())
    if not expected_tokens:
        return True
    overlap = sum(1 for token in expected_tokens if token in actual_tokens)
    return overlap / len(expected_tokens) >= 0.65


def extract_audio_from_video(video_path: str, out_wav: str) -> bool:
    ffmpeg_exe = "ffmpeg"
    if imageio_ffmpeg is not None:
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_exe = "ffmpeg"
    cmd = [ffmpeg_exe, '-y', '-i', str(video_path), '-ar', '16000', '-ac', '1', '-vn', out_wav]
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def run_video_through_pipelines(video_path: str, wav_path: str, target_fps: float = 15.0) -> Dict[str, Any]:
    qm = QueueManager()
    visual_config = ConfigLoader().load().get("visual", {})

    person_model = visual_config.get("person_model", "runs/detect/person_finetune/weights/best.pt")
    weapon_model = visual_config.get("weapon_model", "runs/train/weapon_train4/weights/best.pt")

    # Visual pipeline - configure a bit lower FPS for reliability
    try:
        vis = VisualPipeline(target_fps=target_fps,
                             person_model_path=person_model,
                             weapon_model_path=weapon_model,
                             visual_to_fusion_queue=qm.visual_to_fusion_q,
                             fusion_to_visual_queue=qm.fusion_to_visual_q)
    except FileNotFoundError as e:
        print(f"Warning: Failed to initialize VisualPipeline (weapon model issue): {e}")
        print("Retrying with weapon detection disabled...")
        # Create a minimal pipeline without weapon detection
        vis = VisualPipeline(target_fps=target_fps,
                             weapon_model_path="",
                             visual_to_fusion_queue=qm.visual_to_fusion_q,
                             fusion_to_visual_queue=qm.fusion_to_visual_q)
    vad_model = str((ROOT / "models" / "silero_vad.onnx"))
    if not Path(vad_model).exists():
        vad_model = None
    aud = AudioPipeline(queue_manager=qm, vad_model_path=vad_model, device='cpu')

    vis.start()
    aud.start()

    try:
        # Feed video frames
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
        frame_interval = 1.0 / max(1.0, fps)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            vis.submit_frame(frame)
            time.sleep(frame_interval * 0.5)  # slightly faster than real-time to finish quickly

        cap.release()

        # Feed audio chunks
        data, sr = sf.read(str(wav_path))
        if sr != aud.SAMPLE_RATE:
            # Resampling handled by ffmpeg extraction step; if mismatch, just warn
            print('Warning: WAV sample rate mismatch', sr)

        # Submit in CHUNK_SIZE blocks
        i = 0
        while i < len(data):
            chunk = np.asarray(data[i:i + aud.CHUNK_SIZE], dtype=np.float32)
            aud.submit_audio_chunk(chunk)
            i += aud.CHUNK_SIZE
            time.sleep(0.005)

        # Allow some time for processing
        time.sleep(2.0)

        # Drain visual observations and aggregate audio windows. The audio pipeline emits
        # sliding-window transcripts, so the latest event alone can miss the start
        # of a longer utterance. For video, clip-level validation should report
        # evidence seen anywhere in the clip, not only the final frame.
        visual_events = []
        while True:
            item = qm.get_visual_data(block=False)
            if item is None:
                break
            visual_events.append(item)
        v_count = len(visual_events)
        latest_visual = visual_events[-1] if visual_events else None
        audio_events = []
        while True:
            item = qm.get_audio_event(block=False)
            if item is None:
                break
            audio_events.append(item)
        a_count = len(audio_events)
        latest_audio = audio_events[-1] if audio_events else None
        if latest_audio:
            transcript_parts = []
            for event in audio_events:
                text = (event.transcript or '').strip()
                if text and text not in transcript_parts:
                    transcript_parts.append(text)
            latest_audio.transcript = ' '.join(transcript_parts)

        return {
            'visual_count': v_count,
            'visual': latest_visual,
            'visual_events': visual_events,
            'audio_count': a_count,
            'audio': latest_audio,
        }

    finally:
        vis.stop()
        aud.stop()


def summarize_results(video_file: Path, results: Dict[str, Any]) -> Dict[str, Any]:
    visual = results.get('visual')
    visual_events = results.get('visual_events') or []

    persons_count = 0
    weapons_found = []
    objects_found = []

    if visual:
        persons_count = len(visual.persons) if visual.persons else 0
    for event in visual_events or ([visual] if visual else []):
        if event is None:
            continue
        persons_count = max(persons_count, len(event.persons) if event.persons else 0)
        for w in event.weapons:
            weapons_found.append(getattr(w, 'weapon_type', '').lower())
        for obj in event.processing_stats.get('objects_detected', []):
            objects_found.append(str(obj.get('label', '')).lower())

    audio = results.get('audio')
    return {
        'video_file': video_file.name,
        'visual_events': results.get('visual_count', 0),
        'audio_events': results.get('audio_count', 0),
        'persons_count': persons_count,
        'weapons': sorted(set(weapons_found)),
        'objects': sorted(set(objects_found)),
        'vad_confidence': getattr(audio, 'vad_confidence', 0.0) if audio else 0.0,
        'transcript': getattr(audio, 'transcript', '') if audio else '',
        'intent': getattr(getattr(audio, 'intent', None), 'value', '') if audio else '',
    }


def main():
    videos = sorted(TEST_DIR.glob('test*.mp4'))
    if not videos:
        print('No TEST/test*.mp4 videos found')
        return

    overall = []

    tmpdir = tempfile.mkdtemp(prefix='test_assets_')

    for video_file in videos:
        wav_out = os.path.join(tmpdir, f'{video_file.stem}.wav')
        ok = extract_audio_from_video(str(video_file), wav_out)
        if not ok:
            print('Failed to extract audio for', video_file)
            sidecar_wav = ROOT / 'tests' / 'fixtures' / 'generated' / f'{video_file.stem}.wav'
            if sidecar_wav.exists():
                wav_out = str(sidecar_wav)
                ok = True
                print('Using sidecar audio:', sidecar_wav)
            else:
                print('No sidecar audio found for', video_file)

        print('Running pipelines for', video_file)
        results = run_video_through_pipelines(str(video_file), wav_out)
        report = summarize_results(video_file, results)
        overall.append(report)

    print('\nPipeline observation summary:')
    for r in overall:
        print(r)


if __name__ == '__main__':
    main()
