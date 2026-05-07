"""Play TEST videos through the threat detection UI overlay.

Examples:
  python tools/demo_test_videos.py --all
  python tools/demo_test_videos.py --video TEST/test2.mp4
  python tools/demo_test_videos.py --all --save-output demo_outputs
"""

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None
try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".ultralytics"))

from audio.audio_pipeline import AudioPipeline
from core.config_loader import ConfigLoader
from core.queue_manager import QueueManager
from core.shared_state import SharedStateManager
from display_manager import DisplayManager
from fusion.fusion_manager import FusionManager
from visual.visual_pipeline import VisualPipeline


WINDOW_NAME = "Threat Detection Demo"


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv")


def load_test_videos() -> List[Path]:
    test_dir = REPO_ROOT / "TEST"
    if not test_dir.exists():
        return []
    videos = [
        path
        for path in sorted(test_dir.iterdir())
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return videos


def sidecar_wav_for(video_path: Path) -> Optional[Path]:
    wav_path = REPO_ROOT / "tests" / "fixtures" / "generated" / f"{video_path.stem}.wav"
    return wav_path if wav_path.exists() else None


def extract_audio_from_video(video_path: Path) -> Optional[Path]:
    """Extract mono 16kHz WAV from a video file using ffmpeg if available."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = Path(tmp.name)

    ffmpeg_exe = "ffmpeg"
    if imageio_ffmpeg is not None:
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_exe = "ffmpeg"

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(video_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-vn",
        str(out_path),
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return out_path
    except Exception:
        try:
            out_path.unlink(missing_ok=True)
        except Exception:
            pass
        return None


def load_audio(video_path: Path):
    if sf is None:
        return None, None, None

    wav_path = sidecar_wav_for(video_path)
    temp_wav = None
    if not wav_path:
        temp_wav = extract_audio_from_video(video_path)
        wav_path = temp_wav
    if not wav_path:
        return None, None, None

    data, sample_rate = sf.read(str(wav_path))
    data = np.asarray(data, dtype=np.float32)
    if data.ndim > 1:
        data = data[:, 0]
    return data, sample_rate, temp_wav


def visual_summary_from_latest(visual_data) -> dict:
    if not visual_data:
        return {
            "persons_count": 0,
            "unknown_count": 0,
            "armed_count": 0,
            "weapons_count": 0,
            "unknown_with_weapon": False,
        }

    persons = getattr(visual_data, "persons", []) or []
    unknown_count = 0
    armed_count = 0
    unknown_with_weapon = False
    for person in persons:
        identity = person.get("identity", "UNKNOWN") if isinstance(person, dict) else "UNKNOWN"
        has_weapon = person.get("has_weapon", False) if isinstance(person, dict) else False
        if str(identity).upper() == "UNKNOWN":
            unknown_count += 1
            if has_weapon:
                unknown_with_weapon = True
        if has_weapon:
            armed_count += 1

    return {
        "persons_count": len(persons),
        "unknown_count": unknown_count,
        "armed_count": armed_count,
        "weapons_count": len(getattr(visual_data, "weapons", []) or []),
        "unknown_with_weapon": unknown_with_weapon,
    }


class TestVideoDemo:
    def __init__(self, args):
        self.args = args
        self.queues = QueueManager(maxsize=200)
        self.shared = SharedStateManager()
        self.display = DisplayManager(target_fps=args.display_fps)

        self.visual = VisualPipeline(
            person_model_path=args.person_model,
            weapon_model_path=args.weapon_model,
            person_conf=args.person_conf,
            person_min_area=args.person_min_area,
            target_fps=args.process_fps,
            weapon_detect_every_n_frames=args.weapon_every,
            face_recognize_every_n_frames=args.face_every,
            visual_to_fusion_queue=self.queues.visual_to_fusion_q,
            fusion_to_visual_queue=self.queues.fusion_to_visual_q,
        )
        self.audio = AudioPipeline(
            queue_manager=self.queues,
            vad_model_path=args.vad_model if args.vad_model and Path(args.vad_model).exists() else None,
            device=args.device,
        )
        self.fusion = FusionManager(
            shared_state=self.shared,
            queue_manager=self.queues,
            target_hz=args.fusion_hz,
        )

    def start(self) -> None:
        self.fusion.start()
        self.visual.start()
        if not self.args.no_audio:
            self.audio.start()

    def stop(self) -> None:
        if not self.args.no_audio:
            self.audio.stop()
        self.visual.stop()
        self.fusion.stop()
        self.shared.close()
        cv2.destroyAllWindows()

    def play(self, videos: List[Path]) -> None:
        self.start()
        try:
            for video_path in videos:
                should_continue = self.play_one(video_path)
                if not should_continue:
                    break
        finally:
            self.stop()

    def play_one(self, video_path: Path) -> bool:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Could not open {video_path}")
            return True

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        delay = 1.0 / max(fps * self.args.speed, 1.0)
        audio, audio_sr, temp_wav = (None, None, None) if self.args.no_audio else load_audio(video_path)
        audio_cursor = 0

        writer = None
        if self.args.save_output:
            output_dir = Path(self.args.save_output)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{video_path.stem}_annotated.mp4"
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"Writing annotated output: {output_path}")

        self.queues.clear_all()
        self.shared.reset()
        self.visual.tracker.clear()
        self.visual.latest_visual_data = None
        self.audio.audio_buffer.clear()
        self.audio.samples_submitted = 0
        self.audio._last_transcribed_sample = 0

        print(f"Playing {video_path.name} ({total_frames} frames @ {fps:.1f} fps)")
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = time.time()
            self.visual.submit_frame(frame, timestamp=timestamp)
            self._feed_audio_until(audio, audio_sr, frame_idx / fps, audio_cursor)
            if audio is not None and audio_sr:
                audio_cursor = min(len(audio), int((frame_idx / fps) * audio_sr))

            visual_data = self.visual.latest_visual_data
            try:
                risk = self.shared.get_risk()
                state = self.shared.get_state()
                visual_summary = self.shared.get_visual_summary()
                audio_summary = self.shared.get_audio_summary()
                risk_details = self.shared.get_risk_details()
            except Exception:
                risk = 0.0
                state = "IDLE"
                visual_summary = visual_summary_from_latest(visual_data)
                audio_summary = {}
                risk_details = {}

            if visual_summary.get("persons_count", 0) == 0:
                visual_summary = visual_summary_from_latest(visual_data)

            annotated = self.display.annotate_frame(
                frame,
                risk,
                state,
                visual_summary,
                audio_summary,
                visual_data=visual_data,
                risk_details=risk_details,
                title=f"{video_path.name} | q/esc quit | n next | space pause",
            )

            if writer:
                writer.write(annotated)

            if not self.args.headless:
                cv2.imshow(WINDOW_NAME, annotated)
                key = cv2.waitKey(max(1, int(delay * 1000))) & 0xFF
                if key in (ord("q"), 27):
                    cap.release()
                    if writer:
                        writer.release()
                    return False
                if key == ord("n"):
                    break
                if key == ord(" "):
                    self._pause()
            else:
                time.sleep(delay)

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()
        if temp_wav:
            try:
                temp_wav.unlink(missing_ok=True)
            except Exception:
                pass
        return True

    def _feed_audio_until(self, audio, sample_rate, video_time_sec: float, cursor: int) -> None:
        if audio is None or not sample_rate:
            return

        target = min(len(audio), int(video_time_sec * sample_rate))
        chunk_size = self.audio.CHUNK_SIZE
        pos = cursor
        while pos < target:
            chunk = audio[pos:pos + chunk_size]
            self.audio.submit_audio_chunk(chunk)
            pos += chunk_size

    @staticmethod
    def _pause() -> None:
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key in (ord(" "), ord("n"), ord("q"), 27):
                break


def parse_args():
    visual_defaults = ConfigLoader().load().get("visual", {})
    parser = argparse.ArgumentParser(description="Demonstrate threat detection on TEST videos")
    parser.add_argument("--all", action="store_true", help="Run all supported videos in TEST/")
    parser.add_argument("--video", help="Single video path to run")
    parser.add_argument("--headless", action="store_true", help="Do not open a window")
    parser.add_argument("--save-output", help="Directory for annotated MP4 output")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--process-fps", type=float, default=15.0, help="Visual pipeline processing FPS")
    parser.add_argument("--display-fps", type=float, default=30.0, help="Display FPS")
    parser.add_argument("--fusion-hz", type=float, default=4.0, help="Fusion update frequency")
    parser.add_argument("--person-model", default=visual_defaults.get("person_model", "runs/detect/person_finetune/weights/best.pt"))
    parser.add_argument("--weapon-model", default=visual_defaults.get("weapon_model", "models/weapon_syncrobotic_yolov8n_v2.pt"))
    parser.add_argument("--person-conf", type=float, default=visual_defaults.get("person_conf", 0.60))
    parser.add_argument("--person-min-area", type=int, default=visual_defaults.get("person_min_area", 3500))
    parser.add_argument("--weapon-every", type=int, default=visual_defaults.get("weapon_detect_every_n", 3))
    parser.add_argument("--face-every", type=int, default=visual_defaults.get("face_recognize_every_n", 30))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vad-model", default="models/silero_vad.onnx")
    parser.add_argument("--no-audio", action="store_true", help="Skip sidecar WAV audio playback")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    os.environ.setdefault("THREAT_USE_TRANSFORMER_INTENT", "1")

    if args.video:
        videos = [Path(args.video)]
    else:
        videos = load_test_videos()

    videos = [path if path.is_absolute() else REPO_ROOT / path for path in videos]
    videos = [path for path in videos if path.exists()]
    if not videos:
        print("No supported videos found in TEST/. Add .mp4/.avi/.mov/.mkv/.m4v/.wmv or use --video.")
        return 1

    demo = TestVideoDemo(args)
    demo.play(videos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
