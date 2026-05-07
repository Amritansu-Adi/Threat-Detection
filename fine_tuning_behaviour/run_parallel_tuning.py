import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path


def touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("stop", encoding="utf-8")


def remove_if_exists(path: Path):
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Run audio+video behaviour tuning in parallel with safe interrupt handling")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audio-max-files", type=int, default=0)
    parser.add_argument("--video-max-videos", type=int, default=0)
    parser.add_argument("--video-frame-stride", type=int, default=12)
    parser.add_argument("--video-max-frames", type=int, default=120)
    args = parser.parse_args()

    stop_audio = Path("fine_tuning_behaviour/control/STOP_AUDIO")
    stop_video = Path("fine_tuning_behaviour/control/STOP_VIDEO")
    remove_if_exists(stop_audio)
    remove_if_exists(stop_video)

    audio_cmd = [
        sys.executable,
        "fine_tuning_behaviour/train_audio_behavior.py",
        "--epochs",
        str(args.epochs),
        "--max-files",
        str(args.audio_max_files),
    ]
    video_cmd = [
        sys.executable,
        "fine_tuning_behaviour/train_video_behavior.py",
        "--epochs",
        str(args.epochs),
        "--max-videos",
        str(args.video_max_videos),
        "--frame-stride",
        str(args.video_frame_stride),
        "--max-frames",
        str(args.video_max_frames),
    ]
    if args.resume:
        audio_cmd.append("--resume")
        video_cmd.append("--resume")

    procs = [
        subprocess.Popen(audio_cmd),
        subprocess.Popen(video_cmd),
    ]

    interrupted = False

    def _handle_interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True
        touch(stop_audio)
        touch(stop_video)

    signal.signal(signal.SIGINT, _handle_interrupt)

    try:
        while True:
            states = [p.poll() for p in procs]
            if all(s is not None for s in states):
                break
            if interrupted:
                time.sleep(2.0)
                for p in procs:
                    if p.poll() is None:
                        p.terminate()
                break
            time.sleep(1.0)
    finally:
        remove_if_exists(stop_audio)
        remove_if_exists(stop_video)

    codes = [p.poll() for p in procs]
    print("Parallel tuning finished. Exit codes:", codes)


if __name__ == "__main__":
    main()
