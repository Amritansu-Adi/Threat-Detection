import argparse
import json
import signal
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import os
os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".ultralytics"))

from visual.detector import PersonDetector


STOP_REQUESTED = False


def _sigint_handler(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def collect_videos(root: Path):
    videos = []
    for cls in ("Normal", "Violence", "Weaponized"):
        cls_dir = root / "Train" / cls
        if not cls_dir.exists():
            continue
        for p in cls_dir.rglob("*"):
            if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}:
                videos.append((p, cls.lower()))
    return sorted(videos, key=lambda x: str(x[0]))


def extract_clip_features(video_path: Path, detector: PersonDetector, frame_stride: int = 12, max_frames: int = 120):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.zeros(8, dtype=np.float32)

    total = 0
    persons_total = 0
    weapon_like_total = 0
    brightness = []
    motion_vals = []
    prev_gray = None
    weapon_like_labels = {"knife", "scissors", "gun", "pistol", "rifle"}

    frame_idx = 0
    while total < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_stride != 0:
            continue
        total += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness.append(float(np.mean(gray)))
        if prev_gray is not None:
            motion_vals.append(float(np.mean(cv2.absdiff(gray, prev_gray))))
        prev_gray = gray

        detections = detector.detect(frame)
        persons_total += len(detections)
        for obj in detector.last_objects:
            if str(obj.class_name).lower() in weapon_like_labels and obj.confidence >= 0.2:
                weapon_like_total += 1

    cap.release()
    if total == 0:
        return np.zeros(8, dtype=np.float32)

    motion_mean = float(np.mean(motion_vals)) if motion_vals else 0.0
    motion_std = float(np.std(motion_vals)) if motion_vals else 0.0
    bright_mean = float(np.mean(brightness))
    bright_std = float(np.std(brightness))

    return np.array(
        [
            float(total),
            persons_total / total,
            weapon_like_total / total,
            bright_mean,
            bright_std,
            motion_mean,
            motion_std,
            float(persons_total > 0),
        ],
        dtype=np.float32,
    )


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    expz = np.exp(z)
    return expz / np.sum(expz, axis=1, keepdims=True)


def save_checkpoint(path: Path, epoch: int, w: np.ndarray, b: np.ndarray, mean: np.ndarray, std: np.ndarray, label_names: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, epoch=epoch, w=w, b=b, mean=mean, std=std, labels=np.array(label_names))


def load_checkpoint(path: Path):
    data = np.load(path, allow_pickle=True)
    return (
        int(data["epoch"]),
        data["w"],
        data["b"],
        data["mean"],
        data["std"],
        [str(x) for x in data["labels"].tolist()],
    )


def main():
    parser = argparse.ArgumentParser(description="Low-epoch video behaviour tuning with resume support")
    parser.add_argument("--data-root", default="fine_tuning_behaviour/video_dataset/SCVD/SCVD_converted")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="fine_tuning_behaviour/checkpoints/video")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-file", default="fine_tuning_behaviour/control/STOP_VIDEO")
    parser.add_argument("--frame-stride", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=120)
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _sigint_handler)

    root = Path(args.data_root)
    ckpt_dir = Path(args.checkpoint_dir)
    stop_file = Path(args.stop_file)
    stop_file.parent.mkdir(parents=True, exist_ok=True)

    video_items = collect_videos(root)
    if args.max_videos > 0:
        video_items = video_items[: args.max_videos]
    if not video_items:
        raise RuntimeError(f"No training videos found under {root / 'Train'}")

    label_names = sorted({label for _, label in video_items})
    label_to_idx = {k: i for i, k in enumerate(label_names)}
    detector = PersonDetector(model_path="yolov8n.pt", conf_threshold=0.3, min_area=1800)

    x_list = []
    y_list = []
    for p, label in video_items:
        if STOP_REQUESTED or stop_file.exists():
            break
        x_list.append(extract_clip_features(p, detector, args.frame_stride, args.max_frames))
        y_list.append(label_to_idx[label])
    x = np.stack(x_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    rng = np.random.default_rng(42)
    idx = np.arange(len(x))
    rng.shuffle(idx)
    x = x[idx]
    y = y[idx]
    n_train = max(1, int(0.8 * len(x)))
    x_train, x_val = x[:n_train], x[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std if len(x_val) else x_val

    num_features = x_train.shape[1]
    num_classes = len(label_names)
    w = np.zeros((num_features, num_classes), dtype=np.float32)
    b = np.zeros((1, num_classes), dtype=np.float32)
    start_epoch = 0

    latest_ckpt = ckpt_dir / "latest.npz"
    if args.resume and latest_ckpt.exists():
        start_epoch, w_loaded, b_loaded, mean_loaded, std_loaded, label_names_loaded = load_checkpoint(latest_ckpt)
        same_labels = label_names_loaded == label_names
        same_shapes = (
            w_loaded.shape == (num_features, num_classes)
            and b_loaded.shape == (1, num_classes)
            and mean_loaded.shape == (1, num_features)
            and std_loaded.shape == (1, num_features)
        )
        if same_labels and same_shapes:
            w = w_loaded
            b = b_loaded
            mean = mean_loaded
            std = std_loaded
            x_train = (x[:n_train] - mean) / std
            x_val = (x[n_train:] - mean) / std if len(x_val) else x_val
        else:
            # Resume requested but checkpoint does not match current dataset/classes.
            # Start a fresh run with current shapes.
            start_epoch = 0
            w = np.zeros((num_features, num_classes), dtype=np.float32)
            b = np.zeros((1, num_classes), dtype=np.float32)

    y_train_oh = one_hot(y_train, num_classes)
    history = []
    for epoch in range(start_epoch + 1, args.epochs + 1):
        if STOP_REQUESTED or stop_file.exists():
            break
        logits = x_train @ w + b
        probs = softmax(logits)
        grad_logits = (probs - y_train_oh) / len(x_train)
        grad_w = x_train.T @ grad_logits
        grad_b = np.sum(grad_logits, axis=0, keepdims=True)
        w -= args.lr * grad_w
        b -= args.lr * grad_b

        train_pred = np.argmax(probs, axis=1)
        train_acc = float(np.mean(train_pred == y_train))
        if len(x_val):
            val_probs = softmax(x_val @ w + b)
            val_pred = np.argmax(val_probs, axis=1)
            val_acc = float(np.mean(val_pred == y_val))
        else:
            val_acc = train_acc
        history.append({"epoch": epoch, "train_acc": train_acc, "val_acc": val_acc})

        save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.npz", epoch, w, b, mean, std, label_names)
        save_checkpoint(latest_ckpt, epoch, w, b, mean, std, label_names)

    out_metrics = ckpt_dir / "metrics.json"
    out_metrics.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Video tuning completed/paused. Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()
