# Fine Tuning Behaviour

This folder contains low-epoch behaviour tuning utilities for large datasets.

## Datasets Detected

- `video_dataset/SCVD/SCVD_converted` (Train/Test with `Normal`, `Violence`, `Weaponized`)
- `audio_dataset/audios_VSD/audios_VSD` (wav files like `angry_*`, `noviolence_*`)

These scripts are intended for behaviour matching and calibration, not full model replacement.

## Parallel Run (Audio + Video)

```bash
python fine_tuning_behaviour/run_parallel_tuning.py --epochs 3 --resume
```

Useful limits for quick tests:

```bash
python fine_tuning_behaviour/run_parallel_tuning.py --epochs 1 --audio-max-files 200 --video-max-videos 50 --video-max-frames 60
```

## Safe Stop / Resume

- Press `Ctrl+C` once while runner is active.
- It creates stop signals, children save checkpoints, then exit.
- Resume with `--resume`.

Checkpoints:

- `fine_tuning_behaviour/checkpoints/audio/latest.npz`
- `fine_tuning_behaviour/checkpoints/video/latest.npz`

Per-epoch snapshots and `metrics.json` are also saved in each checkpoint folder.

## 2GB GPU Note

This framework keeps training lightweight and uses low epochs by default.
Video feature extraction still relies on your existing detector pipeline, so it may be slow on large sets.
Use `--video-max-videos` and `--video-max-frames` during iterative tuning.
