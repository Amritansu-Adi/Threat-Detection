# Training weapon-specific YOLOv8 weights

This document explains how to prepare data and train a weapon-specific YOLOv8 model for use in the project.

1) Prepare dataset (YOLO format)
- Directory layout (example):
  - datasets/weapon/images/train/*.jpg
  - datasets/weapon/images/val/*.jpg
  - datasets/weapon/labels/train/*.txt
  - datasets/weapon/labels/val/*.txt

- Label format: one `.txt` per image with lines `class x_center y_center width height` (normalized 0-1).

2) Update data YAML
- Edit `data/weapon_data.yaml` and set `train` and `val` to correct absolute or relative paths.

3) Train locally (recommended: GPU)
- PowerShell example (uses your conda environment):
```powershell
$env:USE_GPU='1'
C:/ProgramData/anaconda3/Scripts/conda.exe run -p c:\programing\MachineLearning\person_detect\yolo\.conda --no-capture-output python camera_detection/train_weapon_yolov8.py --data data/weapon_data.yaml --epochs 100 --imgsz 640 --batch 16
```

If you have a single CUDA GPU and want to set device explicitly, add `--device 0`.

4) Tips to improve performance
- Start from a small model (`yolov8n.pt`) then try `yolov8s.pt`, `yolov8m.pt` if you have more compute.
- Use data augmentation (built-in ultralytics augmentations are used by default).
- Use `--cache` for faster epochs if dataset fits in RAM.

5) Use trained weights in the runtime
- After training, `runs/train/weapon_train/weights/best.pt` (or `last.pt`) is produced. Set env var to point to it before running the detector:
```powershell
$env:YOLOV8_WEIGHTS='runs/train/weapon_train/weights/best.pt'
C:/ProgramData/anaconda3/Scripts/conda.exe run -p c:\programing\MachineLearning\person_detect\yolo\.conda --no-capture-output python camera_detection/yolo_detect.py
```

6) Quick alternatives
- Use Roboflow to create/train a dataset and export YOLOv8 weights (easy UI) then download the `.pt` weights and use them as above.
- Search community hubs for `weapon detection yolov8` pretrained weights, but verify license and trustworthiness before use.

If you want, I can:
- Inspect your dataset structure and fix `data/weapon_data.yaml` paths.
- Kick off a short training run (requires GPU and time) on your machine.
