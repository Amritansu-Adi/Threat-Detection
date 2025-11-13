"""Train a weapon-specific YOLOv8 model (ultralytics) on a local dataset.

Usage (PowerShell):
  $env:USE_GPU='1'
  C:/ProgramData/anaconda3/Scripts/conda.exe run -p c:\programing\MachineLearning\person_detect\yolo\.conda --no-capture-output python camera_detection/train_weapon_yolov8.py --data data/weapon_data.yaml --epochs 50 --imgsz 640 --batch 16

Notes:
- Prepare a dataset in YOLO format (images + .txt labels per image).
- Update `data/weapon_data.yaml` with correct train/val paths and class names.
- For faster iteration use small model (`yolov8n.pt`) then increase size.
"""
import argparse
import sys

def parse_args():
    p = argparse.ArgumentParser(description='Train YOLOv8 on weapon dataset')
    p.add_argument('--data', default='data/weapon_data.yaml', help='path to data YAML')
    p.add_argument('--weights', default='yolov8n.pt', help='base weights (or path to custom .pt)')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--device', default=None, help="Device to use (e.g. '0' or 'cpu'). If None, ultralytics chooses automatically.")
    p.add_argument('--project', default='runs/train', help='save results to project/name')
    p.add_argument('--name', default='weapon_train', help='run name inside project')
    p.add_argument('--resume', action='store_true', help='resume training from last checkpoint')
    p.add_argument('--cache', action='store_true', help='cache dataset for faster training')
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except Exception as e:
        print('ultralytics not installed or failed to import:', e)
        print('Install with: pip install ultralytics')
        sys.exit(1)

    print(f'Loading weights: {args.weights}')
    model = YOLO(args.weights)

    train_kwargs = {
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'project': args.project,
        'name': args.name,
        'resume': args.resume,
    }
    if args.device is not None:
        train_kwargs['device'] = args.device
    if args.cache:
        train_kwargs['cache'] = True

    print('Starting training with args:', train_kwargs)
    model.train(**train_kwargs)


if __name__ == '__main__':
    main()
