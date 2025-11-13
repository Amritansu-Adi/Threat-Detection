"""Test runner for YOLOv8 detector on `TEST/KNIFE.webp`.

This will load the weights specified by `YOLOV8_WEIGHTS` env or `--weights` argument.
If `ultralytics` is not installed, it'll raise a clear error explaining how to install it.
"""
import os
from pathlib import Path
from yolov8_detect import detect_image

ROOT = Path(__file__).resolve().parents[1]
TEST_IMG = ROOT / 'TEST' / 'KNIFE.webp'
OUT = ROOT / 'alerts' / 'yolov8_test_result.jpg'

WEIGHTS = os.environ.get('YOLOV8_WEIGHTS', 'yolov8n.pt')

if __name__ == '__main__':
    if not TEST_IMG.exists():
        print('Test image not found:', TEST_IMG)
        raise SystemExit(1)
    print('Running YOLOv8 with weights=', WEIGHTS)
    dets, names = detect_image(WEIGHTS, str(TEST_IMG), conf=0.25, imgsz=640, save_out=str(OUT))
    print('Detections:', dets)
    print('Annotated result saved to', OUT)
