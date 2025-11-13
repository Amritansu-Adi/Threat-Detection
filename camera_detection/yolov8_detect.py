"""Simple YOLOv8 local inference wrapper.

This script provides a small API to run YOLOv8 inference using the
`ultralytics` package. It is written to be robust: it checks for the
dependency and raises an informative error if not installed.

Usage examples:
  - Python API: from yolov8_detect import detect_image
  - CLI/test: python camera_detection/yolov8_detect.py --weights yolov8n.pt --source TEST/KNIFE.webp

Notes:
  - For weapon detection you must supply weights that include weapon classes
    (COCO weights do not contain weapon labels). Pass a custom weights file
    via `--weights` or set `YOLOV8_WEIGHTS` env.
"""
import os
import cv2
import numpy as np
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception as e:
    raise ImportError(
        "ultralytics package is required. Install with: `pip install ultralytics`\n"
        "Or add it to your conda env and re-run. Original error: " + str(e)
    )


def annotate_and_save(image, detections, out_path):
    # detections: list of (x1,y1,x2,y2,conf,class_id,class_name)
    for (x1, y1, x2, y2, conf, cid, cname) in detections:
        color = (0, 255, 0) if cname == 'person' else (0, 0, 255)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(image, f"{cname} {conf:.2f}", (int(x1), max(10, int(y1)-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(str(out_path), image)


def detect_image(weights, image_path, conf=0.25, imgsz=640, classes=None, save_out=None):
    """Run YOLOv8 on a single image.

    - weights: path to weights (local .pt) or model alias (e.g., 'yolov8n.pt')
    - image_path: path to input image
    - classes: optional list of class indices to keep (None = all)
    - save_out: optional path to write annotated result

    Returns: list of detections as tuples and the model's class-name map
    """
    model = YOLO(weights)
    res = model(str(image_path), imgsz=imgsz, conf=conf)
    r = res[0]
    names = model.names if hasattr(model, 'names') else {}
    dets = []
    if hasattr(r, 'boxes') and r.boxes is not None:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clsids = r.boxes.cls.cpu().numpy().astype(int)
        for (b, c, cid) in zip(boxes, confs, clsids):
            x1, y1, x2, y2 = b.tolist()
            if classes is not None and int(cid) not in classes:
                continue
            cname = names.get(int(cid), str(int(cid)))
            dets.append((x1, y1, x2, y2, float(c), int(cid), cname))

    if save_out:
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Failed to read image {image_path}")
        annotate_and_save(img, dets, save_out)

    return dets, names


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', '-w', default=os.environ.get('YOLOV8_WEIGHTS', 'yolov8n.pt'))
    parser.add_argument('--source', '-s', default=str(Path(__file__).resolve().parents[1] / 'TEST' / 'KNIFE.webp'))
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--out', '-o', default=str(Path(__file__).resolve().parents[1] / 'alerts' / 'yolov8_test_result.jpg'))
    parser.add_argument('--classes', '-c', nargs='+', type=int, help='class ids to keep')
    args = parser.parse_args()

    dets, names = detect_image(args.weights, args.source, conf=args.conf, imgsz=args.imgsz, classes=args.classes, save_out=args.out)
    print('Detections:', dets)
    print('Class map sample:', {k: names[k] for k in list(names)[:10]})
