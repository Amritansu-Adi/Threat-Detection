#!/usr/bin/env python3
"""Quick test to see if YOLO model works at all."""

import sys
sys.path.insert(0, '.')

import numpy as np
from ultralytics import YOLO
import cv2

print("Loading YOLO model...")
model = YOLO('yolov8n.pt')
print(f"✓ Model loaded: {model}")

# Create a dummy frame
frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
print(f"✓ Created dummy frame: {frame.shape}")

print("\nRunning inference on dummy frame...")
results = model(frame, conf=0.3, verbose=False)
print(f"✓ Inference complete: {results}")

if results and hasattr(results[0], 'boxes'):
    boxes = results[0].boxes
    print(f"  - Boxes: {boxes}")
    print(f"  - Count: {len(boxes)}")
    print(f"  - Classes: {results[0].boxes.cls if boxes else 'none'}")
else:
    print("  - No boxes found")

print("\n✓ YOLO works!")
