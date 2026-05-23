#!/usr/bin/env python3
"""Test PersonDetector with real camera frame."""

import sys
sys.path.insert(0, '.')

import cv2
import numpy as np
from visual.detector import PersonDetector

print("Testing PersonDetector with camera...")

# Initialize detector
detector = PersonDetector(conf_threshold=0.3, min_area=2000)
print("✓ PersonDetector initialized")

# Capture frame from camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    sys.exit(1)

ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Cannot capture frame")
    sys.exit(1)

print(f"✓ Captured frame: {frame.shape}")

# Run detection
print("Running person detection...")
detections = detector.detect(frame)

print(f"✓ Detection complete: {len(detections)} persons found")

for i, det in enumerate(detections):
    print(f"  Person {i}: bbox={det.bbox}, conf={det.confidence:.3f}")

if len(detections) == 0:
    print("⚠️  No persons detected - this might be the issue!")
else:
    print("✅ Persons detected successfully!")