import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)

from yolo_detect import detect_humans_and_weapons
import cv2

ROOT = Path(__file__).resolve().parents[1]
TEST_IMG = ROOT / 'TEST' / 'KNIFE.webp'
OUT_DIR = ROOT / 'alerts'
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not TEST_IMG.exists():
    print(f"Test image not found: {TEST_IMG}")
    exit(1)

frame = cv2.imread(str(TEST_IMG))
if frame is None:
    print("Failed to load image (cv2 returned None)")
    exit(1)

# Allow skipping Roboflow remote inference (set env NO_WEAPON=1 to skip)
run_weapon = os.environ.get('NO_WEAPON', '0') != '1'
humans, weapons = detect_humans_and_weapons(frame, run_weapon_detection=run_weapon)
print('Humans detected:', humans)
print('Weapons detected:', weapons)

# annotate and save result
for (x1, y1, x2, y2, conf) in humans:
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
for (wx1, wy1, wx2, wy2, wconf, wlabel) in weapons:
    cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), (0,0,255), 2)
    cv2.putText(frame, f"{wlabel} {wconf:.2f}", (wx1, max(10, wy1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

out_path = OUT_DIR / 'test_result.jpg'
cv2.imwrite(str(out_path), frame)
print('Annotated result saved to', out_path)
