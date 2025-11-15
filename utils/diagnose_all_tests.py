from ultralytics import YOLO
import cv2
from pathlib import Path

repo = Path(r'C:/programing/MachineLearning/person_detect/yolo')
model_path = str(repo / 'camera_detection' / 'models' / 'weapon_best.pt')
print('Model:', model_path)
model = YOLO(model_path)

test_dir = repo / 'TEST'
imgs = [p for p in test_dir.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.webp')]
if not imgs:
    print('No test images found in', test_dir)

for img in imgs:
    print('\n===', img.name, '===')
    frame = cv2.imread(str(img))
    if frame is None:
        print('Failed to read', img)
        continue
    h,w = frame.shape[:2]
    # full-frame inference
    res = model(frame, conf=0.25)
    r = res[0]
    if hasattr(r, 'boxes') and r.boxes is not None:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clsids = r.boxes.cls.cpu().numpy().astype(int)
        for b,cid,c in zip(boxes, clsids, confs):
            name = model.names.get(int(cid), str(int(cid)))
            print('Full box:', [float(x) for x in b.tolist()], 'class:', name, 'conf:', float(c))
    else:
        print('Full-frame: no boxes')
    # focused refinement: for each detected full-frame weapon, run a small local crop around it and re-infer
    if hasattr(r, 'boxes') and r.boxes is not None:
        for idx, b in enumerate(boxes):
            x1,y1,x2,y2 = [int(v) for v in b.tolist()]
            # expand a small region around detected box
            padx = int((x2-x1)*0.25)
            pady = int((y2-y1)*0.25)
            cx1 = max(0, x1 - padx)
            cy1 = max(0, y1 - pady)
            cx2 = min(w, x2 + padx)
            cy2 = min(h, y2 + pady)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            # resize to at most 640 for speed
            target = 640
            scale = target / max(1, max(crop.shape[0], crop.shape[1]))
            if scale > 1.0:
                proc = cv2.resize(crop, (int(crop.shape[1]*scale), int(crop.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)
            else:
                proc = crop
            res2 = model(proc, conf=0.15)
            r2 = res2[0]
            if hasattr(r2, 'boxes') and r2.boxes is not None and len(r2.boxes) > 0:
                boxes2 = r2.boxes.xyxy.cpu().numpy()
                confs2 = r2.boxes.conf.cpu().numpy()
                clsids2 = r2.boxes.cls.cpu().numpy().astype(int)
                for b2,cid2,c2 in zip(boxes2, clsids2, confs2):
                    name2 = model.names.get(int(cid2), str(int(cid2)))
                    # map back to frame coords
                    orig_h, orig_w = crop.shape[:2]
                    proc_h, proc_w = proc.shape[:2]
                    sx = orig_w / proc_w
                    sy = orig_h / proc_h
                    obx1 = int(b2[0] * sx)
                    oby1 = int(b2[1] * sy)
                    obx2 = int(b2[2] * sx)
                    oby2 = int(b2[3] * sy)
                    wx1 = cx1 + obx1
                    wy1 = cy1 + oby1
                    wx2 = cx1 + obx2
                    wy2 = cy1 + oby2
                    print(' Refined box for detection', idx, ':', (wx1,wy1,wx2,wy2), 'class:', name2, 'conf:', float(c2))
            else:
                print('  No refined boxes for detection', idx)
print('\nAll done')
