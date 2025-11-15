from ultralytics import YOLO
import cv2
from pathlib import Path

repo = Path(r'C:/programing/MachineLearning/person_detect/yolo')
model_path = str(repo / 'camera_detection' / 'models' / 'weapon_best.pt')
img_path = str(repo / 'TEST' / 'KNIFE.webp')
print('Model:', model_path)
print('Image:', img_path)
model = YOLO(model_path)
frame = cv2.imread(img_path)
print('Frame shape:', frame.shape)
# Full frame inference
res = model(frame, conf=0.25)
r = res[0]
print('\nFull-frame detections:')
if hasattr(r, 'boxes') and r.boxes is not None:
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clsids = r.boxes.cls.cpu().numpy().astype(int)
    for b,cid,c in zip(boxes, clsids, confs):
        print('  box:', b.tolist(), 'class:', cid, 'name:', model.names.get(int(cid)), 'conf:', float(c))
else:
    print('  no boxes')

# Also run a COCO-pretrained YOLOv8 to detect humans (person class) for comparison
print('\nRunning COCO-pretrained YOLOv8 (yolov8n.pt) for human detection:')
try:
    coco = YOLO('yolov8n.pt')
    res_c = coco(frame, conf=0.25)
    rc = res_c[0]
    humans = []
    if hasattr(rc, 'boxes') and rc.boxes is not None:
        boxes_c = rc.boxes.xyxy.cpu().numpy()
        confs_c = rc.boxes.conf.cpu().numpy()
        clsids_c = rc.boxes.cls.cpu().numpy().astype(int)
        for b,cid,c in zip(boxes_c, clsids_c, confs_c):
            name = coco.names.get(int(cid), str(int(cid)))
            if name.lower() == 'person' or int(cid) == 0:
                humans.append((b.tolist(), float(c)))
    if humans:
        print(' Human detections from COCO model:')
        for hb, hc in humans:
            print('  box:', hb, 'conf:', hc)
    else:
        print('  no humans detected by COCO model')
except Exception as e:
    print('  failed to run COCO model for humans:', e)

# Simulate a person bbox near center for focused crop debug
h,w = frame.shape[:2]
px1,py1,px2,py2 = int(w*0.2), int(h*0.2), int(w*0.8), int(h*0.8)
print('\nSimulated person bbox:', (px1,py1,px2,py2))
pad = int((px2-px1)*0.05)
cx1 = max(0, px1 - pad)
cy1 = max(0, py1 - pad)
cx2 = min(w, px2 + pad)
cy2 = min(h, py2 + pad)
crop = frame[cy1:cy2, cx1:cx2]
print('Crop shape:', crop.shape)
# Upscale processed crop to at most 1024 in largest dimension
target = 1024
scale = target / max(1, max(crop.shape[0], crop.shape[1]))
if scale > 1.0:
    proc_w = int(crop.shape[1] * scale)
    proc_h = int(crop.shape[0] * scale)
    proc = cv2.resize(crop, (proc_w, proc_h), interpolation=cv2.INTER_CUBIC)
else:
    proc = crop
print('Processed crop shape:', proc.shape)
res2 = model(proc, conf=0.2, imgsz=max(proc.shape[:2]))
r2 = res2[0]
print('\nFocused-crop detections:')
if hasattr(r2, 'boxes') and r2.boxes is not None:
    boxes2 = r2.boxes.xyxy.cpu().numpy()
    confs2 = r2.boxes.conf.cpu().numpy()
    clsids2 = r2.boxes.cls.cpu().numpy().astype(int)
    for b,cid,c in zip(boxes2, clsids2, confs2):
        print('  box(proc):', b.tolist(), 'class:', cid, 'name:', model.names.get(int(cid)), 'conf:', float(c))
        # map back to frame coords
        orig_h, orig_w = crop.shape[:2]
        proc_h, proc_w = proc.shape[:2]
        sx = orig_w / proc_w
        sy = orig_h / proc_h
        obx1 = int(b[0] * sx)
        oby1 = int(b[1] * sy)
        obx2 = int(b[2] * sx)
        oby2 = int(b[3] * sy)
        wx1 = cx1 + obx1
        wy1 = cy1 + oby1
        wx2 = cx1 + obx2
        wy2 = cy1 + oby2
        print('   mapped to frame:', (wx1,wy1,wx2,wy2))
else:
    print('  no boxes')

print('\nDone')
