import os
import sys
import time
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import mediapipe as mp

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from facenet_recognition import recognize_face
from email_alert import send_email_alert

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

import torch

logging.basicConfig(level=logging.INFO)

# Config
YOLOV8_WEIGHTS = os.environ.get('YOLOV8_WEIGHTS', os.path.join(os.path.dirname(__file__), 'models', 'weapon_best.pt'))
YOLOV8_CONF = float(os.environ.get('YOLOV8_CONF', 0.25))
YOLOV8_PERSON_CONF = float(os.environ.get('YOLOV8_PERSON_CONF', 0.45))
YOLOV8_PERSON_MIN_AREA = int(os.environ.get('YOLOV8_PERSON_MIN_AREA', 2500))
WEAPON_CLASS_NAMES = [s.strip().lower() for s in os.environ.get('YOLOV8_WEAPON_CLASSES', 'knife,scissors,gun,pistol,rifle').split(',') if s.strip()]
USE_GPU = os.environ.get('USE_GPU', '1') == '1'
WEAPON_DETECT_EVERY_N_FRAMES = int(os.environ.get('WEAPON_DETECT_EVERY_N_FRAMES', 5))
ALERT_SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'alerts')
os.makedirs(ALERT_SNAPSHOT_DIR, exist_ok=True)
# Focused re-inference settings (run on upscaled/processed per-person crops)
# Focused re-inference settings (run on upscaled/processed per-person crops)
YOLOV8_FOCUS_CONF = float(os.environ.get('YOLOV8_FOCUS_CONF', 0.20))
YOLOV8_FOCUS_IMGSZ = int(os.environ.get('YOLOV8_FOCUS_IMGSZ', 1024))
YOLOV8_FOCUS_MIN_CONF = float(os.environ.get('YOLOV8_FOCUS_MIN_CONF', 0.20))
YOLOV8_FOCUS_SYNC = os.environ.get('YOLOV8_FOCUS_SYNC', '0') == '1'
YOLOV8_FOCUS_WORKERS = int(os.environ.get('YOLOV8_FOCUS_WORKERS', 2))
# Fractional padding to expand person bbox when creating focused crop around a person.
# Lowering this reduces how much context is included and makes mapped boxes tighter.
YOLOV8_FOCUS_PAD = float(os.environ.get('YOLOV8_FOCUS_PAD', 0.15))


# MediaPipe setup (hands + face mesh)
mp_hands = mp.solutions.hands.Hands(static_image_mode=False,
                                    max_num_hands=4,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)
mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                          max_num_faces=5,
                                          refine_landmarks=True,
                                          min_detection_confidence=0.5,
                                          min_tracking_confidence=0.5)


# Load YOLOv8 model
if YOLO is None:
    logging.error('ultralytics not installed; install with `pip install ultralytics`')
    model = None
else:
    device = 'cuda' if (USE_GPU and torch.cuda.is_available()) else 'cpu'
    logging.info(f'Loading YOLOv8 weights={YOLOV8_WEIGHTS} device={device}')
    model = YOLO(YOLOV8_WEIGHTS)
    try:
        model.to(device)
    except Exception:
        pass
    # build mappings
    CLASS_NAME_MAP = {v.lower(): k for k, v in model.names.items()} if hasattr(model, 'names') else {}
    PERSON_CLASS_IDS = [CLASS_NAME_MAP.get('person', 0)] if CLASS_NAME_MAP else [0]
    WEAPON_CLASS_IDS = [CLASS_NAME_MAP.get(n) for n in WEAPON_CLASS_NAMES if CLASS_NAME_MAP.get(n) is not None]
    logging.info(f'Person class ids: {PERSON_CLASS_IDS} Weapon class ids: {WEAPON_CLASS_IDS}')

# Async focused inference executor and shared state
executor = ThreadPoolExecutor(max_workers=YOLOV8_FOCUS_WORKERS)
pending_futures = []  # list of futures
latest_refined_weapons = []
latest_refined_frame_idx = -9999


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    if boxAArea + boxBArea - interArea == 0:
        return 0.0
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def clamp_bbox(bbox, max_w, max_h):
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), max_w - 1))
    y1 = max(0, min(int(y1), max_h - 1))
    x2 = max(0, min(int(x2), max_w - 1))
    y2 = max(0, min(int(y2), max_h - 1))
    return x1, y1, x2, y2


def preprocess_crop(img, target_size=None):
    """Apply simple enhancements to a person crop to improve small-object visibility.

    Steps: convert to LAB + CLAHE on L channel, bilateral denoise, optional resize.
    """
    if img is None or img.size == 0:
        return img
    # CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # bilateral filter to reduce noise but keep edges
    enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=75, sigmaSpace=75)
    # optional resize (upscale) to target_size
    if target_size is not None and target_size > 0:
        h, w = enhanced.shape[:2]
        scale = target_size / max(h, w)
        if scale > 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            enhanced = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return enhanced


def focused_infer_on_person(frame, person_bbox):
    """Run a focused YOLOv8 inference on an expanded, preprocessed crop around a person.

    Returns weapon detections mapped to frame coordinates.
    """
    if model is None:
        return []
    x1, y1, x2, y2, _ = person_bbox
    h, w = frame.shape[:2]
    pad_x = int((x2 - x1) * YOLOV8_FOCUS_PAD)
    pad_y = int((y2 - y1) * YOLOV8_FOCUS_PAD)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return []
    proc = preprocess_crop(crop, target_size=YOLOV8_FOCUS_IMGSZ)
    try:
        # run model on processed crop with lower conf and larger imgsz
        res = model(proc, conf=YOLOV8_FOCUS_CONF, imgsz=YOLOV8_FOCUS_IMGSZ)
        r = res[0]
        detections = []
        if hasattr(r, 'boxes') and r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clsids = r.boxes.cls.cpu().numpy().astype(int)
            for (b, c, cid) in zip(boxes, confs, clsids):
                if c < YOLOV8_FOCUS_MIN_CONF:
                    continue
                if int(cid) in WEAPON_CLASS_IDS:
                    # boxes are relative to the processed crop size -> map back
                    bx1, by1, bx2, by2 = [int(v) for v in b.tolist()]
                    # processed crop may have been upscaled; compute scale to original crop
                    orig_h, orig_w = crop.shape[:2]
                    proc_h, proc_w = proc.shape[:2]
                    sx = orig_w / proc_w
                    sy = orig_h / proc_h
                    # map to original crop coords
                    obx1 = int(bx1 * sx)
                    oby1 = int(by1 * sy)
                    obx2 = int(bx2 * sx)
                    oby2 = int(by2 * sy)
                    # map to frame coords
                    wx1 = cx1 + obx1
                    wy1 = cy1 + oby1
                    wx2 = cx1 + obx2
                    wy2 = cy1 + oby2
                    cname = model.names.get(int(cid), str(int(cid))) if hasattr(model, 'names') else str(int(cid))
                    detections.append((wx1, wy1, wx2, wy2, float(c), cname))
        return detections
    except Exception as e:
        logging.warning(f'Focused inference failed: {e}')
        return []


def get_hand_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mp_hands.process(rgb)
    hands = []
    h, w, _ = frame.shape
    if res and getattr(res, 'multi_hand_landmarks', None):
        for hand_landmarks in res.multi_hand_landmarks:
            coords = []
            for lm in hand_landmarks.landmark:
                coords.append((int(lm.x * w), int(lm.y * h)))
            hands.append(coords)
    return hands


def get_face_mesh_boxes(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mp_face.process(rgb)
    faces = []
    h, w, _ = frame.shape
    if res and getattr(res, 'multi_face_landmarks', None):
        for face_landmarks in res.multi_face_landmarks:
            xs = [lm.x for lm in face_landmarks.landmark]
            ys = [lm.y for lm in face_landmarks.landmark]
            x1 = int(min(xs) * w)
            y1 = int(min(ys) * h)
            x2 = int(max(xs) * w)
            y2 = int(max(ys) * h)
            faces.append({'bbox': (x1, y1, x2, y2), 'landmarks': face_landmarks})
    return faces


def detect_humans_and_weapons(frame, run_weapon_detection=True):
    """Run YOLOv8 on the `frame` and return lists of humans and weapons.

    Returns:
      humans: list of (x1,y1,x2,y2,conf)
      weapons: list of (x1,y1,x2,y2,conf,label)
    """
    humans = []
    weapons = []
    if model is None:
        logging.error('YOLOv8 model not available; skipping detection')
        return humans, weapons

    # single shot detection with ultralytics
    try:
        res = model(frame, conf=YOLOV8_CONF)
        r = res[0]
        if hasattr(r, 'boxes') and r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clsids = r.boxes.cls.cpu().numpy().astype(int)
            for (b, c, cid) in zip(boxes, confs, clsids):
                x1, y1, x2, y2 = [int(x) for x in b.tolist()]
                if int(cid) in PERSON_CLASS_IDS:
                    # filter persons by confidence and a minimal box area to reduce false positives
                    area = max(0, (x2 - x1)) * max(0, (y2 - y1))
                    if float(c) < YOLOV8_PERSON_CONF or area < YOLOV8_PERSON_MIN_AREA:
                        continue
                    humans.append((x1, y1, x2, y2, float(c)))
                elif int(cid) in WEAPON_CLASS_IDS and run_weapon_detection:
                    cname = model.names.get(int(cid), str(int(cid))) if hasattr(model, 'names') else str(int(cid))
                    weapons.append((x1, y1, x2, y2, float(c), cname))
    except Exception as e:
        logging.exception(f'YOLOv8 inference failed: {e}')

    # Focused re-inference is handled asynchronously by the main loop by default.
    # If synchronous focused inference is explicitly requested, run it here.
    if YOLOV8_FOCUS_SYNC and run_weapon_detection and len(weapons) == 0 and len(humans) > 0:
        for p in humans:
            try:
                more = focused_infer_on_person(frame, p)
                if more:
                    logging.info(f'Focused inference found {len(more)} weapons for person bbox={p[0:4]}')
                    weapons.extend(more)
            except Exception:
                logging.exception('Error during focused per-person inference')

    return humans, weapons


def main():
    cap = cv2.VideoCapture(0)
    frame_idx = 0
    email_sent = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # clamp any upstream detection coordinates to frame bounds before drawing
        fh, fw = frame.shape[:2]
        run_weapon = (frame_idx % WEAPON_DETECT_EVERY_N_FRAMES) == 0
        humans, weapons = detect_humans_and_weapons(frame, run_weapon_detection=run_weapon)
        hands = get_hand_landmarks(frame)
        face_meshes = get_face_mesh_boxes(frame)

        # recognition
        persons = []
        unknown_detected = False
        for (x1, y1, x2, y2, conf) in humans:
            x1, y1, x2, y2 = clamp_bbox((x1, y1, x2, y2), fw, fh)
            face_img = None
            for fm in face_meshes:
                fx1, fy1, fx2, fy2 = fm['bbox']
                cx = int((fx1 + fx2) / 2)
                cy = int((fy1 + fy2) / 2)
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    face_img = frame[max(0, fy1):fy2, max(0, fx1):fx2]
                    break
            if face_img is None:
                face_img = frame[max(0, y1):y2, max(0, x1):x2]
            name, dist = recognize_face(face_img)
            display_name = name if name else 'Unknown'
            if name is None:
                unknown_detected = True
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)
            persons.append({'bbox': (x1, y1, x2, y2), 'name': display_name, 'dist': dist})
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'Human: {display_name}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Map weapons to persons using hands -> center -> iou -> nearest
        weapon_holders = []
        for (wx1, wy1, wx2, wy2, wconf, wlabel) in weapons:
            wx1, wy1, wx2, wy2 = clamp_bbox((wx1, wy1, wx2, wy2), fw, fh)
            holder = None
            wcenter = bbox_center((wx1, wy1, wx2, wy2))
            mapped = False
            for hand in hands:
                for (hx, hy) in hand:
                    if wx1 <= hx <= wx2 and wy1 <= hy <= wy2:
                        for p in persons:
                            px1, py1, px2, py2 = p['bbox']
                            if px1 <= hx <= px2 and py1 <= hy <= py2:
                                holder = p['name']
                                mapped = True
                                break
                        if mapped:
                            break
                if mapped:
                    break
            if not mapped:
                for p in persons:
                    x1, y1, x2, y2 = p['bbox']
                    if x1 <= wcenter[0] <= x2 and y1 <= wcenter[1] <= y2:
                        holder = p['name']
                        mapped = True
                        break
            if not mapped and len(persons) > 0:
                best_iou = 0.0
                best_name = None
                for p in persons:
                    iou = bbox_iou((wx1, wy1, wx2, wy2), p['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_name = p['name']
                if best_iou > 0.05:
                    holder = best_name
                    mapped = True
            if not mapped and len(persons) > 0:
                min_dist = None
                nearest = None
                for p in persons:
                    pc = bbox_center(p['bbox'])
                    d = (pc[0]-wcenter[0])**2 + (pc[1]-wcenter[1])**2
                    if min_dist is None or d < min_dist:
                        min_dist = d
                        nearest = p['name']
                holder = nearest
            weapon_holders.append(((wx1, wy1, wx2, wy2, wconf, wlabel), holder))
            # Draw the detection box and a filled label background for readability
            cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), (0, 0, 255), 2)
            holder_text = holder if holder is not None else 'None'
            label = f'{wlabel} {wconf:.2f} Holder:{holder_text}'
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            # position label background above the box (clamped to image)
            lx1 = wx1
            ly1 = max(0, wy1 - th - 8)
            lx2 = wx1 + tw + 10
            ly2 = wy1
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 0, 255), -1)
            cv2.putText(frame, label, (lx1 + 5, ly2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Threat logic
        threat_level = None
        high_threat = False
        for w, holder in weapon_holders:
            if holder is None or holder == 'Unknown':
                high_threat = True
                break
        if high_threat:
            threat_level = 'HIGH'
        elif unknown_detected:
            threat_level = 'MINOR'

        if threat_level:
            print(f"THREAT DETECTED: {threat_level}")
            cv2.putText(frame, f'THREAT: {threat_level}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
            if not email_sent:
                holder_lines = []
                for (wx1, wy1, wx2, wy2, wconf, wlabel), holder in weapon_holders:
                    holder_lines.append(f"Weapon={wlabel} conf={wconf:.2f} holder={holder}")
                holders_text = '\n'.join(holder_lines) if holder_lines else 'No weapons detected'
                subject = f"THREAT ALERT: {threat_level}"
                snapshot_name = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                snapshot_path = os.path.join(ALERT_SNAPSHOT_DIR, snapshot_name)
                cv2.imwrite(snapshot_path, frame)
                body = f"A {threat_level} threat has been detected by the system.\n\nDetails:\nUnknown person present: {unknown_detected}\n{holders_text}\nSnapshot: {snapshot_path}"
                send_email_alert(subject, body, to_email=os.environ.get('ALERT_TO_EMAIL', 'amritansuaditya2@gmail.com'))
                email_sent = True
        else:
            email_sent = False

        # Scale display to fit common screens (max 1280x720) while preserving aspect
        max_dw, max_dh = 1280, 720
        fh, fw = frame.shape[:2]
        scale = min(1.0, max_dw / fw, max_dh / fh)
        if scale < 1.0:
            disp = cv2.resize(frame, (int(fw * scale), int(fh * scale)), interpolation=cv2.INTER_AREA)
        else:
            disp = frame
        cv2.imshow('YOLOv8 Human & Weapon Detection + FaceNet', disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

