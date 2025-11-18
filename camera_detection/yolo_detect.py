import os
import sys
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2
import mediapipe as mp
import torch

# Add project directories to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

try:
    from ultralytics import YOLO
    from facenet_pytorch import MTCNN, InceptionResnetV1
    from person_tracker import PersonTracker
    from facenet_recognition import recognize_face, load_known_faces
    from email_alert import send_email_alert
    from drawing import draw_weapon_box, draw_person_box, draw_threat_level
except ImportError as e:
    logging.critical(f"Failed to import a critical library: {e}")
    sys.exit(1)

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

YOLOV8_PERSON_WEIGHTS = os.environ.get('YOLOV8_PERSON_WEIGHTS', 'yolov8n.pt')
YOLOV8_WEAPON_WEIGHTS = os.environ.get('YOLOV8_WEAPON_WEIGHTS', os.path.join(os.path.dirname(__file__), '..', 'models', 'weapon_best.pt'))
YOLOV8_PERSON_CONF = float(os.environ.get('YOLOV8_PERSON_CONF', 0.60))
YOLOV8_PERSON_MIN_AREA = int(os.environ.get('YOLOV8_PERSON_MIN_AREA', 3500))
WEAPON_CLASS_NAMES = [s.strip().lower() for s in os.environ.get('YOLOV8_WEAPON_CLASSES', 'knife,scissors,gun,pistol,rifle').split(',') if s.strip()]
USE_GPU = os.environ.get('USE_GPU', '1') == '1'
WEAPON_DETECT_EVERY_N_FRAMES = int(os.environ.get('WEAPON_DETECT_EVERY_N_FRAMES', 3))
ALERT_SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'alerts')
os.makedirs(ALERT_SNAPSHOT_DIR, exist_ok=True)

YOLOV8_FOCUS_CONF = float(os.environ.get('YOLOV8_FOCUS_CONF', 0.45))
YOLOV8_FOCUS_IMGSZ = int(os.environ.get('YOLOV8_FOCUS_IMGSZ', 640))
YOLOV8_FOCUS_MIN_CONF = float(os.environ.get('YOLOV8_FOCUS_MIN_CONF', 0.45))
YOLOV8_FOCUS_PAD = float(os.environ.get('YOLOV8_FOCUS_PAD', 0.15))
WEAPON_PERSON_OVERLAP_THRESH = float(os.environ.get('WEAPON_PERSON_OVERLAP_THRESH', 0.85))

# --- Model and Device Initialization ---
device = 'cuda' if (USE_GPU and torch.cuda.is_available()) else 'cpu'
logging.info(f"Main processing device: {device}")

# Load YOLOv8 models
logging.info(f'Loading Person Model: weights={YOLOV8_PERSON_WEIGHTS} device={device}')
person_model = YOLO(YOLOV8_PERSON_WEIGHTS)
person_model.to(device)

logging.info(f'Loading Weapon Model: weights={YOLOV8_WEAPON_WEIGHTS} device={device}')
weapon_model = YOLO(YOLOV8_WEAPON_WEIGHTS)
weapon_model.to(device)

CLASS_NAME_MAP = {v.lower(): k for k, v in weapon_model.names.items()}
WEAPON_CLASS_IDS = [CLASS_NAME_MAP.get(n) for n in WEAPON_CLASS_NAMES if CLASS_NAME_MAP.get(n) is not None]
PERSON_CLASS_IDS = [0]
logging.info(f'Weapon class ids from weapon model: {WEAPON_CLASS_IDS}')

# Load Face Recognition Models
# Use the single-face version of MTCNN for live recognition
mtcnn_face_detector = MTCNN(keep_all=False, min_face_size=20, device=device)
facenet_resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
load_known_faces()

# MediaPipe setup
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=4, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Async focused inference executor
executor = ThreadPoolExecutor(max_workers=2)
pending_futures = []
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
    if weapon_model is None:
        return []
    # The person_bbox is a 6-tuple (x1, y1, x2, y2, conf, cname), we only need the coordinates.
    x1, y1, x2, y2 = person_bbox[:4]
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
        # run weapon_model on processed crop with lower conf and larger imgsz
        res = weapon_model(proc, conf=YOLOV8_FOCUS_CONF, imgsz=YOLOV8_FOCUS_IMGSZ)
        r = res[0]
        detections = []
        if hasattr(r, 'boxes') and r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clsids = r.boxes.cls.cpu().numpy().astype(int)
            for (b, c, cid) in zip(boxes, confs, clsids):
                if c < YOLOV8_FOCUS_MIN_CONF:
                    continue
                # We only care about weapon classes from the specialized model
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
                    cname = weapon_model.names.get(int(cid), str(int(cid))) if hasattr(weapon_model, 'names') else str(int(cid))
                    detections.append((wx1, wy1, wx2, wy2, float(c), cname))
        return detections
    except Exception as e:
        logging.warning(f'Focused weapon inference failed: {e}')
        return []


def focused_infer_for_humans(frame, humans):
    """Run focused_infer_on_person for a list of humans and aggregate results."""
    all_dets = []
    for p in humans:
        try:
            more = focused_infer_on_person(frame, p)
            if more:
                all_dets.extend(more)
        except Exception:
            logging.exception('Error during batch focused per-person inference')
    return all_dets


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


def map_weapon_to_person(persons, hands, weapon_bbox):
    """Maps a weapon to a person using a cascading logic of hands, bbox containment, IoU, and proximity.
    
    Args:
        persons (list): A list of dictionaries, where each dict contains 'id', 'bbox', and 'name'.
        hands (list): A list of hand landmarks.
        weapon_bbox (tuple): The bounding box of the weapon.

    Returns:
        int or None: The ID of the associated person, or None if no association is made.
    """
    wx1, wy1, wx2, wy2 = weapon_bbox
    wcenter = bbox_center(weapon_bbox)

    # Weapon area (avoid zero)
    w_area = max(1, (wx2 - wx1) * (wy2 - wy1))

    # 1. Hand Overlap (highest priority)
    for hand in hands:
        for (hx, hy) in hand:
            if wx1 <= hx <= wx2 and wy1 <= hy <= wy2:
                for p in persons:
                    px1, py1, px2, py2 = p['bbox']
                    if px1 <= hx <= px2 and py1 <= hy <= py2:
                        return p['id']

    # 2. High overlap
    best_overlap = 0.0
    best_person_id = None
    for p in persons:
        px1, py1, px2, py2 = p['bbox']
        ix1, iy1, ix2, iy2 = max(wx1, px1), max(wy1, py1), min(wx2, px2), min(wy2, py2)
        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        overlap_ratio = inter_area / float(w_area)
        if overlap_ratio > best_overlap:
            best_overlap = overlap_ratio
            best_person_id = p['id']

    if best_overlap >= WEAPON_PERSON_OVERLAP_THRESH:
        return best_person_id

    # 3. Bbox center containment
    for p in persons:
        x1, y1, x2, y2 = p['bbox']
        if x1 <= wcenter[0] <= x2 and y1 <= wcenter[1] <= y2:
            return p['id']

    # 4. Best IoU
    if len(persons) > 0:
        best_iou = 0.0
        best_person_id_iou = None
        for p in persons:
            iou = bbox_iou(weapon_bbox, p['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_person_id_iou = p['id']
        if best_iou > 0.05:
            return best_person_id_iou

    # 5. Nearest person by centroid
    if len(persons) > 0:
        min_dist = float('inf')
        nearest_person_id = None
        for p in persons:
            pc = bbox_center(p['bbox'])
            d = (pc[0] - wcenter[0])**2 + (pc[1] - wcenter[1])**2
            if d < min_dist:
                min_dist = d
                nearest_person_id = p['id']
        return nearest_person_id

    return None


def detect_humans(frame):
    """Run YOLOv8 inference on a single frame to detect humans."""
    try:
        res = person_model(frame, conf=YOLOV8_PERSON_CONF)
        r = res[0]
        detections = []
        if hasattr(r, 'boxes') and r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clsids = r.boxes.cls.cpu().numpy().astype(int)
            for (b, c, cid) in zip(boxes, confs, clsids):
                if int(cid) in PERSON_CLASS_IDS:
                    x1, y1, x2, y2 = [int(v) for v in b.tolist()]
                    area = (x2 - x1) * (y2 - y1)
                    if c >= YOLOV8_PERSON_CONF and area >= YOLOV8_PERSON_MIN_AREA:
                        cname = person_model.names.get(int(cid), str(int(cid)))
                        detections.append((x1, y1, x2, y2, float(c), cname))
        return detections
    except Exception as e:
        logging.error(f"Error during person detection: {e}")
        return []


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.critical("Could not open webcam. Exiting.")
        return

    frame_idx = 0
    email_sent = False
    last_known_threat_level = None
    tracker = PersonTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        fh, fw = frame.shape[:2]
        
        humans = detect_humans(frame)
        tracked_persons_dict = tracker.update(humans, frame_idx)
        tracked_persons = list(tracked_persons_dict.values()) if isinstance(tracked_persons_dict, dict) else tracked_persons_dict

        run_weapon_detection_this_frame = (frame_idx % WEAPON_DETECT_EVERY_N_FRAMES == 0) and tracked_persons

        global pending_futures, latest_refined_weapons, latest_refined_frame_idx
        if pending_futures:
            done_futures = [f for f in pending_futures if f.done()]
            for f in done_futures:
                try:
                    res = f.result()
                    if res:
                        latest_refined_weapons = res
                        latest_refined_frame_idx = frame_idx
                        logging.info(f'Integrated {len(res)} async focused weapon detections at frame {frame_idx}')
                except Exception:
                    logging.exception('Async focused task failed')
                pending_futures.remove(f)

        if run_weapon_detection_this_frame and not pending_futures:
            person_bboxes = [p.bbox for p in tracked_persons]
            future = executor.submit(focused_infer_for_humans, frame.copy(), person_bboxes)
            pending_futures.append(future)

        current_weapons_to_draw = latest_refined_weapons if (frame_idx - latest_refined_frame_idx) <= WEAPON_DETECT_EVERY_N_FRAMES else []
        
        hands = get_hand_landmarks(frame)
        
        persons_for_mapping = []
        unknown_person_present = False
        known_person_present = False

        for person in tracked_persons:
            x1, y1, x2, y2 = [int(c) for c in person.bbox]
            face_img = frame[max(0, y1):y2, max(0, x1):x2]
            
            if person.label_lock_frames <= 0 and face_img.size > 0:
                # Attempt to re-identify the person
                name, _ = recognize_face(face_img, mtcnn_face_detector, facenet_resnet, device)
                # Only update if a face was actually recognized.
                # If name is None, we do nothing, preserving the last known identity.
                if name:
                    tracker.update_person_recognition(person.id, name)
                # If recognition fails (no face visible), we no longer reset to "Unknown".
                # The person keeps their last known label until a new face is seen and identified.
            
            display_name = person.get_label()
            if display_name == "Unknown":
                unknown_person_present = True
            else:
                known_person_present = True

            persons_for_mapping.append({'bbox': (x1, y1, x2, y2), 'name': display_name, 'id': person.id})
            draw_person_box(frame, person)

        associated_person_ids = set()
        unassociated_weapons = []
        for (wx1, wy1, wx2, wy2, wconf, wlabel) in current_weapons_to_draw:
            holder_id = map_weapon_to_person(persons_for_mapping, hands, (wx1, wy1, wx2, wy2))
            if holder_id is not None:
                associated_person_ids.add(holder_id)
            else:
                unassociated_weapons.append((wx1, wy1, wx2, wy2, wconf, wlabel))

        for person in tracked_persons:
            tracker.update_weapon_association(person.id, person.id in associated_person_ids)
            # draw_person_box(frame, person) # Already drawn above

        for person in tracked_persons:
            if person.has_weapon:
                for (wx1, wy1, wx2, wy2, wconf, wlabel) in current_weapons_to_draw:
                    holder_id = map_weapon_to_person(persons_for_mapping, hands, (wx1, wy1, wx2, wy2))
                    if holder_id == person.id:
                        draw_weapon_box(frame, (wx1, wy1, wx2, wy2), f'{wlabel} {wconf:.2f}', person.get_label())
                        break

        for (wx1, wy1, wx2, wy2, wconf, wlabel) in unassociated_weapons:
            draw_weapon_box(frame, (wx1, wy1, wx2, wy2), f'{wlabel} {wconf:.2f}', None)

        # Refined Threat Logic
        threat_level = 'NO_THREAT'
        unknown_with_weapon = any(p.has_weapon and p.get_label() == 'Unknown' for p in tracked_persons)
        known_with_weapon = any(p.has_weapon and p.get_label() != 'Unknown' for p in tracked_persons)
        person_with_weapon_present = unknown_with_weapon or known_with_weapon

        if unknown_with_weapon:
            threat_level = 'HIGH'
        elif known_with_weapon:
            threat_level = 'MINOR'
        elif unknown_person_present and not known_person_present and not person_with_weapon_present:
            threat_level = 'MINOR'
        
        # Display Threat & Send Alerts
        display_threat = threat_level
        if threat_level != 'NO_THREAT':
            last_known_threat_level = threat_level
        elif not unknown_person_present and not person_with_weapon_present and not unassociated_weapons:
             last_known_threat_level = None # Reset if clear

        draw_threat_level(frame, display_threat)
        
        alert_triggered = False
        if display_threat != 'NO_THREAT' and not email_sent:
            # Check conditions for sending an email
            send_now = False
            for p in tracked_persons:
                # Condition 1: Person is identified and has a weapon.
                if p.has_weapon and p.get_label() != "Unknown":
                    send_now = True
                    break
                # Condition 2: Person has a weapon, is "Unknown", and has been in view for > 20 frames.
                if p.has_weapon and p.get_label() == "Unknown" and p.frames_in_view > 20:
                    send_now = True
                    break
            
            if send_now:
                alert_triggered = True

        if alert_triggered:
            holder_lines = [f"Holder={p.get_label()} (ID: {p.id})" for p in tracked_persons if p.has_weapon]
            holders_text = '\n'.join(holder_lines) if holder_lines else 'No confirmed weapon holders.'
            if unassociated_weapons:
                holders_text += "\nUnassociated weapons detected."

            subject = f"THREAT ALERT: {display_threat}"
            snapshot_name = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            snapshot_path = os.path.join(ALERT_SNAPSHOT_DIR, snapshot_name)
            cv2.imwrite(snapshot_path, frame)
            send_email_alert(subject, holders_text, snapshot_path, to_email='amritansuaditya1@gmail.com')
            email_sent = True
        elif display_threat == 'NO_THREAT':
            email_sent = False

        frame_idx += 1
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown()

if __name__ == "__main__":
    main()

