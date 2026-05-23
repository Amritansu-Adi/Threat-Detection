"""
Visual Detection Components for Late Fusion State Machine

This module contains the core visual detection components:
- PersonDetector: YOLOv8 person detection
- WeaponDetector: YOLOv8 weapon detection with focused inference
- FaceRecognizer: FaceNet-based identity recognition
"""

import os
import logging
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / ".ultralytics"))

try:
    from ultralytics import YOLO
except ImportError as e:
    logging.critical(f"Failed to import ultralytics YOLO: {e}")
    raise

try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    from utils.facenet_recognition import recognize_face, load_known_faces
    HAS_FACENET = True
except Exception as e:
    MTCNN = None
    InceptionResnetV1 = None
    def recognize_face(*args, **kwargs):
        return None, 0.0
    def load_known_faces(*args, **kwargs):
        return None
    HAS_FACENET = False
    logging.warning(f"FaceNet unavailable - face recognition disabled: {e}")

try:
    from utils.person_tracker import PersonTracker as ByteTrackPersonTracker
except ImportError:
    ByteTrackPersonTracker = None


@dataclass
class DetectionResult:
    """Result from YOLOv8 detection."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str


@dataclass
class PersonDetection:
    """Person detection result."""
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_name: str = "person"


@dataclass
class WeaponDetection:
    """Weapon detection result."""
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_name: str
    associated_person_id: Optional[int] = None


@dataclass
class FaceRecognitionResult:
    """Face recognition result."""
    name: Optional[str]
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None


class PersonDetector:
    """YOLOv8-based person detection component."""

    def __init__(self,
                 model_path: str = "yolov8n.pt",
                 conf_threshold: float = 0.60,
                 min_area: int = 3500,
                 device: str = "auto"):
        """
        Initialize person detector.

        Args:
            model_path: Path to YOLOv8 model weights
            conf_threshold: Confidence threshold for detections
            min_area: Minimum bounding box area to consider
            device: Device to run on ('cuda', 'cpu', or 'auto')
        """
        self.conf_threshold = conf_threshold
        self.min_area = min_area
        self.device = device if device != "auto" else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # Person class ID (COCO dataset)
        self.person_class_ids = [0]
        self.nms_iou_threshold = 0.5
        self.last_objects: List[DetectionResult] = []

        logging.info(f"PersonDetector initialized: model={model_path}, device={self.device}")

    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        """
        Detect persons in frame.

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of person detections
        """
        try:
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            detections = []
            self.last_objects = []

            logging.info(f"🔍 YOLO raw results: {results is not None}, has boxes: {results and hasattr(results[0], 'boxes')}")
            
            if results and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                boxes = self._to_numpy(results[0].boxes.xyxy)
                confs = self._to_numpy(results[0].boxes.conf)
                class_ids = self._to_numpy(results[0].boxes.cls).astype(int)
                
                logging.info(f"🔍 YOLO boxes: {len(boxes)} detections, classes: {set(class_ids)}")

                for i, (box, conf, class_id) in enumerate(zip(boxes, confs, class_ids)):
                    x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                    area = (x2 - x1) * (y2 - y1)
                    class_name = self.model.names.get(int(class_id), str(int(class_id)))
                    self.last_objects.append(DetectionResult(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(conf),
                        class_id=int(class_id),
                        class_name=class_name,
                    ))
                    logging.info(f"  Box {i}: class_id={class_id}, conf={conf:.3f}, area={area}, threshold={self.conf_threshold}, min_area={self.min_area}")
                    
                    if int(class_id) in self.person_class_ids:
                        if conf >= self.conf_threshold and area > self.min_area:
                            detections.append(PersonDetection(
                                bbox=(x1, y1, x2, y2),
                                confidence=float(conf),
                                class_name=class_name
                            ))
                            logging.info(f"    ✅ PASSED (added to detections)")
                        else:
                            logging.info(f"    ❌ REJECTED (conf check: {conf >= self.conf_threshold}, area check: {area >= self.min_area})")
                    else:
                        logging.info(f"    ❌ NOT PERSON CLASS")

            return self._deduplicate_person_detections(detections)

        except Exception as e:
            logging.error(f"Person detection failed: {e}", exc_info=True)
            self.last_objects = []
            return []

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        """Convert torch tensors, numpy arrays, and mocked tensor-like values to ndarray."""
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    def _deduplicate_person_detections(
        self, detections: List[PersonDetection]
    ) -> List[PersonDetection]:
        """Remove overlapping person boxes that likely represent the same person."""
        if len(detections) <= 1:
            return detections

        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        filtered: List[PersonDetection] = []

        for candidate in detections:
            if all(self._bbox_iou(candidate.bbox, kept.bbox) < self.nms_iou_threshold for kept in filtered):
                filtered.append(candidate)
            else:
                logging.debug("Suppressed duplicate person detection: %s", candidate.bbox)

        return filtered

    @staticmethod
    def _bbox_iou(
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int],
    ) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


class WeaponDetector:
    """YOLOv8-based weapon detection with focused inference."""

    def __init__(self,
                 model_path: str,
                 weapon_classes: List[str] = None,
                 conf_threshold: float = 0.35,
                 min_conf: float = 0.40,
                 focus_imgsz: int = 640,
                 focus_pad: float = 0.40,
                 device: str = "auto",
                 max_workers: int = 2):
        """
        Initialize weapon detector.

        Args:
            model_path: Path to weapon detection model
            weapon_classes: List of weapon class names to detect
            conf_threshold: Confidence threshold for focused inference
            min_conf: Minimum confidence to keep detection
            focus_imgsz: Image size for focused inference
            focus_pad: Padding factor for person crop expansion
            device: Device to run on
            max_workers: Max threads for async inference
        """
        self.conf_threshold = conf_threshold
        self.min_conf = min_conf
        self.focus_imgsz = focus_imgsz
        self.focus_pad = focus_pad
        self.device = device if device != "auto" else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Default weapon classes
        self.weapon_classes = weapon_classes or ['knife', 'scissors', 'gun', 'guns', 'pistol', 'rifle']

        # Load model (handle missing models gracefully)
        self.model = None
        self.class_name_map = {}
        self.weapon_class_ids = []
        
        using_fallback_model = not (model_path and Path(model_path).exists())
        self.using_fallback_model = using_fallback_model
        load_path = model_path if not using_fallback_model else "yolov8n.pt"
        if using_fallback_model:
            self.conf_threshold = max(self.conf_threshold, 0.12)
            self.min_conf = max(self.min_conf, 0.12)
            self.focus_imgsz = max(self.focus_imgsz, 960)
        if load_path and Path(load_path).exists():
            try:
                self.model = YOLO(load_path)
                self.model.to(self.device)
                
                # Build class ID mapping
                self.class_name_map = {v.lower(): k for k, v in self.model.names.items()}
                self.weapon_class_ids = [
                    self.class_name_map.get(name.lower()) for name in self.weapon_classes
                    if self.class_name_map.get(name.lower()) is not None
                ]
                logging.info(f"WeaponDetector initialized: model={load_path}, classes={self.weapon_classes}")
            except Exception as e:
                logging.warning(f"Failed to load weapon model {load_path}: {e}. Weapon detection disabled.")
        else:
            logging.warning(f"Weapon model not found and no COCO fallback is available: {model_path}. Weapon detection disabled.")

        # Async inference executor
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending_futures: List[Future] = []

    def preprocess_crop(self, img: np.ndarray, target_size: Optional[int] = None) -> np.ndarray:
        """Apply enhancements to improve weapon detection in person crops."""
        if img is None or img.size == 0:
            return img

        # CLAHE enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # Bilateral filter for noise reduction
        enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=75, sigmaSpace=75)

        # Optional resize
        if target_size is not None and target_size > 0:
            h, w = enhanced.shape[:2]
            scale = target_size / max(h, w)
            if scale > 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                enhanced = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        return enhanced

    def focused_infer_on_person(self, frame: np.ndarray, person_bbox: Tuple[int, int, int, int]) -> List[WeaponDetection]:
        """Run focused weapon detection on a person crop."""
        if not self.model:
            # Model not loaded, return empty detections
            return []
            
        x1, y1, x2, y2 = person_bbox
        h, w = frame.shape[:2]

        # Expand bbox with padding
        pad_x = int((x2 - x1) * self.focus_pad)
        pad_y = int((y2 - y1) * self.focus_pad)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return []

        # Preprocess crop
        proc = self.preprocess_crop(crop, target_size=self.focus_imgsz)

        try:
            # Run inference
            results = self.model(proc, conf=self.conf_threshold, imgsz=self.focus_imgsz, verbose=False)
            detections = []

            if results and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                boxes = PersonDetector._to_numpy(results[0].boxes.xyxy)
                confs = PersonDetector._to_numpy(results[0].boxes.conf)
                class_ids = PersonDetector._to_numpy(results[0].boxes.cls).astype(int)

                for box, conf, class_id in zip(boxes, confs, class_ids):
                    if conf < self.min_conf or int(class_id) not in self.weapon_class_ids:
                        continue

                    # Map coordinates back to original frame
                    bx1, by1, bx2, by2 = [int(v) for v in box.tolist()]

                    # Scale back to original crop size
                    orig_h, orig_w = crop.shape[:2]
                    proc_h, proc_w = proc.shape[:2]
                    sx = orig_w / proc_w
                    sy = orig_h / proc_h

                    obx1 = int(bx1 * sx)
                    oby1 = int(by1 * sy)
                    obx2 = int(bx2 * sx)
                    oby2 = int(by2 * sy)

                    # Map to frame coordinates
                    wx1 = cx1 + obx1
                    wy1 = cy1 + oby1
                    wx2 = cx1 + obx2
                    wy2 = cy1 + oby2
                    weapon_cx = (wx1 + wx2) / 2.0
                    weapon_cy = (wy1 + wy2) / 2.0
                    margin_x = int((x2 - x1) * 0.10)
                    margin_y = int((y2 - y1) * 0.10)
                    if not (
                        x1 - margin_x <= weapon_cx <= x2 + margin_x
                        and y1 - margin_y <= weapon_cy <= y2 + margin_y
                    ):
                        continue

                    class_name = self.model.names.get(int(class_id), str(int(class_id)))
                    detections.append(WeaponDetection(
                        bbox=(wx1, wy1, wx2, wy2),
                        confidence=float(conf),
                        class_name=class_name
                    ))

            return detections

        except Exception as e:
            logging.warning(f"Focused weapon inference failed: {e}")
            return []

    def detect_async(self, frame: np.ndarray, person_bboxes: List[Tuple[int, int, int, int]]) -> None:
        """Start async weapon detection for multiple persons."""
        if not self.model:
            # Model not loaded, skip detection
            return
            
        if self.pending_futures:
            return  # Already processing

        future = self.executor.submit(self._detect_all_persons, frame.copy(), person_bboxes)
        self.pending_futures.append(future)

    def _detect_all_persons(self, frame: np.ndarray, person_bboxes: List[Tuple[int, int, int, int]]) -> List[WeaponDetection]:
        """Detect weapons for all persons (called async)."""
        all_detections = []
        for bbox in person_bboxes:
            try:
                detections = self.focused_infer_on_person(frame, bbox)
                all_detections.extend(detections)
            except Exception as e:
                logging.exception("Error during focused weapon detection")
        return self._deduplicate_weapon_detections(all_detections)

    @staticmethod
    def _bbox_iou(
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int],
    ) -> float:
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    def _deduplicate_weapon_detections(
        self,
        detections: List[WeaponDetection],
        iou_threshold: float = 0.45,
    ) -> List[WeaponDetection]:
        if len(detections) <= 1:
            return detections
        ranked = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: List[WeaponDetection] = []
        for det in ranked:
            if any(self._bbox_iou(det.bbox, existing.bbox) >= iou_threshold for existing in kept):
                continue
            kept.append(det)
        return kept

    def get_results(self) -> Optional[List[WeaponDetection]]:
        """Get completed async detection results."""
        if not self.pending_futures:
            return None

        # Check for completed futures
        completed = [f for f in self.pending_futures if f.done()]
        if not completed:
            return None

        # Get first completed result
        future = completed[0]
        try:
            result = future.result()
            self.pending_futures.remove(future)
            return result
        except Exception as e:
            logging.exception("Async weapon detection failed")
            self.pending_futures.remove(future)
            return []

    def shutdown(self):
        """Shutdown async executor."""
        self.executor.shutdown(wait=True)


class FaceRecognizer:
    """FaceNet-based face recognition component."""

    def __init__(self, device: str = "auto"):
        """
        Initialize face recognizer.

        Args:
            device: Device to run on
        """
        self.device = device if device != "auto" else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.available = MTCNN is not None and InceptionResnetV1 is not None

        if not self.available:
            self.mtcnn = None
            self.facenet = None
            logging.warning("FaceRecognizer initialized in disabled mode")
            return

        # Initialize models
        self.mtcnn = MTCNN(keep_all=False, min_face_size=20, device=self.device)
        self.facenet = InceptionResnetV1(pretrained='vggface2', device=self.device).eval()

        # Load known faces
        load_known_faces()

        logging.info(f"FaceRecognizer initialized: device={self.device}")

    def recognize(self, face_image: np.ndarray) -> FaceRecognitionResult:
        """
        Recognize face in image.

        Args:
            face_image: Face crop image

        Returns:
            Recognition result
        """
        if face_image is None or face_image.size == 0:
            return FaceRecognitionResult(name=None, confidence=0.0)

        if not self.available:
            return FaceRecognitionResult(name=None, confidence=0.0)

        try:
            name, confidence = recognize_face(face_image, self.mtcnn, self.facenet, self.device)
            return FaceRecognitionResult(
                name=name,
                confidence=confidence if confidence is not None else 0.0
            )
        except Exception as e:
            logging.error(f"Face recognition failed: {e}")
            return FaceRecognitionResult(name=None, confidence=0.0)
