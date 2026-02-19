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
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future

try:
    from ultralytics import YOLO
    from facenet_pytorch import MTCNN, InceptionResnetV1
    from person_tracker import PersonTracker
    from facenet_recognition import recognize_face, load_known_faces
except ImportError as e:
    logging.critical(f"Failed to import visual detection library: {e}")
    raise


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

            if results and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                for box, conf, class_id in zip(boxes, confs, class_ids):
                    if int(class_id) in self.person_class_ids:
                        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                        area = (x2 - x1) * (y2 - y1)

                        if conf >= self.conf_threshold and area >= self.min_area:
                            class_name = self.model.names.get(int(class_id), str(int(class_id)))
                            detections.append(PersonDetection(
                                bbox=(x1, y1, x2, y2),
                                confidence=float(conf),
                                class_name=class_name
                            ))

            return detections

        except Exception as e:
            logging.error(f"Person detection failed: {e}")
            return []


class WeaponDetector:
    """YOLOv8-based weapon detection with focused inference."""

    def __init__(self,
                 model_path: str,
                 weapon_classes: List[str] = None,
                 conf_threshold: float = 0.45,
                 min_conf: float = 0.45,
                 focus_imgsz: int = 640,
                 focus_pad: float = 0.15,
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
        self.weapon_classes = weapon_classes or ['knife', 'scissors', 'gun', 'pistol', 'rifle']

        # Load model
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # Build class ID mapping
        self.class_name_map = {v.lower(): k for k, v in self.model.names.items()}
        self.weapon_class_ids = [
            self.class_name_map.get(name.lower()) for name in self.weapon_classes
            if self.class_name_map.get(name.lower()) is not None
        ]

        # Async inference executor
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending_futures: List[Future] = []

        logging.info(f"WeaponDetector initialized: model={model_path}, classes={self.weapon_classes}")

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
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

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
        return all_detections

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

        try:
            name, confidence = recognize_face(face_image, self.mtcnn, self.facenet, self.device)
            return FaceRecognitionResult(
                name=name,
                confidence=confidence if confidence is not None else 0.0
            )
        except Exception as e:
            logging.error(f"Face recognition failed: {e}")
            return FaceRecognitionResult(name=None, confidence=0.0)