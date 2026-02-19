"""
Visual Pipeline Orchestrator for Late Fusion State Machine

This module contains the main VisualPipeline class that coordinates:
- Person detection
- Weapon detection
- Tracking
- Face recognition
- Weapon association
- VisualData emission to fusion queue
"""

import threading
import time
import logging
from typing import Optional, Dict, List, Tuple
from queue import Queue, Full
import cv2
import numpy as np

from .detector import PersonDetector, WeaponDetector, FaceRecognizer
from .tracker import PersonTracker, TrackedPerson
from .weapon_mapper import WeaponMapper
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.data_structures import VisualData, WeaponDetection as CoreWeaponDetection


class VisualPipeline:
    """Main visual processing pipeline for person detection, tracking, and weapon association."""

    def __init__(self,
                 # Detection config
                 person_model_path: str = "yolov8n.pt",
                 weapon_model_path: str = "models/weapon_best.pt",
                 person_conf: float = 0.60,
                 person_min_area: int = 3500,
                 weapon_classes: List[str] = None,

                 # Tracking config
                 tracker_max_age: int = 30,
                 tracker_min_hits: int = 3,
                 tracker_iou_threshold: float = 0.3,

                 # Processing config
                 target_fps: float = 30.0,
                 weapon_detect_every_n_frames: int = 3,
                 face_recognize_every_n_frames: int = 10,

                 # Queue config
                 visual_to_fusion_queue: Optional[Queue] = None,
                 fusion_to_visual_queue: Optional[Queue] = None):

        """
        Initialize visual pipeline.

        Args:
            person_model_path: Path to person detection model
            weapon_model_path: Path to weapon detection model
            person_conf: Person detection confidence threshold
            person_min_area: Minimum person bbox area
            weapon_classes: List of weapon classes to detect
            tracker_max_age: Max frames to keep lost tracks
            tracker_min_hits: Min hits to start tracking
            tracker_iou_threshold: IOU threshold for association
            target_fps: Target processing FPS
            weapon_detect_every_n_frames: Weapon detection frequency
            face_recognize_every_n_frames: Face recognition frequency
            visual_to_fusion_queue: Queue to emit VisualData
            fusion_to_visual_queue: Queue to receive interrupts
        """
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps if target_fps > 0 else 0
        self.weapon_detect_every_n_frames = weapon_detect_every_n_frames
        self.face_recognize_every_n_frames = face_recognize_every_n_frames

        # Queues
        self.visual_to_fusion_q = visual_to_fusion_queue
        self.fusion_to_visual_q = fusion_to_visual_queue

        # Initialize components
        self.person_detector = PersonDetector(
            model_path=person_model_path,
            conf_threshold=person_conf,
            min_area=person_min_area
        )

        self.weapon_detector = WeaponDetector(
            model_path=weapon_model_path,
            weapon_classes=weapon_classes or ['knife', 'scissors', 'gun', 'pistol', 'rifle']
        )

        self.tracker = PersonTracker(
            max_age=tracker_max_age,
            min_hits=tracker_min_hits,
            iou_threshold=tracker_iou_threshold
        )

        self.face_recognizer = FaceRecognizer()
        self.weapon_mapper = WeaponMapper()

        # Processing state
        self.running = False
        self.pipeline_thread: Optional[threading.Thread] = None
        self.frame_idx = 0
        self.last_process_time = 0.0

        # Statistics
        self.stats = {
            'frames_processed': 0,
            'persons_detected': 0,
            'weapons_detected': 0,
            'faces_recognized': 0,
            'processing_fps': 0.0,
            'avg_processing_time': 0.0
        }

        logging.info("VisualPipeline initialized")

    def start(self):
        """Start the visual processing pipeline."""
        if self.running:
            logging.warning("VisualPipeline already running")
            return

        self.running = True
        self.pipeline_thread = threading.Thread(target=self._pipeline_loop, name="VisualPipeline")
        self.pipeline_thread.daemon = True
        self.pipeline_thread.start()
        logging.info("VisualPipeline started")

    def stop(self):
        """Stop the visual processing pipeline."""
        if not self.running:
            return

        self.running = False
        if self.pipeline_thread and self.pipeline_thread.is_alive():
            self.pipeline_thread.join(timeout=2.0)

        # Cleanup
        self.weapon_detector.shutdown()
        logging.info("VisualPipeline stopped")

    def submit_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> bool:
        """
        Submit a frame for processing.

        Args:
            frame: Input frame (BGR format)
            timestamp: Frame timestamp (uses current time if None)

        Returns:
            True if frame was accepted for processing
        """
        if not self.running:
            return False

        # Rate limiting
        current_time = time.time()
        if current_time - self.last_process_time < self.frame_interval:
            return False  # Skip frame to maintain target FPS

        self.last_process_time = current_time

        # Process frame in background
        try:
            threading.Thread(
                target=self._process_frame,
                args=(frame.copy(), timestamp or current_time, self.frame_idx),
                name=f"VisualProcess-{self.frame_idx}",
                daemon=True
            ).start()

            self.frame_idx += 1
            return True

        except Exception as e:
            logging.error(f"Failed to submit frame: {e}")
            return False

    def _pipeline_loop(self):
        """Main pipeline processing loop (currently unused - processing is per-frame)."""
        logging.info("VisualPipeline loop started")
        while self.running:
            time.sleep(0.1)  # Idle when not processing frames
        logging.info("VisualPipeline loop ended")

    def _process_frame(self, frame: np.ndarray, timestamp: float, frame_idx: int):
        """Process a single frame through the visual pipeline."""
        start_time = time.time()

        try:
            # 1. Person Detection
            person_detections = self.person_detector.detect(frame)
            self.stats['persons_detected'] += len(person_detections)

            # Convert to tracker format
            dets = [(p.bbox[0], p.bbox[1], p.bbox[2], p.bbox[3], p.confidence, p.class_name)
                   for p in person_detections]

            # 2. Tracking
            tracked_persons = self.tracker.update(dets, frame_idx)

            # 3. Weapon Detection (every N frames)
            weapon_detections = []
            if frame_idx % self.weapon_detect_every_n_frames == 0:
                person_bboxes = [p.bbox for p in tracked_persons.values()]
                if person_bboxes:
                    self.weapon_detector.detect_async(frame, person_bboxes)

                # Get any completed results
                results = self.weapon_detector.get_results()
                if results:
                    weapon_detections = [(w.bbox[0], w.bbox[1], w.bbox[2], w.bbox[3], w.confidence, w.class_name)
                                       for w in results]
                    self.stats['weapons_detected'] += len(weapon_detections)

            # 4. Face Recognition (every N frames, for persons without locked labels)
            if frame_idx % self.face_recognize_every_n_frames == 0:
                self._perform_face_recognition(frame, tracked_persons)

            # 5. Weapon Association
            persons_for_mapping = []
            for person in tracked_persons.values():
                persons_for_mapping.append({
                    'id': person.id,
                    'bbox': person.bbox,
                    'name': person.get_label()
                })

            weapon_associations = self.weapon_mapper.map_weapons_to_persons(
                frame, weapon_detections, persons_for_mapping
            )

            # Update tracker with weapon associations
            for person_id in tracked_persons.keys():
                has_weapon = person_id in weapon_associations
                self.tracker.update_weapon_association(person_id, has_weapon)

            # 6. Check for fusion interrupts
            self._check_fusion_interrupts()

            # 7. Emit VisualData to fusion queue
            self._emit_visual_data(frame, timestamp, tracked_persons, weapon_associations, weapon_detections)

            # Update stats
            processing_time = time.time() - start_time
            self.stats['frames_processed'] += 1
            self.stats['processing_fps'] = 1.0 / processing_time if processing_time > 0 else 0.0
            self.stats['avg_processing_time'] = (
                (self.stats['avg_processing_time'] * (self.stats['frames_processed'] - 1)) + processing_time
            ) / self.stats['frames_processed']

        except Exception as e:
            logging.exception(f"Frame processing failed: {e}")

    def _perform_face_recognition(self, frame: np.ndarray, tracked_persons: Dict[int, TrackedPerson]):
        """Perform face recognition for persons that need it."""
        for person in tracked_persons.values():
            if person.label_lock_frames <= 0 and person.frames_in_view > 5:
                # Extract face crop
                x1, y1, x2, y2 = person.bbox
                face_img = frame[max(0, y1):y2, max(0, x1):x2]

                if face_img.size > 0:
                    result = self.face_recognizer.recognize(face_img)
                    if result.name:
                        self.tracker.update_person_recognition(person.id, result.name)
                        self.stats['faces_recognized'] += 1

    def _check_fusion_interrupts(self):
        """Check for interrupts from fusion manager."""
        if self.fusion_to_visual_q:
            try:
                while True:  # Drain all pending interrupts
                    interrupt = self.fusion_to_visual_q.get_nowait()
                    self._handle_fusion_interrupt(interrupt)
            except:
                pass  # Queue empty

    def _handle_fusion_interrupt(self, interrupt):
        """Handle interrupt from fusion manager."""
        # For now, just log - future: re-prioritize tracking, focus on specific persons, etc.
        logging.info(f"Received fusion interrupt: {interrupt}")

    def _emit_visual_data(self,
                         frame: np.ndarray,
                         timestamp: float,
                         tracked_persons: Dict[int, TrackedPerson],
                         weapon_associations: Dict[int, List],
                         weapon_detections: List[Tuple]):
        """Emit VisualData to fusion queue."""
        if not self.visual_to_fusion_q:
            return

        try:
            # Convert tracked persons to core format
            persons = []
            for person in tracked_persons.values():
                persons.append({
                    'id': person.id,
                    'bbox': person.bbox,
                    'confidence': person.confidence,
                    'label': person.get_label(),
                    'has_weapon': person.has_weapon,
                    'frames_in_view': person.frames_in_view,
                    'velocity': person.velocity
                })

            # Convert weapon detections to core format
            weapons = []
            for wx1, wy1, wx2, wy2, wconf, wlabel in weapon_detections:
                # Find associated person
                associated_person = None
                for person_id, weapon_list in weapon_associations.items():
                    if (wx1, wy1, wx2, wy2, wconf, wlabel) in weapon_list:
                        associated_person = person_id
                        break

                weapons.append(CoreWeaponDetection(
                    bbox=(wx1, wy1, wx2, wy2),
                    confidence=wconf,
                    class_name=wlabel,
                    associated_person_id=associated_person
                ))

            # Create VisualData
            visual_data = VisualData(
                timestamp=timestamp,
                frame_shape=frame.shape,
                persons=persons,
                weapons=weapons,
                processing_stats=self.stats.copy()
            )

            # Emit to queue (non-blocking)
            self.visual_to_fusion_q.put(visual_data, block=False)

        except Full:
            logging.warning("Visual-to-fusion queue full, dropping frame")
        except Exception as e:
            logging.error(f"Failed to emit visual data: {e}")

    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset pipeline statistics."""
        self.stats = {
            'frames_processed': 0,
            'persons_detected': 0,
            'weapons_detected': 0,
            'faces_recognized': 0,
            'processing_fps': 0.0,
            'avg_processing_time': 0.0
        }