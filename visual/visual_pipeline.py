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
import queue
import cv2
import numpy as np

from .detector import PersonDetector, WeaponDetector, FaceRecognizer
from .tracker import PersonTracker, TrackedPerson
from .weapon_mapper import WeaponMapper
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.data_structures import BoundingBox, VisualData, WeaponDetection as CoreWeaponDetection


class VisualPipeline:
    """Main visual processing pipeline for person detection, tracking, and weapon association."""

    def __init__(self,
                 # Detection config
                 person_model_path: str = "yolov8n.pt",
                 weapon_model_path: str = "camera_detection/models/weapon_best.pt",
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
                 face_recognize_every_n_frames: int = 30,

                 # Queue config
                 visual_to_fusion_queue: Optional[queue.Queue] = None,
                 fusion_to_visual_queue: Optional[queue.Queue] = None):

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
        self.unknown_face_recognize_every_n_frames = max(2, face_recognize_every_n_frames // 6)

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
        if getattr(self.weapon_detector, "using_fallback_model", False):
            self.weapon_detect_every_n_frames = 1
            logging.warning(
                "Custom weapon model unavailable; increasing fallback weapon detection cadence to every frame"
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
        self._last_frame: Optional[np.ndarray] = None  # For display
        self.latest_visual_data: Optional[VisualData] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)  # For frame passing
        self._last_weapon_detections: List[Tuple] = []
        self._last_weapon_frame_idx = -10_000
        self._weapon_id_counter = 0
        self._weapon_tracks: Dict[str, Dict] = {}
        self.weapon_detection_ttl_frames = max(weapon_detect_every_n_frames * 2, 3)
        self.weapon_association_hold_frames = max(weapon_detect_every_n_frames * 2, 6)
        self._last_frame_context: Optional[Tuple[np.ndarray, float, int, Dict[int, TrackedPerson]]] = None
        self.weapon_detection_ttl_frames = max(self.weapon_detect_every_n_frames * 3, 6)
        self.weapon_association_hold_frames = max(self.weapon_detect_every_n_frames * 2, 6)
        self._weapon_track_ttl_frames = max(self.weapon_detection_ttl_frames * 2, 12)

        # Statistics
        self.stats = {
            'frames_processed': 0,
            'frames_submitted': 0,
            'persons_detected': 0,
            'weapons_detected': 0,
            'faces_recognized': 0,
            'objects_detected': [],
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
            logging.warning(f"Frame submit rejected: pipeline not running")
            return False

        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_process_time
        if time_since_last < self.frame_interval:
            return False  # Skip frame to maintain target FPS

        self.last_process_time = current_time

        # Store frame for display
        frame_copy = frame.copy()
        self._last_frame = frame_copy

        # Queue only the latest frame for processing.
        try:
            self.stats['frames_submitted'] += 1
            if self.frame_idx % 30 == 0:
                logging.info(f"Frame {self.frame_idx}: Submitted for processing (total submitted: {self.stats['frames_submitted']})")

            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                    logging.debug("Dropped stale frame before queueing latest frame")
                except queue.Empty:
                    pass

            self._frame_queue.put_nowait((frame_copy, timestamp or current_time, self.frame_idx))

            self.frame_idx += 1
            return True

        except Exception as e:
            logging.error(f"Failed to submit frame: {e}")
            return False

    def _pipeline_loop(self):
        """Main pipeline processing loop."""
        logging.info("VisualPipeline loop started")
        while self.running:
            try:
                frame, timestamp, frame_idx = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                self._emit_completed_weapon_detection_if_idle()
                continue

            self._process_frame(frame, timestamp, frame_idx)
        logging.info("VisualPipeline loop ended")

    def _process_frame(self, frame: np.ndarray, timestamp: float, frame_idx: int):
        """Process a single frame through the visual pipeline."""
        start_time = time.time()

        try:
            # 1. Person Detection
            logging.debug(f"Frame {frame_idx}: Starting person detection on {frame.shape} frame")
            start_detect = time.time()
            try:
                person_detections = self.person_detector.detect(frame)
            except Exception as e:
                logging.error(f"Frame {frame_idx}: detect() FAILED: {e}", exc_info=True)
                person_detections = []
            detect_time = time.time() - start_detect
            logging.debug(f"Frame {frame_idx}: detect() returned {len(person_detections)} persons in {detect_time:.3f}s")
            self.stats['objects_detected'] = [
                {
                    'label': obj.class_name,
                    'confidence': obj.confidence,
                    'bbox': obj.bbox,
                }
                for obj in getattr(self.person_detector, 'last_objects', [])
            ]
            
            self.stats['persons_detected'] += len(person_detections)

            # Convert to tracker format
            dets = [(p.bbox[0], p.bbox[1], p.bbox[2], p.bbox[3], p.confidence, p.class_name)
                   for p in person_detections]

            # 2. Tracking
            tracked_persons = self.tracker.update(dets, frame_idx)
            tracked_persons = self._ensure_persons_for_ui(tracked_persons, person_detections)
            tracked_persons = self._prune_duplicate_tracks(tracked_persons)
            self._last_frame_context = (frame.copy(), timestamp, frame_idx, tracked_persons.copy())
            
            # 3. Weapon Detection (every N frames)
            weapon_detections = self._collect_completed_weapon_detections(frame_idx)
            if frame_idx % self.weapon_detect_every_n_frames == 0:
                person_bboxes = [p.bbox for p in tracked_persons.values()]
                if person_bboxes:
                    self.weapon_detector.detect_async(frame, person_bboxes)

            if not weapon_detections and frame_idx - self._last_weapon_frame_idx <= self.weapon_detection_ttl_frames:
                weapon_detections = self._last_weapon_detections

            # 4. Face Recognition (adaptive cadence for faster unknown->known transitions)
            if self._should_run_face_recognition(frame_idx, tracked_persons):
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
                self.tracker.update_weapon_association(
                    person_id,
                    has_weapon,
                    hold_frames=self.weapon_association_hold_frames,
                )

            # 6. Check for fusion interrupts
            self._check_fusion_interrupts()

            # Emit VisualData to fusion queue
            self._emit_visual_data(frame, timestamp, tracked_persons, weapon_associations, weapon_detections, frame_idx)

            # Update stats
            processing_time = time.time() - start_time
            self.stats['frames_processed'] += 1
            self.stats['processing_fps'] = 1.0 / processing_time if processing_time > 0 else 0.0
            self.stats['avg_processing_time'] = (
                (self.stats['avg_processing_time'] * (self.stats['frames_processed'] - 1)) + processing_time
            ) / self.stats['frames_processed']

        except Exception as e:
            logging.exception(f"Frame processing failed: {e}")

    def _collect_completed_weapon_detections(self, frame_idx: int) -> List[Tuple]:
        """Collect any completed async weapon result without waiting."""
        results = self.weapon_detector.get_results()
        if results is None:
            return []

        weapon_detections = [
            (w.bbox[0], w.bbox[1], w.bbox[2], w.bbox[3], w.confidence, w.class_name)
            for w in results
        ]
        weapon_detections = self._deduplicate_weapon_tuples(weapon_detections)
        self._last_weapon_detections = weapon_detections
        self._last_weapon_frame_idx = frame_idx
        self.stats['weapons_detected'] += len(weapon_detections)
        return weapon_detections

    @staticmethod
    def _bbox_iou_tuple(a: Tuple, b: Tuple) -> float:
        ax1, ay1, ax2, ay2 = a[:4]
        bx1, by1, bx2, by2 = b[:4]
        x1 = max(ax1, bx1)
        y1 = max(ay1, by1)
        x2 = min(ax2, bx2)
        y2 = min(ay2, by2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _deduplicate_weapon_tuples(self, detections: List[Tuple], iou_threshold: float = 0.45) -> List[Tuple]:
        if len(detections) <= 1:
            return detections
        ranked = sorted(detections, key=lambda d: d[4], reverse=True)
        kept: List[Tuple] = []
        for det in ranked:
            if any(self._bbox_iou_tuple(det, k) >= iou_threshold for k in kept):
                continue
            kept.append(det)
        return kept

    def _assign_weapon_track_ids(
        self,
        weapon_detections: List[Tuple],
        frame_idx: int,
    ) -> List[Tuple[str, Tuple]]:
        """Assign persistent IDs to weapons based on IoU matching across frames."""
        if not weapon_detections:
            self._expire_weapon_tracks(frame_idx)
            return []

        assignments: List[Tuple[str, Tuple]] = []
        used_track_ids = set()

        for det in weapon_detections:
            best_track_id = None
            best_iou = 0.0
            for track_id, track in self._weapon_tracks.items():
                if track_id in used_track_ids:
                    continue
                if track.get("label") != det[5]:
                    continue
                iou = self._bbox_iou_tuple(det, track["det"])
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None and best_iou >= 0.45:
                self._weapon_tracks[best_track_id]["det"] = det
                self._weapon_tracks[best_track_id]["last_seen_frame"] = frame_idx
                used_track_ids.add(best_track_id)
                assignments.append((best_track_id, det))
            else:
                self._weapon_id_counter += 1
                new_id = f"W{self._weapon_id_counter}"
                self._weapon_tracks[new_id] = {
                    "det": det,
                    "label": det[5],
                    "last_seen_frame": frame_idx,
                }
                used_track_ids.add(new_id)
                assignments.append((new_id, det))

        self._expire_weapon_tracks(frame_idx)
        return assignments

    def _expire_weapon_tracks(self, frame_idx: int) -> None:
        stale_ids = [
            track_id
            for track_id, track in self._weapon_tracks.items()
            if frame_idx - int(track.get("last_seen_frame", frame_idx)) > self._weapon_track_ttl_frames
        ]
        for track_id in stale_ids:
            self._weapon_tracks.pop(track_id, None)

    def _emit_completed_weapon_detection_if_idle(self):
        """Emit a final update when async weapon detection completes after the last frame."""
        if self._last_frame_context is None:
            return

        frame, timestamp, frame_idx, tracked_persons = self._last_frame_context
        weapon_detections = self._collect_completed_weapon_detections(frame_idx)
        if not weapon_detections:
            return

        persons_for_mapping = [
            {
                'id': person.id,
                'bbox': person.bbox,
                'name': person.get_label(),
            }
            for person in tracked_persons.values()
        ]
        weapon_associations = self.weapon_mapper.map_weapons_to_persons(
            frame, weapon_detections, persons_for_mapping
        )

        for person_id in tracked_persons.keys():
            self.tracker.update_weapon_association(
                person_id,
                person_id in weapon_associations,
                hold_frames=self.weapon_association_hold_frames,
            )

        self._emit_visual_data(
            frame,
            timestamp,
            tracked_persons,
            weapon_associations,
            weapon_detections,
            frame_idx,
        )

    def _perform_face_recognition(self, frame: np.ndarray, tracked_persons: Dict[int, TrackedPerson]):
        """Perform face recognition for persons that need it."""
        for person in tracked_persons.values():
            if person.label_lock_frames <= 0 and person.frames_in_view >= 2:
                # Extract upper-body crop first to avoid picking unrelated background faces.
                x1, y1, x2, y2 = person.bbox
                h = max(1, y2 - y1)
                top_focus = y1 + int(0.58 * h)
                face_img = frame[max(0, y1):max(0, top_focus), max(0, x1):x2]

                if face_img.size > 0:
                    result = self.face_recognizer.recognize(face_img)
                    if result.name:
                        self.tracker.update_person_recognition(person.id, result.name)
                        self.stats['faces_recognized'] += 1

    def _should_run_face_recognition(
        self,
        frame_idx: int,
        tracked_persons: Dict[int, TrackedPerson],
    ) -> bool:
        """Run face recognition more frequently while unknown persons are visible."""
        if not tracked_persons:
            return False

        unknown_visible = any(
            p.get_label().strip().upper() in {"", "UNKNOWN", "UNKNOWN_ENTITY", "UNRECOGNIZED", "NONE"}
            and p.label_lock_frames <= 0
            and p.frames_in_view >= 2
            for p in tracked_persons.values()
        )
        cadence = (
            self.unknown_face_recognize_every_n_frames
            if unknown_visible
            else self.face_recognize_every_n_frames
        )
        cadence = max(1, cadence)
        return frame_idx % cadence == 0

    def _ensure_persons_for_ui(
        self,
        tracked_persons: Dict[int, TrackedPerson],
        person_detections: List,
    ) -> Dict[int, TrackedPerson]:
        """Fallback to detector results when tracking returns no persons.

        This keeps fusion/UI person counts aligned with visible YOLO detections
        even if the tracker backend drops or rejects a frame.
        """
        if tracked_persons or not person_detections:
            return tracked_persons

        fallback_tracks: Dict[int, TrackedPerson] = {}
        for idx, detection in enumerate(person_detections, start=1):
            fallback_tracks[-idx] = TrackedPerson(
                id=-idx,
                bbox=detection.bbox,
                confidence=detection.confidence,
                frames_in_view=1,
                last_seen_frame=self.frame_idx,
            )
        logging.warning(
            "Tracker returned no persons despite %d YOLO detections; using detector fallback for fusion/UI",
            len(fallback_tracks),
        )
        return fallback_tracks

    def _prune_duplicate_tracks(
        self,
        tracked_persons: Dict[int, TrackedPerson],
        iou_threshold: float = 0.55,
        center_dist_threshold_px: float = 45.0,
    ) -> Dict[int, TrackedPerson]:
        """Suppress overlapping tracks that likely represent the same person."""
        if len(tracked_persons) <= 1:
            return tracked_persons

        ranked_tracks = sorted(
            tracked_persons.items(),
            key=lambda item: (item[1].frames_in_view, item[1].confidence),
            reverse=True,
        )

        kept: Dict[int, TrackedPerson] = {}
        kept_boxes = []

        for track_id, person in ranked_tracks:
            person_center = self._bbox_center(person.bbox)
            is_duplicate = False
            for bbox in kept_boxes:
                if self._bbox_iou(person.bbox, bbox) >= iou_threshold:
                    is_duplicate = True
                    break
                kept_center = self._bbox_center(bbox)
                center_dist = ((person_center[0] - kept_center[0]) ** 2 + (person_center[1] - kept_center[1]) ** 2) ** 0.5
                if center_dist <= center_dist_threshold_px:
                    is_duplicate = True
                    break
            if is_duplicate:
                logging.debug("Suppressed duplicate track %s with bbox %s", track_id, person.bbox)
                continue

            kept[track_id] = person
            kept_boxes.append(person.bbox)

        return kept

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

    @staticmethod
    def _bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

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
                         weapon_detections: List[Tuple],
                         frame_idx: int):
        """Emit VisualData to fusion queue."""
        try:
            # Convert tracked persons to core format
            persons = []
            for person in tracked_persons.values():
                persons.append({
                    'id': person.id,
                    'bbox': person.bbox,
                    'confidence': person.confidence,
                    'label': person.get_label(),
                    'identity': person.get_label().upper(),  # Normalize for fusion manager compatibility
                    'has_weapon': person.has_weapon,
                    'frames_in_view': person.frames_in_view,
                    'velocity': person.velocity
                })
            
            # Convert weapon detections to core format
            logging.info(f"Frame {frame_idx}: Weapon detection - found {len(weapon_detections)} weapons")
            weapons = []
            weapon_id_pairs = self._assign_weapon_track_ids(weapon_detections, frame_idx)
            for detection_id, (wx1, wy1, wx2, wy2, wconf, wlabel) in weapon_id_pairs:
                # Find associated person
                associated_person = None
                for person_id, weapon_list in weapon_associations.items():
                    if (wx1, wy1, wx2, wy2, wconf, wlabel) in weapon_list:
                        associated_person = person_id
                        break

                weapons.append(CoreWeaponDetection(
                    bbox=BoundingBox(wx1, wy1, wx2, wy2, confidence=wconf),
                    weapon_type=wlabel,
                    confidence=wconf,
                    detection_id=detection_id,
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
            self.latest_visual_data = visual_data

            # Emit to queue (non-blocking)
            if self.visual_to_fusion_q:
                self.visual_to_fusion_q.put(visual_data, block=False)

        except queue.Full:
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
            'objects_detected': [],
            'processing_fps': 0.0,
            'avg_processing_time': 0.0
        }
