"""
Visual Tracking Components for Late Fusion State Machine

This module contains tracking components:
- PersonTracker: Multi-object tracking with identity persistence
- TrackedPerson: Individual person tracking state
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import time

try:
    from person_tracker import PersonTracker as ByteTrackPersonTracker
except ImportError:
    logging.warning("ByteTrack PersonTracker not available, using stub")
    ByteTrackPersonTracker = None


@dataclass
class TrackedPerson:
    """Individual tracked person with identity and state."""
    id: int
    bbox: Tuple[int, int, int, int]  # Current bbox
    confidence: float
    label: str = "Unknown"  # Identity name
    has_weapon: bool = False
    frames_in_view: int = 0
    last_seen_frame: int = 0
    label_lock_frames: int = 0  # Frames until next recognition attempt
    velocity: Tuple[float, float] = (0.0, 0.0)  # Movement velocity
    track_history: deque = field(default_factory=lambda: deque(maxlen=30))  # Position history

    def get_label(self) -> str:
        """Get current display label."""
        return self.label

    def update_position(self, new_bbox: Tuple[int, int, int, int], frame_idx: int):
        """Update position and calculate velocity."""
        if self.track_history:
            prev_bbox = self.track_history[-1]
            prev_center = self._bbox_center(prev_bbox)
            curr_center = self._bbox_center(new_bbox)
            self.velocity = (
                curr_center[0] - prev_center[0],
                curr_center[1] - prev_center[1]
            )

        self.track_history.append(new_bbox)
        self.bbox = new_bbox
        self.last_seen_frame = frame_idx
        self.frames_in_view += 1

    def update_recognition(self, name: str, lock_frames: int = 30):
        """Update identity recognition result."""
        if name and name != "Unknown":
            self.label = name
            self.label_lock_frames = lock_frames

    def decrement_lock(self):
        """Decrement label lock counter."""
        if self.label_lock_frames > 0:
            self.label_lock_frames -= 1

    def is_unknown(self) -> bool:
        """Check if person is unknown."""
        return self.label == "Unknown"

    def has_been_visible_long_enough(self, min_frames: int = 20) -> bool:
        """Check if person has been visible for minimum frames."""
        return self.frames_in_view >= min_frames

    @staticmethod
    def _bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Calculate bbox center."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class PersonTracker:
    """Multi-object tracker for persons with identity persistence."""

    def __init__(self,
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3):
        """
        Initialize person tracker.

        Args:
            max_age: Maximum frames to keep lost tracks
            min_hits: Minimum hits to start tracking
            iou_threshold: IOU threshold for track association
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        # Use ByteTrack if available, otherwise fallback
        if ByteTrackPersonTracker is not None:
            self.tracker = ByteTrackPersonTracker()
            self.use_bytetrack = True
        else:
            self.tracker = None
            self.use_bytetrack = False
            self._fallback_tracks: Dict[int, TrackedPerson] = {}
            self._next_id = 1

        self.active_tracks: Dict[int, TrackedPerson] = {}
        self.frame_idx = 0

        logging.info(f"PersonTracker initialized: use_bytetrack={self.use_bytetrack}")

    def update(self, detections: List[Tuple[int, int, int, int, float, str]], frame_idx: int) -> Dict[int, TrackedPerson]:
        """
        Update tracks with new detections.

        Args:
            detections: List of (x1, y1, x2, y2, conf, class_name)
            frame_idx: Current frame index

        Returns:
            Dictionary of active tracks
        """
        self.frame_idx = frame_idx

        if self.use_bytetrack and self.tracker is not None:
            return self._update_bytetrack(detections)
        else:
            return self._update_fallback(detections)

    def _update_bytetrack(self, detections: List[Tuple]) -> Dict[int, TrackedPerson]:
        """Update using ByteTrack implementation."""
        # Convert detections to expected format
        dets = []
        for x1, y1, x2, y2, conf, _ in detections:
            dets.append([x1, y1, x2, y2, conf])

        # Update ByteTrack
        try:
            tracks = self.tracker.update(np.array(dets), self.frame_idx)
        except:
            tracks = []

        # Convert to our TrackedPerson format
        active_tracks = {}
        for track in tracks:
            track_id = int(track.track_id)
            x1, y1, x2, y2 = [int(v) for v in track.tlbr]
            conf = float(track.score)

            if track_id not in self.active_tracks:
                self.active_tracks[track_id] = TrackedPerson(
                    id=track_id,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf
                )

            person = self.active_tracks[track_id]
            person.update_position((x1, y1, x2, y2), self.frame_idx)
            person.decrement_lock()
            active_tracks[track_id] = person

        # Remove old tracks
        to_remove = []
        for track_id, person in self.active_tracks.items():
            if self.frame_idx - person.last_seen_frame > self.max_age:
                to_remove.append(track_id)

        for track_id in to_remove:
            del self.active_tracks[track_id]

        return active_tracks

    def _update_fallback(self, detections: List[Tuple]) -> Dict[int, TrackedPerson]:
        """Simple fallback tracker when ByteTrack unavailable."""
        # Mark all existing tracks as not seen this frame
        for person in self.active_tracks.values():
            person.last_seen_frame = -1

        # Associate detections with existing tracks
        for det in detections:
            x1, y1, x2, y2, conf, _ = det
            det_bbox = (x1, y1, x2, y2)

            best_iou = 0.0
            best_track_id = None

            # Find best matching existing track
            for track_id, person in self.active_tracks.items():
                if person.last_seen_frame == self.frame_idx:
                    continue  # Already matched this frame

                iou = self._calculate_iou(det_bbox, person.bbox)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None:
                # Update existing track
                person = self.active_tracks[best_track_id]
                person.update_position(det_bbox, self.frame_idx)
                person.confidence = conf
            else:
                # Create new track
                new_id = self._next_id
                self._next_id += 1
                self.active_tracks[new_id] = TrackedPerson(
                    id=new_id,
                    bbox=det_bbox,
                    confidence=conf
                )

        # Remove unmatched tracks
        to_remove = []
        for track_id, person in self.active_tracks.items():
            if person.last_seen_frame != self.frame_idx:
                if self.frame_idx - person.last_seen_frame > self.max_age:
                    to_remove.append(track_id)

        for track_id in to_remove:
            del self.active_tracks[track_id]

        # Decrement locks
        for person in self.active_tracks.values():
            person.decrement_lock()

        return self.active_tracks.copy()

    def update_person_recognition(self, person_id: int, name: str):
        """Update recognition result for a person."""
        if person_id in self.active_tracks:
            self.active_tracks[person_id].update_recognition(name)

    def update_weapon_association(self, person_id: int, has_weapon: bool):
        """Update weapon association for a person."""
        if person_id in self.active_tracks:
            self.active_tracks[person_id].has_weapon = has_weapon

    def get_person(self, person_id: int) -> Optional[TrackedPerson]:
        """Get tracked person by ID."""
        return self.active_tracks.get(person_id)

    def get_all_persons(self) -> List[TrackedPerson]:
        """Get all active tracked persons."""
        return list(self.active_tracks.values())

    def clear(self):
        """Clear all tracks."""
        self.active_tracks.clear()
        if not self.use_bytetrack and hasattr(self, '_fallback_tracks'):
            self._fallback_tracks.clear()

    @staticmethod
    def _calculate_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union between two bboxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0