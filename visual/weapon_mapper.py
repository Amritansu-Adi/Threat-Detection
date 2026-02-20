"""
Weapon Association Components for Late Fusion State Machine

This module contains weapon-to-person association logic:
- WeaponMapper: Associates detected weapons with tracked persons
- HandDetector: MediaPipe-based hand landmark detection
"""

import cv2
import mediapipe as mp
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class HandLandmarks:
    """Hand landmark coordinates."""
    landmarks: List[Tuple[int, int]]  # List of (x, y) coordinates


class HandDetector:
    """MediaPipe-based hand landmark detection."""

    def __init__(self,
                 static_image_mode: bool = False,
                 max_num_hands: int = 4,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize hand detector.

        Args:
            static_image_mode: Whether to treat input as static images
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.available = False
        self.mp_hands = None
        
        try:
            self.mp_hands = mp.solutions.hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.available = True
        except (AttributeError, Exception) as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"mediapipe hand detection not available: {e} - HandDetector will be stubbed")
            self.available = False

    def detect(self, frame: np.ndarray) -> List[HandLandmarks]:
        """
        Detect hand landmarks in frame.

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of hand landmarks
        """
        if not self.available or self.mp_hands is None:
            return []
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb)

        hands = []
        if results and results.multi_hand_landmarks:
            h, w, _ = frame.shape
            for hand_landmarks in results.multi_hand_landmarks:
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.append((int(lm.x * w), int(lm.y * h)))
                hands.append(HandLandmarks(landmarks=coords))

        return hands


class WeaponMapper:
    """Associates detected weapons with tracked persons using multiple heuristics."""

    def __init__(self, overlap_threshold: float = 0.85):
        """
        Initialize weapon mapper.

        Args:
            overlap_threshold: IOU threshold for high-overlap associations
        """
        self.overlap_threshold = overlap_threshold
        self.hand_detector = HandDetector()

    def map_weapons_to_persons(self,
                              frame: np.ndarray,
                              weapons: List[Tuple[int, int, int, int, float, str]],
                              persons: List[Dict]) -> Dict[int, List[Tuple[int, int, int, int, float, str]]]:
        """
        Map weapons to persons using cascading logic.

        Args:
            frame: Current frame
            weapons: List of (x1, y1, x2, y2, conf, label)
            persons: List of person dicts with 'id', 'bbox', 'name'

        Returns:
            Dict mapping person_id to list of associated weapons
        """
        if not weapons or not persons:
            return {}

        # Detect hands
        hands = self.hand_detector.detect(frame)

        # Build mapping
        associations = {}

        for weapon_bbox in weapons:
            wx1, wy1, wx2, wy2, wconf, wlabel = weapon_bbox
            weapon_bbox_coords = (wx1, wy1, wx2, wy2)

            person_id = self._find_weapon_holder(weapon_bbox_coords, persons, hands)
            if person_id is not None:
                if person_id not in associations:
                    associations[person_id] = []
                associations[person_id].append(weapon_bbox)

        return associations

    def _find_weapon_holder(self,
                           weapon_bbox: Tuple[int, int, int, int],
                           persons: List[Dict],
                           hands: List[HandLandmarks]) -> Optional[int]:
        """
        Find which person is holding a weapon using cascading logic.

        Priority order:
        1. Hand overlap (highest priority)
        2. High bbox overlap
        3. Bbox center containment
        4. Best IOU
        5. Nearest person by centroid

        Args:
            weapon_bbox: Weapon bounding box (x1, y1, x2, y2)
            persons: List of person dictionaries
            hands: Detected hand landmarks

        Returns:
            Person ID if associated, None otherwise
        """
        wx1, wy1, wx2, wy2 = weapon_bbox
        wcenter = self._bbox_center(weapon_bbox)

        # 1. Hand Overlap (highest priority)
        for hand in hands:
            for hx, hy in hand.landmarks:
                if wx1 <= hx <= wx2 and wy1 <= hy <= wy2:
                    # Check which person's bbox contains this hand point
                    for person in persons:
                        px1, py1, px2, py2 = person['bbox']
                        if px1 <= hx <= px2 and py1 <= hy <= py2:
                            return person['id']

        # 2. High overlap
        best_overlap = 0.0
        best_person_id = None
        w_area = (wx2 - wx1) * (wy2 - wy1)

        for person in persons:
            px1, py1, px2, py2 = person['bbox']
            ix1, iy1, ix2, iy2 = max(wx1, px1), max(wy1, py1), min(wx2, px2), min(wy2, py2)
            inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            overlap_ratio = inter_area / float(w_area) if w_area > 0 else 0

            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_person_id = person['id']

        if best_overlap >= self.overlap_threshold:
            return best_person_id

        # 3. Bbox center containment
        for person in persons:
            x1, y1, x2, y2 = person['bbox']
            if x1 <= wcenter[0] <= x2 and y1 <= wcenter[1] <= y2:
                return person['id']

        # 4. Best IOU
        if persons:
            best_iou = 0.0
            best_person_id_iou = None
            for person in persons:
                iou = self._calculate_iou(weapon_bbox, person['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_person_id_iou = person['id']

            if best_iou > 0.05:
                return best_person_id_iou

        # 5. Nearest person by centroid (with distance threshold)
        if persons:
            min_dist = float('inf')
            nearest_person_id = None
            max_association_dist = 50  # Maximum pixel distance for association
            
            for person in persons:
                pc = self._bbox_center(person['bbox'])
                dist = ((pc[0] - wcenter[0])**2 + (pc[1] - wcenter[1])**2)**0.5
                if dist < min_dist and dist <= max_association_dist:
                    min_dist = dist
                    nearest_person_id = person['id']
            return nearest_person_id

        return None

    def get_unassociated_weapons(self,
                                weapons: List[Tuple[int, int, int, int, float, str]],
                                associations: Dict[int, List]) -> List[Tuple[int, int, int, int, float, str]]:
        """
        Get weapons that are not associated with any person.

        Args:
            weapons: All detected weapons
            associations: Person-to-weapon associations

        Returns:
            List of unassociated weapons
        """
        associated_weapon_set = set()
        for weapon_list in associations.values():
            for weapon in weapon_list:
                associated_weapon_set.add(weapon)

        return [w for w in weapons if w not in associated_weapon_set]

    @staticmethod
    def _bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Calculate bounding box center."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def _calculate_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union."""
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