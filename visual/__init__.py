"""
Visual Pipeline Module for Late Fusion State Machine

This module provides complete visual processing pipeline including:
- Person detection and tracking
- Weapon detection and association
- Face recognition
- Visual data emission to fusion
"""

from .detector import PersonDetector, WeaponDetector, FaceRecognizer
from .tracker import PersonTracker, TrackedPerson
from .weapon_mapper import WeaponMapper, HandDetector
from .visual_pipeline import VisualPipeline

__all__ = [
    'PersonDetector',
    'WeaponDetector',
    'FaceRecognizer',
    'PersonTracker',
    'TrackedPerson',
    'WeaponMapper',
    'HandDetector',
    'VisualPipeline'
]