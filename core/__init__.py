"""Core module for Autonomous Multi-Modal Intelligence Ecosystem."""

__version__ = "0.1.0"
__author__ = "AMIE Development Team"

from .data_structures import (
    TrackedPerson,
    WeaponDetection,
    AudioEvent,
    VisualData,
    RiskEvent,
    FusionInterrupt,
    AlertData,
)
from .shared_state import SharedStateManager
from .queue_manager import QueueManager
from .config_loader import ConfigLoader

__all__ = [
    "TrackedPerson",
    "WeaponDetection",
    "AudioEvent",
    "VisualData",
    "RiskEvent",
    "FusionInterrupt",
    "AlertData",
    "SharedStateManager",
    "QueueManager",
    "ConfigLoader",
]
