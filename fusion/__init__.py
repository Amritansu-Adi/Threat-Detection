"""Fusion module for Late Fusion State Machine."""

__version__ = "0.1.0"

from .risk_accumulator import RiskAccumulator, RiskWeights
from .state_manager import StateManager, StateTransition, AlertTriggerType
from .fusion_manager import FusionManager

__all__ = [
    "RiskAccumulator",
    "RiskWeights",
    "StateManager",
    "StateTransition",
    "AlertTriggerType",
    "FusionManager",
]

