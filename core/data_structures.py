"""Data structures for Late Fusion State Machine.

This module defines all core data classes used across the system for
communication between visual, audio, and fusion components.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import time


class RiskState(str, Enum):
    """Risk assessment states in the state machine."""

    IDLE = "IDLE"  # 0-25 pts: Normal operation
    CAUTION = "CAUTION"  # 25-50 pts: Minor threat detected
    EVALUATING = "EVALUATING"  # 50-75 pts: Context analysis underway
    ALERT = "ALERT"  # 75-90 pts: High threat confirmed
    CRITICAL = "CRITICAL"  # 90-100 pts: Immediate action required


class FusionCommand(str, Enum):
    """Commands from fusion manager to visual pipeline."""

    FOCUS_RE_INFERENCE = "FOCUS_RE_INFERENCE"  # audio interrupt: re-run detection
    INCREASE_SENSITIVITY = "INCREASE_SENSITIVITY"
    DECREASE_SENSITIVITY = "DECREASE_SENSITIVITY"
    SNAPSHOT_CAPTURE = "SNAPSHOT_CAPTURE"


class EmotionType(str, Enum):
    """Audio emotion classifications."""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    DISGUST = "disgust"
    SURPRISE = "surprise"


class IntentType(str, Enum):
    """Audio intent classifications (zero-shot from DistilBERT)."""

    DISTRESS = "distress"
    THREAT = "threat"
    NEUTRAL = "neutral"
    INFORMATIONAL = "informational"


@dataclass
class BoundingBox:
    """Rectangular bounding box in normalized coordinates."""

    x1: float  # Left edge (0.0-1.0)
    y1: float  # Top edge (0.0-1.0)
    x2: float  # Right edge (0.0-1.0)
    y2: float  # Bottom edge (0.0-1.0)
    confidence: float = 1.0  # Detection confidence (0.0-1.0)

    def center(self) -> tuple:
        """Return (x, y) center of bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def area(self) -> float:
        """Return area of bounding box."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass
class TrackedPerson:
    """Person detection with tracking information.

    Represents a detected and tracked person with optional identity
    and weapon association data.
    """

    track_id: int  # Unique ID from centroid tracker
    bbox: BoundingBox  # Normalized bounding box
    identity: Optional[str] = None  # Face recognition result (e.g., "UNKNOWN", "John Doe")
    has_weapon: bool = False  # Weapon association flag
    pose_keypoints: Optional[List[Dict[str, float]]] = None  # MediaPipe keypoints
    pose_confidence: float = 0.0  # Pose detection confidence
    timestamp: float = field(default_factory=time.time)  # Detection timestamp
    history: List[BoundingBox] = field(default_factory=list)  # Trajectory history (last 30 frames)

    def add_history(self, bbox: BoundingBox, max_history: int = 30):
        """Add bbox to history, maintaining max length."""
        self.history.append(bbox)
        if len(self.history) > max_history:
            self.history.pop(0)


@dataclass
class WeaponDetection:
    """Weapon detection with spatial and ownership information."""

    bbox: BoundingBox  # Weapon bounding box
    weapon_type: str  # "knife", "gun", "rifle", etc.
    confidence: float  # Detection confidence (0.0-1.0)
    associated_person_id: Optional[int] = None  # track_id if held by person
    timestamp: float = field(default_factory=time.time)


@dataclass
class AudioEvent:
    """Audio analysis result from 3-stage audio pipeline.

    Contains emotion (Stage 2), intent (Stage 3), and transcription.
    """

    timestamp: float  # Event timestamp
    transcript: str  # Speech-to-text result (Stage 3: Whisper)
    emotion: EmotionType  # Detected emotion (Stage 2: DistilHuBERT)
    emotion_confidence: float  # Emotion confidence (0.0-1.0)
    intent: IntentType  # Intent classification (Stage 3: DistilBERT)
    intent_confidence: float  # Intent confidence (0.0-1.0)
    sample_duration: float = 0.0  # Duration of audio sample analyzed
    vad_confidence: float = 1.0  # VAD confidence (Stage 1: Silero)
    raw_audio_features: Optional[Dict[str, Any]] = None  # Optional raw audio data


@dataclass
class VisualData:
    """Visual pipeline output with all detection and tracking results."""

    timestamp: float  # Frame timestamp
    frame_idx: int  # Sequential frame index
    tracked_persons: List[TrackedPerson] = field(default_factory=list)
    weapons: List[WeaponDetection] = field(default_factory=list)
    frame_resolution: tuple = (1920, 1080)  # (width, height)
    fps: float = 30.0  # Frames per second
    processing_time_ms: float = 0.0  # Time to process frame


@dataclass
class RiskEvent:
    """Risk assessment result from fusion manager.

    Contains calculated risk score, resulting state, and contributing factors
    for logging and debugging.
    """

    score: float  # Overall risk score (0-100)
    state: RiskState  # Resulting state machine state
    timestamp: float = field(default_factory=time.time)
    visual_risk: float = 0.0  # Visual component risk
    audio_risk: float = 0.0  # Audio component risk
    contributing_factors: List[str] = field(default_factory=list)
    state_duration_ms: float = 0.0  # How long in current state
    previous_state: Optional[RiskState] = None


@dataclass
class FusionInterrupt:
    """Control signal from fusion manager to visual pipeline.

    Used for audio-triggered visual re-inference and other priority changes.
    """

    command: FusionCommand
    timestamp: float = field(default_factory=time.time)
    zones: List[BoundingBox] = field(default_factory=list)  # Priority zones
    priority: str = "HIGH"  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertData:
    """Alert trigger with full context for external systems.

    Contains risk assessment, state transition, and both sensory modalities'
    context for email, logging, or webhook integration.
    """

    timestamp: float = field(default_factory=time.time)
    state: RiskState = RiskState.ALERT
    risk_score: float = field(default=0.0)
    
    # Visual context
    visual_persons: List[TrackedPerson] = field(default_factory=list)
    weapons_detected: List[WeaponDetection] = field(default_factory=list)
    unknown_persons_count: int = 0
    armed_count: int = 0
    
    # Audio context
    audio_transcript: str = ""
    audio_emotion: EmotionType = EmotionType.NEUTRAL
    audio_emotion_confidence: float = 0.0
    audio_intent: IntentType = IntentType.NEUTRAL
    audio_intent_confidence: float = 0.0
    
    # Metadata
    alert_type: str = "UNKNOWN"  # e.g., "ARMED_UNKNOWN_PERSON", "THREAT_SPEECH"
    confidence: float = 0.0
    action_taken: str = ""
    

@dataclass
class SystemMetrics:
    """System performance metrics for monitoring."""

    timestamp: float = field(default_factory=time.time)
    
    # Thread statistics
    visual_fps: float = 0.0
    audio_chunks_per_sec: float = 0.0
    fusion_hz: float = 0.0
    
    # Latency (milliseconds)
    visual_latency_ms: float = 0.0
    audio_latency_ms: float = 0.0
    fusion_latency_ms: float = 0.0
    e2e_latency_ms: float = 0.0  # End-to-end from frame capture to alert
    
    # Queue statistics
    visual_queue_size: int = 0
    audio_queue_size: int = 0
    
    # Resource usage
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    
    # Risk statistics
    current_risk_score: float = 0.0
    current_state: RiskState = RiskState.IDLE
    state_duration_sec: float = 0.0
    total_alerts: int = 0
