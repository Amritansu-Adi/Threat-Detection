"""Thread-safe shared state manager for Late Fusion System.

This module provides a wrapper around multiprocessing.Manager().dict()
to ensure all threads have instant, consistent access to the system's
global state (risk_score, state, etc.) without locks or synchronization delays.

Uses multiprocessing.Manager() instead of threading locks because:
1. Instant visibility across threads (no memory barriers)
2. No contention on high-frequency reads
3. Atomic dict operations (single-writer assumption on fusion manager)
"""

from multiprocessing import Manager
from typing import Any, Dict, Optional
import logging
from datetime import datetime

from .data_structures import RiskState, AlertData

logger = logging.getLogger(__name__)


class SharedStateManager:
    """Thread-safe manager for system global state using multiprocessing.Manager().

    All threads can read state instantly. FusionManager is assumed to be
    the only writer to prevent race conditions.

    Example:
        >>> shared = SharedStateManager()
        >>> shared.update_risk(45.0, RiskState.EVALUATING)
        >>> risk = shared.get_risk()  # Other threads see instantly
        >>> 45.0 == risk
        True
    """

    def __init__(self):
        """Initialize shared state using multiprocessing.Manager()."""
        self.manager = Manager()
        self._state = self.manager.dict()

        # Initialize all keys
        self._state["risk_score"] = 0.0
        self._state["state"] = RiskState.IDLE.value
        self._state["previous_state"] = RiskState.IDLE.value
        self._state["state_entered_timestamp"] = datetime.now().timestamp()
        self._state["last_update_timestamp"] = datetime.now().timestamp()

        # Visual data
        self._state["visual_persons_count"] = 0
        self._state["visual_unknown_count"] = 0
        self._state["visual_armed_count"] = 0
        self._state["weapons_detected_count"] = 0

        # Audio data
        self._state["audio_transcript"] = ""
        self._state["audio_emotion"] = "neutral"
        self._state["audio_emotion_conf"] = 0.0
        self._state["audio_intent"] = "neutral"
        self._state["audio_intent_conf"] = 0.0

        # Alert state
        self._state["alert_active"] = False
        self._state["last_alert_timestamp"] = 0.0
        self._state["alert_count"] = 0

        # System status
        self._state["is_running"] = True
        self._state["error_message"] = ""

        logger.info("SharedStateManager initialized")

    def close(self):
        """Shutdown the manager (should be called at system exit)."""
        if self.manager:
            self.manager.shutdown()
            logger.info("SharedStateManager closed")

    # === Risk Score & State Management ===

    def update_risk(self, score: float, state: RiskState) -> None:
        """Update risk score and state.

        Args:
            score: Risk score (0-100)
            state: RiskState enum value
        """
        if not 0 <= score <= 100:
            logger.warning(f"Risk score {score} out of range [0, 100]")
            score = max(0.0, min(100.0, score))

        # Track state transitions
        if state.value != self._state["state"]:
            self._state["previous_state"] = self._state["state"]
            self._state["state_entered_timestamp"] = datetime.now().timestamp()
            logger.info(f"State transition: {self._state['state']} → {state.value}")

        self._state["risk_score"] = score
        self._state["state"] = state.value
        self._state["last_update_timestamp"] = datetime.now().timestamp()

    def get_risk(self) -> float:
        """Get current risk score.

        Returns:
            float: Current risk score (0-100)
        """
        return float(self._state["risk_score"])

    def get_state(self) -> RiskState:
        """Get current risk state.

        Returns:
            RiskState: Current state
        """
        state_str = str(self._state["state"])
        return RiskState(state_str)

    def get_state_duration_sec(self) -> float:
        """Get how long system has been in current state.

        Returns:
            float: Duration in seconds
        """
        entered = self._state["state_entered_timestamp"]
        now = datetime.now().timestamp()
        return now - entered

    # === Visual Data Access ===

    def update_visual_summary(
        self,
        persons_count: int,
        unknown_count: int,
        armed_count: int,
        weapons_count: int,
    ) -> None:
        """Update visual detection summary.

        Args:
            persons_count: Total tracked persons
            unknown_count: Persons with unknown identity
            armed_count: Persons associated with weapons
            weapons_count: Total weapons detected
        """
        self._state["visual_persons_count"] = persons_count
        self._state["visual_unknown_count"] = unknown_count
        self._state["visual_armed_count"] = armed_count
        self._state["weapons_detected_count"] = weapons_count

    def get_visual_summary(self) -> Dict[str, int]:
        """Get visual detection summary.

        Returns:
            dict: {persons_count, unknown_count, armed_count, weapons_count}
        """
        return {
            "persons_count": int(self._state["visual_persons_count"]),
            "unknown_count": int(self._state["visual_unknown_count"]),
            "armed_count": int(self._state["visual_armed_count"]),
            "weapons_count": int(self._state["weapons_detected_count"]),
        }

    # === Audio Data Access ===

    def update_audio_summary(
        self,
        transcript: str,
        emotion: str,
        emotion_conf: float,
        intent: str,
        intent_conf: float,
    ) -> None:
        """Update audio analysis summary.

        Args:
            transcript: Speech-to-text result
            emotion: Emotion type
            emotion_conf: Emotion confidence (0-1)
            intent: Intent classification
            intent_conf: Intent confidence (0-1)
        """
        self._state["audio_transcript"] = transcript
        self._state["audio_emotion"] = emotion
        self._state["audio_emotion_conf"] = float(emotion_conf)
        self._state["audio_intent"] = intent
        self._state["audio_intent_conf"] = float(intent_conf)

    def get_audio_summary(self) -> Dict[str, Any]:
        """Get audio analysis summary.

        Returns:
            dict: {transcript, emotion, emotion_conf, intent, intent_conf}
        """
        return {
            "transcript": str(self._state["audio_transcript"]),
            "emotion": str(self._state["audio_emotion"]),
            "emotion_conf": float(self._state["audio_emotion_conf"]),
            "intent": str(self._state["audio_intent"]),
            "intent_conf": float(self._state["audio_intent_conf"]),
        }

    # === Alert Management ===

    def set_alert(self, active: bool) -> None:
        """Set alert state.

        Args:
            active: True if alert should be active
        """
        self._state["alert_active"] = active
        if active:
            self._state["last_alert_timestamp"] = datetime.now().timestamp()
            self._state["alert_count"] = int(self._state["alert_count"]) + 1

    def is_alert_active(self) -> bool:
        """Check if alert is currently active.

        Returns:
            bool: True if alert active
        """
        return bool(self._state["alert_active"])

    def get_alert_count(self) -> int:
        """Get total number of alerts triggered.

        Returns:
            int: Total alert count
        """
        return int(self._state["alert_count"])

    def get_last_alert_time(self) -> Optional[float]:
        """Get timestamp of last alert.

        Returns:
            float: Timestamp or None if never alerted
        """
        ts = self._state["last_alert_timestamp"]
        return ts if ts > 0 else None

    # === System Status ===

    def set_running(self, running: bool) -> None:
        """Set system running state.

        Args:
            running: True if system is running
        """
        self._state["is_running"] = running

    def is_running(self) -> bool:
        """Check if system is running.

        Returns:
            bool: True if running
        """
        return bool(self._state["is_running"])

    def set_error(self, error_msg: str) -> None:
        """Set error message.

        Args:
            error_msg: Error description
        """
        self._state["error_message"] = error_msg
        logger.error(f"System error: {error_msg}")

    def get_error(self) -> str:
        """Get current error message.

        Returns:
            str: Error message or empty string
        """
        return str(self._state["error_message"])

    # === Snapshot & Debugging ===

    def get_snapshot(self) -> Dict[str, Any]:
        """Get complete system state snapshot (for logging/debugging).

        Returns:
            dict: Complete state dictionary
        """
        # Convert dict to regular dict for serialization
        snapshot = {}
        for key in self._state.keys():
            snapshot[key] = self._state[key]
        return snapshot

    def reset(self) -> None:
        """Reset to initial state (for testing or system reset)."""
        logger.warning("Resetting shared state")
        self.update_risk(0.0, RiskState.IDLE)
        self.update_visual_summary(0, 0, 0, 0)
        self.update_audio_summary("", "neutral", 0.0, "neutral", 0.0)
        self.set_alert(False)
        self.set_error("")
