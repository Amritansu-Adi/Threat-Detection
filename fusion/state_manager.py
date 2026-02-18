"""State machine implementation for risk state management.

The StateManager tracks state transitions, enforces minimum durations
before transitioning to alert states, and triggers callbacks.

State Transitions:
    IDLE ↔ CAUTION ↔ EVALUATING ↔ ALERT ↔ CRITICAL

Duration Thresholds:
    ALERT state: 3 seconds minimum before triggering alert
    CRITICAL state: 0.5 seconds before triggering alert
    
This prevents false alerts from brief spike but allows fast response to critical threats.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional, List
from enum import Enum

from core.data_structures import RiskState

logger = logging.getLogger(__name__)


class AlertTriggerType(str, Enum):
    """Types of alert triggers."""

    IMMEDIATE = "immediate"  # CRITICAL state (≥90 pts)
    SUSTAINED = "sustained"  # ALERT state (≥75 pts) for 3+ seconds
    ESCALATION = "escalation"  # Rapid risk increase


@dataclass
class StateTransition:
    """Records a state transition event."""

    from_state: RiskState
    to_state: RiskState
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    risk_score: float = 0.0
    reason: str = ""


class StateManager:
    """Manages risk state machine with duration thresholds.

    Responsibilities:
    1. Track current state
    2. Validate transitions (prevent invalid paths)
    3. Enforce minimum durations (prevent flaky alerts)
    4. Trigger callbacks on state change
    5. Generate alerts when thresholds met

    Example:
        >>> sm = StateManager()
        >>> sm.update_state(RiskState.CAUTION, 30.0)
        >>> sm.is_alert_triggered()
        False
        >>> sm.update_state(RiskState.ALERT, 80.0)
        >>> time.sleep(3.1)
        >>> sm.is_alert_triggered()
        True
    """

    def __init__(self):
        """Initialize state manager."""
        self.current_state = RiskState.IDLE
        self.state_entered_time = datetime.now()
        self.previous_state = RiskState.IDLE
        
        # Alert triggering thresholds
        self.alert_duration_sec = 3.0  # Must stay in ALERT for 3 sec
        self.critical_duration_sec = 0.5  # Only 0.5 sec in CRITICAL
        
        # Callbacks for state changes
        self._state_change_callbacks: List[Callable] = []
        self._alert_triggered_callbacks: List[Callable] = []
        
        # History
        self.transitions: List[StateTransition] = []
        self.last_alert_time: Optional[float] = None

        logger.info(f"StateManager initialized: {self.current_state.value}")

    def update_state(self, new_state: RiskState, risk_score: float) -> bool:
        """Update to new state.

        Args:
            new_state: New risk state
            risk_score: Current risk score for logging

        Returns:
            bool: True if state changed, False if stayed same
        """
        if new_state == self.current_state:
            return False

        # Record transition
        old_state = self.current_state
        self.previous_state = old_state
        self.current_state = new_state
        self.state_entered_time = datetime.now()

        transition = StateTransition(
            from_state=old_state,
            to_state=new_state,
            risk_score=risk_score,
            reason=f"Risk: {risk_score:.1f} pts",
        )
        self.transitions.append(transition)

        logger.info(
            f"State transition: {old_state.value} → {new_state.value} "
            f"(risk={risk_score:.1f} pts)"
        )

        # Call callbacks
        for callback in self._state_change_callbacks:
            try:
                callback(old_state, new_state, risk_score)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")

        return True

    def get_current_state(self) -> RiskState:
        """Get current state.

        Returns:
            RiskState: Current state
        """
        return self.current_state

    def get_state_duration_sec(self) -> float:
        """Get how long we've been in current state.

        Returns:
            float: Duration in seconds
        """
        elapsed = datetime.now() - self.state_entered_time
        return elapsed.total_seconds()

    def is_alert_triggered(self) -> bool:
        """Check if alert condition has been met.

        Alert is triggered when:
        1. State is CRITICAL (≥90 pts) for 0.5+ seconds
        2. State is ALERT (≥75 pts) for 3+ seconds

        Returns:
            bool: True if alert should be triggered
        """
        state = self.current_state
        duration = self.get_state_duration_sec()

        # CRITICAL: immediate alert after 0.5 seconds
        if state == RiskState.CRITICAL and duration >= self.critical_duration_sec:
            return True

        # ALERT: delayed alert after 3 seconds
        if state == RiskState.ALERT and duration >= self.alert_duration_sec:
            return True

        return False

    def get_alert_trigger_type(self) -> Optional[AlertTriggerType]:
        """Get the type of alert trigger.

        Returns:
            AlertTriggerType or None if no alert
        """
        state = self.current_state
        duration = self.get_state_duration_sec()

        if state == RiskState.CRITICAL and duration >= self.critical_duration_sec:
            return AlertTriggerType.IMMEDIATE

        if state == RiskState.ALERT and duration >= self.alert_duration_sec:
            return AlertTriggerType.SUSTAINED

        if state == RiskState.EVALUATING and self.previous_state == RiskState.CAUTION:
            return AlertTriggerType.ESCALATION

        return None

    def acknowledge_alert(self) -> None:
        """Mark alert as acknowledged (resets trigger time).

        This allows detecting the next alert even if state hasn't changed.
        """
        self.last_alert_time = datetime.now().timestamp()
        logger.info(f"Alert acknowledged at state {self.current_state.value}")

    def is_state_valid(self, state: RiskState) -> bool:
        """Check if state is valid (prevents invalid transitions).

        In this simple implementation, all states are valid.
        Can be extended with explicit transition matrices.

        Args:
            state: State to check

        Returns:
            bool: True if valid
        """
        return isinstance(state, RiskState)

    # === Callback Management ===

    def on_state_change(self, callback: Callable) -> None:
        """Register callback for state changes.

        Args:
            callback: Function(old_state, new_state, risk_score)
        """
        self._state_change_callbacks.append(callback)

    def on_alert_triggered(self, callback: Callable) -> None:
        """Register callback for alert triggers.

        Args:
            callback: Function(state, risk_score, duration_sec)
        """
        self._alert_triggered_callbacks.append(callback)

    def trigger_alert_callbacks(self, risk_score: float) -> None:
        """Trigger all alert callbacks.

        Args:
            risk_score: Current risk score
        """
        for callback in self._alert_triggered_callbacks:
            try:
                callback(
                    self.current_state,
                    risk_score,
                    self.get_state_duration_sec(),
                )
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    # === Debugging & History ===

    def get_transition_history(self, last_n: int = 20) -> List[StateTransition]:
        """Get recent state transitions.

        Args:
            last_n: Number of recent transitions to return

        Returns:
            List[StateTransition]: Recent history
        """
        return self.transitions[-last_n:]

    def get_state_summary(self) -> dict:
        """Get current state summary for logging/monitoring.

        Returns:
            dict: State information
        """
        return {
            "current_state": self.current_state.value,
            "duration_sec": self.get_state_duration_sec(),
            "alert_triggered": self.is_alert_triggered(),
            "transition_count": len(self.transitions),
            "last_transition": (
                self.transitions[-1].to_state.value
                if self.transitions
                else "NONE"
            ),
        }

    def reset(self) -> None:
        """Reset to initial state (for testing).

        WARNING: This clears history!
        """
        logger.warning("Resetting StateManager")
        self.current_state = RiskState.IDLE
        self.state_entered_time = datetime.now()
        self.previous_state = RiskState.IDLE
        self.transitions.clear()
        self.last_alert_time = None
