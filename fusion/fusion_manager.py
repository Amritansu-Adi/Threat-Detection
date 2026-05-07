"""Late Fusion Manager - Main orchestrator for state machine.

The FusionManager is the core coordinator that:
1. Listens on visual_to_fusion_q for VisualData (non-blocking)
2. Listens on audio_to_fusion_q for AudioEvents (non-blocking)
3. Calculates risk using RiskAccumulator
4. Updates state using StateManager
5. Publishes interrupts via fusion_to_visual_q
6. Updates shared state for all threads to see

Runs at ~1 Hz (1 second updates) to match human perception timescale.
"""

import logging
import threading
import time
from typing import Optional, Callable
from datetime import datetime

from core.data_structures import (
    VisualData,
    AudioEvent,
    RiskState,
    FusionInterrupt,
    FusionCommand,
    AlertData,
)
from core.shared_state import SharedStateManager
from core.queue_manager import QueueManager
from fusion.risk_accumulator import RiskAccumulator
from fusion.state_manager import StateManager, AlertTriggerType

logger = logging.getLogger(__name__)


class FusionManager:
    """Late Fusion orchestrator coordinating visual, audio, and state.

    Design principles:
    1. Non-blocking queue reads (don't wait for slow producers)
    2. Drain queues to get latest data only (drop stale items)
    3. Always update shared state (even if no input received)
    4. Send interrupts only on significant events
    5. Run at ~1 Hz for human-scale perception

    Example:
        >>> shared = SharedStateManager()
        >>> queues = QueueManager()
        >>> fusion = FusionManager(shared, queues)
        >>> fusion.start()
        >>> time.sleep(60)
        >>> fusion.stop()
    """

    def __init__(
        self,
        shared_state: SharedStateManager,
        queue_manager: QueueManager,
        target_hz: float = 1.0,
    ):
        """Initialize FusionManager.

        Args:
            shared_state: SharedStateManager for publishing global state
            queue_manager: QueueManager for inter-thread communication
            target_hz: Fusion loop frequency (default 1 Hz)
        """
        self.shared = shared_state
        self.queues = queue_manager
        self.target_hz = target_hz
        self.target_period_sec = 1.0 / target_hz

        # Core components
        self.risk_accumulator = RiskAccumulator()
        self.state_manager = StateManager()

        # Thread control
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # Metrics
        self.loop_count = 0
        self.last_update_time = datetime.now()
        self._last_loop_time = time.time()
        self.alerts_triggered = 0
        self._latest_visual_data: Optional[VisualData] = None
        self._latest_visual_seen_at = 0.0
        self._latest_audio_event: Optional[AudioEvent] = None
        self._latest_audio_seen_at = 0.0
        self.visual_ttl_sec = 1.0
        self.audio_ttl_sec = 2.5

        # Callbacks
        self._on_alert_callbacks = []
        self._on_state_change_callbacks = []

        logger.info(
            f"FusionManager initialized (target: {target_hz} Hz, "
            f"period: {self.target_period_sec:.3f} sec)"
        )

    def start(self) -> None:
        """Start fusion manager in background thread.

        Safe to call multiple times (second call is no-op if already running).
        """
        with self._lock:
            if self._running:
                logger.warning("FusionManager already running")
                return

            self._running = True
            self.shared.set_running(True)

        self._thread = threading.Thread(target=self._fusion_loop, daemon=False)
        self._thread.start()
        logger.info("FusionManager started")

    def stop(self) -> None:
        """Stop fusion manager and wait for thread to finish.

        Safe to call multiple times.
        """
        with self._lock:
            if not self._running:
                logger.warning("FusionManager not running")
                return
            self._running = False

        self.shared.set_running(False)

        if self._thread:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.error("FusionManager thread did not stop cleanly")
            logger.info("FusionManager stopped")

    def _fusion_loop(self) -> None:
        """Main fusion loop running at target_hz frequency.

        Runs in background thread. Continuous loop:
            1. Drain queues to get latest data
            2. Calculate risk
            3. Update state
            4. Check alert conditions
            5. Send interrupts
            6. Update shared state
            7. Sleep to maintain frequency
        """
        logger.info(f"Fusion loop started (target {self.target_hz} Hz)")

        while self._running:
            try:
                loop_start = time.time()

                # Step 1: Get latest data from queues (non-blocking)
                visual_data = self._get_latest_visual_data()
                audio_event = self._get_latest_audio_event()
                now = time.time()

                if visual_data is not None:
                    self._latest_visual_data = visual_data
                    self._latest_visual_seen_at = now
                elif now - self._latest_visual_seen_at <= self.visual_ttl_sec:
                    visual_data = self._latest_visual_data

                if audio_event is not None:
                    self._latest_audio_event = audio_event
                    self._latest_audio_seen_at = now
                elif now - self._latest_audio_seen_at <= self.audio_ttl_sec:
                    audio_event = self._latest_audio_event

                # Step 2: Calculate risk using Temporal Persistence Multiplier
                current_risk = self.shared.get_risk()
                dt = max(0.001, now - self._last_loop_time)
                self._last_loop_time = now
                risk_event = self.risk_accumulator.calculate_risk(
                    current_risk=current_risk,
                    visual_data=visual_data,
                    audio_event=audio_event,
                    dt_seconds=dt,
                )

                # Step 3: Update state machine
                state_changed = self.state_manager.update_state(
                    risk_event.state, risk_event.score
                )

                # Step 4: Check alert conditions
                alert_triggered = self._check_and_trigger_alert(risk_event.score)
                if risk_event.state not in (RiskState.ALERT, RiskState.CRITICAL):
                    self.shared.set_alert(False)

                # Step 5: Send interrupts if needed
                if state_changed or alert_triggered:
                    self._send_interrupts(risk_event.state, risk_event.score)

                # Step 6: Update shared state (ALL threads see this instantly)
                self.shared.update_risk(
                    risk_event.score,
                    risk_event.state,
                    visual_risk=risk_event.visual_risk,
                    audio_risk=risk_event.audio_risk,
                    contributing_factors=risk_event.contributing_factors,
                )
                if visual_data:
                    persons = self.risk_accumulator._get_visual_persons(visual_data)
                    unknown_count = sum(
                        1
                        for p in persons
                        if self.risk_accumulator._is_unknown_identity(
                            self.risk_accumulator._person_identity(p)
                        )
                    )
                    weapon_count = sum(
                        1 for p in persons if self.risk_accumulator._person_has_weapon(p)
                    )
                    unknown_with_weapon = any(
                        self.risk_accumulator._person_has_weapon(p)
                        and self.risk_accumulator._is_unknown_identity(
                            self.risk_accumulator._person_identity(p)
                        )
                        for p in persons
                    )
                    self.shared.update_visual_summary(
                        len(persons),
                        unknown_count,
                        weapon_count,
                        len(visual_data.weapons),
                        unknown_with_weapon,
                    )
                else:
                    self.shared.update_visual_summary(0, 0, 0, 0)
                if audio_event:
                    self.shared.update_audio_summary(
                        audio_event.transcript,
                        audio_event.emotion.value,
                        audio_event.emotion_confidence,
                        audio_event.intent.value,
                        audio_event.intent_confidence,
                    )
                elif now - self._latest_audio_seen_at > self.audio_ttl_sec:
                    self.shared.update_audio_summary(
                        "",
                        "neutral",
                        0.0,
                        "neutral",
                        0.0,
                    )

                # Step 7: Maintain loop frequency
                self.loop_count += 1
                elapsed = time.time() - loop_start
                if elapsed < self.target_period_sec:
                    sleep_time = self.target_period_sec - elapsed
                    time.sleep(sleep_time)
                else:
                    logger.debug(
                        f"Fusion loop slower than target: {elapsed:.3f}s "
                        f"(target: {self.target_period_sec:.3f}s)"
                    )

            except Exception as e:
                logger.error(f"Error in fusion loop: {e}", exc_info=True)
                self.shared.set_error(f"Fusion loop error: {str(e)}")
                time.sleep(0.1)  # Prevent exception spam

        logger.info(f"Fusion loop stopped (completed {self.loop_count} iterations)")

    def _get_latest_visual_data(self) -> Optional[VisualData]:
        """Get latest visual data, dropping stale items.

        Returns:
            Latest VisualData or None if queue empty
        """
        count, latest = self.queues.drain_visual_queue()
        if count > 1:
            logger.debug(f"Dropped {count - 1} stale visual frames")
        return latest

    def _get_latest_audio_event(self) -> Optional[AudioEvent]:
        """Get latest audio event, dropping stale items.

        Returns:
            Latest AudioEvent or None if queue empty
        """
        count, latest = self.queues.drain_audio_queue()
        if count > 1:
            logger.debug(f"Dropped {count - 1} stale audio events")
        return latest

    def _check_and_trigger_alert(self, risk_score: float) -> bool:
        """Check if alert should be triggered and fire callbacks.

        Args:
            risk_score: Current risk score

        Returns:
            bool: True if alert was triggered
        """
        if not self.state_manager.is_alert_triggered():
            return False

        # Don't retrigger same alert (callback already fired)
        if self.shared.is_alert_active():
            return False

        # Trigger alert
        logger.warning(f"⚠️  ALERT TRIGGERED - Risk: {risk_score:.1f} pts, "
                      f"State: {self.state_manager.current_state.value}")

        self.alerts_triggered += 1
        self.shared.set_alert(True)
        self.state_manager.trigger_alert_callbacks(risk_score)

        # Call registered callbacks
        for callback in self._on_alert_callbacks:
            try:
                callback(self.state_manager.current_state, risk_score)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

        return True

    def _send_interrupts(self, state: RiskState, risk_score: float) -> None:
        """Send control interrupts to visual pipeline.

        Args:
            state: Current state
            risk_score: Current risk score
        """
        # On state escalation to EVALUATING or ALERT, request visual re-inference
        if state in [RiskState.EVALUATING, RiskState.ALERT, RiskState.CRITICAL]:
            interrupt = FusionInterrupt(
                command=FusionCommand.FOCUS_RE_INFERENCE,
                priority="HIGH" if state == RiskState.CRITICAL else "MEDIUM",
            )
            self.queues.put_fusion_interrupt(interrupt)
            logger.debug(f"Sent FOCUS_RE_INFERENCE interrupt (state={state.value})")

    def on_alert(self, callback: Callable) -> None:
        """Register callback for alert triggers.

        Args:
            callback: Function(state, risk_score)
        """
        self._on_alert_callbacks.append(callback)

    def on_state_change(self, callback: Callable) -> None:
        """Register callback for state changes.

        Args:
            callback: Function(old_state, new_state, risk_score)
        """
        self._on_state_change_callbacks.append(callback)

    def get_status(self) -> dict:
        """Get FusionManager status for monitoring.

        Returns:
            dict: Status information
        """
        return {
            "running": self._running,
            "loop_count": self.loop_count,
            "alerts_triggered": self.alerts_triggered,
            "current_state": self.state_manager.current_state.value,
            "state_duration_sec": self.state_manager.get_state_duration_sec(),
            "queue_sizes": self.queues.qsize(),
            "shared_state": self.shared.get_snapshot(),
        }
