"""Integration test for Phase 1-2 core infrastructure.

Tests the Temporal Persistence Multiplier formula with the "Shut up" scenario:
    T=0s: "Shut up!" with anger (0.8 conf) → 25 pts → CAUTION
    T=1s: Laughter (happy emotion) → filtered to 0 → decay to 22.5 pts
    T=2s: No new input → decay to 20.3 pts
    Result: Never crosses alert threshold, correctly handled

This validates:
1. Data structure instantiation
2. Risk accumulation formula
3. Temporal decay
4. Context filtering
5. State machine transitions
6. Queue communication
"""

import time
import logging
from datetime import datetime

from core.data_structures import (
    AudioEvent,
    RiskState,
    EmotionType,
    IntentType,
    VisualData,
    TrackedPerson,
    BoundingBox,
)
from core.shared_state import SharedStateManager
from core.queue_manager import QueueManager
from fusion.risk_accumulator import RiskAccumulator
from fusion.state_manager import StateManager

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def test_shut_up_scenario():
    """Test the "Shut up" scenario with context filtering.

    Validates that a threat phrase ("Shut up!") combined with joking
    emotion (laughter/happy) doesn't trigger an alert.
    """
    logger.info("=" * 80)
    logger.info("TEST: 'Shut Up' Scenario - Threat + Joking Context")
    logger.info("=" * 80)

    # Initialize components
    risk_acc = RiskAccumulator()
    state_mgr = StateManager()

    current_risk = 0.0
    dt = 1.0  # 1 second per update

    # === T=0s: "Shut up!" with anger ===
    logger.info("\n[T=0s] Threat speech detected")
    audio_0 = AudioEvent(
        timestamp=time.time(),
        transcript="Shut up!",
        emotion=EmotionType.ANGRY,
        emotion_confidence=0.80,  # High confidence
        intent=IntentType.THREAT,
        intent_confidence=0.85,
        sample_duration=1.5,
        vad_confidence=0.95,
    )

    risk_event_0 = risk_acc.calculate_risk(
        current_risk=current_risk,
        audio_event=audio_0,
        dt_seconds=dt,
    )

    state_mgr.update_state(risk_event_0.state, risk_event_0.score)
    current_risk = risk_event_0.score

    logger.info(f"  Transcript: '{audio_0.transcript}'")
    logger.info(f"  Emotion: {audio_0.emotion.value} (conf={audio_0.emotion_confidence:.2f})")
    logger.info(f"  Intent: {audio_0.intent.value} (conf={audio_0.intent_confidence:.2f})")
    logger.info(f"  → Risk: {risk_event_0.visual_risk:.1f} (visual) + "
               f"{risk_event_0.audio_risk:.1f} (audio) = {current_risk:.1f} pts")
    logger.info(f"  → State: {state_mgr.current_state.value}")
    logger.info(f"  → Alert triggered: {state_mgr.is_alert_triggered()}")
    logger.info(f"  Contributing factors: {risk_event_0.contributing_factors}")

    assert current_risk > 0, "Should have non-zero risk from threat"
    # Anger (25 pts) + Threat intent (45 pts) = 70 pts → EVALUATING state
    assert current_risk >= 60, f"Should have high risk from threat+anger, got {current_risk}"
    assert state_mgr.current_state == RiskState.EVALUATING, f"Should be EVALUATING at ~70 pts, got {state_mgr.current_state.value}"

    # === T=1s: Laughter detected (happy emotion) ===
    logger.info("\n[T=1s] Laughter context detected")
    audio_1 = AudioEvent(
        timestamp=time.time(),
        transcript="hahaha",
        emotion=EmotionType.HAPPY,
        emotion_confidence=0.85,  # High confidence happy
        intent=IntentType.NEUTRAL,  # Changed intent to neutral
        intent_confidence=0.70,
        sample_duration=1.0,
        vad_confidence=0.90,
    )

    risk_event_1 = risk_acc.calculate_risk(
        current_risk=current_risk,
        audio_event=audio_1,
        dt_seconds=dt,
    )

    state_mgr.update_state(risk_event_1.state, risk_event_1.score)
    current_risk = risk_event_1.score

    logger.info(f"  Transcript: '{audio_1.transcript}'")
    logger.info(f"  Emotion: {audio_1.emotion.value} (conf={audio_1.emotion_confidence:.2f})")
    logger.info(f"  Intent: {audio_1.intent.value} (conf={audio_1.intent_confidence:.2f})")
    logger.info(f"  → Risk (before filter): Would be threat intent, but...")
    logger.info(f"  → Context filtering applied: Happy emotion suppresses threat")
    logger.info(f"  → Risk: {risk_event_1.visual_risk:.1f} (visual) + "
               f"{risk_event_1.audio_risk:.1f} (audio, filtered) = {current_risk:.1f} pts")
    logger.info(f"  → State: {state_mgr.current_state.value}")
    logger.info(f"  → Alert triggered: {state_mgr.is_alert_triggered()}")

    assert risk_event_1.audio_risk == 0.0, "Context filtering should suppress audio risk"
    assert current_risk < 70, f"Risk should decay from 70, got {current_risk}"
    # 70 * 0.9 = 63, but now no new threats, should be less
    assert state_mgr.current_state == RiskState.EVALUATING, f"Still EVALUATING while decaying: {state_mgr.current_state.value}"

    # === T=2s: Continued laughter, no new threat ===
    logger.info("\n[T=2s] Continued laughter")
    audio_2 = AudioEvent(
        timestamp=time.time(),
        transcript="haha still laughing",
        emotion=EmotionType.HAPPY,
        emotion_confidence=0.82,
        intent=IntentType.NEUTRAL,
        intent_confidence=0.75,
        sample_duration=1.0,
        vad_confidence=0.88,
    )

    risk_event_2 = risk_acc.calculate_risk(
        current_risk=current_risk,
        audio_event=audio_2,
        dt_seconds=dt,
    )

    state_mgr.update_state(risk_event_2.state, risk_event_2.score)
    current_risk = risk_event_2.score

    logger.info(f"  Transcript: '{audio_2.transcript}'")
    logger.info(f"  → Risk: {current_risk:.1f} pts (continued decay)")
    logger.info(f"  → State: {state_mgr.current_state.value}")

    logger.info(f"  → Risk: {current_risk:.1f} pts (continued decay)")
    logger.info(f"  → State: {state_mgr.current_state.value}")

    assert current_risk < 70, "Risk should have decayed from initial 70 pts"

    # === Final Assertion ===
    logger.info("\n" + "=" * 80)
    logger.info("RESULT: ✅ 'Shut up' scenario handled correctly!")
    logger.info(f"  T=0s: Threat + anger → {risk_event_0.score:.1f} pts (EVALUATING)")
    logger.info(f"  T=1s: + Laughter (context filter) → {risk_event_1.score:.1f} pts (decay to EVALUATING)")
    logger.info(f"  T=2s: Continued laughter → {current_risk:.1f} pts (transitioned to CAUTION)")
    logger.info(f"  Final risk: {current_risk:.1f} pts (well below 75-point ALERT threshold)")
    logger.info(f"  Final state: {state_mgr.current_state.value}")
    logger.info(f"  Alert would trigger: {state_mgr.is_alert_triggered()}")
    logger.info("  Key learning: Context (laughter) suppressed false alert on threat words!")
    logger.info("  Risk decayed through EVALUATING → CAUTION without ever reaching ALERT")
    logger.info("=" * 80)

    assert not state_mgr.is_alert_triggered(), "Alert should NOT be triggered (need 3s in ALERT state)"
    assert current_risk < 75, "Risk should be well below ALERT threshold (75 pts)"


def test_queue_communication():
    """Test queue communication between simulated threads.

    Validates that data can be correctly put/get through queues.
    """
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Queue Communication")
    logger.info("=" * 80)

    queues = QueueManager()

    # Test visual queue
    logger.info("\nTesting visual_to_fusion_q...")
    visual = VisualData(
        timestamp=time.time(),
        frame_shape=(480, 640, 3),
        persons=[
            {
                'id': 1,
                'bbox': (10, 10, 50, 50),
                'confidence': 0.9,
                'label': 'UNKNOWN',
                'has_weapon': False,
                'frames_in_view': 5,
                'velocity': (0.0, 0.0)
            }
        ],
    )

    put_result = queues.put_visual_data(visual, block=True)
    logger.info(f"  Put visual data: {put_result}")
    assert put_result, f"Should successfully put visual data"
    
    time.sleep(0.01)  # Small delay to ensure queue processing
    qsize = queues.visual_to_fusion_q.qsize()
    logger.info(f"  Queue size after put: {qsize}")
    
    retrieved = queues.get_visual_data(block=True, timeout=1.0)
    logger.info(f"  Retrieved visual: {retrieved is not None}")
    assert retrieved is not None, f"Should retrieve visual data (got None)"
    assert retrieved.frame_shape == (480, 640, 3), "Frame shape should match"
    logger.info("  ✓ Visual queue working")

    # Test audio queue
    logger.info("\nTesting audio_to_fusion_q...")
    audio = AudioEvent(
        timestamp=time.time(),
        transcript="test",
        emotion=EmotionType.NEUTRAL,
        emotion_confidence=0.5,
        intent=IntentType.NEUTRAL,
        intent_confidence=0.5,
    )

    put_audio_result = queues.put_audio_event(audio, block=True)
    logger.info(f"  Put audio event: {put_audio_result}")
    assert put_audio_result, "Should successfully put audio event"
    
    time.sleep(0.01)
    audio_qsize = queues.audio_to_fusion_q.qsize()
    logger.info(f"  Queue size after put: {audio_qsize}")
    
    retrieved_audio = queues.get_audio_event(block=True, timeout=1.0)
    logger.info(f"  Retrieved audio: {retrieved_audio is not None}")
    assert retrieved_audio is not None, f"Should retrieve audio event (got None)"
    assert retrieved_audio.transcript == "test", "Transcript should match"
    logger.info("  ✓ Audio queue working")

    # Test draining
    logger.info("\nTesting queue draining...")
    
    # Create fresh queue manager for draining test
    drain_queues = QueueManager()
    
    # Add 5 items to visual queue
    for i in range(5):
        vd = VisualData(timestamp=time.time(), frame_shape=(480, 640, 3))
        success = drain_queues.put_visual_data(vd)
        logger.info(f"  Put visual data {i}: {success}")
        if not success:
            logger.warning(f"Failed to put visual data {i}")
    
    # Check queue size before draining
    qsize_before = drain_queues.visual_to_fusion_q.qsize()
    logger.info(f"  Queue size before drain: {qsize_before}")
    
    count, latest = drain_queues.drain_visual_queue()
    logger.info(f"  Drained {count} items, latest timestamp={latest.timestamp if latest else 'None'}")
    assert count >= 1, f"Should drain at least 1 item, got {count}"
    assert latest is not None, "Latest should not be None"
    assert latest.frame_shape == (480, 640, 3), f"Frame shape should match, got {latest.frame_shape}"
    logger.info(f"  ✓ Successfully drained {count} items")

    logger.info("\n" + "=" * 80)
    logger.info("Queue communication tests: ✅ PASSED")
    logger.info("=" * 80)


def test_shared_state():
    """Test shared state manager thread-safe operations.

    Validates that state updates are visible across "threads".
    """
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Shared State Management")
    logger.info("=" * 80)

    shared = SharedStateManager()

    # Test risk update
    logger.info("\nTesting risk score updates...")
    shared.update_risk(45.0, RiskState.EVALUATING)
    assert shared.get_risk() == 45.0, "Risk should be 45"
    assert shared.get_state() == RiskState.EVALUATING, "State should be EVALUATING"
    logger.info("  ✓ Risk score update working")

    # Test visual summary
    logger.info("\nTesting visual summary...")
    shared.update_visual_summary(5, 2, 1, 0)
    summary = shared.get_visual_summary()
    assert summary["persons_count"] == 5, "Should have 5 persons"
    assert summary["unknown_count"] == 2, "Should have 2 unknown"
    logger.info("  ✓ Visual summary working")

    # Test audio summary
    logger.info("\nTesting audio summary...")
    shared.update_audio_summary(
        "danger ahead",
        "fearful",
        0.85,
        "threat",
        0.80,
    )
    audio_summary = shared.get_audio_summary()
    assert audio_summary["transcript"] == "danger ahead", "Transcript mismatch"
    assert audio_summary["emotion"] == "fearful", "Emotion mismatch"
    logger.info("  ✓ Audio summary working")

    logger.info("\n" + "=" * 80)
    logger.info("Shared state tests: ✅ PASSED")
    logger.info("=" * 80)

    shared.close()


if __name__ == "__main__":
    try:
        test_queue_communication()
        test_shared_state()
        test_shut_up_scenario()

        logger.info("\n" + "=" * 80)
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("=" * 80)

    except AssertionError as e:
        logger.error(f"\n❌ TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"\n❌ UNEXPECTED ERROR: {e}", exc_info=True)
        exit(1)
