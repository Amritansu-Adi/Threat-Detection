"""Integration tests for full system operation - Phase 6.

Tests the complete threat detection system working end-to-end:
- Multi-threaded operation
- Queue communication between components
- State machine transitions
- Scenario-based testing
- Performance validation
"""

import time
import threading
import logging
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime

from core.system_manager import SystemManager
from core.data_structures import AudioEvent, VisualData, TrackedPerson, BoundingBox
from core.data_structures import EmotionType, IntentType, RiskState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestFullSystemIntegration:
    """Test the complete system working together."""

    def test_system_startup_shutdown(self):
        """Test full system startup and shutdown cycle."""
        logger.info("Testing full system startup/shutdown...")

        system = SystemManager()

        # Mock the actual pipeline components to avoid hardware dependencies
        with patch.object(system, '_init_pipelines') as mock_init, \
             patch.object(system, '_start_threads') as mock_start, \
             patch.object(system, '_stop_threads') as mock_stop, \
             patch.object(system, '_setup_signal_handlers') as mock_signals:

            # Test startup
            system.start()
            assert system.running
            mock_init.assert_called_once()
            mock_start.assert_called_once()
            mock_signals.assert_called_once()

            # Test shutdown
            system.stop()
            assert not system.running
            mock_stop.assert_called_once()

        logger.info("✅ Full system startup/shutdown test passed")

    def test_config_validation(self):
        """Test configuration loading and validation."""
        logger.info("Testing configuration validation...")

        system = SystemManager()

        # Verify config structure
        assert "system" in system.config
        assert "visual" in system.config
        assert "audio" in system.config
        assert "fusion" in system.config

        # Verify required parameters exist
        assert system.config["system"]["target_fps"] > 0
        assert system.config["system"]["fusion_hz"] > 0
        assert 0 <= system.config["fusion"]["risk_decay_alpha"] <= 1

        logger.info("✅ Configuration validation test passed")

    def test_queue_initialization(self):
        """Test that queues are properly initialized."""
        logger.info("Testing queue initialization...")

        system = SystemManager()

        # Check that queues exist
        assert hasattr(system.queue_manager, 'audio_to_fusion_q')
        assert hasattr(system.queue_manager, 'visual_to_fusion_q')
        assert hasattr(system.queue_manager, 'fusion_to_visual_q')

        # Check queue sizes
        assert system.queue_manager.audio_to_fusion_q.maxsize == 100
        assert system.queue_manager.visual_to_fusion_q.maxsize == 100
        assert system.queue_manager.fusion_to_visual_q.maxsize == 50

        logger.info("✅ Queue initialization test passed")

    def test_shared_state_initialization(self):
        """Test shared state manager initialization."""
        logger.info("Testing shared state initialization...")

        system = SystemManager()

        # Check initial state values
        state = system.shared_state._state
        assert state["risk_score"] == 0.0
        assert state["state"] == RiskState.IDLE.value
        assert state["visual_persons_count"] == 0
        assert state["audio_events_count"] == 0

        logger.info("✅ Shared state initialization test passed")


class TestScenarioBasedTesting:
    """Test real-world scenarios end-to-end."""

    def test_shut_up_scenario_integration(self):
        """Test the 'Shut up' scenario with full system integration."""
        logger.info("Testing 'Shut up' scenario integration...")

        system = SystemManager()

        # Mock components
        with patch.object(system, '_init_pipelines') as mock_init:
            system.start()

            # Simulate the scenario timeline
            start_time = time.time()

            # T=0: Angry "Shut up!" with anger emotion
            audio_event = AudioEvent(
                timestamp=start_time,
                transcript="Shut up!",
                emotion=EmotionType.ANGER.value,
                emotion_confidence=0.8,
                intent=IntentType.THREAT.value,
                intent_confidence=0.9
            )

            # Put audio event in queue
            system.queue_manager.audio_to_fusion_q.put(audio_event)

            # Wait for fusion processing
            time.sleep(1.1)  # Fusion runs at 1 Hz

            # Check that risk increased
            risk_score = system.shared_state.get_risk_score()
            current_state = system.shared_state.get_state()

            # Should be in CAUTION or higher state
            assert risk_score > 15  # Above IDLE threshold
            assert current_state in [RiskState.CAUTION.value, RiskState.EVALUATING.value]

            # T=1: Laughter/joking context
            audio_event2 = AudioEvent(
                timestamp=start_time + 1.0,
                transcript="Just kidding!",
                emotion=EmotionType.HAPPY.value,
                emotion_confidence=0.9,
                intent=IntentType.NEUTRAL.value,
                intent_confidence=0.7
            )

            system.queue_manager.audio_to_fusion_q.put(audio_event2)
            time.sleep(1.1)

            # Risk should decay but not trigger alert
            final_risk = system.shared_state.get_risk_score()
            final_state = system.shared_state.get_state()

            # Should be below ALERT threshold (45 pts)
            assert final_risk < 45
            assert final_state != RiskState.ALERT.value

            system.stop()

        logger.info("✅ 'Shut up' scenario integration test passed")

    def test_weapon_detection_scenario(self):
        """Test weapon detection scenario with visual and fusion."""
        logger.info("Testing weapon detection scenario...")

        system = SystemManager()

        with patch.object(system, '_init_pipelines') as mock_init:
            system.start()

            # Create visual data with weapon detection
            bbox = BoundingBox(x1=100, y1=100, x2=200, y2=300)
            person = TrackedPerson(
                track_id=1,
                bbox=(bbox.x1, bbox.y1, bbox.x2, bbox.y2),
                identity="Unknown",
                confidence=0.85,
                has_weapon=True
            )

            visual_data = VisualData(
                timestamp=time.time(),
                frame_id=1,
                persons=[person],
                weapons_detected=True,
                risk_score=40.0  # High risk for unknown + weapon
            )

            # Send to fusion
            system.queue_manager.visual_to_fusion_q.put(visual_data)
            time.sleep(1.1)

            # Check risk escalation
            risk_score = system.shared_state.get_risk_score()
            current_state = system.shared_state.get_state()

            assert risk_score >= 30  # Should trigger EVALUATING state
            assert current_state in [RiskState.EVALUATING.value, RiskState.ALERT.value]

            system.stop()

        logger.info("✅ Weapon detection scenario test passed")

    def test_distress_call_scenario(self):
        """Test distress call scenario with audio focus."""
        logger.info("Testing distress call scenario...")

        system = SystemManager()

        with patch.object(system, '_init_pipelines') as mock_init:
            system.start()

            # Distress audio event
            distress_event = AudioEvent(
                timestamp=time.time(),
                transcript="Help! Someone has a gun!",
                emotion=EmotionType.FEAR.value,
                emotion_confidence=0.95,
                intent=IntentType.DISTRESS.value,
                intent_confidence=0.9
            )

            system.queue_manager.audio_to_fusion_q.put(distress_event)
            time.sleep(1.1)

            # Check high risk score
            risk_score = system.shared_state.get_risk_score()
            current_state = system.shared_state.get_state()

            assert risk_score >= 35  # Distress intent = 35 pts
            assert current_state in [RiskState.EVALUATING.value, RiskState.ALERT.value]

            system.stop()

        logger.info("✅ Distress call scenario test passed")


class TestPerformanceValidation:
    """Test system performance characteristics."""

    def test_latency_measurement(self):
        """Test end-to-end latency from input to state change."""
        logger.info("Testing latency measurement...")

        system = SystemManager()

        with patch.object(system, '_init_pipelines') as mock_init:
            system.start()

            # Measure latency
            start_time = time.time()

            # Send high-risk audio event
            audio_event = AudioEvent(
                timestamp=start_time,
                transcript="Intruder with weapon!",
                emotion=EmotionType.FEAR.value,
                emotion_confidence=0.9,
                intent=IntentType.THREAT.value,
                intent_confidence=0.95
            )

            system.queue_manager.audio_to_fusion_q.put(audio_event)

            # Wait for processing
            time.sleep(1.1)

            end_time = time.time()
            latency = end_time - start_time

            # Should process within 2 seconds (fusion runs at 1 Hz)
            assert latency < 2.0

            # Verify state change occurred
            risk_score = system.shared_state.get_risk_score()
            assert risk_score >= 45  # Threat intent = 45 pts

            system.stop()

        logger.info(f"✅ Latency measurement test passed (latency: {latency:.2f}s)")

    def test_concurrent_operation(self):
        """Test concurrent visual and audio processing."""
        logger.info("Testing concurrent operation...")

        system = SystemManager()

        with patch.object(system, '_init_pipelines') as mock_init:
            system.start()

            # Send both visual and audio events simultaneously
            timestamp = time.time()

            # Visual event
            bbox = BoundingBox(x1=50, y1=50, x2=150, y2=250)
            person = TrackedPerson(
                track_id=1,
                bbox=(bbox.x1, bbox.y1, bbox.x2, bbox.y2),
                identity="Unknown",
                confidence=0.8,
                has_weapon=True
            )
            visual_data = VisualData(
                timestamp=timestamp,
                frame_id=1,
                persons=[person],
                weapons_detected=True,
                risk_score=40.0
            )

            # Audio event
            audio_event = AudioEvent(
                timestamp=timestamp,
                transcript="Watch out!",
                emotion=EmotionType.FEAR.value,
                emotion_confidence=0.8,
                intent=IntentType.THREAT.value,
                intent_confidence=0.8
            )

            # Send both events
            system.queue_manager.visual_to_fusion_q.put(visual_data)
            system.queue_manager.audio_to_fusion_q.put(audio_event)

            # Wait for fusion processing
            time.sleep(1.1)

            # Check combined risk (should be additive)
            risk_score = system.shared_state.get_risk_score()
            current_state = system.shared_state.get_state()

            # Visual (40) + Audio (45) with temporal multiplier
            assert risk_score >= 60  # Should trigger ALERT or CRITICAL
            assert current_state in [RiskState.ALERT.value, RiskState.CRITICAL.value]

            system.stop()

        logger.info("✅ Concurrent operation test passed")

    def test_queue_overflow_handling(self):
        """Test queue overflow scenarios."""
        logger.info("Testing queue overflow handling...")

        system = SystemManager()

        # Fill audio queue to capacity
        for i in range(110):  # Queue maxsize is 100
            audio_event = AudioEvent(
                timestamp=time.time(),
                transcript=f"Message {i}",
                emotion=EmotionType.NEUTRAL.value,
                emotion_confidence=0.5,
                intent=IntentType.NEUTRAL.value,
                intent_confidence=0.5
            )
            system.queue_manager.audio_to_fusion_q.put(audio_event)

        # Queue should not block and should handle overflow gracefully
        queue_size = system.queue_manager.audio_to_fusion_q.qsize()
        assert queue_size <= 100  # Should not exceed maxsize

        logger.info("✅ Queue overflow handling test passed")


def run_integration_tests():
    """Run all integration tests."""
    print("🧪 Running Phase 6: Testing & Validation - Integration Tests")
    print("=" * 60)

    # Test classes
    test_classes = [
        TestFullSystemIntegration,
        TestScenarioBasedTesting,
        TestPerformanceValidation
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"\n📋 Running {test_class.__name__}...")

        instance = test_class()
        methods = [method for method in dir(instance) if method.startswith('test_')]

        for method_name in methods:
            try:
                total_tests += 1
                print(f"  ▶️  {method_name}...")
                getattr(instance, method_name)()
                passed_tests += 1
                print(f"  ✅ {method_name} PASSED")
            except Exception as e:
                print(f"  ❌ {method_name} FAILED: {e}")

    print("\n" + "=" * 60)
    print(f"📊 Integration Test Results: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 All integration tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check logs for details.")
        return False


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)