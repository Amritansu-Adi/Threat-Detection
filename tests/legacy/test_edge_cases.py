"""Edge case and robustness testing for the threat detection system - Phase 6.

Tests system behavior under adverse conditions:
- Missing inputs
- Model failures
- Network issues
- Resource constraints
- Invalid data
"""

import time
import threading
import logging
import numpy as np
from unittest.mock import patch, MagicMock, side_effect
from queue import Full, Empty

from core.system_manager import SystemManager
from core.data_structures import AudioEvent, VisualData, TrackedPerson, BoundingBox
from core.data_structures import EmotionType, IntentType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestEdgeCases:
    """Test system robustness under edge conditions."""

    def test_no_audio_input(self):
        """Test system operation with no audio input."""
        logger.info("Testing no audio input scenario...")

        system = SystemManager()

        with patch.object(system, '_init_pipelines') as mock_init:
            system.start()

            # Start fusion manager
            system.fusion_manager.start()

            # Send only visual data for 10 seconds
            for i in range(10):
                visual_data = VisualData(
                    timestamp=time.time(),
                    frame_id=i,
                    persons=[
                        TrackedPerson(
                            track_id=1,
                            bbox=(100, 100, 200, 300),
                            identity="Unknown",
                            confidence=0.8,
                            has_weapon=False
                        )
                    ],
                    weapons_detected=False,
                    risk_score=10.0
                )
                system.queue_manager.visual_to_fusion_q.put(visual_data)
                time.sleep(1.0)

            # System should still function and maintain state
            risk_score = system.shared_state.get_risk_score()
            assert risk_score >= 0  # Should have some risk from visual

            system.fusion_manager.stop()
            system.stop()

        logger.info("✅ No audio input test passed")

    def test_no_visual_input(self):
        """Test system operation with no visual input."""
        logger.info("Testing no visual input scenario...")

        system = SystemManager()

        with patch.object(system, '_init_pipelines') as mock_init:
            system.start()
            system.fusion_manager.start()

            # Send only audio data for 10 seconds
            for i in range(10):
                audio_event = AudioEvent(
                    timestamp=time.time(),
                    transcript="Test audio",
                    emotion=EmotionType.NEUTRAL.value,
                    emotion_confidence=0.7,
                    intent=IntentType.NEUTRAL.value,
                    intent_confidence=0.7
                )
                system.queue_manager.audio_to_fusion_q.put(audio_event)
                time.sleep(1.0)

            # System should still function
            risk_score = system.shared_state.get_risk_score()
            assert risk_score >= 0

            system.fusion_manager.stop()
            system.stop()

        logger.info("✅ No visual input test passed")

    def test_queue_overflow(self):
        """Test behavior when queues overflow."""
        logger.info("Testing queue overflow handling...")

        system = SystemManager()

        # Fill visual queue beyond capacity (maxsize=100)
        for i in range(120):
            visual_data = VisualData(
                timestamp=time.time(),
                frame_id=i,
                persons=[],
                weapons_detected=False,
                risk_score=0.0
            )
            try:
                system.queue_manager.visual_to_fusion_q.put(visual_data, timeout=0.01)
            except Full:
                break  # Queue is full

        # Queue should not exceed maxsize
        queue_size = system.queue_manager.visual_to_fusion_q.qsize()
        assert queue_size <= 100

        # System should still be operational
        assert system.queue_manager.audio_to_fusion_q.qsize() == 0  # Other queues unaffected

        logger.info("✅ Queue overflow test passed")

    def test_invalid_audio_data(self):
        """Test handling of invalid audio data."""
        logger.info("Testing invalid audio data handling...")

        system = SystemManager()

        with patch.object(system, '_init_pipelines') as mock_init:
            system.start()
            system.fusion_manager.start()

            # Send invalid audio events
            invalid_events = [
                # Missing required fields
                {"timestamp": time.time(), "transcript": "test"},
                # Invalid emotion type
                AudioEvent(
                    timestamp=time.time(),
                    transcript="test",
                    emotion="INVALID_EMOTION",
                    emotion_confidence=0.8,
                    intent=IntentType.NEUTRAL.value,
                    intent_confidence=0.7
                ),
                # Negative confidence
                AudioEvent(
                    timestamp=time.time(),
                    transcript="test",
                    emotion=EmotionType.NEUTRAL.value,
                    emotion_confidence=-0.5,
                    intent=IntentType.NEUTRAL.value,
                    intent_confidence=0.7
                )
            ]

            for event in invalid_events:
                try:
                    system.queue_manager.audio_to_fusion_q.put(event)
                    time.sleep(1.1)  # Wait for processing
                except Exception as e:
                    logger.debug(f"Expected error with invalid data: {e}")

            # System should continue operating despite invalid data
            risk_score = system.shared_state.get_risk_score()
            assert isinstance(risk_score, (int, float))

            system.fusion_manager.stop()
            system.stop()

        logger.info("✅ Invalid audio data test passed")

    def test_invalid_visual_data(self):
        """Test handling of invalid visual data."""
        logger.info("Testing invalid visual data handling...")

        system = SystemManager()

        with patch.object(system, '_init_pipelines') as mock_init:
            system.start()
            system.fusion_manager.start()

            # Send invalid visual events
            invalid_events = [
                # Empty persons list with weapons_detected=True
                VisualData(
                    timestamp=time.time(),
                    frame_id=1,
                    persons=[],
                    weapons_detected=True,
                    risk_score=50.0
                ),
                # Person with invalid bbox
                VisualData(
                    timestamp=time.time(),
                    frame_id=2,
                    persons=[
                        TrackedPerson(
                            track_id=1,
                            bbox=(-10, -10, -5, -5),  # Invalid bbox
                            identity="Test",
                            confidence=0.8,
                            has_weapon=False
                        )
                    ],
                    weapons_detected=False,
                    risk_score=10.0
                ),
                # Person with negative confidence
                VisualData(
                    timestamp=time.time(),
                    frame_id=3,
                    persons=[
                        TrackedPerson(
                            track_id=1,
                            bbox=(100, 100, 200, 300),
                            identity="Test",
                            confidence=-0.2,  # Invalid confidence
                            has_weapon=False
                        )
                    ],
                    weapons_detected=False,
                    risk_score=10.0
                )
            ]

            for event in invalid_events:
                try:
                    system.queue_manager.visual_to_fusion_q.put(event)
                    time.sleep(1.1)
                except Exception as e:
                    logger.debug(f"Expected error with invalid data: {e}")

            # System should continue operating
            risk_score = system.shared_state.get_risk_score()
            assert isinstance(risk_score, (int, float))

            system.fusion_manager.stop()
            system.stop()

        logger.info("✅ Invalid visual data test passed")

    def test_component_failure_recovery(self):
        """Test system behavior when components fail."""
        logger.info("Testing component failure recovery...")

        system = SystemManager()

        with patch.object(system, '_init_pipelines') as mock_init:
            system.start()

            # Simulate fusion manager failure
            original_risk_accumulator = system.fusion_manager.risk_accumulator

            # Replace with failing accumulator
            class FailingRiskAccumulator:
                def calculate_risk(self, *args, **kwargs):
                    raise RuntimeError("Simulated accumulator failure")

            system.fusion_manager.risk_accumulator = FailingRiskAccumulator()

            # Send data that would normally cause processing
            audio_event = AudioEvent(
                timestamp=time.time(),
                transcript="Test",
                emotion=EmotionType.ANGER.value,
                emotion_confidence=0.8,
                intent=IntentType.THREAT.value,
                intent_confidence=0.8
            )

            system.queue_manager.audio_to_fusion_q.put(audio_event)

            # System should handle the failure gracefully
            time.sleep(2.0)  # Allow time for error handling

            # Restore working accumulator
            system.fusion_manager.risk_accumulator = original_risk_accumulator

            # Send another event - should work now
            audio_event2 = AudioEvent(
                timestamp=time.time(),
                transcript="Test after recovery",
                emotion=EmotionType.NEUTRAL.value,
                emotion_confidence=0.7,
                intent=IntentType.NEUTRAL.value,
                intent_confidence=0.7
            )

            system.queue_manager.audio_to_fusion_q.put(audio_event2)
            time.sleep(1.1)

            # System should be operational again
            risk_score = system.shared_state.get_risk_score()
            assert isinstance(risk_score, (int, float))

            system.stop()

        logger.info("✅ Component failure recovery test passed")

    def test_concurrent_thread_safety(self):
        """Test thread safety with concurrent operations."""
        logger.info("Testing concurrent thread safety...")

        system = SystemManager()
        results = {"errors": 0, "operations": 0}

        def producer_worker(queue_name, num_events):
            """Producer thread that sends events."""
            for i in range(num_events):
                try:
                    if queue_name == "audio":
                        event = AudioEvent(
                            timestamp=time.time(),
                            transcript=f"Audio {i}",
                            emotion=EmotionType.NEUTRAL.value,
                            emotion_confidence=0.7,
                            intent=IntentType.NEUTRAL.value,
                            intent_confidence=0.7
                        )
                        system.queue_manager.audio_to_fusion_q.put(event)
                    else:  # visual
                        event = VisualData(
                            timestamp=time.time(),
                            frame_id=i,
                            persons=[],
                            weapons_detected=False,
                            risk_score=5.0
                        )
                        system.queue_manager.visual_to_fusion_q.put(event)

                    results["operations"] += 1
                except Exception as e:
                    results["errors"] += 1
                    logger.debug(f"Producer error: {e}")
                time.sleep(0.01)  # Small delay

        with patch.object(system, '_init_pipelines') as mock_init:
            system.start()
            system.fusion_manager.start()

            # Start multiple producer threads
            threads = []
            for i in range(5):
                t1 = threading.Thread(target=producer_worker, args=("audio", 20))
                t2 = threading.Thread(target=producer_worker, args=("visual", 20))
                threads.extend([t1, t2])

            for t in threads:
                t.start()

            for t in threads:
                t.join(timeout=10.0)

            # Allow time for processing
            time.sleep(3.0)

            # Check results
            assert results["operations"] > 0, "No operations completed"
            assert results["errors"] == 0, f"Thread safety errors: {results['errors']}"

            # System should still be operational
            risk_score = system.shared_state.get_risk_score()
            assert isinstance(risk_score, (int, float))

            system.fusion_manager.stop()
            system.stop()

        logger.info("✅ Concurrent thread safety test passed")

    def test_resource_cleanup(self):
        """Test proper resource cleanup on shutdown."""
        logger.info("Testing resource cleanup...")

        system = SystemManager()

        with patch.object(system, '_init_pipelines') as mock_init:
            system.start()

            # Verify system is running
            assert system.running

            # Check that threads are created
            initial_thread_count = threading.active_count()

            # Start components
            system.fusion_manager.start()

            # Verify threads are running
            assert system.fusion_manager._running

            # Shutdown
            system.stop()

            # Verify cleanup
            assert not system.running
            assert not system.fusion_manager._running

            # Allow time for thread cleanup
            time.sleep(1.0)

            # Check that threads are cleaned up (may not be exact due to main thread)
            final_thread_count = threading.active_count()
            assert final_thread_count <= initial_thread_count + 1  # Allow some tolerance

        logger.info("✅ Resource cleanup test passed")

    def test_configuration_validation(self):
        """Test configuration validation edge cases."""
        logger.info("Testing configuration validation...")

        # Test invalid configurations
        invalid_configs = [
            {"system": {"target_fps": 0}},  # Invalid FPS
            {"system": {"fusion_hz": -1}},  # Negative Hz
            {"fusion": {"risk_decay_alpha": 1.5}},  # Alpha > 1
            {"visual": {"person_conf": -0.1}},  # Negative confidence
            {"audio": {"sample_rate": 0}},  # Zero sample rate
        ]

        from core.config_loader import ConfigLoader

        for invalid_config in invalid_configs:
            loader = ConfigLoader()
            loader.config = invalid_config

            # Should raise ValueError for invalid config
            try:
                loader._validate_config()
                assert False, f"Should have failed validation for: {invalid_config}"
            except ValueError:
                pass  # Expected

        logger.info("✅ Configuration validation test passed")


def run_edge_case_tests():
    """Run all edge case tests."""
    print("🔥 Running Phase 6: Testing & Validation - Edge Case Tests")
    print("=" * 60)

    test_instance = TestEdgeCases()
    methods = [method for method in dir(test_instance) if method.startswith('test_')]

    total_tests = len(methods)
    passed_tests = 0

    for method_name in methods:
        try:
            print(f"\n🧪 Running {method_name}...")
            getattr(test_instance, method_name)()
            passed_tests += 1
            print(f"✅ {method_name} PASSED")
        except Exception as e:
            print(f"❌ {method_name} FAILED: {e}")

    print("\n" + "=" * 60)
    print(f"📊 Edge Case Test Results: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 All edge case tests passed!")
        return True
    else:
        print("⚠️  Some edge case tests failed. Check logs for details.")
        return False


if __name__ == "__main__":
    success = run_edge_case_tests()
    exit(0 if success else 1)