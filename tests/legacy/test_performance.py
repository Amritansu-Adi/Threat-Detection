"""Performance testing for the threat detection system - Phase 6.

Measures:
- CPU usage per component
- Memory usage over time
- End-to-end latency
- Queue throughput
- Sustained operation stability
"""

import time
import threading
import psutil
import logging
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta
import gc

from core.system_manager import SystemManager
from core.data_structures import AudioEvent, VisualData, TrackedPerson, BoundingBox
from core.data_structures import EmotionType, IntentType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor system performance metrics."""

    def __init__(self):
        self.cpu_samples = []
        self.memory_samples = []
        self.start_time = None
        self.end_time = None

    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.cpu_samples = []
        self.memory_samples = []

    def record_sample(self):
        """Record current CPU and memory usage."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_mb = memory.used / (1024 * 1024)

        self.cpu_samples.append(cpu_percent)
        self.memory_samples.append(memory_mb)

    def stop_monitoring(self):
        """Stop monitoring and return summary."""
        self.end_time = time.time()

        if not self.cpu_samples:
            return {"error": "No samples recorded"}

        duration = self.end_time - self.start_time

        return {
            "duration_sec": duration,
            "cpu_avg": np.mean(self.cpu_samples),
            "cpu_max": np.max(self.cpu_samples),
            "cpu_min": np.min(self.cpu_samples),
            "memory_avg_mb": np.mean(self.memory_samples),
            "memory_max_mb": np.max(self.memory_samples),
            "memory_min_mb": np.min(self.memory_samples),
            "samples_count": len(self.cpu_samples)
        }


class TestPerformanceMetrics:
    """Test system performance characteristics."""

    def __init__(self):
        self.monitor = PerformanceMonitor()

    def test_baseline_performance(self):
        """Test baseline system performance without load."""
        logger.info("Testing baseline performance...")

        system = SystemManager()
        self.monitor.start_monitoring()

        # Monitor for 10 seconds with no activity
        for i in range(100):
            self.monitor.record_sample()
            time.sleep(0.1)

        baseline = self.monitor.stop_monitoring()

        logger.info(f"Baseline CPU: {baseline['cpu_avg']:.1f}% avg, {baseline['cpu_max']:.1f}% max")
        logger.info(f"Baseline Memory: {baseline['memory_avg_mb']:.0f}MB avg, {baseline['memory_max_mb']:.0f}MB max")

        # Assert reasonable baseline performance
        assert baseline['cpu_avg'] < 20  # Should be low CPU at idle
        assert baseline['memory_max_mb'] < 1000  # Should be reasonable memory usage

        logger.info("✅ Baseline performance test passed")
        return baseline

    def test_visual_processing_performance(self):
        """Test visual pipeline performance under load."""
        logger.info("Testing visual processing performance...")

        system = SystemManager()
        self.monitor.start_monitoring()

        with system.queue_manager.audio_to_fusion_q.mutex:  # Prevent fusion from running
            # Send 100 visual frames
            for frame_id in range(100):
                # Create realistic visual data
                persons = []
                for person_id in range(np.random.randint(0, 5)):  # 0-4 persons per frame
                    bbox = BoundingBox(
                        x1=np.random.randint(0, 800),
                        y1=np.random.randint(0, 600),
                        x2=np.random.randint(400, 1200),
                        y2=np.random.randint(300, 900)
                    )
                    person = TrackedPerson(
                        track_id=person_id,
                        bbox=(bbox.x1, bbox.y1, bbox.x2, bbox.y2),
                        identity=f"Person_{person_id}" if np.random.random() > 0.5 else "Unknown",
                        confidence=np.random.uniform(0.6, 0.95),
                        has_weapon=np.random.random() > 0.8  # 20% chance of weapon
                    )
                    persons.append(person)

                visual_data = VisualData(
                    timestamp=time.time(),
                    frame_id=frame_id,
                    persons=persons,
                    weapons_detected=len([p for p in persons if p.has_weapon]) > 0,
                    risk_score=np.random.uniform(0, 50)
                )

                system.queue_manager.visual_to_fusion_q.put(visual_data)
                self.monitor.record_sample()

        performance = self.monitor.stop_monitoring()

        logger.info(f"Visual processing CPU: {performance['cpu_avg']:.1f}% avg")
        logger.info(f"Visual processing Memory: {performance['memory_max_mb']:.0f}MB max")

        # Assert reasonable performance
        assert performance['cpu_avg'] < 80  # Should not saturate CPU
        assert performance['memory_max_mb'] < 1500  # Should not use excessive memory

        logger.info("✅ Visual processing performance test passed")
        return performance

    def test_audio_processing_performance(self):
        """Test audio pipeline performance under load."""
        logger.info("Testing audio processing performance...")

        system = SystemManager()
        self.monitor.start_monitoring()

        with system.queue_manager.visual_to_fusion_q.mutex:  # Prevent fusion from running
            # Send 50 audio events (simulating 50 seconds of audio at ~1 event/sec)
            for i in range(50):
                audio_event = AudioEvent(
                    timestamp=time.time(),
                    transcript=f"Audio event {i}",
                    emotion=np.random.choice([
                        EmotionType.NEUTRAL.value,
                        EmotionType.HAPPY.value,
                        EmotionType.ANGER.value,
                        EmotionType.FEAR.value
                    ]),
                    emotion_confidence=np.random.uniform(0.7, 0.95),
                    intent=np.random.choice([
                        IntentType.NEUTRAL.value,
                        IntentType.THREAT.value,
                        IntentType.DISTRESS.value
                    ]),
                    intent_confidence=np.random.uniform(0.6, 0.9)
                )

                system.queue_manager.audio_to_fusion_q.put(audio_event)
                self.monitor.record_sample()
                time.sleep(0.02)  # Simulate audio processing time

        performance = self.monitor.stop_monitoring()

        logger.info(f"Audio processing CPU: {performance['cpu_avg']:.1f}% avg")
        logger.info(f"Audio processing Memory: {performance['memory_max_mb']:.0f}MB max")

        assert performance['cpu_avg'] < 70
        assert performance['memory_max_mb'] < 1200

        logger.info("✅ Audio processing performance test passed")
        return performance

    def test_fusion_engine_performance(self):
        """Test fusion engine performance under load."""
        logger.info("Testing fusion engine performance...")

        system = SystemManager()
        self.monitor.start_monitoring()

        # Start fusion manager
        system.fusion_manager.start()

        # Send mixed visual and audio events
        for i in range(30):  # 30 seconds of data
            # Visual event
            if i % 3 == 0:  # Every 3rd iteration
                visual_data = VisualData(
                    timestamp=time.time(),
                    frame_id=i,
                    persons=[
                        TrackedPerson(
                            track_id=1,
                            bbox=(100, 100, 200, 300),
                            identity="Unknown",
                            confidence=0.8,
                            has_weapon=(i > 15)  # Weapon appears halfway through
                        )
                    ],
                    weapons_detected=(i > 15),
                    risk_score=30.0 if i > 15 else 10.0
                )
                system.queue_manager.visual_to_fusion_q.put(visual_data)

            # Audio event
            if i % 2 == 0:  # Every 2nd iteration
                audio_event = AudioEvent(
                    timestamp=time.time(),
                    transcript="Test audio",
                    emotion=EmotionType.ANGER.value if i > 20 else EmotionType.NEUTRAL.value,
                    emotion_confidence=0.8,
                    intent=IntentType.THREAT.value if i > 20 else IntentType.NEUTRAL.value,
                    intent_confidence=0.8
                )
                system.queue_manager.audio_to_fusion_q.put(audio_event)

            self.monitor.record_sample()
            time.sleep(1.0)  # Wait for fusion cycle

        system.fusion_manager.stop()
        performance = self.monitor.stop_monitoring()

        logger.info(f"Fusion engine CPU: {performance['cpu_avg']:.1f}% avg")
        logger.info(f"Fusion engine Memory: {performance['memory_max_mb']:.0f}MB max")

        # Check that risk escalated during the test
        final_risk = system.shared_state.get_risk_score()
        assert final_risk > 40  # Should have triggered alerts

        assert performance['cpu_avg'] < 60
        assert performance['memory_max_mb'] < 1300

        logger.info("✅ Fusion engine performance test passed")
        return performance

    def test_end_to_end_latency(self):
        """Test end-to-end latency from input to state change."""
        logger.info("Testing end-to-end latency...")

        system = SystemManager()
        latencies = []

        # Start fusion manager
        system.fusion_manager.start()

        # Test 10 events
        for i in range(10):
            start_time = time.time()

            # Send high-priority threat event
            audio_event = AudioEvent(
                timestamp=start_time,
                transcript="Immediate threat!",
                emotion=EmotionType.FEAR.value,
                emotion_confidence=0.95,
                intent=IntentType.DISTRESS.value,
                intent_confidence=0.95
            )

            system.queue_manager.audio_to_fusion_q.put(audio_event)

            # Wait for state change
            initial_state = system.shared_state.get_state()
            timeout = 5.0  # Max wait time
            elapsed = 0

            while elapsed < timeout:
                current_state = system.shared_state.get_state()
                if current_state != initial_state:
                    break
                time.sleep(0.1)
                elapsed += 0.1

            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)

            # Reset for next test
            system.shared_state._state["risk_score"] = 0.0
            system.shared_state._state["state"] = "IDLE"

        system.fusion_manager.stop()

        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        min_latency = np.min(latencies)

        logger.info(f"End-to-end latency: {avg_latency:.2f}s avg, {max_latency:.2f}s max, {min_latency:.2f}s min")

        # Assert reasonable latency (< 2 seconds for 1Hz fusion)
        assert avg_latency < 2.0
        assert max_latency < 3.0

        logger.info("✅ End-to-end latency test passed")
        return {
            "avg_latency": avg_latency,
            "max_latency": max_latency,
            "min_latency": min_latency
        }

    def test_memory_leak_detection(self):
        """Test for memory leaks during sustained operation."""
        logger.info("Testing memory leak detection...")

        system = SystemManager()
        memory_readings = []

        # Start fusion manager
        system.fusion_manager.start()

        # Run for 60 seconds, checking memory every 5 seconds
        for i in range(12):
            # Send some events to keep system active
            if i % 2 == 0:
                audio_event = AudioEvent(
                    timestamp=time.time(),
                    transcript="Memory test",
                    emotion=EmotionType.NEUTRAL.value,
                    emotion_confidence=0.7,
                    intent=IntentType.NEUTRAL.value,
                    intent_confidence=0.7
                )
                system.queue_manager.audio_to_fusion_q.put(audio_event)

            time.sleep(5)
            memory = psutil.virtual_memory()
            memory_readings.append(memory.used / (1024 * 1024))  # MB

        system.fusion_manager.stop()

        # Check for memory growth trend
        if len(memory_readings) >= 3:
            early_avg = np.mean(memory_readings[:3])
            late_avg = np.mean(memory_readings[-3:])

            memory_growth = late_avg - early_avg
            memory_growth_rate = memory_growth / 60  # MB per minute

            logger.info(f"Memory growth: {memory_growth:.1f}MB over 60s ({memory_growth_rate:.2f}MB/min)")

            # Assert no significant memory leak (> 50MB growth is concerning)
            assert abs(memory_growth) < 50, f"Potential memory leak detected: {memory_growth:.1f}MB growth"

        logger.info("✅ Memory leak detection test passed")
        return memory_readings


def run_performance_tests():
    """Run all performance tests."""
    print("⚡ Running Phase 6: Testing & Validation - Performance Tests")
    print("=" * 60)

    tester = TestPerformanceMetrics()
    results = {}

    test_methods = [
        'test_baseline_performance',
        'test_visual_processing_performance',
        'test_audio_processing_performance',
        'test_fusion_engine_performance',
        'test_end_to_end_latency',
        'test_memory_leak_detection'
    ]

    total_tests = len(test_methods)
    passed_tests = 0

    for method_name in test_methods:
        try:
            print(f"\n📊 Running {method_name}...")
            result = getattr(tester, method_name)()
            results[method_name] = result
            passed_tests += 1
            print(f"✅ {method_name} PASSED")
        except Exception as e:
            print(f"❌ {method_name} FAILED: {e}")
            results[method_name] = {"error": str(e)}

    print("\n" + "=" * 60)
    print(f"📊 Performance Test Results: {passed_tests}/{total_tests} tests passed")

    # Print summary
    print("\n📈 Performance Summary:")
    for test_name, result in results.items():
        if "error" not in result:
            if "cpu_avg" in result:
                print(f"  {test_name}: CPU {result['cpu_avg']:.1f}%, Memory {result['memory_max_mb']:.0f}MB")
            elif "avg_latency" in result:
                print(f"  {test_name}: Latency {result['avg_latency']:.2f}s avg")
            elif isinstance(result, list):
                print(f"  {test_name}: Memory stable ({len(result)} readings)")

    if passed_tests == total_tests:
        print("🎉 All performance tests passed!")
        return True
    else:
        print("⚠️  Some performance tests failed. Check logs for details.")
        return False


if __name__ == "__main__":
    success = run_performance_tests()
    exit(0 if success else 1)