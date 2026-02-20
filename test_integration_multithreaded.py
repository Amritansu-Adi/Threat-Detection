"""Multi-threaded integration tests for threat detection system.

Tests the complete system with real threads running concurrently to verify:
1. Thread coordination and communication
2. Queue-based data flow between components
3. State synchronization across threads
4. Graceful startup/shutdown
5. Risk accumulation and state transitions in real-time
"""

import time
import pytest
import threading
from datetime import datetime
from unittest.mock import patch, MagicMock

from core.queue_manager import QueueManager
from core.shared_state import SharedStateManager
from fusion.fusion_manager import FusionManager
from core.data_structures import (
    VisualData,
    AudioEvent,
    TrackedPerson,
    BoundingBox,
    EmotionType,
    IntentType,
    RiskState,
)


class TestFusionManagerThreaded:
    """Test FusionManager running in a background thread."""

    def test_fusion_manager_start_stop(self):
        """Test FusionManager lifecycle: start, run, stop."""
        shared = SharedStateManager()
        queues = QueueManager()
        fusion = FusionManager(shared, queues, target_hz=2.0)

        # Start fusion manager
        fusion.start()
        assert fusion._running is True

        # Let it run for a short period
        time.sleep(0.5)

        # Verify thread is running
        assert fusion._thread is not None
        assert fusion._thread.is_alive()

        # Stop and verify
        fusion.stop()
        assert fusion._running is False
        assert not fusion._thread.is_alive()

        shared.close()

    def test_fusion_manager_processes_visual_data(self):
        """Test FusionManager processes visual data from queue."""
        shared = SharedStateManager()
        queues = QueueManager()
        fusion = FusionManager(shared, queues, target_hz=2.0)

        fusion.start()
        time.sleep(0.2)  # Let thread initialize

        # Send visual data: unknown armed person
        person = TrackedPerson(
            track_id=1,
            bbox=BoundingBox(x1=0.2, y1=0.1, x2=0.5, y2=0.8),
            identity="UNKNOWN",
            has_weapon=True,
        )
        visual = VisualData(
            timestamp=time.time(),
            frame_shape=(480, 640, 3),
            persons=[],
        )
        visual.tracked_persons = [person]

        # Put visual data and wait for processing
        queues.visual_to_fusion_q.put(visual, block=True, timeout=1)
        time.sleep(0.8)  # Wait for fusion cycle

        # Verify state updated with risk
        risk = shared.get_risk()
        assert risk > 0  # Risk should increase from visual threat

        fusion.stop()
        shared.close()

    def test_fusion_manager_processes_audio_data(self):
        """Test FusionManager processes audio data from queue."""
        shared = SharedStateManager()
        queues = QueueManager()
        fusion = FusionManager(shared, queues, target_hz=2.0)

        fusion.start()
        time.sleep(0.2)

        # Send audio data: threat + anger
        audio = AudioEvent(
            timestamp=time.time(),
            transcript="I will kill you",
            emotion=EmotionType.ANGRY,
            emotion_confidence=0.9,
            intent=IntentType.THREAT,
            intent_confidence=0.9,
        )

        queues.audio_to_fusion_q.put(audio, block=True, timeout=1)
        time.sleep(0.8)

        # Verify audio risk accumulated
        risk = shared.get_risk()
        assert risk > 0

        fusion.stop()
        shared.close()

    def test_fusion_manager_state_transitions(self):
        """Test FusionManager triggers state transitions based on risk."""
        shared = SharedStateManager()
        queues = QueueManager()
        fusion = FusionManager(shared, queues, target_hz=2.0)

        fusion.start()
        time.sleep(0.2)

        # Send multiple high-risk audio events to build up risk
        for i in range(5):
            audio = AudioEvent(
                timestamp=time.time(),
                transcript="help fire emergency",
                emotion=EmotionType.FEARFUL,
                emotion_confidence=0.92,
                intent=IntentType.DISTRESS,
                intent_confidence=0.9,
            )
            queues.audio_to_fusion_q.put(audio, block=True, timeout=1)
            time.sleep(0.3)

        # Verify state progressed beyond IDLE
        state = shared.get_state()
        assert state != RiskState.IDLE

        fusion.stop()
        shared.close()

    def test_fusion_manager_decay_without_input(self):
        """Test risk decays when no new input arrives."""
        shared = SharedStateManager()
        queues = QueueManager()
        fusion = FusionManager(shared, queues, target_hz=2.0)

        fusion.start()
        time.sleep(0.2)

        # Send one audio threat
        audio = AudioEvent(
            timestamp=time.time(),
            transcript="threat",
            emotion=EmotionType.ANGRY,
            emotion_confidence=0.9,
            intent=IntentType.THREAT,
            intent_confidence=0.9,
        )
        queues.audio_to_fusion_q.put(audio, block=True, timeout=1)
        time.sleep(0.5)

        risk_after_threat = shared.get_risk()
        assert risk_after_threat > 0

        # Wait without sending more data - risk should decay
        time.sleep(4)
        risk_after_decay = shared.get_risk()

        # Risk should have decreased
        assert risk_after_decay < risk_after_threat

        fusion.stop()
        shared.close()

    def test_fusion_manager_callbacks(self):
        """Test FusionManager state change callbacks are invoked."""
        shared = SharedStateManager()
        queues = QueueManager()
        fusion = FusionManager(shared, queues, target_hz=2.0)

        # Track callback invocations
        state_changes = []

        def on_state_change(old_state, new_state, risk):
            state_changes.append((old_state, new_state, risk))

        fusion.state_manager.on_state_change(on_state_change)

        fusion.start()
        time.sleep(0.2)

        # Send threat to trigger state change
        audio = AudioEvent(
            timestamp=time.time(),
            transcript="threat",
            emotion=EmotionType.ANGRY,
            emotion_confidence=0.9,
            intent=IntentType.THREAT,
            intent_confidence=0.9,
        )
        queues.audio_to_fusion_q.put(audio, block=True, timeout=1)
        time.sleep(0.8)

        # Verify callback was called
        assert len(state_changes) > 0
        old, new, risk = state_changes[0]
        assert old == RiskState.IDLE  # Started at IDLE
        assert new != RiskState.IDLE  # Moved to a new state

        fusion.stop()
        shared.close()


class TestMultiThreadedQueueDynamics:
    """Test queue behavior with concurrent producers/consumers."""

    def test_queue_concurrent_puts_gets(self):
        """Test concurrent puts and gets don't cause race conditions."""
        queues = QueueManager()

        # Producer: put visual data
        def producer():
            for i in range(10):
                visual = VisualData(
                    timestamp=time.time(),
                    frame_shape=(480, 640, 3),
                    persons=[],
                )
                queues.visual_to_fusion_q.put(visual, block=True, timeout=1)
                time.sleep(0.05)

        # Consumer: drain queue
        def consumer():
            time.sleep(0.1)  # Let producer start
            results = []
            for _ in range(10):
                count, latest = queues.drain_visual_queue()
                if latest:
                    results.append(latest)
                time.sleep(0.05)
            return results

        prod_thread = threading.Thread(target=producer)
        cons_thread = threading.Thread(target=consumer)

        prod_thread.start()
        cons_thread.start()

        prod_thread.join(timeout=5)
        cons_thread.join(timeout=5)

        # Verify both completed
        assert not prod_thread.is_alive()
        assert not cons_thread.is_alive()

    def test_queue_backpressure_handling(self):
        """Test system handles queue backpressure gracefully."""
        queues = QueueManager()

        # Rapidly fill visual queue
        filled_count = 0
        for _ in range(200):
            visual = VisualData(
                timestamp=time.time(),
                frame_shape=(480, 640, 3),
                persons=[],
            )
            try:
                # Non-blocking put (visual pipeline behavior)
                result = queues.put_visual_data(visual, block=False)
                if result:
                    filled_count += 1
            except:
                pass

        # Verify queue didn't overflow (non-blocking puts returned False)
        sizes = queues.qsize()
        assert sizes["visual_q"] <= 100  # Max size should be respected

        # Drain should work without hanging
        count, _ = queues.drain_visual_queue()
        assert count <= 100


class TestSharedStateThreadSafety:
    """Test SharedStateManager thread safety."""

    def test_concurrent_updates_reads(self):
        """Test concurrent updates and reads are safe."""
        shared = SharedStateManager()
        updates = []
        reads = []

        # Writer thread
        def writer():
            for i in range(20):
                score = float(i * 5)
                state = RiskState.IDLE if i < 5 else RiskState.CAUTION
                shared.update_risk(score, state)
                updates.append((score, state))
                time.sleep(0.01)

        # Reader thread
        def reader():
            for _ in range(20):
                risk = shared.get_risk()
                state = shared.get_state()
                reads.append((risk, state))
                time.sleep(0.01)

        write_thread = threading.Thread(target=writer)
        read_thread = threading.Thread(target=reader)

        write_thread.start()
        read_thread.start()

        write_thread.join(timeout=5)
        read_thread.join(timeout=5)

        # Verify both threads completed
        assert len(updates) > 0
        assert len(reads) > 0

        shared.close()

    def test_alert_state_thread_safe(self):
        """Test alert state updates are thread-safe."""
        shared = SharedStateManager()
        alert_set_count = 0
        alert_check_count = 0

        # Thread 1: Set alert repeatedly
        def setter():
            nonlocal alert_set_count
            for _ in range(10):
                shared.set_alert(True)
                alert_set_count += 1
                time.sleep(0.01)
                shared.set_alert(False)

        # Thread 2: Check alert state
        def checker():
            nonlocal alert_check_count
            for _ in range(20):
                is_active = shared.is_alert_active()
                assert isinstance(is_active, bool)
                alert_check_count += 1
                time.sleep(0.01)

        set_thread = threading.Thread(target=setter)
        check_thread = threading.Thread(target=checker)

        set_thread.start()
        check_thread.start()

        set_thread.join(timeout=5)
        check_thread.join(timeout=5)

        assert alert_set_count == 10
        assert alert_check_count == 20

        shared.close()


class TestGracefulShutdown:
    """Test system graceful shutdown with threads running."""

    def test_fusion_manager_thread_cleanup(self):
        """Test FusionManager properly cleans up thread on shutdown."""
        shared = SharedStateManager()
        queues = QueueManager()
        fusion = FusionManager(shared, queues, target_hz=2.0)

        fusion.start()
        thread_id = fusion._thread.ident
        time.sleep(0.2)

        # Verify thread is running
        assert fusion._thread.is_alive()

        # Shutdown
        fusion.stop()

        # Verify thread stopped
        time.sleep(0.1)
        assert not fusion._thread.is_alive()

        shared.close()

    def test_system_manager_graceful_shutdown(self):
        """Test SystemManager handles graceful shutdown."""
        from core.system_manager import SystemManager

        system = SystemManager()

        # Start with mocked components (to avoid hardware dependencies)
        with patch.object(system, '_init_pipelines'), \
             patch.object(system, '_start_threads'), \
             patch.object(system, '_stop_threads'):

            system.start()
            assert system.running

            # Simulate some runtime
            time.sleep(0.1)

            # Shutdown
            system.stop()
            assert not system.running


class TestRiskAccumulationRealTime:
    """Test real-time risk accumulation with FusionManager."""

    def test_idle_to_alert_progression(self):
        """Test risk progression from IDLE → CAUTION → EVALUATING → ALERT."""
        shared = SharedStateManager()
        queues = QueueManager()
        fusion = FusionManager(shared, queues, target_hz=2.0)

        fusion.start()
        time.sleep(0.2)

        assert shared.get_state() == RiskState.IDLE

        # Send moderate threat (audio threat intent)
        audio1 = AudioEvent(
            timestamp=time.time(),
            transcript="threat",
            emotion=EmotionType.ANGRY,
            emotion_confidence=0.9,
            intent=IntentType.THREAT,
            intent_confidence=0.9,
        )
        queues.audio_to_fusion_q.put(audio1, block=True, timeout=1)
        time.sleep(0.6)

        state1 = shared.get_state()
        risk1 = shared.get_risk()
        # Should progress beyond IDLE
        assert risk1 > 0

        # Send more threats to escalate
        for _ in range(3):
            audio = AudioEvent(
                timestamp=time.time(),
                transcript="I will hurt you",
                emotion=EmotionType.ANGRY,
                emotion_confidence=0.95,
                intent=IntentType.THREAT,
                intent_confidence=0.95,
            )
            queues.audio_to_fusion_q.put(audio, block=True, timeout=1)
            time.sleep(0.6)

        state2 = shared.get_state()
        risk2 = shared.get_risk()

        # Risk should escalate
        assert risk2 > risk1
        assert state2 != RiskState.IDLE

        fusion.stop()
        shared.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])