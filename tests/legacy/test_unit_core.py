import time
import pytest

from core.queue_manager import QueueManager
from core.shared_state import SharedStateManager
from core.data_structures import (
    VisualData,
    AudioEvent,
    FusionInterrupt,
    EmotionType,
    IntentType,
    FusionCommand,
)


def test_queue_manager_put_get_drain_clear():
    qm = QueueManager(maxsize=10)

    # Test visual data with blocking operations
    visual = VisualData(timestamp=time.time(), frame_shape=(480, 640, 3))
    # Use blocking put to ensure it goes into the queue
    qm.visual_to_fusion_q.put(visual, block=True, timeout=1)

    count, latest = qm.drain_visual_queue()
    assert count == 1
    assert latest.timestamp == visual.timestamp

    # qsize should be zero after drain
    sizes = qm.qsize()
    assert isinstance(sizes, dict)
    assert sizes["visual_q"] == 0

    # Audio event put/get  
    audio = AudioEvent(
        timestamp=time.time(),
        transcript="help",
        emotion=EmotionType.NEUTRAL,
        emotion_confidence=0.9,
        intent=IntentType.DISTRESS,
        intent_confidence=0.9,
    )

    qm.audio_to_fusion_q.put(audio, block=True, timeout=1)
    evt = qm.get_audio_event(block=True, timeout=1)
    assert evt is not None
    assert evt.transcript == "help"

    # Fusion interrupt
    interrupt = FusionInterrupt(command=FusionCommand.SNAPSHOT_CAPTURE)
    qm.fusion_to_visual_q.put(interrupt, block=True, timeout=1)
    got = qm.get_fusion_interrupt(block=True, timeout=1)
    assert got is not None
    assert got.command == FusionCommand.SNAPSHOT_CAPTURE

    # Clear and close
    qm.clear_all()
    sizes2 = qm.qsize()
    assert sizes2["visual_q"] == 0


def test_shared_state_manager_updates_and_reset():
    s = SharedStateManager()

    # Risk update
    from core.data_structures import RiskState

    s.update_risk(50.0, RiskState.CAUTION)
    assert pytest.approx(s.get_risk(), rel=1e-6) == 50.0
    assert s.get_state() == RiskState.CAUTION

    # Visual summary
    s.update_visual_summary(2, 1, 0, 0)
    vs = s.get_visual_summary()
    assert vs["persons_count"] == 2
    assert vs["unknown_count"] == 1

    # Audio summary
    s.update_audio_summary("hi", "happy", 0.9, "neutral", 0.1)
    aud = s.get_audio_summary()
    assert aud["transcript"] == "hi"
    assert aud["emotion"] == "happy"

    # Alert management
    s.set_alert(True)
    assert s.is_alert_active() is True
    assert s.get_alert_count() >= 1

    # Reset
    s.reset()
    assert pytest.approx(s.get_risk(), rel=1e-6) == 0.0

    # Close manager
    s.close()
