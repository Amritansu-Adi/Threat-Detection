import time
import pytest
from datetime import datetime, timedelta

from fusion.risk_accumulator import RiskAccumulator
from fusion.state_manager import StateManager, AlertTriggerType
from core.data_structures import AudioEvent, EmotionType, IntentType, RiskState


def test_risk_accumulator_audio_and_context():
    ra = RiskAccumulator()

    audio = AudioEvent(
        timestamp=time.time(),
        transcript="I'll kill you",
        emotion=EmotionType.ANGRY,
        emotion_confidence=0.9,
        intent=IntentType.THREAT,
        intent_confidence=0.9,
    )

    event = ra.calculate_risk(0.0, None, audio, dt_seconds=1.0)

    # Expect audio risk = anger(25) + threat(45) = 70
    assert pytest.approx(event.score, rel=1e-6) == 70.0
    assert event.state == RiskState.EVALUATING

    # Happy emotion should suppress threat via context filtering
    audio_joke = AudioEvent(
        timestamp=time.time(),
        transcript="Shut up",
        emotion=EmotionType.HAPPY,
        emotion_confidence=0.95,
        intent=IntentType.THREAT,
        intent_confidence=0.95,
    )

    event2 = ra.calculate_risk(0.0, None, audio_joke, dt_seconds=1.0)
    assert event2.audio_risk == 0.0
    assert event2.score == 0.0


def test_risk_accumulator_decay():
    ra = RiskAccumulator()
    current = 50.0
    ev = ra.calculate_risk(current, None, None, dt_seconds=2.0)

    # Compute expected per implementation: decayed = current * alpha**dt; total = decayed * alpha
    expected = (current * (ra.alpha ** 2)) * ra.alpha
    assert pytest.approx(ev.score, rel=1e-6) == expected


def test_state_manager_transitions_and_alerts():
    sm = StateManager()

    assert sm.get_current_state() == RiskState.IDLE

    # Transition to ALERT and simulate duration
    changed = sm.update_state(RiskState.ALERT, 80.0)
    assert changed is True
    sm.state_entered_time = datetime.now() - timedelta(seconds=4)
    assert sm.is_alert_triggered() is True
    assert sm.get_alert_trigger_type() == AlertTriggerType.SUSTAINED

    # Transition to CRITICAL and simulate short duration
    sm.update_state(RiskState.CRITICAL, 95.0)
    sm.state_entered_time = datetime.now() - timedelta(seconds=1)
    assert sm.is_alert_triggered() is True
    assert sm.get_alert_trigger_type() == AlertTriggerType.IMMEDIATE

    # Callback behavior
    called = []

    def cb(old, new, risk):
        called.append((old, new, risk))

    sm.on_state_change(cb)
    sm.update_state(RiskState.CAUTION, 30.0)
    assert len(called) >= 1
