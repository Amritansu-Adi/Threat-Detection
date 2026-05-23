import time
import pytest
from datetime import datetime, timedelta

from fusion.risk_accumulator import RiskAccumulator
from fusion.state_manager import StateManager, AlertTriggerType
from core.data_structures import (
    AudioEvent,
    BoundingBox,
    EmotionType,
    IntentType,
    RiskState,
    WeaponDetection,
)
from core.shared_state import SharedStateManager


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

    # Audio risk is confidence-weighted and bounded as current severity.
    expected_audio = (
        ra.weights.anger_high_conf * audio.emotion_confidence
        + ra.weights.threat_intent * audio.intent_confidence
    )
    assert pytest.approx(event.score, rel=1e-6) == expected_audio
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

    expected = current * (ra.alpha ** 2)
    assert pytest.approx(ev.score, rel=1e-6) == expected


def test_repeated_weak_visual_evidence_does_not_snowball():
    ra = RiskAccumulator()
    visual = type("Visual", (), {})()
    visual.persons = [{
        "id": 1,
        "identity": "UNKNOWN",
        "has_weapon": False,
        "frames_in_view": 30,
    }]
    visual.weapons = []

    risk = 0.0
    for _ in range(10):
        event = ra.calculate_risk(risk, visual, None, dt_seconds=0.25)
        risk = event.score

    assert risk < 20
    assert event.state == RiskState.IDLE


def test_unassociated_weapon_is_weak_visual_evidence_without_person():
    ra = RiskAccumulator()
    visual = type("Visual", (), {})()
    visual.persons = []
    visual.weapons = [
        WeaponDetection(
            bbox=BoundingBox(0.1, 0.1, 0.2, 0.2, confidence=0.7),
            weapon_type="knife",
            confidence=0.7,
            associated_person_id=None,
        )
    ]

    event = ra.calculate_risk(0.0, visual, None, dt_seconds=1.0)

    assert pytest.approx(event.visual_risk, rel=1e-6) == ra.weights.weapon_base * 0.7
    assert 0.0 < event.score < 20.0
    assert event.state == RiskState.IDLE


def test_shared_state_tracks_explainable_risk_details():
    shared = SharedStateManager()
    shared.update_risk(
        42.0,
        RiskState.CAUTION,
        visual_risk=12.0,
        audio_risk=30.0,
        contributing_factors=["Visual: 12.0 pts", "- Unknown person detected"],
    )
    shared.update_visual_summary(1, 1, 1, 1, unknown_with_weapon=True)

    risk_details = shared.get_risk_details()
    visual_summary = shared.get_visual_summary()

    assert risk_details["visual_risk"] == 12.0
    assert risk_details["audio_risk"] == 30.0
    assert "Unknown person detected" in " ".join(risk_details["contributing_factors"])
    assert visual_summary["unknown_with_weapon"] is True


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
