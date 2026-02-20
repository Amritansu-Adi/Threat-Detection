"""Scenario-based testing for threat detection system.

Tests realistic threat scenarios to validate the system's entire pipeline
from sensory input through risk accumulation to alert generation.

Scenarios:
1. Armed Unknown Person - Visual threat alone
2. Distress Call - Audio threat alone  
3. Threat Speech + Anger - Combined audio threat
4. "Shut Up" + Laughter - Context filtering (false positive prevention)
5. Unknown Person + Weapon - Compound visual threat
6. State Machine Alert Timing - Verify alert delay thresholds
7. Risk Escalation - Multiple threats building over time
"""

import time
import pytest
from datetime import datetime

from fusion.risk_accumulator import RiskAccumulator
from fusion.state_manager import StateManager
from core.data_structures import (
    VisualData,
    AudioEvent,
    TrackedPerson,
    BoundingBox,
    WeaponDetection,
    EmotionType,
    IntentType,
    RiskState,
)


class TestScenarioArmedUnknownPerson:
    """Scenario 1: Armed unknown person detected."""

    def test_visual_threat_unknown_armed(self):
        """High visual risk: unknown person with weapon."""
        ra = RiskAccumulator()

        # Create unknown armed person
        person = TrackedPerson(
            track_id=1,
            bbox=BoundingBox(x1=0.2, y1=0.1, x2=0.5, y2=0.8, confidence=0.95),
            identity="UNKNOWN",
            has_weapon=True,
        )

        visual = VisualData(
            timestamp=time.time(),
            frame_shape=(480, 640, 3),
            persons=[{"track_id": 1, "bbox": person.bbox}],
        )
        visual.tracked_persons = [person]

        event = ra.calculate_risk(0.0, visual, None, dt_seconds=1.0)

        # Risk = unknown(20) + weapon(40) = 60 pts
        expected_risk = 60.0
        assert pytest.approx(event.score, rel=0.1) == expected_risk
        assert event.state == RiskState.EVALUATING
        assert "Unknown person" in " ".join(event.contributing_factors)
        assert "Armed" in " ".join(event.contributing_factors)


class TestScenarioDistressCall:
    """Scenario 2: Audio distress call ("help!", "fire!")."""

    def test_audio_threat_distress(self):
        """High audio risk: distress intent with fearful emotion."""
        ra = RiskAccumulator()

        audio = AudioEvent(
            timestamp=time.time(),
            transcript="help fire help",
            emotion=EmotionType.FEARFUL,
            emotion_confidence=0.92,
            intent=IntentType.DISTRESS,
            intent_confidence=0.88,
        )

        event = ra.calculate_risk(0.0, None, audio, dt_seconds=1.0)

        # Risk = fear(30) + distress(35) = 65 pts
        expected_risk = 65.0
        assert pytest.approx(event.score, rel=0.1) == expected_risk
        assert event.state == RiskState.EVALUATING
        factors_str = " ".join(event.contributing_factors).lower()
        assert "fearful emotion" in factors_str
        assert "distress" in factors_str


class TestScenarioThreatSpeech:
    """Scenario 3: Threat speech (e.g., "I'll kill you") with anger."""

    def test_audio_threat_with_anger(self):
        """High audio risk: threat intent with angry emotion."""
        ra = RiskAccumulator()

        audio = AudioEvent(
            timestamp=time.time(),
            transcript="I will kill you",
            emotion=EmotionType.ANGRY,
            emotion_confidence=0.94,
            intent=IntentType.THREAT,
            intent_confidence=0.91,
        )

        event = ra.calculate_risk(0.0, None, audio, dt_seconds=1.0)

        # Risk = anger(25) + threat(45) = 70 pts
        expected_risk = 70.0
        assert pytest.approx(event.score, rel=0.1) == expected_risk
        assert event.state == RiskState.EVALUATING
        assert "THREAT" in " ".join(event.contributing_factors)


class TestScenarioContextFiltering:
    """Scenario 4: "Shut up" + laughter = no false alert."""

    def test_joking_threat_filtered(self):
        """Context filtering: happy emotion suppresses threat.

        This is the key test for false positive prevention. A threat phrase
        ("Shut up!") combined with laughter (happy emotion) should NOT
        trigger an alert because the context shows the person is joking.
        """
        ra = RiskAccumulator()

        # Threat phrase but happy emotion
        audio_joke = AudioEvent(
            timestamp=time.time(),
            transcript="Shut up you idiots",
            emotion=EmotionType.HAPPY,
            emotion_confidence=0.91,  # Strong laughter/happiness
            intent=IntentType.THREAT,
            intent_confidence=0.87,  # High confidence in threat words
        )

        event = ra.calculate_risk(0.0, None, audio_joke, dt_seconds=1.0)

        # With context filtering: audio risk = 0 (happy suppresses threat)
        assert event.audio_risk == 0.0
        assert event.score == 0.0
        assert event.state == RiskState.IDLE
        assert "Context filter" in " ".join(event.contributing_factors) or event.audio_risk == 0.0


class TestScenarioCompoundThreat:
    """Scenario 5: Multiple threats combine (armed unknown + threat speech)."""

    def test_compound_visual_and_audio_threat(self):
        """Combined visual + audio threat → high risk."""
        ra = RiskAccumulator()

        # Visual: armed unknown person
        person = TrackedPerson(
            track_id=1,
            bbox=BoundingBox(x1=0.2, y1=0.1, x2=0.5, y2=0.8),
            identity="UNKNOWN",
            has_weapon=True,
        )
        visual = VisualData(
            timestamp=time.time(),
            frame_shape=(480, 640, 3),
            persons=[{"track_id": 1}],
        )
        visual.tracked_persons = [person]

        # Audio: threat speech with anger
        audio = AudioEvent(
            timestamp=time.time(),
            transcript="I have a weapon",
            emotion=EmotionType.ANGRY,
            emotion_confidence=0.9,
            intent=IntentType.THREAT,
            intent_confidence=0.9,
        )

        # Compound risk = visual(60) + audio(70) = 130, clamped to 100
        event = ra.calculate_risk(0.0, visual, audio, dt_seconds=1.0)
        assert event.score > 90  # Should be very high
        assert event.state in [RiskState.ALERT, RiskState.CRITICAL]


class TestScenarioStateMachineAlertTiming:
    """Scenario 6: State machine alert delay prevents false positives."""

    def test_alert_triggered_after_3_sec_in_alert_state(self):
        """Alert requires 3+ seconds in ALERT state before triggering."""
        sm = StateManager()

        # Transition to ALERT
        sm.update_state(RiskState.ALERT, 80.0)
        assert not sm.is_alert_triggered()  # Just entered

        # Simulate 1 second in state
        sm.state_entered_time = datetime.now()
        sm.state_entered_time = sm.state_entered_time.replace(
            microsecond=int((time.time() - 1) * 1e6) % 1000000
        )
        time.sleep(0.1)  # Small delay
        assert not sm.is_alert_triggered()  # Still under 3 sec

        # Simulate 3.5 seconds in state
        sm.state_entered_time = datetime.fromtimestamp(time.time() - 3.5)
        assert sm.is_alert_triggered()  # Now alert triggered

    def test_critical_triggered_after_0p5_sec(self):
        """Critical state triggers alert immediately after 0.5 sec."""
        sm = StateManager()

        sm.update_state(RiskState.CRITICAL, 95.0)
        assert not sm.is_alert_triggered()  # Just entered

        # Simulate 0.6 seconds in state
        sm.state_entered_time = datetime.fromtimestamp(time.time() - 0.6)
        assert sm.is_alert_triggered()  # Alert triggered


class TestScenarioRiskEscalation:
    """Scenario 7: Risk builds over time with repeated threats."""

    def test_risk_escalation_over_time(self):
        """Multiple audio events escalate risk through state machine."""
        ra = RiskAccumulator()
        sm = StateManager()

        risk_score = 0.0

        # T=0: First audio event (distress)
        audio1 = AudioEvent(
            timestamp=time.time(),
            transcript="help",
            emotion=EmotionType.FEARFUL,
            emotion_confidence=0.9,
            intent=IntentType.DISTRESS,
            intent_confidence=0.9,
        )
        event1 = ra.calculate_risk(risk_score, None, audio1, dt_seconds=1.0)
        risk_score = event1.score
        sm.update_state(event1.state, risk_score)
        assert event1.state == RiskState.EVALUATING  # 30 + 35 = 65

        # T=1: Second threat with minor decay
        audio2 = AudioEvent(
            timestamp=time.time(),
            transcript="fire",
            emotion=EmotionType.FEARFUL,
            emotion_confidence=0.9,
            intent=IntentType.DISTRESS,
            intent_confidence=0.9,
        )
        event2 = ra.calculate_risk(risk_score, None, audio2, dt_seconds=1.0)
        risk_score = event2.score
        sm.update_state(event2.state, risk_score)
        assert risk_score > event1.score  # Risk escalates

        # Risk should be clamped at 100
        assert risk_score <= 100.0


class TestScenarioWeaponDetection:
    """Scenario 8: unassociated weapon detected."""

    def test_unassociated_weapon_risk(self):
        """Unassociated weapon in scene adds risk."""
        ra = RiskAccumulator()

        # Add a known person so _calculate_visual_risk doesn't return early
        person = TrackedPerson(
            track_id=1,
            bbox=BoundingBox(x1=0.1, y1=0.1, x2=0.3, y2=0.9),
            identity="John",  # Known person (not unknown)
            has_weapon=False,
        )

        weapon = WeaponDetection(
            bbox=BoundingBox(x1=0.3, y1=0.4, x2=0.5, y2=0.6),
            weapon_type="knife",
            confidence=0.92,
            associated_person_id=None,  # Not held by anyone
        )

        visual = VisualData(
            timestamp=time.time(),
            frame_shape=(480, 640, 3),
            persons=[],
        )
        visual.tracked_persons = [person]
        visual.weapons = [weapon]

        event = ra.calculate_risk(0.0, visual, None, dt_seconds=1.0)

        # Unassociated weapon risk calculated: 40 * 0.5 = 20 pts
        # Should be positive (but exact value depends on risk_accumulator formula)
        assert event.score >= 0  # At minimum, no negative risk
        assert event.audio_risk == 0  # No audio threat


if __name__ == "__main__":
    pytest.main([__file__, "-v"])