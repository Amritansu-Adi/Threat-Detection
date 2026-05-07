"""Risk accumulator implementing Temporal Persistence Multiplier.

This module implements the core risk calculation algorithm:
    Risk_t = (Risk_{t-1} × α) + (SensoryInput × β)

Where:
    α (alpha) = 0.9 (retention factor - creates ~5 second decay)
    β (beta) = 1.0 (confidence scaling)

The formula implements a behavioral narrative where:
- Risk decays when no new evidence accumulates
- Multiple brief threats build into sustained alert
- False positives (like "shut up!" + laughter) naturally decay
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import math

from core.data_structures import (
    VisualData,
    AudioEvent,
    RiskEvent,
    RiskState,
    TrackedPerson,
    WeaponDetection,
    EmotionType,
    IntentType,
)

logger = logging.getLogger(__name__)


@dataclass
class RiskWeights:
    """Configurable risk weights for visual and audio inputs.
    
    These values determine how different threats contribute to risk score.
    Tuned for 2026 "Conversational Intuition" standard with context awareness.
    """

    # Visual threat weights
    unknown_person_base: float = 8.0  # Unknown person in scene after grace period
    weapon_base: float = 6.0  # Unassociated weapon/object risk
    associated_weapon_base: float = 26.0  # Weapon associated with a person
    authorized_with_weapon: float = 3.0  # Known resident holding a plausible weapon/tool
    unknown_with_weapon: float = 26.0  # Additional points if unknown person has weapon
    unknown_grace_frames: int = 15  # Allow identity recognition before scoring unknowns
    max_unknown_person_risk: float = 20.0
    
    # Audio emotion weights (Stage 2: DistilHuBERT)
    anger_high_conf: float = 15.0  # High confidence anger
    fear_high_conf: float = 20.0  # High confidence fear
    
    # Audio intent weights (Stage 3: DistilBERT zero-shot)
    distress_intent: float = 45.0  # Distress call ("help!", "fire!")
    threat_intent: float = 55.0  # Threat statement ("I'll kill you")
    sudden_sound_anomaly: float = 24.0  # Sudden transient/noise anomaly
    audio_visual_unknown_bonus: float = 12.0
    audio_visual_weapon_bonus: float = 22.0
    visual_weight: float = 0.55
    audio_weight: float = 0.45
    
    # Confidence thresholds
    emotion_confidence_threshold: float = 0.80  # Only strong emotions trigger
    intent_confidence_threshold: float = 0.75  # Intent must be confident


class RiskAccumulator:
    """Accumulates risk over time using Temporal Persistence Multiplier.

    Implements the formula: Risk_t = (Risk_{t-1} × α) + (SensoryInput × β)

    This creates a "behavioral narrative" where:
    1. Risk decays naturally over time (α = 0.9 per second)
    2. New evidence pushes risk up
    3. Lasts ~5 seconds with no new evidence before dropping to zero
    4. Context filtering prevents false positives ("Shut up!" + laughter → no alert)

    Example - "Shut up" Scenario:
        T=0s: "Shut up!" + anger(0.8) → Risk = 0 + 25 = 25 pts → CAUTION
        T=1s: Laughter detected → Risk = 25×0.9^1 + 0 = 22.5 → Context filters threat intent
        T=2s: No new input → Risk = 22.5×0.9^1 = 20.25 → Continues decay
        Result: Never crosses threshold, correctly identified as joking
    """

    def __init__(self, weights: Optional[RiskWeights] = None):
        """Initialize risk accumulator.

        Args:
            weights: Custom RiskWeights, or defaults if None
        """
        self.weights = weights or RiskWeights()
        self.risk_history: List[tuple] = []  # (timestamp, risk_score) for debugging
        
        # Decay parameters
        self.alpha = 0.9  # Retention factor per second
        self.beta = 1.0  # Confidence scaling
        
        logger.info(
            f"RiskAccumulator initialized: α={self.alpha}, β={self.beta}, "
            f"weights={self.weights}"
        )

    def calculate_risk(
        self,
        current_risk: float,
        visual_data: Optional[VisualData] = None,
        audio_event: Optional[AudioEvent] = None,
        dt_seconds: float = 1.0,
    ) -> RiskEvent:
        """Calculate new risk score from current risk and new sensory input.

        Implements: Risk_t = (Risk_{t-1} × α^dt) + (InputRisk × β)

        Args:
            current_risk: Previous risk score (0-100)
            visual_data: Latest visual detection data (optional)
            audio_event: Latest audio analysis (optional)
            dt_seconds: Time delta since last update (for decay calculation)

        Returns:
            RiskEvent with new score, state, and contributing factors
        """
        # Calculate decay: older updates decay more
        decay_factor = self.alpha ** dt_seconds
        decayed_risk = current_risk * decay_factor

        # Calculate new input risk
        visual_risk = self._calculate_visual_risk(visual_data) if visual_data else 0.0
        audio_risk = self._calculate_audio_risk(audio_event) if audio_event else 0.0

        # Apply context filtering (context can suppress audio if contradictory)
        audio_risk = self._apply_context_filtering(
            audio_risk, audio_event, visual_data
        )

        # Weighted-sum scoring model:
        # total = decayed_previous + (wv*visual + wa*audio + cross_modal_bonus) * dt_scale
        cross_modal_bonus = self._cross_modal_bonus(visual_data, audio_event, audio_risk)
        weighted_input = (
            self.weights.visual_weight * visual_risk
            + self.weights.audio_weight * audio_risk
            + cross_modal_bonus
        )
        evidence_scale = max(0.20, min(1.0, dt_seconds))
        total_risk = decayed_risk + (weighted_input * self.beta * evidence_scale)

        # Clamp to [0, 100]
        total_risk = max(0.0, min(100.0, total_risk))

        # Determine new state
        new_state = self._risk_to_state(total_risk)

        # Build contributing factors for logging
        factors = self._build_contributing_factors(
            visual_data, audio_event, visual_risk, audio_risk, decay_factor
        )

        event = RiskEvent(
            score=total_risk,
            state=new_state,
            visual_risk=visual_risk,
            audio_risk=audio_risk,
            contributing_factors=factors,
        )

        # Log for debugging
        self.risk_history.append((datetime.now().timestamp(), total_risk))
        logger.debug(
            f"Risk update: {current_risk:.1f} → {total_risk:.1f} "
            f"(visual={visual_risk:.1f}, audio={audio_risk:.1f}, decay={decay_factor:.3f}) "
            f"→ {new_state.value}"
        )

        return event

    def _calculate_visual_risk(self, visual_data: VisualData) -> float:
        """Calculate risk from visual detections.

        Scoring emphasizes combinations:
        - Unknown person alone is low risk after a grace period.
        - Unassociated weapon-like objects are low risk.
        - Weapon associated with a person is moderate/high depending identity.
        - Unknown person with associated weapon is high risk.

        Args:
            visual_data: Visual detection data

        Returns:
            float: Visual risk score (0-100+, will be clamped in main formula)
        """
        risk = 0.0

        if not visual_data:
            return 0.0
        persons = self._get_visual_persons(visual_data)
        if not persons:
            return self._calculate_unassociated_weapon_risk(visual_data.weapons)

        associated_weapon_ids = {
            weapon.associated_person_id
            for weapon in visual_data.weapons
            if weapon.associated_person_id is not None
        }
        unknown_person_risk = 0.0

        # Score each tracked person
        for person in persons:
            identity = self._person_identity(person)
            has_weapon = self._person_has_weapon(person)
            person_id = self._person_id(person)
            frames_in_view = self._person_frames_in_view(person)
            is_unknown = self._is_unknown_identity(identity)
            person_has_associated_weapon = has_weapon or person_id in associated_weapon_ids

            if is_unknown and frames_in_view >= self.weights.unknown_grace_frames:
                unknown_person_risk += self.weights.unknown_person_base

            if person_has_associated_weapon:
                if is_unknown:
                    risk += self.weights.associated_weapon_base + self.weights.unknown_with_weapon
                else:
                    risk += self.weights.authorized_with_weapon

        risk += min(unknown_person_risk, self.weights.max_unknown_person_risk)

        risk += self._calculate_unassociated_weapon_risk(visual_data.weapons)

        return risk

    def _calculate_unassociated_weapon_risk(self, weapons: List[WeaponDetection]) -> float:
        """Score weapon-like detections that are not tied to a visible person."""
        risk = 0.0
        for weapon in weapons or []:
            if weapon.associated_person_id is None:
                # Unassociated weapon-like detections are weak evidence.
                risk += self.weights.weapon_base * max(0.1, weapon.confidence)
        return risk

    def _calculate_audio_risk(self, audio_event: AudioEvent) -> float:
        """Calculate risk from audio analysis.

        Stage 2 (Emotion): Strong emotions (anger, fear) boost risk
        Stage 3 (Intent): Threat/distress intents boost risk

        With confidence thresholds to avoid false positives on uncertain signals.

        Args:
            audio_event: Audio analysis event

        Returns:
            float: Audio risk score
        """
        risk = 0.0

        if not audio_event:
            return 0.0

        if audio_event.vad_confidence < 0.5:
            return 0.0

        # Stage 2: Emotion-based risk (requires high confidence)
        if audio_event.emotion_confidence >= self.weights.emotion_confidence_threshold:
            if audio_event.emotion == EmotionType.ANGRY:
                risk += self.weights.anger_high_conf * audio_event.emotion_confidence
            elif audio_event.emotion == EmotionType.FEARFUL:
                risk += self.weights.fear_high_conf * audio_event.emotion_confidence

        # Stage 3: Intent-based risk (requires high confidence)
        if audio_event.intent_confidence >= self.weights.intent_confidence_threshold:
            if audio_event.intent == IntentType.THREAT:
                risk += self.weights.threat_intent * audio_event.intent_confidence
            elif audio_event.intent == IntentType.DISTRESS:
                risk += self.weights.distress_intent * audio_event.intent_confidence

        # Stage 4: Sudden acoustic anomaly (bang/crash-like transient).
        features = audio_event.raw_audio_features or {}
        anomaly_score = float(features.get("anomaly_score", 0.0))
        sudden_anomaly = bool(features.get("sudden_anomaly", False))
        if sudden_anomaly:
            risk += self.weights.sudden_sound_anomaly * max(0.35, anomaly_score)

        return risk

    def _cross_modal_bonus(
        self,
        visual_data: Optional[VisualData],
        audio_event: Optional[AudioEvent],
        audio_risk: float,
    ) -> float:
        """Cross-modal bonus on top of weighted visual/audio components."""
        bonus = 0.0
        if audio_risk <= 0 or not audio_event or not visual_data:
            return 0.0

        if audio_event.intent not in (IntentType.THREAT, IntentType.DISTRESS):
            return 0.0

        persons = self._get_visual_persons(visual_data)
        unknown_present = any(
            self._is_unknown_identity(self._person_identity(p))
            for p in persons
        )
        weapon_present = bool(visual_data.weapons) or any(
            self._person_has_weapon(p)
            for p in persons
        )

        if unknown_present:
            bonus += self.weights.audio_visual_unknown_bonus
        if weapon_present:
            bonus += self.weights.audio_visual_weapon_bonus

        return bonus

    def _apply_context_filtering(
        self,
        audio_risk: float,
        audio_event: Optional[AudioEvent],
        visual_data: Optional[VisualData],
    ) -> float:
        """Apply context filtering to suppress contradictory signals.

        Example: "Shut up!" (threat intent 45 pts) + laughter (happy emotion)
                 → Filter: Context shows joking → audio_risk = 0

        This is the key to avoiding false positives when people use
        threatening language in joking contexts.

        Args:
            audio_risk: Audio risk before filtering
            audio_event: Audio analysis for context
            visual_data: Visual data for additional context (future expansion)

        Returns:
            float: Filtered audio risk
        """
        if not audio_event or audio_risk == 0:
            return audio_risk

        # If happy/sad emotion but threat intent, it's likely joking/sarcasm
        if audio_event.emotion == EmotionType.HAPPY:
            logger.debug(
                f"Context filter: Happy emotion suppressing threat intent "
                f"({audio_event.intent.value})"
            )
            return 0.0  # Joking threat = no risk

        # If disgust emotion, less likely to be genuine threat
        if audio_event.emotion == EmotionType.DISGUST:
            logger.debug(
                f"Context filter: Disgust emotion suppressing threat intent"
            )
            return audio_risk * 0.3  # Reduce by 70%

        # Add visual context capabilities here in future (e.g., if laughing faces visible)

        return audio_risk

    def _risk_to_state(self, risk_score: float) -> RiskState:
        """Convert risk score to state machine state.

        Risk thresholds:
        - IDLE: 0-25 pts
        - CAUTION: 25-50 pts
        - EVALUATING: 50-75 pts
        - ALERT: 75-90 pts
        - CRITICAL: 90-100 pts

        Args:
            risk_score: Risk score (0-100)

        Returns:
            RiskState: Corresponding state
        """
        if risk_score < 20:
            return RiskState.IDLE
        elif risk_score < 45:
            return RiskState.CAUTION
        elif risk_score < 70:
            return RiskState.EVALUATING
        elif risk_score < 90:
            return RiskState.ALERT
        else:
            return RiskState.CRITICAL

    @staticmethod
    def _is_unknown_identity(identity: Optional[str]) -> bool:
        normalized = str(identity or "").strip().upper()
        return normalized in {"", "UNKNOWN", "UNKNOWN_ENTITY", "UNRECOGNIZED", "NONE"}

    @staticmethod
    def _get_visual_persons(visual_data: Optional[VisualData]) -> List:
        if not visual_data:
            return []
        tracked = getattr(visual_data, 'tracked_persons', None)
        if tracked:
            return list(tracked)
        return list(visual_data.persons or [])

    @staticmethod
    def _person_identity(person) -> Optional[str]:
        if isinstance(person, dict):
            return person.get('identity') or person.get('label') or person.get('name') or "UNKNOWN"
        return getattr(person, 'identity', None) or getattr(person, 'label', None) or "UNKNOWN"

    @staticmethod
    def _person_has_weapon(person) -> bool:
        if isinstance(person, dict):
            return bool(person.get('has_weapon', False))
        return bool(getattr(person, 'has_weapon', False))

    @staticmethod
    def _person_id(person):
        if isinstance(person, dict):
            return person.get('id', person.get('track_id'))
        return getattr(person, 'id', getattr(person, 'track_id', None))

    @staticmethod
    def _person_frames_in_view(person) -> int:
        if isinstance(person, dict):
            return int(person.get('frames_in_view', 0))
        return int(getattr(person, 'frames_in_view', 999))

    def _build_contributing_factors(
        self,
        visual_data: Optional[VisualData],
        audio_event: Optional[AudioEvent],
        visual_risk: float,
        audio_risk: float,
        decay_factor: float,
    ) -> List[str]:
        """Build human-readable list of factors contributing to risk.

        Used for logging and debugging.

        Args:
            visual_data: Visual data
            audio_event: Audio event
            visual_risk: Calculated visual risk
            audio_risk: Calculated audio risk
            decay_factor: Decay factor applied this update

        Returns:
            List[str]: Human-readable factors
        """
        factors = []

        if visual_risk > 0:
            factors.append(f"Visual: {visual_risk:.1f} pts")
            if visual_data:
                persons = self._get_visual_persons(visual_data)
                if any(self._is_unknown_identity(self._person_identity(p)) for p in persons):
                    factors.append("- Unknown person detected")
                armed_persons = [p for p in persons if self._person_has_weapon(p)]
                unknown_armed = [
                    p
                    for p in armed_persons
                    if self._is_unknown_identity(self._person_identity(p))
                ]
                known_armed = len(armed_persons) - len(unknown_armed)
                if unknown_armed:
                    factors.append(f"- Unknown person with associated weapon ({len(unknown_armed)})")
                if known_armed:
                    factors.append(f"- Known person with associated object/tool ({known_armed})")
                elif armed_persons:
                    factors.append("- Armed person detected")
                if visual_data.weapons:
                    factors.append(f"- {len(visual_data.weapons)} weapon(s) detected")
                    unassociated = sum(
                        1 for w in visual_data.weapons if w.associated_person_id is None
                    )
                    if unassociated:
                        factors.append(f"- Unassociated weapon-like object ({unassociated})")

        if audio_risk > 0:
            factors.append(f"Audio: {audio_risk:.1f} pts")
            if audio_event:
                if audio_event.emotion_confidence > 0.8:
                    factors.append(
                        f"- Strong {audio_event.emotion.value} emotion detected"
                    )
                if audio_event.intent_confidence > 0.75:
                    factors.append(f"- {audio_event.intent.value.upper()} intent detected")
                if bool((audio_event.raw_audio_features or {}).get("sudden_anomaly", False)):
                    score = float((audio_event.raw_audio_features or {}).get("anomaly_score", 0.0))
                    factors.append(f"- Sudden sound anomaly ({score:.2f})")

        if decay_factor < 0.95:
            factors.append(f"Temporal decay applied ({decay_factor:.3f})")

        return factors

    def get_decay_rate(self) -> str:
        """Get human-readable decay rate information.

        Shows how long it takes for risk to decay after threat is gone.

        Returns:
            str: Decay rate description
        """
        # Calculate time to decay current_risk to 50% (half-life)
        half_life = math.log(0.5) / math.log(self.alpha)
        full_decay = math.log(0.01) / math.log(self.alpha)  # To 1% of original

        return (
            f"Half-life: {half_life:.2f} sec, Full decay (>1%): {full_decay:.2f} sec"
        )

    def get_risk_history(self, last_n: int = 100) -> List[tuple]:
        """Get recent risk history for debugging.

        Args:
            last_n: Number of recent samples to return

        Returns:
            List[(timestamp, risk_score)]: Recent history
        """
        return self.risk_history[-last_n:]
