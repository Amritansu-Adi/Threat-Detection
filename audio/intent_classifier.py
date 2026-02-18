"""Stage 3b: DistilBERT Zero-Shot - Intent classification from transcript.

Intent Classification using DistilBERT-MNLI:
- Zero-shot learning (no fine-tuning needed)
- Classifies text into: distress, threat, neutral, informational
- Fast on CPU with distilled model
- Confidence scores for each intent

Purpose in 3-stage pipeline:
  Silero VAD → DistilHuBERT (emotion) → Faster-Whisper (STT) + DistilBERT (intent)

Intent scoring:
  - Distress intent (e.g., "help!", "fire!") → +35 pts
  - Threat intent (e.g., "I'll kill you") → +45 pts
  - Neutral intent → 0 pts

Zero-shot classification:
  No training data needed - DistilBERT can classify new intents by natural language.
  Pass candidate labels like ["distress", "threat", "neutral", "informational"]
  Model returns confidence for each label.

Context filtering:
  - If audio_emotion is HAPPY, reduce/suppress threat intent risk
  - Solution to "Shut up!" problem: threat words + happy prosody = no alert

Model: distilbert-base-uncased-finetuned-mnli (Hugging Face)
Paper: DistilBERT: A distilled version of BERT (Sanh et al.)
"""

import logging
from typing import Tuple, Optional, Dict, List
from enum import Enum

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger = logging.getLogger(__name__)
    logger.warning("transformers not installed - IntentClassifier will be stubbed")

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    """Intent classifications for threat assessment."""
    DISTRESS = "distress"  # "Help!", "Fire!", "Emergency!"
    THREAT = "threat"  # "I'll kill you", "Watch out"
    NEUTRAL = "neutral"  # Normal conversation
    INFORMATIONAL = "informational"  # Statement of fact


class IntentClassifier:
    """Zero-shot intent classification using DistilBERT.

    Classifies speech intent (distress, threat, neutral, informational) from text.

    Attributes:
        model_name: Hugging Face model ID
        device: "cpu" or "cuda"
        pipeline: Transformers zero-shot classifier
        candidate_labels: Intent labels to classify against
    """

    MODEL_NAME = "distilbert-base-uncased-finetuned-mnli"  # MNLI = inference
    CANDIDATE_LABELS = [
        Intent.DISTRESS.value,
        Intent.THREAT.value,
        Intent.NEUTRAL.value,
        Intent.INFORMATIONAL.value,
    ]

    def __init__(self, device: str = "cpu"):
        """Initialize intent classifier.

        Args:
            device: "cpu" or "cuda"
        """
        self.device = device
        self.pipeline = None
        self.loaded = False

        if HAS_TRANSFORMERS:
            self._load_model()
        else:
            logger.warning("IntentClassifier running in stub mode (transformers not installed)")

    def _load_model(self) -> None:
        """Load DistilBERT zero-shot classifier."""
        try:
            self.pipeline = pipeline(
                task="zero-shot-classification",
                model=self.MODEL_NAME,
                device=0 if self.device == "cuda" else -1,  # -1 for CPU
            )
            self.loaded = True
            logger.info(f"✓ DistilBERT loaded: {self.MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to load DistilBERT: {e}")
            self.loaded = False

    def classify(
        self, text: str, candidate_labels: Optional[List[str]] = None
    ) -> Tuple[Optional[Intent], float]:
        """Classify text intent.

        Args:
            text: Text to classify (from speech transcript)
            candidate_labels: Labels to classify against (default: standard intents)

        Returns:
            Tuple[intent, confidence]:
                - intent: Intent enum or None if classification failed
                - confidence: Confidence in top prediction (0.0-1.0)
        """
        if not self.loaded:
            logger.debug("IntentClassifier in stub mode - returning neutral")
            return Intent.NEUTRAL, 0.5

        if not text or not text.strip():
            return Intent.NEUTRAL, 0.0

        try:
            labels = candidate_labels or self.CANDIDATE_LABELS

            # Run zero-shot classification
            result = self.pipeline(
                text,
                labels,
                multi_class=False,  # Single-label classification
            )

            # Extract top result
            top_label = result["labels"][0].lower()
            confidence = float(result["scores"][0])

            # Map to Intent enum
            try:
                intent = Intent(top_label)
            except ValueError:
                logger.warning(f"Unknown intent label: {top_label}")
                intent = Intent.NEUTRAL

            logger.debug(f"Classified intent: {intent.value} (conf={confidence:.2f})")
            return intent, confidence

        except Exception as e:
            logger.error(f"Error in intent classification: {e}")
            return Intent.NEUTRAL, 0.0

    def get_all_scores(
        self, text: str, candidate_labels: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Get confidence scores for all intent labels.

        Args:
            text: Text to classify
            candidate_labels: Labels to classify against

        Returns:
            Dict mapping intent label to confidence score
        """
        if not self.loaded:
            return {label: 0.25 for label in (candidate_labels or self.CANDIDATE_LABELS)}

        if not text or not text.strip():
            return {label: 0.0 for label in (candidate_labels or self.CANDIDATE_LABELS)}

        try:
            labels = candidate_labels or self.CANDIDATE_LABELS

            result = self.pipeline(
                text,
                labels,
                multi_class=False,
            )

            # Create score dict
            scores = {}
            for label, score in zip(result["labels"], result["scores"]):
                scores[label.lower()] = float(score)

            return scores

        except Exception as e:
            logger.error(f"Error getting all scores: {e}")
            return {label: 0.0 for label in (candidate_labels or self.CANDIDATE_LABELS)}

    def is_high_confidence(self, confidence: float, threshold: float = 0.75) -> bool:
        """Check if confidence meets high-confidence threshold.

        Used for risk scoring - only act on confident intent classifications.

        Args:
            confidence: Intent confidence (0.0-1.0)
            threshold: High-confidence threshold (default 0.75 = 75%)

        Returns:
            bool: True if confidence >= threshold
        """
        return confidence >= threshold

    def should_filter_as_context(
        self, intent: Intent, audio_emotion: Optional[str] = None
    ) -> bool:
        """Determine if this intent should be suppressed by audio context.

        Example: threat_intent + happy_emotion = joking (filter out threat)

        Args:
            intent: Classified intent
            audio_emotion: Emotion type (e.g., "happy", "angry")

        Returns:
            bool: True if this intent should be filtered/suppressed
        """
        # If threat intent + happy emotion = joking context
        if intent == Intent.THREAT and audio_emotion == "happy":
            logger.debug(f"Context filter: Threat intent suppressed by happy emotion")
            return True

        # If threat intent + disgust emotion = less credible
        if intent == Intent.THREAT and audio_emotion == "disgust":
            logger.debug(f"Context filter: Threat intent reduced by disgust emotion")
            return True

        return False
