"""Stage 2: DistilHuBERT - Emotion recognition from audio prosody.

Speech Emotion Recognition using DistilHuBERT:
- Lightweight 0.02MB model (distilled from HuBERT)
- Recognizes prosody (tone, pitch, rhythm, emphasis)
- Real-time capable on CPU
- Outputs: Emotion type + confidence

Emotions detected: neutral, happy, sad, angry, fearful, disgust, surprise

Purpose in 3-stage pipeline:
  Silero VAD → DistilHuBERT (emotion) → Faster-Whisper (STT) + DistilBERT (intent)

Emotion feedback to risk accumulator:
  - Anger (high conf) → +25 pts
  - Fear (high conf) → +30 pts
  - Context filter: Happy emotion suppresses threat intents (joking)

Model: ntu-spml/distilhubert (Hugging Face)
Paper: DistilHuBERT: Speech Representation Learning by Layer-wise Distillation
"""

import logging
from typing import Tuple, Optional, Dict
import numpy as np
from enum import Enum

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger = logging.getLogger(__name__)
    logger.warning("transformers not installed - EmotionAnalyzer will be stubbed")

logger = logging.getLogger(__name__)


class Emotion(str, Enum):
    """Emotion classifications from DistilHuBERT."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    DISGUST = "disgust"
    SURPRISE = "surprise"


class EmotionAnalyzer:
    """Speech Emotion Recognition using DistilHuBERT.

    Analyzes audio for emotional prosody (tone, pitch, emphasis).

    Attributes:
        model_name: Hugging Face model ID
        device: "cpu" or "cuda" for inference
        pipeline: Transformers pipeline for inference
    """

    MODEL_NAME = "ntu-spml/distilhubert"  # 0.02MB model (distilled)
    DEFAULT_EMOTIONS = [
        "neutral", "happy", "sad", "angry", "fearful", "disgust", "surprise"
    ]

    def __init__(self, device: str = "cpu", use_fast: bool = True):
        """Initialize emotion analyzer.

        Args:
            device: "cpu" or "cuda"
            use_fast: Use fast tokenizer if available
        """
        self.device = device
        self.use_fast = use_fast
        self.pipeline = None
        self.loaded = False

        if HAS_TRANSFORMERS:
            self._load_model()
        else:
            logger.warning("EmotionAnalyzer running in stub mode (transformers not installed)")

    def _load_model(self) -> None:
        """Load DistilHuBERT model from Hugging Face."""
        try:
            # Initialize audio classification pipeline
            self.pipeline = pipeline(
                task="audio-classification",
                model=self.MODEL_NAME,
                device=0 if self.device == "cuda" else -1,  # -1 for CPU
            )
            self.loaded = True
            logger.info(f"✓ DistilHuBERT loaded: {self.MODEL_NAME} (0.02MB model)")
        except Exception as e:
            logger.error(f"Failed to load DistilHuBERT: {e}")
            self.loaded = False

    def analyze(
        self, audio_array: np.ndarray, sample_rate: int = 16000
    ) -> Tuple[Optional[Emotion], float]:
        """Analyze audio for emotional content.

        Args:
            audio_array: Audio samples (numpy array, 1D or 2D)
            sample_rate: Sample rate of audio (default 16kHz)

        Returns:
            Tuple[emotion, confidence]:
                - emotion: Emotion enum or None if analysis failed
                - confidence: Detection confidence (0.0-1.0)
        """
        if not self.loaded:
            logger.debug("EmotionAnalyzer in stub mode - returning neutral")
            return Emotion.NEUTRAL, 0.5

        try:
            # Ensure audio is 1D
            audio_array = np.asarray(audio_array, dtype=np.float32)
            if audio_array.ndim > 1:
                audio_array = audio_array.flatten()

            # Normalize to [-1, 1]
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array / max_val

            # Run inference
            # Pipeline expects dict with "array" and "sampling_rate"
            result = self.pipeline(
                {"array": audio_array, "sampling_rate": sample_rate},
                top_k=1,  # Get top emotion
            )

            if not result or len(result) == 0:
                return Emotion.NEUTRAL, 0.0

            # Extract top result
            top = result[0]
            emotion_str = top["label"].lower()
            confidence = float(top["score"])

            # Map to Emotion enum
            try:
                emotion = Emotion(emotion_str)
            except ValueError:
                logger.warning(f"Unknown emotion label: {emotion_str}")
                emotion = Emotion.NEUTRAL

            logger.debug(f"Detected emotion: {emotion.value} (conf={confidence:.2f})")
            return emotion, confidence

        except Exception as e:
            logger.error(f"Error in emotion analysis: {e}")
            return Emotion.NEUTRAL, 0.0

    def get_top_emotions(
        self, audio_array: np.ndarray, sample_rate: int = 16000, top_k: int = 3
    ) -> list:
        """Get top-k emotions with confidence scores.

        Args:
            audio_array: Audio samples
            sample_rate: Sample rate (default 16kHz)
            top_k: Number of top emotions to return

        Returns:
            List of {"emotion": str, "confidence": float} dicts
        """
        if not self.loaded:
            return [{"emotion": "neutral", "confidence": 0.5}]

        try:
            audio_array = np.asarray(audio_array, dtype=np.float32).flatten()
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array / max_val

            result = self.pipeline(
                {"array": audio_array, "sampling_rate": sample_rate},
                top_k=top_k,
            )

            return [
                {
                    "emotion": item["label"].lower(),
                    "confidence": float(item["score"]),
                }
                for item in result
            ]

        except Exception as e:
            logger.error(f"Error getting top emotions: {e}")
            return [{"emotion": "neutral", "confidence": 0.0}]

    def is_high_confidence(self, confidence: float, threshold: float = 0.80) -> bool:
        """Check if confidence meets high-confidence threshold.

        Used for risk scoring - only act on high-confidence emotions.

        Args:
            confidence: Emotion confidence (0.0-1.0)
            threshold: High-confidence threshold (default 0.80 = 80%)

        Returns:
            bool: True if confidence >= threshold
        """
        return confidence >= threshold
