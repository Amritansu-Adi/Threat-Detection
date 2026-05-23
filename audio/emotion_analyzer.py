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
import os
import re
from typing import Tuple, Optional, Dict
import numpy as np
from enum import Enum

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except Exception as e:
    HAS_TRANSFORMERS = False
    logger = logging.getLogger(__name__)
    logger.warning(f"transformers unavailable - EmotionAnalyzer will be stubbed: {e}")

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

    # Use a speech-emotion model with classification head.
    MODEL_NAME = "Dpngtm/wav2vec2-emotion-recognition"
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
        self._id2label: Dict[int, str] = {}

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
                local_files_only=(os.getenv("THREAT_TRANSFORMER_LOCAL_ONLY", "1") == "1"),
            )
            try:
                model_cfg = getattr(self.pipeline, "model", None).config
                self._id2label = {
                    int(k): str(v).lower()
                    for k, v in getattr(model_cfg, "id2label", {}).items()
                }
            except Exception:
                self._id2label = {}
            self.loaded = True
            logger.info(f"✓ Emotion model loaded: {self.MODEL_NAME}")
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
            emotion_str = self._normalize_label(str(top["label"]))
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

    def _normalize_label(self, raw_label: str) -> str:
        """Map model labels to canonical emotion labels."""
        label = raw_label.strip().lower()
        alias_map = {
            "anger": "angry",
            "fear": "fearful",
            "happiness": "happy",
            "sadness": "sad",
            "calm": "neutral",
            "surprised": "surprise",
        }
        if label in alias_map:
            return alias_map[label]

        match = re.match(r"label[_\s-]*(\d+)$", label, flags=re.IGNORECASE)
        if match:
            idx = int(match.group(1))
            mapped = self._id2label.get(idx, "").strip().lower()
            if mapped:
                if mapped in alias_map:
                    return alias_map[mapped]
                return mapped
            fallback_by_index = {
                0: "sad",
                1: "angry",
                2: "disgust",
                3: "fearful",
                4: "happy",
                5: "neutral",
                6: "surprise",
            }
            return fallback_by_index.get(idx, "neutral")

        return label

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
