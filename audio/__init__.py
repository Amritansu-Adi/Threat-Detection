"""Audio pipeline module for Autonomous Multi-Modal Intelligence Ecosystem.

3-stage hierarchical speech analysis:
  Stage 1: Silero VAD (voice detection)
  Stage 2: DistilHuBERT (emotion recognition)
  Stage 3: Faster-Whisper (STT) + DistilBERT (intent classification)
"""

__version__ = "0.1.0"

from .vad import SileroVAD
from .emotion_analyzer import EmotionAnalyzer, Emotion
from .transcriber import Transcriber
from .intent_classifier import IntentClassifier, Intent
from .audio_pipeline import AudioPipeline

__all__ = [
    "SileroVAD",
    "EmotionAnalyzer",
    "Emotion",
    "Transcriber",
    "IntentClassifier",
    "Intent",
    "AudioPipeline",
]

