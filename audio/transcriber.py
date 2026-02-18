"""Stage 3a: Faster-Whisper - Speech-to-Text with INT8 quantization.

Automatic Speech Recognition using Faster-Whisper:
- INT8 quantized for CPU efficiency
- Beam size = 1 (greedy decoding) for speed
- Real-time capable on consumer hardware
- Outputs: Transcript with timestamps

Purpose in 3-stage pipeline:
  Silero VAD → DistilHuBERT (emotion) → Faster-Whisper (STT) + DistilBERT (intent)

Faster-Whisper benefits:
  - Distilled from OpenAI Whisper for smaller size
  - INT8 quantization reduces model size by 4x
  - Beam size = 1 prioritizes speed over accuracy
  - Language detection built-in

Configuration:
  - Model: tiny.en (English only, fastest)
  - Compute type: int8 (quantized)
  - Beam size: 1 (fastest)
  - Language: en (English)

Model: openai/whisper-tiny.en
Paper: Robust Speech Recognition via Large-Scale Weak Supervision (OpenAI)
Optimization: Faster-Whisper (CTranslate2 backend)
"""

import logging
from typing import Optional, Tuple, List, Dict
import numpy as np

try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False
    logger = logging.getLogger(__name__)
    logger.warning("faster_whisper not installed - Transcriber will be stubbed")

logger = logging.getLogger(__name__)


class Transcriber:
    """Speech-to-Text using Faster-Whisper (INT8 quantized).

    Converts speech audio to text with high speed/accuracy tradeoff.

    Attributes:
        model_size: Whisper model size (tiny, base, small, medium, large)
        device: "cpu" or "cuda"
        compute_type: "float32", "float16", or "int8"
        model: Faster-Whisper model instance
    """

    # Model configurations (size vs speed tradeoff)
    MODEL_CONFIGS = {
        "tiny": {"size_mb": 39, "speed": "fastest", "accuracy": "low"},
        "base": {"size_mb": 140, "speed": "fast", "accuracy": "medium"},
        "small": {"size_mb": 466, "speed": "medium", "accuracy": "high"},
        "medium": {"size_mb": 1500, "speed": "slow", "accuracy": "very-high"},
    }

    DEFAULT_MODEL = "tiny"  # Smallest/fastest model
    DEFAULT_COMPUTE_TYPE = "int8"  # 4x compression
    DEFAULT_BEAM_SIZE = 1  # Greedy (fastest)

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL,
        device: str = "cpu",
        compute_type: str = DEFAULT_COMPUTE_TYPE,
        beam_size: int = DEFAULT_BEAM_SIZE,
        language: str = "en",
    ):
        """Initialize Faster-Whisper transcriber.

        Args:
            model_size: Model size (tiny, base, small, medium)
            device: "cpu" or "cuda"
            compute_type: "float32", "float16", or "int8"
            beam_size: Beam search size (1=greedy/fastest)
            language: Language code (default "en" for English)
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.language = language
        self.model = None
        self.loaded = False

        if HAS_FASTER_WHISPER:
            self._load_model()
        else:
            logger.warning("Transcriber running in stub mode (faster_whisper not installed)")

    def _load_model(self) -> None:
        """Load Faster-Whisper model."""
        try:
            self.model = WhisperModel(
                model_size_or_path=self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                num_workers=1,
                cpu_threads=4,  # For CPU execution
                download_root=None,  # Use default cache dir
            )
            self.loaded = True
            config = self.MODEL_CONFIGS.get(self.model_size, {})
            logger.info(
                f"✓ Faster-Whisper loaded: {self.model_size}"
                f" ({config.get('size_mb', '?')}MB, beam={self.beam_size}, int8={self.compute_type == 'int8'})"
            )
        except Exception as e:
            logger.error(f"Failed to load Faster-Whisper: {e}")
            self.loaded = False

    def transcribe(
        self, audio_array: np.ndarray, sample_rate: int = 16000
    ) -> Tuple[str, float]:
        """Transcribe audio to text.

        Args:
            audio_array: Audio samples (numpy array)
            sample_rate: Sample rate (default 16kHz)

        Returns:
            Tuple[transcript, confidence]:
                - transcript: Recognized text
                - confidence: Average confidence (0.0-1.0)
        """
        if not self.loaded:
            logger.debug("Transcriber in stub mode - returning empty transcript")
            return "", 0.0

        try:
            # Convert to float32 if needed
            audio_array = np.asarray(audio_array, dtype=np.float32)

            # Ensure 1D
            if audio_array.ndim > 1:
                audio_array = audio_array.flatten()

            # Normalize
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array / max_val

            # Transcribe
            segments, info = self.model.transcribe(
                audio=audio_array,
                language=self.language,
                beam_size=self.beam_size,
                vad_filter=True,  # Use VAD to filter silence
                vad_parameters={"min_speech_duration_ms": 300},
            )

            # Collect segments and confidence
            transcript_parts = []
            confidences = []

            for segment in segments:
                transcript_parts.append(segment.text)
                # Whisper doesn't provide per-word confidence, estimate from no_speech_prob
                conf = 1.0 - segment.no_speech_prob
                confidences.append(conf)

            transcript = " ".join(transcript_parts).strip()
            avg_confidence = (
                np.mean(confidences) if confidences else 0.0
            )

            logger.debug(
                f"Transcribed: '{transcript}' (conf={avg_confidence:.2f})"
            )
            return transcript, avg_confidence

        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return "", 0.0

    def transcribe_with_timestamps(
        self, audio_array: np.ndarray, sample_rate: int = 16000
    ) -> List[Dict]:
        """Transcribe with word-level timestamps.

        Args:
            audio_array: Audio samples
            sample_rate: Sample rate (default 16kHz)

        Returns:
            List of {"word": str, "start": float, "end": float} dicts
        """
        if not self.loaded:
            return []

        try:
            audio_array = np.asarray(audio_array, dtype=np.float32).flatten()
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array / max_val

            segments, info = self.model.transcribe(
                audio=audio_array,
                language=self.language,
                beam_size=self.beam_size,
                vad_filter=True,
            )

            results = []
            for segment in segments:
                # Whisper provides segment-level timing
                results.append({
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                    "confidence": 1.0 - segment.no_speech_prob,
                })

            return results

        except Exception as e:
            logger.error(f"Error getting timestamps: {e}")
            return []

    def get_language(self, audio_array: np.ndarray) -> Tuple[str, float]:
        """Auto-detect language from audio.

        Args:
            audio_array: Audio samples

        Returns:
            Tuple[language_code, confidence]:
                - language_code: ISO 639-1 code (e.g., "en")
                - confidence: Detection confidence (0.0-1.0)
        """
        if not self.loaded:
            return "en", 1.0  # Default to English

        try:
            audio_array = np.asarray(audio_array, dtype=np.float32).flatten()
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array / max_val

            _, info = self.model.transcribe(
                audio=audio_array,
                language=None,  # Auto-detect
                beam_size=1,
            )

            return info.language, info.language_probability

        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return "en", 0.0
