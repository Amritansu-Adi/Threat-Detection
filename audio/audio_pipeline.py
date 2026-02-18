"""Main Audio Pipeline - 3-stage hierarchical speech analysis.

Audio Pipeline Architecture:
  1. Audio Capture (16kHz mono, 2-second chunks)
  2. Stage 1: Silero VAD (voice detection gate)
  3. Stage 2: DistilHuBERT (emotion from prosody)
  4. Stage 3: Faster-Whisper (STT) + DistilBERT (intent)
  5. Emit AudioEvent to fusion manager

The pipeline maintains a 5-second sliding window buffer to ensure complete
utterances are analyzed with context (including trailing silence for prosody).

Data Flow:
  Audio Input (16kHz)
       ↓
  [VAD Gate] → if no voice, skip to next chunk
       ↓
  [SER: Emotion] → anger_high_conf +25 pts, fear_high_conf +30 pts
       ↓
  [ASR: Transcribe] → get text
       ↓
  [Intent: DistilBERT] → threat +45 pts, distress +35 pts
       ↓
  [Context Filter] → happy emotion suppresses threat (no false alert on "Shut up")
       ↓
  → Emit AudioEvent to audio_to_fusion_q

Threading:
  - Runs in background thread
  - Non-blocking queue insertion
  - Logs timing for latency monitoring
"""

import logging
import threading
import time
import numpy as np
from typing import Optional, Callable
from collections import deque
from datetime import datetime

from core.data_structures import AudioEvent, EmotionType, IntentType
from core.queue_manager import QueueManager
from audio.vad import SileroVAD
from audio.emotion_analyzer import EmotionAnalyzer, Emotion
from audio.transcriber import Transcriber
from audio.intent_classifier import IntentClassifier, Intent

logger = logging.getLogger(__name__)


class AudioPipeline:
    """3-stage hierarchical audio pipeline.

    Captures audio, analyzes with VAD → emotion → intent, emits AudioEvent.

    Attributes:
        queue_manager: Queue manager for emitting events
        vad: Silero VAD for voice detection
        emotion: DistilHuBERT for emotion recognition
        transcriber: Faster-Whisper for speech-to-text
        intent: DistilBERT for intent classification
    """

    # Audio configuration
    SAMPLE_RATE = 16000  # 16 kHz
    FRAME_DURATION_MS = 100  # 100ms frames
    CHUNK_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 1600 samples
    BUFFER_DURATION_SEC = 5.0  # 5-second sliding window
    BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DURATION_SEC)  # 80,000 samples

    def __init__(
        self,
        queue_manager: QueueManager,
        vad_model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        """Initialize audio pipeline.

        Args:
            queue_manager: QueueManager for emit AudioEvents
            vad_model_path: Path to Silero VAD ONNX model (optional)
            device: "cpu" or "cuda"
        """
        self.queue_manager = queue_manager
        self.device = device

        # Initialize stages
        self.vad = SileroVAD(model_path=vad_model_path)
        self.emotion = EmotionAnalyzer(device=device)
        self.transcriber = Transcriber(device=device)
        self.intent = IntentClassifier(device=device)

        # Audio buffer (5-second sliding window)
        self.audio_buffer = deque(maxlen=self.BUFFER_SIZE)

        # Thread control
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # Metrics
        self.chunks_processed = 0
        self.events_emitted = 0
        self.vad_triggers = 0

        logger.info(
            f"AudioPipeline initialized: "
            f"sr={self.SAMPLE_RATE}Hz, "
            f"chunk={self.CHUNK_SIZE}, "
            f"buffer={self.BUFFER_SIZE}"
        )

    def start(self) -> None:
        """Start audio pipeline in background thread."""
        with self._lock:
            if self._running:
                logger.warning("AudioPipeline already running")
                return
            self._running = True

        self._thread = threading.Thread(target=self._audio_loop, daemon=False)
        self._thread.start()
        logger.info("AudioPipeline started")

    def stop(self) -> None:
        """Stop audio pipeline."""
        with self._lock:
            if not self._running:
                logger.warning("AudioPipeline not running")
                return
            self._running = False

        if self._thread:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.error("AudioPipeline thread did not stop cleanly")
            logger.info("AudioPipeline stopped")

    def submit_audio_chunk(self, audio_chunk: np.ndarray) -> None:
        """Submit audio chunk for processing.

        Args:
            audio_chunk: Audio samples (numpy array, should be 16kHz mono)
        """
        if len(audio_chunk) == 0:
            return

        # Add to buffer
        audio_chunk = np.asarray(audio_chunk, dtype=np.float32)
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.flatten()

        for sample in audio_chunk:
            self.audio_buffer.append(sample)

    def _audio_loop(self) -> None:
        """Main audio processing loop (background thread).

        Runs continuously while _running=True.
        """
        logger.info("Audio loop started")
        vad_state = {"h": np.zeros((2, 1, 64), dtype=np.float32),
                     "c": np.zeros((2, 1, 64), dtype=np.float32)}

        while self._running:
            try:
                # Process buffer every 100ms
                if len(self.audio_buffer) >= self.CHUNK_SIZE:
                    chunk = np.array(list(self.audio_buffer)[-self.CHUNK_SIZE:])
                    self.chunks_processed += 1

                    # Stage 1: VAD check
                    vad_conf, vad_state = self.vad.process_chunk(chunk, vad_state)

                    if vad_conf >= 0.5:  # Voice detected
                        self.vad_triggers += 1
                        logger.debug(f"VAD trigger: conf={vad_conf:.2f}")

                        # Get full buffer for Stage 2-3 analysis
                        buffer_array = np.array(list(self.audio_buffer), dtype=np.float32)

                        # Stage 2: Emotion analysis
                        emotion, emotion_conf = self.emotion.analyze(buffer_array)

                        # Stage 3a: Transcription
                        transcript, transcript_conf = self.transcriber.transcribe(buffer_array)

                        # Stage 3b: Intent classification
                        intent, intent_conf = self.intent.classify(transcript)

                        # Create AudioEvent
                        event = AudioEvent(
                            timestamp=datetime.now().timestamp(),
                            transcript=transcript,
                            emotion=EmotionType(emotion.value) if emotion else EmotionType.NEUTRAL,
                            emotion_confidence=emotion_conf,
                            intent=IntentType(intent.value) if intent else IntentType.NEUTRAL,
                            intent_confidence=intent_conf,
                            sample_duration=len(self.audio_buffer) / self.SAMPLE_RATE,
                            vad_confidence=vad_conf,
                        )

                        # Emit event to fusion manager
                        success = self.queue_manager.put_audio_event(event, block=False)
                        if success:
                            self.events_emitted += 1
                            logger.info(
                                f"AudioEvent: '{transcript[:50]}...' "
                                f"emotion={emotion.value}({emotion_conf:.1f}) "
                                f"intent={intent.value}({intent_conf:.1f})"
                            )
                        else:
                            logger.warning("Audio queue full - event dropped")

                # Sleep briefly to prevent busy-waiting
                time.sleep(0.05)

            except Exception as e:
                logger.error(f"Error in audio loop: {e}", exc_info=True)
                time.sleep(0.1)

        logger.info(f"Audio loop stopped (processed={self.chunks_processed}, "
                   f"emitted={self.events_emitted}, vad_triggers={self.vad_triggers})")

    def get_status(self) -> dict:
        """Get audio pipeline status.

        Returns:
            dict: Status information
        """
        return {
            "running": self._running,
            "chunks_processed": self.chunks_processed,
            "vad_triggers": self.vad_triggers,
            "events_emitted": self.events_emitted,
            "buffer_size": len(self.audio_buffer),
        }
