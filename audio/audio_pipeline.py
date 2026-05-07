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
from audio.intent_classifier import IntentClassifier

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
    PRE_ROLL_SEC = 0.4
    MAX_SPEECH_SEC = 4.0
    MIN_SPEECH_SEC = 0.1
    END_SILENCE_SEC = 0.35
    VAD_TRIGGER_THRESHOLD = 0.5
    IDLE_FLUSH_SEC = 0.35

    def __init__(
        self,
        queue_manager: QueueManager,
        vad_model_path: Optional[str] = None,
        device: str = "cpu",
        should_process_audio: Optional[Callable[[], bool]] = None,
    ):
        """Initialize audio pipeline.

        Args:
            queue_manager: QueueManager for emit AudioEvents
            vad_model_path: Path to Silero VAD ONNX model (optional)
            device: "cpu" or "cuda"
        """
        self.queue_manager = queue_manager
        self.device = device
        self.should_process_audio = should_process_audio

        # Initialize stages
        self.vad = SileroVAD(model_path=vad_model_path)
        self.emotion = EmotionAnalyzer(device=device)
        self.transcriber = Transcriber(device=device)
        self.intent = IntentClassifier(device=device)

        # Audio buffer (5-second sliding window)
        self.audio_buffer = deque(maxlen=self.BUFFER_SIZE)
        self._pending_chunks = deque()

        # Thread control
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # Metrics
        self.chunks_processed = 0
        self.events_emitted = 0
        self.vad_triggers = 0
        self.samples_submitted = 0
        self.segment_flushes = 0
        self.gated_drops = 0

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

        if self.should_process_audio is not None:
            try:
                if not self.should_process_audio():
                    self.gated_drops += len(audio_chunk)
                    return
            except Exception:
                # Fail-open: keep audio running if visibility gate errors.
                pass

        with self._lock:
            for sample in audio_chunk:
                self.audio_buffer.append(sample)
            for start in range(0, len(audio_chunk), self.CHUNK_SIZE):
                chunk_slice = audio_chunk[start:start + self.CHUNK_SIZE]
                if len(chunk_slice) > 0:
                    self._pending_chunks.append(chunk_slice.copy())
            self.samples_submitted += len(audio_chunk)

    def _audio_loop(self) -> None:
        """Main audio processing loop (background thread).

        Runs continuously while _running=True.
        """
        logger.info("Audio loop started")
        vad_state = {"h": np.zeros((2, 1, 64), dtype=np.float32),
                     "c": np.zeros((2, 1, 64), dtype=np.float32)}
        speech_active = False
        speech_chunks = []
        speech_samples = 0
        silence_samples = 0
        last_activity_time = time.time()
        last_vad_conf = 0.0
        pre_roll_samples = int(self.SAMPLE_RATE * self.PRE_ROLL_SEC)
        max_speech_samples = int(self.SAMPLE_RATE * self.MAX_SPEECH_SEC)
        min_speech_samples = int(self.SAMPLE_RATE * self.MIN_SPEECH_SEC)
        end_silence_samples = int(self.SAMPLE_RATE * self.END_SILENCE_SEC)

        while self._running:
            try:
                if self.should_process_audio is not None:
                    try:
                        if not self.should_process_audio():
                            speech_active = False
                            speech_chunks = []
                            speech_samples = 0
                            silence_samples = 0
                            with self._lock:
                                self._pending_chunks.clear()
                            time.sleep(0.02)
                            continue
                    except Exception:
                        pass

                with self._lock:
                    chunk = self._pending_chunks.popleft() if self._pending_chunks else None

                if chunk is None:
                    if speech_active and speech_samples >= min_speech_samples:
                        if time.time() - last_activity_time >= self.IDLE_FLUSH_SEC:
                            self._emit_segment(
                                speech_chunks,
                                speech_samples,
                                silence_samples,
                                last_vad_conf,
                            )
                            self.segment_flushes += 1
                            speech_active = False
                            speech_chunks = []
                            speech_samples = 0
                            silence_samples = 0
                    time.sleep(0.02)
                    continue

                self.chunks_processed += 1
                vad_conf, vad_state = self.vad.process_chunk(chunk, vad_state)
                last_activity_time = time.time()
                last_vad_conf = vad_conf

                if vad_conf >= self.VAD_TRIGGER_THRESHOLD:
                    self.vad_triggers += 1
                    logger.debug(f"VAD trigger: conf={vad_conf:.2f}")

                    if not speech_active:
                        speech_active = True
                        silence_samples = 0
                        history = list(self.audio_buffer)
                        if len(chunk) <= len(history):
                            history = history[:-len(chunk)]
                        pre_roll = np.array(history[-pre_roll_samples:], dtype=np.float32)
                        speech_chunks = [pre_roll] if pre_roll.size else []
                        speech_samples = int(pre_roll.size)

                    speech_chunks.append(chunk.copy())
                    speech_samples += len(chunk)
                    silence_samples = 0

                    if speech_samples >= max_speech_samples:
                        self._emit_segment(
                            speech_chunks,
                            speech_samples,
                            0,
                            vad_conf,
                        )
                        self.segment_flushes += 1
                        speech_active = False
                        speech_chunks = []
                        speech_samples = 0
                        silence_samples = 0
                elif speech_active:
                    speech_chunks.append(chunk.copy())
                    speech_samples += len(chunk)
                    silence_samples += len(chunk)

                    if silence_samples >= end_silence_samples:
                        voiced_samples = speech_samples - silence_samples
                        if voiced_samples >= min_speech_samples:
                            self._emit_segment(
                                speech_chunks,
                                speech_samples,
                                silence_samples,
                                vad_conf,
                            )
                            self.segment_flushes += 1
                        speech_active = False
                        speech_chunks = []
                        speech_samples = 0
                        silence_samples = 0

                time.sleep(0.005)

            except Exception as e:
                logger.error(f"Error in audio loop: {e}", exc_info=True)
                time.sleep(0.1)

        if speech_active and speech_samples >= min_speech_samples:
            self._emit_segment(
                speech_chunks,
                speech_samples,
                silence_samples,
                vad_conf=0.0,
            )
            self.segment_flushes += 1

        logger.info(f"Audio loop stopped (processed={self.chunks_processed}, "
                   f"emitted={self.events_emitted}, vad_triggers={self.vad_triggers})")

    def _emit_segment(
        self,
        speech_chunks,
        speech_samples: int,
        trailing_silence_samples: int,
        vad_conf: float,
    ) -> None:
        """Analyze one speech segment and emit a single AudioEvent."""
        if not speech_chunks:
            return

        buffer_array = np.concatenate(speech_chunks).astype(np.float32, copy=False)
        if trailing_silence_samples > 0 and trailing_silence_samples < len(buffer_array):
            buffer_array = buffer_array[:-trailing_silence_samples]

        if buffer_array.size == 0:
            return

        # Stage 2: Emotion analysis
        emotion, emotion_conf = self.emotion.analyze(buffer_array, sample_rate=self.SAMPLE_RATE)
        if emotion is None:
            emotion, emotion_conf = Emotion.NEUTRAL, 0.0

        transcript, transcript_conf = self.transcriber.transcribe(buffer_array)
        intent, intent_conf = self.intent.classify(transcript)
        raw_features = self._extract_audio_features(buffer_array)
        if emotion == Emotion.NEUTRAL and emotion_conf <= 0.55:
            emotion, emotion_conf = self._fallback_emotion_from_audio_features(raw_features)
        if not transcript.strip():
            fallback_intent, fallback_conf = self._fallback_intent_from_audio_features(
                raw_features,
                vad_conf,
            )
            if fallback_intent is not None and fallback_conf > intent_conf:
                intent, intent_conf = fallback_intent, fallback_conf

        event = AudioEvent(
            timestamp=datetime.now().timestamp(),
            transcript=transcript,
            emotion=EmotionType(emotion.value) if emotion else EmotionType.NEUTRAL,
            emotion_confidence=emotion_conf,
            intent=IntentType(intent.value) if intent else IntentType.NEUTRAL,
            intent_confidence=intent_conf,
            sample_duration=len(buffer_array) / self.SAMPLE_RATE,
            vad_confidence=vad_conf,
            raw_audio_features=raw_features,
        )

        success = self.queue_manager.put_audio_event(event, block=False)
        if success:
            self.events_emitted += 1
            logger.info(
                f"AudioEvent: '{transcript[:50]}...' "
                f"emotion={event.emotion.value}({emotion_conf:.2f}) "
                f"intent={intent.value}({intent_conf:.1f}) "
                f"stt_conf={transcript_conf:.1f}"
            )
        else:
            logger.warning("Audio queue full - event dropped")

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
            "pending_chunks": len(self._pending_chunks),
            "segment_flushes": self.segment_flushes,
            "gated_drops": self.gated_drops,
        }

    @staticmethod
    def _extract_audio_features(audio_array: np.ndarray) -> dict:
        """Extract simple robust features for fallback intent inference."""
        if audio_array.size == 0:
            return {"rms": 0.0, "peak": 0.0, "zcr": 0.0, "anomaly_score": 0.0, "sudden_anomaly": False}
        rms = float(np.sqrt(np.mean(np.square(audio_array))))
        peak = float(np.max(np.abs(audio_array)))
        if audio_array.size > 1:
            zcr = float(np.mean(np.abs(np.diff(np.signbit(audio_array)))))
            diff = np.diff(audio_array)
            transient = float(np.percentile(np.abs(diff), 99))
        else:
            zcr = 0.0
            transient = 0.0

        crest = peak / max(rms, 1e-6)
        anomaly_score = min(1.0, max(0.0, 0.25 * crest + 6.0 * transient + 1.8 * zcr))
        sudden_anomaly = anomaly_score >= 0.72 and peak >= 0.25

        return {
            "rms": rms,
            "peak": peak,
            "zcr": zcr,
            "transient": transient,
            "crest": crest,
            "anomaly_score": anomaly_score,
            "sudden_anomaly": sudden_anomaly,
        }

    @staticmethod
    def _fallback_intent_from_audio_features(features: dict, vad_conf: float):
        """Fallback when STT is unavailable: infer coarse distress from vocal intensity."""
        if vad_conf < 0.6:
            return IntentType.NEUTRAL, 0.0

        rms = float(features.get("rms", 0.0))
        peak = float(features.get("peak", 0.0))
        zcr = float(features.get("zcr", 0.0))

        high_arousal = (rms >= 0.03 and peak >= 0.25) or (peak >= 0.5 and zcr >= 0.03)
        if high_arousal:
            return IntentType.DISTRESS, min(0.84, 0.7 + vad_conf * 0.12)

        return IntentType.NEUTRAL, 0.0

    @staticmethod
    def _fallback_emotion_from_audio_features(features: dict):
        """Fallback emotion from acoustic arousal cues when transformer SER is weak/unavailable."""
        rms = float(features.get("rms", 0.0))
        peak = float(features.get("peak", 0.0))
        zcr = float(features.get("zcr", 0.0))
        anomaly = float(features.get("anomaly_score", 0.0))
        if anomaly >= 0.78 and peak >= 0.35:
            return Emotion.FEARFUL, min(0.86, 0.62 + 0.25 * anomaly)
        if (rms >= 0.03 and peak >= 0.30) or (zcr >= 0.06 and peak >= 0.24):
            return Emotion.ANGRY, 0.68
        return Emotion.NEUTRAL, 0.55
