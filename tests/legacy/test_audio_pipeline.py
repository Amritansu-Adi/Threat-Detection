"""Integration test for Phase 3 audio pipeline.

Tests each stage of the audio pipeline:
1. VAD: Voice detection
2. Emotion: Emotion recognition from audio
3. Transcriber: Speech-to-text
4. Intent: Intent classification from text

Note: This test uses synthetic/stub data since actual audio models require downloads.
In production, these models would be downloaded from Hugging Face/GitHub on first run.
"""

import logging
import numpy as np
from typing import Tuple

from audio.vad import SileroVAD
from audio.emotion_analyzer import EmotionAnalyzer, Emotion
from audio.transcriber import Transcriber
from audio.intent_classifier import IntentClassifier, Intent
from core.data_structures import AudioEvent, EmotionType, IntentType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def generate_synthetic_audio(duration_sec: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate synthetic audio for testing.

    Creates a simple sine wave at 440 Hz (A note).

    Args:
        duration_sec: Duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        Audio array (numpy)
    """
    samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, samples, dtype=np.float32)
    frequency = 440.0  # A note
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    # Add some noise for realism
    audio += np.random.normal(0, 0.01, samples).astype(np.float32)
    return audio


def test_vad_stage():
    """Test Stage 1: VAD initialization and basic operation."""
    logger.info("=" * 80)
    logger.info("TEST: Stage 1 - Silero VAD")
    logger.info("=" * 80)

    try:
        vad = SileroVAD(model_path=None)  # None = stub mode
        logger.info("✓ VAD initialized (stub mode)")

        # Test with synthetic audio
        audio = generate_synthetic_audio(0.5)
        state = {"h": np.zeros((2, 1, 64), dtype=np.float32),
                 "c": np.zeros((2, 1, 64), dtype=np.float32)}

        conf, new_state = vad.process_chunk(audio[:512], state)
        logger.info(f"✓ VAD processed chunk: conf={conf:.2f}")

        # Verify state update
        assert new_state is not None, "State should be updated"
        logger.info(f"✓ VAD state updated")

        logger.info("=" * 80)
        logger.info("Stage 1 tests: ✅ PASSED")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ Stage 1 FAILED: {e}", exc_info=True)
        return False

    return True


def test_emotion_stage():
    """Test Stage 2: Emotion analyzer initialization and operation."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Stage 2 - Emotion Analysis (DistilHuBERT)")
    logger.info("=" * 80)

    try:
        em = EmotionAnalyzer(device="cpu")
        logger.info(f"✓ Emotion analyzer initialized (loaded: {em.loaded})")

        # Test with synthetic audio
        audio = generate_synthetic_audio(1.0)

        emotion, conf = em.analyze(audio)
        logger.info(f"✓ Emotion analyzed: {emotion.value} (conf={conf:.2f})")

        # Verify result
        assert isinstance(emotion, Emotion), "Should return Emotion enum"
        assert 0 <= conf <= 1, "Confidence should be 0-1"
        logger.info(f"✓ Emotion result valid")

        # Test high-confidence check
        is_high = em.is_high_confidence(0.85)
        assert is_high, "0.85 should be high confidence"
        logger.info(f"✓ High-confidence check working")

        logger.info("=" * 80)
        logger.info("Stage 2 tests: ✅ PASSED")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ Stage 2 FAILED: {e}", exc_info=True)
        return False

    return True


def test_transcriber_stage():
    """Test Stage 3a: Transcriber initialization and operation."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Stage 3a - Transcription (Faster-Whisper)")
    logger.info("=" * 80)

    try:
        trans = Transcriber(model_size="tiny", device="cpu")
        logger.info(f"✓ Transcriber initialized (loaded: {trans.loaded})")

        # Test with synthetic audio
        audio = generate_synthetic_audio(2.0)

        transcript, conf = trans.transcribe(audio)
        logger.info(f"✓ Transcribe result: '{transcript}' (conf={conf:.2f})")

        # Verify result
        assert isinstance(transcript, str), "Should return string"
        assert 0 <= conf <= 1, "Confidence should be 0-1"
        logger.info(f"✓ Transcription result valid")

        logger.info("=" * 80)
        logger.info("Stage 3a tests: ✅ PASSED")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ Stage 3a FAILED: {e}", exc_info=True)
        return False

    return True


def test_intent_stage():
    """Test Stage 3b: Intent classifier."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Stage 3b - Intent Classification (DistilBERT)")
    logger.info("=" * 80)

    try:
        ic = IntentClassifier(device="cpu")
        logger.info(f"✓ Intent classifier initialized (loaded: {ic.loaded})")

        # Test with various text examples
        test_cases = [
            ("Help! Fire!", Intent.DISTRESS),
            ("Watch out!", Intent.DISTRESS),
            ("I'll kill you", Intent.THREAT),
            ("The weather is nice", Intent.NEUTRAL),
            ("Did you know it rained yesterday?", Intent.INFORMATIONAL),
        ]

        logger.info("\nTesting intent classification on example texts:")
        for text, expected_intent in test_cases:
            intent, conf = ic.classify(text)
            logger.info(
                f"  Text: '{text}'"
                f"\n    → Intent: {intent.value} (conf={conf:.2f})"
                f"\n    → Expected: {expected_intent.value}"
            )

        # Test get_all_scores
        text = "I'll hurt you"
        scores = ic.get_all_scores(text)
        logger.info(f"\n  All scores for '{text}':")
        for label, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"    - {label}: {score:.2f}")

        # Test high-confidence threshold
        is_high = ic.is_high_confidence(0.80)
        assert is_high, "0.80 should be high confidence"
        logger.info(f"\n✓ High-confidence check working")

        # Test context filtering
        should_filter = ic.should_filter_as_context(Intent.THREAT, "happy")
        assert should_filter, "Threat intent with happy emotion should be filtered"
        logger.info(f"✓ Context filtering working (threat + happy = filter)")

        logger.info("=" * 80)
        logger.info("Stage 3b tests: ✅ PASSED")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ Stage 3b FAILED: {e}", exc_info=True)
        return False

    return True


def test_audio_event_creation():
    """Test AudioEvent data class creation."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: AudioEvent Data Structure")
    logger.info("=" * 80)

    try:
        event = AudioEvent(
            timestamp=1000.0,
            transcript="Help me!",
            emotion=EmotionType.FEARFUL,
            emotion_confidence=0.85,
            intent=IntentType.DISTRESS,
            intent_confidence=0.90,
            sample_duration=5.0,
            vad_confidence=0.95,
        )

        logger.info(f"✓ AudioEvent created:")
        logger.info(f"    Timestamp: {event.timestamp}")
        logger.info(f"    Transcript: '{event.transcript}'")
        logger.info(f"    Emotion: {event.emotion.value} (conf={event.emotion_confidence})")
        logger.info(f"    Intent: {event.intent.value} (conf={event.intent_confidence})")
        logger.info(f"    Duration: {event.sample_duration}s")
        logger.info(f"    VAD: {event.vad_confidence}")

        # Verify all fields
        assert event.transcript == "Help me!", "Transcript mismatch"
        assert event.emotion == EmotionType.FEARFUL, "Emotion mismatch"
        assert event.intent == IntentType.DISTRESS, "Intent mismatch"
        logger.info(f"✓ All fields validated")

        logger.info("=" * 80)
        logger.info("AudioEvent tests: ✅ PASSED")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ AudioEvent test FAILED: {e}", exc_info=True)
        return False

    return True


if __name__ == "__main__":
    logger.info("\n" + "🎤 AUDIO PIPELINE TESTS" + "\n")

    results = []
    results.append(("Stage 1: VAD", test_vad_stage()))
    results.append(("Stage 2: Emotion", test_emotion_stage()))
    results.append(("Stage 3a: Transcriber", test_transcriber_stage()))
    results.append(("Stage 3b: Intent", test_intent_stage()))
    results.append(("AudioEvent", test_audio_event_creation()))

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if not passed:
            all_passed = False

    logger.info("=" * 80)

    if all_passed:
        logger.info("\n🎉 ALL AUDIO TESTS PASSED!\n")
        exit(0)
    else:
        logger.info("\n❌ SOME TESTS FAILED\n")
        exit(1)
