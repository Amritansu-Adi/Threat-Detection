import numpy as np
import pytest
from audio.transcriber import Transcriber


def test_transcriber_stub_or_loaded():
    t = Transcriber()
    samples = np.zeros(16000 * 1, dtype=np.float32)

    transcript, conf = t.transcribe(samples)
    # If model not loaded, expect stubbed empty transcript
    if not t.loaded:
        assert transcript == ""
        assert conf == 0.0
    else:
        # When loaded, ensure it returns a string and a numeric confidence
        assert isinstance(transcript, str)
        assert isinstance(conf, float)
