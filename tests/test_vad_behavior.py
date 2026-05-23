import numpy as np
from audio.vad import SileroVAD


def test_vad_stub_mode():
    vad = SileroVAD(model_path=None)
    # In stub mode, process_chunk should return 0.0 confidence and preserve state
    state = {"h": np.zeros((2, 1, 64), dtype=np.float32), "c": np.zeros((2, 1, 64), dtype=np.float32)}
    conf, new_state = vad.process_chunk(np.zeros(512, dtype=np.float32), state)
    assert isinstance(conf, float)
    assert conf == 0.0


def test_is_speech_thresholds():
    vad = SileroVAD(model_path=None)
    assert not vad.is_speech(0.1, threshold=0.5)
    assert vad.is_speech(0.6, threshold=0.5)
