"""Exercise the real decoder API without loading or downloading a Whisper model."""

import math
import struct
import wave

import numpy as np
import pytest
from faster_whisper.audio import decode_audio


@pytest.fixture
def audio_file(tmp_path):
    """Generate 0.1 seconds of mono PCM audio using only the standard library."""
    path = tmp_path / "tone.wav"
    sample_rate = 8000
    samples = [
        round(16000 * math.sin(2 * math.pi * 440 * index / sample_rate))
        for index in range(sample_rate // 10)
    ]
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return path


def test_faster_whisper_decodes_with_real_pyav(audio_file):
    audio = decode_audio(str(audio_file), sampling_rate=16000)

    assert audio.dtype == np.float32
    assert audio.shape == (1600,)
    assert np.isfinite(audio).all()
    assert 0.3 < np.max(np.abs(audio)) < 0.6
