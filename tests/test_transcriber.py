import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from video_transcriber.config import TranscriptionConfig
from video_transcriber.transcriber import Segment, Transcriber


@pytest.mark.parametrize("language", [None, "es"])
def test_model_options_and_segments(monkeypatch, language):
    model = Mock()
    model.transcribe.return_value = (
        iter([SimpleNamespace(start=1.25, end=8.75, text=" Hola ")]),
        SimpleNamespace(language="es", language_probability=0.99),
    )
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=factory))
    engine = Transcriber(
        TranscriptionConfig(language=language, model_dir=Path("models"), local_files_only=True)
    )
    for _ in range(2):
        result = engine.transcribe(Path("video.webm"))
    factory.assert_called_once_with(
        "medium", device="cpu", compute_type="int8", download_root="models", local_files_only=True
    )
    model.transcribe.assert_called_with("video.webm", language=language, vad_filter=True)
    # First generator was consumed; verify materialization separately.
    model.transcribe.return_value = (
        iter([SimpleNamespace(start=1.25, end=8.75, text=" Hola ")]),
        SimpleNamespace(language="es", language_probability=0.99),
    )
    result = engine.transcribe(Path("video.webm"))
    assert result.segments == (Segment(1.25, 8.75, "Hola"),)
    assert result.language == "es"
    assert result.language_probability == 0.99


def test_lazy_decoder_failure(monkeypatch):
    def broken():
        raise RuntimeError("decode failed")
        yield  # pragma: no cover

    model = Mock()
    model.transcribe.return_value = (broken(), SimpleNamespace(language="en"))
    monkeypatch.setitem(
        sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Mock(return_value=model))
    )
    with pytest.raises(RuntimeError, match="decode failed"):
        Transcriber(TranscriptionConfig()).transcribe(Path("bad.webm"))


@pytest.mark.parametrize(
    "options", [{"model": "invalid"}, {"device": "bad"}, {"compute_type": "bad"}]
)
def test_config_validation(options):
    with pytest.raises(ValueError):
        TranscriptionConfig(**options)
