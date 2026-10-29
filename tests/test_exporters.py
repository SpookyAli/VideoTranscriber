import pytest

from video_transcriber.exporters import TxtExporter, format_timestamp
from video_transcriber.transcriber import Segment, Transcription


@pytest.mark.parametrize(
    ("seconds", "expected"), [(0, "00:00:00"), (61.9, "00:01:01"), (360000, "100:00:00")]
)
def test_timestamp(seconds, expected):
    assert format_timestamp(seconds) == expected


@pytest.mark.parametrize("seconds", [-1, float("inf"), float("nan")])
def test_invalid_timestamp(seconds):
    with pytest.raises(ValueError):
        format_timestamp(seconds)


def test_txt_output(tmp_path):
    path = tmp_path / "nested" / "video.txt"
    result = Transcription((Segment(1, 8, "¡Hola!"), Segment(9, 15, "Adiós.")), "es", 1)
    TxtExporter().export(result, path)
    assert (
        path.read_bytes()
        == ("[00:00:01 - 00:00:08] ¡Hola!\n\n[00:00:09 - 00:00:15] Adiós.\n\n").encode()
    )
    with pytest.raises(FileExistsError):
        TxtExporter().export(result, path)
    TxtExporter(timestamps=False).export(result, path, overwrite=True)
    assert path.read_text(encoding="utf-8") == "¡Hola!\n\nAdiós.\n\n"
    assert list(path.parent.iterdir()) == [path]


def test_empty(tmp_path):
    path = tmp_path / "empty.txt"
    TxtExporter().export(Transcription((), "en", 0), path)
    assert path.read_bytes() == b""


def test_failed_write_preserves_output(tmp_path, monkeypatch):
    path = tmp_path / "video.txt"
    path.write_text("original")

    def fail(*args):
        raise OSError("disk failure")

    monkeypatch.setattr("video_transcriber.exporters.os.replace", fail)
    with pytest.raises(OSError, match="disk failure"):
        TxtExporter().export(Transcription((), "en", 1), path, overwrite=True)
    assert path.read_text() == "original"
    assert list(tmp_path.iterdir()) == [path]
