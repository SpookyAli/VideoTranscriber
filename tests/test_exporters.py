import json

import pytest

from video_transcriber.exporters import (
    JsonExporter,
    SrtExporter,
    TxtExporter,
    exporters_for,
    format_srt_timestamp,
    format_timestamp,
)
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


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "00:00:00,000"),
        (1.9996, "00:00:02,000"),
        (59.9999, "00:01:00,000"),
        (360000.125, "100:00:00,125"),
    ],
)
def test_srt_millisecond_timestamps(seconds, expected):
    assert format_srt_timestamp(seconds) == expected


@pytest.mark.parametrize("seconds", [-0.1, float("inf"), float("nan")])
def test_invalid_srt_timestamp(seconds):
    with pytest.raises(ValueError):
        format_srt_timestamp(seconds)


def test_srt_unicode_cue_numbering_and_blank_segments(tmp_path):
    result = Transcription(
        (
            Segment(1.25, 3.75, "¡Hola!"),
            Segment(4, 5, " "),
            Segment(7, 9.5, "Adiós.\nHasta luego."),
        ),
        "es",
        0.97,
    )
    target = tmp_path / "captions.srt"
    SrtExporter().export(result, target)
    assert target.read_text(encoding="utf-8") == (
        "1\n00:00:01,250 --> 00:00:03,750\n¡Hola!\n\n"
        "2\n00:00:07,000 --> 00:00:09,500\nAdiós.\nHasta luego.\n\n"
    )
    with pytest.raises(FileExistsError):
        SrtExporter().export(result, target)


def test_json_schema_preserves_fractional_seconds_and_unicode(tmp_path):
    target = tmp_path / "captions.json"
    result = Transcription((Segment(1.2345, 2.5678, "¡Hola!"),), "es", 0.94)
    JsonExporter().export(result, target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": 1,
        "language": "es",
        "language_probability": 0.94,
        "segments": [{"start": 1.2345, "end": 2.5678, "text": "¡Hola!"}],
    }
    assert "¡Hola!" in target.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        JsonExporter().export(result, target)


def test_empty_srt_and_json(tmp_path):
    result = Transcription((), "en", 0.5)
    SrtExporter().export(result, tmp_path / "empty.srt")
    JsonExporter().export(result, tmp_path / "empty.json")
    assert (tmp_path / "empty.srt").read_bytes() == b""
    assert json.loads((tmp_path / "empty.json").read_text())["segments"] == []


@pytest.mark.parametrize(
    "segment",
    [
        Segment(-1, 2, "x"),
        Segment(4, 3, "x"),
        Segment(float("nan"), 1, "x"),
        Segment(0, float("inf"), "x"),
    ],
)
@pytest.mark.parametrize("exporter", [SrtExporter(), JsonExporter()])
def test_reject_invalid_segment_timing_without_creating_output(tmp_path, segment, exporter):
    target = tmp_path / ("bad" + exporter.suffix)
    with pytest.raises(ValueError):
        exporter.export(Transcription((segment,), "en", 0.9), target)
    assert not target.exists()


def test_json_rejects_nonfinite_language_probability(tmp_path):
    target = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        JsonExporter().export(Transcription((), "en", float("nan")), target)
    assert not target.exists()


@pytest.mark.parametrize("exporter", [SrtExporter(), JsonExporter()])
def test_multiformat_atomic_overwrite_and_cleanup(tmp_path, monkeypatch, exporter):
    target = tmp_path / ("result" + exporter.suffix)
    result = Transcription((Segment(0.5, 1.5, "Hi"),), "en", 1)
    target.write_text("original", encoding="utf-8")

    def fail(*args):
        raise OSError("disk failure")

    monkeypatch.setattr("video_transcriber.exporters.os.replace", fail)
    with pytest.raises(OSError):
        exporter.export(result, target, overwrite=True)
    assert target.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.iterdir()) == [target]


def test_export_selection():
    assert [exporter.suffix for exporter in exporters_for("all")] == [".txt", ".srt", ".json"]
    assert [exporter.suffix for exporter in exporters_for("srt")] == [".srt"]
    with pytest.raises(ValueError):
        exporters_for("pdf")
