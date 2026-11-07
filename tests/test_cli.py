import json
from unittest.mock import Mock

import pytest

from video_transcriber.cli import build_parser, discover_inputs, main
from video_transcriber.transcriber import Segment, Transcription


@pytest.fixture
def engine(monkeypatch):
    model = Mock()
    model.transcribe.return_value = Transcription((Segment(1, 2, "Hello"),), "en", 1)
    factory = Mock(return_value=model)
    monkeypatch.setattr("video_transcriber.cli.Transcriber", factory)
    return factory, model


def test_arguments():
    args = build_parser().parse_args(
        [
            "transcribe",
            "--input",
            "input",
            "--language",
            "es",
            "--model",
            "large-v3",
            "--no-timestamps",
            "--overwrite",
            "--device",
            "cuda",
            "--compute-type",
            "float16",
        ]
    )
    assert (args.language, args.model, args.device, args.compute_type) == (
        "es",
        "large-v3",
        "cuda",
        "float16",
    )
    assert args.no_timestamps and args.overwrite
    defaults = build_parser().parse_args(["transcribe", "--input", "input"])
    assert (defaults.model, defaults.device, defaults.compute_type) == ("medium", "cpu", "int8")


@pytest.mark.parametrize(
    "arguments", [[], ["transcribe"], ["transcribe", "--input", "x", "--model", "bad"]]
)
def test_bad_arguments(arguments):
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(arguments)
    assert error.value.code == 2


def test_input_validation(tmp_path, engine):
    with pytest.raises(ValueError, match="does not exist"):
        discover_inputs(tmp_path / "missing")
    with pytest.raises(ValueError, match="No WEBM"):
        discover_inputs(tmp_path)
    text = tmp_path / "x.txt"
    text.touch()
    with pytest.raises(ValueError, match="Expected a WEBM"):
        discover_inputs(text)
    assert main(["transcribe", "--input", str(text)]) == 2
    engine[0].assert_not_called()


def test_batch_skip_and_overwrite(tmp_path, engine, capsys):
    inputs, output = tmp_path / "input", tmp_path / "output"
    inputs.mkdir()
    output.mkdir()
    for name in ("a.webm", "b.WEBM", "ignored.txt"):
        (inputs / name).touch()
    (output / "a.txt").write_text("existing")
    arguments = ["transcribe", "--input", str(inputs), "--output", str(output)]
    assert main(arguments) == 0
    engine[0].assert_called_once()
    engine[1].transcribe.assert_called_once_with(inputs / "b.WEBM")
    assert (output / "a.txt").read_text() == "existing"
    assert "1 succeeded, 1 skipped, 0 failed" in capsys.readouterr().out
    engine[0].reset_mock()
    assert main(arguments) == 0
    engine[0].assert_not_called()
    assert main(arguments + ["--overwrite", "--no-timestamps"]) == 0
    assert (output / "a.txt").read_text() == "Hello\n\n"


def test_batch_continues_after_failure(tmp_path, engine, capsys):
    for name in ("a.webm", "b.webm"):
        (tmp_path / name).touch()
    engine[1].transcribe.side_effect = [RuntimeError("bad media"), Transcription((), "en", 1)]
    assert main(["transcribe", "--input", str(tmp_path), "--output", str(tmp_path / "out")]) == 1
    assert engine[1].transcribe.call_count == 2
    assert (tmp_path / "out" / "b.txt").exists()
    captured = capsys.readouterr()
    assert "bad media" in captured.err
    assert "1 succeeded, 0 skipped, 1 failed" in captured.out


def test_model_load_failure(tmp_path, engine, capsys):
    (tmp_path / "a.webm").touch()
    engine[0].side_effect = RuntimeError("model unavailable")
    assert main(["transcribe", "--input", str(tmp_path), "--output", str(tmp_path / "out")]) == 1
    assert "Error loading model: model unavailable" in capsys.readouterr().err


def test_output_failure(tmp_path, engine):
    source = tmp_path / "a.webm"
    source.touch()
    output = tmp_path / "out"
    output.write_text("not a directory")
    assert main(["transcribe", "--input", str(source), "--output", str(output)]) == 2
    engine[0].assert_not_called()


def test_export_failure(tmp_path, engine, monkeypatch):
    source = tmp_path / "a.webm"
    source.touch()
    monkeypatch.setattr(
        "video_transcriber.cli.TxtExporter.export", Mock(side_effect=OSError("disk"))
    )
    assert main(["transcribe", "--input", str(source), "--output", str(tmp_path / "out")]) == 1


def test_local_progress_opt_in(tmp_path, engine, capsys):
    media = tmp_path / "clip.webm"
    media.touch()
    out = tmp_path / "out"
    assert main(["transcribe", "--input", str(media), "--output", str(out), "--progress"]) == 0
    output = capsys.readouterr().out
    assert "[PROGRESS] model: starting" in output
    assert "[PROGRESS] transcription: done" in output
    assert "[PERF] Total elapsed:" in output


def test_local_default_no_progress(tmp_path, engine, capsys):
    media = tmp_path / "clip.webm"
    media.touch()
    assert main(["transcribe", "--input", str(media), "--output", str(tmp_path / "out")]) == 0
    output = capsys.readouterr().out
    assert "[PROGRESS]" not in output
    assert "[PERF]" not in output



def test_local_all_formats_one_transcription_and_skip(tmp_path, engine, capsys):
    media = tmp_path / "demo.webm"
    media.touch()
    out = tmp_path / "out"
    args = ["transcribe", "--input", str(media), "--output", str(out), "--format", "all"]
    assert main(args) == 0
    engine[1].transcribe.assert_called_once_with(media)
    assert {p.suffix for p in out.iterdir()} == {".txt", ".srt", ".json"}
    assert (out / "demo.txt").read_text() == "[00:00:01 - 00:00:02] Hello\n\n"
    assert "00:00:01,000 --> 00:00:02,000" in (out / "demo.srt").read_text()
    assert json.loads((out / "demo.json").read_text())["segments"][0]["text"] == "Hello"
    engine[0].reset_mock()
    assert main(args) == 0
    engine[0].assert_not_called()
    assert "all requested outputs exist" in capsys.readouterr().out


def test_local_partial_missing_format_retains_existing(tmp_path, engine):
    media = tmp_path / "demo.webm"
    media.touch()
    out = tmp_path / "out"
    out.mkdir()
    existing = out / "demo.txt"
    existing.write_text("original")
    args = ["transcribe", "--input", str(media), "--output", str(out), "--format", "all"]
    assert main(args) == 0
    assert existing.read_text() == "original"
    assert (out / "demo.srt").exists()
    assert (out / "demo.json").exists()
    assert engine[1].transcribe.call_count == 1


def test_local_srt_only_and_json_only(tmp_path, engine):
    media = tmp_path / "demo.webm"
    media.touch()
    out = tmp_path / "out"
    for fmt in ("srt", "json"):
        args = ["transcribe", "--input", str(media), "--output", str(out), "--format", fmt]
        assert main(args) == 0
    assert not (out / "demo.txt").exists()
    assert (out / "demo.srt").exists()
    assert (out / "demo.json").exists()


def test_local_no_timestamps_does_not_change_srt(tmp_path, engine):
    media = tmp_path / "demo.webm"
    media.touch()
    out = tmp_path / "out"
    args = [
        "transcribe", "--input", str(media), "--output", str(out),
        "--format", "all", "--no-timestamps",
    ]
    assert main(args) == 0
    assert (out / "demo.txt").read_text() == "Hello\n\n"
    assert "-->" in (out / "demo.srt").read_text()


def test_local_formats_are_race_safe(tmp_path, engine, monkeypatch):
    media = tmp_path / "demo.webm"
    media.touch()
    out = tmp_path / "out"
    out.mkdir()
    target = out / "demo.json"

    from video_transcriber.exporters import JsonExporter

    original = JsonExporter.export

    def raced(self, result, path, overwrite=False):
        if not path.exists():
            path.write_text("other writer")
        return original(self, result, path, overwrite=overwrite)

    monkeypatch.setattr("video_transcriber.exporters.JsonExporter.export", raced)
    args = ["transcribe", "--input", str(media), "--output", str(out), "--format", "all"]
    assert main(args) == 0
    assert target.read_text() == "other writer"
    assert (out / "demo.srt").exists()
