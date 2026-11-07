"""Offline source and lifecycle tests; no browser profile or network is used."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from video_transcriber.cli import main
from video_transcriber.sources.youtube import (
    YouTubeError,
    YouTubeSource,
    select_audio,
    validate_url,
)
from video_transcriber.transcriber import Segment, Transcription

VIDEO = "Jqe34VMhFXI"
URL = f"https://youtu.be/{VIDEO}"


def track(identifier="a", language="es", note="Spanish original", **kwargs):
    return dict(
        format_id=identifier,
        language=language,
        format_note=note,
        vcodec="none",
        acodec="opus",
        abr=128,
        ext="webm",
        **kwargs,
    )


@pytest.mark.parametrize(
    "url",
    [
        URL,
        f"https://www.youtube.com/watch?v={VIDEO}&list=ignored",
        f"https://m.youtube.com/watch?v={VIDEO}",
        f"https://youtube.com/shorts/{VIDEO}",
        f"https://youtube.com/embed/{VIDEO}",
        f"https://youtube.com/live/{VIDEO}",
        f"https://music.youtube.com/watch?v={VIDEO}",
    ],
)
def test_urls(url):
    assert validate_url(url) == f"https://www.youtube.com/watch?v={VIDEO}"


@pytest.mark.parametrize(
    "url",
    [
        "bad",
        "https://youtube.com/playlist?list=x",
        "https://evil.com/watch?v=" + VIDEO,
        "https://youtube.com.evil.com/watch?v=" + VIDEO,
        "file:///etc/passwd",
        "https://user:secret@youtube.com/watch?v=" + VIDEO,
        "https://youtu.be:99999/" + VIDEO,
        "https://youtu.be/short",
        "https://youtu.be/" + VIDEO + "/extra",
    ],
)
def test_invalid_urls(url):
    with pytest.raises(ValueError, match="Invalid YouTube URL"):
        validate_url(url)


def test_language_and_original_selection():
    original = track(language="es-US")
    dubbed = track("dub", "es", "Spanish dubbed")
    english = track("en", "en", "English")
    assert select_audio([english, dubbed, original], "es_ES") is original
    assert select_audio([dubbed, original], "es", True) is original
    assert select_audio([english, original]) is original
    assert select_audio([english]) is None


def test_quality_non_drc_and_video_exclusion():
    drc = track("drc", note="Spanish original DRC")
    clean = track("clean")
    high = track("high")
    high["abr"] = 256
    video = track("video")
    video["vcodec"] = "vp9"
    assert select_audio([video, drc, clean, high], "es") is high
    assert select_audio([drc, clean], "es") is clean


def test_unavailable_language_and_audio():
    with pytest.raises(YouTubeError, match="Available languages: en, pt-br"):
        select_audio([track(language="en"), track(language="pt-BR")], "es")
    with pytest.raises(YouTubeError, match="audio-only"):
        select_audio([])
    drm = track()
    drm["has_drm"] = True
    with pytest.raises(YouTubeError):
        select_audio([drm])


@pytest.fixture
def downloader(monkeypatch):
    import yt_dlp

    instances = []
    info = dict(id=VIDEO, title="A title: / safe % name", formats=[track()])

    class FakeDL:
        def __init__(self, options):
            self.options = options
            instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download):
            assert not download
            return info

        def process_ie_result(self, result, download):
            assert download
            fmt = list(self.options["format"]({"formats": result["formats"]}))[0]
            assert fmt["format_id"] == "a"
            path = Path(self.options["outtmpl"].replace("%%", "%").replace("%(ext)s", "m4a"))
            path.write_bytes(b"audio")
            return {"requested_downloads": [{"filepath": str(path)}]}

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeDL)
    return instances, info, FakeDL


@pytest.mark.parametrize("browser", [None, "firefox", "chrome", "edge"])
def test_download_configuration_and_actual_path(tmp_path, downloader, browser, monkeypatch):
    monkeypatch.setattr("video_transcriber.sources.youtube.shutil.which", lambda _: None)
    source = YouTubeSource(URL, browser=browser, audio_language="es")
    source.inspect()
    preexisting = tmp_path / "user.webm"
    preexisting.write_bytes(b"keep")
    audio = source.download(tmp_path)
    assert audio.suffix == ".m4a" and VIDEO in audio.name
    assert audio.parent.parent == tmp_path
    options = downloader[0][0].options
    assert options["js_runtimes"] == {"deno": {}}
    assert not options.get("nocheckcertificate")
    assert "cookiefile" not in options
    assert options.get("cookiesfrombrowser") == ((browser,) if browser else None)
    assert source.deno is False
    source.cleanup()
    assert not audio.exists()
    assert preexisting.read_bytes() == b"keep"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Sign in with cookies SECRET", "access/authentication failed"),
        ("Video unavailable SECRET", "Video unavailable"),
        ("Signature JavaScript challenge SECRET", "Deno was not found on PATH"),
        ("Connection reset SECRET", "metadata extraction failed"),
    ],
)
def test_safe_errors(downloader, message, expected, monkeypatch):
    monkeypatch.setattr("video_transcriber.sources.youtube.shutil.which", lambda _: None)
    downloader[2].extract_info = Mock(side_effect=RuntimeError(message))
    with pytest.raises(YouTubeError) as error:
        YouTubeSource(URL).inspect()
    assert expected in str(error.value)
    assert "SECRET" not in str(error.value)
    assert error.value.stage == "METADATA_EXTRACTION"
    assert error.value.code in {
        "AUTHENTICATION_FAILED",
        "VIDEO_UNAVAILABLE",
        "JS_RUNTIME_UNAVAILABLE",
        "METADATA_EXTRACTION_FAILED",
    }


def test_download_failure_preserves_partial(tmp_path, downloader):
    source = YouTubeSource(URL)
    source.inspect()

    def fail(self, *args, **kwargs):
        (source.directory / "audio.part").write_bytes(b"partial")
        raise RuntimeError("network interrupted SECRET")

    downloader[2].process_ie_result = fail
    with pytest.raises(YouTubeError, match="media download failed") as failure:
        source.download(tmp_path)
    assert failure.value.stage == "DOWNLOAD"
    assert failure.value.code == "DOWNLOAD_FAILED"
    assert (source.directory / "audio.part").read_bytes() == b"partial"


@pytest.fixture
def youtube_cli(tmp_path, downloader, monkeypatch):
    engine = Mock()
    engine.transcribe.return_value = Transcription((Segment(1, 2, "Hola"),), "es", 1)
    factory = Mock(return_value=engine)
    monkeypatch.setattr("video_transcriber.cli.Transcriber", factory)
    downloads, output = tmp_path / "downloads", tmp_path / "out"
    args = [
        "transcribe",
        "--url",
        URL,
        "--download-dir",
        str(downloads),
        "--output",
        str(output),
        "--audio-language",
        "es",
        "--language",
        "en",
    ]
    return args, downloads, output, factory, engine


@pytest.mark.parametrize("keep", [False, True])
def test_cli_success_cleanup(youtube_cli, keep):
    args, downloads, output, factory, engine = youtube_cli
    assert main(args + (["--keep-audio"] if keep else [])) == 0
    assert len(list(output.glob("*.txt"))) == 1
    assert list(output.glob("*.txt"))[0].read_text() == "[00:00:01 - 00:00:02] Hola\n\n"
    assert bool(list(downloads.rglob("*.m4a"))) == keep
    assert factory.call_args.args[0].language == "en"


def test_cli_failure_preserves_audio(youtube_cli):
    args, downloads, output, factory, engine = youtube_cli
    engine.transcribe.side_effect = RuntimeError("corrupt audio")
    assert main(args) == 1
    assert list(downloads.rglob("*.m4a"))
    assert not list(output.glob("*.txt"))


def test_cli_skip_before_download_and_overwrite(youtube_cli, downloader):
    args, downloads, output, factory, engine = youtube_cli
    assert main(args + ["--keep-audio"]) == 0
    count = len(downloader[0])
    factory.reset_mock()
    assert main(args) == 0
    factory.assert_not_called()
    assert len(downloader[0]) == count + 1  # metadata only
    assert main(args + ["--overwrite"]) == 0
    assert len(list(downloads.rglob("*.m4a"))) == 1  # prior operation's audio retained


def test_cleanup_error_keeps_transcript(youtube_cli, monkeypatch, capsys):
    monkeypatch.setattr(YouTubeSource, "cleanup", Mock(side_effect=OSError("disk")))
    assert main(youtube_cli[0]) == 0
    assert list(youtube_cli[2].glob("*.txt"))
    assert "TXT saved" in capsys.readouterr().err


@pytest.mark.parametrize(
    "options",
    [
        ["--browser", "firefox"],
        ["--audio-language", "es"],
        ["--keep-audio"],
        ["--prefer-original-audio"],
        ["--download-dir", "input"],
        ["--url", URL],
    ],
)
def test_cli_local_rejects_youtube_options(options):
    with pytest.raises(SystemExit) as error:
        main(["transcribe", "--input", "local.webm", *options])
    assert error.value.code == 2


def test_cli_invalid_url(capsys):
    assert main(["transcribe", "--url", "https://evil.com"]) == 2
    assert "Invalid YouTube URL" in capsys.readouterr().err


def test_real_ytdlp_processing_without_network(tmp_path, monkeypatch):
    """Exercise the pinned library's selector/template/result contract offline."""
    from yt_dlp import YoutubeDL

    info = dict(
        id=VIDEO,
        title="Título % safe",
        extractor="youtube",
        extractor_key="Youtube",
        webpage_url=URL,
        formats=[dict(track(), url="https://example.invalid/audio", protocol="https")],
    )
    monkeypatch.setattr(YoutubeDL, "extract_info", lambda *a, **kw: dict(info))

    def local_download(self, filename, video_info, subtitle=False):
        Path(filename).write_bytes(b"local fixture")
        return True, True

    monkeypatch.setattr(YoutubeDL, "dl", local_download)
    source = YouTubeSource(URL, audio_language="es")
    source.inspect()
    audio = source.download(tmp_path)
    assert audio.is_file()
    assert "%" in audio.name and "Título" in audio.name
    assert audio.suffix == ".webm"
    source.cleanup()


def test_model_load_failure_retains_download(youtube_cli):
    args, downloads, output, factory, engine = youtube_cli
    factory.side_effect = RuntimeError("missing model")
    assert main(args) == 1
    assert list(downloads.rglob("*.m4a"))
    assert not list(output.glob("*.txt"))


def test_export_failure_retains_download(youtube_cli, monkeypatch):
    monkeypatch.setattr(
        "video_transcriber.cli.TxtExporter.export", Mock(side_effect=OSError("disk full"))
    )
    assert main(youtube_cli[0]) == 1
    assert list(youtube_cli[1].rglob("*.m4a"))


def test_no_original_delegates_to_bestaudio(downloader, tmp_path):
    downloader[1]["formats"] = [track(note="Spanish")]
    original_process = downloader[2].process_ie_result

    def process(self, info, download):
        assert self.options["format"] == "bestaudio"
        self.options["format"] = lambda ctx: iter(info["formats"])
        return original_process(self, info, download)

    downloader[2].process_ie_result = process
    source = YouTubeSource(URL)
    source.inspect()
    assert source.download(tmp_path).is_file()


def test_download_path_must_be_owned(tmp_path, downloader):
    outside = tmp_path / "user.m4a"
    outside.write_bytes(b"keep")
    downloader[2].process_ie_result = Mock(
        return_value={"requested_downloads": [{"filepath": str(outside)}]}
    )
    source = YouTubeSource(URL)
    source.inspect()
    with pytest.raises(YouTubeError, match="valid owned audio"):
        source.download(tmp_path)
    assert outside.read_bytes() == b"keep"


def test_output_race_preserves_audio(youtube_cli, monkeypatch):
    def raced(self, result, target, overwrite=False):
        target.write_text("other writer", encoding="utf-8")
        raise FileExistsError

    monkeypatch.setattr("video_transcriber.cli.TxtExporter.export", raced)
    assert main(youtube_cli[0]) == 0
    assert list(youtube_cli[1].rglob("*.m4a"))
    assert next(youtube_cli[2].glob("*.txt")).read_text() == "other writer"


@pytest.mark.parametrize("language", ["", " ", "es/token", "123"])
def test_invalid_audio_language(language):
    with pytest.raises(ValueError, match="Invalid audio language"):
        YouTubeSource(URL, audio_language=language)


def test_long_unicode_title(downloader, tmp_path):
    downloader[1]["title"] = "日" * 200
    source = YouTubeSource(URL)
    source.inspect()
    assert len((source.stem + ".webm").encode("utf-8")) < 255
    assert source.download(tmp_path).is_file()


def test_interruption_retains_audio(youtube_cli, capsys):
    youtube_cli[4].transcribe.side_effect = KeyboardInterrupt
    assert main(youtube_cli[0]) == 130
    assert list(youtube_cli[1].rglob("*.m4a"))
    assert "Interrupted" in capsys.readouterr().err


def test_processed_metadata_does_not_download_video(tmp_path, monkeypatch):
    from yt_dlp import YoutubeDL

    audio = dict(track(), url="https://example.invalid/audio", protocol="https")
    video = dict(track("video"), url="https://example.invalid/video", vcodec="vp9", ext="webm")
    info = dict(
        id=VIDEO,
        title="Video",
        extractor="youtube",
        extractor_key="Youtube",
        webpage_url=URL,
        formats=[audio, video],
        requested_formats=[video, audio],
    )
    monkeypatch.setattr(YoutubeDL, "extract_info", lambda *a, **kw: dict(info))
    calls = []

    def download(self, filename, video_info, subtitle=False):
        assert not video_info.get("requested_formats")
        assert video_info["vcodec"] == "none"
        assert video_info["url"].endswith("/audio")
        calls.append(filename)
        Path(filename).write_bytes(b"audio")
        return True, True

    monkeypatch.setattr(YoutubeDL, "dl", download)
    source = YouTubeSource(URL, audio_language="es")
    source.inspect()
    assert source.download(tmp_path).is_file()
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (403, "DOWNLOAD_HTTP_403"),
        (404, "DOWNLOAD_HTTP_404"),
        (429, "DOWNLOAD_HTTP_429"),
        (500, "DOWNLOAD_HTTP_5XX"),
        (503, "DOWNLOAD_HTTP_5XX"),
    ],
)
def test_download_http_error_stage_and_privacy(tmp_path, downloader, status, code, capsys):
    source = YouTubeSource(URL, audio_language="es")
    source.inspect()
    downloader[2].process_ie_result = Mock(
        side_effect=RuntimeError(
            f"ERROR: unable to download video data: HTTP Error {status}: Forbidden "
            "https://example.invalid/videoplayback?token=SECRET"
        )
    )
    with pytest.raises(YouTubeError) as failure:
        source.download(tmp_path)
    assert failure.value.code == code
    assert failure.value.stage == "DOWNLOAD"
    assert failure.value.http_status == status
    assert "SECRET" not in str(failure.value)
    assert "example.invalid" not in str(failure.value)
    assert "SECRET" not in capsys.readouterr().err


def test_metadata_http_error_not_labeled_download(downloader):
    downloader[2].extract_info = Mock(side_effect=RuntimeError("HTTP Error 403: Forbidden"))
    with pytest.raises(YouTubeError) as failure:
        YouTubeSource(URL).inspect()
    assert failure.value.code == "METADATA_EXTRACTION_FAILED"
    assert failure.value.stage == "METADATA_EXTRACTION"
    assert failure.value.http_status == 403


def test_audio_errors_have_structured_stage():
    with pytest.raises(YouTubeError) as missing:
        select_audio([])
    assert (missing.value.code, missing.value.stage) == (
        "AUDIO_FORMAT_NOT_FOUND",
        "AUDIO_SELECTION",
    )
    with pytest.raises(YouTubeError) as language:
        select_audio([track(language="en")], "es")
    assert (language.value.code, language.value.stage) == (
        "AUDIO_LANGUAGE_UNAVAILABLE",
        "AUDIO_SELECTION",
    )


def test_debug_stage_reporting_safe(youtube_cli, capsys):
    assert main(youtube_cli[0] + ["--debug"]) == 0
    logs = capsys.readouterr()
    assert "[DEBUG] Metadata extraction: OK" in logs.out
    assert "[DEBUG] Audio selection: OK" in logs.out
    assert "[DEBUG] Download: OK" in logs.out
    assert "Transcription: OK" in logs.out
    assert "TXT export: OK" in logs.out


def test_debug_http_403_safe(youtube_cli, downloader, capsys):
    downloader[2].process_ie_result = Mock(
        side_effect=RuntimeError(
            "unable to download video data: HTTP Error 403: Forbidden "
            "https://example.invalid/?Authorization=SECRET"
        )
    )
    assert main(youtube_cli[0] + ["--debug"]) == 1
    logs = capsys.readouterr()
    assert "[DEBUG] DOWNLOAD: FAILED" in logs.out
    assert "[DEBUG] HTTP status: 403" in logs.out
    assert "Failed [DOWNLOAD_HTTP_403] (DOWNLOAD)" in logs.err
    assert "SECRET" not in logs.out + logs.err
    assert "Authorization" not in logs.out + logs.err


def test_debug_rejects_malicious_format_metadata(youtube_cli, downloader, capsys):
    downloader[1]["formats"] = [
        track(identifier="a", language="es-US;SECRET", note="Spanish original")
    ]
    assert main(youtube_cli[0] + ["--debug"]) == 0
    logs = capsys.readouterr()
    assert "token=SECRET" not in logs.out + logs.err


def test_missing_deno_not_reported_for_unrelated_http_error(tmp_path, downloader, monkeypatch):
    monkeypatch.setattr("video_transcriber.sources.youtube.shutil.which", lambda _: None)
    source = YouTubeSource(URL)
    source.inspect()
    downloader[2].process_ie_result = Mock(side_effect=RuntimeError("HTTP Error 403: Forbidden"))
    with pytest.raises(YouTubeError) as failure:
        source.download(tmp_path)
    assert failure.value.code == "DOWNLOAD_HTTP_403"
    assert "Deno" not in str(failure.value)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("JavaScript challenge failed", "JS_CHALLENGE_FAILED"),
        ("Sign in with cookies", "AUTHENTICATION_FAILED"),
        ("Video unavailable", "VIDEO_UNAVAILABLE"),
    ],
)
def test_structured_extractor_errors(downloader, monkeypatch, message, expected):
    monkeypatch.setattr("video_transcriber.sources.youtube.shutil.which", lambda _: "deno")
    downloader[2].extract_info = Mock(side_effect=RuntimeError(message))
    with pytest.raises(YouTubeError) as failure:
        YouTubeSource(URL).inspect()
    assert failure.value.code == expected
    assert failure.value.stage == "METADATA_EXTRACTION"


def test_youtube_transcription_and_export_failure_codes(youtube_cli, monkeypatch, capsys):
    youtube_cli[4].transcribe.side_effect = RuntimeError("SECRET private model error")
    assert main(youtube_cli[0] + ["--debug"]) == 1
    stderr = capsys.readouterr().err
    assert "TRANSCRIPTION_FAILED" in stderr
    assert "SECRET" not in stderr

    youtube_cli[4].transcribe.side_effect = None
    monkeypatch.setattr(
        "video_transcriber.cli.TxtExporter.export",
        Mock(side_effect=RuntimeError("SECRET disk error")),
    )
    assert main(youtube_cli[0] + ["--debug"]) == 1
    stderr = capsys.readouterr().err
    assert "EXPORT_FAILED" in stderr
    assert "SECRET" not in stderr


def test_youtube_progress_hooks_and_cli(tmp_path, downloader, youtube_cli, capsys):
    args, _, _, factory, engine = youtube_cli
    assert main(args + ["--progress"]) == 0
    output = capsys.readouterr().out
    assert "[PROGRESS] metadata: done" in output
    assert "[PROGRESS] download: done" in output
    assert "[PROGRESS] transcription: done" in output
    assert "[PERF] Total elapsed:" in output
    # The progress hook must be passed into yt-dlp without changing normal paths.
    assert len(downloader[0][-1].options["progress_hooks"]) == 1
    engine.transcribe.assert_called_once()
    assert callable(engine.transcribe.call_args.kwargs["progress"])
