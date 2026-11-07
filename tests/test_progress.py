"""Offline progress reporting tests with a deterministic clock."""

from video_transcriber.progress import ProgressReporter


def test_opt_in_stage_timing_and_summary(capsys):
    times = iter([0.0, 1.0, 3.5, 4.0])
    reporter = ProgressReporter(True, clock=lambda: next(times))
    reporter.start("download")
    reporter.finish("download")
    reporter.summary()
    output = capsys.readouterr().out
    assert "[PROGRESS] download: starting" in output
    assert "[PROGRESS] download: done (2.5s)" in output
    assert "[PERF] Total elapsed: 4.0s" in output


def test_default_silent(capsys):
    reporter = ProgressReporter()
    reporter.start("download")
    reporter.download_hook({"status": "downloading", "downloaded_bytes": 50, "total_bytes": 100})
    reporter.segment_hook(4, 10)
    reporter.finish("download")
    reporter.summary()
    assert capsys.readouterr().out == ""


def test_progress_clamps_throttles_and_rejects_unknown_values(capsys):
    reporter = ProgressReporter(True)
    reporter.download_hook({"status": "downloading", "downloaded_bytes": 1, "total_bytes": 0})
    reporter.download_hook({"status": "downloading", "downloaded_bytes": 5, "total_bytes": 100})
    reporter.download_hook({"status": "downloading", "downloaded_bytes": 6, "total_bytes": 100})
    reporter.download_hook({"status": "downloading", "downloaded_bytes": 50, "total_bytes": 100})
    reporter.download_hook({"status": "finished", "filename": "token=SECRET"})
    output = capsys.readouterr().out
    assert "[PROGRESS] download: 5%" in output
    assert "[PROGRESS] download: 50%" in output
    assert "[PROGRESS] download: 100%" in output
    assert output.count("[PROGRESS] download:") == 3
    assert "SECRET" not in output


def test_transcription_uses_known_duration_only(capsys):
    reporter = ProgressReporter(True)
    reporter.segment_hook(3, None)
    reporter.segment_hook(3, 10)
    reporter.segment_hook(3.5, 10)
    reporter.segment_hook(11, 10)
    output = capsys.readouterr().out
    assert "[PROGRESS] transcription: 30%" in output
    assert "[PROGRESS] transcription: 100%" in output
    assert output.count("[PROGRESS] transcription:") == 2


def test_completed_transcription_reports_100_percent_only_on_success(capsys):
    reporter = ProgressReporter(True)
    reporter.start("transcription")
    reporter.segment_hook(9, 10)
    reporter.finish("transcription")
    logs = capsys.readouterr().out
    assert "[PROGRESS] transcription: 90%" in logs
    assert "[PROGRESS] transcription: 100%" in logs


def test_progress_resets_for_next_batch_file(capsys):
    reporter = ProgressReporter(True)
    for _ in range(2):
        reporter.start("transcription")
        reporter.segment_hook(3, 10)
        reporter.finish("transcription")
    logs = capsys.readouterr().out
    assert logs.count("[PROGRESS] transcription: 30%") == 2
    assert logs.count("[PROGRESS] transcription: 100%") == 2
