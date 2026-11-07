"""Bounded, privacy-safe CLI progress and timing output."""

import math
import time
from collections.abc import Callable


class ProgressReporter:
    """Emit concise milestones without exposing paths or upstream metadata."""

    def __init__(self, enabled: bool = False, *, clock: Callable[[], float] = time.perf_counter):
        self.enabled = enabled
        self.clock = clock
        self.started = clock()
        self.stage_started: dict[str, float] = {}
        self.last_percent: dict[str, int] = {}
        self.stage_durations: dict[str, float] = {}
        self.media_seconds: float | None = None

    def start(self, stage: str) -> None:
        self.stage_started[stage] = self.clock()
        if self.enabled:
            print(f"[PROGRESS] {stage}: starting", flush=True)

    def finish(self, stage: str) -> None:
        since = self.stage_started.pop(stage, None)
        if since is None:
            return
        seconds = max(0.0, self.clock() - since)
        self.stage_durations[stage] = seconds
        if self.enabled:
            print(f"[PROGRESS] {stage}: done ({seconds:.1f}s)", flush=True)

    def fraction(self, stage: str, done: float, total: float | None) -> None:
        if not self.enabled or total is None:
            return
        if not (math.isfinite(done) and math.isfinite(total)) or total <= 0:
            return
        percent = min(100, max(0, int(100 * done / total)))
        previous = self.last_percent.get(stage, -10)
        if percent < 100 and percent < previous + 10:
            return
        if percent <= previous:
            return
        self.last_percent[stage] = percent
        print(f"[PROGRESS] {stage}: {percent}%", flush=True)

    def download_hook(self, event: dict) -> None:
        """Consume yt-dlp progress events without printing URLs, filenames or tokens."""
        if event.get("status") == "finished":
            self.fraction("download", 1, 1)
        elif event.get("status") == "downloading":
            downloaded = event.get("downloaded_bytes")
            total = event.get("total_bytes") or event.get("total_bytes_estimate")
            if isinstance(downloaded, (int, float)) and isinstance(total, (int, float)):
                self.fraction("download", downloaded, total)

    def segment_hook(self, end_seconds: float, duration_seconds: float | None) -> None:
        if duration_seconds is not None and math.isfinite(duration_seconds) and duration_seconds > 0:
            self.media_seconds = duration_seconds
        self.fraction("transcription", end_seconds, duration_seconds)

    def summary(self) -> None:
        if not self.enabled:
            return
        elapsed = max(0.0, self.clock() - self.started)
        print(f"[PERF] Total elapsed: {elapsed:.1f}s", flush=True)
        transcription = self.stage_durations.get("transcription")
        if transcription and self.media_seconds and self.media_seconds > 0:
            ratio = transcription / self.media_seconds
            print(f"[PERF] Transcription RTF: {ratio:.2f}x", flush=True)
