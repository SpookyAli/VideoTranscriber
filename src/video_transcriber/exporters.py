"""Atomic export of structured transcriptions to TXT, SRT and JSON."""

import json
import math
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TextIO

from video_transcriber.transcriber import Transcription

FORMATS = ("txt", "srt", "json", "all")


class Exporter(Protocol):
    suffix: str

    def export(self, result: Transcription, path: Path, *, overwrite: bool = False) -> None: ...


def format_timestamp(seconds: float) -> str:
    """Use whole seconds, flooring fractional timestamps; hours never wrap."""
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Timestamp must be finite and nonnegative")
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_srt_timestamp(seconds: float) -> str:
    """Render timestamps in SRT HH:MM:SS,mmm format, rounded to milliseconds."""
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Timestamp must be finite and nonnegative")
    milliseconds = int(seconds * 1000 + 0.5)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _validate_segments(result: Transcription) -> None:
    for segment in result.segments:
        if (
            not math.isfinite(segment.start)
            or not math.isfinite(segment.end)
            or segment.start < 0
            or segment.end < segment.start
        ):
            raise ValueError("Invalid segment timestamps: require finite 0 <= start <= end")


def _atomic_write(path: Path, overwrite: bool, write: Callable[[TextIO], None]) -> None:
    """Publish a finished file without clobbering concurrent writers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class TxtExporter:
    suffix = ".txt"

    def __init__(self, *, timestamps: bool = True) -> None:
        self.timestamps = timestamps

    def export(self, result: Transcription, path: Path, *, overwrite: bool = False) -> None:
        def write(stream: TextIO) -> None:
            for segment in result.segments:
                text = segment.text.strip()
                if self.timestamps:
                    text = (
                        f"[{format_timestamp(segment.start)} - "
                        f"{format_timestamp(segment.end)}] {text}"
                    )
                stream.write(text + "\n\n")

        _atomic_write(path, overwrite, write)


class SrtExporter:
    """Standard numbered subtitle cues; empty segments are omitted."""

    suffix = ".srt"

    def export(self, result: Transcription, path: Path, *, overwrite: bool = False) -> None:
        _validate_segments(result)

        def write(stream: TextIO) -> None:
            cue = 0
            for segment in result.segments:
                text = segment.text.strip()
                if not text:
                    continue
                cue += 1
                stream.write(
                    f"{cue}\n{format_srt_timestamp(segment.start)} --> "
                    f"{format_srt_timestamp(segment.end)}\n{text}\n\n"
                )

        _atomic_write(path, overwrite, write)


class JsonExporter:
    """Stable, simple schema preserving original fractional seconds."""

    suffix = ".json"

    def export(self, result: Transcription, path: Path, *, overwrite: bool = False) -> None:
        _validate_segments(result)
        if not math.isfinite(result.language_probability):
            raise ValueError("Language probability must be finite")
        document = {
            "schema_version": 1,
            "language": result.language,
            "language_probability": result.language_probability,
            "segments": [
                {"start": seg.start, "end": seg.end, "text": seg.text} for seg in result.segments
            ],
        }

        def write(stream: TextIO) -> None:
            json.dump(document, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")

        _atomic_write(path, overwrite, write)


def exporters_for(format_name: str, *, timestamps: bool = True) -> tuple[Exporter, ...]:
    """Resolve user-facing format; TXT flags never change subtitle timing."""
    if format_name not in FORMATS:
        raise ValueError("Unknown export format")
    exporters: dict[str, Exporter] = {
        "txt": TxtExporter(timestamps=timestamps),
        "srt": SrtExporter(),
        "json": JsonExporter(),
    }
    names = ("txt", "srt", "json") if format_name == "all" else (format_name,)
    return tuple(exporters[name] for name in names)
