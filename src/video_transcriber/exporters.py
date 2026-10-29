"""Export structured results without coupling the engine to a file format."""

import math
import os
import tempfile
from pathlib import Path
from typing import Protocol

from video_transcriber.transcriber import Transcription


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


class TxtExporter:
    suffix = ".txt"

    def __init__(self, *, timestamps: bool = True) -> None:
        self.timestamps = timestamps

    def export(self, result: Transcription, path: Path, *, overwrite: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                for segment in result.segments:
                    text = segment.text.strip()
                    if self.timestamps:
                        text = (
                            f"[{format_timestamp(segment.start)} - "
                            f"{format_timestamp(segment.end)}] {text}"
                        )
                    stream.write(text + "\n\n")
                stream.flush()
                os.fsync(stream.fileno())
            if overwrite:
                os.replace(temporary, path)
            else:
                # An atomic no-clobber publication, including concurrent writers.
                os.link(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
