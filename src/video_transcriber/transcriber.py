"""Model ownership and structured transcription; no output formatting."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from video_transcriber.config import TranscriptionConfig


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcription:
    segments: tuple[Segment, ...]
    language: str
    language_probability: float


class Transcriber:
    """Create once and reuse for every file in a batch."""

    def __init__(self, config: TranscriptionConfig) -> None:
        # Lazy import keeps help and argument validation independent of native libraries.
        from faster_whisper import WhisperModel

        self.config = config
        self.model = WhisperModel(
            config.model,
            device=config.device,
            compute_type=config.compute_type,
            download_root=str(config.model_dir) if config.model_dir else None,
            local_files_only=config.local_files_only,
        )

    def transcribe(
        self, path: Path, *, progress: Callable[[float, float | None], None] | None = None
    ) -> Transcription:
        options = {"language": self.config.language, "vad_filter": True}
        if self.config.initial_prompt:
            options["initial_prompt"] = self.config.initial_prompt
        if self.config.beam_size is not None:
            options["beam_size"] = self.config.beam_size
        if self.config.temperature is not None:
            options["temperature"] = self.config.temperature
        segments, info = self.model.transcribe(str(path), **options)
        # Faster-Whisper executes lazily: consume inside the caller's error boundary.
        # Segment timestamps are the only available progress signal without probing media.
        duration = getattr(info, "duration", None)
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            duration = None
        collected = []
        for segment in segments:
            collected.append(Segment(segment.start, segment.end, segment.text.strip()))
            if progress is not None:
                progress(segment.end, duration)
        return Transcription(tuple(collected), info.language, info.language_probability)
