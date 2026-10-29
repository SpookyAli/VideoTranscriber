"""Model ownership and structured transcription; no output formatting."""

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

    def transcribe(self, path: Path) -> Transcription:
        segments, info = self.model.transcribe(
            str(path), language=self.config.language, vad_filter=True
        )
        # Faster-Whisper executes lazily: consume inside the caller's error boundary.
        return Transcription(
            tuple(Segment(s.start, s.end, s.text.strip()) for s in segments),
            info.language,
            info.language_probability,
        )
