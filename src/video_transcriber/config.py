"""Validated transcription settings, independent of the CLI."""

from dataclasses import dataclass
from pathlib import Path

MODELS = ("tiny", "base", "small", "medium", "large-v3")
COMPUTE_TYPES = ("default", "auto", "int8", "int8_float32", "int8_float16", "float16", "float32")


@dataclass(frozen=True)
class TranscriptionConfig:
    model: str = "medium"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str | None = None
    model_dir: Path | None = None
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"Unsupported model: {self.model}")
        if self.device not in ("cpu", "cuda", "auto"):
            raise ValueError(f"Unsupported device: {self.device}")
        if self.compute_type not in COMPUTE_TYPES:
            raise ValueError(f"Unsupported compute type: {self.compute_type}")
