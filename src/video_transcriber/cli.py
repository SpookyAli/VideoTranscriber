"""Command-line orchestration and per-file failure reporting."""

import argparse
import sys
from pathlib import Path

from video_transcriber.config import COMPUTE_TYPES, MODELS, TranscriptionConfig
from video_transcriber.exporters import Exporter, TxtExporter
from video_transcriber.transcriber import Transcriber


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe WEBM videos locally to UTF-8 TXT.")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser(
        "transcribe", help="Transcribe a file or directory (nonrecursive)"
    )
    command.add_argument("--input", required=True, type=Path)
    command.add_argument("--output", type=Path, default=Path("output"))
    command.add_argument("--language", help="Whisper language code, e.g. es; default: auto-detect")
    command.add_argument("--model", choices=MODELS, default="medium")
    command.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    command.add_argument("--compute-type", choices=COMPUTE_TYPES, default="int8")
    command.add_argument("--model-dir", type=Path, help="Persistent model cache directory")
    command.add_argument("--local-files-only", action="store_true", help="Never download models")
    command.add_argument("--no-timestamps", action="store_true")
    command.add_argument("--overwrite", action="store_true")
    return parser


def discover_inputs(path: Path) -> list[Path]:
    if not path.exists():
        raise ValueError(f"Input does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() != ".webm":
            raise ValueError(f"Expected a WEBM file: {path}")
        return [path]
    if not path.is_dir():
        raise ValueError(f"Input must be a file or directory: {path}")
    files = sorted(
        (p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".webm"),
        key=lambda p: p.name,
    )
    if not files:
        raise ValueError(f"No WEBM files found in: {path}")
    return files


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        files = discover_inputs(args.input)
        # Case folding prevents collisions on Windows even when running on Linux.
        names = [p.stem.casefold() for p in files]
        if len(names) != len(set(names)):
            raise ValueError("Input filenames map to duplicate output names; rename them first")
        args.output.mkdir(parents=True, exist_ok=True)
        if not args.output.is_dir():
            raise ValueError(f"Output is not a directory: {args.output}")
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    exporter: Exporter = TxtExporter(timestamps=not args.no_timestamps)
    pending: list[tuple[Path, Path]] = []
    skipped = 0
    for source in files:
        target = args.output / (source.stem + exporter.suffix)
        if target.exists() and not args.overwrite:
            print(f"Skipped: {source.name} (output exists)")
            skipped += 1
        else:
            pending.append((source, target))

    succeeded = failed = 0
    if pending:
        try:
            engine = Transcriber(
                TranscriptionConfig(
                    model=args.model,
                    device=args.device,
                    compute_type=args.compute_type,
                    language=args.language,
                    model_dir=args.model_dir,
                    local_files_only=args.local_files_only,
                )
            )
        except Exception as exc:
            print(f"Error loading model: {exc}", file=sys.stderr)
            failed = len(pending)
        else:
            for source, target in pending:
                try:
                    result = engine.transcribe(source)
                    exporter.export(result, target, overwrite=args.overwrite)
                except FileExistsError:
                    if not args.overwrite and target.is_file():
                        skipped += 1
                        print(f"Skipped: {source.name} (output appeared during processing)")
                    else:
                        failed += 1
                        print(f"Failed: {source.name}: output path exists", file=sys.stderr)
                except Exception as exc:
                    failed += 1
                    print(f"Failed: {source.name}: {exc}", file=sys.stderr)
                else:
                    succeeded += 1
                    print(f"Transcribed: {source.name} -> {target} ({result.language})")
    print(f"Summary: {succeeded} succeeded, {skipped} skipped, {failed} failed")
    return 1 if failed else 0
