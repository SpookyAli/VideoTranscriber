"""Command-line orchestration and per-file failure reporting."""

import argparse
import sys
from pathlib import Path

from video_transcriber.config import COMPUTE_TYPES, MODELS, TranscriptionConfig
from video_transcriber.exporters import FORMATS, Exporter, exporters_for
from video_transcriber.exporters import TxtExporter as TxtExporter
from video_transcriber.progress import ProgressReporter
from video_transcriber.sources.youtube import YouTubeError, YouTubeSource, select_audio
from video_transcriber.transcriber import Transcriber


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe local WEBM files or YouTube audio to TXT, SRT or JSON."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser(
        "transcribe", help="Transcribe a local file/directory (nonrecursive) or a YouTube video"
    )
    source = command.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--url", help="Single YouTube video URL")
    command.add_argument("--browser", choices=("firefox", "chrome", "edge"))
    command.add_argument("--audio-language", help="YouTube audio track language, e.g. es")
    command.add_argument("--prefer-original-audio", action="store_true")
    command.add_argument("--keep-audio", action="store_true")
    command.add_argument("--download-dir", type=Path, default=None, help="Default: input/")
    command.add_argument("--output", type=Path, default=Path("output"))
    command.add_argument("--language", help="Whisper language code, e.g. es; default: auto-detect")
    command.add_argument("--model", choices=MODELS, default="medium")
    command.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    command.add_argument("--compute-type", choices=COMPUTE_TYPES, default="int8")
    command.add_argument("--model-dir", type=Path, help="Persistent model cache directory")
    command.add_argument("--local-files-only", action="store_true", help="Never download models")
    command.add_argument("--no-timestamps", action="store_true")
    command.add_argument("--format", choices=FORMATS, default="txt", help="Default: txt")
    command.add_argument("--overwrite", action="store_true")
    command.add_argument("--debug", action="store_true", help="Show safe stage-based diagnostics")
    command.add_argument(
        "--progress", action="store_true", help="Show stage, download and transcription progress"
    )
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


def _config(args: argparse.Namespace) -> TranscriptionConfig:
    return TranscriptionConfig(
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        model_dir=args.model_dir,
        local_files_only=args.local_files_only,
    )


def _pending_exports(
    stem: str, output: Path, exporters: tuple[Exporter, ...], overwrite: bool
) -> list[tuple[Exporter, Path]]:
    """Select missing outputs, retaining any existing formats by default."""
    return [
        (exporter, output / (stem + exporter.suffix))
        for exporter in exporters
        if overwrite or not (output / (stem + exporter.suffix)).exists()
    ]


def transcribe_youtube(args: argparse.Namespace) -> int:
    reporter = ProgressReporter(args.progress)
    exporters = exporters_for(args.format, timestamps=not args.no_timestamps)

    def debug(message: str) -> None:
        if args.debug:
            print(f"[DEBUG] {message}")

    pending: list[tuple[Exporter, Path]] = []
    completed: list[Path] = []
    raced_output = False
    audio: Path | None = None
    try:
        source = YouTubeSource(
            args.url,
            browser=args.browser,
            audio_language=args.audio_language,
            prefer_original=args.prefer_original_audio,
        )
        args.output.mkdir(parents=True, exist_ok=True)
    except (ValueError, OSError) as exc:
        if isinstance(exc, ValueError):
            print(f"Error [INVALID_YOUTUBE_URL]: {exc}", file=sys.stderr)
        else:
            print("Error: Could not create the output directory", file=sys.stderr)
        return 2

    debug("Source: YouTube")
    debug(f"Video ID: {source.url.rsplit('=', 1)[1]}")
    try:
        reporter.start("metadata")
        source.inspect()
        reporter.finish("metadata")
        debug("Metadata extraction: OK")
        selected = select_audio(
            source.info.get("formats", []), source.audio_language, source.prefer_original
        )
        debug("Audio selection: OK")
        if args.audio_language:
            debug(f"Requested language: {args.audio_language}")
        if selected:
            import re

            language = str(selected.get("language") or "")
            fmt_id = str(selected.get("format_id") or "")
            ext = str(selected.get("ext") or "")
            if re.fullmatch(r"[A-Za-z0-9-]{1,35}", language):
                debug(f"Selected language: {language}")
            if re.fullmatch(r"[A-Za-z0-9_-]{1,35}", fmt_id):
                debug(f"Selected format: {fmt_id}")
            if re.fullmatch(r"[a-z0-9]{1,12}", ext):
                debug(f"Audio container: {ext}")

        pending = _pending_exports(source.stem, args.output, exporters, args.overwrite)
        if not pending:
            print(f"Skipped: {source.stem} (all requested outputs exist)")
            return 0

        debug("Download: starting")
        reporter.start("download")
        if args.progress:
            audio = source.download(
                args.download_dir or Path("input"), progress_hook=reporter.download_hook
            )
        else:
            audio = source.download(args.download_dir or Path("input"))
        reporter.finish("download")
        debug("Download: OK")

        try:
            reporter.start("model")
            engine = Transcriber(_config(args))
            reporter.finish("model")
            reporter.start("transcription")
            if args.progress:
                result = engine.transcribe(audio, progress=reporter.segment_hook)
            else:
                result = engine.transcribe(audio)
            reporter.finish("transcription")
        except Exception:
            raise YouTubeError(
                "Model loading or audio transcription failed; downloaded audio retained",
                code="TRANSCRIPTION_FAILED",
                stage="TRANSCRIPTION",
            ) from None

        debug("Transcription: OK")
        for exporter, target in pending:
            try:
                reporter.start("export")
                exporter.export(result, target, overwrite=args.overwrite)
                reporter.finish("export")
                completed.append(target)
            except FileExistsError:
                if not args.overwrite and target.is_file():
                    print(f"Skipped: {target.name} (output appeared during processing)")
                    raced_output = True
                    continue
                raise YouTubeError(
                    "Export target already exists; downloaded audio retained",
                    code="EXPORT_FAILED",
                    stage="TXT_EXPORT",
                ) from None
            except Exception:
                raise YouTubeError(
                    "Export failed; downloaded audio retained",
                    code="EXPORT_FAILED",
                    stage="TXT_EXPORT",
                ) from None
        debug("TXT export: OK" if args.format == "txt" else "Export: OK")
    except YouTubeError as exc:
        debug(f"{exc.stage}: FAILED")
        if exc.http_status is not None:
            debug(f"HTTP status: {exc.http_status}")
        print(f"Failed [{exc.code}] ({exc.stage}): {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted: any downloaded/partial audio has been retained", file=sys.stderr)
        return 130
    except Exception:
        print(
            "Failed: unexpected processing failure; downloaded audio retained",
            file=sys.stderr,
        )
        return 1

    for target in completed:
        print(f"Transcribed: {audio.name} -> {target} ({result.language})")
    reporter.summary()
    if not args.keep_audio and not raced_output:
        try:
            source.cleanup()
        except OSError:
            print(
                "Warning: TXT saved, but audio cleanup failed; remove audio manually",
                file=sys.stderr,
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.url:
        return transcribe_youtube(args)
    if (
        args.browser
        or args.audio_language is not None
        or args.prefer_original_audio
        or args.keep_audio
        or args.download_dir is not None
    ):
        parser.error("YouTube options require --url")
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

    exporters = exporters_for(args.format, timestamps=not args.no_timestamps)
    reporter = ProgressReporter(args.progress)
    pending: list[tuple[Path, list[tuple[Exporter, Path]]]] = []
    skipped = 0
    for source in files:
        outputs = _pending_exports(source.stem, args.output, exporters, args.overwrite)
        if not outputs:
            print(f"Skipped: {source.name} (all requested outputs exist)")
            skipped += 1
        else:
            pending.append((source, outputs))

    succeeded = failed = 0
    if pending:
        try:
            reporter.start("model")
            engine = Transcriber(_config(args))
            reporter.finish("model")
        except Exception as exc:
            print(f"Error loading model: {exc}", file=sys.stderr)
            failed = len(pending)
        else:
            for source, outputs in pending:
                try:
                    reporter.start("transcription")
                    if args.progress:
                        result = engine.transcribe(source, progress=reporter.segment_hook)
                    else:
                        result = engine.transcribe(source)
                    reporter.finish("transcription")
                except Exception as exc:
                    failed += 1
                    print(f"Failed: {source.name}: {exc}", file=sys.stderr)
                    continue
                export_failed = False
                for exporter, target in outputs:
                    try:
                        reporter.start("export")
                        exporter.export(result, target, overwrite=args.overwrite)
                        reporter.finish("export")
                    except FileExistsError:
                        if not args.overwrite and target.is_file():
                            print(f"Skipped: {target.name} (output appeared during processing)")
                            continue
                        export_failed = True
                        print(f"Failed: {target.name}: output path exists", file=sys.stderr)
                    except Exception as exc:
                        export_failed = True
                        print(f"Failed: {target.name}: {exc}", file=sys.stderr)
                    else:
                        print(f"Transcribed: {source.name} -> {target} ({result.language})")
                if export_failed:
                    failed += 1
                else:
                    succeeded += 1

    print(f"Summary: {succeeded} succeeded, {skipped} skipped, {failed} failed")
    reporter.summary()
    return 1 if failed else 0
