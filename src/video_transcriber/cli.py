"""Command-line orchestration and per-file failure reporting."""

import argparse
import sys
from pathlib import Path

from video_transcriber.config import COMPUTE_TYPES, MODELS, TranscriptionConfig
from video_transcriber.exporters import Exporter, TxtExporter
from video_transcriber.sources.youtube import YouTubeError, YouTubeSource, select_audio
from video_transcriber.transcriber import Transcriber


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe local WEBM files or YouTube audio to UTF-8 TXT."
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
    command.add_argument("--overwrite", action="store_true")
    command.add_argument("--debug", action="store_true", help="Show safe stage-based diagnostics")
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


def transcribe_youtube(args: argparse.Namespace) -> int:
    def debug(message: str) -> None:
        if args.debug:
            print(f"[DEBUG] {message}")

    target: Path | None = None
    try:
        source = YouTubeSource(
            args.url,
            browser=args.browser,
            audio_language=args.audio_language,
            prefer_original=args.prefer_original_audio,
        )
        args.output.mkdir(parents=True, exist_ok=True)
    except (ValueError, OSError) as exc:
        # URL validation produces a fixed error, but avoid echoing arbitrary filesystem errors.
        if isinstance(exc, ValueError):
            print(f"Error [INVALID_YOUTUBE_URL]: {exc}", file=sys.stderr)
        else:
            print("Error: Could not create the output directory", file=sys.stderr)
        return 2

    debug("Source: YouTube")
    debug(f"Video ID: {source.url.rsplit('=', 1)[1]}")
    try:
        source.inspect()
        debug("Metadata extraction: OK")
        selected = select_audio(
            source.info.get("formats", []), source.audio_language, source.prefer_original
        )
        debug("Audio selection: OK")
        if args.audio_language:
            debug(f"Requested language: {args.audio_language}")
        if selected:
            # Only allowlist safe language tags and format identifiers in debug output.
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
        target = args.output / (source.stem + ".txt")
        if target.exists() and not args.overwrite:
            print(f"Skipped: {target.name} (output exists)")
            return 0
        debug("Download: starting")
        audio = source.download(args.download_dir or Path("input"))
        debug("Download: OK")
        try:
            result = Transcriber(_config(args)).transcribe(audio)
        except Exception:
            raise YouTubeError(
                "Model loading or audio transcription failed; downloaded audio retained",
                code="TRANSCRIPTION_FAILED",
                stage="TRANSCRIPTION",
            ) from None
        debug("Transcription: OK")
        try:
            TxtExporter(timestamps=not args.no_timestamps).export(
                result, target, overwrite=args.overwrite
            )
        except FileExistsError:
            raise
        except Exception:
            raise YouTubeError(
                "TXT export failed; downloaded audio retained",
                code="EXPORT_FAILED",
                stage="TXT_EXPORT",
            ) from None
        debug("TXT export: OK")
    except FileExistsError:
        if not args.overwrite and target is not None and target.is_file():
            print(f"Skipped: {target.name} (output appeared during processing); audio retained")
            return 0
        print("Failed [EXPORT_FAILED]: output path exists; downloaded audio retained", file=sys.stderr)
        return 1
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
    print(f"Transcribed: {audio.name} -> {target} ({result.language})")
    if not args.keep_audio:
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
            engine = Transcriber(_config(args))
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
