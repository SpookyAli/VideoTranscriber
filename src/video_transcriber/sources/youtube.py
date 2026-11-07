"""Local yt-dlp integration; never expose extractor messages or browser secrets."""

import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


class YouTubeError(RuntimeError):
    """A credential-free, structured YouTube source failure."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "DOWNLOAD_FAILED",
        stage: str = "DOWNLOAD",
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.http_status = http_status


def validate_url(url: str) -> str:
    """Accept single-video HTTPS/HTTP URLs and discard unrelated query parameters."""
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"https", "http"} or parsed.username or parsed.password:
            raise ValueError
        if parsed.port is not None:
            raise ValueError
        host = (parsed.hostname or "").lower()
        if host == "youtu.be":
            video_id = parsed.path.removeprefix("/")
        elif host in {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}:
            if parsed.path == "/watch":
                video_id = parse_qs(parsed.query).get("v", [""])[0]
            else:
                match = re.fullmatch(r"/(?:shorts|embed|live)/([\w-]{11})/?", parsed.path)
                video_id = match[1] if match else ""
        else:
            raise ValueError
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            raise ValueError
    except ValueError:
        raise ValueError(
            "Invalid YouTube URL: use a single youtube.com or youtu.be video URL"
        ) from None
    return f"https://www.youtube.com/watch?v={video_id}"


def _language(value: str | None) -> str:
    return (value or "").strip().replace("_", "-").lower()


def _original(fmt: dict) -> bool:
    return "original" in str(fmt.get("format_note", "")).lower()


def select_audio(
    formats: list[dict], language: str | None = None, prefer_original: bool = False
) -> dict | None:
    """Return a preferred audio-only track; None delegates to bestaudio."""
    audio = [
        f
        for f in formats
        if f.get("vcodec") == "none"
        and f.get("acodec") not in {None, "none"}
        and not f.get("has_drm")
    ]
    if not audio:
        raise YouTubeError(
            "No permitted audio-only formats are available for this video",
            code="AUDIO_FORMAT_NOT_FOUND",
            stage="AUDIO_SELECTION",
        )
    wanted = _language(language)
    if wanted:
        matching = [
            f for f in audio if _language(f.get("language")).split("-")[0] == wanted.split("-")[0]
        ]
        if not matching:
            available = sorted({_language(f.get("language")) or "unknown" for f in audio})
            # Metadata is untrusted: only display language-like tags.
            available = [x for x in available if re.fullmatch(r"[a-z0-9-]{1,35}", x)]
            raise YouTubeError(
                "Requested audio language is unavailable. "
                f"Available languages: {', '.join(available)}",
                code="AUDIO_LANGUAGE_UNAVAILABLE",
                stage="AUDIO_SELECTION",
            )
        audio = matching
    elif not any(_original(f) for f in audio):
        return None

    def rank(fmt: dict) -> tuple:
        note = str(fmt.get("format_note", "")).lower()
        return (
            _original(fmt),
            "drc" not in note and not fmt.get("is_drc", False),
            _language(fmt.get("language")) == wanted if wanted else False,
            fmt.get("abr") or fmt.get("tbr") or 0,
            fmt.get("asr") or 0,
            fmt.get("filesize") or fmt.get("filesize_approx") or 0,
        )

    return max(audio, key=rank)


class _QuietLogger:
    def debug(self, message: str) -> None:
        pass

    info = warning = error = debug


def _http_status(exc: Exception) -> int | None:
    """Read HTTP status from known wrappers before a narrow yt-dlp text fallback."""
    from urllib.error import HTTPError as UrllibHTTPError

    from yt_dlp.utils import HTTPError as YtdlpHTTPError

    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, (YtdlpHTTPError, UrllibHTTPError)):
            status = getattr(current, "status", None) or getattr(current, "code", None)
            if isinstance(status, int) and 100 <= status <= 599:
                return status
        current = current.__cause__ or current.__context__
    # DownloadError commonly flattens an HTTPError into its message.
    # Read only a numeric status, never present upstream text to the caller.
    match = re.search(r"\\bHTTP Error ([1-5][0-9]{2})\\b", str(exc), re.I)
    return int(match[1]) if match else None


def _failure(exc: Exception, deno: bool, stage: str) -> YouTubeError:
    """Map external errors to allowlisted messages without exposing upstream text."""
    status = _http_status(exc)
    if status is not None:
        if stage == "DOWNLOAD":
            if status in {403, 404, 429}:
                code = f"DOWNLOAD_HTTP_{status}"
            elif status >= 500:
                code = "DOWNLOAD_HTTP_5XX"
            else:
                code = "DOWNLOAD_FAILED"
            descriptions = {
                403: (
                    "YouTube rejected the media download request (HTTP 403). "
                    "Verify download permissions and yt-dlp compatibility."
                ),
                404: "The requested media resource was not found (HTTP 404).",
                429: "The remote server rejected the request due to rate limiting (HTTP 429).",
            }
            detail = descriptions.get(
                status, "The remote server rejected the media download request."
            )
        else:
            code = "METADATA_EXTRACTION_FAILED"
            detail = f"YouTube metadata request failed (HTTP {status})."
        return YouTubeError(detail, code=code, stage=stage, http_status=status)

    message = str(exc).lower()
    if any(word in message for word in ("javascript", "challenge", "js runtime", "signature")):
        code = "JS_CHALLENGE_FAILED" if deno else "JS_RUNTIME_UNAVAILABLE"
        detail = (
            "JavaScript challenge failed; check Deno and yt-dlp-ejs"
            if deno
            else "JavaScript runtime unavailable; Deno was not found on PATH"
        )
    elif any(word in message for word in ("cookie", "sign in", "login", "members-only", "private")):
        code = "AUTHENTICATION_FAILED"
        detail = "YouTube access/authentication failed; verify legitimate access and browser login"
    elif any(word in message for word in ("unavailable", "removed", "not available")):
        code = "VIDEO_UNAVAILABLE"
        detail = "Video unavailable; check its URL and availability in your account or region"
    else:
        code = "METADATA_EXTRACTION_FAILED" if stage == "METADATA_EXTRACTION" else "DOWNLOAD_FAILED"
        detail = (
            "YouTube metadata extraction failed; check connectivity and yt-dlp compatibility"
            if stage == "METADATA_EXTRACTION"
            else (
                "YouTube media download failed; check connectivity, download permissions, "
                "and yt-dlp compatibility"
            )
        )
    return YouTubeError(detail, code=code, stage=stage)


class YouTubeSource:
    """One video with explicit metadata, download, and owned-artifact lifecycle."""

    def __init__(
        self,
        url: str,
        *,
        browser: str | None = None,
        audio_language: str | None = None,
        prefer_original: bool = False,
    ) -> None:
        self.url = validate_url(url)
        if browser not in {None, "firefox", "chrome", "edge"}:
            raise ValueError("Unsupported browser; choose firefox, chrome, or edge")
        if audio_language is not None and not re.fullmatch(
            r"[a-z]{2,3}(?:-[a-z0-9]{2,8})*", _language(audio_language)
        ):
            raise ValueError("Invalid audio language: use a language tag such as es or es-US")
        self.audio_language = audio_language
        self.prefer_original = prefer_original
        self.deno = shutil.which("deno") is not None
        self.options = {
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "logger": _QuietLogger(),
            "noplaylist": True,
            "windowsfilenames": True,
            "overwrites": False,
            "format": "bestaudio",
            "js_runtimes": {"deno": {}},
        }
        if browser:
            self.options["cookiesfrombrowser"] = (browser,)
        self.info: dict = {}
        self.stem = ""
        self.directory: Path | None = None

    def inspect(self) -> None:
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import sanitize_filename

        try:
            with YoutubeDL(self.options) as ydl:
                info = ydl.extract_info(self.url, download=False)
            if not info or info.get("_type", "video") != "video":
                raise YouTubeError(
                    "Expected a single YouTube video, not a playlist",
                    code="METADATA_EXTRACTION_FAILED",
                    stage="METADATA_EXTRACTION",
                )
            video_id = self.url.rsplit("=", 1)[1]
            if info.get("id") != video_id:
                raise YouTubeError(
                    "YouTube returned an unexpected video identity",
                    code="METADATA_EXTRACTION_FAILED",
                    stage="METADATA_EXTRACTION",
                )
            select_audio(info.get("formats", []), self.audio_language, self.prefer_original)
            title = sanitize_filename(str(info.get("title") or video_id), restricted=False)
            # The appended video ID also prevents bare Windows reserved device names.
            title = title.strip(" .") or "video"
            title = title[:140].encode("utf-8")[:180].decode("utf-8", errors="ignore").rstrip(" .")
            self.stem = f"{title} [{video_id}]"
            self.info = info
        except YouTubeError:
            raise
        except Exception as exc:
            raise _failure(exc, self.deno, "METADATA_EXTRACTION") from None

    def download(self, directory: Path) -> Path:
        from yt_dlp import YoutubeDL

        if not self.info:
            raise YouTubeError(
                "Inspect video metadata before downloading",
                code="METADATA_EXTRACTION_FAILED",
                stage="METADATA_EXTRACTION",
            )
        try:
            directory.mkdir(parents=True, exist_ok=True)
            self.directory = Path(tempfile.mkdtemp(prefix="youtube-", dir=directory)).resolve()
            selected = select_audio(
                self.info.get("formats", []), self.audio_language, self.prefer_original
            )
            options = dict(self.options)
            options["format"] = (lambda ctx: iter([selected])) if selected else "bestaudio"
            # Escape metadata percent signs in yt-dlp's output template.
            options["outtmpl"] = str(self.directory / self.stem).replace("%", "%%") + ".%(ext)s"
            # extract_info(download=False) still adds a selection. Do not carry a
            # prior video/audio merge into the audio-only download's process_info.
            info = dict(self.info)
            for key in ("requested_formats", "requested_downloads", "filepath", "_filename"):
                info.pop(key, None)
            with YoutubeDL(options) as ydl:
                result = ydl.process_ie_result(info, download=True)
                downloads = result.get("requested_downloads") or [result]
                path = Path(downloads[0].get("filepath") or ydl.prepare_filename(result)).resolve()
            if path.parent != self.directory or not path.is_file() or path.is_symlink():
                raise YouTubeError(
                    "Downloader did not return a valid owned audio file",
                    code="DOWNLOADED_FILE_INVALID",
                    stage="DOWNLOADED_FILE_VALIDATION",
                )
            return path
        except YouTubeError:
            raise
        except Exception as exc:
            raise _failure(exc, self.deno, "DOWNLOAD") from None

    def cleanup(self) -> None:
        """Only call after successful export, and only remove our private directory."""
        if self.directory is not None:
            shutil.rmtree(self.directory)
