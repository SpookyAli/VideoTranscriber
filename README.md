# VideoTranscriber

A local Python CLI that transcribes WEBM audio/video to UTF-8 TXT with
Faster-Whisper 1.2.1. Supports automatic language detection, explicit languages,
VAD, timestamped segments, and sequential batch processing with one model load.
No cloud API or account is required. Model downloads require Internet access on
first use; inference can run completely offline afterward.

## Windows 11 and PyCharm installation

Use 64-bit CPython. Python 3.12 is the conservative baseline; Python 3.14 requires
compatible wheels for native dependencies (CTranslate2, PyAV, ONNX Runtime, and
tokenizers). If pip reports no matching distribution on 3.14, select Python 3.12
in PyCharm and recreate `.venv`. Do not disable package verification.

Open this repository in PyCharm, open its PowerShell terminal at the repository
root, and run (replace `3.14` with `3.12` when using the baseline):

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

In **Settings > Project > Python Interpreter**, select the existing interpreter
`.venv\Scripts\python.exe`. Use the repository root as the run working directory.
Activation is optional; using the explicit interpreter also avoids PowerShell
execution policy changes. PyAV includes FFmpeg libraries in its wheels; a separate
FFmpeg executable is normally unnecessary.

Faster-Whisper is pinned to `1.2.1`, with PyAV constrained to `av>=11,<18`.
Newer PyAV releases can remove decoder API arguments used by Faster-Whisper
(such as `metadata_errors`), causing transcription to fail. Install through this
project rather than upgrading PyAV independently. To repair an existing `.venv`,
rerun `.\.venv\Scripts\python.exe -m pip install -e ".[dev]"`; pip will apply the
compatible constraint.

## First transcription

Place a video at `input/video.webm`, then run:

```powershell
.\.venv\Scripts\python.exe main.py transcribe --input input/video.webm --model-dir models
```

The default model is `medium`, the device is CPU, and the compute type is `int8`.
The first invocation downloads model weights from Hugging Face. To test a smaller
model first, pass `--model tiny`. For offline use, reuse the same model/cache:

```powershell
.\.venv\Scripts\python.exe main.py transcribe --input input/video.webm --model-dir models --local-files-only
```

The same commands work as `python main.py ...` when `.venv` is activated. Installing
the package also exposes `video-transcriber transcribe ...`.

## Usage

```powershell
python main.py transcribe --input input/
python main.py transcribe --input input/ --language es
python main.py transcribe --input input/ --model large-v3
python main.py transcribe --input input/ --no-timestamps
python main.py transcribe --input input/ --output output/custom --overwrite
python main.py transcribe --input input/ --device cuda --compute-type float16
python main.py transcribe --help
```

Directories are scanned nonrecursively for `.webm` files (case insensitive).
Other file types in a batch are ignored. Each video produces `output/<stem>.txt`;
existing outputs are skipped unless `--overwrite` is set. Duplicate output names
are rejected before model loading. Outputs use atomic publication and UTF-8;
the output filesystem must support hard links for atomic no-overwrite publication
(standard local NTFS and Linux filesystems do). Empty transcriptions create empty
TXT files. Batch failures are reported individually and remaining files continue.
Exit codes: `0` success (including skips), `1` processing/model/export failure,
`2` invalid arguments or input/output configuration.

Default output:

```text
[00:00:01 - 00:00:08] Welcome to today's analysis.

[00:00:09 - 00:00:15] Let's examine the statistics.
```

Timestamps floor fractional seconds for display; structured engine results retain
original fractional start/end values and detected-language metadata.

## Models and hardware

| Model | Typical tradeoff |
| --- | --- |
| tiny | Fastest, lowest resource use, lower accuracy |
| base | Fast, modest accuracy |
| small | Balance for limited hardware |
| medium | Default; higher accuracy and greater memory/CPU requirements |
| large-v3 | Highest resource requirements; often better accuracy |

CPU `int8` is the portable default. Medium/large models can be slow on CPU and
require substantial RAM and disk space. CUDA requires a supported NVIDIA GPU,
sufficient VRAM, and CUDA/cuDNN versions compatible with the installed CTranslate2.
Consult Faster-Whisper/CTranslate2 installation documentation for GPU libraries;
this project does not install GPU drivers. Choose a compute type supported by your
device (`--compute-type`); unsupported combinations produce a model-loading error.
Language defaults to automatic detection; use Whisper language codes such as `es`
or `en` to override. Recognition quality and timing depend on the recording.

## Development

```powershell
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

CI runs these commands on Windows and Linux with Python 3.12 and 3.14. Unit tests
mock model inference and do not download models or require real WEBM files.
A decoder regression test generates a small WAV fixture locally and exercises
real Faster-Whisper/PyAV decoding and resampling without network access.
An actual media transcription is a separate manual integration check.

`src/video_transcriber` separates settings, model/segment handling, exporters, and
CLI orchestration. Exporters consume structured results through a small protocol,
so JSON/SRT can be added without changing the engine. `main.py` is the repository
entry point. `input/`, `output/`, and `models/` hold ignored local artifacts; never
commit recordings, generated text, credentials, or model weights.

## YouTube sources (V1.1)

YouTube support runs locally through pinned `yt-dlp==2026.10.26` and
`yt-dlp-ejs==0.8.0`. Installing `.[dev]` installs both; Faster-Whisper remains
1.2.1 and the PyAV constraint remains `av>=11,<18`. CPython 3.12 is the conservative
baseline; the same commands support Python 3.14 when native wheels are available.
No separate FFmpeg executable or audio transcoding is needed for WEBM/M4A audio.

### Windows setup

From the repository root in the PyCharm PowerShell terminal:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
winget install --id DenoLand.Deno --exact
```

Install Deno separately; this application never installs system software. The
supported existing setup uses Deno 2.9.8. If already installed, keep that version;
check `deno --version` and reopen PyCharm/its terminal after changing PATH. For a
manual installation, use the official [Deno installation guide](https://docs.deno.com/runtime/getting_started/installation/).
Deno must be on PATH in the process running Python. yt-dlp is explicitly configured
to use Deno and its installed EJS challenge scripts. TLS verification stays enabled.
On Linux use the same Python virtual environment workflow and install Deno separately.

### First YouTube transcription

For the first local validation, sign in to YouTube in **Firefox** with an account
that can legitimately access the video, close Firefox if its cookie database is
locked, and run:

```powershell
.\.venv\Scripts\python.exe main.py transcribe --url "https://youtu.be/Jqe34VMhFXI" --browser firefox --audio-language es --language es --model medium --model-dir models
```

Firefox is recommended on Windows. Chrome and Edge are also supported through
`--browser chrome` and `--browser edge`, but OS encryption/profile locks can prevent
cookie extraction. Authentication is optional for public videos:

```powershell
.\.venv\Scripts\python.exe main.py transcribe --url "https://www.youtube.com/watch?v=Jqe34VMhFXI" --audio-language es --language es --model medium --model-dir models
```

`--input` and `--url` are mutually exclusive. Browser, audio-track, and download
options require `--url`. Watch, short-link, Shorts, embed, and live-video URLs are
accepted; playlist-only URLs are rejected, and playlist parameters on a video URL
are ignored. Only a single video's audio is downloaded.

`--audio-language es` selects the **YouTube audio track**. `--language es` tells
**Whisper which language to recognize**; neither option changes the other. Language
matching normalizes case and underscores and accepts regional variants (`es`,
`es-US`, `es-ES`). Matching original audio outranks dubbed audio, then non-DRC audio
outranks DRC, with exact regional tags and bitrate used to rank remaining choices.
A requested language that is unavailable produces an error listing available
languages; the application never silently selects another language. Without an
audio-language request, identifiable original audio is preferred; otherwise yt-dlp
selects its normal best audio. `--prefer-original-audio` makes that preference
explicit (it already applies by default). Selection never uses fixed format IDs.

### Download lifecycle and output

Downloads go into unique `input/youtube-*/` subdirectories by default. Override the
parent using `--download-dir "D:\Audio"`. The filename uses a sanitized, length-limited
title and YouTube ID, such as `Title [Jqe34VMhFXI].webm`; the actual container can
also be M4A. No full video stream or unnecessary transcoding is used.

TXT files use `output/<sanitized-title> [<id>].txt` and preserve UTF-8, timestamps,
and atomic publication. `--output` changes the output directory. An existing TXT
is skipped after metadata lookup and before audio download, unless `--overwrite`
is supplied. `--overwrite` applies to the transcript; each download has a new
private directory and never overwrites an existing recording.

After successful TXT export the operation's download directory is removed. Keep
its audio with:

```powershell
.\.venv\Scripts\python.exe main.py transcribe --url "https://youtu.be/Jqe34VMhFXI" --browser firefox --audio-language es --language es --keep-audio
```

Model, decoder, and export failures preserve downloaded audio; interrupted downloads
may leave `.part` files in that operation's directory. Cleanup failure emits a
warning and leaves the completed TXT intact. Local user inputs and previous
operations' downloads are never removed. The local `--input` workflow still scans
WEBM files only; retained M4A files are supported by the engine's PyAV decoder but
are not accepted by the local-file CLI in this release.

### Troubleshooting and security

- **Missing Deno / JavaScript challenge:** verify `deno --version`, PATH, and the
  pinned yt-dlp/EJS packages with `.\.venv\Scripts\python.exe -m pip check`.
  Reopen the terminal after installation. YouTube changes can require a future
  coordinated dependency update; do not disable TLS or bypass challenges/access rules.
- **Authentication / membership:** sign in locally with an entitled account and
  choose the correct browser. Close the browser if its profile is locked. This
  version uses the browser's default profile and does not automatically log in.
- **Unavailable language / no audio formats:** check the displayed language list
  or the video's availability. There is no fallback to another requested language,
  video streams, DRM content, or unauthorized access methods.
- **Network / download failure:** check connectivity, video availability, access,
  and permission to download. Unknown errors are not labeled authentication errors.
  Upstream messages are suppressed to avoid leaking cookies, headers, and signed URLs.
- **Transcription / TXT failure:** check the model cache (especially with
  `--local-files-only`), audio validity, disk space, and output permissions. Audio
  remains in the operation's download directory. First model use requires a download;
  cached models support offline inference.

Browser cookies are used only by local yt-dlp and the requested service. They are
never intentionally exported, written to a persistent cookie file, printed, or sent
to telemetry. Do not commit browser profiles, cookies, media, or generated output.
Access permission and downloading permission are separate: account access does not
itself grant permission to save media. Use this feature only where both permit it.
No authentication bypass or DRM circumvention is implemented.

All tests run offline with mocked yt-dlp/browser authentication. They include a
real yt-dlp API-contract test with a local fake download and the existing real
Faster-Whisper/PyAV decoding regression. CI does not need Deno, browser profiles,
YouTube access, model weights, or credentials. Live YouTube authentication/challenge
solving and transcription remain separate manual checks on the user's machine.

## YouTube diagnostics (V1.1.1)

Use `--debug` with `--url` to show safe diagnostic stage transitions,
selected audio language/format when available, and classified failure codes:

```powershell
python main.py transcribe --url "https://www.youtube.com/watch?v=IfWWpDrF8hk" --browser firefox --audio-language es --language es --model medium --model-dir models --debug
```

The debugger prints only allowlisted data: video ID, recognized language tags,
format identifier, container, stage and HTTP status. It does not print raw upstream
logs, authentication cookies, media URLs, request headers, or exception tracebacks.

Key error codes include `METADATA_EXTRACTION_FAILED`,
`AUDIO_LANGUAGE_UNAVAILABLE`, `AUDIO_FORMAT_NOT_FOUND`,
`DOWNLOAD_HTTP_403`, `DOWNLOAD_HTTP_404`, `DOWNLOAD_HTTP_429`,
`DOWNLOAD_HTTP_5XX`, `DOWNLOAD_FAILED`, `AUTHENTICATION_FAILED`,
`JS_RUNTIME_UNAVAILABLE`, `JS_CHALLENGE_FAILED`,
`TRANSCRIPTION_FAILED`, and `EXPORT_FAILED`.

An HTTP 403 at the download stage means that the media server rejected the
download request. Metadata being accessible, or a video being playable in a
browser, does not establish that a separate download is allowed; HTTP 403 does
not identify a single underlying cause or prove a missing token. The patch does
not implement new download strategies, bypass access restrictions, or resolve
server-side HTTP 403. HTTP 404 indicates the requested resource was not found,
HTTP 429 is rate limiting, and HTTP 5xx is a server-side response. If metadata
fails, check URL, connection, service availability and authentication. If Deno
challenge solving fails, inspect its installation and the yt-dlp/EJS dependency
versions. Audio-track mismatch is reported without silently choosing a different
language. Existing downloaded audio is retained after transcription/export failure.

## Progress and performance monitoring (V1.2)

Add `--progress` to `transcribe` for opt-in stage timing, bounded percentage
updates when duration/byte totals are available, and an end-of-run timing summary:

```powershell
python main.py transcribe --input input/video.webm --language es --model medium --model-dir models --progress
python main.py transcribe --url "https://www.youtube.com/watch?v=VIDEO_ID" --browser firefox --audio-language es --language es --model medium --model-dir models --progress
```

The default CLI remains concise; `--debug` is a separate diagnostic option and
may be combined with `--progress`. For local batches the model is loaded once
and timing covers each transcription/export stage. For YouTube, progress starts
at metadata extraction and reports media download progress when yt-dlp supplies
downloaded byte totals. Unknown totals never produce fabricated percentages.

Transcription progress is based on the end timestamps of *completed* Whisper
segments divided by Faster-Whisper's reported source duration. If the duration
is unknown or no segment has completed, only stage timing is reported. VAD,
silence, and variable inference speed mean these percentages are not wall-clock
completion predictions. Stage timing uses a monotonic clock. The transcription
real-time factor (RTF) is measured transcription wall time divided by audio
duration; lower RTF means faster than real time. All counters and updates are
bounded, and no source filenames, URLs, headers, cookies or signed URLs are
emitted by the progress reporter.

This version does not change authorization, downloading or access restrictions;
HTTP 403 download failures continue to be reported by V1.1.1 diagnostics.
