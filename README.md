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
