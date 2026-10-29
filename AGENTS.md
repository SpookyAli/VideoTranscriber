# Development conventions

- Use the src layout. `config.py` owns settings, `transcriber.py` owns Faster-Whisper and structured results, `exporters.py` owns output formats, and `cli.py` owns orchestration.
- Keep the engine independent of CLI parsing and export formats. Reuse one model per batch.
- Use type hints, pathlib, explicit UTF-8, and atomic output publication. Preserve no-overwrite behavior.
- Run `python -m ruff check .`, `python -m ruff format --check .`, and `python -m pytest -q` before submitting.
- Update tests whenever behavior changes. Mock Faster-Whisper; tests must never download models or require real media, CUDA, network access, or API credentials.
- Declare runtime and development dependencies in pyproject.toml. Keep Faster-Whisper pinned; validate native dependency compatibility before updating Python or dependencies. Do not add dependencies without a concrete need.
- Never commit videos, generated transcriptions, model weights, virtual environments, caches, or credentials. Use input/, output/, and models/ for local artifacts.
- Keep execution local; do not add cloud APIs, databases, GUI, diarization, or NFLQuant integration without an explicit requirement.
