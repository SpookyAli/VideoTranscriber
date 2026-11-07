"""Local, bounded vocabulary hints for Faster-Whisper; no prompt content is logged."""

from pathlib import Path

PROFILES: dict[str, tuple[str, ...]] = {
    "nfl": (
        "NFL", "Patrick Mahomes", "Josh Allen", "Lamar Jackson", "Chiefs", "Seahawks",
        "Ravens", "Bears", "Patriots", "touchdown", "red zone", "moneyline",
        "spread", "over/under", "quarterback", "yards", "interception",
    ),
    "gaming": (
        "gameplay", "speedrun", "frame rate", "FPS", "NPC", "RPG",
        "Nintendo", "PlayStation", "Xbox", "Steam", "patch notes",
    ),
    "technology": (
        "Python", "GitHub", "Power BI", "Databricks", "SQL", "API",
        "machine learning", "Docker", "JSON", "JavaScript", "Kubernetes",
    ),
}
MAX_FILE_BYTES = 16_384
MAX_TERMS = 64
MAX_TERM_CHARS = 100
MAX_PROMPT_CHARS = 1200


def vocabulary_file(path: Path) -> tuple[str, ...]:
    """Load one term per line; reject unexpectedly large or malformed local input."""
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("Vocabulary file exceeds the 16 KiB limit")
        raw = path.read_text(encoding="utf-8-sig")
    except OSError:
        raise ValueError("Cannot read vocabulary file; check the path and permissions") from None
    except UnicodeError:
        raise ValueError("Vocabulary file must be UTF-8 text") from None
    terms = [line.strip() for line in raw.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if len(terms) > MAX_TERMS:
        raise ValueError("Vocabulary file contains too many terms (limit: 64)")
    if any(len(term) > MAX_TERM_CHARS or any(ord(ch) < 32 for ch in term) for term in terms):
        raise ValueError("Vocabulary entries must be short, printable lines")
    return tuple(terms)


def build_prompt(
    initial_prompt: str | None = None,
    profile: str | None = None,
    vocabulary_path: Path | None = None,
) -> str | None:
    """Compose explicit user context and unique vocabulary; no implicit postprocessing."""
    if initial_prompt is not None and (
        len(initial_prompt) > MAX_PROMPT_CHARS
        or any(ord(char) < 32 and char not in "\n\t" for char in initial_prompt)
    ):
        raise ValueError("Initial prompt is too long or contains control characters")
    if profile is not None and profile not in PROFILES:
        raise ValueError("Unknown vocabulary profile")
    terms = list(PROFILES.get(profile, ()))
    if vocabulary_path is not None:
        terms.extend(vocabulary_file(vocabulary_path))
    unique: list[str] = []
    seen: set[str] = set()
    for term in terms:
        key = term.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(term)
    if len(unique) > MAX_TERMS:
        raise ValueError("Combined vocabulary exceeds 64 terms")
    context = (initial_prompt or "").strip()
    if unique:
        hint = "Terminology: " + ", ".join(unique) + "."
        context = f"{context}\n{hint}".strip() if context else hint
    if len(context) > MAX_PROMPT_CHARS:
        raise ValueError("Combined initial prompt exceeds 1200 characters")
    return context or None
