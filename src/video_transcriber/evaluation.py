"""Reproducible offline WER/CER for human-reference and model-transcript files."""

import json
import re
import unicodedata
from pathlib import Path

from rapidfuzz.distance import Levenshtein

MAX_EVALUATION_BYTES = 2_000_000
_TIMESTAMP_PREFIX = re.compile(r"^\[\d{2,}:\d{2}:\d{2} - \d{2,}:\d{2}:\d{2}\]\s*")


def read_transcript(path: Path) -> str:
    """Read plain TXT or V1.3 JSON, stripping our standard TXT timestamp prefix."""
    if path.suffix.lower() not in {".txt", ".json"}:
        raise ValueError("Evaluation input must be a TXT or V1.3 JSON file")
    try:
        if path.stat().st_size > MAX_EVALUATION_BYTES:
            raise ValueError("Evaluation file exceeds the 2 MB limit")
        content = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError):
        raise ValueError("Cannot read evaluation file as UTF-8") from None
    if path.suffix.lower() == ".json":
        try:
            document = json.loads(content)
            if (
                not isinstance(document, dict)
                or document.get("schema_version") != 1
                or not isinstance(document.get("segments"), list)
                or any(not isinstance(s, dict) or not isinstance(s.get("text"), str)
                       for s in document["segments"])
            ):
                raise ValueError
            return " ".join(s["text"] for s in document["segments"])
        except (ValueError, TypeError):
            raise ValueError("Invalid V1.3 transcription JSON") from None
    return " ".join(_TIMESTAMP_PREFIX.sub("", line) for line in content.splitlines())


def normalize(text: str) -> str:
    """NFC + casefold + Unicode word tokens; punctuation ignored, spaces collapsed."""
    value = unicodedata.normalize("NFC", text).casefold()
    return " ".join(re.findall(r"\w+", value, flags=re.UNICODE))


def evaluate(reference_text: str, hypothesis_text: str) -> dict:
    """Compute word and character error rates against a nonempty human reference."""
    reference = normalize(reference_text)
    hypothesis = normalize(hypothesis_text)
    if not reference:
        raise ValueError("Reference text has no words after normalization")
    ref_words, hyp_words = reference.split(), hypothesis.split()
    word_errors = Levenshtein.distance(ref_words, hyp_words)
    char_errors = Levenshtein.distance(reference, hypothesis)
    return {
        "normalization": "NFC,casefold,unicode_words,no_punctuation,collapsed_spaces",
        "reference_words": len(ref_words),
        "reference_characters": len(reference),
        "word_errors": word_errors,
        "character_errors": char_errors,
        "wer": round(word_errors / len(ref_words), 6),
        "cer": round(char_errors / len(reference), 6),
    }


def evaluate_files(reference: Path, hypothesis: Path) -> dict:
    return evaluate(read_transcript(reference), read_transcript(hypothesis))
