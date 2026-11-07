"""Reproducible offline WER/CER with Unicode and V1.3 output."""

import json

import pytest

from video_transcriber.evaluation import evaluate, evaluate_files, normalize, read_transcript


def test_normalization_unicode_punctuation_and_whitespace():
    assert normalize("  ¡HÓLA,    Mundo! \n") == "hóla mundo"


def test_identical_reference_has_zero_error():
    report = evaluate("Hola, MUNDO.", "hola mundo")
    assert report["wer"] == 0
    assert report["cer"] == 0
    assert report["reference_words"] == 2


def test_word_and_character_error_rates():
    report = evaluate("hola mundo", "hola mudo")
    assert report["word_errors"] == 1
    assert report["wer"] == 0.5
    assert report["character_errors"] == 1
    assert report["cer"] == 0.1


def test_empty_hypothesis_and_invalid_reference():
    report = evaluate("uno dos", "")
    assert report["wer"] == 1
    assert report["cer"] == 1
    with pytest.raises(ValueError, match="Reference"):
        evaluate(" ... ", "hola")


def test_json_and_timestamped_txt(tmp_path):
    reference = tmp_path / "reference.txt"
    reference.write_text("Hola mundo\nAdiós", encoding="utf-8")
    hypothesis = tmp_path / "hypothesis.json"
    hypothesis.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "language": "es",
                "language_probability": 1,
                "segments": [
                    {"start": 0, "end": 1, "text": "Hola mundo"},
                    {"start": 1, "end": 2, "text": "Adiós"},
                ],
            }
        ),
        encoding="utf-8",
    )
    assert evaluate_files(reference, hypothesis)["wer"] == 0
    txt = tmp_path / "hypothesis.txt"
    txt.write_text(
        "[00:00:00 - 00:00:01] Hola mundo\n\n"
        "[00:00:01 - 00:00:02] Adiós\n\n",
        encoding="utf-8",
    )
    assert evaluate_files(reference, txt)["cer"] == 0


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"schema_version": 2, "segments": []},
        {"schema_version": 1, "segments": [{"start": 0}]},
    ],
)
def test_invalid_json_schema(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid V1.3"):
        read_transcript(path)


def test_reject_large_or_wrong_file(tmp_path):
    path = tmp_path / "large.txt"
    path.write_text("a" * 2_000_001)
    with pytest.raises(ValueError, match="2 MB"):
        read_transcript(path)
    with pytest.raises(ValueError, match="TXT or"):
        read_transcript(tmp_path / "bad.srt")
