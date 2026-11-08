"""Offline vocabulary preparation, validation and regression tests."""

import pytest

from video_transcriber.vocabulary import build_prompt, vocabulary_file


def test_builtin_nfl_vocabulary_is_opt_in():
    assert build_prompt() is None
    assert "Patrick Mahomes" in build_prompt(profile="nfl")
    assert "PlayStation" in build_prompt(profile="gaming")
    assert "Databricks" in build_prompt(profile="technology")


def test_prompt_file_dedup_and_preserve_unicode(tmp_path):
    path = tmp_path / "terms.txt"
    path.write_text("# Glossary\nPatrick Mahomes\nMÁQUINA\npatrick mahomes\n", encoding="utf-8")
    prompt = build_prompt("NFL en español.", "nfl", path)
    assert prompt.startswith("NFL en español.")
    assert prompt.count("Patrick Mahomes") == 1
    assert "MÁQUINA" in prompt


def test_file_can_use_utf8_bom(tmp_path):
    path = tmp_path / "vocab.txt"
    path.write_bytes(b"\xef\xbb\xbfNFL\n")
    assert vocabulary_file(path) == ("NFL",)


@pytest.mark.parametrize("bad", ["a" * 1201, "\x00foo"])
def test_reject_bad_initial_prompt(bad):
    with pytest.raises(ValueError, match="Initial prompt"):
        build_prompt(bad)


def test_reject_file_too_large_and_avoid_echo(tmp_path):
    path = tmp_path / "words.txt"
    path.write_text("secret" * 3000, encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        build_prompt(vocabulary_path=path)
    assert "secret" not in str(exc.value)


def test_missing_and_non_utf8_file(tmp_path):
    with pytest.raises(ValueError, match="Cannot read"):
        build_prompt(vocabulary_path=tmp_path / "missing.txt")
    path = tmp_path / "binary.txt"
    path.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="UTF-8"):
        vocabulary_file(path)


def test_overflow_and_unprintable_terms(tmp_path):
    path = tmp_path / "terms.txt"
    path.write_text("\n".join(f"term_{i}" for i in range(65)), encoding="utf-8")
    with pytest.raises(ValueError, match="too many"):
        vocabulary_file(path)
    path.write_text("a" * 101, encoding="utf-8")
    with pytest.raises(ValueError, match="printable"):
        vocabulary_file(path)


def test_profile_validation():
    with pytest.raises(ValueError, match="Unknown vocabulary"):
        build_prompt(profile="unknown")
