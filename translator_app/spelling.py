"""Offline "did you mean" spelling correction for the source text, using pyspellchecker (optional extra `text`)."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from translator_app.models import SpellingCorrection, SpellingFix

# pyspellchecker ships word-frequency dictionaries for these languages.
SUPPORTED_LANGS = {"EN", "ES", "FR", "PT", "DE", "IT", "RU", "AR", "NL", "LV", "EU", "FA"}
MAX_WORDS = 200
MAX_SUGGESTIONS = 3

_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")


@lru_cache(maxsize=8)
def _checker(lang: str, distance: int) -> Any:
    from spellchecker import SpellChecker

    return SpellChecker(language=lang.lower(), distance=distance)


def available() -> bool:
    try:
        _checker("EN", 1)
    except Exception:  # noqa: BLE001 - package not installed
        return False
    return True


def _should_skip(word: str, start: int, text: str) -> bool:
    if len(word) < 3 or word.isupper():
        return True  # short words and acronyms ("USA", "OK")
    if any(ch.isupper() for ch in word[1:]):
        return True  # mixed case like "iPhone", "GitHub"
    if word[0].isupper():
        # Capitalized mid-sentence is probably a name; at a sentence start it's just capitalization.
        before = text[:start].rstrip()
        return bool(before) and before[-1] not in ".!?\n\"'“‘(:"
    return False


def _match_case(original: str, suggestion: str) -> str:
    return suggestion[:1].upper() + suggestion[1:] if original[:1].isupper() else suggestion


def check(text: str, *, source_lang: str) -> SpellingFix | None:
    """Return corrections for misspelled words, or None if nothing to fix (or checking doesn't apply).

    An explicit source language allows edits up to distance 2. With "auto" we only get here for Latin-script
    text, can't be sure it's English, and use distance 1 plus a mostly-known-words check so that e.g. French
    isn't "corrected" into English.
    """
    from translator_app.prompting import detect_language

    lang = source_lang.upper()
    if lang == "AUTO":
        if detect_language(text) != "EN":
            return None
        lang, distance = "EN", 1
    elif lang in SUPPORTED_LANGS:
        distance = 2
    else:
        return None
    if not available():
        return None

    matches = list(_WORD.finditer(text))
    if not matches or len(matches) > MAX_WORDS:
        return None
    checker = _checker(lang, distance)

    words = [m.group().replace("’", "'").lower() for m in matches]
    unknown = checker.unknown(words)
    if not unknown:
        return None
    if source_lang.upper() == "AUTO" and len(words) > 1 and len(unknown) * 2 > len(words):
        return None  # mostly unknown words: probably not English at all

    corrections: list[SpellingCorrection] = []
    pieces: list[str] = []
    pos = 0
    for m, lowered in zip(matches, words):
        word = m.group()
        if lowered not in unknown or _should_skip(word, m.start(), text):
            continue
        candidates = checker.candidates(lowered) or set()
        ranked = sorted(candidates - {lowered}, key=lambda w: (-checker.word_frequency[w], w))[:MAX_SUGGESTIONS]
        if not ranked:
            continue
        suggestion = _match_case(word, ranked[0])
        corrections.append(
            SpellingCorrection(
                word=word,
                suggestion=suggestion,
                alternatives=[_match_case(word, w) for w in ranked[1:]],
            )
        )
        pieces.append(text[pos : m.start()])
        pieces.append(suggestion)
        pos = m.end()
    if not corrections:
        return None
    pieces.append(text[pos:])
    return SpellingFix(original=text, corrected="".join(pieces), corrections=corrections)
