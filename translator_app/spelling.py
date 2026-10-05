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
MIN_LANGUAGE_CONFIDENCE = 0.4
MIN_LANGUAGE_MARGIN = 0.2
# Latin-script dictionaries consulted before treating an undetected word as an English typo.
_LATIN_LANGS = SUPPORTED_LANGS - {"EN", "RU", "AR", "FA"}

_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")


@lru_cache(maxsize=2 * len(SUPPORTED_LANGS))
def _checker(lang: str, distance: int) -> Any:
    from spellchecker import SpellChecker

    return SpellChecker(language=lang.lower(), distance=distance)


def available() -> bool:
    try:
        _checker("EN", 1)
    except Exception:  # noqa: BLE001 - package not installed
        return False
    return True


@lru_cache(maxsize=1)
def _detector() -> Any:
    from lingua import IsoCode639_1, LanguageDetectorBuilder
    from translator_app.prompting import LANGUAGES

    # Include translation languages without spelling dictionaries too, so they aren't forced
    # into one of the spellcheck languages. The bundled models run entirely offline.
    codes = {code.split("-")[0] for code in LANGUAGES} | SUPPORTED_LANGS
    return LanguageDetectorBuilder.from_iso_codes_639_1(
        *(getattr(IsoCode639_1, code) for code in sorted(codes))
    ).build()


def detect_source_language(text: str) -> str | None:
    """Identify an app-supported language, declining ambiguous input or missing dependencies."""
    if not text.strip():
        return None
    try:
        detector = _detector()
    except ImportError:
        return None
    scores = detector.compute_language_confidence_values(text)
    if not scores:
        return None
    best = scores[0]
    runner_up = scores[1].value if len(scores) > 1 else 0.0
    if best.value < MIN_LANGUAGE_CONFIDENCE or best.value - runner_up < MIN_LANGUAGE_MARGIN:
        return None
    return best.language.iso_code_639_1.name


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

    Auto mode first identifies the language with the offline detector; explicit or confidently detected
    supported languages allow edits up to distance 2. The detector is often unsure about short or misspelled
    Latin-script input ("boook"), so that falls back to English at distance 1, unless most words aren't English
    or an unknown word is a real word in another dictionary ("hola", "merci").
    """
    from translator_app.prompting import detect_language

    lang, distance = source_lang.upper(), 2
    fallback = False
    if lang == "AUTO":
        lang = detect_source_language(text)
        if lang is None and detect_language(text) == "EN":
            lang, distance, fallback = "EN", 1, True
    if lang not in SUPPORTED_LANGS:
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
    if fallback:
        if len(words) > 1 and sum(w in unknown for w in words) * 2 > len(words):
            return None  # mostly unknown words: probably not English at all
        if any(_checker(other, 2).known(unknown) for other in sorted(_LATIN_LANGS)):
            return None  # a real word elsewhere, not an English typo

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
                start=m.start(),
                end=m.end(),
            )
        )
        pieces.append(text[pos : m.start()])
        pieces.append(suggestion)
        pos = m.end()
    if not corrections:
        return None
    pieces.append(text[pos:])
    return SpellingFix(original=text, corrected="".join(pieces), corrections=corrections)
