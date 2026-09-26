"""Annotate Japanese kanji with readings in 漢字（かんじ） form, using fugashi + unidic-lite (optional extra `text`)."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

# UniDic's default reading differs from everyday usage for a few very common words.
_READING_OVERRIDES = {"日本": "ニホン", "日本人": "ニホンジン", "日本語": "ニホンゴ", "明日": "アシタ"}


def _is_kanji(ch: str) -> bool:
    cp = ord(ch)
    return 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0xF900 <= cp <= 0xFAFF or ch in "々〆ヶ"


def _is_kana(ch: str) -> bool:
    return 0x3040 <= ord(ch) <= 0x30FF


def _to_hiragana(text: str) -> str:
    return "".join(chr(ord(ch) - 0x60) if 0x30A1 <= ord(ch) <= 0x30F6 else ch for ch in text)


def _to_katakana(text: str) -> str:
    return "".join(chr(ord(ch) + 0x60) if 0x3041 <= ord(ch) <= 0x3096 else ch for ch in text)


def has_kanji(text: str) -> bool:
    return any(_is_kanji(ch) for ch in text)


@lru_cache(maxsize=1)
def _tagger() -> Any:
    import fugashi

    return fugashi.Tagger()


def available() -> bool:
    try:
        _tagger()
    except Exception:  # noqa: BLE001 - missing package or dictionary
        return False
    return True


def _annotate_token(surface: str, reading: str) -> str:
    """Attach the reading to each kanji run: 打ち合わせ + ウチアワセ -> 打（う）ち合（あ）わせ."""
    runs = re.findall(r"[^぀-ヿ]+|[぀-ヿ]+", surface)
    pattern = "".join(re.escape(_to_katakana(run)) if _is_kana(run[0]) else "(.+?)" for run in runs)
    match = re.fullmatch(pattern, reading)
    if not match:
        return f"{surface}（{_to_hiragana(reading)}）"
    groups = iter(match.groups())
    out = []
    for run in runs:
        if _is_kana(run[0]):
            out.append(run)
        else:
            out.append(f"{run}（{_to_hiragana(next(groups))}）" if has_kanji(run) else run)
    return "".join(out)


def annotate(text: str) -> str:
    """Return `text` with readings after every kanji word. Text without kanji is returned unchanged."""
    if not has_kanji(text):
        return text
    tagger = _tagger()
    out = []
    for line in text.split("\n"):
        pieces = []
        pos = 0
        for word in tagger(line):
            surface = word.surface
            start = line.find(surface, pos)
            if start > pos:
                pieces.append(line[pos:start])  # whitespace the tagger skipped
            pos = start + len(surface) if start >= 0 else pos
            reading = _READING_OVERRIDES.get(surface) or getattr(word.feature, "kana", None)
            if has_kanji(surface) and reading and reading != "*":
                pieces.append(_annotate_token(surface, reading))
            else:
                pieces.append(surface)
        pieces.append(line[pos:])
        out.append("".join(pieces))
    return "\n".join(out)


def _is_japanese_lang(code: str | None) -> bool:
    return (code or "").upper().startswith("JA")


def annotate_result(
    payload: dict[str, Any], *, source_text: str, source_lang: str, target_lang: str
) -> dict[str, Any]:
    """Annotate the Japanese parts of a translate/dictionary result (as a dict from asdict).

    Returns {"result": annotated copy of payload, "source": annotated source text or None}.
    Only text known to be Japanese is annotated, so Chinese hanzi never gets Japanese readings.
    """
    from translator_app.prompting import detect_language

    target_ja = _is_japanese_lang(target_lang)
    source_ja = _is_japanese_lang(source_lang) or (
        source_lang.lower() == "auto"
        and (detect_language(source_text) == "JA" or _is_japanese_lang(payload.get("detected_source_lang")))
    )
    result = dict(payload)

    if "entries" in payload:  # dictionary result
        entries = []
        for entry in payload.get("entries") or []:
            senses = []
            for sense in entry.get("senses") or []:
                sense = dict(sense)
                if source_ja and sense.get("example_source"):
                    sense["example_source"] = annotate(sense["example_source"])
                if target_ja:
                    for key in ("meaning", "example_target"):
                        if sense.get(key):
                            sense[key] = annotate(sense[key])
                senses.append(sense)
            entries.append({**entry, "senses": senses})
        result["entries"] = entries
        if source_ja and payload.get("term"):
            result["term"] = annotate(payload["term"])
    elif target_ja:
        if payload.get("translation"):
            result["translation"] = annotate(payload["translation"])
        if payload.get("alternatives"):
            result["alternatives"] = [annotate(alt) for alt in payload["alternatives"]]

    source = annotate(source_text) if source_ja and has_kanji(source_text) else None
    return {"result": result, "source": source}
