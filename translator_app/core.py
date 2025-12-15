from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, TypeVar

from translator_app.config import AppConfig
from translator_app.models import (
    DictionaryEntry,
    DictionaryResult,
    DictionarySense,
    TranslateRequest,
    TranslateResult,
)
from translator_app.ollama import OllamaError, chat_json, generate_json
from translator_app.prompting import build_system_prompt, build_user_prompt


class ProviderResponseParseError(RuntimeError):
    def __init__(self, message: str, *, raw_response: str | None = None):
        super().__init__(message)
        self.raw_response = raw_response


T = TypeVar("T")


def _extract_first_json_object(text: str) -> str:
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    start = s.find("{")
    if start == -1:
        raise ProviderResponseParseError("Model did not return a JSON object.", raw_response=text)
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    raise ProviderResponseParseError("Unterminated JSON object in model output.", raw_response=text)


def _parse_json(text: str) -> dict[str, Any]:
    blob = _extract_first_json_object(text)
    try:
        obj = json.loads(blob)
    except Exception as exc:  # noqa: BLE001
        raise ProviderResponseParseError("Failed to parse JSON from model output.", raw_response=text) from exc
    if not isinstance(obj, dict):
        raise ProviderResponseParseError("Expected a JSON object from model output.", raw_response=text)
    return obj


def _coerce_str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(x) for x in value]
    return None


def _as_translate_result(obj: dict[str, Any], *, provider: str, model: str | None, latency_ms: int | None) -> TranslateResult:
    translation = str(obj.get("translation", "")).strip()
    if not translation:
        raise ProviderResponseParseError("Missing 'translation' in model output.", raw_response=json.dumps(obj))
    return TranslateResult(
        translation=translation,
        alternatives=_coerce_str_list(obj.get("alternatives")),
        notes=(str(obj["notes"]).strip() if obj.get("notes") not in (None, "") else None),
        detected_source_lang=(
            str(obj["detected_source_lang"]).strip() if obj.get("detected_source_lang") not in (None, "") else None
        ),
        provider=provider,
        model=model,
        latency_ms=latency_ms,
    )


def _as_dictionary_result(
    request: TranslateRequest,
    obj: dict[str, Any],
    *,
    provider: str,
    model: str | None,
    latency_ms: int | None,
) -> DictionaryResult:
    term = str(obj.get("term") or request.text).strip()
    entries_raw = obj.get("entries") or []
    if not isinstance(entries_raw, list):
        raise ProviderResponseParseError("Expected 'entries' to be a list.", raw_response=json.dumps(obj))
    entries: list[DictionaryEntry] = []
    for entry_raw in entries_raw:
        if not isinstance(entry_raw, dict):
            continue
        pos = entry_raw.get("pos")
        pos_s = None if pos in (None, "") else str(pos)
        senses_raw = entry_raw.get("senses") or []
        if not isinstance(senses_raw, list):
            senses_raw = []
        senses: list[DictionarySense] = []
        for sense_raw in senses_raw:
            if not isinstance(sense_raw, dict):
                continue
            meaning = str(sense_raw.get("meaning", "")).strip()
            if not meaning:
                continue
            senses.append(
                DictionarySense(
                    meaning=meaning,
                    example_source=(str(sense_raw["example_source"]) if sense_raw.get("example_source") not in (None, "") else None),
                    example_target=(str(sense_raw["example_target"]) if sense_raw.get("example_target") not in (None, "") else None),
                    usage_notes=(str(sense_raw["usage_notes"]) if sense_raw.get("usage_notes") not in (None, "") else None),
                )
            )
        if senses:
            entries.append(DictionaryEntry(pos=pos_s, senses=senses))
    if not entries:
        raise ProviderResponseParseError("No dictionary entries produced.", raw_response=json.dumps(obj))
    return DictionaryResult(term=term, entries=entries, provider=provider, model=model, latency_ms=latency_ms)


def translate_text(request: TranslateRequest, *, config: AppConfig) -> TranslateResult | DictionaryResult:
    if config.provider.name != "ollama":
        raise ValueError(f"Unsupported provider: {config.provider.name!r} (only 'ollama' implemented)")

    system = build_system_prompt(request)
    user = build_user_prompt(request)

    try:
        resp = chat_json(
            host=config.ollama.host,
            model=config.ollama.model,
            system=system,
            user=user,
            temperature=request.temperature,
            seed=request.seed,
        )
        content = resp.content
    except OllamaError:
        prompt = system + "\n\n" + user
        resp = generate_json(
            host=config.ollama.host,
            model=config.ollama.model,
            prompt=prompt,
            temperature=request.temperature,
            seed=request.seed,
        )
        content = resp.content

    obj = _parse_json(content)

    if request.mode == "dictionary":
        return _as_dictionary_result(request, obj, provider="ollama", model=resp.model, latency_ms=resp.latency_ms)

    return _as_translate_result(obj, provider="ollama", model=resp.model, latency_ms=resp.latency_ms)

