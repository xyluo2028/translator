from __future__ import annotations

import json
import random
from dataclasses import replace
from typing import Any, TypeVar

from translator_app import spelling
from translator_app.config import AppConfig
from translator_app.hf_transformers import TransformersError, chat_json as hf_chat_json, chat_messages as hf_chat_messages
from translator_app.models import (
    DictionaryEntry,
    DictionaryResult,
    DictionarySense,
    TranslateRequest,
    TranslateResult,
)
from translator_app.ollama import OllamaError, chat_json, chat_text, generate_json
from translator_app.prompting import (
    PromptStyle,
    build_plain_messages,
    build_plain_prompt,
    build_system_prompt,
    build_user_prompt,
    detect_language,
    prompt_style_for,
)


class ProviderResponseParseError(RuntimeError):
    def __init__(self, message: str, *, raw_response: str | None = None):
        super().__init__(message)
        self.raw_response = raw_response


T = TypeVar("T")


def _response_schema_for(request: TranslateRequest) -> dict[str, Any]:
    if request.mode == "dictionary":
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["term", "entries"],
            "properties": {
                "term": {"type": "string"},
                "entries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["pos", "senses"],
                        "properties": {
                            "pos": {"type": ["string", "null"]},
                            "senses": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["meaning", "example_source", "example_target", "usage_notes"],
                                    "properties": {
                                        "meaning": {"type": "string"},
                                        "example_source": {"type": ["string", "null"]},
                                        "example_target": {"type": ["string", "null"]},
                                        "usage_notes": {"type": ["string", "null"]},
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["translation", "alternatives", "notes", "detected_source_lang"],
        "properties": {
            "translation": {"type": "string"},
            "alternatives": {"type": ["array", "null"], "items": {"type": "string"}},
            "notes": {"type": ["string", "null"]},
            "detected_source_lang": {"type": ["string", "null"]},
        },
    }


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


def resolve_model(request: TranslateRequest, config: AppConfig) -> tuple[str, PromptStyle]:
    """Pick the model for a request on the active provider. Translation-only models can't do dictionary mode."""
    if config.provider.name == "transformers":
        model, dictionary_model = request.model or config.transformers.model, config.transformers.dictionary_model
    else:
        model, dictionary_model = request.model or config.ollama.model, config.ollama.dictionary_model
    style = prompt_style_for(model)
    if style != "json" and request.mode == "dictionary":
        model = dictionary_model
        style = prompt_style_for(model)
    if style != "json" and request.mode == "dictionary":
        raise ValueError(f"Dictionary mode needs a general chat model; {model!r} is translation-only.")
    return model, style


def translate_text(request: TranslateRequest, *, config: AppConfig) -> TranslateResult | DictionaryResult:
    if config.provider.name not in ("ollama", "transformers"):
        raise ValueError(f"Unsupported provider: {config.provider.name!r}")

    fix = spelling.check(request.text, source_lang=request.source_lang) if request.spellcheck else None
    if fix is None:
        return _translate(request, config=config)
    result = _translate(replace(request, text=fix.corrected), config=config)
    return replace(result, spelling=fix)


def _translate(request: TranslateRequest, *, config: AppConfig) -> TranslateResult | DictionaryResult:

    model, style = resolve_model(request, config)
    if style != "json":
        return _translate_plain(request, config=config, model=model, style=style)

    system = build_system_prompt(request)
    user = build_user_prompt(request)
    if config.provider.name == "ollama":
        return _translate_with_ollama(
            request, config=config, model=model, system=system, user=user, response_schema=_response_schema_for(request)
        )
    config = replace(config, transformers=replace(config.transformers, model=model))
    return _translate_with_transformers(request, config=config, system=system, user=user)


def _translate_plain(
    request: TranslateRequest,
    *,
    config: AppConfig,
    model: str,
    style: PromptStyle,
) -> TranslateResult:
    """Translation-only models (HY-MT, TranslateGemma): official prompt in, bare translation out."""
    source_lang = request.source_lang
    detected = None
    if source_lang.lower() == "auto":
        detected = detect_language(request.text)
        source_lang = detected or ""
        # Latin script is only a guess at English (could be French, German, ...); don't report it as detected.
        if detected == "EN":
            detected = None

    sampling: dict[str, Any] = {"temperature": request.temperature}
    if style == "hunyuan":
        # Tencent's recommended sampling settings for HY-MT1.5.
        sampling.update({"top_k": 20, "top_p": 0.6, "repeat_penalty": 1.05})
    seed = request.seed
    if request.rerun:
        # No schema to steer, so a rerun just samples a different candidate.
        sampling["temperature"] = max(request.temperature, 0.7)
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

    if config.provider.name == "transformers":
        messages, add_generation_prompt = build_plain_messages(request, style=style, source_lang=source_lang)
        resp = hf_chat_messages(
            config=replace(config.transformers, model=model),
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            sampling=sampling,
            seed=seed,
        )
        tone_model = config.transformers.dictionary_model
    else:
        options = dict(sampling)
        if seed is not None:
            options["seed"] = seed
        prompt = build_plain_prompt(request, style=style, source_lang=source_lang)
        resp = chat_text(host=config.ollama.host, model=model, user=prompt, options=options)
        tone_model = config.ollama.dictionary_model

    # HY-MT ends lines with markdown hard breaks ("  \n"); drop trailing whitespace per line.
    translation = "\n".join(line.rstrip() for line in resp.content.strip().splitlines())
    if not translation:
        raise ProviderResponseParseError("Model returned an empty translation.", raw_response=resp.raw)

    notes = []
    if request.tone not in ("", "neutral") or request.tone_instructions:
        notes.append(f"Tone is not supported by translation-only model {model}; use {tone_model} for tone.")
    if request.rerun and request.rerun.style != "retry":
        notes.append(f"{request.rerun.style} is not supported by {model}; returned a fresh sample instead.")

    return TranslateResult(
        translation=translation,
        notes=" ".join(notes) or None,
        detected_source_lang=detected,
        provider=config.provider.name,
        model=resp.model or model,
        latency_ms=resp.latency_ms,
    )


def _translate_with_ollama(
    request: TranslateRequest,
    *,
    config: AppConfig,
    model: str,
    system: str,
    user: str,
    response_schema: dict[str, Any],
) -> TranslateResult | DictionaryResult:

    resp = None
    content = ""
    last_parse_error: ProviderResponseParseError | None = None

    for attempt in range(1, 4):
        temperature = request.temperature if attempt == 1 else 0.0
        try:
            try:
                resp = chat_json(
                    host=config.ollama.host,
                    model=model,
                    system=system,
                    user=user,
                    response_format=response_schema,
                    temperature=temperature,
                    seed=request.seed,
                )
                content = resp.content
            except OllamaError:
                resp = chat_json(
                    host=config.ollama.host,
                    model=model,
                    system=system,
                    user=user,
                    response_format="json",
                    temperature=temperature,
                    seed=request.seed,
                )
                content = resp.content

            obj = _parse_json(content)
            last_parse_error = None
            break
        except ProviderResponseParseError as e:
            last_parse_error = e
            if attempt == 1:
                prompt = system + "\n\n" + user
                resp = generate_json(
                    host=config.ollama.host,
                    model=model,
                    prompt=prompt,
                    response_format="json",
                    temperature=0.0,
                    seed=request.seed,
                )
                content = resp.content
                try:
                    obj = _parse_json(content)
                    last_parse_error = None
                    break
                except ProviderResponseParseError as e2:
                    last_parse_error = e2
            if attempt >= 3:
                raise

    if last_parse_error is not None:
        raise last_parse_error

    if request.mode == "dictionary":
        assert resp is not None
        return _as_dictionary_result(request, obj, provider="ollama", model=resp.model, latency_ms=resp.latency_ms)

    assert resp is not None
    return _as_translate_result(obj, provider="ollama", model=resp.model, latency_ms=resp.latency_ms)


def _translate_with_transformers(
    request: TranslateRequest,
    *,
    config: AppConfig,
    system: str,
    user: str,
) -> TranslateResult | DictionaryResult:
    resp = None
    content = ""
    last_parse_error: ProviderResponseParseError | None = None

    for attempt in range(1, 4):
        temperature = request.temperature if attempt == 1 else 0.0
        user_prompt = user
        if attempt >= 2:
            user_prompt += (
                "\nReturn exactly one JSON object."
                "\nDo not add markdown, commentary, or any text before or after the JSON."
            )
        try:
            resp = hf_chat_json(
                config=config.transformers,
                system=system,
                user=user_prompt,
                temperature=temperature,
                seed=request.seed,
            )
            content = resp.content
            obj = _parse_json(content)
            last_parse_error = None
            break
        except ProviderResponseParseError as exc:
            last_parse_error = exc
            if attempt >= 3:
                raise
        except TransformersError as exc:
            raise exc

    if last_parse_error is not None:
        raise last_parse_error

    assert resp is not None
    if request.mode == "dictionary":
        return _as_dictionary_result(
            request,
            obj,
            provider="transformers",
            model=resp.model,
            latency_ms=resp.latency_ms,
        )
    return _as_translate_result(obj, provider="transformers", model=resp.model, latency_ms=resp.latency_ms)
