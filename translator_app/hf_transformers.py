from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Any

from translator_app.config import TransformersConfig


class TransformersError(RuntimeError):
    def __init__(self, message: str, *, raw_response: str | None = None):
        super().__init__(message)
        self.raw_response = raw_response


@dataclass(frozen=True)
class TransformersResponse:
    content: str
    model: str | None
    latency_ms: int | None
    raw: str


@dataclass(frozen=True)
class _LoadedModel:
    processor: Any
    model: Any


_MODEL_CACHE: dict[tuple[str, str, str, bool], _LoadedModel] = {}
_MODEL_CACHE_LOCK = Lock()


def _load_dependencies() -> tuple[Any, Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise TransformersError(
            "Transformers backend requires the optional dependencies. "
            "Install them with: pip install -e .[transformers]"
        ) from exc
    return AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor


def _resolve_torch_dtype(value: str) -> Any:
    if value == "auto":
        return value
    try:
        import torch
    except ImportError as exc:
        raise TransformersError(
            "Transformers backend requires PyTorch. Install it with: pip install -e .[transformers]"
        ) from exc
    if not hasattr(torch, value):
        raise TransformersError(f"Unsupported torch dtype in config: {value!r}")
    return getattr(torch, value)


def _load_model(config: TransformersConfig) -> _LoadedModel:
    cache_key = (config.model, config.device_map, config.dtype, config.trust_remote_code)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor = _load_dependencies()
    model_kwargs = {
        "device_map": config.device_map,
        "trust_remote_code": config.trust_remote_code,
        "torch_dtype": _resolve_torch_dtype(config.dtype),
    }

    processor = AutoProcessor.from_pretrained(config.model, trust_remote_code=config.trust_remote_code)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
    try:
        model = AutoModelForImageTextToText.from_pretrained(config.model, **model_kwargs)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(config.model, **model_kwargs)

    loaded = _LoadedModel(
        processor=processor,
        model=model,
    )
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.setdefault(cache_key, loaded)
        return _MODEL_CACHE[cache_key]


def _build_inputs(processor: Any, *, system: str, user: str, enable_thinking: bool) -> Any:
    rich_messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [{"type": "text", "text": user}]},
    ]
    plain_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if hasattr(processor, "apply_chat_template"):
        base_kwargs = {
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "add_generation_prompt": True,
        }
        try:
            return processor.apply_chat_template(
                rich_messages,
                enable_thinking=enable_thinking,
                **base_kwargs,
            )
        except Exception:
            pass
        try:
            return processor.apply_chat_template(
                rich_messages,
                **base_kwargs,
            )
        except Exception:
            pass
        try:
            return processor.apply_chat_template(
                plain_messages,
                enable_thinking=enable_thinking,
                **base_kwargs,
            )
        except TypeError:
            return processor.apply_chat_template(
                plain_messages,
                **base_kwargs,
            )
    prompt = f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:\n"
    return processor(text=prompt, return_tensors="pt")


def _pad_token_id(model: Any, processor: Any) -> int | None:
    token_id = getattr(getattr(model, "generation_config", None), "pad_token_id", None)
    if token_id is not None:
        return token_id
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        if getattr(tokenizer, "pad_token_id", None) is not None:
            return tokenizer.pad_token_id
        return tokenizer.eos_token_id
    return getattr(model.config, "eos_token_id", None)


def chat_json(
    *,
    config: TransformersConfig,
    system: str,
    user: str,
    temperature: float = 0.2,
    seed: int | None = None,
) -> TransformersResponse:
    start = time.time()
    loaded = _load_model(config)
    inputs = _build_inputs(loaded.processor, system=system, user=user, enable_thinking=config.enable_thinking).to(
        loaded.model.device
    )
    input_len = int(inputs["input_ids"].shape[-1])

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "pad_token_id": _pad_token_id(loaded.model, loaded.processor),
    }
    if temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
    else:
        generation_kwargs["do_sample"] = False

    if seed is not None:
        try:
            import torch
        except ImportError:
            pass
        else:
            torch.manual_seed(seed)

    try:
        outputs = loaded.model.generate(**inputs, **generation_kwargs)
    except Exception as exc:  # noqa: BLE001
        raise TransformersError(f"Transformers generation failed for model {config.model!r}") from exc

    generated_ids = outputs[0][input_len:]
    if hasattr(loaded.processor, "decode"):
        content = loaded.processor.decode(generated_ids, skip_special_tokens=True).strip()
    else:
        tokenizer = getattr(loaded.processor, "tokenizer", None)
        if tokenizer is None:
            raise TransformersError(f"Could not decode output for model {config.model!r}")
        content = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    latency_ms = int((time.time() - start) * 1000)
    return TransformersResponse(content=content, model=config.model, latency_ms=latency_ms, raw=content)
