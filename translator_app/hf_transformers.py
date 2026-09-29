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


# Keep at most two models resident (typically a translation model + the dictionary model).
_MAX_CACHED_MODELS = 2
_MODEL_CACHE: dict[tuple[str, str, str, bool], _LoadedModel] = {}
_MODEL_CACHE_LOCK = Lock()
# One generate() at a time: the web UI serves requests from threads and models are shared.
_GENERATE_LOCK = Lock()


def _load_dependencies() -> tuple[Any, Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor
        from transformers.utils import logging as hf_logging
    except ImportError as exc:
        raise TransformersError(
            "Transformers backend requires the optional dependencies. "
            "Install them with: pip install -e .[transformers]"
        ) from exc
    # Model configs trigger harmless warnings (rope keys, tokenizer cleanup); keep the CLI output clean.
    hf_logging.set_verbosity_error()
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


def _describe_load_error(model: str, exc: Exception) -> str:
    message = f"Could not load Hugging Face model {model!r}: {type(exc).__name__}: {exc}"
    text = str(exc).lower()
    if "gated" in text or "401" in text or "403" in text or "restricted" in text:
        message += (
            f"\nThis model is gated: accept its license at https://huggingface.co/{model}"
            " and log in with `hf auth login`."
        )
    return message


def is_model_cached(model: str) -> bool:
    """True when the model's weights are already in the local Hugging Face cache (no download needed)."""
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return False
    try:
        for repo in scan_cache_dir().repos:
            if repo.repo_id == model and repo.repo_type == "model":
                return any(f.file_name.endswith(".safetensors") for rev in repo.revisions for f in rev.files)
    except Exception:  # noqa: BLE001
        return False
    return False


def dependencies_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _release_accelerator_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_model(config: TransformersConfig) -> _LoadedModel:
    cache_key = (config.model, config.device_map, config.dtype, config.trust_remote_code)
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.pop(cache_key, None)
        if cached is not None:
            _MODEL_CACHE[cache_key] = cached  # move to most-recently-used
            return cached
        if len(_MODEL_CACHE) >= _MAX_CACHED_MODELS:
            while len(_MODEL_CACHE) >= _MAX_CACHED_MODELS:
                _MODEL_CACHE.pop(next(iter(_MODEL_CACHE)))
            _release_accelerator_memory()
        loaded = _load_model_uncached(config)
        _MODEL_CACHE[cache_key] = loaded
        return loaded


def _load_model_uncached(config: TransformersConfig) -> _LoadedModel:
    AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor = _load_dependencies()
    model_kwargs = {
        "device_map": config.device_map,
        "trust_remote_code": config.trust_remote_code,
        "dtype": _resolve_torch_dtype(config.dtype),
    }

    try:
        processor = AutoProcessor.from_pretrained(config.model, trust_remote_code=config.trust_remote_code)
    except Exception as exc:  # noqa: BLE001
        raise TransformersError(_describe_load_error(config.model, exc)) from exc
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
    try:
        model = AutoModelForImageTextToText.from_pretrained(config.model, **model_kwargs)
    except Exception:
        try:
            model = AutoModelForCausalLM.from_pretrained(config.model, **model_kwargs)
        except Exception as exc:  # noqa: BLE001
            raise TransformersError(_describe_load_error(config.model, exc)) from exc

    return _LoadedModel(processor=processor, model=model)


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


def _build_message_inputs(processor: Any, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> Any:
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=add_generation_prompt,
    )


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
    content = _generate(loaded, inputs, config=config, sampling={"temperature": temperature}, seed=seed)
    latency_ms = int((time.time() - start) * 1000)
    return TransformersResponse(content=content, model=config.model, latency_ms=latency_ms, raw=content)


def chat_messages(
    *,
    config: TransformersConfig,
    messages: list[dict[str, Any]],
    add_generation_prompt: bool = True,
    sampling: dict[str, Any] | None = None,
    seed: int | None = None,
) -> TransformersResponse:
    """Generate from explicit chat messages (used for translation-only models and their own templates)."""
    start = time.time()
    loaded = _load_model(config)
    try:
        inputs = _build_message_inputs(loaded.processor, messages, add_generation_prompt=add_generation_prompt)
    except Exception as exc:  # noqa: BLE001
        raise TransformersError(f"Chat template failed for model {config.model!r}: {exc}") from exc
    inputs = inputs.to(loaded.model.device)
    content = _generate(loaded, inputs, config=config, sampling=sampling or {}, seed=seed)
    latency_ms = int((time.time() - start) * 1000)
    return TransformersResponse(content=content, model=config.model, latency_ms=latency_ms, raw=content)


def _generate(
    loaded: _LoadedModel,
    inputs: Any,
    *,
    config: TransformersConfig,
    sampling: dict[str, Any],
    seed: int | None,
) -> str:
    input_len = int(inputs["input_ids"].shape[-1])
    temperature = float(sampling.get("temperature", 0.0))

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "pad_token_id": _pad_token_id(loaded.model, loaded.processor),
    }
    if temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
        for key in ("top_k", "top_p"):
            if key in sampling:
                generation_kwargs[key] = sampling[key]
    else:
        generation_kwargs["do_sample"] = False
    if "repeat_penalty" in sampling:
        generation_kwargs["repetition_penalty"] = sampling["repeat_penalty"]

    if seed is not None:
        try:
            import torch
        except ImportError:
            pass
        else:
            torch.manual_seed(seed)

    try:
        with _GENERATE_LOCK:
            outputs = loaded.model.generate(**inputs, **generation_kwargs)
    except Exception as exc:  # noqa: BLE001
        raise TransformersError(f"Transformers generation failed for model {config.model!r}: {exc}") from exc

    generated_ids = outputs[0][input_len:]
    if hasattr(loaded.processor, "decode"):
        return loaded.processor.decode(generated_ids, skip_special_tokens=True).strip()
    tokenizer = getattr(loaded.processor, "tokenizer", None)
    if tokenizer is None:
        raise TransformersError(f"Could not decode output for model {config.model!r}")
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
