from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib


PROVIDERS = ("ollama", "transformers", "openai", "gemini")


@dataclass(frozen=True)
class CloudConfig:
    model: str = "gpt-6-luna"
    models: tuple[str, ...] = field(default_factory=tuple)
    api_key_env: str = "OPENAI_API_KEY"
    max_output_tokens: int = 4096
    timeout_s: float = 120.0
    reasoning_effort: str = "none"


@dataclass(frozen=True)
class ProviderConfig:
    name: str = "ollama"


@dataclass(frozen=True)
class OllamaConfig:
    host: str = "http://localhost:11434"
    model: str = "hf.co/tencent/HY-MT1.5-7B-GGUF:Q4_K_M"
    # Used for dictionary mode (and tone) when `model` is a translation-only model.
    dictionary_model: str = "gpt-oss:latest"
    options: dict[str, Any] = field(default_factory=dict)
    enable_thinking: bool | None = None
    structured_output: bool = True
    api: str = "chat"


@dataclass(frozen=True)
class TransformersConfig:
    model: str = "tencent/HY-MT1.5-7B"
    # Used for dictionary mode (and tone) when `model` is a translation-only model.
    dictionary_model: str = "google/gemma-4-E4B-it"
    # Extra models offered in the web UI's model picker.
    models: tuple[str, ...] = field(default_factory=tuple)
    device_map: str = "auto"
    dtype: str = "auto"
    quantization: str = "none"
    attn_implementation: str = "auto"
    max_cached_models: int = 2
    max_new_tokens: int = 512
    enable_thinking: bool = False
    trust_remote_code: bool = False


@dataclass(frozen=True)
class DefaultsConfig:
    source_lang: str = "auto"
    target_lang: str = "ZH"
    tone: str = "neutral"
    explain_lang: str = "EN"
    temperature: float = 0.2


@dataclass(frozen=True)
class EnhancementConfig:
    num_ctx: int = 8192
    max_new_tokens: int = 2048


@dataclass(frozen=True)
class AppConfig:
    provider: ProviderConfig = ProviderConfig()
    ollama: OllamaConfig = OllamaConfig()
    transformers: TransformersConfig = TransformersConfig()
    defaults: DefaultsConfig = DefaultsConfig()
    enhancement: EnhancementConfig = EnhancementConfig()
    openai: CloudConfig = CloudConfig()
    gemini: CloudConfig = CloudConfig(
        model="gemini-3.1-flash-lite", api_key_env="GEMINI_API_KEY", reasoning_effort="minimal",
    )


def _get_table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f'Expected "{key}" to be a table in config.toml')
    return value


def _cloud_config(data: dict[str, Any], name: str, defaults: CloudConfig) -> CloudConfig:
    table = _get_table(data, name)
    values = {key: table.get(key, getattr(defaults, key)) for key in (
        "model", "api_key_env", "max_output_tokens", "timeout_s", "reasoning_effort",
    )}
    for key in ("model", "api_key_env"):
        if not isinstance(values[key], str) or not values[key].strip():
            raise ValueError(f'Expected "{name}.{key}" to be a nonempty string')
    if type(values["max_output_tokens"]) is not int or values["max_output_tokens"] < 1:
        raise ValueError(f'Expected "{name}.max_output_tokens" to be a positive integer')
    timeout = values["timeout_s"]
    if type(timeout) not in (int, float) or not 0 < timeout < float("inf"):
        raise ValueError(f'Expected "{name}.timeout_s" to be a positive finite number')
    if values["reasoning_effort"] not in ("", "none", "minimal", "low", "medium", "high", "xhigh", "max"):
        raise ValueError(f'Invalid "{name}.reasoning_effort"')
    models = table.get("models", [])
    if not isinstance(models, list) or any(not isinstance(m, str) or not m.strip() for m in models):
        raise ValueError(f'Expected "{name}.models" to be a list of nonempty strings')
    return CloudConfig(**values, models=tuple(models))


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        return AppConfig()

    data = tomllib.loads(config_path.read_text(encoding="utf-8"))

    provider_table = _get_table(data, "provider")
    ollama_table = _get_table(data, "ollama")
    transformers_table = _get_table(data, "transformers")
    defaults_table = _get_table(data, "defaults")
    enhancement_table = _get_table(data, "enhancement")

    provider = ProviderConfig(name=str(provider_table.get("name", "ollama")))
    ollama = OllamaConfig(
        host=str(ollama_table.get("host", "http://localhost:11434")),
        model=str(ollama_table.get("model", OllamaConfig.model)),
        dictionary_model=str(ollama_table.get("dictionary_model", OllamaConfig.dictionary_model)),
        options=_get_table(ollama_table, "options"),
        enable_thinking=ollama_table.get("enable_thinking"),
        structured_output=ollama_table.get("structured_output", True),
        api=str(ollama_table.get("api", "chat")),
    )
    if ollama.enable_thinking is not None and not isinstance(ollama.enable_thinking, bool):
        raise TypeError('Expected "ollama.enable_thinking" to be a boolean')
    if not isinstance(ollama.structured_output, bool):
        raise TypeError('Expected "ollama.structured_output" to be a boolean')
    if ollama.api not in ("chat", "generate"):
        raise ValueError('Expected "ollama.api" to be "chat" or "generate"')
    max_new_tokens = transformers_table.get("max_new_tokens", 512)
    if not isinstance(max_new_tokens, int):
        raise TypeError('Expected "transformers.max_new_tokens" to be an integer')
    enable_thinking = transformers_table.get("enable_thinking", False)
    if not isinstance(enable_thinking, bool):
        raise TypeError('Expected "transformers.enable_thinking" to be a boolean')
    trust_remote_code = transformers_table.get("trust_remote_code", False)
    if not isinstance(trust_remote_code, bool):
        raise TypeError('Expected "transformers.trust_remote_code" to be a boolean')
    quantization = transformers_table.get("quantization", "none")
    if quantization not in ("none", "4bit", "8bit"):
        raise ValueError('Expected "transformers.quantization" to be "none", "4bit", or "8bit"')
    max_cached_models = transformers_table.get("max_cached_models", 2)
    if type(max_cached_models) is not int or max_cached_models < 1:
        raise ValueError('Expected "transformers.max_cached_models" to be a positive integer')

    models = transformers_table.get("models", [])
    if not isinstance(models, list) or not all(isinstance(m, str) for m in models):
        raise TypeError('Expected "transformers.models" to be a list of strings')

    transformers = TransformersConfig(
        model=str(transformers_table.get("model", TransformersConfig.model)),
        dictionary_model=str(transformers_table.get("dictionary_model", TransformersConfig.dictionary_model)),
        models=tuple(models),
        device_map=str(transformers_table.get("device_map", "auto")),
        dtype=str(transformers_table.get("dtype", "auto")),
        quantization=quantization,
        attn_implementation=str(transformers_table.get("attn_implementation", "auto")),
        max_cached_models=max_cached_models,
        max_new_tokens=max_new_tokens,
        enable_thinking=enable_thinking,
        trust_remote_code=trust_remote_code,
    )

    temperature = defaults_table.get("temperature", 0.2)
    try:
        temperature_f = float(temperature)
    except Exception as exc:  # noqa: BLE001
        raise TypeError('Expected "defaults.temperature" to be a number') from exc

    defaults = DefaultsConfig(
        source_lang=str(defaults_table.get("source_lang", "auto")),
        target_lang=str(defaults_table.get("target_lang", "ZH")),
        tone=str(defaults_table.get("tone", "neutral")),
        explain_lang=str(defaults_table.get("explain_lang", "EN")),
        temperature=temperature_f,
    )

    enhancement_values = {}
    for key in ("num_ctx", "max_new_tokens"):
        value = enhancement_table.get(key, getattr(EnhancementConfig(), key))
        if type(value) is not int or value < 1:
            raise ValueError(f'Expected "enhancement.{key}" to be a positive integer')
        enhancement_values[key] = value
    enhancement = EnhancementConfig(**enhancement_values)

    return AppConfig(
        provider=provider, ollama=ollama, transformers=transformers, defaults=defaults, enhancement=enhancement,
        openai=_cloud_config(data, "openai", AppConfig.openai),
        gemini=_cloud_config(data, "gemini", AppConfig.gemini),
    )
