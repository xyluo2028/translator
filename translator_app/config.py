from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib


@dataclass(frozen=True)
class ProviderConfig:
    name: str = "ollama"


@dataclass(frozen=True)
class OllamaConfig:
    host: str = "http://localhost:11434"
    model: str = "gpt-oss:20b"


@dataclass(frozen=True)
class TransformersConfig:
    model: str = "google/gemma-4-E4B-it"
    device_map: str = "auto"
    dtype: str = "auto"
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
class AppConfig:
    provider: ProviderConfig = ProviderConfig()
    ollama: OllamaConfig = OllamaConfig()
    transformers: TransformersConfig = TransformersConfig()
    defaults: DefaultsConfig = DefaultsConfig()


def _get_table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f'Expected "{key}" to be a table in config.toml')
    return value


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        return AppConfig()

    data = tomllib.loads(config_path.read_text(encoding="utf-8"))

    provider_table = _get_table(data, "provider")
    ollama_table = _get_table(data, "ollama")
    transformers_table = _get_table(data, "transformers")
    defaults_table = _get_table(data, "defaults")

    provider = ProviderConfig(name=str(provider_table.get("name", "ollama")))
    ollama = OllamaConfig(
        host=str(ollama_table.get("host", "http://localhost:11434")),
        model=str(ollama_table.get("model", "gpt-oss:20b")),
    )
    max_new_tokens = transformers_table.get("max_new_tokens", 512)
    if not isinstance(max_new_tokens, int):
        raise TypeError('Expected "transformers.max_new_tokens" to be an integer')
    enable_thinking = transformers_table.get("enable_thinking", False)
    if not isinstance(enable_thinking, bool):
        raise TypeError('Expected "transformers.enable_thinking" to be a boolean')
    trust_remote_code = transformers_table.get("trust_remote_code", False)
    if not isinstance(trust_remote_code, bool):
        raise TypeError('Expected "transformers.trust_remote_code" to be a boolean')

    transformers = TransformersConfig(
        model=str(transformers_table.get("model", "google/gemma-4-E4B-it")),
        device_map=str(transformers_table.get("device_map", "auto")),
        dtype=str(transformers_table.get("dtype", "auto")),
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

    return AppConfig(provider=provider, ollama=ollama, transformers=transformers, defaults=defaults)
