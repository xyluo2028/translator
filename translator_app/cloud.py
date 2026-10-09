"""Dependency-free OpenAI and Gemini Chat Completions clients."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from translator_app.config import CloudConfig


ENDPOINTS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
}


class CloudError(RuntimeError):
    pass


@dataclass(frozen=True)
class CloudResponse:
    content: str
    model: str
    latency_ms: int


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward a provider credential to a redirected endpoint.
        return None


def key_available(config: CloudConfig) -> bool:
    return bool(os.environ.get(config.api_key_env, "").strip())


def chat_json(
    *, provider: str, config: CloudConfig, model: str, system: str, user: str,
    response_schema: dict[str, Any], temperature: float,
) -> CloudResponse:
    key = os.environ.get(config.api_key_env, "").strip()
    if not key:
        raise CloudError(f"Set {config.api_key_env} in the environment before using {provider}.")
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "translator_result", "strict": True, "schema": response_schema},
        },
    }
    if config.reasoning_effort:
        payload["reasoning_effort"] = config.reasoning_effort
    if provider == "openai":
        payload["max_completion_tokens"] = config.max_output_tokens
        payload["store"] = False
        # Sampling controls are not supported by all OpenAI reasoning models.
    else:
        payload["max_tokens"] = config.max_output_tokens
        payload["temperature"] = temperature
    request = urllib.request.Request(
        ENDPOINTS[provider], data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.build_opener(_NoRedirect()).open(request, timeout=config.timeout_s) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        # Upstream bodies can contain credentials or user text; never surface them.
        hints = {
            400: "Check the model's support for structured output and reasoning_effort in config.toml.",
            401: f"Check {config.api_key_env}; the API key was rejected.",
            403: "Check API key permissions, model access, and supported region.",
            404: "Check the model ID and your account's access to it.",
            429: "Rate limit or quota exceeded. Check usage/billing and retry later.",
        }
        exc.close()
        raise CloudError(f"{provider} HTTP {exc.code}: {hints.get(exc.code, 'Provider request failed; retry later.')}") from None
    except (urllib.error.URLError, OSError, TimeoutError):
        raise CloudError(f"Could not reach {provider} or the request timed out. Check your connection and retry.") from None
    except UnicodeError:
        raise CloudError(f"{provider} returned an invalid response encoding.") from None
    try:
        obj = json.loads(raw)
        choice = obj["choices"][0]
        message = choice["message"]
        if message.get("refusal") or choice.get("finish_reason") == "content_filter":
            raise CloudError(f"{provider} declined this request.")
        if choice.get("finish_reason") == "length":
            raise CloudError(f"{provider} output was truncated. Increase {provider}.max_output_tokens or shorten the input.")
        content = message["content"]
        if not isinstance(content, str) or not content.strip():
            raise CloudError(f"{provider} returned no text. Try a larger max_output_tokens budget.")
        returned_model = obj.get("model")
        return CloudResponse(
            content=content, model=returned_model if isinstance(returned_model, str) and returned_model else model,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        raise CloudError(f"{provider} returned an invalid Chat Completions response.") from None
