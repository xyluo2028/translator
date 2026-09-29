from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin


class OllamaError(RuntimeError):
    def __init__(self, message: str, *, raw_response: str | None = None):
        super().__init__(message)
        self.raw_response = raw_response


@dataclass(frozen=True)
class OllamaResponse:
    content: str
    model: str | None
    latency_ms: int | None
    raw: str


ResponseFormat = str | dict[str, Any]


def _http_post_json(url: str, payload: dict[str, Any], timeout_s: float = 120.0) -> tuple[int, str]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        method="POST",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = getattr(resp, "status", 200)
            text = resp.read().decode("utf-8", errors="replace")
            return status, text
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        raise OllamaError(_describe_http_error(e.code, str(e.reason), raw, payload.get("model")), raw_response=raw) from e
    except urllib.error.URLError as e:
        raise OllamaError(f"Failed to reach Ollama at {url} (is `ollama serve` running?): {e.reason}") from e


def _describe_http_error(code: int, reason: str, raw: str, model: str | None) -> str:
    try:
        detail = str(json.loads(raw).get("error") or "")
    except Exception:  # noqa: BLE001
        detail = ""
    message = f"Ollama HTTP {code}: {detail or reason}"
    if code == 404 and model:
        message += f"\nPull the model first: ollama pull {model}"
    return message


def list_models(*, host: str, timeout_s: float = 5.0) -> list[str]:
    url = urljoin(host.rstrip("/") + "/", "api/tags")
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            obj = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise OllamaError(f"Failed to reach Ollama at {url} (is `ollama serve` running?): {e}") from e
    return sorted(str(m["name"]) for m in obj.get("models", []) if m.get("name"))


def chat_json(
    *,
    host: str,
    model: str,
    system: str,
    user: str,
    response_format: ResponseFormat = "json",
    temperature: float = 0.2,
    seed: int | None = None,
    timeout_s: float = 120.0,
) -> OllamaResponse:
    start = time.time()
    url = urljoin(host.rstrip("/") + "/", "api/chat")
    options: dict[str, Any] = {"temperature": temperature}
    if seed is not None:
        options["seed"] = seed

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": response_format,
        "options": options,
    }
    status, raw = _http_post_json(url, payload, timeout_s=timeout_s)
    try:
        obj = json.loads(raw)
        content = obj.get("message", {}).get("content", "")
        model_name = obj.get("model")
    except Exception as exc:  # noqa: BLE001
        raise OllamaError(f"Invalid JSON from Ollama (status {status})", raw_response=raw) from exc

    latency_ms = int((time.time() - start) * 1000)
    return OllamaResponse(content=content, model=model_name, latency_ms=latency_ms, raw=raw)


def generate_json(
    *,
    host: str,
    model: str,
    prompt: str,
    response_format: ResponseFormat = "json",
    temperature: float = 0.2,
    seed: int | None = None,
    timeout_s: float = 120.0,
) -> OllamaResponse:
    start = time.time()
    url = urljoin(host.rstrip("/") + "/", "api/generate")
    options: dict[str, Any] = {"temperature": temperature}
    if seed is not None:
        options["seed"] = seed

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": response_format,
        "options": options,
    }
    status, raw = _http_post_json(url, payload, timeout_s=timeout_s)
    try:
        obj = json.loads(raw)
        content = obj.get("response", "")
        model_name = obj.get("model")
    except Exception as exc:  # noqa: BLE001
        raise OllamaError(f"Invalid JSON from Ollama (status {status})", raw_response=raw) from exc

    latency_ms = int((time.time() - start) * 1000)
    return OllamaResponse(content=content, model=model_name, latency_ms=latency_ms, raw=raw)


def chat_text(
    *,
    host: str,
    model: str,
    user: str,
    options: dict[str, Any] | None = None,
    timeout_s: float = 300.0,
) -> OllamaResponse:
    """Single-turn chat with no system prompt or output format, for translation-only models."""
    start = time.time()
    url = urljoin(host.rstrip("/") + "/", "api/chat")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user}],
        "stream": False,
        "options": options or {},
    }
    status, raw = _http_post_json(url, payload, timeout_s=timeout_s)
    try:
        obj = json.loads(raw)
        content = obj.get("message", {}).get("content", "")
        model_name = obj.get("model")
    except Exception as exc:  # noqa: BLE001
        raise OllamaError(f"Invalid JSON from Ollama (status {status})", raw_response=raw) from exc

    latency_ms = int((time.time() - start) * 1000)
    return OllamaResponse(content=content, model=model_name, latency_ms=latency_ms, raw=raw)
