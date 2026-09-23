#!/usr/bin/env python3
"""Local web UI for the translator: `.venv/bin/python webui.py --open` (or plain `python3` for Ollama only)."""
from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from translator_app.config import AppConfig, ProviderConfig, load_config
from translator_app.core import ProviderResponseParseError, translate_text
from translator_app.hf_transformers import TransformersError, dependencies_available, is_model_cached
from translator_app.models import RerunHint, TranslateRequest
from translator_app.ollama import OllamaError, list_models
from translator_app.prompting import LANGUAGES, prompt_style_for

INDEX_HTML = Path(__file__).resolve().parent / "translator_app" / "web" / "index.html"
MAX_TEXT_CHARS = 20_000


def _app_info(config: AppConfig) -> dict[str, Any]:
    """Models from both backends. The UI sends back (provider, model) and the server routes accordingly."""
    info: dict[str, Any] = {
        "provider": config.provider.name,
        "defaults": asdict(config.defaults),
        "languages": [{"code": code, "name": names[0]} for code, names in LANGUAGES.items()],
        "models": [],
        "default_model": {"provider": config.provider.name, "name": _default_model(config)},
        "dictionary_models": {
            "ollama": config.ollama.dictionary_model,
            "transformers": config.transformers.dictionary_model,
        },
        "errors": [],
    }

    try:
        ollama_names = list_models(host=config.ollama.host)
    except OllamaError as exc:
        ollama_names = []
        info["errors"].append(str(exc))
    for name in dict.fromkeys([*ollama_names, config.ollama.model]):
        info["models"].append(_model_entry("ollama", name, available=name in ollama_names, reason="not pulled"))

    hf_ready = dependencies_available()
    hf_names = [config.transformers.model, *config.transformers.models, config.transformers.dictionary_model]
    for name in dict.fromkeys(hf_names):
        if hf_ready:
            entry = _model_entry("transformers", name, available=is_model_cached(name), reason="not downloaded")
        else:
            entry = _model_entry("transformers", name, available=False, reason="run with .venv/bin/python")
        info["models"].append(entry)

    info["models"].sort(key=lambda m: (m["provider"] != "ollama", not m["translation_only"], m["name"]))
    return info


def _default_model(config: AppConfig) -> str:
    return config.transformers.model if config.provider.name == "transformers" else config.ollama.model


def _model_entry(provider: str, name: str, *, available: bool, reason: str) -> dict[str, Any]:
    return {
        "provider": provider,
        "name": name,
        "translation_only": prompt_style_for(name) != "json",
        "available": available,
        "reason": None if available else reason,
    }


def _request_from_body(body: dict[str, Any], config: AppConfig) -> TranslateRequest:
    text = str(body.get("text") or "")
    if not text.strip():
        raise ValueError("Enter some text to translate.")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"Text is too long ({len(text)} characters, max {MAX_TEXT_CHARS}).")
    mode = body.get("mode") or "translate"
    if mode not in ("translate", "dictionary"):
        raise ValueError(f"Unknown mode: {mode!r}")
    rerun = body.get("rerun")
    if rerun not in (None, "", "retry", "more_literal", "more_natural"):
        raise ValueError(f"Unknown rerun style: {rerun!r}")
    return TranslateRequest(
        text=text,
        source_lang=str(body.get("source_lang") or config.defaults.source_lang),
        target_lang=str(body.get("target_lang") or config.defaults.target_lang),
        mode=mode,
        tone=str(body.get("tone") or config.defaults.tone),
        tone_instructions=(str(body["tone_instructions"]) if body.get("tone_instructions") else None),
        explain_lang=str(body.get("explain_lang") or config.defaults.explain_lang),
        rerun=RerunHint(style=rerun) if rerun else None,
        temperature=config.defaults.temperature,
        model=(str(body["model"]) if body.get("model") else None),
    )


def make_handler(config: AppConfig) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "TranslatorWebUI"

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write(f"[webui] {fmt % args}\n")

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, obj: Any) -> None:
            self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                self._send(200, INDEX_HTML.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/info":
                self._send_json(200, _app_info(config))
            else:
                self._send_json(404, {"error": "Not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/translate":
                self._send_json(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object.")
                request = _request_from_body(body, config)
            except (ValueError, json.JSONDecodeError) as exc:
                self._send_json(400, {"error": str(exc)})
                return
            provider = body.get("provider") or config.provider.name
            if provider not in ("ollama", "transformers"):
                self._send_json(400, {"error": f"Unknown provider: {provider!r}"})
                return
            try:
                result = translate_text(request, config=replace(config, provider=ProviderConfig(name=provider)))
            except (OllamaError, TransformersError, ProviderResponseParseError, ValueError) as exc:
                self._send_json(502, {"error": str(exc)})
                return
            payload = asdict(result)
            payload["mode"] = request.mode
            self._send_json(200, payload)

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local web UI for the LLM translator.")
    parser.add_argument("--config", default="config.toml", help="Path to config TOML.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (keep 127.0.0.1 to stay local-only).")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the UI in your browser.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(config))
    url = f"http://{args.host}:{args.port}"
    hf = "available" if dependencies_available() else "unavailable (run with .venv/bin/python)"
    print(f"Translator UI running at {url}  (default: {config.provider.name}, Hugging Face: {hf}; Ctrl+C to stop)")
    if args.open:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
