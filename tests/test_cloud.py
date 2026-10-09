import contextlib
import io
import json
import os
import tempfile
import unittest
import urllib.error
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import translate
import webui
from translator_app import cloud, core
from translator_app.config import AppConfig, ProviderConfig, load_config
from translator_app.models import TranslateRequest


class CloudTests(unittest.TestCase):
    def request(self, mode="translate", **kwargs):
        return TranslateRequest(text="hello", source_lang="EN", target_lang="JA", mode=mode,
                                spellcheck=False, **kwargs)

    def completion(self, content='{"translation":"こんにちは"}', **choice_fields):
        choice = {"message": {"content": content}, "finish_reason": "stop", **choice_fields}
        return json.dumps({"model": "returned-model", "choices": [choice]}).encode()

    def invoke(self, provider="openai", **overrides):
        config = getattr(AppConfig(), provider)
        return cloud.chat_json(provider=provider, config=config, model=config.model,
                               system="system", user="hello", response_schema={"type": "object"},
                               temperature=0.2, **overrides)

    def mock_opener(self, raw):
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value.read.return_value = raw
        return patch.object(cloud.urllib.request, "build_opener", return_value=opener), opener

    def test_provider_payload_authentication_and_response(self):
        for provider in ("openai", "gemini"):
            mock, opener = self.mock_opener(self.completion())
            with self.subTest(provider=provider), mock, patch.dict(os.environ, {
                "OPENAI_API_KEY": "secret-openai", "GEMINI_API_KEY": "secret-gemini",
            }, clear=True):
                result = self.invoke(provider)
            req = opener.open.call_args.args[0]
            payload = json.loads(req.data)
            self.assertEqual(req.full_url, cloud.ENDPOINTS[provider])
            self.assertEqual(req.get_header("Authorization"), f"Bearer secret-{provider}")
            self.assertNotIn("secret", req.full_url + req.data.decode())
            self.assertEqual(payload["response_format"]["type"], "json_schema")
            self.assertTrue(payload["response_format"]["json_schema"]["strict"])
            self.assertEqual(payload["messages"][1]["content"], "hello")
            self.assertEqual(result.model, "returned-model")
            self.assertGreaterEqual(result.latency_ms, 0)
            if provider == "openai":
                self.assertEqual(payload["max_completion_tokens"], 4096)
                self.assertEqual(payload["reasoning_effort"], "none")
                self.assertIs(payload["store"], False)
                self.assertNotIn("temperature", payload)
                self.assertNotIn("max_tokens", payload)
            else:
                self.assertEqual(payload["max_tokens"], 4096)
                self.assertEqual(payload["reasoning_effort"], "minimal")
                self.assertEqual(payload["temperature"], 0.2)
                self.assertNotIn("store", payload)
            opener.open.assert_called_once()

    def test_missing_key_fails_before_network(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(cloud.urllib.request, "build_opener") as build:
            for provider in ("openai", "gemini"):
                with self.assertRaisesRegex(cloud.CloudError, getattr(AppConfig(), provider).api_key_env):
                    self.invoke(provider)
        build.assert_not_called()

    def test_all_modes_route_to_cloud_without_local_inference(self):
        outputs = {
            "translate": {"translation": "こんにちは"},
            "dictionary": {"term": "hello", "entries": [{"pos": "interjection", "senses": [{"meaning": "挨拶"}]}]},
            "enhance": {"enhanced_prompt": "Write a greeting.", "clarifications": []},
        }
        for provider in ("gemini", "openai"):
            config = replace(AppConfig(), provider=ProviderConfig(provider))
            for mode, output in outputs.items():
                with self.subTest(provider=provider, mode=mode), \
                        patch.object(core, "cloud_chat_json", return_value=cloud.CloudResponse(json.dumps(output), "custom", 7)) as api, \
                        patch.object(core, "chat_json") as ollama, patch.object(core, "hf_chat_json") as hf:
                    result = core.translate_text(self.request(mode, model="custom"), config=config)
                self.assertEqual(result.provider, provider)
                self.assertEqual(result.model, "custom")
                self.assertEqual(result.latency_ms, 7)
                api.assert_called_once()
                self.assertEqual(api.call_args.kwargs["model"], "custom")
                self.assertEqual(api.call_args.kwargs["response_schema"], core._response_schema_for(self.request(mode)))
                self.assertEqual(api.call_args.kwargs["config"].max_output_tokens, 4096)
                ollama.assert_not_called()
                hf.assert_not_called()

    def test_parse_failure_does_not_trigger_billable_repair(self):
        config = replace(AppConfig(), provider=ProviderConfig("openai"))
        with patch.object(core, "cloud_chat_json", return_value=cloud.CloudResponse("invalid", "model", 0)) as api:
            with self.assertRaises(core.ProviderResponseParseError):
                core.translate_text(self.request(), config=config)
        api.assert_called_once()

    def test_http_errors_are_actionable_and_do_not_expose_body(self):
        for code, expected in ((400, "reasoning_effort"), (401, "API key"), (403, "permissions"),
                               (404, "model ID"), (429, "quota"), (500, "retry later"), (302, "failed")):
            mock, opener = self.mock_opener(b"")
            opener.open.side_effect = urllib.error.HTTPError("https://example.com", code, "secret-reason", {},
                                                            io.BytesIO(b"secret-key private-text"))
            with self.subTest(code=code), mock, patch.dict(os.environ, {"OPENAI_API_KEY": "secret-key"}):
                with self.assertRaisesRegex(cloud.CloudError, expected) as raised:
                    self.invoke()
            self.assertNotIn("secret", str(raised.exception))
            self.assertNotIn("private-text", str(raised.exception))
            opener.open.assert_called_once()

    def test_timeout_and_network_failure_are_sanitized(self):
        for exc in (TimeoutError("secret"), urllib.error.URLError("secret")):
            mock, opener = self.mock_opener(b"")
            opener.open.side_effect = exc
            with mock, patch.dict(os.environ, {"OPENAI_API_KEY": "secret"}):
                with self.assertRaisesRegex(cloud.CloudError, "connection") as raised:
                    self.invoke()
            self.assertNotIn("secret", str(raised.exception))

    def test_refusal_truncation_and_malformed_responses(self):
        cases = [
            (self.completion(finish_reason="length"), "truncated"),
            (self.completion(finish_reason="content_filter"), "declined"),
            (self.completion(message={"refusal": "Cannot comply", "content": None}), "declined"),
            (self.completion(content=""), "no text"),
            (b"not-json", "invalid"), (b'{"choices": []}', "invalid"),
            (b'{"choices": [null]}', "invalid"), (b"null", "invalid"),
            (b"\xff", "encoding"),
        ]
        for raw, expected in cases:
            mock, opener = self.mock_opener(raw)
            with self.subTest(expected=expected, raw=raw), mock, patch.dict(os.environ, {"OPENAI_API_KEY": "secret"}):
                with self.assertRaisesRegex(cloud.CloudError, expected):
                    self.invoke()
            opener.open.assert_called_once()

    def test_redirect_is_never_followed(self):
        self.assertIsNone(cloud._NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.com"))

    def test_config_loading_and_legacy_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.toml"
            path.write_text('[provider]\nname="gemini"\n[gemini]\nmodel="custom"\nmodels=["extra"]\n'
                            'api_key_env="CUSTOM_KEY"\nmax_output_tokens=1234\ntimeout_s=30\nreasoning_effort=""\n')
            config = load_config(path)
            self.assertEqual(config.gemini.model, "custom")
            self.assertEqual(config.gemini.models, ("extra",))
            self.assertEqual(config.gemini.max_output_tokens, 1234)
            self.assertEqual(config.gemini.api_key_env, "CUSTOM_KEY")
            self.assertEqual(config.openai.model, "gpt-6-luna")
            self.assertEqual(config.ollama, AppConfig().ollama)
            for setting in ('max_output_tokens=0', 'max_output_tokens=true', 'timeout_s=nan',
                            'timeout_s=0', 'model=""', 'models="bad"', 'reasoning_effort="bad"'):
                path.write_text(f"[openai]\n{setting}\n")
                with self.subTest(setting=setting), self.assertRaises(ValueError):
                    load_config(path)

    def test_web_picker_keys_stay_server_side(self):
        config = replace(AppConfig(), provider=ProviderConfig("gemini"))
        config = replace(config, gemini=replace(config.gemini, api_key_env="CUSTOM_KEY", models=("extra",)))
        with patch.dict(os.environ, {"CUSTOM_KEY": "secret-value"}, clear=True), \
                patch.object(webui, "list_models", return_value=[]), \
                patch.object(webui, "dependencies_available", return_value=False):
            info = webui._app_info(config)
        self.assertNotIn("secret-value", json.dumps(info))
        gemini = [m for m in info["models"] if m["provider"] == "gemini"]
        self.assertEqual(len(gemini), 2)
        self.assertTrue(all(m["available"] and not m["translation_only"] for m in gemini))
        openai = next(m for m in info["models"] if m["provider"] == "openai")
        self.assertFalse(openai["available"])
        self.assertEqual(openai["reason"], "API KEY not set")
        self.assertEqual(info["default_model"], {"provider": "gemini", "name": config.gemini.model})

    def test_cli_cloud_selection_and_missing_key_error(self):
        with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stderr(io.StringIO()) as stderr:
            status = translate.main(["hello", "--provider", "gemini", "--from", "EN", "--no-spellcheck"])
        self.assertEqual(status, 1)
        self.assertIn("GEMINI_API_KEY", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_web_post_routes_cloud_and_returns_api_errors(self):
        handler_type = webui.make_handler(AppConfig())
        for provider in ("gemini", "openai"):
            for failure in (False, True):
                body = json.dumps({"text": "draft", "mode": "enhance", "provider": provider}).encode()
                handler = object.__new__(handler_type)
                handler.path = "/api/translate"
                handler.headers = {"Content-Length": str(len(body))}
                handler.rfile = io.BytesIO(body)
                handler._send_json = MagicMock()
                with self.subTest(provider=provider, failure=failure), patch.object(core, "cloud_chat_json") as api:
                    if failure:
                        api.side_effect = cloud.CloudError("Rate limit or quota exceeded")
                    else:
                        api.return_value = cloud.CloudResponse('{"enhanced_prompt":"Better draft","clarifications":[]}', "model", 1)
                    handler.do_POST()
                api.assert_called_once()
                self.assertEqual(api.call_args.kwargs["provider"], provider)
                status, response = handler._send_json.call_args.args
                self.assertEqual(status, 502 if failure else 200)
                if failure:
                    self.assertIn("quota", response["error"])
                else:
                    self.assertEqual(response["enhanced_prompt"], "Better draft")
                    self.assertEqual(response["provider"], provider)


if __name__ == "__main__":
    unittest.main()
