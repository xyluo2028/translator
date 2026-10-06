import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from translator_app import core, ollama
from translator_app.config import AppConfig, OllamaConfig
from translator_app.models import TranslateRequest


class OllamaOptionsTests(unittest.TestCase):
    def test_gpu_and_thinking_controls_reach_chat_and_generate(self):
        options = {"num_gpu": 999, "num_ctx": 2048, "temperature": 1.0}
        raw = json.dumps({"model": "gemma", "message": {"content": "{}"}, "response": "{}"})
        for function, messages in (
            (ollama.chat_json, {"system": "system", "user": "user"}),
            (ollama.generate_json, {"prompt": "prompt"}),
        ):
            with self.subTest(function=function.__name__), \
                    patch.object(ollama, "_http_post_json", return_value=(200, raw)) as post:
                function(host="http://localhost:11435", model="gemma", options=options,
                         enable_thinking=False, temperature=0.2, seed=42, **messages)
                payload = post.call_args.args[1]
                self.assertEqual(payload["options"], {"num_gpu": 999, "num_ctx": 2048, "temperature": 0.2, "seed": 42})
                self.assertIs(payload["think"], False)
        self.assertEqual(options["temperature"], 1.0, "Request controls mutated shared configuration")

    def test_legacy_configuration_omits_thinking_control(self):
        raw = json.dumps({"message": {"content": "{}"}})
        with patch.object(ollama, "_http_post_json", return_value=(200, raw)) as post:
            ollama.chat_json(host="http://localhost:11434", model="model", system="system", user="user")
        self.assertNotIn("think", post.call_args.args[1])

    def test_dictionary_retries_preserve_gpu_controls(self):
        config = replace(AppConfig(), ollama=OllamaConfig(
            model="gemma", dictionary_model="gemma", options={"num_gpu": 999}, enable_thinking=False,
        ))
        good = ollama.OllamaResponse(
            content='{"term":"book","entries":[{"pos":"noun","senses":[{"meaning":"书"}]}]}',
            model="gemma", latency_ms=1, raw="",
        )
        invalid = replace(good, content="invalid JSON")
        with patch.object(core, "chat_json", side_effect=[ollama.OllamaError("schema unsupported"), invalid]) as chat, \
                patch.object(core, "generate_json", return_value=good) as generate:
            result = core.translate_text(TranslateRequest(
                text="book", source_lang="EN", target_lang="ZH", mode="dictionary", spellcheck=False,
            ), config=config)
        self.assertEqual(result.term, "book")
        for call in [*chat.call_args_list, *generate.call_args_list]:
            self.assertEqual(call.kwargs["options"], {"num_gpu": 999})
            self.assertIs(call.kwargs["enable_thinking"], False)

    def test_json_profile_completes_dictionary_without_schema_or_repair(self):
        config = replace(AppConfig(), ollama=OllamaConfig(model="gemma", structured_output=False))
        response = ollama.OllamaResponse(
            content='{"term":"book","entries":[{"pos":"noun","senses":[{"meaning":"书"}]}]}',
            model="gemma", latency_ms=1, raw="",
        )
        with patch.object(core, "chat_json", return_value=response) as chat, \
                patch.object(core, "generate_json") as repair:
            result = core.translate_text(TranslateRequest(
                text="book", source_lang="EN", target_lang="ZH", mode="dictionary", spellcheck=False,
            ), config=config)
        self.assertEqual(result.term, "book")
        chat.assert_called_once()
        self.assertEqual(chat.call_args.kwargs["response_format"], "json")
        repair.assert_not_called()

    def test_generate_profile_completes_dictionary_in_one_request(self):
        config = replace(AppConfig(), ollama=OllamaConfig(model="gemma", structured_output=False, api="generate"))
        response = ollama.OllamaResponse(
            content='{"term":"book","entries":[{"pos":"noun","senses":[{"meaning":"书"}]}]}',
            model="gemma", latency_ms=1, raw="",
        )
        with patch.object(core, "generate_json", return_value=response) as generate, \
                patch.object(core, "chat_json") as chat:
            result = core.translate_text(TranslateRequest(
                text="book", source_lang="EN", target_lang="ZH", mode="dictionary", spellcheck=False,
            ), config=config)
        self.assertEqual(result.term, "book")
        generate.assert_called_once()
        self.assertEqual(generate.call_args.kwargs["response_format"], "json")
        self.assertIn("book", generate.call_args.kwargs["prompt"])
        chat.assert_not_called()


if __name__ == "__main__":
    unittest.main()
