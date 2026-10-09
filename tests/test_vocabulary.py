import contextlib
import io
import json
import unittest
from dataclasses import asdict, replace
from unittest.mock import MagicMock, patch

import translate
import webui
from translator_app import core
from translator_app.config import AppConfig, OllamaConfig, ProviderConfig, TransformersConfig
from translator_app.models import TranslateRequest, RelativesResult
from translator_app.ollama import OllamaResponse
from translator_app.prompting import build_system_prompt, build_user_prompt
from translator_app.vocabulary import validate_relatives_text


class VocabularyTests(unittest.TestCase):
    def request(self, **kwargs):
        return replace(TranslateRequest("create", "EN", "ZH", mode="relatives", spellcheck=False), **kwargs)

    def relatives(self):
        return {
            "term": "create",
            "derivations": [{"term": "creation", "pos": "noun", "meaning": "创造"},
                            {"term": "creative", "pos": "adjective", "meaning": "有创造力的"},
                            {"term": "creatively", "pos": "adverb", "meaning": "创造性地"}],
            "synonyms": [{"term": "make", "pos": "verb", "meaning": "制作"}],
            "antonyms": [{"term": "destroy", "pos": "verb", "meaning": "毁坏"}], "notes": None,
        }

    def dictionary(self):
        return {"term": "schedule", "entries": [{"pos": "noun", "senses": [{"meaning": "日程"}]}],
                "pronunciations": [{"label": "UK", "ipa": "/ˈʃedjuːl/"}, {"label": "US", "ipa": "/ˈskedʒuːl/"}]}

    def test_short_input_limits_cover_unicode_phrases_and_paragraphs(self):
        for term in ("creative", "break the ice", "well-being", "don't", "打ち合わせ", "创造力", "one two three four five"):
            with self.subTest(term=term):
                validate_relatives_text(term)
        for term in ("", "1234", "a" * 61, "one two three four five six", "hello\nworld", "This is a sentence.", "这是一句话。"):
            with self.subTest(term=term), self.assertRaisesRegex(ValueError, "short phrases"):
                validate_relatives_text(term)

    def test_invalid_input_fails_before_any_model_request(self):
        with patch.object(core, "_translate") as provider:
            with self.assertRaisesRegex(ValueError, "short phrases"):
                core.translate_text(self.request(text="too many words to be a short phrase"), config=AppConfig())
        provider.assert_not_called()
        with self.assertRaisesRegex(ValueError, "short phrases"):
            webui._request_from_body({"mode": "relatives", "text": "a" * 61}, AppConfig())

    def test_relatives_reaches_every_provider_with_model_fallback_and_budget(self):
        for provider, patch_name in (("ollama", "chat_json"), ("transformers", "hf_chat_json"),
                                     ("gemini", "cloud_chat_json"), ("openai", "cloud_chat_json")):
            config = replace(AppConfig(), provider=ProviderConfig(provider),
                             ollama=OllamaConfig(model="translategemma:4b", dictionary_model="general"),
                             transformers=TransformersConfig(model="tencent/HY-MT1.5-7B", dictionary_model="general"))
            response = OllamaResponse(json.dumps(self.relatives(), ensure_ascii=False), "general", 3, "")
            with self.subTest(provider=provider), patch.object(core, patch_name, return_value=response) as api:
                result = core.translate_text(self.request(), config=config)
            api.assert_called_once()
            self.assertIsInstance(result, RelativesResult)
            self.assertEqual(result.provider, provider)
            self.assertEqual(result.derivations[2].pos, "adverb")
            self.assertEqual(result.synonyms[0].term, "make")
            self.assertEqual(result.antonyms[0].term, "destroy")
            if provider == "ollama":
                self.assertEqual(api.call_args.kwargs["model"], "general")
                self.assertEqual(api.call_args.kwargs["options"]["num_predict"], config.enhancement.max_new_tokens)
            elif provider == "transformers":
                self.assertEqual(api.call_args.kwargs["config"].model, "general")
                self.assertEqual(api.call_args.kwargs["config"].max_new_tokens, config.enhancement.max_new_tokens)

    def test_relatives_parsing_accepts_absent_relationships_and_rejects_bad_shapes(self):
        result = core._as_relatives_result(self.request(), {
            "term": "unique", "derivations": [], "synonyms": [], "antonyms": [], "notes": "No established antonym.",
        }, provider="gemini", model="model", latency_ms=1)
        self.assertEqual(result.antonyms, [])
        self.assertIn("antonym", result.notes)
        for changes in ({"synonyms": "bad"}, {"derivations": [{}]}, {"antonyms": [{"term": "no", "pos": None, "meaning": "无"}]},
                        {"notes": []}, {"term": 123}):
            with self.subTest(changes=changes), self.assertRaises(core.ProviderResponseParseError):
                core._as_relatives_result(self.request(), {**self.relatives(), **changes}, provider="ollama", model="m", latency_ms=1)

    def test_dictionary_ipa_survives_parsing_and_legacy_results(self):
        request = self.request(mode="dictionary", text="schedule")
        result = core._as_dictionary_result(request, self.dictionary(), provider="gemini", model="m", latency_ms=1)
        self.assertEqual(asdict(result)["pronunciations"][1], {"label": "US", "ipa": "/ˈskedʒuːl/"})
        older = self.dictionary()
        del older["pronunciations"]
        result = core._as_dictionary_result(request, older, provider="ollama", model="m", latency_ms=1)
        self.assertEqual(result.pronunciations, [])
        for value in (None, "schedule", [{}], [{"label": "UK", "ipa": None}]):
            with self.subTest(value=value), self.assertRaises(core.ProviderResponseParseError):
                core._as_dictionary_result(request, {**older, "pronunciations": value}, provider="ollama", model="m", latency_ms=1)

    def test_relatives_uses_input_language_and_ignores_translation_language_settings(self):
        request = self.request()
        self.assertIn("SOURCE language", build_system_prompt(request))
        self.assertIn("Do not invent word forms", build_system_prompt(request))
        self.assertIn("Detect the term's language", build_user_prompt(request))
        self.assertEqual(build_user_prompt(request), build_user_prompt(replace(request, source_lang="JA", target_lang="FR")))
        self.assertIn("same language", build_system_prompt(request))
        dictionary_prompt = build_system_prompt(replace(request, mode="dictionary"))
        self.assertIn("International Phonetic Alphabet", dictionary_prompt)
        self.assertIn("UK and US", dictionary_prompt)
        schema = core._response_schema_for(replace(request, mode="dictionary"))
        self.assertIn("pronunciations", schema["required"])
        relatives_schema = core._response_schema_for(request)
        self.assertEqual(set(relatives_schema["required"]), {"term", "derivations", "synonyms", "antonyms", "notes"})

    def test_web_relatives_validation_and_metadata(self):
        handler_type = webui.make_handler(AppConfig())
        for text, expected_status in (("create", 200), ("This sentence should not become vocabulary input.", 400)):
            body = json.dumps({"mode": "relatives", "text": text, "provider": "gemini", "target_lang": "JA", "spellcheck": False}).encode()
            handler = object.__new__(handler_type)
            handler.path = "/api/translate"
            handler.headers = {"Content-Length": str(len(body))}
            handler.rfile = io.BytesIO(body)
            handler._send_json = MagicMock()
            with patch.object(core, "cloud_chat_json", return_value=OllamaResponse(json.dumps(self.relatives()), "m", 1, "")) as api:
                handler.do_POST()
            status, payload = handler._send_json.call_args.args
            self.assertEqual(status, expected_status)
            if status == 200:
                self.assertEqual(payload["mode"], "relatives")
                self.assertNotIn("target_lang", payload)
                api.assert_called_once()
            else:
                api.assert_not_called()
        request = webui._request_from_body({"mode": "relatives", "text": "jump", "source_lang": "JA"}, AppConfig())
        self.assertEqual(request.source_lang, "auto")

    def test_cli_prints_related_groups_and_ipa(self):
        for mode, obj, expected in (("relatives", self.relatives(), "creatively\tadverb\t创造性地"),
                                    ("dictionary", self.dictionary(), "UK IPA: /ˈʃedjuːl/")):
            with patch.object(core, "cloud_chat_json", return_value=OllamaResponse(json.dumps(obj), "m", 1, "")), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                status = translate.main(["create", "--mode", mode, "--provider", "gemini", "--from", "EN", "--no-spellcheck"])
            self.assertEqual(status, 0)
            self.assertIn(expected, output.getvalue())


if __name__ == "__main__":
    unittest.main()
