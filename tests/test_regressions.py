import random
import importlib.util
import sys
import threading
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from translator_app import core, hf_transformers as hf, spelling
from translator_app.config import AppConfig, TransformersConfig
from translator_app.models import DictionaryResult, TranslateRequest, TranslateResult
from translator_app.prompting import build_user_prompt


class SpellingTests(unittest.TestCase):
    def test_auto_preserves_foreign_words_before_provider_call(self):
        for mode in ("translate", "dictionary"):
            for text in ("hola", "merci", "salut", "hola hola", "Je suis ici", "tyype"):
                with self.subTest(mode=mode, text=text):
                    request = TranslateRequest(text=text, source_lang="auto", target_lang="EN", mode=mode)
                    with patch.object(core, "_translate", return_value=TranslateResult("result")) as provider:
                        core.translate_text(request, config=AppConfig())
                    self.assertEqual(provider.call_args.args[0].text, text)

    @unittest.skipUnless(
        spelling.available() and importlib.util.find_spec("lingua"), "requires the updated text extra"
    )
    def test_auto_detects_language_and_corrects_before_translation_or_dictionary(self):
        cases = (
            ("Where is teh station?", "EN", "Where is the station?"),
            ("Bonjor tout le monde", "FR", "Bonjour tout le monde"),
            ("Das ist ein gutes beispil", "DE", "Das ist ein gutes beispiel"),
        )
        for mode in ("translate", "dictionary"):
            for text, lang, corrected in cases:
                with self.subTest(mode=mode, lang=lang):
                    self.assertEqual(spelling.check(text, source_lang="auto").corrected, corrected)
                    request = TranslateRequest(text=text, source_lang="auto", target_lang="JA", mode=mode)
                    response = DictionaryResult(corrected, []) if mode == "dictionary" else TranslateResult("result")
                    with patch.object(core, "_translate", return_value=response) as provider:
                        result = core.translate_text(request, config=AppConfig())
                    sent = provider.call_args.args[0]
                    self.assertEqual((sent.text, sent.source_lang), (corrected, lang))
                    self.assertEqual(result.spelling.original, text)
                    if mode == "translate":
                        self.assertEqual(result.detected_source_lang, lang)

    def test_uncertain_detection_skips_spelling_and_preserves_auto(self):
        request = TranslateRequest(text="ambiguous", source_lang="auto", target_lang="JA")
        with patch.object(spelling, "detect_source_language", return_value=None), \
                patch.object(spelling, "check") as check, \
                patch.object(core, "_translate", return_value=TranslateResult("result")) as provider:
            core.translate_text(request, config=AppConfig())
        check.assert_not_called()
        self.assertEqual(provider.call_args.args[0], request)

    def test_detected_language_is_used_without_spelling_edits(self):
        for enabled in (True, False):
            with self.subTest(spellcheck=enabled):
                request = TranslateRequest(text="Bonjour", source_lang="auto", target_lang="JA", spellcheck=enabled)
                with patch.object(spelling, "detect_source_language", return_value="FR"), \
                        patch.object(spelling, "check", return_value=None) as check, \
                        patch.object(core, "_translate", return_value=TranslateResult("result")) as provider:
                    result = core.translate_text(request, config=AppConfig())
                self.assertEqual(provider.call_args.args[0].source_lang, "FR")
                self.assertEqual(provider.call_args.args[0].text, request.text)
                self.assertEqual(result.detected_source_lang, "FR")
                self.assertEqual(check.call_count, int(enabled))

    def test_explicit_source_bypasses_detection(self):
        request = TranslateRequest(text="hola", source_lang="ES", target_lang="JA", spellcheck=False)
        with patch.object(spelling, "detect_source_language") as detect, \
                patch.object(core, "_translate", return_value=TranslateResult("result")):
            core.translate_text(request, config=AppConfig())
        detect.assert_not_called()

    @unittest.skipUnless(spelling.available(), "requires the optional text extra")
    def test_explicit_language_still_corrects_and_records_exact_positions(self):
        text = "😀 The echelon helo helo"
        fix = spelling.check(text, source_lang="EN")
        self.assertIsNotNone(fix)
        self.assertEqual([c.word for c in fix.corrections], ["helo", "helo"])
        self.assertEqual([(c.start, c.end) for c in fix.corrections], [(14, 18), (19, 23)])
        corrected = text
        for correction in reversed(fix.corrections):
            self.assertEqual(text[correction.start:correction.end], correction.word)
            corrected = corrected[:correction.start] + correction.suggestion + corrected[correction.end:]
        self.assertEqual(corrected, fix.corrected)
        self.assertTrue(fix.corrected.startswith("😀 The echelon "))

    @unittest.skipUnless(spelling.available(), "requires the optional text extra")
    def test_explicit_source_and_opt_out_reach_provider(self):
        request = TranslateRequest(text="Where is teh station?", source_lang="EN", target_lang="JA")
        with patch.object(core, "_translate", return_value=TranslateResult("result")) as provider:
            result = core.translate_text(request, config=AppConfig())
            self.assertEqual(provider.call_args.args[0].text, "Where is the station?")
            self.assertEqual(result.spelling.original, request.text)
            core.translate_text(replace(request, spellcheck=False), config=AppConfig())
            self.assertEqual(provider.call_args.args[0].text, request.text)

    def test_dictionary_prompt_respects_selected_source_language(self):
        request = TranslateRequest(text="gift", source_lang="DE", target_lang="EN", mode="dictionary")
        self.assertIn("Source language: DE.", build_user_prompt(request))


class LanguageDetectionTests(unittest.TestCase):
    def test_low_confidence_and_close_scores_are_declined(self):
        def score(code, value):
            return SimpleNamespace(language=SimpleNamespace(iso_code_639_1=SimpleNamespace(name=code)), value=value)

        for best, second, expected in ((0.3, 0.05, None), (0.6, 0.5, None), (0.7, 0.1, "FR")):
            with self.subTest(best=best, second=second):
                detector = SimpleNamespace(compute_language_confidence_values=lambda _: [score("FR", best), score("EN", second)])
                with patch.object(spelling, "_detector", return_value=detector):
                    self.assertEqual(spelling.detect_source_language("input"), expected)

    def test_missing_detector_preserves_input(self):
        with patch.object(spelling, "_detector", side_effect=ImportError):
            self.assertIsNone(spelling.detect_source_language("hola"))
            self.assertIsNone(spelling.check("hola", source_lang="auto"))

    def test_detected_language_without_dictionary_is_not_corrected(self):
        with patch.object(spelling, "detect_source_language", return_value="JA"), \
                patch.object(spelling, "_checker") as checker:
            self.assertIsNone(spelling.check("これは teh です", source_lang="auto"))
        checker.assert_not_called()


class GenerationTests(unittest.TestCase):
    def test_waiting_request_cannot_reseed_active_generation(self):
        active = threading.Event()
        waiting = threading.Event()
        lock = threading.Lock()
        rng = random.Random()
        observed = {}
        errors = []

        class ObservedLock:
            def __enter__(self):
                if threading.current_thread().name == "second":
                    waiting.set()
                lock.acquire()

            def __exit__(self, *args):
                lock.release()

        def generate(**kwargs):
            name = kwargs["name"]
            if name == "first":
                active.set()
                if not waiting.wait(5):
                    raise AssertionError("Second request did not reach the generation lock")
            observed[name] = rng.random()
            return [[0, 1]]

        loaded = hf._LoadedModel(
            processor=SimpleNamespace(decode=lambda *args, **kwargs: "result"),
            model=SimpleNamespace(generate=generate, config=SimpleNamespace(eos_token_id=0)),
        )

        def run(name, seed):
            try:
                hf._generate(
                    loaded,
                    {"input_ids": SimpleNamespace(shape=(1, 1)), "name": name},
                    config=TransformersConfig(), sampling={"temperature": 0.2}, seed=seed,
                )
            except Exception as exc:
                errors.append(exc)

        with patch.dict(sys.modules, {"torch": SimpleNamespace(manual_seed=rng.seed)}), \
                patch.object(hf, "_GENERATE_LOCK", ObservedLock()):
            first = threading.Thread(target=run, args=("first", 1), name="first")
            second = threading.Thread(target=run, args=("second", 2), name="second")
            first.start()
            try:
                self.assertTrue(active.wait(5), "First request did not start generating")
                second.start()
            finally:
                first.join(10)
                if second.ident is not None:
                    second.join(10)
            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(observed["first"], random.Random(1).random())
        self.assertEqual(observed["second"], random.Random(2).random())


if __name__ == "__main__":
    unittest.main()
