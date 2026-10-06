import threading
import unittest
import weakref
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from translator_app import hf_transformers as hf
from translator_app.config import TransformersConfig, load_config


class TransformersLoadingTests(unittest.TestCase):
    def test_invalid_quantization_and_cache_limits_are_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            for value in ('quantization = "fp4"', "max_cached_models = 0", "max_cached_models = true"):
                with self.subTest(value=value):
                    path.write_text("[transformers]\n" + value)
                    with self.assertRaises(ValueError):
                        load_config(path)

    def test_eviction_releases_previous_model_before_loading_next(self):
        references = []

        def load(config):
            if references:
                self.assertIsNone(references[-1](), "Evicted weights are still resident during loading")
            loaded = hf._LoadedModel(processor=None, model=object())
            references.append(weakref.ref(loaded))
            return loaded

        config = TransformersConfig(model="first", max_cached_models=1)
        with patch.object(hf, "_MODEL_CACHE", {}), \
                patch.object(hf, "_load_model_uncached", side_effect=load), \
                patch.object(hf, "_release_accelerator_memory") as release:
            hf._load_model(config)
            hf._load_model(replace(config, model="second"))
            self.assertEqual(len(hf._MODEL_CACHE), 1)
            release.assert_called_once()

    def test_quantization_and_attention_changes_load_distinct_models(self):
        config = TransformersConfig(model="same", max_cached_models=3)
        with patch.object(hf, "_MODEL_CACHE", {}), \
                patch.object(hf, "_load_model_uncached", side_effect=lambda _: object()) as load:
            first = hf._load_model(config)
            self.assertIs(first, hf._load_model(config))
            self.assertIsNot(first, hf._load_model(replace(config, quantization="4bit")))
            self.assertIsNot(first, hf._load_model(replace(config, attn_implementation="sdpa")))
            self.assertEqual(load.call_count, 3)

    def test_gpu_only_fails_when_cuda_is_unavailable(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict("sys.modules", {"torch": torch}), \
                patch.object(hf, "_load_dependencies", return_value=(Mock(), Mock(), Mock())):
            with self.assertRaisesRegex(hf.TransformersError, "requires CUDA"):
                hf._load_model_uncached(TransformersConfig(device_map="cuda"))

    def test_gpu_only_rejects_cpu_parameters(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        model = SimpleNamespace(parameters=lambda: iter([
            SimpleNamespace(device=SimpleNamespace(type="cuda")),
            SimpleNamespace(device=SimpleNamespace(type="cpu")),
        ]))
        multimodal = Mock()
        multimodal.from_pretrained.return_value = model
        with patch.dict("sys.modules", {"torch": torch}), \
                patch.object(hf, "_load_dependencies", return_value=(Mock(), multimodal, Mock())):
            with self.assertRaisesRegex(hf.TransformersError, "GPU-only loading failed"):
                hf._load_model_uncached(TransformersConfig(device_map="cuda:0"))

    def test_loading_failure_does_not_retry_a_second_architecture(self):
        for error in (RuntimeError("CUDA out of memory"), ValueError("Invalid quantization settings")):
            with self.subTest(error=error):
                causal, multimodal, processor = Mock(), Mock(), Mock()
                multimodal.from_pretrained.side_effect = error
                with patch.object(hf, "_load_dependencies", return_value=(causal, multimodal, processor)):
                    with self.assertRaises(hf.TransformersError):
                        hf._load_model_uncached(TransformersConfig())
                causal.from_pretrained.assert_not_called()

    def test_text_only_architecture_falls_back_to_causal_loader(self):
        causal, multimodal, processor = Mock(), Mock(), Mock()
        multimodal.from_pretrained.side_effect = ValueError("Unrecognized configuration class TextConfig")
        with patch.object(hf, "_load_dependencies", return_value=(causal, multimodal, processor)):
            result = hf._load_model_uncached(TransformersConfig())
        self.assertIs(result.model, causal.from_pretrained.return_value)

    def test_waiting_request_cannot_load_or_evict_during_generation(self):
        active = threading.Event()
        waiting = threading.Event()
        lock = threading.RLock()
        order, errors = [], []

        class ObservedLock:
            def __enter__(self):
                if threading.current_thread().name == "second":
                    waiting.set()
                lock.acquire()

            def __exit__(self, *args):
                lock.release()

        def load(config):
            order.append("load " + config.model)
            if config.model == "first":
                active.set()
            return hf._LoadedModel(processor=None, model=SimpleNamespace(device="cuda"))

        def generate(*args, config, **kwargs):
            if config.model == "first" and not waiting.wait(5):
                raise AssertionError("Second request did not reach the lock")
            order.append("generate " + config.model)
            return "{}"

        def request(name):
            try:
                hf.chat_json(config=TransformersConfig(model=name), system="system", user="user")
            except Exception as exc:
                errors.append(exc)

        with patch.object(hf, "_GENERATE_LOCK", ObservedLock()), \
                patch.object(hf, "_load_model", side_effect=load), \
                patch.object(hf, "_build_inputs", return_value=Mock()), \
                patch.object(hf, "_generate", side_effect=generate):
            first = threading.Thread(target=request, args=("first",), name="first")
            second = threading.Thread(target=request, args=("second",), name="second")
            first.start()
            self.assertTrue(active.wait(5), "First request did not acquire the lock")
            second.start()
            first.join(10)
            second.join(10)
            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(order, ["load first", "generate first", "load second", "generate second"])


if __name__ == "__main__":
    unittest.main()
