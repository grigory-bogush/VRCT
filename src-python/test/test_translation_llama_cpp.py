"""llama.cpp (Vulkan) backend for the local AI translation engine.

- GGUF weight check/download (models/translation/translation_llama_cpp.py)
- TranslateGemma prompt and the /completion call (restart once on crash)
- weight <-> device compatibility, and the controller keeping them aligned
- Translator routing GGUF weights to llama-server instead of CTranslate2
- optional: a real llama-server (set VRCT_TEST_LLAMA_CPP_SERVER and
  VRCT_TEST_LLAMA_CPP_MODEL)
"""

import copy
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests

import models.translation.translation_llama_cpp as lc
from config import config
from controller import Controller
from model import model
from models.translation.translation_languages import translation_lang

_CPU = {"device": "cpu", "device_index": 0, "device_name": "cpu", "compute_types": ["auto", "int8"]}
_VULKAN = {"device": "vulkan", "device_index": 0, "device_name": "AMD Radeon RX 7800 XT (Vulkan)", "compute_types": ["auto"]}


def _gguf_file(path: str, size: int = 2 * 1024 * 1024) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(b"\0" * (size - 4))


class WeightTests(unittest.TestCase):
    def test_backends(self) -> None:
        self.assertEqual(lc.getWeightBackend("translategemma-4b-it-Q4_K_M"), lc.BACKEND_LLAMA_CPP)
        self.assertEqual(lc.getWeightBackend("nllb-200-distilled-600M-ct2-int8"), lc.BACKEND_CTRANSLATE2)
        self.assertEqual(lc.getDeviceBackend(_VULKAN), lc.BACKEND_LLAMA_CPP)
        self.assertEqual(lc.getDeviceBackend(_CPU), lc.BACKEND_CTRANSLATE2)
        self.assertTrue(lc.isWeightCompatibleWithDevice("translategemma-12b-it-Q4_K_M", _VULKAN))
        self.assertFalse(lc.isWeightCompatibleWithDevice("translategemma-12b-it-Q4_K_M", _CPU))
        self.assertFalse(lc.isWeightCompatibleWithDevice("m2m100_418M-ct2-int8", _VULKAN))

    def test_check_requires_gguf_magic(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            weight = "translategemma-4b-it-Q4_K_M"
            self.assertFalse(lc.checkLlamaCppWeight(root, weight))
            path = lc.getLlamaCppWeightPath(root, weight)
            _gguf_file(path)
            self.assertTrue(lc.checkLlamaCppWeight(root, weight))
            with open(path, "r+b") as f:
                f.write(b"XXXX")
            self.assertFalse(lc.checkLlamaCppWeight(root, weight))
        self.assertIsNone(lc.getLlamaCppWeightPath("/x", "nllb-200-distilled-600M-ct2-int8"))

    def test_download_renames_part_file_and_always_calls_end_callback(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            weight = "translategemma-4b-it-Q4_K_M"
            end = MagicMock()

            def fake_download(url, path, func=None):
                self.assertTrue(path.endswith(".part"))
                self.assertIn("mradermacher/translategemma-4b-it-GGUF", url)
                _gguf_file(path)
                if func:
                    func(1.0)
                return True

            with patch("models.transcription.transcription_whisper.downloadFile", side_effect=fake_download):
                self.assertTrue(lc.downloadLlamaCppWeight(root, weight, end_callback=end))
            self.assertTrue(lc.checkLlamaCppWeight(root, weight))
            end.assert_called_once()

            end.reset_mock()
            with patch("models.transcription.transcription_whisper.downloadFile", return_value=False):
                self.assertFalse(lc.downloadLlamaCppWeight(root, "translategemma-4b-it-Q8_0", end_callback=end))
            end.assert_called_once()

    def test_every_gguf_weight_has_a_language_table_known_to_the_prompt(self) -> None:
        for weight in lc.llama_cpp_weights:
            table = translation_lang["CTranslate2"][weight]
            self.assertIn("Japanese", table["source"])
            self.assertEqual(table["source"]["Norwegian"], "no")  # not YAML's False
            for code in table["source"].values():
                self.assertIn(code, lc._LANGUAGE_NAMES, code)

    def test_weights_are_offered_by_config(self) -> None:
        for weight in lc.llama_cpp_weights:
            self.assertIn(weight, config.SELECTABLE_CTRANSLATE2_WEIGHT_TYPE_LIST)
            self.assertIn(weight, config.SELECTABLE_CTRANSLATE2_WEIGHT_TYPE_DICT)


class PromptAndTranslateTests(unittest.TestCase):
    def test_translategemma_prompt(self) -> None:
        prompt = lc.buildTranslateGemmaPrompt("  こんにちは  ", "ja", "ru")
        self.assertTrue(prompt.startswith("<start_of_turn>user\nYou are a professional Japanese (ja) to Russian (ru) translator."))
        self.assertIn("Please translate the following Japanese text into Russian:\n\n\nこんにちは<end_of_turn>\n", prompt)
        self.assertTrue(prompt.endswith("<start_of_turn>model\n"))

    def _response(self, payload):
        response = MagicMock()
        response.json.return_value = payload
        response.raise_for_status.return_value = None
        return response

    def setUp(self) -> None:
        self.server = MagicMock()
        self.server.completion_url = "http://127.0.0.1:1/completion"
        self.translator = lc.LlamaCppTranslator(self.server)

    @patch("models.translation.translation_llama_cpp.requests.post")
    def test_translate_posts_greedy_completion(self, post) -> None:
        post.return_value = self._response({"content": " Привет \n", "timings": {"predicted_n": 3}})
        self.assertEqual(self.translator.translate("Hello", "en", "ru"), "Привет")
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["temperature"], 0.0)
        self.assertIn("<end_of_turn>", body["stop"])
        self.assertIn("English (en) to Russian (ru)", body["prompt"])

    @patch("models.translation.translation_llama_cpp.requests.post")
    def test_restarts_dead_server_once(self, post) -> None:
        post.side_effect = [requests.ConnectionError("died"), self._response({"content": "ok"})]
        self.assertEqual(self.translator.translate("x", "en", "ja"), "ok")
        self.server.restart.assert_called_once()

    @patch("models.translation.translation_llama_cpp.requests.post")
    def test_second_failure_propagates(self, post) -> None:
        post.side_effect = requests.ConnectionError("still dead")
        with self.assertRaises(requests.ConnectionError):
            self.translator.translate("x", "en", "ja")


class ServerSharingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        for weight in ("translategemma-4b-it-Q4_K_M", "translategemma-4b-it-Q8_0"):
            _gguf_file(lc.getLlamaCppWeightPath(self.root, weight))
        lc.shutdownLlamaCppServer()

    def tearDown(self) -> None:
        lc.shutdownLlamaCppServer()
        self.tmp.cleanup()

    @patch.object(lc, "findLlamaCppServer", return_value="/bin/llama-server")
    @patch.object(lc.LlamaCppServer, "ensureStarted")
    @patch.object(lc.LlamaCppServer, "stop")
    def test_shared_until_weight_or_gpu_changes(self, stop, _ensure, _find) -> None:
        a = lc.acquireLlamaCppServer(self.root, "translategemma-4b-it-Q4_K_M", 0)
        self.assertIs(a, lc.acquireLlamaCppServer(self.root, "translategemma-4b-it-Q4_K_M", 0))
        stop.assert_not_called()
        b = lc.acquireLlamaCppServer(self.root, "translategemma-4b-it-Q8_0", 0)
        self.assertIsNot(a, b)
        c = lc.acquireLlamaCppServer(self.root, "translategemma-4b-it-Q8_0", 1)
        self.assertIsNot(b, c)
        self.assertEqual(stop.call_count, 2)

    @patch.object(lc, "findLlamaCppServer", return_value=None)
    def test_missing_runtime_is_a_clear_error(self, _find) -> None:
        with self.assertRaisesRegex(RuntimeError, "not bundled"):
            lc.acquireLlamaCppServer(self.root, "translategemma-4b-it-Q4_K_M", 0)

    def test_server_args_pin_the_gpu(self) -> None:
        server = lc.LlamaCppServer("/bin/llama-server", "/m.gguf", vulkan_index=1)
        args = server._serverArgs()
        self.assertEqual(args[args.index("--device") + 1], "Vulkan0")
        self.assertIn("--no-jinja", args)

    def test_no_vulkan_devices_without_bundled_server(self) -> None:
        with patch.object(lc, "findLlamaCppServer", return_value=None), \
             patch.object(lc, "getVulkanDeviceList") as enumerate_devices:
            self.assertEqual(lc.getTranslationVulkanComputeDeviceList("/x"), [])
        enumerate_devices.assert_not_called()


class TranslatorRoutingTests(unittest.TestCase):
    def test_gguf_weight_goes_through_llama_server(self) -> None:
        from models.translation.translation_translator import Translator

        translator = Translator()
        fake_server = MagicMock()
        with patch("models.translation.translation_translator.acquireLlamaCppServer", return_value=fake_server) as acquire:
            translator.changeCTranslate2Model(path="/root", model_type="translategemma-4b-it-Q4_K_M", device="vulkan", device_index=1)
        acquire.assert_called_once_with("/root", "translategemma-4b-it-Q4_K_M", 1)
        self.assertTrue(translator.isLoadedCTranslate2Model())

        translator.llama_cpp_translator = MagicMock()
        translator.llama_cpp_translator.translate.return_value = "こんにちは"
        result = translator.translate("CTranslate2", "translategemma-4b-it-Q4_K_M", "English", "Japanese", "Japan", "Hello")
        self.assertEqual(result, "こんにちは")
        translator.llama_cpp_translator.translate.assert_called_once_with("Hello", "en", "ja")

    def test_gguf_weight_refuses_non_vulkan_device(self) -> None:
        from models.translation.translation_translator import Translator

        with self.assertRaises(RuntimeError):
            Translator().changeCTranslate2Model(path="/root", model_type="translategemma-4b-it-Q4_K_M", device="cpu")


class _ConfigSnapshot:
    _NAMES = (
        "_SELECTABLE_TRANSLATION_COMPUTE_DEVICE_LIST",
        "_SELECTED_TRANSLATION_COMPUTE_DEVICE",
        "_SELECTED_TRANSLATION_COMPUTE_TYPE",
        "_SELECTABLE_CTRANSLATE2_WEIGHT_TYPE_DICT",
        "_SELECTABLE_TRANSLATION_ENGINE_STATUS",
        "_CTRANSLATE2_WEIGHT_TYPE",
    )

    def __enter__(self):
        self._saved = {n: copy.deepcopy(getattr(config, n)) for n in self._NAMES}
        return self

    def __exit__(self, *exc):
        for name, value in self._saved.items():
            setattr(config, name, value)


class ControllerAlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = Controller.__new__(Controller)
        self.controller.run_mapping = {k: k for k in (
            "selected_translation_compute_type",
            "selected_translation_compute_device",
            "selected_ctranslate2_weight_type",
            "selectable_ctranslate2_weight_type_dict",
        )}
        self.calls = []
        self.controller.run = lambda status, endpoint, result: self.calls.append((endpoint, result))
        self.cpu = copy.deepcopy(config.SELECTABLE_COMPUTE_DEVICE_LIST[0])

    def _setup_devices(self) -> None:
        config._SELECTABLE_TRANSLATION_COMPUTE_DEVICE_LIST = [self.cpu, _VULKAN]

    def test_switching_to_vulkan_picks_downloaded_gguf_weight(self) -> None:
        with _ConfigSnapshot():
            self._setup_devices()
            config.SELECTED_TRANSLATION_COMPUTE_DEVICE = self.cpu
            config.CTRANSLATE2_WEIGHT_TYPE = "nllb-200-distilled-600M-ct2-int8"
            downloaded = {"translategemma-4b-it-Q8_0"}
            with patch.object(model, "checkTranslatorCTranslate2ModelWeight", side_effect=lambda w: w in downloaded), \
                 patch.object(Controller, "updateTranslationEngineAndEngineList") as update:
                self.controller.setSelectedTranslationComputeDevice(_VULKAN)
            self.assertEqual(config.CTRANSLATE2_WEIGHT_TYPE, "translategemma-4b-it-Q8_0")
            self.assertTrue(config.SELECTABLE_TRANSLATION_ENGINE_STATUS["CTranslate2"])
            update.assert_called_once()
            pushed = dict(self.calls)
            self.assertEqual(pushed["selected_ctranslate2_weight_type"], "translategemma-4b-it-Q8_0")

    def test_switching_to_vulkan_without_downloads_selects_default_and_disables_engine(self) -> None:
        with _ConfigSnapshot():
            self._setup_devices()
            config.SELECTED_TRANSLATION_COMPUTE_DEVICE = self.cpu
            config.CTRANSLATE2_WEIGHT_TYPE = "nllb-200-distilled-600M-ct2-int8"
            with patch.object(model, "checkTranslatorCTranslate2ModelWeight", return_value=False), \
                 patch.object(Controller, "updateTranslationEngineAndEngineList"):
                self.controller.setSelectedTranslationComputeDevice(_VULKAN)
            self.assertEqual(config.CTRANSLATE2_WEIGHT_TYPE, lc.DEFAULT_LLAMA_CPP_WEIGHT)
            self.assertFalse(config.SELECTABLE_TRANSLATION_ENGINE_STATUS["CTranslate2"])

    def test_switching_back_to_cpu_restores_a_ctranslate2_weight(self) -> None:
        with _ConfigSnapshot():
            self._setup_devices()
            config.SELECTED_TRANSLATION_COMPUTE_DEVICE = _VULKAN
            config.CTRANSLATE2_WEIGHT_TYPE = "translategemma-4b-it-Q4_K_M"
            with patch.object(model, "checkTranslatorCTranslate2ModelWeight", side_effect=lambda w: w == "m2m100_1.2B-ct2-int8"), \
                 patch.object(Controller, "updateTranslationEngineAndEngineList"):
                self.controller.setSelectedTranslationComputeDevice(self.cpu)
            self.assertEqual(config.CTRANSLATE2_WEIGHT_TYPE, "m2m100_1.2B-ct2-int8")

    def test_compatible_device_change_keeps_weight(self) -> None:
        with _ConfigSnapshot():
            self._setup_devices()
            config.SELECTED_TRANSLATION_COMPUTE_DEVICE = self.cpu
            config.CTRANSLATE2_WEIGHT_TYPE = "m2m100_418M-ct2-int8"
            with patch.object(Controller, "refreshAiTranslationWeightStatus") as refresh:
                self.controller.setSelectedTranslationComputeDevice(self.cpu)
            refresh.assert_not_called()
            self.assertEqual(config.CTRANSLATE2_WEIGHT_TYPE, "m2m100_418M-ct2-int8")

    def test_choosing_gguf_weight_on_cpu_moves_device_to_vulkan(self) -> None:
        with _ConfigSnapshot():
            self._setup_devices()
            config.SELECTED_TRANSLATION_COMPUTE_DEVICE = self.cpu
            with patch.object(model, "checkTranslatorCTranslate2ModelWeight", return_value=True), \
                 patch.object(Controller, "updateTranslationEngineAndEngineList"):
                self.controller.setCtranslate2WeightType("translategemma-12b-it-Q4_K_M")
            self.assertEqual(config.SELECTED_TRANSLATION_COMPUTE_DEVICE, _VULKAN)
            self.assertEqual(dict(self.calls)["selected_translation_compute_device"], _VULKAN)

    def test_vulkan_translation_device_requires_runtime(self) -> None:
        from config import ConfigValidationError

        with _ConfigSnapshot():
            config._SELECTABLE_TRANSLATION_COMPUTE_DEVICE_LIST = [self.cpu]
            with self.assertRaises(ConfigValidationError):
                config.SELECTED_TRANSLATION_COMPUTE_DEVICE = _VULKAN


@unittest.skipUnless(
    os.environ.get("VRCT_TEST_LLAMA_CPP_SERVER") and os.environ.get("VRCT_TEST_LLAMA_CPP_MODEL"),
    "set VRCT_TEST_LLAMA_CPP_SERVER and VRCT_TEST_LLAMA_CPP_MODEL to run against a real llama-server",
)
class RealServerTests(unittest.TestCase):
    def test_translate_crash_recover_stop(self) -> None:
        server = lc.LlamaCppServer(
            os.environ["VRCT_TEST_LLAMA_CPP_SERVER"], os.environ["VRCT_TEST_LLAMA_CPP_MODEL"],
            vulkan_index=int(os.environ.get("VRCT_TEST_VULKAN_INDEX", "0")),
        )
        try:
            translator = lc.LlamaCppTranslator(server)
            text = translator.translate("Good morning! How are you?", "en", "ja")
            self.assertTrue(text)
            self.assertNotIn("<end_of_turn>", text)

            server._process.kill()
            server._process.wait()
            self.assertTrue(translator.translate("Thank you very much.", "en", "ru"))
            self.assertTrue(server.isAlive())
        finally:
            server.stop()
        self.assertFalse(server.isAlive())


if __name__ == "__main__":
    unittest.main()
