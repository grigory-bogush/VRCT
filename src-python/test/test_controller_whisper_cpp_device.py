"""whisper.cpp (Vulkan) 文字起こしデバイスの config/controller/model 側の配線テスト。

- 文字起こし用のデバイス一覧 (SELECTABLE_TRANSCRIPTION_COMPUTE_DEVICE_LIST) は
  CPU/CUDA 一覧 + Vulkan GPU で、翻訳用の一覧 (CTranslate2 専用) には
  Vulkan が混ざらないこと
- Vulkan デバイスを選ぶと Whisper の重みチェック/ダウンロードが GGML 側に
  切り替わり、ダウンロード済みフラグ・エンジン可用性・UI が追従すること
"""

import copy
import unittest
from unittest.mock import patch

from config import ConfigValidationError, config
from controller import Controller
from model import model

_VULKAN = {"device": "vulkan", "device_index": 0, "device_name": "AMD Radeon RX 7800 XT (Vulkan)", "compute_types": ["auto"]}


class _ConfigSnapshot:
    _NAMES = (
        "_SELECTABLE_TRANSCRIPTION_COMPUTE_DEVICE_LIST",
        "_SELECTABLE_TRANSLATION_COMPUTE_DEVICE_LIST",
        "_SELECTED_TRANSCRIPTION_COMPUTE_DEVICE",
        "_SELECTED_TRANSCRIPTION_COMPUTE_TYPE",
        "_SELECTABLE_WHISPER_WEIGHT_TYPE_DICT",
        "_SELECTABLE_TRANSCRIPTION_ENGINE_STATUS",
        "_SELECTED_TRANSCRIPTION_ENGINE",
        "_WHISPER_WEIGHT_TYPE",
    )

    def __enter__(self):
        self._saved = {n: copy.deepcopy(getattr(config, n)) for n in self._NAMES}
        return self

    def __exit__(self, *exc):
        for name, value in self._saved.items():
            setattr(config, name, value)


class TranscriptionDeviceListTests(unittest.TestCase):
    def test_transcription_list_extends_shared_list(self) -> None:
        shared = list(config.SELECTABLE_COMPUTE_DEVICE_LIST)
        self.assertEqual(config.SELECTABLE_TRANSCRIPTION_COMPUTE_DEVICE_LIST[: len(shared)], shared)

    def test_translation_list_never_contains_vulkan(self) -> None:
        self.assertFalse(any(d.get("device") == "vulkan" for d in config.SELECTABLE_COMPUTE_DEVICE_LIST))

    def test_vulkan_transcription_device_needs_its_own_list(self) -> None:
        # Translation has its own Vulkan entries only when llama-server is
        # bundled (test_translation_llama_cpp.py); without them a
        # transcription-only Vulkan device must not leak into translation.
        with _ConfigSnapshot():
            config._SELECTABLE_TRANSCRIPTION_COMPUTE_DEVICE_LIST = list(config.SELECTABLE_COMPUTE_DEVICE_LIST) + [_VULKAN]
            config._SELECTABLE_TRANSLATION_COMPUTE_DEVICE_LIST = list(config.SELECTABLE_COMPUTE_DEVICE_LIST)
            translation_before = config.SELECTED_TRANSLATION_COMPUTE_DEVICE

            config.SELECTED_TRANSCRIPTION_COMPUTE_DEVICE = _VULKAN
            with self.assertRaises(ConfigValidationError):
                config.SELECTED_TRANSLATION_COMPUTE_DEVICE = _VULKAN

            self.assertEqual(config.SELECTED_TRANSCRIPTION_COMPUTE_DEVICE, _VULKAN)
            self.assertEqual(config.SELECTED_TRANSLATION_COMPUTE_DEVICE, translation_before)

    def test_saved_vulkan_device_is_dropped_when_gpu_or_runtime_disappears(self) -> None:
        with _ConfigSnapshot():
            config._SELECTABLE_TRANSCRIPTION_COMPUTE_DEVICE_LIST = list(config.SELECTABLE_COMPUTE_DEVICE_LIST)
            before = config.SELECTED_TRANSCRIPTION_COMPUTE_DEVICE
            with self.assertRaises(ConfigValidationError):
                config.SELECTED_TRANSCRIPTION_COMPUTE_DEVICE = _VULKAN
            self.assertEqual(config.SELECTED_TRANSCRIPTION_COMPUTE_DEVICE, before)


class ModelWeightRoutingTests(unittest.TestCase):
    def test_vulkan_device_uses_ggml_weights(self) -> None:
        with _ConfigSnapshot():
            config._SELECTABLE_TRANSCRIPTION_COMPUTE_DEVICE_LIST = list(config.SELECTABLE_COMPUTE_DEVICE_LIST) + [_VULKAN]
            config.SELECTED_TRANSCRIPTION_COMPUTE_DEVICE = _VULKAN
            with patch("model.checkWhisperCppWeight", return_value=True) as cpp, \
                 patch("model.checkWhisperWeight", return_value=False) as ct2:
                self.assertTrue(model.checkTranscriptionWhisperModelWeight("base"))
            cpp.assert_called_once()
            ct2.assert_not_called()

            with patch("model.downloadWhisperCppWeight") as cpp_dl, patch("model.downloadWhisperWeight") as ct2_dl:
                model.downloadWhisperModelWeight("base")
            cpp_dl.assert_called_once()
            ct2_dl.assert_not_called()

    def test_cpu_device_keeps_faster_whisper_weights(self) -> None:
        with _ConfigSnapshot():
            config.SELECTED_TRANSCRIPTION_COMPUTE_DEVICE = config.SELECTABLE_COMPUTE_DEVICE_LIST[0]
            with patch("model.checkWhisperCppWeight") as cpp, patch("model.checkWhisperWeight", return_value=True) as ct2:
                self.assertTrue(model.checkTranscriptionWhisperModelWeight("base"))
            ct2.assert_called_once()
            cpp.assert_not_called()


class SetTranscriptionDeviceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = Controller.__new__(Controller)
        self.controller.run_mapping = {
            "selected_transcription_compute_type": "selected_transcription_compute_type",
            "selectable_whisper_weight_type_dict": "selectable_whisper_weight_type_dict",
            "selectable_language_list": "selectable_language_list",
            "selected_your_languages": "selected_your_languages",
            "selected_target_languages": "selected_target_languages",
        }
        self.calls = []
        self.controller.run = lambda status, endpoint, result: self.calls.append((endpoint, dict(result) if hasattr(result, "items") else result))

    def test_switching_to_vulkan_refreshes_weight_status_and_engine(self) -> None:
        with _ConfigSnapshot():
            config._SELECTABLE_TRANSCRIPTION_COMPUTE_DEVICE_LIST = list(config.SELECTABLE_COMPUTE_DEVICE_LIST) + [_VULKAN]
            config.SELECTED_TRANSCRIPTION_COMPUTE_DEVICE = config.SELECTABLE_COMPUTE_DEVICE_LIST[0]
            config._WHISPER_WEIGHT_TYPE = "base"
            config._SELECTED_TRANSCRIPTION_ENGINE = "Whisper"
            config.SELECTABLE_WHISPER_WEIGHT_TYPE_DICT = {k: True for k in config.SELECTABLE_WHISPER_WEIGHT_TYPE_DICT}

            # Only the GGML "small" weight exists so far.
            with patch("model.checkWhisperCppWeight", side_effect=lambda root, w: w == "small"):
                self.controller.setSelectedTranscriptionComputeDevice(_VULKAN)

            self.assertEqual(config.SELECTED_TRANSCRIPTION_COMPUTE_DEVICE, _VULKAN)
            weights = config.SELECTABLE_WHISPER_WEIGHT_TYPE_DICT
            self.assertTrue(weights["small"])
            self.assertFalse(weights["base"])
            self.assertFalse(config.SELECTABLE_TRANSCRIPTION_ENGINE_STATUS["Whisper"])
            self.assertFalse(self.controller._whisper_available_cache)
            # The selected weight is missing, so the engine must stay on Whisper
            # (the UI then offers the download) rather than flip to Google.
            self.assertEqual(config.SELECTED_TRANSCRIPTION_ENGINE, "Whisper")
            pushed = dict(self.calls)
            self.assertEqual(pushed["selectable_whisper_weight_type_dict"], dict(weights))

    def test_switching_between_cpu_and_cuda_does_not_touch_weights(self) -> None:
        with _ConfigSnapshot():
            config.SELECTED_TRANSCRIPTION_COMPUTE_DEVICE = config.SELECTABLE_COMPUTE_DEVICE_LIST[0]
            with patch.object(Controller, "refreshWhisperWeightStatusForBackend") as refresh:
                self.controller.setSelectedTranscriptionComputeDevice(config.SELECTABLE_COMPUTE_DEVICE_LIST[0])
            refresh.assert_not_called()

    def test_finished_ggml_download_of_selected_weight_reenables_whisper(self) -> None:
        with _ConfigSnapshot():
            config._WHISPER_WEIGHT_TYPE = "base"
            config.SELECTABLE_TRANSCRIPTION_ENGINE_STATUS["Whisper"] = False
            download = Controller.DownloadWhisper(self.controller.run_mapping | {"downloaded_whisper_weight": "downloaded_whisper_weight"}, "base", self.controller.run)
            with patch("model.model.checkTranscriptionWhisperModelWeight", return_value=True):
                download.downloaded()
            self.assertTrue(config.SELECTABLE_TRANSCRIPTION_ENGINE_STATUS["Whisper"])
            self.assertTrue(config.SELECTABLE_WHISPER_WEIGHT_TYPE_DICT["base"])


if __name__ == "__main__":
    unittest.main()
