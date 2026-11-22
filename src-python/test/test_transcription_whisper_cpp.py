"""models/transcription/transcription_whisper_cpp.py のテスト。

whisper-server 自体 (whisper.cpp) は信頼し、VRCT 側の責務である
- GGML 重みの検出・ダウンロード (.part からの rename)
- whisper-server バイナリの探索と、それが無い場合に Vulkan デバイスを出さないこと
- /inference への送信内容 (言語・閾値・形式) とレスポンスの解釈
  (セグメントの閾値フィルタ、言語名→コード変換、is_definitive)
- 子プロセスが落ちた場合の 1 回だけの再起動
を検証する。

実際の whisper-server を使う結合テストは、環境変数
VRCT_TEST_WHISPER_CPP_SERVER (バイナリのパス) と
VRCT_TEST_WHISPER_CPP_MODEL (GGML モデルのパス) が両方設定されている
場合のみ実行する。
"""

import io
import os
import tempfile
import unittest
import wave
from unittest.mock import MagicMock, patch

import requests
from speech_recognition import AudioData

import models.transcription.transcription_whisper_cpp as wc


def _ggml_file(path: str, size: int = 2 * 1024 * 1024) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write((0x67676D6C).to_bytes(4, "little"))
        f.write(b"\0" * (size - 4))


def _silence(seconds: float = 0.5) -> AudioData:
    return AudioData(b"\0\0" * int(16000 * seconds), 16000, 2)


class TestWeights(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_every_faster_whisper_weight_type_has_a_ggml_file(self) -> None:
        from models.transcription.transcription_whisper import _MODELS

        self.assertEqual(set(_MODELS), set(wc._GGML_MODELS))

    def test_missing_weight_is_not_available(self) -> None:
        self.assertFalse(wc.checkWhisperCppWeight(self.root, "base"))

    def test_valid_ggml_weight_is_available(self) -> None:
        _ggml_file(wc.getWhisperCppWeightPath(self.root, "base"))
        self.assertTrue(wc.checkWhisperCppWeight(self.root, "base"))

    def test_wrong_magic_is_rejected(self) -> None:
        path = wc.getWhisperCppWeightPath(self.root, "base")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"<html>" + b"\0" * (2 * 1024 * 1024))  # e.g. an HTML error page
        self.assertFalse(wc.checkWhisperCppWeight(self.root, "base"))

    def test_turbo_int8_maps_to_q8_0(self) -> None:
        self.assertTrue(wc.getWhisperCppWeightPath(self.root, "large-v3-turbo-int8").endswith("ggml-large-v3-turbo-q8_0.bin"))

    def test_unknown_weight_type(self) -> None:
        self.assertIsNone(wc.getWhisperCppWeightPath(self.root, "nope"))
        self.assertFalse(wc.checkWhisperCppWeight(self.root, "nope"))

    def test_download_writes_part_file_then_renames(self) -> None:
        seen_paths = []

        def fake_download(url, path, func=None):
            seen_paths.append((url, path))
            _ggml_file(path)
            if func:
                func(1.0)
            return True

        end = MagicMock()
        with patch("models.transcription.transcription_whisper.downloadFile", side_effect=fake_download):
            wc.downloadWhisperCppWeight(self.root, "small", end_callback=end)

        url, part_path = seen_paths[0]
        self.assertEqual(url, "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin")
        self.assertTrue(part_path.endswith(".part"))
        self.assertFalse(os.path.exists(part_path))
        self.assertTrue(wc.checkWhisperCppWeight(self.root, "small"))
        end.assert_called_once()

    def test_failed_download_leaves_no_weight_and_still_calls_end_callback(self) -> None:
        end = MagicMock()
        with patch("models.transcription.transcription_whisper.downloadFile", return_value=False):
            wc.downloadWhisperCppWeight(self.root, "small", end_callback=end)
        self.assertFalse(wc.checkWhisperCppWeight(self.root, "small"))
        end.assert_called_once()

    def test_download_is_skipped_when_weight_exists(self) -> None:
        _ggml_file(wc.getWhisperCppWeightPath(self.root, "tiny"))
        with patch("models.transcription.transcription_whisper.downloadFile") as download:
            wc.downloadWhisperCppWeight(self.root, "tiny")
        download.assert_not_called()


class TestRuntimeAndDevices(unittest.TestCase):
    def test_no_vulkan_devices_without_bundled_server(self) -> None:
        with patch.object(wc, "findWhisperCppServer", return_value=None), \
             patch.object(wc, "getVulkanDeviceList") as enumerate_devices:
            self.assertEqual(wc.getVulkanComputeDeviceList("/x"), [])
        # The Vulkan loader is not even touched when the runtime is missing.
        enumerate_devices.assert_not_called()

    def test_vulkan_devices_use_raw_vulkan_index(self) -> None:
        with patch.object(wc, "findWhisperCppServer", return_value="/x/whisper-server"), \
             patch.object(wc, "getVulkanDeviceList", return_value=[
                 {"vulkan_index": 1, "name": "AMD Radeon RX 7800 XT", "vendor_id": 0x1002},
             ]):
            devices = wc.getVulkanComputeDeviceList("/x")
        self.assertEqual(devices, [{
            "device": "vulkan",
            "device_index": 1,
            "device_name": "AMD Radeon RX 7800 XT (Vulkan)",
            "compute_types": ["auto"],
        }])

    def test_env_override_is_searched_first(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            exe = os.path.join(d, wc._SERVER_EXE)
            open(exe, "w").close()
            with patch.dict(os.environ, {"VRCT_WHISPER_CPP_DIR": d}):
                self.assertEqual(wc.findWhisperCppServer("/nonexistent"), exe)

    def test_vulkan_enumeration_never_raises(self) -> None:
        # In CI there is usually no Vulkan driver at all.
        self.assertIsInstance(wc.getVulkanDeviceList(), list)


class TestLanguageCodes(unittest.TestCase):
    def test_full_names_map_to_codes(self) -> None:
        self.assertEqual(wc._toWhisperLanguageCode("english"), "en")
        self.assertEqual(wc._toWhisperLanguageCode("Japanese"), "ja")
        self.assertEqual(wc._toWhisperLanguageCode("cantonese"), "yue")

    def test_short_codes_pass_through(self) -> None:
        self.assertEqual(wc._toWhisperLanguageCode("ru"), "ru")
        self.assertIsNone(wc._toWhisperLanguageCode(None))

    def test_every_vrct_whisper_code_is_known_to_whisper_cpp(self) -> None:
        from models.transcription.transcription_languages import transcription_lang

        known = set(wc._WHISPER_LANGUAGE_NAME_TO_CODE.values())
        used = {c["Whisper"] for countries in transcription_lang.values() for c in countries.values()}
        self.assertEqual(used - known, set())


class TestProvider(unittest.TestCase):
    def setUp(self) -> None:
        self.server = MagicMock()
        self.server.inference_url = "http://127.0.0.1:1234/inference"
        self.provider = wc.WhisperCppProvider(self.server)

    def _response(self, payload: dict) -> MagicMock:
        response = MagicMock()
        response.json.return_value = payload
        response.raise_for_status.return_value = None
        return response

    def _transcribe(self, force_language: bool, **kwargs):
        return self.provider.transcribe(
            _silence(),
            kwargs.get("language", "English"),
            kwargs.get("country", "United States"),
            avg_logprob=-0.8,
            no_speech_prob=0.6,
            no_repeat_ngram_size=0,
            force_language=force_language,
        )

    @patch("models.transcription.transcription_whisper_cpp.requests.post")
    def test_sends_wav_and_forced_language(self, post) -> None:
        post.return_value = self._response({"language": "english", "segments": []})

        self._transcribe(force_language=True)

        _, kwargs = post.call_args
        self.assertEqual(post.call_args[0][0], "http://127.0.0.1:1234/inference")
        name, wav_bytes, mime = kwargs["files"]["file"]
        self.assertEqual((name, mime), ("audio.wav", "audio/wav"))
        with wave.open(io.BytesIO(wav_bytes)) as w:
            self.assertEqual((w.getframerate(), w.getsampwidth(), w.getnchannels()), (16000, 2, 1))
        form = kwargs["data"]
        self.assertEqual(form["language"], "en")
        self.assertEqual(form["response_format"], "verbose_json")
        self.assertEqual(form["logprob_thold"], "-0.8")
        self.assertEqual(form["no_speech_thold"], "0.6")
        self.assertEqual(form["no_language_probabilities"], "true")

    @patch("models.transcription.transcription_whisper_cpp.requests.post")
    def test_auto_detects_when_several_candidate_languages(self, post) -> None:
        post.return_value = self._response({"language": "english", "segments": []})
        self._transcribe(force_language=False)
        self.assertEqual(post.call_args[1]["data"]["language"], "auto")

    @patch("models.transcription.transcription_whisper_cpp.requests.post")
    def test_filters_segments_like_local_whisper(self, post) -> None:
        post.return_value = self._response({
            "language": "english",
            "segments": [
                {"text": " Hello", "avg_logprob": -0.2, "no_speech_prob": 0.01},
                {"text": " garbage", "avg_logprob": -1.5, "no_speech_prob": 0.01},  # low confidence
                {"text": " silence", "avg_logprob": -0.1, "no_speech_prob": 0.9},  # probably no speech
                {"text": " world", "avg_logprob": -0.4, "no_speech_prob": 0.1},
            ],
        })

        text, confidence, is_definitive = self._transcribe(force_language=True)

        self.assertEqual(text, " Hello world")
        self.assertAlmostEqual(confidence, 0.7408, places=3)  # exp(mean(-0.2, -0.4))
        self.assertTrue(is_definitive)

    @patch("models.transcription.transcription_whisper_cpp.requests.post")
    def test_detected_language_decides_early_exit(self, post) -> None:
        segment = {"text": " konnichiwa", "avg_logprob": -0.3, "no_speech_prob": 0.0}
        post.return_value = self._response({"language": "japanese", "segments": [segment]})
        self.assertTrue(self._transcribe(force_language=False, language="Japanese", country="Japan")[2])
        self.assertFalse(self._transcribe(force_language=False)[2])

    @patch("models.transcription.transcription_whisper_cpp.requests.post")
    def test_empty_result(self, post) -> None:
        post.return_value = self._response({"language": "english", "segments": [], "text": ""})
        self.assertEqual(self._transcribe(force_language=True), ("", 0.0, False))

    @patch("models.transcription.transcription_whisper_cpp.requests.post")
    def test_restarts_dead_server_once_and_retries(self, post) -> None:
        post.side_effect = [
            requests.ConnectionError("child died"),
            self._response({"language": "english", "segments": [{"text": " ok", "avg_logprob": -0.1, "no_speech_prob": 0.0}]}),
        ]

        text, _, _ = self._transcribe(force_language=True)

        self.assertEqual(text, " ok")
        self.server.restart.assert_called_once()
        self.assertEqual(post.call_count, 2)

    @patch("models.transcription.transcription_whisper_cpp.requests.post")
    def test_second_failure_propagates(self, post) -> None:
        post.side_effect = requests.ConnectionError("still dead")
        with self.assertRaises(requests.ConnectionError):
            self._transcribe(force_language=True)
        self.server.restart.assert_called_once()


class TestServerSharing(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        _ggml_file(wc.getWhisperCppWeightPath(self.root, "base"))
        _ggml_file(wc.getWhisperCppWeightPath(self.root, "small"))
        wc.shutdownWhisperCppServer()

    def tearDown(self) -> None:
        wc.shutdownWhisperCppServer()
        self.tmp.cleanup()

    @patch.object(wc, "findWhisperCppServer", return_value="/bin/whisper-server")
    @patch.object(wc.WhisperCppServer, "ensureStarted")
    @patch.object(wc.WhisperCppServer, "stop")
    def test_mic_and_speaker_share_one_server_until_model_changes(self, stop, _ensure, _find) -> None:
        mic = wc.acquireWhisperCppServer(self.root, "base", 0)
        speaker = wc.acquireWhisperCppServer(self.root, "base", 0)
        self.assertIs(mic, speaker)
        stop.assert_not_called()

        other = wc.acquireWhisperCppServer(self.root, "small", 0)
        self.assertIsNot(other, mic)
        stop.assert_called_once()

    @patch.object(wc, "findWhisperCppServer", return_value=None)
    def test_missing_runtime_is_a_clear_error(self, _find) -> None:
        with self.assertRaisesRegex(RuntimeError, "not bundled"):
            wc.acquireWhisperCppServer(self.root, "base", 0)


@unittest.skipUnless(
    os.environ.get("VRCT_TEST_WHISPER_CPP_SERVER") and os.environ.get("VRCT_TEST_WHISPER_CPP_MODEL"),
    "set VRCT_TEST_WHISPER_CPP_SERVER and VRCT_TEST_WHISPER_CPP_MODEL to run against a real whisper-server",
)
class TestRealServer(unittest.TestCase):
    def test_start_transcribe_crash_recover_stop(self) -> None:
        server = wc.WhisperCppServer(
            os.environ["VRCT_TEST_WHISPER_CPP_SERVER"], os.environ["VRCT_TEST_WHISPER_CPP_MODEL"], vulkan_index=0,
        )
        try:
            server.ensureStarted()
            provider = wc.WhisperCppProvider(server)
            kwargs = dict(avg_logprob=-100.0, no_speech_prob=1.0, no_repeat_ngram_size=0, force_language=True)
            provider.transcribe(_silence(1.0), "English", "United States", **kwargs)

            server._process.kill()
            server._process.wait()
            provider.transcribe(_silence(1.0), "English", "United States", **kwargs)
            self.assertTrue(server.isAlive())
        finally:
            server.stop()
        self.assertFalse(server.isAlive())


if __name__ == "__main__":
    unittest.main()
