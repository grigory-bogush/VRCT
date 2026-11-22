"""whisper.cpp (Vulkan) backend for the local "Whisper" transcription engine.

faster-whisper/CTranslate2 only accelerates on NVIDIA (CUDA). whisper.cpp's
Vulkan backend runs on any Vulkan 1.2+ GPU - AMD Radeon, Intel Arc/iGPU and
NVIDIA alike - with no vendor SDK. This module lets the existing local
"Whisper" engine use it when the user picks a "(Vulkan)" device in the
transcription compute-device dropdown; the engine name, weight-size choice
(tiny ... large-v3-turbo) and Whisper thresholds stay exactly the same.

Design
------
whisper.cpp is run as a child process (`whisper-server`) on 127.0.0.1 and
called over HTTP, instead of being loaded into the sidecar process:

* No compiled Python binding is needed - the server binary is a stock
  whisper.cpp build (`-DGGML_VULKAN=ON`), see tools/whisper_cpp/.
* A GPU driver crash/hang (TDR) kills only the child, not VRCT; the provider
  restarts it on the next request.
* Mic and speaker transcribers share one server (one copy of the model in
  VRAM). whisper-server serialises requests with its own mutex, which is the
  same behaviour as a single faster-whisper model instance.

Weights are whisper.cpp's GGML files from the `ggerganov/whisper.cpp`
Hugging Face repo, stored under `weights/whisper_ggml/`. They are separate
from the CTranslate2 weights in `weights/whisper/` because the formats differ.
"""

from __future__ import annotations

import atexit
import math
import os
import threading
from os import path as os_path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from models.transcription.transcription_languages import transcription_lang
from models.vulkan_runtime import (
    LocalServerProcess,
    exeName,
    findRuntimeExe,
    getVulkanDeviceList,
    vulkanComputeDeviceEntries,
)
from utils import errorLogging, printLog

# --- weights -----------------------------------------------------------------

_GGML_REPO = "ggerganov/whisper.cpp"

# VRCT weight type -> GGML file in the whisper.cpp HF repo. Keys mirror
# transcription_whisper._MODELS so the existing weight-size selector works
# unchanged. "large-v3-turbo-int8" maps to the q8_0 quantisation, which is the
# whisper.cpp equivalent (similar size, ~same accuracy).
_GGML_MODELS: Dict[str, str] = {
    "tiny": "ggml-tiny.bin",
    "base": "ggml-base.bin",
    "small": "ggml-small.bin",
    "medium": "ggml-medium.bin",
    "large-v1": "ggml-large-v1.bin",
    "large-v2": "ggml-large-v2.bin",
    "large-v3": "ggml-large-v3.bin",
    "large-v3-turbo-int8": "ggml-large-v3-turbo-q8_0.bin",
    "large-v3-turbo": "ggml-large-v3-turbo.bin",
}

# whisper.cpp reads the first uint32 (little endian) and expects
# GGML_FILE_MAGIC = 0x67676d6c ("ggml").
_GGML_MAGIC = (0x67676D6C).to_bytes(4, "little")
_MIN_WEIGHT_BYTES = 1024 * 1024


def getWhisperCppWeightPath(root: str, weight_type: str) -> Optional[str]:
    filename = _GGML_MODELS.get(weight_type)
    if filename is None:
        return None
    return os_path.join(root, "weights", "whisper_ggml", filename)


def checkWhisperCppWeight(root: str, weight_type: str) -> bool:
    """Cheap availability check: the file exists, is not tiny and starts with
    the GGML magic. Downloads go to a `.part` file that is renamed only on
    success, so a file at the final path is never a truncated download."""
    weight_path = getWhisperCppWeightPath(root, weight_type)
    if weight_path is None or not os_path.isfile(weight_path):
        return False
    try:
        if os_path.getsize(weight_path) < _MIN_WEIGHT_BYTES:
            return False
        with open(weight_path, "rb") as f:
            return f.read(4) == _GGML_MAGIC
    except OSError:
        return False


def downloadWhisperCppWeight(
    root: str,
    weight_type: str,
    callback: Optional[Callable[[float], None]] = None,
    end_callback: Optional[Callable[[], None]] = None,
) -> None:
    """Download the GGML weight for `weight_type` if it is missing.

    Same contract as transcription_whisper.downloadWhisperWeight (progress
    callback 0.0-1.0, end_callback always called) so the controller's
    download flow and UI progress bar work unchanged.
    """
    # Imported lazily: transcription_whisper pulls in faster_whisper, which
    # is irrelevant for this backend.
    from models.transcription.transcription_whisper import downloadFile

    try:
        weight_path = getWhisperCppWeightPath(root, weight_type)
        if weight_path is not None and not checkWhisperCppWeight(root, weight_type):
            os.makedirs(os_path.dirname(weight_path), exist_ok=True)
            url = f"https://huggingface.co/{_GGML_REPO}/resolve/main/{_GGML_MODELS[weight_type]}"
            part_path = weight_path + ".part"
            if downloadFile(url, part_path, func=callback):
                os.replace(part_path, weight_path)
    except Exception:
        errorLogging()
    finally:
        if callable(end_callback):
            end_callback()


# --- server binary -------------------------------------------------------------

_SERVER_EXE = exeName("whisper-server")


def findWhisperCppServer(root: Optional[str] = None) -> Optional[str]:
    return findRuntimeExe("whisper-server", "whisper_cpp", "VRCT_WHISPER_CPP_DIR", root)


def isWhisperCppAvailable(root: Optional[str] = None) -> bool:
    return findWhisperCppServer(root) is not None


# --- Vulkan devices ---------------------------------------------------------------

def getVulkanComputeDeviceList(root: Optional[str] = None) -> List[Dict[str, Any]]:
    """Compute-device entries (same shape as utils.getComputeDeviceList) for
    the transcription device dropdown. Empty unless the whisper.cpp runtime
    is bundled, so builds without it never show a device that can't work."""
    if not isWhisperCppAvailable(root):
        return []
    return vulkanComputeDeviceEntries(getVulkanDeviceList())


# --- server process -------------------------------------------------------------

_REQUEST_TIMEOUT = (5, 120)


class WhisperCppServer(LocalServerProcess):
    """One whisper-server child process bound to 127.0.0.1:<random port>."""

    name = "whisper.cpp server"
    log_filename = "whisper_cpp_server.log"

    def __init__(self, exe: str, model_path: str, vulkan_index: int, log_dir: Optional[str] = None, threads: int = 4) -> None:
        super().__init__(exe, model_path, vulkan_index, log_dir)
        self.threads = threads

    @property
    def inference_url(self) -> str:
        return f"{self.base_url}/inference"

    def _serverArgs(self) -> List[str]:
        return [
            "--model", self.model_path,
            "--threads", str(self.threads),
            # GGML_VK_VISIBLE_DEVICES (set by LocalServerProcess) narrows ggml
            # to exactly the GPU the user picked, which then is ggml device 0.
            "--device", "0",
            # Language probabilities need an extra encoder pass per request;
            # VRCT does not use them (confidence comes from avg_logprob).
            "--no-language-probabilities",
        ]


# Shared by the mic and speaker transcribers. Re-created only when the model
# file or GPU changes, so toggling mic/speaker on and off does not reload the
# model.
_server_lock = threading.Lock()
_server: Optional[WhisperCppServer] = None


def acquireWhisperCppServer(root: str, weight_type: str, vulkan_index: int) -> WhisperCppServer:
    global _server
    exe = findWhisperCppServer(root)
    if exe is None:
        raise RuntimeError("whisper.cpp runtime (whisper-server) is not bundled with this build")
    model_path = getWhisperCppWeightPath(root, weight_type)
    if model_path is None or not checkWhisperCppWeight(root, weight_type):
        raise RuntimeError(f"whisper.cpp weight for {weight_type} is not downloaded")

    with _server_lock:
        if _server is not None and (_server.model_path != model_path or _server.vulkan_index != vulkan_index or _server.exe != exe):
            _server.stop()
            _server = None
        if _server is None:
            _server = WhisperCppServer(exe, model_path, vulkan_index, log_dir=os_path.join(root, "logs"))
        server = _server
    server.ensureStarted()
    return server


def shutdownWhisperCppServer() -> None:
    global _server
    with _server_lock:
        if _server is not None:
            _server.stop()
            _server = None


atexit.register(shutdownWhisperCppServer)


# --- provider -------------------------------------------------------------------

# whisper-server's verbose_json reports the detected language by its full
# name ("english"), while VRCT's transcription_lang uses Whisper's short codes
# ("en"). Generated from g_lang in whisper.cpp/src/whisper.cpp; short codes
# are accepted as-is too, in case a future server version switches.
_WHISPER_LANGUAGE_NAME_TO_CODE: Dict[str, str] = {
    "english": "en", "chinese": "zh", "german": "de", "spanish": "es", "russian": "ru",
    "korean": "ko", "french": "fr", "japanese": "ja", "portuguese": "pt", "turkish": "tr",
    "polish": "pl", "catalan": "ca", "dutch": "nl", "arabic": "ar", "swedish": "sv",
    "italian": "it", "indonesian": "id", "hindi": "hi", "finnish": "fi", "vietnamese": "vi",
    "hebrew": "he", "ukrainian": "uk", "greek": "el", "malay": "ms", "czech": "cs",
    "romanian": "ro", "danish": "da", "hungarian": "hu", "tamil": "ta", "norwegian": "no",
    "thai": "th", "urdu": "ur", "croatian": "hr", "bulgarian": "bg", "lithuanian": "lt",
    "latin": "la", "maori": "mi", "malayalam": "ml", "welsh": "cy", "slovak": "sk", "telugu": "te",
    "persian": "fa", "latvian": "lv", "bengali": "bn", "serbian": "sr", "azerbaijani": "az",
    "slovenian": "sl", "kannada": "kn", "estonian": "et", "macedonian": "mk", "breton": "br",
    "basque": "eu", "icelandic": "is", "armenian": "hy", "nepali": "ne", "mongolian": "mn",
    "bosnian": "bs", "kazakh": "kk", "albanian": "sq", "swahili": "sw", "galician": "gl",
    "marathi": "mr", "punjabi": "pa", "sinhala": "si", "khmer": "km", "shona": "sn", "yoruba": "yo",
    "somali": "so", "afrikaans": "af", "occitan": "oc", "georgian": "ka", "belarusian": "be",
    "tajik": "tg", "sindhi": "sd", "gujarati": "gu", "amharic": "am", "yiddish": "yi", "lao": "lo",
    "uzbek": "uz", "faroese": "fo", "haitian creole": "ht", "pashto": "ps", "turkmen": "tk",
    "nynorsk": "nn", "maltese": "mt", "sanskrit": "sa", "luxembourgish": "lb", "myanmar": "my",
    "tibetan": "bo", "tagalog": "tl", "malagasy": "mg", "assamese": "as", "tatar": "tt",
    "hawaiian": "haw", "lingala": "ln", "hausa": "ha", "bashkir": "ba", "javanese": "jw",
    "sundanese": "su", "cantonese": "yue",
}


def _toWhisperLanguageCode(language: Optional[str]) -> Optional[str]:
    if not language:
        return None
    language = language.strip().lower()
    return _WHISPER_LANGUAGE_NAME_TO_CODE.get(language, language)


class WhisperCppProvider:
    """TranscriptionProvider (see transcription_providers.py) backed by a
    whisper-server child process. Mirrors LocalWhisperProvider: same language
    codes (the "Whisper" column of transcription_lang), same segment
    filtering by avg_logprob / no_speech_prob, same early-exit rule."""

    def __init__(self, server: WhisperCppServer, beam_size: int = 5) -> None:
        self._server = server
        self._beam_size = beam_size

    def _post(self, wav_bytes: bytes, form: Dict[str, str]) -> Dict[str, Any]:
        response = requests.post(
            self._server.inference_url,
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data=form,
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    def transcribe(
        self,
        audio_data,
        language: str,
        country: str,
        *,
        avg_logprob: float,
        no_speech_prob: float,
        no_repeat_ngram_size: int,
        force_language: bool,
    ) -> Tuple[str, float, bool]:
        # no_repeat_ngram_size has no whisper.cpp equivalent; whisper.cpp's
        # entropy-based temperature fallback covers the same repetition loops.
        wav_bytes = audio_data.get_wav_data(convert_rate=16000, convert_width=2)
        requested_code = transcription_lang[language][country]["Whisper"]
        form = {
            "response_format": "verbose_json",
            "language": requested_code if force_language else "auto",
            "temperature": "0.0",
            "beam_size": str(self._beam_size),
            "logprob_thold": str(avg_logprob),
            "no_speech_thold": str(no_speech_prob),
            "no_timestamps": "true",
            "no_language_probabilities": "true",
        }

        try:
            self._server.ensureStarted()
            payload = self._post(wav_bytes, form)
        except (requests.ConnectionError, requests.Timeout):
            # The child died (driver reset, OOM, ...) or hung: restart once
            # and retry this clip, so a single GPU hiccup does not leave the
            # mic dead until VRCT is restarted.
            errorLogging()
            self._server.restart()
            payload = self._post(wav_bytes, form)

        accepted_logprobs: List[float] = []
        text = ""
        for segment in payload.get("segments") or []:
            seg_logprob = float(segment.get("avg_logprob", 0.0))
            seg_no_speech = float(segment.get("no_speech_prob", 0.0))
            if seg_logprob < avg_logprob or seg_no_speech > no_speech_prob:
                continue
            text += segment.get("text", "")
            accepted_logprobs.append(seg_logprob)

        if not text.strip():
            return "", 0.0, False

        confidence = math.exp(sum(accepted_logprobs) / len(accepted_logprobs)) if accepted_logprobs else 0.5
        is_definitive = force_language or _toWhisperLanguageCode(payload.get("language")) == requested_code
        return text, confidence, is_definitive
