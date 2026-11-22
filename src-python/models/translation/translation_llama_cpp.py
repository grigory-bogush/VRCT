"""llama.cpp (Vulkan) backend for the local AI translation engine.

CTranslate2 (m2m100 / NLLB) only accelerates on NVIDIA (CUDA). This module
adds GGUF translation weights - Google's TranslateGemma, a Gemma 3 model
fine-tuned for translation - that run through llama.cpp's Vulkan backend on
any Vulkan 1.2+ GPU: AMD Radeon, Intel Arc/iGPU and NVIDIA alike.

They are offered as extra weight types of the existing local engine
("CTranslate2" in the code, "AI model" in the UI), so the engine selector,
the CTranslate2 fallback and the weight download UI keep working. Which
backend a weight needs is `getWeightBackend()`:

* "ctranslate2" weights run on the CPU/CUDA devices,
* "llama_cpp" weights run on the "(Vulkan)" devices.

llama-server runs as a child process on 127.0.0.1 (see models/vulkan_runtime)
and is called with a raw prompt on /completion: TranslateGemma's embedded
chat template is not usable through llama-server's chat endpoint (it wants
structured content with language codes), so the documented prompt is built
here instead.
"""

from __future__ import annotations

import atexit
import os
import threading
from os import path as os_path
from typing import Any, Callable, Dict, List, Optional

import requests

from models.vulkan_runtime import (
    LocalServerProcess,
    exeName,
    findRuntimeExe,
    getVulkanDeviceList,
    vulkanComputeDeviceEntries,
)
from utils import errorLogging, printLog

# --- weights -----------------------------------------------------------------

# weight type -> GGUF file. Community GGUF conversions (not gated) of
# google/translategemma-*-it, licensed under the Gemma Terms of Use
# (https://ai.google.dev/gemma/terms).
llama_cpp_weights: Dict[str, Dict[str, str]] = {
    "translategemma-4b-it-Q4_K_M": {
        "hf_repo": "mradermacher/translategemma-4b-it-GGUF",
        "filename": "translategemma-4b-it.Q4_K_M.gguf",
    },
    "translategemma-4b-it-Q8_0": {
        "hf_repo": "mradermacher/translategemma-4b-it-GGUF",
        "filename": "translategemma-4b-it.Q8_0.gguf",
    },
    "translategemma-12b-it-Q4_K_M": {
        "hf_repo": "mradermacher/translategemma-12b-it-GGUF",
        "filename": "translategemma-12b-it.Q4_K_M.gguf",
    },
}

DEFAULT_LLAMA_CPP_WEIGHT = "translategemma-4b-it-Q4_K_M"

BACKEND_CTRANSLATE2 = "ctranslate2"
BACKEND_LLAMA_CPP = "llama_cpp"

_GGUF_MAGIC = b"GGUF"
_MIN_WEIGHT_BYTES = 1024 * 1024


def isLlamaCppWeight(weight_type: str) -> bool:
    return weight_type in llama_cpp_weights


def getWeightBackend(weight_type: str) -> str:
    return BACKEND_LLAMA_CPP if isLlamaCppWeight(weight_type) else BACKEND_CTRANSLATE2


def getDeviceBackend(device: Optional[Dict[str, Any]]) -> str:
    """Backend a translation compute device runs: "(Vulkan)" devices run the
    llama.cpp weights, CPU/CUDA devices the CTranslate2 ones."""
    if isinstance(device, dict) and device.get("device") == "vulkan":
        return BACKEND_LLAMA_CPP
    return BACKEND_CTRANSLATE2


def isWeightCompatibleWithDevice(weight_type: str, device: Optional[Dict[str, Any]]) -> bool:
    return getWeightBackend(weight_type) == getDeviceBackend(device)


def getLlamaCppWeightPath(root: str, weight_type: str) -> Optional[str]:
    info = llama_cpp_weights.get(weight_type)
    if info is None:
        return None
    return os_path.join(root, "weights", "llama_cpp", info["filename"])


def checkLlamaCppWeight(root: str, weight_type: str) -> bool:
    """Cheap availability check: the file exists, is not tiny and starts with
    the GGUF magic. Downloads go to a `.part` file that is renamed only on
    success, so a file at the final path is never a truncated download."""
    weight_path = getLlamaCppWeightPath(root, weight_type)
    if weight_path is None or not os_path.isfile(weight_path):
        return False
    try:
        if os_path.getsize(weight_path) < _MIN_WEIGHT_BYTES:
            return False
        with open(weight_path, "rb") as f:
            return f.read(4) == _GGUF_MAGIC
    except OSError:
        return False


def downloadLlamaCppWeight(
    root: str,
    weight_type: str,
    callback: Optional[Callable[[float], None]] = None,
    end_callback: Optional[Callable[[], None]] = None,
) -> bool:
    """Download the GGUF weight for `weight_type` if it is missing.

    Same contract as translation_utils.downloadCTranslate2Weight (progress
    callback 0.0-1.0, end_callback always called) so the controller's
    download flow and UI progress bar work unchanged.
    """
    # Imported lazily: transcription_whisper pulls in faster_whisper.
    from models.transcription.transcription_whisper import downloadFile

    ok = False
    try:
        weight_path = getLlamaCppWeightPath(root, weight_type)
        if weight_path is None:
            return False
        if checkLlamaCppWeight(root, weight_type):
            ok = True
            return True
        info = llama_cpp_weights[weight_type]
        os.makedirs(os_path.dirname(weight_path), exist_ok=True)
        url = f"https://huggingface.co/{info['hf_repo']}/resolve/main/{info['filename']}"
        part_path = weight_path + ".part"
        if downloadFile(url, part_path, func=callback):
            os.replace(part_path, weight_path)
            ok = checkLlamaCppWeight(root, weight_type)
        return ok
    except Exception:
        errorLogging()
        return False
    finally:
        if callable(end_callback):
            end_callback()


# --- server binary -----------------------------------------------------------

_SERVER_EXE = exeName("llama-server")


def findLlamaCppServer(root: Optional[str] = None) -> Optional[str]:
    return findRuntimeExe("llama-server", "llama_cpp", "VRCT_LLAMA_CPP_DIR", root)


def isLlamaCppAvailable(root: Optional[str] = None) -> bool:
    return findLlamaCppServer(root) is not None


def getTranslationVulkanComputeDeviceList(root: Optional[str] = None) -> List[Dict[str, Any]]:
    """"(Vulkan)" entries for the translation compute-device dropdown. Empty
    unless llama-server is bundled, so builds without it never offer a
    device that can't work."""
    if not isLlamaCppAvailable(root):
        return []
    return vulkanComputeDeviceEntries(getVulkanDeviceList())


# --- server process ------------------------------------------------------------

_REQUEST_TIMEOUT = (5, 60)
# Chat lines are short; this bounds a runaway generation, not real output.
_MAX_OUTPUT_TOKENS = 512


class LlamaCppServer(LocalServerProcess):
    """One llama-server child process bound to 127.0.0.1:<random port>."""

    name = "llama.cpp server"
    log_filename = "llama_cpp_server.log"

    def __init__(self, exe: str, model_path: str, vulkan_index: int, log_dir: Optional[str] = None, context_size: int = 2048) -> None:
        super().__init__(exe, model_path, vulkan_index, log_dir)
        self.context_size = context_size

    @property
    def completion_url(self) -> str:
        return f"{self.base_url}/completion"

    def _startLocked(self) -> None:
        super()._startLocked()
        # The first request on a fresh Vulkan context compiles pipelines and
        # takes ~2 s; do it now so the user's first chat line isn't slow.
        try:
            requests.post(
                self.completion_url,
                json={"prompt": buildTranslateGemmaPrompt("Hello", "en", "ja"), "n_predict": 4, "temperature": 0.0},
                timeout=_REQUEST_TIMEOUT,
            )
        except requests.RequestException:
            pass

    def _serverArgs(self) -> List[str]:
        return [
            "--model", self.model_path,
            # GGML_VK_VISIBLE_DEVICES (set by LocalServerProcess) leaves only
            # the GPU the user picked, which then is "Vulkan0".
            "--device", "Vulkan0",
            "--gpu-layers", "999",
            "--ctx-size", str(self.context_size),
            "--parallel", "1",
            "--no-webui",
            # TranslateGemma's embedded Jinja template can't be used through
            # the chat endpoint (and can fail to parse); /completion is
            # called with a prompt built in Python, see buildTranslateGemmaPrompt.
            "--no-jinja",
            "--chat-template", "gemma",
        ]


_server_lock = threading.Lock()
_server: Optional[LlamaCppServer] = None


def acquireLlamaCppServer(root: str, weight_type: str, vulkan_index: int) -> LlamaCppServer:
    """Start (or reuse) the shared llama-server for `weight_type` on the GPU
    with raw Vulkan index `vulkan_index`. Mic, speaker and chat translations
    share it; it is re-created only when the weight or GPU changes."""
    global _server
    exe = findLlamaCppServer(root)
    if exe is None:
        raise RuntimeError("llama.cpp runtime (llama-server) is not bundled with this build")
    model_path = getLlamaCppWeightPath(root, weight_type)
    if model_path is None or not checkLlamaCppWeight(root, weight_type):
        raise RuntimeError(f"llama.cpp weight {weight_type} is not downloaded")

    with _server_lock:
        if _server is not None and (_server.model_path != model_path or _server.vulkan_index != vulkan_index or _server.exe != exe):
            _server.stop()
            _server = None
        if _server is None:
            _server = LlamaCppServer(exe, model_path, vulkan_index, log_dir=os_path.join(root, "logs"))
        server = _server
    server.ensureStarted()
    return server


def shutdownLlamaCppServer() -> None:
    global _server
    with _server_lock:
        if _server is not None:
            _server.stop()
            _server = None


atexit.register(shutdownLlamaCppServer)


# --- translation ---------------------------------------------------------------

# Language code (the values in languages.yml under CTranslate2:<gguf weight>)
# -> English name, as TranslateGemma's prompt names both: "Japanese (ja)".
_LANGUAGE_NAMES: Dict[str, str] = {
    "af": "Afrikaans", "ar": "Arabic", "hy": "Armenian", "az": "Azerbaijani", "be": "Belarusian",
    "bs": "Bosnian", "bg": "Bulgarian", "ca": "Catalan", "zh": "Chinese", "zh-TW": "Chinese",
    "hr": "Croatian", "cs": "Czech", "da": "Danish", "nl": "Dutch", "en": "English", "et": "Estonian",
    "fi": "Finnish", "fr": "French", "gl": "Galician", "de": "German", "el": "Greek", "he": "Hebrew",
    "hi": "Hindi", "hu": "Hungarian", "is": "Icelandic", "id": "Indonesian", "it": "Italian",
    "ja": "Japanese", "kn": "Kannada", "kk": "Kazakh", "ko": "Korean", "lv": "Latvian",
    "lt": "Lithuanian", "mk": "Macedonian", "ms": "Malay", "mr": "Marathi", "mi": "Maori",
    "ne": "Nepali", "no": "Norwegian", "fa": "Persian", "pl": "Polish", "pt": "Portuguese",
    "ro": "Romanian", "ru": "Russian", "sr": "Serbian", "sk": "Slovak", "sl": "Slovenian",
    "es": "Spanish", "sw": "Swahili", "sv": "Swedish", "tl": "Tagalog", "ta": "Tamil", "th": "Thai",
    "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese", "cy": "Welsh",
}


def _languageLabel(code: str) -> str:
    return _LANGUAGE_NAMES.get(code, code)


def buildTranslateGemmaPrompt(message: str, source_code: str, target_code: str) -> str:
    """TranslateGemma's documented single-turn prompt (from the chat
    template in the model's GGUF metadata). llama-server adds <bos>."""
    src = _languageLabel(source_code)
    tgt = _languageLabel(target_code)
    instruction = (
        f"You are a professional {src} ({source_code}) to {tgt} ({target_code}) translator. "
        f"Your goal is to accurately convey the meaning and nuances of the original {src} text "
        f"while adhering to {tgt} grammar, vocabulary, and cultural sensitivities.\n"
        f"Produce only the {tgt} translation, without any additional explanations or commentary. "
        f"Please translate the following {src} text into {tgt}:\n\n\n"
    )
    return f"<start_of_turn>user\n{instruction}{message.strip()}<end_of_turn>\n<start_of_turn>model\n"


class LlamaCppTranslator:
    """Translates through a shared llama-server (see acquireLlamaCppServer)."""

    def __init__(self, server: LlamaCppServer) -> None:
        self._server = server

    @property
    def server(self) -> LlamaCppServer:
        return self._server

    def _post(self, prompt: str) -> Dict[str, Any]:
        response = requests.post(
            self._server.completion_url,
            json={
                "prompt": prompt,
                "n_predict": _MAX_OUTPUT_TOKENS,
                "temperature": 0.0,
                "top_k": 1,
                "stop": ["<end_of_turn>", "<start_of_turn>"],
                "cache_prompt": True,
            },
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    def translate(self, message: str, source_code: str, target_code: str) -> str:
        prompt = buildTranslateGemmaPrompt(message, source_code, target_code)
        try:
            self._server.ensureStarted()
            payload = self._post(prompt)
        except (requests.ConnectionError, requests.Timeout):
            # The child died (driver reset, OOM, ...) or hung: restart once
            # and retry, so one GPU hiccup does not break translation until
            # VRCT is restarted.
            errorLogging()
            self._server.restart()
            payload = self._post(prompt)
        text = str(payload.get("content", "")).strip()
        timings = payload.get("timings") or {}
        if timings:
            printLog(
                "[MT-stats] llama.cpp",
                f"prompt={timings.get('prompt_ms', 0):.0f}ms gen={timings.get('predicted_ms', 0):.0f}ms "
                f"tokens={timings.get('predicted_n', 0)}",
            )
        return text
