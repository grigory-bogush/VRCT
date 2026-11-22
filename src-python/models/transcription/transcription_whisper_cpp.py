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
import ctypes
import math
import os
import socket
import subprocess
import sys
import threading
import time
from os import path as os_path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from models.transcription.transcription_languages import transcription_lang
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

_SERVER_EXE = "whisper-server.exe" if sys.platform == "win32" else "whisper-server"


def _candidateRuntimeDirs(root: Optional[str]) -> List[str]:
    dirs: List[str] = []
    env_dir = os.environ.get("VRCT_WHISPER_CPP_DIR")
    if env_dir:
        dirs.append(env_dir)
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        # PyInstaller onedir: spec/backend.spec copies the runtime here.
        dirs.append(os_path.join(meipass, "whisper_cpp"))
    if root:
        dirs.append(os_path.join(root, "whisper_cpp"))
    # Dev checkout: tools/whisper_cpp/fetch_runtime.* unpacks here.
    repo_root = os_path.abspath(os_path.join(os_path.dirname(__file__), "..", "..", ".."))
    dirs.append(os_path.join(repo_root, "tools", "whisper_cpp", "bin"))
    return dirs


def findWhisperCppServer(root: Optional[str] = None) -> Optional[str]:
    for directory in _candidateRuntimeDirs(root):
        candidate = os_path.join(directory, _SERVER_EXE)
        if os_path.isfile(candidate):
            return candidate
    return None


def isWhisperCppAvailable(root: Optional[str] = None) -> bool:
    return findWhisperCppServer(root) is not None


# --- Vulkan device enumeration ------------------------------------------------

_VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1
_VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2


class _VkApplicationInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("pApplicationName", ctypes.c_char_p),
        ("applicationVersion", ctypes.c_uint32),
        ("pEngineName", ctypes.c_char_p),
        ("engineVersion", ctypes.c_uint32),
        ("apiVersion", ctypes.c_uint32),
    ]


class _VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.POINTER(_VkApplicationInfo)),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.c_void_p),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.c_void_p),
    ]


def _loadVulkanLoader():
    names = ["vulkan-1.dll"] if sys.platform == "win32" else ["libvulkan.so.1", "libvulkan.so"]
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def getVulkanDeviceList() -> List[Dict[str, Any]]:
    """Return GPUs visible through the Vulkan loader as
    [{"vulkan_index": int, "name": str, "vendor_id": int}, ...].

    `vulkan_index` is the raw vkEnumeratePhysicalDevices index, which is what
    ggml's GGML_VK_VISIBLE_DEVICES expects. Only discrete/integrated GPUs are
    returned (same filter ggml-vulkan applies), so software rasterisers such
    as llvmpipe never show up. Returns [] when there is no Vulkan loader or
    driver; never raises.
    """
    loader = _loadVulkanLoader()
    if loader is None:
        return []

    try:
        vkCreateInstance = loader.vkCreateInstance
        vkCreateInstance.restype = ctypes.c_int
        vkCreateInstance.argtypes = [ctypes.POINTER(_VkInstanceCreateInfo), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        vkEnumeratePhysicalDevices = loader.vkEnumeratePhysicalDevices
        vkEnumeratePhysicalDevices.restype = ctypes.c_int
        vkEnumeratePhysicalDevices.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
        vkGetPhysicalDeviceProperties = loader.vkGetPhysicalDeviceProperties
        vkGetPhysicalDeviceProperties.restype = None
        vkGetPhysicalDeviceProperties.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        vkDestroyInstance = loader.vkDestroyInstance
        vkDestroyInstance.restype = None
        vkDestroyInstance.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        app_info = _VkApplicationInfo(
            sType=0,  # VK_STRUCTURE_TYPE_APPLICATION_INFO
            pApplicationName=b"VRCT",
            apiVersion=(1 << 22) | (2 << 12),  # VK_API_VERSION_1_2
        )
        create_info = _VkInstanceCreateInfo(
            sType=1,  # VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
            pApplicationInfo=ctypes.pointer(app_info),
        )
        instance = ctypes.c_void_p()
        if vkCreateInstance(ctypes.byref(create_info), None, ctypes.byref(instance)) != 0:
            return []

        devices: List[Dict[str, Any]] = []
        try:
            count = ctypes.c_uint32(0)
            if vkEnumeratePhysicalDevices(instance, ctypes.byref(count), None) != 0 or count.value == 0:
                return []
            handles = (ctypes.c_void_p * count.value)()
            if vkEnumeratePhysicalDevices(instance, ctypes.byref(count), handles) not in (0, 5):  # VK_SUCCESS / VK_INCOMPLETE
                return []

            for index in range(count.value):
                # VkPhysicalDeviceProperties is ~824 bytes; only the leading
                # fields are read: apiVersion, driverVersion, vendorID,
                # deviceID, deviceType (uint32 each) then char deviceName[256].
                props = ctypes.create_string_buffer(4096)
                vkGetPhysicalDeviceProperties(handles[index], props)
                raw = props.raw
                api_version = int.from_bytes(raw[0:4], "little")
                vendor_id = int.from_bytes(raw[8:12], "little")
                device_type = int.from_bytes(raw[16:20], "little")
                name = raw[20:276].split(b"\0", 1)[0].decode("utf-8", errors="replace").strip()
                if device_type not in (_VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, _VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU):
                    continue
                # ggml-vulkan needs Vulkan 1.2.
                if (api_version >> 22) == 1 and ((api_version >> 12) & 0x3FF) < 2:
                    continue
                devices.append({"vulkan_index": index, "name": name or f"Vulkan GPU {index}", "vendor_id": vendor_id})
        finally:
            vkDestroyInstance(instance, None)
        return devices
    except Exception:
        errorLogging()
        return []


def getVulkanComputeDeviceList(root: Optional[str] = None) -> List[Dict[str, Any]]:
    """Compute-device entries (same shape as utils.getComputeDeviceList) for
    the transcription device dropdown. Empty unless the whisper.cpp runtime
    is bundled, so builds without it never show a device that can't work."""
    if not isWhisperCppAvailable(root):
        return []
    return [
        {
            "device": "vulkan",
            "device_index": d["vulkan_index"],
            "device_name": f"{d['name']} (Vulkan)",
            "compute_types": ["auto"],
        }
        for d in getVulkanDeviceList()
    ]


# --- server process -------------------------------------------------------------

_HEALTH_TIMEOUT_SEC = 180  # large-v3 from a slow HDD can take a while to load
_REQUEST_TIMEOUT = (5, 120)


def _pickFreePort() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _KillOnCloseJob:
    """Windows Job Object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, so the
    whisper-server child (and the VRAM it holds) dies with the sidecar even
    if the sidecar is killed without running atexit handlers."""

    def __init__(self) -> None:
        self._handle = None
        if sys.platform != "win32":
            return
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

            class _IO_COUNTERS(ctypes.Structure):
                _fields_ = [(n, ctypes.c_ulonglong) for n in (
                    "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                    "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

            class _BASIC_LIMIT(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", ctypes.c_longlong),
                    ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", ctypes.c_uint32),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", ctypes.c_uint32),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", ctypes.c_uint32),
                    ("SchedulingClass", ctypes.c_uint32),
                ]

            class _EXTENDED_LIMIT(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", _BASIC_LIMIT),
                    ("IoInfo", _IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            kernel32.CreateJobObjectW.restype = ctypes.c_void_p
            kernel32.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
            kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            handle = kernel32.CreateJobObjectW(None, None)
            if not handle:
                return
            info = _EXTENDED_LIMIT()
            info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            # 9 = JobObjectExtendedLimitInformation
            if not kernel32.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
                return
            self._kernel32 = kernel32
            self._handle = handle
        except Exception:
            errorLogging()
            self._handle = None

    def assign(self, process: subprocess.Popen) -> None:
        if self._handle is None:
            return
        try:
            self._kernel32.AssignProcessToJobObject(self._handle, ctypes.c_void_p(int(process._handle)))  # type: ignore[attr-defined]
        except Exception:
            errorLogging()


_job: Optional[_KillOnCloseJob] = None


class WhisperCppServer:
    """One whisper-server child process bound to 127.0.0.1:<random port>."""

    def __init__(self, exe: str, model_path: str, vulkan_index: int, log_dir: Optional[str] = None, threads: int = 4) -> None:
        self.exe = exe
        self.model_path = model_path
        self.vulkan_index = vulkan_index
        self.log_dir = log_dir
        self.threads = threads
        self.port: Optional[int] = None
        self._process: Optional[subprocess.Popen] = None
        self._log_file = None
        self._lock = threading.Lock()

    @property
    def inference_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/inference"

    def isAlive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def ensureStarted(self) -> None:
        with self._lock:
            if self.isAlive():
                return
            self._stopLocked()
            self._startLocked()

    def restart(self) -> None:
        with self._lock:
            self._stopLocked()
            self._startLocked()

    def stop(self) -> None:
        with self._lock:
            self._stopLocked()

    def _startLocked(self) -> None:
        global _job
        self.port = _pickFreePort()
        args = [
            self.exe,
            "--model", self.model_path,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--threads", str(self.threads),
            # GGML_VK_VISIBLE_DEVICES below narrows ggml to exactly the GPU the
            # user picked, which then is ggml device 0.
            "--device", "0",
            # Language probabilities need an extra encoder pass per request;
            # VRCT does not use them (confidence comes from avg_logprob).
            "--no-language-probabilities",
        ]
        env = dict(os.environ)
        env["GGML_VK_VISIBLE_DEVICES"] = str(self.vulkan_index)

        stdout = subprocess.DEVNULL
        if self.log_dir:
            try:
                os.makedirs(self.log_dir, exist_ok=True)
                self._log_file = open(os_path.join(self.log_dir, "whisper_cpp_server.log"), "ab")
                stdout = self._log_file
            except OSError:
                self._log_file = None

        creationflags = 0
        if sys.platform == "win32":
            creationflags = 0x08000000  # CREATE_NO_WINDOW

        printLog("Starting whisper.cpp server", f"{' '.join(args[1:])} GGML_VK_VISIBLE_DEVICES={self.vulkan_index}")
        self._process = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=os_path.dirname(self.exe),
            creationflags=creationflags,
        )
        if _job is None:
            _job = _KillOnCloseJob()
        _job.assign(self._process)

        deadline = time.monotonic() + _HEALTH_TIMEOUT_SEC
        health_url = f"http://127.0.0.1:{self.port}/health"
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                code = self._process.returncode
                self._stopLocked()
                raise RuntimeError(f"whisper.cpp server exited during startup (code {code}); see logs/whisper_cpp_server.log")
            try:
                if requests.get(health_url, timeout=1).status_code == 200:
                    printLog("whisper.cpp server ready", self.port)
                    return
            except requests.RequestException:
                pass
            time.sleep(0.2)
        self._stopLocked()
        raise RuntimeError("whisper.cpp server did not become ready in time")

    def _stopLocked(self) -> None:
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        if self._log_file is not None:
            try:
                self._log_file.close()
            except OSError:
                pass
            self._log_file = None


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
