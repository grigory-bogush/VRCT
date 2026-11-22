"""Shared plumbing for the Vulkan (ggml) runtimes VRCT runs as child processes.

Both local GPU backends that work on AMD/Intel as well as NVIDIA are stock
ggml programs started next to the sidecar and called over HTTP on 127.0.0.1:

* whisper.cpp's ``whisper-server`` for the local Whisper transcription engine
  (models/transcription/transcription_whisper_cpp.py)
* llama.cpp's ``llama-server`` for the local AI translation engine's GGUF
  weights (models/translation/translation_llama_cpp.py)

This module holds what they share: locating the bundled binary, Vulkan GPU
enumeration, and a child-process manager (random port, health wait, Windows
Job Object kill-on-close, log file).
"""

from __future__ import annotations

import ctypes
import os
import socket
import subprocess
import sys
import threading
import time
from os import path as os_path
from typing import Any, Dict, List, Optional

import requests

from utils import errorLogging, printLog


# --- bundled binaries -----------------------------------------------------------

def exeName(stem: str) -> str:
    return f"{stem}.exe" if sys.platform == "win32" else stem


def candidateRuntimeDirs(subdir: str, env_var: str, root: Optional[str]) -> List[str]:
    """Directories searched for a runtime named `subdir` ("whisper_cpp",
    "llama_cpp"), in priority order: env override, PyInstaller bundle,
    VRCT folder, dev checkout (tools/<subdir>/bin)."""
    dirs: List[str] = []
    env_dir = os.environ.get(env_var)
    if env_dir:
        dirs.append(env_dir)
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        # PyInstaller onedir: spec/backend.spec copies the runtime here.
        dirs.append(os_path.join(meipass, subdir))
    if root:
        dirs.append(os_path.join(root, subdir))
    repo_root = os_path.abspath(os_path.join(os_path.dirname(__file__), "..", ".."))
    dirs.append(os_path.join(repo_root, "tools", subdir, "bin"))
    return dirs


def findRuntimeExe(stem: str, subdir: str, env_var: str, root: Optional[str] = None) -> Optional[str]:
    for directory in candidateRuntimeDirs(subdir, env_var, root):
        candidate = os_path.join(directory, exeName(stem))
        if os_path.isfile(candidate):
            return candidate
    return None


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


_device_cache_lock = threading.Lock()
_device_cache: Optional[List[Dict[str, Any]]] = None


def getVulkanDeviceList(use_cache: bool = True) -> List[Dict[str, Any]]:
    """Return GPUs visible through the Vulkan loader as
    [{"vulkan_index": int, "name": str, "vendor_id": int}, ...].

    `vulkan_index` is the raw vkEnumeratePhysicalDevices index, which is what
    ggml's GGML_VK_VISIBLE_DEVICES expects. Only discrete/integrated GPUs are
    returned (same filter ggml-vulkan applies), so software rasterisers such
    as llvmpipe never show up. Returns [] when there is no Vulkan loader or
    driver; never raises. The result is cached: both the transcription and
    the translation device lists ask for it at startup.
    """
    global _device_cache
    with _device_cache_lock:
        if use_cache and _device_cache is not None:
            return [dict(d) for d in _device_cache]
        devices = _enumerateVulkanDevices()
        _device_cache = devices
        return [dict(d) for d in devices]


def _enumerateVulkanDevices() -> List[Dict[str, Any]]:
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


def vulkanComputeDeviceEntries(devices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute-device dropdown entries (same shape as utils.getComputeDeviceList)."""
    return [
        {
            "device": "vulkan",
            "device_index": d["vulkan_index"],
            "device_name": f"{d['name']} (Vulkan)",
            "compute_types": ["auto"],
        }
        for d in devices
    ]


# --- child process ------------------------------------------------------------

def pickFreePort() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _KillOnCloseJob:
    """Windows Job Object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, so the
    server child (and the VRAM it holds) dies with the sidecar even if the
    sidecar is killed without running atexit handlers."""

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


_job_lock = threading.Lock()
_job: Optional[_KillOnCloseJob] = None


def _assignToKillOnCloseJob(process: subprocess.Popen) -> None:
    global _job
    with _job_lock:
        if _job is None:
            _job = _KillOnCloseJob()
    _job.assign(process)


class LocalServerProcess:
    """One ggml server child process bound to 127.0.0.1:<random port>.

    Subclasses provide `_serverArgs()` (everything after the exe, without
    host/port) and set `name` (for logs) and `log_filename`.
    """

    name = "server"
    log_filename = "server.log"
    health_timeout_sec = 180  # large models from a slow HDD can take a while

    def __init__(self, exe: str, model_path: str, vulkan_index: int, log_dir: Optional[str] = None) -> None:
        self.exe = exe
        self.model_path = model_path
        self.vulkan_index = vulkan_index
        self.log_dir = log_dir
        self.port: Optional[int] = None
        self._process: Optional[subprocess.Popen] = None
        self._log_file = None
        self._lock = threading.Lock()

    # -- subclass hooks
    def _serverArgs(self) -> List[str]:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- public
    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

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

    # -- internals
    def _startLocked(self) -> None:
        self.port = pickFreePort()
        args = [self.exe, "--host", "127.0.0.1", "--port", str(self.port)] + self._serverArgs()
        env = dict(os.environ)
        # Narrows ggml to exactly the GPU the user picked, which then is ggml
        # device 0 (raw vkEnumeratePhysicalDevices index, see
        # getVulkanDeviceList).
        env["GGML_VK_VISIBLE_DEVICES"] = str(self.vulkan_index)

        stdout = subprocess.DEVNULL
        if self.log_dir:
            try:
                os.makedirs(self.log_dir, exist_ok=True)
                self._log_file = open(os_path.join(self.log_dir, self.log_filename), "ab")
                stdout = self._log_file
            except OSError:
                self._log_file = None

        creationflags = 0
        if sys.platform == "win32":
            creationflags = 0x08000000  # CREATE_NO_WINDOW

        printLog(f"Starting {self.name}", f"{' '.join(args[1:])} GGML_VK_VISIBLE_DEVICES={self.vulkan_index}")
        self._process = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=os_path.dirname(self.exe),
            creationflags=creationflags,
        )
        _assignToKillOnCloseJob(self._process)

        deadline = time.monotonic() + self.health_timeout_sec
        health_url = f"{self.base_url}/health"
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                code = self._process.returncode
                self._stopLocked()
                raise RuntimeError(f"{self.name} exited during startup (code {code}); see logs/{self.log_filename}")
            try:
                # llama-server answers 503 while the model is loading.
                if requests.get(health_url, timeout=1).status_code == 200:
                    printLog(f"{self.name} ready", self.port)
                    return
            except requests.RequestException:
                pass
            time.sleep(0.2)
        self._stopLocked()
        raise RuntimeError(f"{self.name} did not become ready in time")

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
