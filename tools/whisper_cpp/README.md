# whisper.cpp (Vulkan) runtime

Lets the local **Whisper** transcription engine run on any Vulkan GPU
(AMD Radeon, Intel Arc / iGPU, NVIDIA) via [whisper.cpp](https://github.com/ggml-org/whisper.cpp),
not just on NVIDIA through CUDA/CTranslate2.

ローカル Whisper を whisper.cpp (Vulkan) 経由で AMD / Intel GPU でも動かすための
ランタイムです。CUDA 版 (CTranslate2) はそのまま残り、文字起こしの計算デバイスで
「(Vulkan)」付きのデバイスを選んだ場合だけこちらが使われます。

## How it works

* `whisper-server.exe` (stock whisper.cpp, `GGML_VULKAN=ON`, statically linked)
  is bundled as `_internal/whisper_cpp/`.
* If it is present, every Vulkan 1.2+ GPU is added to the **transcription**
  compute-device dropdown as `<GPU name> (Vulkan)`. The translation dropdown
  (CTranslate2) is unchanged.
* Selecting such a device switches the Whisper weights to the GGML files from
  `ggerganov/whisper.cpp` on Hugging Face (`weights/whisper_ggml/`); the
  weight-size selector and download button work as before.
* VRCT starts `whisper-server` on `127.0.0.1:<random port>` the first time
  transcription starts and talks to it over HTTP. Mic and speaker share one
  server (one copy of the model in VRAM). If the server dies (driver reset,
  OOM) it is restarted on the next phrase. It is killed together with VRCT
  (Windows Job Object). Its log is `logs/whisper_cpp_server.log`.

Code: `src-python/models/transcription/transcription_whisper_cpp.py`.

## Building the runtime

CI: `.github/workflows/build-whisper-cpp-vulkan.yml`.

Locally (Windows, needs Git, CMake, VS 2022 Build Tools and the
[Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows)):

```powershell
powershell -ExecutionPolicy Bypass -File tools\whisper_cpp\build_whisper_cpp_vulkan.ps1
npm run build
```

The script writes `tools/whisper_cpp/bin/whisper-server.exe` (git-ignored);
`spec/backend*.spec` bundle it when it exists. For `npm run dev-fast` the
sidecar finds it there directly. `VRCT_WHISPER_CPP_DIR` overrides the search
path.

## Weight mapping

| VRCT weight | GGML file |
|---|---|
| tiny / base / small / medium | `ggml-<size>.bin` |
| large-v1 / large-v2 / large-v3 | `ggml-<size>.bin` |
| large-v3-turbo | `ggml-large-v3-turbo.bin` |
| large-v3-turbo-int8 | `ggml-large-v3-turbo-q8_0.bin` |

## Differences from faster-whisper

* `no_repeat_ngram_size` has no whisper.cpp equivalent and is ignored
  (whisper.cpp's entropy-based temperature fallback handles repetition loops).
* Confidence for choosing between several candidate languages is
  `exp(mean avg_logprob)` of the accepted segments, as for the API engines.
