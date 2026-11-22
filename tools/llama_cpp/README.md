# llama.cpp (Vulkan) runtime

Lets the local AI translation engine run on any Vulkan GPU (AMD Radeon,
Intel Arc / iGPU, NVIDIA) via [llama.cpp](https://github.com/ggml-org/llama.cpp).
CTranslate2 (m2m100 / NLLB) only uses NVIDIA GPUs through CUDA; this adds
GGUF weights of Google's [TranslateGemma](https://huggingface.co/google/translategemma-4b-it),
a Gemma 3 model fine-tuned for translation.

ローカル AI 翻訳を llama.cpp (Vulkan) 経由で AMD / Intel GPU でも動かすための
ランタイムです。翻訳の処理デバイスで「(Vulkan)」付きの GPU を選ぶと、AI 翻訳
モデルが TranslateGemma (GGUF) に切り替わります。CTranslate2 のモデルは
CPU / NVIDIA 用としてそのまま残ります。

## How it works

* `llama-server.exe` (stock llama.cpp, `GGML_VULKAN=ON`, statically linked)
  is bundled as `_internal/llama_cpp/`.
* If it is present, every Vulkan 1.2+ GPU is added to the **translation**
  compute-device dropdown as `<GPU name> (Vulkan)`.
* The AI translation weight list gains three GGUF weights (marked
  "AMD / Intel GPU" in the UI):

  | Weight | File | Size |
  |---|---|---|
  | `translategemma-4b-it-Q4_K_M` (default) | `mradermacher/translategemma-4b-it-GGUF` / `translategemma-4b-it.Q4_K_M.gguf` | 2.5 GB |
  | `translategemma-4b-it-Q8_0` | `mradermacher/translategemma-4b-it-GGUF` / `translategemma-4b-it.Q8_0.gguf` | 4.1 GB |
  | `translategemma-12b-it-Q4_K_M` | `mradermacher/translategemma-12b-it-GGUF` / `translategemma-12b-it.Q4_K_M.gguf` | 7.3 GB |

  They are stored in `weights/llama_cpp/`.
* GGUF weights run only on a "(Vulkan)" device and CTranslate2 weights only
  on CPU/CUDA. The sidecar keeps the two aligned: picking a Vulkan device
  switches to a GGUF weight (a downloaded one if any), picking a GGUF weight
  switches the device to the first Vulkan GPU, and back.
* VRCT starts `llama-server` on `127.0.0.1:<random port>` when translation is
  enabled and calls `/completion` with TranslateGemma's documented prompt
  (built in Python; the model's embedded Jinja template is not used). The
  server is killed together with VRCT (Windows Job Object). Its log is
  `logs/llama_cpp_server.log`.

Measured on an RX 7800 XT with the 4B Q4_K_M weight: ~150-250 ms per chat
line after warm-up.

Code: `src-python/models/translation/translation_llama_cpp.py`, shared process
and Vulkan plumbing in `src-python/models/vulkan_runtime.py`.

## Building the runtime

```powershell
powershell -ExecutionPolicy Bypass -File tools\llama_cpp\build_llama_cpp_vulkan.ps1
```

Same requirements as `tools/whisper_cpp/` (Git, CMake, VS 2022 C++ tools,
Vulkan SDK). The script writes `tools/llama_cpp/bin/llama-server.exe`
(git-ignored); `spec/backend*.spec` bundle it when it exists.
`VRCT_LLAMA_CPP_DIR` overrides the search path.

## License

llama.cpp is MIT. TranslateGemma is provided under the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms); VRCT downloads the
weights from Hugging Face on demand and does not redistribute them.
