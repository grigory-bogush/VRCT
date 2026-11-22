# Builds llama.cpp's `llama-server.exe` with the Vulkan backend and puts it in
# tools/llama_cpp/bin/, where spec/backend*.spec pick it up and bundle it as
# `llama_cpp/` next to the sidecar. Used by the local AI translation engine
# for its GGUF (TranslateGemma) weights on AMD/Intel/NVIDIA GPUs.
#
# Requirements (Windows): same as tools/whisper_cpp/build_whisper_cpp_vulkan.ps1
#   - Git, CMake 3.21+
#   - Visual Studio 2022 Build Tools (C++ workload)
#   - Vulkan SDK (sets %VULKAN_SDK%; provides glslc for shader compilation)
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File tools\llama_cpp\build_llama_cpp_vulkan.ps1
#   powershell -ExecutionPolicy Bypass -File tools\llama_cpp\build_llama_cpp_vulkan.ps1 -Version b12359
#
# The result is a single statically linked llama-server.exe. At run time it
# only needs vulkan-1.dll, which every GPU driver installs.

param(
    # b12359 is the build behind llama.cpp's v1.0.0 release.
    [string]$Version = "b12359"
)

$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$src = Join-Path $here "src"
$build = Join-Path $here "build"
$bin = Join-Path $here "bin"

if (-not $env:VULKAN_SDK) {
    throw "VULKAN_SDK is not set. Install the Vulkan SDK from https://vulkan.lunarg.com/sdk/home#windows first."
}

# A shallow clone of a tag carries no tag ref, so `git describe` can't tell
# which version is checked out; remember it in a stamp file instead.
$stamp = Join-Path $here ".src-version"
if (Test-Path $src) {
    $current = if (Test-Path $stamp) { (Get-Content $stamp -Raw).Trim() } else { "" }
    if ($current -ne $Version) {
        Remove-Item -Recurse -Force $src
        if (Test-Path $build) { Remove-Item -Recurse -Force $build }
    }
}
if (-not (Test-Path $src)) {
    git clone --depth 1 --branch $Version https://github.com/ggml-org/llama.cpp.git $src
    if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
    Set-Content -Path $stamp -Value $Version -Encoding ascii
}

# The shallow clone has no history, so llama.cpp would report build number 1
# and fail to fetch its (mandatory, embedded) server UI assets for "b1".
$buildNumberArgs = @()
if ($Version -match '^b(\d+)$') {
    $buildNumberArgs = @("-DLLAMA_BUILD_NUMBER=$($Matches[1])")
}

cmake -S $src -B $build @buildNumberArgs `
    -DCMAKE_BUILD_TYPE=Release `
    -DBUILD_SHARED_LIBS=OFF `
    -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded `
    -DGGML_VULKAN=ON `
    -DGGML_NATIVE=OFF `
    -DLLAMA_CURL=OFF `
    -DLLAMA_BUILD_TESTS=OFF `
    -DLLAMA_BUILD_EXAMPLES=OFF `
    -DLLAMA_BUILD_TOOLS=ON `
    -DLLAMA_BUILD_SERVER=ON
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

cmake --build $build --config Release --target llama-server -j
if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }

New-Item -ItemType Directory -Force -Path $bin | Out-Null
$exe = Get-ChildItem -Path $build -Recurse -Filter "llama-server.exe" | Select-Object -First 1
if (-not $exe) { throw "llama-server.exe not found under $build" }
Copy-Item $exe.FullName (Join-Path $bin "llama-server.exe") -Force
Copy-Item (Join-Path $src "LICENSE") (Join-Path $bin "LICENSE-llama.cpp.txt") -Force
Set-Content -Path (Join-Path $bin "VERSION.txt") -Value "llama.cpp $Version (GGML_VULKAN=ON)" -Encoding ascii

Write-Host "llama-server.exe ($Version, Vulkan) -> $bin"
