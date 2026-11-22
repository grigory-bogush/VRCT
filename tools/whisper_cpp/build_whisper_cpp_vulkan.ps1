# Builds whisper.cpp's `whisper-server.exe` with the Vulkan backend and puts it
# in tools/whisper_cpp/bin/, where spec/backend*.spec pick it up and bundle it
# as `whisper_cpp/` next to the sidecar.
#
# Requirements (Windows):
#   - Git, CMake 3.21+
#   - Visual Studio 2022 Build Tools (C++ workload)
#   - Vulkan SDK (sets %VULKAN_SDK%; provides glslc for shader compilation)
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File tools\whisper_cpp\build_whisper_cpp_vulkan.ps1
#   powershell -ExecutionPolicy Bypass -File tools\whisper_cpp\build_whisper_cpp_vulkan.ps1 -Version v1.9.5
#
# The result is a single statically linked whisper-server.exe (no ggml/whisper
# DLLs, no MSVC runtime DLL). At run time it only needs vulkan-1.dll, which
# every AMD/NVIDIA/Intel GPU driver installs.

param(
    [string]$Version = "v1.9.5"
)

$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$src = Join-Path $here "src"
$build = Join-Path $here "build"
$bin = Join-Path $here "bin"

if (-not $env:VULKAN_SDK) {
    throw "VULKAN_SDK is not set. Install the Vulkan SDK from https://vulkan.lunarg.com/sdk/home#windows first."
}

if (Test-Path $src) {
    $current = (git -C $src describe --tags --exact-match 2>$null)
    if ($current -ne $Version) {
        Remove-Item -Recurse -Force $src
    }
}
if (-not (Test-Path $src)) {
    git clone --depth 1 --branch $Version https://github.com/ggml-org/whisper.cpp.git $src
    if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
}

cmake -S $src -B $build `
    -DCMAKE_BUILD_TYPE=Release `
    -DBUILD_SHARED_LIBS=OFF `
    -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded `
    -DGGML_VULKAN=ON `
    -DGGML_NATIVE=OFF `
    -DWHISPER_SDL2=OFF `
    -DWHISPER_BUILD_TESTS=OFF `
    -DWHISPER_BUILD_EXAMPLES=ON `
    -DWHISPER_BUILD_SERVER=ON
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

cmake --build $build --config Release --target whisper-server -j
if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }

New-Item -ItemType Directory -Force -Path $bin | Out-Null
$exe = Get-ChildItem -Path $build -Recurse -Filter "whisper-server.exe" | Select-Object -First 1
if (-not $exe) { throw "whisper-server.exe not found under $build" }
Copy-Item $exe.FullName (Join-Path $bin "whisper-server.exe") -Force
Copy-Item (Join-Path $src "LICENSE") (Join-Path $bin "LICENSE-whisper.cpp.txt") -Force
Set-Content -Path (Join-Path $bin "VERSION.txt") -Value "whisper.cpp $Version (GGML_VULKAN=ON)" -Encoding ascii

Write-Host "whisper-server.exe ($Version, Vulkan) -> $bin"
