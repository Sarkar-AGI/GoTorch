# GoTorch Windows Installer
# Copyright (c) 2024-2026 Sarkar-AGI. MIT License.
#
# Usage (PowerShell as Administrator):
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#   .\install_windows.ps1
#
# Or with custom install dir:
#   .\install_windows.ps1 -InstallDir "C:\GoTorch"

param(
    [string]$InstallDir = "$env:USERPROFILE\GoTorch",
    [string]$GoVersion  = "1.22.2",
    [switch]$SkipGo     = $false,
    [switch]$SkipBuild  = $false,
    [switch]$WithCUDA   = $false
)

# ─── Colors ───────────────────────────────────────────────────────────────────

function Write-Green($msg)  { Write-Host $msg -ForegroundColor Green }
function Write-Red($msg)    { Write-Host $msg -ForegroundColor Red }
function Write-Yellow($msg) { Write-Host $msg -ForegroundColor Yellow }
function Write-Blue($msg)   { Write-Host $msg -ForegroundColor Cyan }

# ─── Banner ───────────────────────────────────────────────────────────────────

Write-Blue ""
Write-Blue "╔════════════════════════════════════════════╗"
Write-Blue "║      GoTorch Windows Installer v1.0.0      ║"
Write-Blue "║   PyTorch C++ engine, wrapped for Go       ║"
Write-Blue "╚════════════════════════════════════════════╝"
Write-Blue ""

# ─── Admin check ──────────────────────────────────────────────────────────────

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Yellow "⚠ Not running as Administrator."
    Write-Yellow "  Some steps may fail. Re-run as Administrator for best results."
    Write-Yellow "  Right-click PowerShell → 'Run as Administrator'"
    Write-Yellow ""
}

# ─── Check / Install requirements ─────────────────────────────────────────────

Write-Host "── Checking requirements ────────────────────────────────────────────────"

# Check winget
$hasWinget = $null -ne (Get-Command winget -ErrorAction SilentlyContinue)

# Check Git
$hasGit = $null -ne (Get-Command git -ErrorAction SilentlyContinue)
if (-not $hasGit) {
    Write-Yellow "  Git not found. Installing via winget..."
    if ($hasWinget) {
        winget install --id Git.Git -e --source winget --silent
        $env:PATH += ";C:\Program Files\Git\bin"
    } else {
        Write-Red "✗ Git not found. Install from: https://git-scm.com/download/win"
        exit 1
    }
}
Write-Green "✓ Git: $(git --version)"

# Check CMake
$hasCMake = $null -ne (Get-Command cmake -ErrorAction SilentlyContinue)
if (-not $hasCMake) {
    Write-Yellow "  CMake not found. Installing via winget..."
    if ($hasWinget) {
        winget install --id Kitware.CMake -e --silent
        $env:PATH += ";C:\Program Files\CMake\bin"
    } else {
        Write-Red "✗ CMake not found. Install from: https://cmake.org/download"
        exit 1
    }
}
Write-Green "✓ CMake: $(cmake --version | Select-Object -First 1)"

# Check Visual Studio / MSVC
$hasMSVC = $null -ne (Get-Command cl -ErrorAction SilentlyContinue)
if (-not $hasMSVC) {
    # Try to find VS
    $vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vsWhere) {
        $vsPath = & $vsWhere -latest -property installationPath
        $vcvars = "$vsPath\VC\Auxiliary\Build\vcvars64.bat"
        if (Test-Path $vcvars) {
            Write-Green "✓ Visual Studio found at: $vsPath"
        }
    } else {
        Write-Yellow "⚠ Visual Studio not found."
        Write-Yellow "  Install Visual Studio 2022 with 'Desktop development with C++'"
        Write-Yellow "  https://visualstudio.microsoft.com/downloads/"
        Write-Yellow ""
        Write-Yellow "  Or install Build Tools only:"
        Write-Yellow "  winget install Microsoft.VisualStudio.2022.BuildTools"
        if ($hasWinget) {
            $install = Read-Host "  Install now? (y/n)"
            if ($install -eq 'y') {
                winget install Microsoft.VisualStudio.2022.BuildTools `
                    --override "--quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
            }
        }
    }
}

# Check Ninja
$hasNinja = $null -ne (Get-Command ninja -ErrorAction SilentlyContinue)
if (-not $hasNinja) {
    Write-Yellow "  Ninja not found. Installing via winget..."
    if ($hasWinget) {
        winget install --id Ninja-build.Ninja -e --silent
    } else {
        Write-Red "✗ Ninja not found. Install from: https://ninja-build.org"
        exit 1
    }
}
Write-Green "✓ Ninja: $(ninja --version)"

# Check Python
$hasPython = $null -ne (Get-Command python -ErrorAction SilentlyContinue)
if (-not $hasPython) {
    Write-Yellow "  Python not found. Installing via winget..."
    if ($hasWinget) {
        winget install --id Python.Python.3.11 -e --silent
        $env:PATH += ";$env:USERPROFILE\AppData\Local\Programs\Python\Python311"
    } else {
        Write-Red "✗ Python not found. Install from: https://python.org/downloads"
        exit 1
    }
}
Write-Green "✓ Python: $(python --version)"

# ─── Install Go ───────────────────────────────────────────────────────────────

if (-not $SkipGo) {
    $hasGoBin = $null -ne (Get-Command go -ErrorAction SilentlyContinue)
    if (-not $hasGoBin) {
        Write-Host ""
        Write-Host "── Installing Go $GoVersion ──────────────────────────────────────────────"

        $goInstaller = "$env:TEMP\go$GoVersion.windows-amd64.msi"
        $goUrl = "https://go.dev/dl/go$GoVersion.windows-amd64.msi"

        Write-Host "  Downloading Go $GoVersion..."
        Invoke-WebRequest -Uri $goUrl -OutFile $goInstaller -UseBasicParsing

        Write-Host "  Installing Go..."
        Start-Process msiexec.exe -Wait -ArgumentList "/i `"$goInstaller`" /quiet"
        Remove-Item $goInstaller -ErrorAction SilentlyContinue

        $env:PATH += ";C:\Program Files\Go\bin"
        [System.Environment]::SetEnvironmentVariable("PATH", $env:PATH, [System.EnvironmentVariableTarget]::User)
    }
    Write-Green "✓ Go: $(go version)"
}

# ─── Clone GoTorch ────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "── Cloning GoTorch ──────────────────────────────────────────────────────"

if (-not (Test-Path "$InstallDir\gotorch.go")) {
    git clone https://github.com/Sarkar-AGI/GoTorch.git $InstallDir
} else {
    Write-Green "✓ GoTorch already at: $InstallDir"
}

Set-Location $InstallDir
Write-Green "✓ GoTorch source at: $InstallDir"

# ─── Submodules ───────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "── Initializing submodules ──────────────────────────────────────────────"
git submodule update --init --recursive --depth 1 `
    third_party/eigen `
    third_party/fmt `
    third_party/glog `
    third_party/googletest `
    third_party/protobuf `
    third_party/pybind11 `
    third_party/sleef 2>$null
Write-Green "✓ Submodules ready"

# ─── Python build deps ────────────────────────────────────────────────────────

Write-Host ""
Write-Host "── Installing Python build dependencies ─────────────────────────────────"
python -m pip install --quiet pyyaml numpy typing_extensions
Write-Green "✓ Python deps ready"

# ─── Build libtorch ───────────────────────────────────────────────────────────

if (-not $SkipBuild) {
    Write-Host ""
    Write-Host "── Building libtorch C++ backend ────────────────────────────────────────"
    Write-Yellow "   This may take 20-60 minutes on first build..."
    Write-Host ""

    $BuildDir = "$InstallDir\build"
    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
    Set-Location $BuildDir

    # CMake configure
    $cmakeArgs = @(
        "..",
        "-GNinja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_PYTHON=OFF",
        "-DBUILD_TEST=OFF",
        "-DBUILD_CAFFE2=OFF",
        "-DBUILD_SHARED_LIBS=ON",
        "-DUSE_CUDA=$(if ($WithCUDA) { 'ON' } else { 'OFF' })",
        "-DUSE_DISTRIBUTED=OFF",
        "-DUSE_MKLDNN=ON",
        "-DUSE_QNNPACK=OFF",
        "-DUSE_PYTORCH_QNNPACK=OFF",
        "-DUSE_XNNPACK=OFF",
        "-DUSE_NNPACK=OFF",
        "-DUSE_OPENMP=ON",
        "-DBUILD_BINARY=OFF",
        "-DCMAKE_INSTALL_PREFIX=$InstallDir\build\install"
    )

    cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Red "✗ CMake configuration failed"
        exit 1
    }

    # Build
    $cores = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
    ninja -j$cores torch torch_cpu c10
    if ($LASTEXITCODE -ne 0) {
        Write-Red "✗ Build failed"
        exit 1
    }

    ninja install
    Set-Location $InstallDir
    Write-Green "✓ libtorch built successfully"
}

# ─── Set environment variables ────────────────────────────────────────────────

Write-Host ""
Write-Host "── Setting environment variables ────────────────────────────────────────"

$LibDir   = "$InstallDir\build\lib"
$IncDir1  = "$InstallDir\torch\csrc\api\include"
$IncDir2  = "$InstallDir\csrc\go_binding"
$IncDir3  = "$InstallDir\build\install\include"

# CGo flags
[System.Environment]::SetEnvironmentVariable(
    "CGO_CFLAGS",
    "-I$IncDir1 -I$IncDir2 -I$IncDir3",
    [System.EnvironmentVariableTarget]::User
)
[System.Environment]::SetEnvironmentVariable(
    "CGO_LDFLAGS",
    "-L$LibDir -ltorch -ltorch_cpu -lc10",
    [System.EnvironmentVariableTarget]::User
)
[System.Environment]::SetEnvironmentVariable(
    "GOTORCH",
    $InstallDir,
    [System.EnvironmentVariableTarget]::User
)

# Add lib to PATH so DLLs are found
$currentPath = [System.Environment]::GetEnvironmentVariable("PATH", [System.EnvironmentVariableTarget]::User)
if ($currentPath -notlike "*$LibDir*") {
    [System.Environment]::SetEnvironmentVariable(
        "PATH",
        "$LibDir;$currentPath",
        [System.EnvironmentVariableTarget]::User
    )
}

# For current session
$env:CGO_CFLAGS  = "-I$IncDir1 -I$IncDir2 -I$IncDir3"
$env:CGO_LDFLAGS = "-L$LibDir -ltorch -ltorch_cpu -lc10"
$env:GOTORCH     = $InstallDir
$env:PATH        = "$LibDir;$env:PATH"

Write-Green "✓ Environment variables set"

# ─── Build Go binding ─────────────────────────────────────────────────────────

Write-Host ""
Write-Host "── Building Go binding ──────────────────────────────────────────────────"
go build ./...
if ($LASTEXITCODE -ne 0) {
    Write-Red "✗ Go build failed. Check CGO_CFLAGS and CGO_LDFLAGS."
    exit 1
}
Write-Green "✓ Go binding built"

# ─── Quick test ───────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "── Running quick tests ──────────────────────────────────────────────────"
$env:GOTORCH_SKIP_CPP_TESTS = "1"
go test -timeout 60s -run "^TestGoTensorCreation|TestGoLinear|TestGoAdd" .
if ($LASTEXITCODE -eq 0) {
    Write-Green "✓ Tests passed"
} else {
    Write-Yellow "⚠ Some tests failed — check environment variables"
}

# ─── Run example ──────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "── Running MNIST example ────────────────────────────────────────────────"
go run ./examples/mnist/ 2>&1 | Select-Object -First 15

# ─── Done ─────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Green "╔════════════════════════════════════════════╗"
Write-Green "║       GoTorch installed! 🎉                ║"
Write-Green "╚════════════════════════════════════════════╝"
Write-Host ""
Write-Host "Next steps:"
Write-Host ""
Write-Host "  # Restart PowerShell to apply environment variables"
Write-Host ""
Write-Host "  # Run examples"
Write-Host "  cd $InstallDir"
Write-Host "  go run ./examples/mnist/"
Write-Host "  go run ./examples/text_classification/"
Write-Host "  go run ./examples/image_classification/"
Write-Host ""
Write-Host "  # Run tests"
Write-Host "  go test -v ./..."
Write-Host ""
Write-Host "  # Run benchmarks"
Write-Host "  go test -bench=. -benchmem ."
Write-Host ""
Write-Host "  # Use in your project"
Write-Host "  go get github.com/Sarkar-AGI/GoTorch@v1.0.0"
Write-Host ""
Write-Host "  import gt `"github.com/Sarkar-AGI/GoTorch`""
Write-Host ""
