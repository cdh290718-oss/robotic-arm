# Install the isolated environment for the MolmoAct2 NF4 offline verification.
#
# Does NOT touch G:\LeRobot\.venv or any other existing environment.
# TEMP/TMP are redirected into the project because ensurepip failed in another
# directory when the system temp dir was not writable (that was a permissions
# problem, not a Python incompatibility).
#
# Usage:  pwsh -NoProfile -File .\install_env.ps1

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
if (-not $root) { $root = 'G:\MolmoAct2_NF4' }

$basePython = 'G:\LeRobot\python\cpython-3.12-windows-x86_64-none\python.exe'
if (-not (Test-Path $basePython)) { throw "base CPython not found: $basePython" }

$env:TEMP = Join-Path $root 'tmp'
$env:TMP  = Join-Path $root 'tmp'
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
$env:PIP_CACHE_DIR = Join-Path $root 'pip_cache'

$venv = Join-Path $root '.venv'
$vpy  = Join-Path $venv 'Scripts\python.exe'

if (-not (Test-Path $vpy)) {
    Write-Host "creating venv $venv from $basePython"
    & $basePython -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed ($LASTEXITCODE)" }
} else {
    Write-Host "venv already present: $venv"
}

# Versions are pinned to the ones the verification ran with.
$torchIndex = 'https://download.pytorch.org/whl/cu128'
& $vpy -m pip install --upgrade pip
& $vpy -m pip install --index-url $torchIndex 'torch==2.10.0+cu128'
if ($LASTEXITCODE -ne 0) { throw "torch install failed ($LASTEXITCODE)" }

& $vpy -m pip install `
    'transformers==5.5.4' `
    'tokenizers' `
    'safetensors==0.8.0' `
    'huggingface-hub==1.31.0' `
    'accelerate==1.15.0' `
    'bitsandbytes' `
    'numpy' `
    'pillow' `
    'psutil'
if ($LASTEXITCODE -ne 0) { throw "dependency install failed ($LASTEXITCODE)" }

& $vpy -m pip freeze | Set-Content -Encoding UTF8 (Join-Path $root 'results\pip_freeze.txt')
Write-Host "done. interpreter: $vpy"
