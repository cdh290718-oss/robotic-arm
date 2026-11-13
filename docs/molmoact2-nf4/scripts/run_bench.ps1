# Offline verification runner for the pinned MolmoAct2 SO-100/101 NF4 checkpoint.
#
# Each configuration runs in its own process (independent CUDA context), so
# VRAM peaks cannot leak between configurations.  Reports are written to tmp\
# first and then moved into results\ by this script: the Python child runs at
# low integrity under the session sandbox and cannot create files in results\.
#
# No robot, camera or serial access happens here.
#
# Usage:
#   pwsh -NoProfile -File .\run_bench.ps1                # all configurations
#   pwsh -NoProfile -File .\run_bench.ps1 -Only patch_on
#   pwsh -NoProfile -File .\run_bench.ps1 -Only saved

param(
    [string]$Only = 'all',
    [int]$Chunks = 30,
    [int]$Warmup = 3
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
if (-not $root) { $root = 'G:\MolmoAct2_NF4' }

$env:TEMP = Join-Path $root 'tmp'
$env:TMP = Join-Path $root 'tmp'
$env:PIP_CACHE_DIR = Join-Path $root 'pip_cache'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'

$py = Join-Path $root '.venv\Scripts\python.exe'
$tmp = Join-Path $root 'tmp'
$results = Join-Path $root 'results'
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

# saved observation: a two-up collage from the SmolVLA run in the sibling project
$mat = 'D:\桌面\萌芽杯\policy_v1_refined\results\20261112_214119_107_Grasp_bf16'
$collage = Join-Path $mat 'camera_last_observation.jpg'
$obsTop = Join-Path $tmp 'obs_top.jpg'
$obsWrist = Join-Path $tmp 'obs_wrist.jpg'
$state = @('-11.5165', '26.8571', '-27.6044', '88.3956', '0.2198', '2.1438')
$task = 'put the yellow block into the basket'

function Move-Report {
    param([string]$Name)
    $runDir = Get-ChildItem (Join-Path $tmp 'runs') -Directory -Filter "*_$Name" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $runDir) { Write-Warning "no run directory for $Name"; return }
    New-Item -ItemType Directory -Force -Path $results | Out-Null
    $dst = Join-Path $results $runDir.Name
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    Copy-Item -Path (Join-Path $runDir.FullName '*') -Destination $dst -Recurse -Force
    Write-Host "run dir -> $dst"
}

function Invoke-Run {
    param([string]$Name, [string[]]$Extra)
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    & $py (Join-Path $root 'offline_inference.py') `
        --model-dir (Join-Path $root 'model') `
        --run-name $Name `
        --run-root (Join-Path $tmp 'runs') `
        --chunks $Chunks --warmup $Warmup @Extra
    Write-Host "exit=$LASTEXITCODE"
    Move-Report $Name
}

$baseCommon = @('--synthetic', '--state') + $state + @('--task', $task)

# ---- D1 baseline: upstream behaviour, CUDA graph on, no patch
if ($Only -eq 'all' -or $Only -eq 'baseline') {
    Invoke-Run 'd1_baseline_graph_on' ($baseCommon + @('--cuda-graph', 'on'))
}
# ---- D2 embedding no-cat patch, CUDA graph on
if ($Only -eq 'all' -or $Only -eq 'patch_on') {
    Invoke-Run 'd2_patch_graph_on' ($baseCommon + @('--cuda-graph', 'on', '--patch-embedding'))
}
# ---- D3 no patch, CUDA graph off
if ($Only -eq 'all' -or $Only -eq 'nograph') {
    Invoke-Run 'd3_baseline_graph_off' ($baseCommon + @('--cuda-graph', 'off'))
}
# ---- D4 patch + CUDA graph off
if ($Only -eq 'all' -or $Only -eq 'patch_off') {
    Invoke-Run 'd4_patch_graph_off' ($baseCommon + @('--cuda-graph', 'off', '--patch-embedding'))
}
# ---- F saved site observation (throughput only; not a synced observation)
if ($Only -eq 'all' -or $Only -eq 'saved') {
    & $py (Join-Path $root 'split_collage.py') --input $collage --left $obsTop --right $obsWrist --info
    Invoke-Run 'f1_saved_observation' (@(
        '--image-top', $obsTop, '--image-wrist', $obsWrist,
        '--state') + $state + @(
        '--task', $task,
        '--cuda-graph', 'on', '--patch-embedding'))
}

Write-Host "`ndone. reports in $results" -ForegroundColor Green
