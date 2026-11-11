[CmdletBinding()]
param(
    [ValidateSet('Check', 'PrepareCache', 'Run')]
    [string]$Mode = 'Check',
    [string]$LeRobotRoot = 'G:\LeRobot',
    [string]$ModelRoot = '',
    [ValidateRange(1, 10000)]
    [int]$Iterations = 100,
    [ValidateRange(0, 100)]
    [int]$Warmup = 5
)
$ErrorActionPreference = 'Stop'
$lerobotPython = Join-Path $LeRobotRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $lerobotPython)) {
    throw "Interpreter missing: $lerobotPython. Use -LeRobotRoot to select the existing environment."
}
$env:PYTHONUTF8 = '1'
$env:PYTHONNOUSERSITE = '1'
$env:HF_HOME = Join-Path $LeRobotRoot 'cache\huggingface'
$env:TORCH_HOME = Join-Path $LeRobotRoot 'cache\torch'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
if ([string]::IsNullOrWhiteSpace($ModelRoot)) {
    $ModelRoot = Join-Path $PSScriptRoot 'models'
}
$modeMap = @{ Check = 'check'; PrepareCache = 'cache'; Run = 'run' }
$reportPath = Join-Path $PSScriptRoot ('results\' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_' + $Mode)
$scriptPath = Join-Path $PSScriptRoot 'offline_eval.py'
$pythonArgs = @('-u', $scriptPath, '--mode', $modeMap[$Mode], '--model-root', $ModelRoot,
                '--samples', (Join-Path $PSScriptRoot 'samples'), '--output', $reportPath,
                '--iterations', "$Iterations", '--warmup', "$Warmup")
Write-Host "Interpreter: $lerobotPython"
Write-Host "Report folder: $reportPath"
Write-Host 'Offline benchmark only. No cameras or robot ports will be opened.'
& $lerobotPython @pythonArgs
$taskExitCode = $LASTEXITCODE
if ($taskExitCode -ne 0) {
    Write-Host "Stopped with code $taskExitCode. Send back the generated *_return.zip, including failed runs."
    Write-Host 'If the pinned base cache is missing, run again with -Mode PrepareCache, then -Mode Run.'
}
exit $taskExitCode
