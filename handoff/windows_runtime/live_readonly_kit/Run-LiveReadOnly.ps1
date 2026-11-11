[CmdletBinding()]
param(
    [ValidateSet('Inputs','Live','Replay')][string]$Mode='Inputs',
    [ValidateSet('bf16','fp16')][string]$Precision='bf16',
    [ValidateRange(1,600)][int]$Seconds=60,
    [string]$LeRobotRoot='G:\LeRobot',
    [string]$OfflineKit='',
    [string]$ModelDir='',
    [string]$RobotConfig='',
    [string]$Calibration='',
    [string]$Port='',
    [switch]$NoBrowser
)
$ErrorActionPreference='Stop'
$taskPython=Join-Path $LeRobotRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw "Interpreter not found: $taskPython" }
$env:PYTHONUTF8='1'
$env:PYTHONNOUSERSITE='1'
$env:HF_HOME=Join-Path $LeRobotRoot 'cache\huggingface'
$env:TORCH_HOME=Join-Path $LeRobotRoot 'cache\torch'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
$env:TOKENIZERS_PARALLELISM='false'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$taskOutput=Join-Path $PSScriptRoot ('results\'+(Get-Date -Format 'yyyyMMdd_HHmmss_fff')+'_'+$Mode+'_'+$Precision)
$taskArgs=@('-u',(Join-Path $PSScriptRoot 'live_probe.py'),'--mode',$Mode.ToLower(),
            '--precision',$Precision,'--seconds',"$Seconds",'--lerobot-root',$LeRobotRoot,'--output',$taskOutput)
if ($OfflineKit) { $taskArgs+=@('--offline-kit',$OfflineKit) }
if ($ModelDir) { $taskArgs+=@('--model-dir',$ModelDir) }
if ($RobotConfig) { $taskArgs+=@('--robot-config',$RobotConfig) }
if ($Calibration) { $taskArgs+=@('--calibration',$Calibration) }
if ($Port) { $taskArgs+=@('--port',$Port) }
if ($NoBrowser) { $taskArgs+='--no-browser' }
Write-Host "Using: $taskPython"
Write-Host 'READ ONLY: no motor position, torque, or calibration writes. Ctrl+C stops the probe.'
& $taskPython @taskArgs
$taskExitCode=$LASTEXITCODE
Write-Host "Results: $taskOutput"
Write-Host 'Send back the *_return.zip, including failed runs.'
exit $taskExitCode
