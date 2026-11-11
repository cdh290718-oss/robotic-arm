[CmdletBinding()]
param(
    [string]$LeRobotRoot='G:\LeRobot',
    [string]$Port='',
    [ValidateRange(1,60)][int]$Seconds=15
)
$ErrorActionPreference='Stop'
$taskPython=Join-Path $LeRobotRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw "Interpreter not found: $taskPython" }
$env:PYTHONUTF8='1'
$env:PYTHONNOUSERSITE='1'
$taskOutput=Join-Path $PSScriptRoot ('results\'+(Get-Date -Format 'yyyyMMdd_HHmmss_fff')+'_MotorDiagnostic')
$taskArgs=@('-u',(Join-Path $PSScriptRoot 'motor_diagnostic.py'),'--lerobot-root',$LeRobotRoot,
            '--seconds',"$Seconds",'--output',$taskOutput)
if ($Port) {$taskArgs+=@('--port',$Port)}
Write-Host 'READ ONLY. Close other robot-control programs. No model, cameras, movement or torque changes.'
& $taskPython @taskArgs
exit $LASTEXITCODE
