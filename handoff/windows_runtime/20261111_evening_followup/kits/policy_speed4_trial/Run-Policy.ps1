[CmdletBinding()]
param(
    [ValidateSet('Predict','Execute','GraspPredict','Grasp','Release','SelfTest')][string]$Mode='Predict',
    [ValidateRange(1,120)][int]$Seconds=10,
    [string]$LeRobotRoot='G:\LeRobot',
    [switch]$FixedWrist
)
$ErrorActionPreference='Stop'
$taskPython=Join-Path $LeRobotRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw "Python missing: $taskPython" }
$env:PYTHONUTF8='1'
$env:PYTHONNOUSERSITE='1'
$env:HF_HOME=Join-Path $LeRobotRoot 'cache\huggingface'
$env:TORCH_HOME=Join-Path $LeRobotRoot 'cache\torch'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:TOKENIZERS_PARALLELISM='false'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
if ($Mode -eq 'SelfTest') {
    Push-Location $PSScriptRoot
    try { & $taskPython -B -m unittest test_control test_integration test_policy_startup test_grasp_limits test_fixed_wrist; $taskExit=$LASTEXITCODE }
    finally { Pop-Location }
    exit $taskExit
}
if ($Mode -eq 'Execute' -and $Seconds -gt 10) { throw 'Execute is limited to 10 seconds; Predict supports up to 120.' }
if ($Mode -eq 'GraspPredict' -and $Seconds -gt 60) { throw 'GraspPredict is limited to 60 seconds.' }
$taskOutput=Join-Path $PSScriptRoot ('results\'+(Get-Date -Format 'yyyyMMdd_HHmmss_fff')+'_'+$Mode+'_bf16')
$taskMode=if ($Mode -eq 'Predict') { 'dryrun' } elseif ($Mode -eq 'GraspPredict') { 'grasp_dryrun' } else { $Mode.ToLower() }
if ($Mode -in @('Execute','Grasp')) {
    if ($Mode -eq 'Grasp') {
        Write-Host 'FULL GRASP ATTEMPT: saved calibration workspace, <=120s. Gripper WILL open/close.'
        Write-Host 'Goal steps: up to 4 ticks/period (~10.55 deg/s); gripper goal <=20 points/s.'
        Write-Host 'Body tracking <=4deg; gripper tracking <=5 points. Actual motion speed is not independently limited.'
    } else { Write-Host 'MODEL-DRIVEN MOTION: <=10s, +/-8deg, fixed gripper, 4deg tracking guard. P16 unchanged.' }
    Write-Host 'The initial current-position write may activate torque. Close all other controllers.'
    Write-Host 'Keep terminal focused. Ctrl+C / Esc / Space stops. Successful-start stop HOLDS torque ON.'
    Write-Host 'Failed startup attempts torque OFF. Support the arm before starting and before Release.'
} elseif ($Mode -eq 'Release') {
    Write-Host 'TORQUE OFF: physically support the arm first.'
} else { Write-Host 'FULL MODEL PREDICTION, no motor writes. Both cameras must be uncovered.' }
$taskArgs=@('-B','-u',(Join-Path $PSScriptRoot 'bounded_run.py'),'--mode',$taskMode,'--precision','bf16','--seconds',"$Seconds",'--lerobot-root',$LeRobotRoot,'--output',$taskOutput)
if ($FixedWrist) {
    Write-Host 'wrist_roll: passive; actual position still sent to model.'
    Write-Host 'Fixed-wrist camera config: DSHOW wrist index 0, external index 2 (recheck after USB changes).'
    $taskArgs+=@('--fixed-wrist','--robot-config',(Join-Path $PSScriptRoot 'robot_config_fixed_wrist.json'))
}
& $taskPython @taskArgs
exit $LASTEXITCODE
