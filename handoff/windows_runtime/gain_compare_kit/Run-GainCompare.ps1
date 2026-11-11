[CmdletBinding()]
param(
    [ValidateSet('Check','Compare','Large','SelfTest')][string]$Mode='Check',
    [string]$LeRobotRoot='G:\LeRobot',
    [string]$Port='COM5'
)
$ErrorActionPreference='Stop'
$taskPython=Join-Path $LeRobotRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw "Python missing: $taskPython" }
$env:PYTHONUTF8='1'
$env:PYTHONNOUSERSITE='1'
if ($Mode -eq 'SelfTest') {
    Push-Location $PSScriptRoot
    try { & $taskPython -B -m unittest -v test_gain_compare; $taskExit=$LASTEXITCODE }
    finally { Pop-Location }
    exit $taskExit
}
if ($Mode -in @('Compare','Large')) {
    if ($Mode -eq 'Large') { Write-Host 'ACTUAL MOTION: elbow <=2.91 deg; P16 then P20. Each trial <=10s.' }
    else { Write-Host 'ACTUAL MOTION: elbow <=0.88 deg; P16 then P20. Each trial <=5s.' }
    Write-Host 'Torque OFF after EACH trial. Support the arm so releasing torque will not cause a fall.'
    Write-Host 'Close calibration/teleoperation/control sessions. Keep this terminal focused. Ctrl+C/Esc/Space stops.'
} else { Write-Host 'READ ONLY CHECK: no motor register writes.' }
& $taskPython -B (Join-Path $PSScriptRoot 'gain_compare.py') --mode $Mode.ToLower() --root $LeRobotRoot --port $Port
exit $LASTEXITCODE
