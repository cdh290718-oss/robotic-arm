[CmdletBinding()]
param([ValidateRange(1,300)][int]$Seconds=120,[string]$LeRobotRoot='G:\LeRobot',[string]$Port='COM5',[switch]$SelfTest)
$ErrorActionPreference='Stop'
$taskPython=Join-Path $LeRobotRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw "Python missing: $taskPython" }
$env:PYTHONUTF8='1'
$env:PYTHONNOUSERSITE='1'
if ($SelfTest) {
    Push-Location $PSScriptRoot
    try { & $taskPython -B -m unittest test_watch_wrist; $taskExit=$LASTEXITCODE }
    finally { Pop-Location }
    exit $taskExit
}
& $taskPython -B -u (Join-Path $PSScriptRoot 'watch_wrist.py') --root $LeRobotRoot --port $Port --seconds $Seconds
exit $LASTEXITCODE
