param([int]$External=2, [int]$Wrist=0)
$ErrorActionPreference='Stop'
$python='G:\LeRobot\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "Python not found: $python" }
& $python -u (Join-Path $PSScriptRoot 'preview.py') --external $External --wrist $Wrist
exit $LASTEXITCODE
