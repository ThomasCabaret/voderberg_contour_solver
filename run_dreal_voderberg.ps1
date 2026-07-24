$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$exitCode = 1
try {
    if (Get-Command py -ErrorAction SilentlyContinue) { py -3 project_cli.py dreal-voderberg } else { python project_cli.py dreal-voderberg }
    $exitCode = $LASTEXITCODE
} finally {
    Read-Host "Press Enter to close"
}
exit $exitCode
