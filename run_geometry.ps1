Set-Location $PSScriptRoot
$python = Get-Command py -ErrorAction SilentlyContinue
if ($python) {
    & py -3 project_cli.py geometry
} else {
    & python project_cli.py geometry
}
$exitCode = $LASTEXITCODE
Write-Host ""
if ($exitCode -ne 0) {
    Write-Host "Geometry search failed with exit code $exitCode."
}
Read-Host "Press Enter to close"
exit $exitCode
