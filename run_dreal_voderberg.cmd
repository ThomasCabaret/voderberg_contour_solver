@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if not errorlevel 1 (
    py -3 project_cli.py dreal-voderberg
) else (
    python project_cli.py dreal-voderberg
)
set "exit_code=%errorlevel%"
echo.
if not "%exit_code%"=="0" echo dReal problem generation failed with exit code %exit_code%.
pause
exit /b %exit_code%
