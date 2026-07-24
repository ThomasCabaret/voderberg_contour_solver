@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if not errorlevel 1 (
    py -3 project_cli.py web
) else (
    python project_cli.py web
)
set "exit_code=%errorlevel%"
echo.
if not "%exit_code%"=="0" echo Web viewer stopped with exit code %exit_code%.
pause
exit /b %exit_code%
