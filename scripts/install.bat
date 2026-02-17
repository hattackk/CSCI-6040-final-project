@echo off
setlocal

set SCRIPT_DIR=%~dp0
set INSTALL_PY=%SCRIPT_DIR%install.py

if defined PYTHON_BIN (
  "%PYTHON_BIN%" "%INSTALL_PY%" %*
  exit /b %ERRORLEVEL%
)

where py >nul 2>nul
if %ERRORLEVEL%==0 (
  py -3 "%INSTALL_PY%" %*
  exit /b %ERRORLEVEL%
)

where python >nul 2>nul
if %ERRORLEVEL%==0 (
  python "%INSTALL_PY%" %*
  exit /b %ERRORLEVEL%
)

echo Python 3.10+ is required but was not found in PATH.
exit /b 1
