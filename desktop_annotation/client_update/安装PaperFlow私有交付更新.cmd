@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-PaperFlow-PrivateDelivery.ps1"
set "code=%errorlevel%"
if not "%code%"=="0" (
  echo.
  echo Update did not finish. Follow the message and run this file again.
  pause
)
exit /b %code%
