@echo off
setlocal
cd /d "%~dp0"

if not exist "logs" mkdir "logs"

rem A direct Python child keeps this console alive for the whole GUI session.
rem Delegate to WScript so the familiar .cmd launcher leaves no cmd window open.
start "" /b wscript.exe "%~dp0launch_cutter_gui.vbs"

endlocal
