@echo off
setlocal
cd /d "%~dp0"

if not exist "logs" mkdir "logs"

rem Do not depend on Windows Script Host: some company desktops disable .vbs.
rem The child is detached and all startup failures remain inspectable here.
start "PaperFlow" /b cmd.exe /d /c "python.exe -B cut_questions_gui.py >> logs\gui_startup.log 2>&1"

endlocal
