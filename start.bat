@echo off
rem stage07 — запуск без консоли через pythonw.exe
setlocal
cd /d "%~dp0"

set "PYW=.venv\Scripts\pythonw.exe"
if not exist "%PYW%" (
    set "PYW=pythonw.exe"
)

start "" "%PYW%" main.py
endlocal
