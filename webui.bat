@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1

rem First run downloads llama-server automatically (LLAMA_BACKEND=auto).
rem Examples:
rem   set LLAMA_BACKEND=cuda
rem   set LLAMA_SKIP_INSTALL=1

if not exist "venv\Scripts\python.exe" (
  echo Creating venv...
  python -m venv venv
  if errorlevel 1 (
    echo Failed to create venv. Is Python on PATH?
    exit /b 1
  )
  call venv\Scripts\activate.bat
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Failed to install requirements.
    exit /b 1
  )
) else (
  call venv\Scripts\activate.bat
)

python launch.py %*
