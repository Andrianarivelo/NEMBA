@echo off
rem ---------------------------------------------------------------------------
rem  NEMBA launcher (double-click): opens the GUI with the conda env below.
rem  Extra arguments are passed to main.py, e.g.  NEMBA.bat --demo --speed fast
rem  Edit ENV_NAME if your environment has another name.
rem ---------------------------------------------------------------------------
set ENV_NAME=analysis
set PY=%USERPROFILE%\miniconda3\envs\%ENV_NAME%\python.exe
if not exist "%PY%" set PY=%USERPROFILE%\anaconda3\envs\%ENV_NAME%\python.exe
if not exist "%PY%" set PY=python
cd /d "%~dp0"
"%PY%" main.py %*
if errorlevel 1 pause
