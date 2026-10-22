@echo off
REM Double-click to start the SRIM Damage Calculator in your web browser.
cd /d "%~dp0"
REM Uses the Python 3.13 environment (.venv313) if it exists, otherwise the older .venv.
set PY=.venv313\Scripts\python.exe
if not exist "%PY%" set PY=.venv\Scripts\python.exe
REM --server.address localhost: only this PC can open the app (remove it to share on the lab network).
"%PY%" -m streamlit run srim_app.py --server.address localhost --browser.gatherUsageStats false
pause
