@echo off
REM Double-click to start the SRIM Damage Calculator in your web browser.
cd /d "%~dp0"
REM --server.address localhost: only this PC can open the app (remove it to share on the lab network).
".venv\Scripts\python.exe" -m streamlit run srim_app.py --server.address localhost --browser.gatherUsageStats false
pause
