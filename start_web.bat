@echo off
cd /d %~dp0
echo ============================================
echo   XingHuo Customer-Service Agent - Web Server
echo   Keep this window open. Closing it stops the server.
echo   The browser will open automatically in a few seconds.
echo ============================================
start "" /min cmd /c "timeout /t 4 /nobreak >nul & start http://127.0.0.1:8000"
".venv\Scripts\python.exe" -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
echo.
echo Server stopped.
pause
