@echo off
echo ========================================================
echo AEGIS Deepfake Research Platform - Local Startup
echo ========================================================

echo Starting AEGIS Inference API (Backend) on Port 8000...
start "AEGIS Backend" cmd /c ".\venv\Scripts\Activate && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"

echo Waiting 5 seconds for backend to initialize...
timeout /t 5 /nobreak >nul

echo Starting AEGIS Glassmorphism UI (Frontend) on Port 3000...
start "AEGIS Frontend" cmd /c "cd frontend && npm run dev"

echo.
echo ========================================================
echo Services launched!
echo - API is running in the background.
echo - Frontend is opening your default browser.
echo Please go to http://localhost:3000 to use AEGIS!
echo ========================================================
pause
