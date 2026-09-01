@echo off
REM Startup script for AEGIS Inference API (Windows)

REM Set default values if not provided
if not defined API_HOST set API_HOST=0.0.0.0
if not defined API_PORT set API_PORT=8000
if not defined LOG_LEVEL set LOG_LEVEL=INFO
if not defined API_RELOAD set API_RELOAD=false

REM Navigate to project root
cd /d "%~dp0.."

REM Start the API server
if "%API_RELOAD%"=="true" (
    echo Starting API server in development mode with auto-reload...
    python -m api.main
) else (
    echo Starting API server in production mode...
    uvicorn api.main:app --host %API_HOST% --port %API_PORT% --log-level %LOG_LEVEL%
)
