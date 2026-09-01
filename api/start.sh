#!/bin/bash
# Startup script for AEGIS Inference API

# Set default values if not provided
export API_HOST=${API_HOST:-0.0.0.0}
export API_PORT=${API_PORT:-8000}
export LOG_LEVEL=${LOG_LEVEL:-INFO}
export API_RELOAD=${API_RELOAD:-false}

# Navigate to project root
cd "$(dirname "$0")/.."

# Start the API server
if [ "$API_RELOAD" = "true" ]; then
    echo "Starting API server in development mode with auto-reload..."
    python -m api.main
else
    echo "Starting API server in production mode..."
    uvicorn api.main:app --host "$API_HOST" --port "$API_PORT" --log-level "$LOG_LEVEL"
fi
