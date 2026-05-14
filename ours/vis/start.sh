#!/bin/bash

set -e

BACKEND_PORT=8011
FRONTEND_PORT=8012
PROJECT_DIR="/home/gujing/Graduation_Design/ours/vis"
BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  if [ -n "$FRONTEND_PID" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
}

trap cleanup INT TERM EXIT

# Ensure dependencies without requiring network on every start.
echo "Checking Python dependencies..."
MISSING_DEPS=$(python -c "import importlib.util; deps={'fastapi':'fastapi','uvicorn':'uvicorn','websockets':'websockets'}; print(' '.join(pkg for pkg, mod in deps.items() if importlib.util.find_spec(mod) is None))")
if [ -n "$MISSING_DEPS" ]; then
  echo "Missing dependencies: ${MISSING_DEPS}"
  echo "Install them with: python -m pip install fastapi uvicorn websockets"
  exit 1
fi

# Clean up old processes on the ports we want to use
echo "Cleaning up old processes on ports ${BACKEND_PORT} and ${FRONTEND_PORT}..."
fuser -k "${BACKEND_PORT}/tcp" 2>/dev/null || true
fuser -k "${FRONTEND_PORT}/tcp" 2>/dev/null || true
sleep 1

# Start Backend
echo "Starting FastAPI Backend Server on port ${BACKEND_PORT}..."
cd "${PROJECT_DIR}/backend"
python -m uvicorn main:app --host 127.0.0.1 --port "${BACKEND_PORT}" &
BACKEND_PID=$!

# Start Frontend
echo "Starting Frontend Web Server on port ${FRONTEND_PORT}..."
cd "${PROJECT_DIR}/frontend"
python -m http.server "${FRONTEND_PORT}" --bind 127.0.0.1 &
FRONTEND_PID=$!

sleep 1

if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
  echo "Backend failed to start on port ${BACKEND_PORT}."
  exit 1
fi

if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
  echo "Frontend failed to start on port ${FRONTEND_PORT}."
  exit 1
fi

echo ""
echo "=================================================="
echo "Visualization System is running!"
echo "Please open your browser and navigate to: http://127.0.0.1:${FRONTEND_PORT}"
echo "=================================================="
echo ""
echo "Press Ctrl+C to stop servers."

# Wait for process to exit
wait -n "$BACKEND_PID" "$FRONTEND_PID"
echo "A server process exited; shutting down remaining processes."
exit 1
