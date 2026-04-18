#!/bin/bash

# Ensure dependencies
echo "Installing dependencies..."
pip install fastapi uvicorn websockets

# Clean up old processes on the ports we want to use
echo "Cleaning up old processes on ports 8002 and 8003..."
fuser -k 8002/tcp 2>/dev/null || true
fuser -k 8003/tcp 2>/dev/null || true
sleep 1

# Start Backend
echo "Starting FastAPI Backend Server on port 8002..."
cd /home/gujing/Graduation_Design/ours/vis/backend
uvicorn main:app --host 127.0.0.1 --port 8002 &
BACKEND_PID=$!

# Start Frontend
echo "Starting Frontend Web Server on port 8003..."
cd /home/gujing/Graduation_Design/ours/vis/frontend
python3 -m http.server 8003 &
FRONTEND_PID=$!

echo ""
echo "=================================================="
echo "Visualization System is running!"
echo "Please open your browser and navigate to: http://127.0.0.1:8003"
echo "=================================================="
echo ""
echo "Press Ctrl+C to stop servers."

# Wait for process to exit
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
