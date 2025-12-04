#!/bin/bash

# FinDocAI Startup Script
# This script starts all required services for the FinDocAI application

set -e  # Exit immediately if a command exits with a non-zero status

# Default values that can be overridden by environment variables
export REDIS_HOST=${REDIS_HOST:-localhost}
export REDIS_PORT=${REDIS_PORT:-6380}
export DB_HOST=${DB_HOST:-localhost}
export DB_PORT=${DB_PORT:-5434}
export DB_NAME=${DB_NAME:-findocai}
export DB_USER=${DB_USER:-findocai_user}
export DB_PASSWORD=${DB_PASSWORD:-findocai_password}

echo "Starting FinDocAI services..."

# Function to check if a port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "Port $port is already in use"
        return 1
    fi
    return 0
}

# Function to wait for a service to be ready
wait_for_service() {
    local host=$1
    local port=$2
    local service=$3
    local timeout=${4:-60}
    
    echo "Waiting for $service at $host:$port..."
    local count=0
    while ! nc -z $host $port; do
        sleep 2
        count=$((count + 2))
        if [ $count -gt $timeout ]; then
            echo "Timeout waiting for $service"
            return 1
        fi
        echo -n "."
    done
    echo ""
    echo "$service is ready!"
}

# Check if required ports are available
echo "Checking required ports..."
if ! check_port 8000; then
    echo "Error: Port 8000 is in use. Please stop the service using this port first."
    exit 1
fi

if ! check_port $REDIS_PORT; then
    echo "Error: Redis port $REDIS_PORT is in use."
    # Don't exit here, just warn because Docker might manage it
    echo "Warning: Redis port in use, Docker may manage it"
fi

if ! check_port $DB_PORT; then
    echo "Error: Database port $DB_PORT is in use."
    echo "Warning: DB port in use, Docker may manage it"
fi

# Start database and Redis using Docker Compose
echo "Starting PostgreSQL and Redis with Docker Compose..."
docker-compose up -d postgres redis

# Wait for database to be ready
wait_for_service $DB_HOST $DB_PORT "PostgreSQL" 45

# Wait for Redis to be ready
wait_for_service $REDIS_HOST $REDIS_PORT "Redis" 30

# Initialize the database
echo "Initializing database..."
python scripts/init_db.py

# Check if GEMINI_API_KEY is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "Warning: GEMINI_API_KEY environment variable is not set."
    echo "Some features may be limited without a valid API key."
    echo "Please set it with: export GEMINI_API_KEY='your_api_key'"
fi

# Start the FastAPI server in the background
echo "Starting FastAPI server..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# Wait a moment for the server to start
sleep 3

# Check if the server is running
if ! kill -0 $FASTAPI_PID 2>/dev/null; then
    echo "Error: Failed to start FastAPI server"
    exit 1
fi

echo "FastAPI server started with PID: $FASTAPI_PID"

# Start the Celery worker in the background
echo "Starting Celery worker..."
celery -A app.worker worker --loglevel=info --concurrency=2 &
CELERY_PID=$!

# Wait a moment for the worker to start
sleep 2

# Check if the worker is running
if ! kill -0 $CELERY_PID 2>/dev/null; then
    echo "Error: Failed to start Celery worker"
    kill $FASTAPI_PID 2>/dev/null || true
    exit 1
fi

echo "Celery worker started with PID: $CELERY_PID"

# Create a file to track PIDs for easy stopping
cat > .service_pids << EOF
FASTAPI_PID=$FASTAPI_PID
CELERY_PID=$CELERY_PID
EOF

echo ""
echo "==========================================="
echo "FinDocAI is now running!"
echo "==========================================="
echo "API Server: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo "Metrics: http://localhost:8000/metrics"
echo ""
echo "To stop all services, run: ./scripts/stop.sh"
echo "==========================================="

# Wait for any process to exit
wait -n

# If we get here, one of the processes has exited
echo "One of the services has stopped unexpectedly."
echo "Stopping remaining services..."

# Kill both processes
kill $FASTAPI_PID $CELERY_PID 2>/dev/null || true

# Clean up the PID file
rm -f .service_pids

exit 1