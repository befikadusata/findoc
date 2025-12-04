#!/bin/bash

# FinDocAI Stop Script
# This script stops all services started by run.sh

set -e

echo "Stopping FinDocAI services..."

# Stop the services using their PIDs
if [ -f .service_pids ]; then
    source .service_pids
    
    if [ ! -z "$FASTAPI_PID" ] && kill -0 $FASTAPI_PID 2>/dev/null; then
        echo "Stopping FastAPI server (PID: $FASTAPI_PID)..."
        kill $FASTAPI_PID
    else
        echo "FastAPI server not running or already stopped"
    fi
    
    if [ ! -z "$CELERY_PID" ] && kill -0 $CELERY_PID 2>/dev/null; then
        echo "Stopping Celery worker (PID: $CELERY_PID)..."
        kill $CELERY_PID
    else
        echo "Celery worker not running or already stopped"
    fi
    
    # Clean up the PID file
    rm -f .service_pids
else
    echo "Service PID file not found. Services may not have been started with run.sh or may already be stopped."
fi

# Also stop Docker services
echo "Stopping PostgreSQL and Redis containers..."
docker-compose down

echo "All FinDocAI services stopped."