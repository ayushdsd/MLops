#!/bin/bash
# Quick start script for Customer Churn MLOps Pipeline

set -e

echo "=========================================="
echo "Customer Churn MLOps Pipeline"
echo "Docker Quick Start"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed"
    echo "Please install Docker from https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Error: Docker Compose is not installed"
    echo "Please install Docker Compose from https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"
echo ""

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo "❌ Error: Docker daemon is not running"
    echo "Please start Docker Desktop or Docker daemon"
    exit 1
fi

echo "✅ Docker daemon is running"
echo ""

# Build and start services
echo "Building Docker images (this may take a few minutes)..."
docker-compose build

echo ""
echo "Starting all services..."
docker-compose up -d

echo ""
echo "Waiting for services to be healthy..."
sleep 10

# Check service health
echo ""
echo "Checking service status..."
docker-compose ps

echo ""
echo "=========================================="
echo "✅ Services are starting!"
echo "=========================================="
echo ""
echo "Access the services at:"
echo "  - Streamlit UI:    http://localhost:8501"
echo "  - Prediction API:  http://localhost:8000"
echo "  - API Docs:        http://localhost:8000/docs"
echo "  - MLflow UI:       http://localhost:5000"
echo "  - Airflow UI:      http://localhost:8080"
echo ""
echo "Default Airflow credentials:"
echo "  Username: admin"
echo "  Password: admin"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f"
echo ""
echo "To stop services:"
echo "  docker-compose down"
echo ""
echo "For more commands, see DOCKER_DEPLOYMENT.md"
echo "=========================================="
