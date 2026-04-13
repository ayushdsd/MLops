@echo off
REM Quick start script for Customer Churn MLOps Pipeline (Windows)

echo ==========================================
echo Customer Churn MLOps Pipeline
echo Docker Quick Start
echo ==========================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not installed
    echo Please install Docker Desktop from https://docs.docker.com/desktop/install/windows-install/
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo Error: Docker Compose is not installed
    echo Please install Docker Compose from https://docs.docker.com/compose/install/
    exit /b 1
)

echo Docker and Docker Compose are installed
echo.

REM Check if Docker daemon is running
docker info >nul 2>&1
if errorlevel 1 (
    echo Error: Docker daemon is not running
    echo Please start Docker Desktop
    exit /b 1
)

echo Docker daemon is running
echo.

REM Build and start services
echo Building Docker images (this may take a few minutes)...
docker-compose build

echo.
echo Starting all services...
docker-compose up -d

echo.
echo Waiting for services to be healthy...
timeout /t 10 /nobreak >nul

REM Check service status
echo.
echo Checking service status...
docker-compose ps

echo.
echo ==========================================
echo Services are starting!
echo ==========================================
echo.
echo Access the services at:
echo   - Streamlit UI:    http://localhost:8501
echo   - Prediction API:  http://localhost:8000
echo   - API Docs:        http://localhost:8000/docs
echo   - MLflow UI:       http://localhost:5000
echo   - Airflow UI:      http://localhost:8080
echo.
echo Default Airflow credentials:
echo   Username: admin
echo   Password: admin
echo.
echo To view logs:
echo   docker-compose logs -f
echo.
echo To stop services:
echo   docker-compose down
echo.
echo For more commands, see DOCKER_DEPLOYMENT.md
echo ==========================================
