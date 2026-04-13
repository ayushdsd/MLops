# Makefile for Customer Churn MLOps Pipeline
# Provides convenient commands for Docker operations

.PHONY: help build up down restart logs ps clean test

# Default target
help:
	@echo "Customer Churn MLOps Pipeline - Docker Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  make build       - Build all Docker images"
	@echo "  make up          - Start all services"
	@echo "  make down        - Stop all services"
	@echo "  make restart     - Restart all services"
	@echo "  make logs        - View logs from all services"
	@echo "  make ps          - Show running containers"
	@echo "  make clean       - Remove containers, networks, and volumes"
	@echo "  make test        - Run tests in Docker container"
	@echo "  make shell-api   - Open shell in API container"
	@echo "  make shell-ui    - Open shell in UI container"
	@echo ""

# Build all Docker images
build:
	@echo "Building Docker images..."
	docker-compose build

# Start all services
up:
	@echo "Starting all services..."
	docker-compose up -d
	@echo ""
	@echo "Services started! Access them at:"
	@echo "  - Streamlit UI:    http://localhost:8501"
	@echo "  - Prediction API:  http://localhost:8000"
	@echo "  - API Docs:        http://localhost:8000/docs"
	@echo "  - MLflow UI:       http://localhost:5000"
	@echo "  - Airflow UI:      http://localhost:8080 (admin/admin)"
	@echo ""

# Stop all services
down:
	@echo "Stopping all services..."
	docker-compose down

# Restart all services
restart:
	@echo "Restarting all services..."
	docker-compose restart

# View logs from all services
logs:
	docker-compose logs -f

# Show running containers
ps:
	docker-compose ps

# Remove containers, networks, and volumes
clean:
	@echo "WARNING: This will remove all containers, networks, and volumes!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker-compose down -v --rmi all; \
		echo "Cleanup complete!"; \
	else \
		echo "Cleanup cancelled."; \
	fi

# Run tests in Docker container
test:
	@echo "Running tests..."
	docker-compose exec prediction-api pytest -v

# Open shell in API container
shell-api:
	docker-compose exec prediction-api bash

# Open shell in UI container
shell-ui:
	docker-compose exec streamlit-ui bash

# View API logs
logs-api:
	docker-compose logs -f prediction-api

# View UI logs
logs-ui:
	docker-compose logs -f streamlit-ui

# View MLflow logs
logs-mlflow:
	docker-compose logs -f mlflow

# View Airflow logs
logs-airflow:
	docker-compose logs -f airflow-webserver airflow-scheduler

# Check health of all services
health:
	@echo "Checking service health..."
	@echo ""
	@echo "Prediction API:"
	@curl -s http://localhost:8000/health | python -m json.tool || echo "  ❌ Not responding"
	@echo ""
	@echo "MLflow:"
	@curl -s http://localhost:5000/health > /dev/null 2>&1 && echo "  ✅ Healthy" || echo "  ❌ Not responding"
	@echo ""
	@echo "Streamlit UI:"
	@curl -s http://localhost:8501/_stcore/health > /dev/null 2>&1 && echo "  ✅ Healthy" || echo "  ❌ Not responding"
	@echo ""
	@echo "Airflow:"
	@curl -s http://localhost:8080/health > /dev/null 2>&1 && echo "  ✅ Healthy" || echo "  ❌ Not responding"
	@echo ""

# Rebuild and restart specific service
rebuild-api:
	docker-compose build prediction-api
	docker-compose up -d prediction-api

rebuild-ui:
	docker-compose build streamlit-ui
	docker-compose up -d streamlit-ui

rebuild-mlflow:
	docker-compose build mlflow
	docker-compose up -d mlflow
