# Docker Deployment Guide

This guide explains how to deploy the Customer Churn MLOps Pipeline using Docker and Docker Compose.

## Prerequisites

- Docker Engine 20.10 or higher
- Docker Compose 2.0 or higher
- At least 4GB of available RAM
- At least 10GB of available disk space

## Quick Start

### 1. Build and Start All Services

```bash
# Build all Docker images and start services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service status
docker-compose ps
```

### 2. Initialize Airflow (First Time Only)

The `airflow-init` service will automatically initialize the Airflow database and create an admin user on first run.

**Default Airflow Credentials:**
- Username: `admin`
- Password: `admin`

### 3. Access Services

Once all services are running, you can access:

- **Streamlit UI**: http://localhost:8501
- **Prediction API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000
- **Airflow UI**: http://localhost:8080

## Service Architecture

The Docker Compose setup includes the following services:

### Core Services

1. **postgres** (Port 5432)
   - PostgreSQL database for Airflow metadata
   - Persistent storage via Docker volume

2. **mlflow** (Port 5000)
   - MLflow tracking server
   - Experiment tracking and model registry
   - Persistent artifact storage

3. **prediction-api** (Port 8000)
   - FastAPI REST API for predictions
   - Loads models from MLflow registry
   - Health check endpoint at `/health`

4. **streamlit-ui** (Port 8501)
   - User-friendly web interface
   - Connects to prediction API
   - Real-time churn predictions

5. **airflow-webserver** (Port 8080)
   - Airflow web UI
   - DAG management and monitoring

6. **airflow-scheduler**
   - Executes scheduled DAGs
   - Manages task orchestration

7. **airflow-init**
   - One-time initialization service
   - Creates database schema and admin user

## Volume Mounts

The following directories are mounted as volumes:

- `./data` → `/app/data` (API) and `/opt/airflow/data` (Airflow)
- `./models` → `/app/models` (API) and `/opt/airflow/models` (Airflow)
- `./logs` → `/app/logs` (API/UI) and `/opt/airflow/logs` (Airflow)
- `./dags` → `/opt/airflow/dags` (Airflow)
- `./src` → `/opt/airflow/src` (Airflow)

Persistent volumes:
- `churn-postgres-data` → PostgreSQL data
- `churn-mlflow-artifacts` → MLflow artifacts
- `churn-mlflow-backend` → MLflow backend store

## Environment Variables

Environment variables can be configured in the `docker-compose.yml` file or via a `.env` file.

Key variables:

```bash
# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000

# API
API_PORT=8000
MODEL_NAME=churn_model
MODEL_STAGE=Production

# UI
UI_PORT=8501
API_URL=http://prediction-api:8000

# Airflow
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow
```

## Common Operations

### Start Services

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d prediction-api
```

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f prediction-api

# Last 100 lines
docker-compose logs --tail=100 streamlit-ui
```

### Rebuild Images

```bash
# Rebuild all images
docker-compose build

# Rebuild specific service
docker-compose build prediction-api

# Rebuild without cache
docker-compose build --no-cache
```

### Scale Services

```bash
# Scale prediction API to 3 instances
docker-compose up -d --scale prediction-api=3
```

### Execute Commands in Containers

```bash
# Open bash shell in API container
docker-compose exec prediction-api bash

# Run Python script
docker-compose exec prediction-api python scripts/generate_sample_data.py

# Check API health
docker-compose exec prediction-api curl http://localhost:8000/health
```

## Health Checks

All services include health checks:

```bash
# Check health status
docker-compose ps

# View health check logs
docker inspect churn-prediction-api | grep -A 10 Health
```

Health check endpoints:
- API: `http://localhost:8000/health`
- MLflow: `http://localhost:5000/health`
- Streamlit: `http://localhost:8501/_stcore/health`
- Airflow: `http://localhost:8080/health`

## Troubleshooting

### Service Won't Start

1. Check logs:
   ```bash
   docker-compose logs <service-name>
   ```

2. Verify dependencies are healthy:
   ```bash
   docker-compose ps
   ```

3. Restart service:
   ```bash
   docker-compose restart <service-name>
   ```

### Port Already in Use

If a port is already in use, modify the port mapping in `docker-compose.yml`:

```yaml
ports:
  - "8001:8000"  # Map to different host port
```

### Out of Memory

Increase Docker memory limit in Docker Desktop settings or add memory limits to services:

```yaml
services:
  prediction-api:
    deploy:
      resources:
        limits:
          memory: 2G
```

### Database Connection Issues

1. Ensure PostgreSQL is healthy:
   ```bash
   docker-compose ps postgres
   ```

2. Reset database (WARNING: deletes data):
   ```bash
   docker-compose down -v
   docker-compose up -d
   ```

### Model Not Loading

1. Check if model exists in MLflow:
   - Visit http://localhost:5000
   - Verify model is registered and in "Production" stage

2. Check API logs:
   ```bash
   docker-compose logs prediction-api
   ```

3. Manually load model:
   ```bash
   docker-compose exec prediction-api python -c "from src.api.predictor import Predictor; p = Predictor('http://mlflow:5000'); p.load_model('churn_model', 'Production')"
   ```

## Production Considerations

### Security

1. **Change default passwords** in `docker-compose.yml`:
   - PostgreSQL password
   - Airflow admin password
   - Airflow Fernet key

2. **Use secrets management**:
   ```yaml
   secrets:
     postgres_password:
       file: ./secrets/postgres_password.txt
   ```

3. **Enable HTTPS** with reverse proxy (nginx, traefik)

4. **Restrict CORS** in API configuration

### Performance

1. **Increase worker count** for API:
   ```yaml
   environment:
     - API_WORKERS=8
   ```

2. **Use production database** (PostgreSQL instead of SQLite for MLflow):
   ```yaml
   command: ["mlflow", "server", 
             "--backend-store-uri", "postgresql://user:pass@postgres/mlflow"]
   ```

3. **Enable caching** for predictions

4. **Use load balancer** for multiple API instances

### Monitoring

1. **Add monitoring stack** (Prometheus, Grafana):
   ```yaml
   prometheus:
     image: prom/prometheus
     # ... configuration
   ```

2. **Enable metrics endpoints**

3. **Set up alerting** for service failures

### Backup

1. **Backup volumes regularly**:
   ```bash
   docker run --rm -v churn-postgres-data:/data -v $(pwd):/backup ubuntu tar czf /backup/postgres-backup.tar.gz /data
   ```

2. **Backup MLflow artifacts**:
   ```bash
   docker run --rm -v churn-mlflow-artifacts:/data -v $(pwd):/backup ubuntu tar czf /backup/mlflow-backup.tar.gz /data
   ```

## Cleanup

### Remove All Containers and Images

```bash
# Stop and remove containers
docker-compose down

# Remove images
docker-compose down --rmi all

# Remove volumes (WARNING: deletes all data)
docker-compose down -v --rmi all
```

### Remove Unused Resources

```bash
# Remove unused containers, networks, images
docker system prune -a

# Remove unused volumes
docker volume prune
```

## Next Steps

1. Train your first model (see main README.md)
2. Configure Airflow DAG for automated retraining
3. Test the complete workflow
4. Set up monitoring and alerting
5. Configure backups

For more information, see the main [README.md](README.md) file.
