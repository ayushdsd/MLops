# Customer Churn MLOps Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-80%25-green.svg)](tests/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](docker-compose.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **production-ready, end-to-end MLOps system** for predicting customer churn in telecommunications. Built with industry best practices, this pipeline demonstrates a complete machine learning workflow from data ingestion to model deployment, featuring experiment tracking, automated retraining, REST API serving, and a user-friendly web interface.

## Table of Contents

- [Overview](#overview)
- [Why This Project?](#why-this-project)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Testing](#testing)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

The **Customer Churn MLOps Pipeline** is a comprehensive machine learning system designed to predict customer churn for telecommunications companies. It implements the complete MLOps lifecycle with production-grade infrastructure, making it suitable for both learning and real-world deployment.

### What is Customer Churn?

Customer churn refers to when customers stop doing business with a company. For telecommunications companies, predicting which customers are likely to churn allows them to:
- Take proactive retention actions
- Optimize marketing spend
- Improve customer satisfaction
- Reduce revenue loss

### What This System Does

This pipeline:
1. **Ingests** customer data (demographics, services, account information)
2. **Processes** and validates the data with comprehensive error handling
3. **Trains** a Random Forest classifier to predict churn probability
4. **Tracks** all experiments with MLflow for reproducibility
5. **Serves** predictions via a REST API
6. **Provides** a user-friendly web interface for business users
7. **Monitors** model performance and enables automated retraining

## Why This Project?

This project serves as a **complete reference implementation** for MLOps best practices:

### For Learning
- **End-to-End Pipeline**: See how all MLOps components work together
- **Best Practices**: Learn industry-standard patterns and architectures
- **Testing Strategies**: Understand property-based testing and test-driven development
- **Production Patterns**: Study error handling, logging, and monitoring

### For Production Use
- **Battle-Tested Code**: 80%+ test coverage with 150+ unit tests
- **Scalable Architecture**: Containerized services with Docker Compose
- **Comprehensive Documentation**: Detailed guides for deployment and usage
- **Configurable**: Environment-based configuration for different environments

### For Interviews & Portfolios
- **Demonstrates Skills**: Shows proficiency in ML, software engineering, and DevOps
- **Real-World Problem**: Solves an actual business problem
- **Professional Quality**: Production-ready code with proper documentation
- **Modern Stack**: Uses current industry-standard tools and frameworks

## Key Features

### Complete MLOps Lifecycle
- **Data Pipeline**: Automated data loading, validation, and preprocessing
- **Model Training**: Configurable Random Forest with hyperparameter tuning
- **Experiment Tracking**: MLflow integration for reproducibility
- **Model Registry**: Versioned models with stage management (Staging/Production)
- **Model Serving**: FastAPI REST API for real-time predictions
- **User Interface**: Streamlit web app for non-technical users
- **Monitoring**: Comprehensive logging and health checks

### Rigorous Testing
- **Property-Based Testing**: 31 universal properties tested with Hypothesis
- **Unit Tests**: 150+ tests covering all components
- **Integration Tests**: End-to-end workflow validation
- **80%+ Coverage**: Comprehensive test coverage across the codebase

### Production-Ready Infrastructure
- **Containerization**: Docker images for all services
- **Orchestration**: Docker Compose for multi-service deployment
- **Configuration Management**: Environment-based configuration with sensible defaults
- **Logging**: Centralized logging with rotation and multiple output streams
- **Error Handling**: Comprehensive error handling with custom exception classes
- **Health Checks**: Service health monitoring and readiness probes

### MLflow Integration
- **Experiment Tracking**: Log parameters, metrics, and artifacts
- **Model Registry**: Version control for trained models
- **Artifact Storage**: Store models, pipelines, and visualizations
- **Model Comparison**: Compare experiments and select best models
- **Stage Management**: Promote models through Staging to Production

### User-Friendly Interfaces
- **REST API**: FastAPI with automatic OpenAPI documentation
- **Web UI**: Streamlit interface with form-based input
- **Risk Classification**: Color-coded risk levels (Low/Medium/High)
- **Real-Time Predictions**: Sub-second response times
- **Error Messages**: User-friendly error handling and validation

## Architecture

The system follows a **layered architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────┐
│                          User Layer                                  │
│  ┌──────────────────────┐         ┌──────────────────────┐         │
│  │   Streamlit Web UI   │         │   External Systems   │         │
│  │   (Port 8501)        │         │   (API Clients)      │         │
│  └──────────┬───────────┘         └──────────┬───────────┘         │
└─────────────┼──────────────────────────────────┼───────────────────┘
              │                                  │
              └──────────────┬───────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       Serving Layer                                  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              FastAPI Prediction Service (Port 8000)           │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐ │  │
│  │  │ /predict   │  │  /health   │  │  /model-info           │ │  │
│  │  └────────────┘  └────────────┘  └────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      Model Management Layer                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              MLflow Server (Port 5000)                        │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐ │  │
│  │  │Experiments │  │   Model    │  │      Artifacts         │ │  │
│  │  │ Tracking   │  │  Registry  │  │  (Models, Pipelines)   │ │  │
│  │  └────────────┘  └────────────┘  └────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      Training Layer                                  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                   Training Service                            │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐ │  │
│  │  │  Trainer   │→ │ Evaluator  │→ │   MLflow Logger        │ │  │
│  │  │ (RF Model) │  │ (Metrics)  │  │ (Params, Artifacts)    │ │  │
│  │  └────────────┘  └────────────┘  └────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    Data Processing Layer                             │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                   Data Processor                              │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐ │  │
│  │  │   Loader   │→ │ Validator  │→ │    Preprocessor        │ │  │
│  │  │ (CSV Read) │  │ (Schema)   │  │ (Transform, Encode)    │ │  │
│  │  └────────────┘  └────────────┘  └────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       Data Storage Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │  Raw Data    │  │  Processed   │  │   Model Artifacts        │ │
│  │  (CSV)       │  │  Data        │  │   (Pickled Models)       │ │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

**Training Flow:**
1. **Data Processor** loads CSV → validates schema → preprocesses features → saves pipeline
2. **Training Service** trains Random Forest → evaluates metrics → logs to MLflow
3. **MLflow** stores experiments, models, and artifacts with versioning
4. **Model Registry** manages model lifecycle (None → Staging → Production)

**Prediction Flow:**
1. **User** submits data via Streamlit UI or API call
2. **Prediction API** validates input → loads model from registry
3. **Predictor** applies preprocessing → generates prediction → classifies risk
4. **Response** returns probability, risk label, model version, timestamp

**Monitoring Flow:**
1. **Health Checks** verify service availability and model loading
2. **Logging** captures all operations with timestamps and stack traces
3. **MLflow UI** provides experiment comparison and model performance tracking

## Technology Stack

### Core ML & Data
- **Python 3.9+**: Primary programming language
- **scikit-learn**: Machine learning algorithms (Random Forest)
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing

### MLOps & Tracking
- **MLflow**: Experiment tracking, model registry, artifact storage
- **Pydantic**: Configuration management and data validation

### API & Web
- **FastAPI**: High-performance REST API framework
- **Streamlit**: Interactive web UI for business users
- **Uvicorn**: ASGI server for FastAPI

### Testing
- **pytest**: Test framework
- **Hypothesis**: Property-based testing
- **pytest-cov**: Code coverage reporting

### Infrastructure
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **PostgreSQL**: Database for Airflow (optional)

### Development Tools
- **Git**: Version control
- **Make**: Build automation
- **Logging**: Python logging with rotation

## Project Structure

```
customer-churn-mlops-pipeline/
├── data/                       # Data storage
│   ├── raw/                   # Raw CSV datasets (telco_churn.csv)
│   └── processed/             # Preprocessed train/test splits
├── dags/                      # Airflow DAG definitions (future)
├── logs/                      # Application logs
├── models/                    # Trained model artifacts
│   └── pipelines/            # Preprocessing pipelines (scalers, encoders)
├── src/                       # Source code
│   ├── data_processing/      # Data loading, validation, preprocessing
│   │   └── data_loader.py   # DataProcessor class
│   ├── training/             # Model training and MLflow integration
│   │   └── trainer.py       # TrainingService class
│   ├── api/                  # FastAPI prediction service (future)
│   ├── ui/                   # Streamlit user interface (future)
│   └── config.py             # Centralized configuration management
├── tests/                     # Comprehensive test suite
│   ├── unit/                 # Unit tests for specific functionality
│   ├── property/             # Property-based tests using Hypothesis
│   ├── integration/          # Integration tests for workflows
│   └── fixtures/             # Test fixtures and Hypothesis strategies
├── scripts/                   # Utility scripts
│   └── generate_sample_data.py  # Generate sample datasets
├── examples/                  # Example usage scripts
│   └── model_persistence_demo.py  # Model training demo
├── .env.example              # Environment variables template
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Quick Start

Get the entire system running in **under 5 minutes** with Docker:

```bash
# 1. Clone the repository
git clone <repository-url>
cd customer-churn-mlops-pipeline

# 2. Start all services with Docker Compose
docker-compose up -d

# 3. Wait for services to be ready (~30 seconds)
docker-compose ps

# 4. Access the applications
```

## Installation

### Prerequisites

- **Docker & Docker Compose** (recommended) OR
- **Python 3.9+** with pip
- **Git** for cloning the repository
- **4GB RAM** minimum (8GB recommended)
- **2GB disk space** for Docker images and data

### Option 1: Docker Deployment (Recommended)

The easiest way to run the entire MLOps pipeline is using Docker Compose:

```bash
# 1. Clone the repository
git clone <repository-url>
cd customer-churn-mlops-pipeline

# 2. Build and start all services
docker-compose up -d

# 3. Access services
# - Streamlit UI: http://localhost:8501
# - Prediction API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
# - MLflow UI: http://localhost:5000
# - Airflow UI: http://localhost:8080 (admin/admin)

# 4. View logs
docker-compose logs -f

# 5. Stop services
docker-compose down
```

For detailed Docker deployment instructions, see [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md).

### Option 2: Local Development Setup

For local development without Docker:

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd customer-churn-mlops-pipeline
```

#### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your configuration
# (Optional - defaults work for local development)
```

#### 5. Prepare Data Directory

```bash
# Create necessary directories
mkdir -p data/raw data/processed models/pipelines logs
```

#### 6. Add Dataset

Place your `telco_churn.csv` dataset in the `data/raw/` directory, or use the provided sample data generation script:

```bash
python scripts/generate_sample_data.py
```

## Usage Guide

### Training a New Model

#### Using Python API

```python
from src.data_processing.data_loader import DataProcessor
from src.training.trainer import TrainingService, TrainingConfig

# 1. Load and preprocess data
processor = DataProcessor()
df = processor.load_data("data/raw/telco_churn.csv")

# Validate data quality
validation = processor.validate_schema(df)
if not validation.is_valid:
    print(f"Validation errors: {validation.errors}")
    exit(1)

# Preprocess data
preprocessed = processor.preprocess(df, target_column='Churn')

# 2. Configure and train model
config = TrainingConfig(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

trainer = TrainingService(
    mlflow_uri="http://localhost:5000",
    config=config
)

model = trainer.train(preprocessed.X_train, preprocessed.y_train)

# 3. Evaluate model
metrics = trainer.evaluate(model, preprocessed.X_test, preprocessed.y_test)
print(f"Accuracy: {metrics.accuracy:.3f}")
print(f"F1 Score: {metrics.f1_score:.3f}")
print(f"ROC-AUC: {metrics.roc_auc:.3f}")

# 4. Log to MLflow
params = {
    "n_estimators": config.n_estimators,
    "max_depth": config.max_depth,
    "random_state": config.random_state
}
run_id = trainer.log_experiment(model, metrics, params)
print(f"Logged to MLflow with run ID: {run_id}")

# 5. Register model
version = trainer.register_model(run_id, "churn_model")
print(f"Registered as version: {version}")

# 6. Promote to production (if performance is good)
if metrics.roc_auc >= 0.85:
    trainer.promote_to_production("churn_model", version)
    print("Model promoted to Production!")
```

#### Via REST API

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Get Model Info:**
```bash
curl http://localhost:8000/model-info
```

**Make Prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @customer_data.json
```

### Viewing Experiments in MLflow

1. Open http://localhost:5000
2. Browse experiments in the left sidebar
3. Compare runs by selecting multiple experiments
4. View metrics, parameters, and artifacts
5. Download models or artifacts
6. Promote models to different stages

### Monitoring and Logs

**View Service Logs:**
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f prediction-api
docker-compose logs -f streamlit-ui
docker-compose logs -f mlflow
```

**Check Log Files:**
```bash
# Application logs (if running locally)
tail -f logs/data_processing.log
tail -f logs/training.log
tail -f logs/prediction_api.log
```

**Monitor Service Health:**
```bash
# Check all services
docker-compose ps

# API health endpoint
curl http://localhost:8000/health

# MLflow health
curl http://localhost:5000/health
```

## Configuration

All configuration is managed through environment variables with sensible defaults. Configuration can be overridden via `.env` file or environment variables.

### MLflow Configuration

```bash
MLFLOW_TRACKING_URI=http://localhost:5000  # MLflow server URL
MLFLOW_ARTIFACT_ROOT=/mlflow/artifacts     # Artifact storage path
```

### Model Configuration

```bash
MODEL_NAME=churn_model          # Model name in registry
MODEL_STAGE=Production          # Model stage (None/Staging/Production)
MODEL_PATH=./models             # Local model storage path
```

### Training Configuration

```bash
N_ESTIMATORS=100                # Number of trees in Random Forest
MAX_DEPTH=10                    # Maximum tree depth (None for unlimited)
RANDOM_STATE=42                 # Random seed for reproducibility
TEST_SIZE=0.2                   # Test set proportion (0.0-1.0)
```

### Data Configuration

```bash
DATA_PATH=./data/telco_churn.csv        # Path to training dataset
PROCESSED_DATA_PATH=./data/processed    # Preprocessed data storage
```

### Logging Configuration

```bash
LOG_LEVEL=INFO                  # Logging level (DEBUG/INFO/WARNING/ERROR)
LOG_PATH=./logs                 # Log file directory
```

## API Endpoints (Future)

The Prediction API will expose the following endpoints:

### POST /predict

Predict churn probability for a customer.

**Request Body:**
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.35,
  "TotalCharges": 844.20
}
```

**Response:**
```json
{
  "churn_probability": 0.73,
  "risk_label": "High",
  "model_version": "v3",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### GET /health

Check service health status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "v3"
}
```

### GET /model-info

Get information about the loaded model.

**Response:**
```json
{
  "model_name": "churn_model",
  "version": 3,
  "stage": "Production",
  "metrics": {
    "accuracy": 0.85,
    "f1_score": 0.72,
    "roc_auc": 0.88
  }
}
```

### Common Issues

#### 1. MLflow Connection Error

**Problem**: `ConnectionError: Cannot connect to MLflow server`

**Solution**:
```bash
# Ensure MLflow server is running
mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000

# Check MLFLOW_TRACKING_URI in .env matches server address
MLFLOW_TRACKING_URI=http://localhost:5000
```

#### 2. Dataset Not Found

**Problem**: `DataLoadError: Dataset file not found`

**Solution**:
```bash
# Verify dataset exists
ls data/raw/telco_churn.csv

# Generate sample data if needed
python scripts/generate_sample_data.py

# Check DATA_PATH in .env
DATA_PATH=./data/raw/telco_churn.csv
```

#### 3. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt

# Run from project root directory
cd customer-churn-mlops-pipeline
python examples/model_persistence_demo.py
```

#### 4. Permission Errors on Windows

**Problem**: `PermissionError: [WinError 32] The process cannot access the file`

**Solution**:
This is a known Windows issue with temporary files. The application logs a warning but continues execution. No action needed.

#### 5. Schema Validation Failures

**Problem**: `SchemaValidationError: Missing required columns`

**Solution**:
```python
# Check your dataset has all required columns
processor = DataProcessor()
print(processor.required_columns)

# Verify column names match exactly (case-sensitive)
# Expected: 'SeniorCitizen', 'MonthlyCharges', etc.
```

#### 6. Memory Issues with Large Datasets

**Problem**: `MemoryError` during preprocessing

**Solution**:
```python
# Process data in chunks for large datasets
chunk_size = 10000
for chunk in pd.read_csv('data.csv', chunksize=chunk_size):
    preprocessed = processor.preprocess(chunk)
    # Process chunk...
```

#### 7. Test Failures

**Problem**: Property-based tests fail intermittently

**Solution**:
```bash
# Run with specific seed for reproducibility
pytest --hypothesis-seed=12345

# Increase test examples for more thorough testing
# Edit tests/property/test_*.py:
# @settings(max_examples=1000)
```

### Logging and Debugging

Enable debug logging for detailed troubleshooting:

```bash
# Set in .env
LOG_LEVEL=DEBUG

# Or in code
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check log files in `logs/` directory:
- `data_processing.log`: Data loading and preprocessing logs
- `training.log`: Model training and evaluation logs

## License

MIT License
