# Customer Churn Prediction API

FastAPI-based REST API for predicting customer churn probability.

## Features

- **POST /predict**: Generate churn predictions with input validation
- **GET /health**: Health check endpoint (200 when healthy, 503 when model not loaded)
- **GET /model-info**: Get model metadata and version information
- **CORS Support**: Configured for cross-origin requests
- **Comprehensive Error Handling**: Structured error responses with appropriate HTTP status codes
- **Request Logging**: All requests and errors are logged
- **Input Validation**: Validates all customer attributes before prediction

## Quick Start

### 1. Start the API Server

```bash
# Using uvicorn directly
python -m uvicorn src.api.app:app --reload --port 8000

# Or using the app's main entry point
python -m src.api.app
```

The API will be available at `http://localhost:8000`

### 2. View API Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 3. Make a Prediction

```python
import requests

customer_data = {
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure": 12,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "Yes",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 70.35,
    "total_charges": 844.20
}

response = requests.post("http://localhost:8000/predict", json=customer_data)
print(response.json())
```

## API Endpoints

### POST /predict

Generate a churn prediction for a customer.

**Request Body:**
```json
{
  "gender": "Female",
  "senior_citizen": 0,
  "partner": "Yes",
  "dependents": "No",
  "tenure": 12,
  "phone_service": "Yes",
  "multiple_lines": "No",
  "internet_service": "Fiber optic",
  "online_security": "No",
  "online_backup": "Yes",
  "device_protection": "No",
  "tech_support": "No",
  "streaming_tv": "Yes",
  "streaming_movies": "Yes",
  "contract": "Month-to-month",
  "paperless_billing": "Yes",
  "payment_method": "Electronic check",
  "monthly_charges": 70.35,
  "total_charges": 844.20
}
```

**Success Response (200):**
```json
{
  "churn_probability": 0.75,
  "risk_label": "High",
  "model_version": "3",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

**Error Response (400 - Validation Error):**
```json
{
  "error": "Validation Error",
  "detail": "Validation failed with 2 error(s):\n- tenure: Field 'tenure' must be greater than or equal to 0, got -5\n- gender: Field 'gender' must be one of ['Male', 'Female'], got 'Other'",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "path": "/predict"
}
```

**Error Response (503 - Model Not Loaded):**
```json
{
  "error": "Service Unavailable",
  "detail": "Model not loaded. Please try again later.",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "path": "/predict"
}
```

### GET /health

Check the health status of the API service.

**Success Response (200 - Healthy):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "3"
}
```

**Error Response (503 - Unhealthy):**
```json
{
  "status": "unhealthy",
  "model_loaded": false,
  "model_version": null
}
```

### GET /model-info

Get information about the loaded model.

**Response (200):**
```json
{
  "model_name": "churn_model",
  "model_version": "3",
  "model_loaded": true,
  "preprocessing_pipeline_loaded": true
}
```

## Configuration

The API can be configured using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | MLflow tracking server URI | `http://localhost:5000` |
| `MODEL_NAME` | Name of the model in MLflow registry | `churn_model` |
| `MODEL_STAGE` | Model stage to load (None, Staging, Production) | `Production` |
| `PREPROCESSING_RUN_ID` | MLflow run ID containing preprocessing pipeline | None |
| `PREPROCESSING_PIPELINE_PATH` | Local path to preprocessing pipeline | None |
| `API_PORT` | Port for the API server | `8000` |

### Example Configuration

```bash
# .env file
MLFLOW_TRACKING_URI=http://localhost:5000
MODEL_NAME=churn_model
MODEL_STAGE=Production
API_PORT=8000
```

## Input Validation

The API validates all customer attributes:

### Required Fields
All 19 customer attributes must be present in the request.

### Numerical Fields
- `tenure`: Integer ≥ 0
- `monthly_charges`: Float > 0
- `total_charges`: Float ≥ 0

### Categorical Fields
- `gender`: "Male" or "Female"
- `senior_citizen`: 0 or 1
- `partner`: "Yes" or "No"
- `dependents`: "Yes" or "No"
- `phone_service`: "Yes" or "No"
- `multiple_lines`: "Yes", "No", or "No phone service"
- `internet_service`: "DSL", "Fiber optic", or "No"
- `online_security`: "Yes", "No", or "No internet service"
- `online_backup`: "Yes", "No", or "No internet service"
- `device_protection`: "Yes", "No", or "No internet service"
- `tech_support`: "Yes", "No", or "No internet service"
- `streaming_tv`: "Yes", "No", or "No internet service"
- `streaming_movies`: "Yes", "No", or "No internet service"
- `contract`: "Month-to-month", "One year", or "Two year"
- `paperless_billing`: "Yes" or "No"
- `payment_method`: "Electronic check", "Mailed check", "Bank transfer (automatic)", or "Credit card (automatic)"

## Risk Classification

Churn probability is classified into three risk levels:

- **Low**: 0.0 ≤ probability < 0.33
- **Medium**: 0.33 ≤ probability < 0.66
- **High**: 0.66 ≤ probability ≤ 1.0

## Error Handling

The API provides structured error responses for all error conditions:

- **400 Bad Request**: Invalid input data (validation errors)
- **500 Internal Server Error**: Unexpected server errors
- **503 Service Unavailable**: Model not loaded or service not ready

All error responses include:
- `error`: Error type/category
- `detail`: Detailed error message
- `timestamp`: ISO format timestamp
- `path`: Request path where error occurred

## Logging

The API logs all requests and errors:

- Request method and path
- Response status codes
- Validation errors
- Prediction results
- Model loading status
- Exception stack traces

Logs are written to both console and log files.

## CORS Configuration

CORS middleware is configured to allow cross-origin requests. For production, configure the `allow_origins` parameter appropriately:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Testing

Run the unit tests:

```bash
pytest tests/unit/test_app.py -v
```

Run the demo script:

```bash
python examples/api_demo.py
```

## Architecture

The API integrates with:

- **Predictor Module** (`src/api/predictor.py`): Handles model loading and prediction
- **Validators Module** (`src/api/validators.py`): Validates customer input data
- **Models Module** (`src/api/models.py`): Pydantic models for request/response
- **MLflow**: Loads models from Model Registry

## Development

### Running in Development Mode

```bash
uvicorn src.api.app:app --reload --port 8000
```

The `--reload` flag enables auto-reload on code changes.

### Running in Production

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

Use multiple workers for better performance in production.

## Docker Deployment

The API can be containerized using Docker:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t churn-api .
docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://mlflow:5000 churn-api
```

## Troubleshooting

### Model Not Loading

If the health endpoint returns 503:

1. Check MLflow server is running
2. Verify `MLFLOW_TRACKING_URI` is correct
3. Ensure model exists in registry with specified name and stage
4. Check logs for detailed error messages

### Validation Errors

If predictions return 400:

1. Verify all 19 required fields are present
2. Check field values match expected categorical values
3. Ensure numerical fields are within valid ranges
4. Review error detail message for specific issues

### Connection Errors

If unable to connect to API:

1. Verify server is running
2. Check port is not in use by another process
3. Ensure firewall allows connections on the port
4. Check network connectivity
