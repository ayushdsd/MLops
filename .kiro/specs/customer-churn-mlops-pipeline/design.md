# Design Document: Customer Churn MLOps Pipeline

## Overview

The Customer Churn MLOps Pipeline is a production-ready machine learning system that predicts customer churn using a Random Forest classifier. The architecture follows MLOps best practices with clear separation between data processing, model training, serving, and orchestration layers.

The system consists of six primary components:

1. **Data Processing Layer**: Handles data ingestion, validation, preprocessing, and feature engineering
2. **Training Service**: Manages model training, evaluation, and experiment tracking
3. **MLflow Integration**: Provides experiment tracking, model registry, and versioning
4. **Prediction API**: FastAPI-based REST service for real-time predictions
5. **User Interface**: Streamlit application for business users
6. **Orchestration Layer**: Airflow DAG for automated retraining workflows

The design emphasizes modularity, testability, and production readiness with comprehensive error handling, logging, health monitoring, and containerized deployment.

## Architecture

### System Architecture Diagram

```mermaid
graph TB
    subgraph "Data Layer"
        CSV[CSV Dataset]
        DataStore[(Data Storage)]
    end
    
    subgraph "Processing Layer"
        DataProc[Data Processor]
        Validator[Data Validator]
    end
    
    subgraph "Training Layer"
        Trainer[Training Service]
        MLflow[MLflow Server]
        Registry[(Model Registry)]
    end
    
    subgraph "Serving Layer"
        API[Prediction API<br/>FastAPI]
        UI[Streamlit UI]
    end
    
    subgraph "Orchestration Layer"
        Airflow[Airflow Scheduler]
        DAG[Retraining DAG]
    end
    
    CSV --> DataProc
    DataProc --> DataStore
    DataProc --> Validator
    Validator --> Trainer
    Trainer --> MLflow
    MLflow --> Registry
    Registry --> API
    API --> UI
    Airflow --> DAG
    DAG --> Trainer
    
    style MLflow fill:#f9f,stroke:#333
    style API fill:#bbf,stroke:#333
    style UI fill:#bfb,stroke:#333
```

### Component Interaction Flow

**Training Flow:**
1. Data Processor loads CSV → validates schema → preprocesses features → saves transformation pipeline
2. Training Service loads preprocessed data → trains Random Forest → evaluates metrics
3. MLflow logs hyperparameters, metrics, and model artifacts
4. Model Registry stores versioned model with metadata
5. Training Service promotes model to Production if performance threshold met

**Prediction Flow:**
1. User submits data via Streamlit UI or direct API call
2. Prediction API validates input schema
3. API loads transformation pipeline → applies preprocessing
4. API loads Production model from registry → generates prediction
5. API returns churn probability and risk label

**Retraining Flow:**
1. Airflow scheduler triggers Retraining DAG on weekly schedule
2. DAG executes training pipeline with fresh data
3. New model logged to MLflow with experiment tracking
4. DAG compares new model ROC-AUC vs current Production model
5. If improvement ≥ 0.01, DAG promotes new model to Production
6. On failure, DAG sends alert notification

## Components and Interfaces

### 1. Data Processor Component

**Responsibilities:**
- Load raw CSV data from file system
- Validate data schema and quality
- Handle missing values via imputation
- Encode categorical variables (label encoding, one-hot encoding)
- Scale numerical features using standardization
- Split data into train/test sets (80-20)
- Save preprocessing pipeline for inference

**Interface:**

```python
class DataProcessor:
    def load_data(self, file_path: str) -> pd.DataFrame
    def validate_schema(self, df: pd.DataFrame) -> ValidationResult
    def preprocess(self, df: pd.DataFrame) -> PreprocessedData
    def save_pipeline(self, pipeline: Pipeline, path: str) -> None
    def load_pipeline(self, path: str) -> Pipeline
```

**Key Methods:**
- `load_data()`: Reads CSV, returns DataFrame or raises FileNotFoundError
- `validate_schema()`: Checks required columns, data types, value ranges
- `preprocess()`: Applies full transformation pipeline, returns train/test splits
- `save_pipeline()`: Serializes sklearn Pipeline to disk
- `load_pipeline()`: Deserializes pipeline for inference

### 2. Training Service Component

**Responsibilities:**
- Train Random Forest classifier with configurable hyperparameters
- Evaluate model on test set (accuracy, F1, ROC-AUC)
- Log experiments to MLflow (params, metrics, artifacts)
- Register models in Model Registry with versioning
- Promote models to Production stage based on performance

**Interface:**

```python
class TrainingService:
    def __init__(self, mlflow_uri: str, config: TrainingConfig)
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier
    def evaluate(self, model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray) -> Metrics
    def log_experiment(self, model: RandomForestClassifier, metrics: Metrics, params: dict) -> str
    def register_model(self, run_id: str, model_name: str) -> ModelVersion
    def promote_to_production(self, model_name: str, version: int) -> None
```

**Configuration:**
```python
@dataclass
class TrainingConfig:
    n_estimators: int = 100
    max_depth: Optional[int] = None
    random_state: int = 42
    test_size: float = 0.2
```

### 3. MLflow Integration

**Responsibilities:**
- Track experiments with parameters, metrics, and artifacts
- Store model artifacts with versioning
- Manage model lifecycle stages (None, Staging, Production)
- Provide web UI for experiment comparison
- Serve models via REST API

**Configuration:**
- Tracking URI: `http://mlflow:5000`
- Artifact storage: `/mlflow/artifacts`
- Backend store: SQLite or PostgreSQL
- Model registry: Integrated with tracking server

**Logged Artifacts:**
- Trained model (pickle or MLflow format)
- Preprocessing pipeline
- Feature importance plots
- Confusion matrix
- ROC curve

### 4. Prediction API Component

**Responsibilities:**
- Expose REST endpoints for predictions and health checks
- Load Production model from Model Registry on startup
- Validate incoming prediction requests
- Apply preprocessing pipeline to input data
- Return churn probability and risk classification
- Handle errors with appropriate HTTP status codes

**Interface:**

```python
# FastAPI endpoints
@app.post("/predict")
async def predict(customer: CustomerInput) -> PredictionResponse

@app.get("/health")
async def health() -> HealthResponse

@app.get("/model-info")
async def model_info() -> ModelInfo
```

**Request/Response Models:**

```python
class CustomerInput(BaseModel):
    gender: str
    senior_citizen: int
    partner: str
    dependents: str
    tenure: int
    phone_service: str
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    contract: str
    paperless_billing: str
    payment_method: str
    monthly_charges: float
    total_charges: float

class PredictionResponse(BaseModel):
    churn_probability: float
    risk_label: str  # "Low", "Medium", "High"
    model_version: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str  # "healthy", "unhealthy"
    model_loaded: bool
    model_version: Optional[str]
```

**Validation Rules:**
- All required fields must be present
- Numerical fields: tenure ≥ 0, monthly_charges > 0, total_charges ≥ 0
- Categorical fields: must match expected values from training data
- Gender: ["Male", "Female"]
- Binary fields: ["Yes", "No"]

**Risk Classification:**
- Low: 0.0 ≤ probability < 0.33
- Medium: 0.33 ≤ probability < 0.66
- High: 0.66 ≤ probability ≤ 1.0

### 5. Streamlit UI Component

**Responsibilities:**
- Provide user-friendly form for customer data entry
- Call Prediction API with form data
- Display prediction results with visual indicators
- Handle API errors gracefully
- Show model version and prediction timestamp

**UI Layout:**
```
┌─────────────────────────────────────┐
│  Customer Churn Prediction System   │
├─────────────────────────────────────┤
│  Customer Information Form          │
│  ├─ Demographics (gender, age, etc) │
│  ├─ Services (phone, internet, etc) │
│  └─ Account (contract, billing)     │
├─────────────────────────────────────┤
│  [Predict Churn] Button             │
├─────────────────────────────────────┤
│  Prediction Results                 │
│  ├─ Churn Probability: 67.3%        │
│  ├─ Risk Level: HIGH ⚠️             │
│  └─ Model Version: v3               │
└─────────────────────────────────────┘
```

**Implementation:**
```python
def render_input_form() -> dict
def call_prediction_api(customer_data: dict) -> dict
def display_prediction(response: dict) -> None
def display_error(error: str) -> None
```

### 6. Airflow Orchestration Component

**Responsibilities:**
- Schedule periodic retraining (weekly by default)
- Execute training pipeline with fresh data
- Compare new model performance vs Production model
- Promote model if ROC-AUC improvement ≥ 0.01
- Send alerts on training failures
- Log DAG execution history

**DAG Structure:**

```python
# retraining_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator

dag = DAG(
    'churn_model_retraining',
    schedule_interval='@weekly',
    default_args={
        'retries': 2,
        'retry_delay': timedelta(minutes=5)
    }
)

load_data_task = PythonOperator(task_id='load_data', ...)
preprocess_task = PythonOperator(task_id='preprocess', ...)
train_task = PythonOperator(task_id='train_model', ...)
evaluate_task = PythonOperator(task_id='evaluate', ...)
compare_task = PythonOperator(task_id='compare_models', ...)
promote_task = PythonOperator(task_id='promote_if_better', ...)
alert_task = PythonOperator(task_id='send_alert', trigger_rule='one_failed')

load_data_task >> preprocess_task >> train_task >> evaluate_task >> compare_task >> promote_task
[load_data_task, preprocess_task, train_task, evaluate_task] >> alert_task
```

**Model Promotion Logic:**
```python
def should_promote_model(new_roc_auc: float, current_roc_auc: float) -> bool:
    return new_roc_auc >= current_roc_auc + 0.01
```

## Data Models

### Customer Data Schema

```python
@dataclass
class CustomerData:
    # Demographics
    gender: str  # "Male", "Female"
    senior_citizen: int  # 0 or 1
    partner: str  # "Yes", "No"
    dependents: str  # "Yes", "No"
    
    # Account Information
    tenure: int  # months with company (0-72)
    contract: str  # "Month-to-month", "One year", "Two year"
    paperless_billing: str  # "Yes", "No"
    payment_method: str  # "Electronic check", "Mailed check", "Bank transfer", "Credit card"
    monthly_charges: float  # monthly bill amount
    total_charges: float  # total amount charged
    
    # Services
    phone_service: str  # "Yes", "No"
    multiple_lines: str  # "Yes", "No", "No phone service"
    internet_service: str  # "DSL", "Fiber optic", "No"
    online_security: str  # "Yes", "No", "No internet service"
    online_backup: str  # "Yes", "No", "No internet service"
    device_protection: str  # "Yes", "No", "No internet service"
    tech_support: str  # "Yes", "No", "No internet service"
    streaming_tv: str  # "Yes", "No", "No internet service"
    streaming_movies: str  # "Yes", "No", "No internet service"
    
    # Target (for training only)
    churn: Optional[str] = None  # "Yes", "No"
```

### Model Metadata Schema

```python
@dataclass
class ModelMetadata:
    model_name: str
    version: int
    stage: str  # "None", "Staging", "Production"
    created_at: datetime
    metrics: dict  # {"accuracy": 0.85, "f1": 0.72, "roc_auc": 0.88}
    hyperparameters: dict
    run_id: str
    artifact_uri: str
```

### Preprocessing Pipeline Schema

```python
@dataclass
class PreprocessingPipeline:
    numerical_features: List[str]
    categorical_features: List[str]
    scaler: StandardScaler
    encoders: Dict[str, LabelEncoder]
    feature_order: List[str]  # Ensures consistent feature ordering
```

### Training Metrics Schema

```python
@dataclass
class TrainingMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: np.ndarray
    feature_importance: Dict[str, float]
```

### Validation Result Schema

```python
@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
@dataclass
class ValidationError:
    field: str
    message: str
    value: Any
```

## Database Schema

The system uses MLflow's backend store for experiment tracking and model registry. The default configuration uses SQLite for development and can be upgraded to PostgreSQL for production.

**MLflow Tables (managed by MLflow):**
- `experiments`: Experiment metadata
- `runs`: Training run records
- `metrics`: Logged metrics per run
- `params`: Logged parameters per run
- `tags`: Run tags and metadata
- `model_versions`: Registered model versions
- `registered_models`: Model registry entries

**File System Storage:**
- `/data/raw/`: Raw CSV datasets
- `/data/processed/`: Preprocessed train/test splits
- `/models/`: Trained model artifacts
- `/models/pipelines/`: Preprocessing pipelines
- `/mlflow/artifacts/`: MLflow artifact storage
- `/logs/`: Application logs


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Data Loading Succeeds for Valid CSV Files

*For any* valid CSV file containing customer churn data with the required schema, the Data_Processor should successfully load it into a DataFrame without errors.

**Validates: Requirements 1.1, 1.4**

### Property 2: Data Persistence Round-Trip

*For any* loaded customer dataset, storing it to the /data directory and then loading it back should produce an equivalent dataset.

**Validates: Requirements 1.2**

### Property 3: Null Value Imputation Completeness

*For any* dataset containing null values, after preprocessing, the resulting dataset should contain no null values in any column.

**Validates: Requirements 2.1**

### Property 4: Categorical Encoding Produces Numerical Values

*For any* dataset containing categorical variables, after preprocessing, all categorical columns should be transformed into numerical representations.

**Validates: Requirements 2.2**

### Property 5: Numerical Standardization Properties

*For any* dataset containing numerical variables, after standardization, each numerical feature should have a mean approximately equal to 0 and standard deviation approximately equal to 1.

**Validates: Requirements 2.3**

### Property 6: Train-Test Split Ratio

*For any* dataset, splitting it with an 80-20 ratio should produce a training set containing 80% of the samples and a test set containing 20% of the samples (within rounding tolerance).

**Validates: Requirements 2.4**

### Property 7: Preprocessing Pipeline Round-Trip

*For any* preprocessing pipeline, saving it to disk and then loading it back should produce a pipeline that generates identical transformations on the same input data.

**Validates: Requirements 2.5**

### Property 8: Training Produces Fitted Model

*For any* valid preprocessed training dataset, the Training_Service should produce a fitted RandomForestClassifier that can generate predictions.

**Validates: Requirements 3.1**

### Property 9: Model Evaluation Returns Required Metrics

*For any* trained model and test dataset, evaluation should return all three required metrics: accuracy, F1 score, and ROC-AUC.

**Validates: Requirements 3.2**

### Property 10: Model Persistence After Training

*For any* completed training run, a model artifact file should exist in the /models directory.

**Validates: Requirements 3.3**

### Property 11: Hyperparameter Configuration Respected

*For any* set of hyperparameters provided to the Training_Service, the resulting trained model should have those exact hyperparameter values.

**Validates: Requirements 3.4**

### Property 12: Complete Experiment Logging

*For any* training run, MLflow should contain a logged experiment with all hyperparameters, all evaluation metrics, the model artifact, and timestamp/run identifier tags.

**Validates: Requirements 4.1, 4.2, 4.3, 4.4**

### Property 13: Model Registration with Metadata

*For any* trained model registered in the Model_Registry, it should have a unique version number and complete metadata including training date, metrics, and hyperparameters.

**Validates: Requirements 5.1, 5.2**

### Property 14: Model Stage Transitions

*For any* registered model, it should be possible to transition it through all valid stages: None → Staging → Production, and the stage should be correctly reflected in the registry.

**Validates: Requirements 5.3**

### Property 15: Performance-Based Model Promotion

*For any* two models where the new model's ROC-AUC exceeds the current production model's ROC-AUC by at least 0.01, the promotion logic should return true indicating the new model should be promoted.

**Validates: Requirements 5.4**

### Property 16: Valid Prediction Returns Probability

*For any* valid customer data submitted to the /predict endpoint, the response should contain a churn_probability value between 0.0 and 1.0.

**Validates: Requirements 6.2**

### Property 17: Invalid Input Returns 400 Status

*For any* invalid customer data (missing fields, wrong types, out-of-range values), the /predict endpoint should return a 400 status code with error details.

**Validates: Requirements 6.3**

### Property 18: Risk Label Classification

*For any* churn probability value, the assigned risk label should be "Low" for [0, 0.33), "Medium" for [0.33, 0.66), and "High" for [0.66, 1.0].

**Validates: Requirements 7.4**

### Property 19: DAG Triggers Training

*For any* execution of the Retraining_DAG, it should trigger the Training_Service and result in a new model being logged to MLflow.

**Validates: Requirements 8.2, 8.3**

### Property 20: Conditional Model Promotion in DAG

*For any* retraining run where the new model's ROC-AUC exceeds the production model's ROC-AUC by at least 0.01, the DAG should promote the new model to Production stage.

**Validates: Requirements 8.4**

### Property 21: Retraining Failure Alerts

*For any* retraining DAG execution that fails, an alert notification should be triggered.

**Validates: Requirements 8.5**

### Property 22: Model Loading Before Predictions

*For any* Prediction_API instance, prediction requests should fail or be rejected until model loading is complete.

**Validates: Requirements 10.3**

### Property 23: Configuration from Environment Variables

*For any* environment variable set for configuration (API ports, MLflow URI, model paths), the system should use that value instead of defaults.

**Validates: Requirements 11.2, 11.3**

### Property 24: Default Configuration Values

*For any* configuration parameter without a set environment variable, the system should use a sensible default value and function correctly.

**Validates: Requirements 11.4**

### Property 25: Error Logging Completeness

*For any* error that occurs in any component, a log entry should be created containing timestamp, component name, and stack trace.

**Validates: Requirements 14.1**

### Property 26: Structured Error Responses

*For any* error in the Prediction_API, the response should be a structured JSON object with an appropriate HTTP status code (4xx or 5xx).

**Validates: Requirements 14.2**

### Property 27: Training Progress Logging

*For any* training run, log messages should be generated that include progress information and metric updates.

**Validates: Requirements 14.3**

### Property 28: Data Quality Warning Logging

*For any* dataset with detected quality issues (e.g., high null percentage, outliers), the Data_Processor should log warning messages.

**Validates: Requirements 14.4**

### Property 29: Dual Logging Output

*For any* log message generated by the system, it should appear in both console output and log files.

**Validates: Requirements 14.5**

### Property 30: Comprehensive Input Validation

*For any* prediction request, the Prediction_API should validate that all required fields are present, numerical fields contain valid numbers, and categorical fields contain expected values, returning descriptive error messages for any validation failures.

**Validates: Requirements 15.1, 15.2, 15.3, 15.4**

### Property 31: Schema Validation Before Preprocessing

*For any* training dataset, the Data_Processor should validate the schema (required columns, data types) before beginning preprocessing, rejecting invalid schemas with descriptive errors.

**Validates: Requirements 15.5**

## Error Handling

### Error Categories

The system implements comprehensive error handling across four categories:

1. **Data Errors**: Missing files, invalid schemas, data quality issues
2. **Model Errors**: Training failures, evaluation errors, loading failures
3. **API Errors**: Invalid requests, validation failures, service unavailability
4. **Infrastructure Errors**: MLflow connection failures, file system errors, container issues

### Error Handling Strategy

**Data Processor Errors:**
```python
class DataProcessorError(Exception):
    """Base exception for data processing errors"""
    pass

class DataLoadError(DataProcessorError):
    """Raised when data cannot be loaded"""
    pass

class SchemaValidationError(DataProcessorError):
    """Raised when data schema is invalid"""
    pass

class PreprocessingError(DataProcessorError):
    """Raised when preprocessing fails"""
    pass
```

**Error Handling Pattern:**
```python
def load_data(file_path: str) -> pd.DataFrame:
    try:
        if not os.path.exists(file_path):
            raise DataLoadError(f"Dataset file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {len(df)} records from {file_path}")
        return df
        
    except pd.errors.EmptyDataError:
        raise DataLoadError(f"Dataset file is empty: {file_path}")
    except pd.errors.ParserError as e:
        raise DataLoadError(f"Failed to parse CSV file: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error loading data: {str(e)}", exc_info=True)
        raise DataProcessorError(f"Data loading failed: {str(e)}")
```

**Training Service Errors:**
```python
class TrainingError(Exception):
    """Base exception for training errors"""
    pass

class ModelEvaluationError(TrainingError):
    """Raised when model evaluation fails"""
    pass

class MLflowError(TrainingError):
    """Raised when MLflow operations fail"""
    pass
```

**API Error Responses:**
```python
class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str
    path: str

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="Validation Error",
            detail=str(exc),
            timestamp=datetime.utcnow().isoformat(),
            path=request.url.path
        ).dict()
    )

@app.exception_handler(ModelNotLoadedError)
async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
    return JSONResponse(
        status_code=503,
        content=ErrorResponse(
            error="Service Unavailable",
            detail="Model not loaded. Please try again later.",
            timestamp=datetime.utcnow().isoformat(),
            path=request.url.path
        ).dict()
    )
```

### Retry Logic

**MLflow Operations:**
```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(MLflowError)
)
def log_model_to_mlflow(model, run_id: str):
    mlflow.sklearn.log_model(model, "model")
```

**API Calls:**
```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type(requests.RequestException)
)
def call_prediction_api(data: dict) -> dict:
    response = requests.post(f"{API_URL}/predict", json=data)
    response.raise_for_status()
    return response.json()
```

### Logging Configuration

```python
import logging
from logging.handlers import RotatingFileHandler

def setup_logging(component_name: str):
    logger = logging.getLogger(component_name)
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        f'logs/{component_name}.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
```

### Graceful Degradation

**API Service:**
- If MLflow is unavailable, log warning and use local model file
- If model loading fails, return 503 with retry-after header
- If preprocessing pipeline missing, attempt to recreate from training data

**UI Application:**
- If API is unavailable, display user-friendly error message
- Implement timeout for API calls (5 seconds)
- Cache last successful prediction for display

**Retraining DAG:**
- If training fails, preserve current production model
- Send alert but don't block future DAG runs
- Log detailed error information for debugging

## Testing Strategy

### Dual Testing Approach

The system employs both unit testing and property-based testing for comprehensive coverage:

**Unit Tests:**
- Specific examples demonstrating correct behavior
- Edge cases (missing files, empty datasets, model not loaded)
- Integration points between components
- Error conditions and exception handling

**Property-Based Tests:**
- Universal properties that hold for all inputs
- Comprehensive input coverage through randomization
- Validation of invariants and transformations
- Round-trip properties for serialization/deserialization

### Property-Based Testing Configuration

**Framework:** Hypothesis (Python)

**Configuration:**
```python
from hypothesis import given, settings, strategies as st

# Minimum 100 iterations per property test
@settings(max_examples=100)
@given(
    data=st.data(),
    customer_data=customer_strategy()
)
def test_property(data, customer_data):
    # Test implementation
    pass
```

**Test Tagging:**
Each property-based test must reference its design document property:

```python
def test_data_loading_succeeds():
    """
    Feature: customer-churn-mlops-pipeline, Property 1: Data Loading Succeeds for Valid CSV Files
    
    For any valid CSV file containing customer churn data with the required schema,
    the Data_Processor should successfully load it into a DataFrame without errors.
    """
    # Test implementation
```

### Test Organization

```
tests/
├── unit/
│   ├── test_data_processor.py
│   ├── test_training_service.py
│   ├── test_prediction_api.py
│   └── test_ui_application.py
├── property/
│   ├── test_data_properties.py
│   ├── test_training_properties.py
│   ├── test_api_properties.py
│   └── test_validation_properties.py
├── integration/
│   ├── test_end_to_end.py
│   ├── test_mlflow_integration.py
│   └── test_airflow_dag.py
└── fixtures/
    ├── sample_data.csv
    ├── test_models.pkl
    └── strategies.py  # Hypothesis strategies
```

### Hypothesis Strategies

```python
import hypothesis.strategies as st
from hypothesis import assume

@st.composite
def customer_strategy(draw):
    """Generate random valid customer data"""
    return {
        "gender": draw(st.sampled_from(["Male", "Female"])),
        "senior_citizen": draw(st.integers(min_value=0, max_value=1)),
        "partner": draw(st.sampled_from(["Yes", "No"])),
        "dependents": draw(st.sampled_from(["Yes", "No"])),
        "tenure": draw(st.integers(min_value=0, max_value=72)),
        "phone_service": draw(st.sampled_from(["Yes", "No"])),
        "multiple_lines": draw(st.sampled_from(["Yes", "No", "No phone service"])),
        "internet_service": draw(st.sampled_from(["DSL", "Fiber optic", "No"])),
        "contract": draw(st.sampled_from(["Month-to-month", "One year", "Two year"])),
        "paperless_billing": draw(st.sampled_from(["Yes", "No"])),
        "payment_method": draw(st.sampled_from([
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])),
        "monthly_charges": draw(st.floats(min_value=0.0, max_value=200.0)),
        "total_charges": draw(st.floats(min_value=0.0, max_value=10000.0))
    }

@st.composite
def invalid_customer_strategy(draw):
    """Generate random invalid customer data for validation testing"""
    customer = draw(customer_strategy())
    
    # Randomly introduce validation errors
    error_type = draw(st.sampled_from(["missing_field", "invalid_type", "invalid_value"]))
    
    if error_type == "missing_field":
        field = draw(st.sampled_from(list(customer.keys())))
        del customer[field]
    elif error_type == "invalid_type":
        customer["tenure"] = "not a number"
    elif error_type == "invalid_value":
        customer["gender"] = "Invalid"
    
    return customer

@st.composite
def dataset_strategy(draw, min_rows=10, max_rows=1000):
    """Generate random customer dataset"""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    customers = [draw(customer_strategy()) for _ in range(n_rows)]
    
    # Add churn labels
    for customer in customers:
        customer["Churn"] = draw(st.sampled_from(["Yes", "No"]))
    
    return pd.DataFrame(customers)
```

### Example Property Tests

```python
from hypothesis import given, settings
import hypothesis.strategies as st

@settings(max_examples=100)
@given(dataset=dataset_strategy())
def test_property_1_data_loading_succeeds(dataset, tmp_path):
    """
    Feature: customer-churn-mlops-pipeline, Property 1: Data Loading Succeeds for Valid CSV Files
    """
    # Save dataset to temporary CSV
    csv_path = tmp_path / "test_data.csv"
    dataset.to_csv(csv_path, index=False)
    
    # Load data
    processor = DataProcessor()
    loaded_df = processor.load_data(str(csv_path))
    
    # Verify successful loading
    assert loaded_df is not None
    assert len(loaded_df) == len(dataset)
    assert list(loaded_df.columns) == list(dataset.columns)

@settings(max_examples=100)
@given(dataset=dataset_strategy())
def test_property_6_train_test_split_ratio(dataset):
    """
    Feature: customer-churn-mlops-pipeline, Property 6: Train-Test Split Ratio
    """
    processor = DataProcessor()
    X_train, X_test, y_train, y_test = processor.split_data(dataset, test_size=0.2)
    
    total_samples = len(dataset)
    train_samples = len(X_train)
    test_samples = len(X_test)
    
    # Verify 80-20 split (within rounding tolerance)
    assert abs(train_samples / total_samples - 0.8) < 0.01
    assert abs(test_samples / total_samples - 0.2) < 0.01
    assert train_samples + test_samples == total_samples

@settings(max_examples=100)
@given(probability=st.floats(min_value=0.0, max_value=1.0))
def test_property_18_risk_label_classification(probability):
    """
    Feature: customer-churn-mlops-pipeline, Property 18: Risk Label Classification
    """
    risk_label = classify_risk(probability)
    
    if 0.0 <= probability < 0.33:
        assert risk_label == "Low"
    elif 0.33 <= probability < 0.66:
        assert risk_label == "Medium"
    elif 0.66 <= probability <= 1.0:
        assert risk_label == "High"

@settings(max_examples=100)
@given(customer=invalid_customer_strategy())
def test_property_17_invalid_input_returns_400(customer, test_client):
    """
    Feature: customer-churn-mlops-pipeline, Property 17: Invalid Input Returns 400 Status
    """
    response = test_client.post("/predict", json=customer)
    
    assert response.status_code == 400
    error_data = response.json()
    assert "error" in error_data
    assert "detail" in error_data
```

### Unit Test Examples

```python
def test_missing_file_error():
    """Test edge case: missing dataset file"""
    processor = DataProcessor()
    
    with pytest.raises(DataLoadError) as exc_info:
        processor.load_data("nonexistent.csv")
    
    assert "not found" in str(exc_info.value).lower()

def test_health_endpoint_returns_200():
    """Test example: health endpoint when healthy"""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_health_endpoint_returns_503_when_model_not_loaded():
    """Test edge case: health endpoint when model not loaded"""
    # Simulate model not loaded state
    app.state.model = None
    
    response = client.get("/health")
    
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unhealthy"
    assert data["model_loaded"] is False
```

### Integration Tests

```python
def test_end_to_end_training_and_prediction():
    """Test complete workflow from training to prediction"""
    # 1. Load and preprocess data
    processor = DataProcessor()
    df = processor.load_data("data/telco_churn.csv")
    X_train, X_test, y_train, y_test = processor.preprocess(df)
    
    # 2. Train model
    trainer = TrainingService(mlflow_uri="http://localhost:5000")
    model = trainer.train(X_train, y_train)
    metrics = trainer.evaluate(model, X_test, y_test)
    
    # 3. Register model
    run_id = trainer.log_experiment(model, metrics, {"n_estimators": 100})
    trainer.register_model(run_id, "churn_model")
    
    # 4. Make prediction
    api = PredictionAPI()
    api.load_model("churn_model", stage="Production")
    
    sample_customer = {...}
    prediction = api.predict(sample_customer)
    
    assert 0.0 <= prediction["churn_probability"] <= 1.0
    assert prediction["risk_label"] in ["Low", "Medium", "High"]
```

### Test Coverage Goals

- Unit test coverage: ≥ 80%
- Property test coverage: All 31 properties implemented
- Integration test coverage: All component interactions
- Edge case coverage: All error conditions and boundary cases

### Continuous Testing

```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest hypothesis pytest-cov
      
      - name: Run unit tests
        run: pytest tests/unit/ -v --cov=src
      
      - name: Run property tests
        run: pytest tests/property/ -v
      
      - name: Run integration tests
        run: pytest tests/integration/ -v
      
      - name: Generate coverage report
        run: pytest --cov=src --cov-report=html
```


## Deployment Architecture

### Docker Containerization

**Service Containers:**

1. **Prediction API Container**
```dockerfile
# Dockerfile.api
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/api/ ./api/
COPY src/data_processing/ ./data_processing/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

2. **Streamlit UI Container**
```dockerfile
# Dockerfile.ui
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ui/ ./ui/

EXPOSE 8501

CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

3. **MLflow Server Container**
```dockerfile
# Dockerfile.mlflow
FROM python:3.9-slim

WORKDIR /app

RUN pip install mlflow psycopg2-binary

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
```

### Docker Compose Orchestration

```yaml
# docker-compose.yml
version: '3.8'

services:
  mlflow:
    build:
      context: .
      dockerfile: Dockerfile.mlflow
    container_name: mlflow-server
    ports:
      - "5000:5000"
    volumes:
      - mlflow-data:/mlflow
    networks:
      - churn-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  prediction-api:
    build:
      context: .
      dockerfile: Dockerfile.api
    container_name: prediction-api
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - MODEL_NAME=churn_model
      - MODEL_STAGE=Production
    depends_on:
      - mlflow
    networks:
      - churn-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  streamlit-ui:
    build:
      context: .
      dockerfile: Dockerfile.ui
    container_name: streamlit-ui
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://prediction-api:8000
    depends_on:
      - prediction-api
    networks:
      - churn-network

  airflow-webserver:
    image: apache/airflow:2.7.0
    container_name: airflow-webserver
    command: webserver
    ports:
      - "8080:8080"
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./data:/opt/airflow/data
      - ./src:/opt/airflow/src
    environment:
      - AIRFLOW__CORE__EXECUTOR=LocalExecutor
      - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow
      - AIRFLOW__CORE__LOAD_EXAMPLES=False
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - postgres
      - mlflow
    networks:
      - churn-network

  airflow-scheduler:
    image: apache/airflow:2.7.0
    container_name: airflow-scheduler
    command: scheduler
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./data:/opt/airflow/data
      - ./src:/opt/airflow/src
    environment:
      - AIRFLOW__CORE__EXECUTOR=LocalExecutor
      - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - postgres
      - mlflow
    networks:
      - churn-network

  postgres:
    image: postgres:13
    container_name: airflow-postgres
    environment:
      - POSTGRES_USER=airflow
      - POSTGRES_PASSWORD=airflow
      - POSTGRES_DB=airflow
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - churn-network

volumes:
  mlflow-data:
  postgres-data:

networks:
  churn-network:
    driver: bridge
```

### Environment Configuration

**.env.example:**
```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_ARTIFACT_ROOT=/mlflow/artifacts

# Model Configuration
MODEL_NAME=churn_model
MODEL_STAGE=Production
MODEL_PATH=./models

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# UI Configuration
UI_PORT=8501
API_URL=http://localhost:8000

# Training Configuration
N_ESTIMATORS=100
MAX_DEPTH=10
RANDOM_STATE=42
TEST_SIZE=0.2

# Data Configuration
DATA_PATH=./data/telco_churn.csv
PROCESSED_DATA_PATH=./data/processed

# Logging Configuration
LOG_LEVEL=INFO
LOG_PATH=./logs

# Airflow Configuration
AIRFLOW_HOME=/opt/airflow
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__CORE__LOAD_EXAMPLES=False
```

### Configuration Management

```python
# src/config.py
from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_artifact_root: str = "/mlflow/artifacts"
    
    # Model
    model_name: str = "churn_model"
    model_stage: str = "Production"
    model_path: str = "./models"
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    # Training
    n_estimators: int = 100
    max_depth: Optional[int] = None
    random_state: int = 42
    test_size: float = 0.2
    
    # Data
    data_path: str = "./data/telco_churn.csv"
    processed_data_path: str = "./data/processed"
    
    # Logging
    log_level: str = "INFO"
    log_path: str = "./logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

## Project Structure

```
customer-churn-mlops-pipeline/
├── .github/
│   └── workflows/
│       └── test.yml                 # CI/CD pipeline
├── data/
│   ├── raw/
│   │   └── telco_churn.csv         # Raw dataset
│   └── processed/                   # Preprocessed data
├── dags/
│   ├── retraining_dag.py           # Airflow retraining DAG
│   └── utils.py                     # DAG utilities
├── models/
│   ├── churn_model.pkl             # Trained models
│   └── pipelines/
│       └── preprocessing.pkl        # Preprocessing pipelines
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py                    # Configuration management
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── data_loader.py          # Data loading
│   │   ├── preprocessor.py         # Preprocessing logic
│   │   └── validator.py            # Data validation
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py              # Training service
│   │   ├── evaluator.py            # Model evaluation
│   │   └── mlflow_utils.py         # MLflow integration
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI application
│   │   ├── models.py               # Pydantic models
│   │   ├── predictor.py            # Prediction logic
│   │   └── validators.py           # Input validation
│   └── ui/
│       ├── __init__.py
│       ├── app.py                  # Streamlit application
│       └── components.py           # UI components
├── tests/
│   ├── unit/
│   │   ├── test_data_processor.py
│   │   ├── test_training_service.py
│   │   ├── test_prediction_api.py
│   │   └── test_validators.py
│   ├── property/
│   │   ├── test_data_properties.py
│   │   ├── test_training_properties.py
│   │   ├── test_api_properties.py
│   │   └── test_validation_properties.py
│   ├── integration/
│   │   ├── test_end_to_end.py
│   │   ├── test_mlflow_integration.py
│   │   └── test_airflow_dag.py
│   └── fixtures/
│       ├── sample_data.csv
│       └── strategies.py           # Hypothesis strategies
├── logs/                            # Application logs
├── mlflow/                          # MLflow artifacts
├── .env.example                     # Environment variables template
├── .gitignore
├── docker-compose.yml               # Docker orchestration
├── Dockerfile.api                   # API container
├── Dockerfile.ui                    # UI container
├── Dockerfile.mlflow                # MLflow container
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package setup
└── README.md                        # Project documentation
```

## API Specification

### Prediction Endpoint

**POST /predict**

Request:
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
  "streaming_movies": "No",
  "contract": "Month-to-month",
  "paperless_billing": "Yes",
  "payment_method": "Electronic check",
  "monthly_charges": 70.35,
  "total_charges": 844.20
}
```

Response (200 OK):
```json
{
  "churn_probability": 0.73,
  "risk_label": "High",
  "model_version": "3",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

Response (400 Bad Request):
```json
{
  "error": "Validation Error",
  "detail": "Field 'tenure' must be a non-negative integer",
  "timestamp": "2024-01-15T10:30:00Z",
  "path": "/predict"
}
```

Response (503 Service Unavailable):
```json
{
  "error": "Service Unavailable",
  "detail": "Model not loaded. Please try again later.",
  "timestamp": "2024-01-15T10:30:00Z",
  "path": "/predict"
}
```

### Health Check Endpoint

**GET /health**

Response (200 OK):
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "3",
  "model_stage": "Production",
  "uptime_seconds": 3600
}
```

Response (503 Service Unavailable):
```json
{
  "status": "unhealthy",
  "model_loaded": false,
  "model_version": null,
  "uptime_seconds": 120
}
```

### Model Info Endpoint

**GET /model-info**

Response (200 OK):
```json
{
  "model_name": "churn_model",
  "version": "3",
  "stage": "Production",
  "created_at": "2024-01-15T08:00:00Z",
  "metrics": {
    "accuracy": 0.85,
    "f1_score": 0.72,
    "roc_auc": 0.88
  },
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
  }
}
```

## Security Considerations

### API Security

1. **Input Validation**: All inputs validated against strict schemas
2. **Rate Limiting**: Implement rate limiting to prevent abuse
3. **CORS Configuration**: Configure CORS for production deployments
4. **Authentication**: Add API key authentication for production

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict(customer: CustomerInput):
    # Prediction logic
    pass
```

### Data Security

1. **PII Handling**: Customer data should be anonymized in logs
2. **Data Encryption**: Encrypt sensitive data at rest and in transit
3. **Access Control**: Implement role-based access for MLflow and Airflow

### Model Security

1. **Model Versioning**: Track all model versions for audit trail
2. **Model Validation**: Validate model performance before promotion
3. **Rollback Capability**: Maintain ability to rollback to previous models

## Performance Optimization

### API Performance

1. **Model Caching**: Load model once at startup, cache in memory
2. **Batch Predictions**: Support batch prediction endpoint for efficiency
3. **Async Processing**: Use async/await for I/O operations
4. **Connection Pooling**: Pool database connections for MLflow

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def load_model():
    """Cache model in memory"""
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

@app.post("/predict/batch")
async def predict_batch(customers: List[CustomerInput]):
    """Batch prediction endpoint"""
    model = load_model()
    predictions = model.predict_proba([c.dict() for c in customers])
    return [
        {"churn_probability": float(p[1]), "risk_label": classify_risk(p[1])}
        for p in predictions
    ]
```

### Training Performance

1. **Parallel Processing**: Use joblib for parallel model training
2. **Data Sampling**: Sample large datasets for faster experimentation
3. **Incremental Learning**: Consider incremental learning for large datasets

### Monitoring and Observability

1. **Metrics Collection**: Track prediction latency, throughput, error rates
2. **Model Drift Detection**: Monitor prediction distribution over time
3. **Performance Dashboards**: Create dashboards for system health

```python
from prometheus_client import Counter, Histogram

prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')

@app.post("/predict")
@prediction_latency.time()
async def predict(customer: CustomerInput):
    prediction_counter.inc()
    # Prediction logic
    pass
```

## Maintenance and Operations

### Model Retraining Strategy

1. **Scheduled Retraining**: Weekly automated retraining via Airflow
2. **Triggered Retraining**: Manual trigger when data drift detected
3. **A/B Testing**: Test new models against production before full rollout

### Monitoring Checklist

- [ ] API response times < 500ms
- [ ] Model prediction accuracy > 80%
- [ ] MLflow server availability > 99%
- [ ] Data quality checks passing
- [ ] No critical errors in logs
- [ ] Disk space sufficient for artifacts
- [ ] Container health checks passing

### Backup and Recovery

1. **Model Backups**: MLflow artifacts backed up daily
2. **Data Backups**: Raw and processed data backed up weekly
3. **Configuration Backups**: Environment configs version controlled
4. **Disaster Recovery**: Document recovery procedures

### Scaling Considerations

**Horizontal Scaling:**
- Deploy multiple API instances behind load balancer
- Use Redis for shared model cache
- Distribute Airflow workers for parallel DAG execution

**Vertical Scaling:**
- Increase container resources for training workloads
- Use GPU instances for large-scale model training
- Optimize memory usage for large datasets

## Future Enhancements

1. **Advanced Models**: Experiment with XGBoost, LightGBM, neural networks
2. **Feature Store**: Implement feature store for feature reuse
3. **Real-time Predictions**: Add streaming prediction capability
4. **Model Explainability**: Integrate SHAP or LIME for predictions
5. **Multi-model Serving**: Support multiple model versions simultaneously
6. **Automated Hyperparameter Tuning**: Integrate Optuna or Ray Tune
7. **Data Drift Detection**: Implement automated drift detection
8. **Model Performance Monitoring**: Track model performance in production

