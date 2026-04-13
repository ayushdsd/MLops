# Implementation Plan: Customer Churn MLOps Pipeline

## Overview

This implementation plan breaks down the Customer Churn MLOps Pipeline into discrete, actionable coding tasks. The pipeline is a production-ready ML system with data processing, model training, MLflow experiment tracking, FastAPI prediction service, Streamlit UI, and Airflow orchestration. Each task builds incrementally, ensuring integration at every step.

## Tasks

- [x] 1. Set up project structure and dependencies
  - Create directory structure: `/data`, `/src`, `/models`, `/dags`, `/tests`, `/logs`
  - Create subdirectories: `/src/data_processing`, `/src/training`, `/src/api`, `/src/ui`
  - Create `requirements.txt` with all dependencies (pandas, scikit-learn, mlflow, fastapi, streamlit, airflow, hypothesis, pytest)
  - Create `.env.example` file with configuration variables
  - Create `src/config.py` for centralized configuration management using pydantic BaseSettings
  - _Requirements: 11.1, 11.4, 13.1, 13.2, 13.5_

- [x] 2. Implement data processing layer
  - [x] 2.1 Create data loader module
    - Implement `DataProcessor.load_data()` to read CSV files and return DataFrame
    - Add error handling for missing files, empty files, and parse errors
    - Add logging for successful loads and errors
    - _Requirements: 1.1, 1.3, 14.1_
  
  - [x] 2.2 Create data validator module
    - Implement `DataProcessor.validate_schema()` to check required columns and data types
    - Validate numerical ranges (tenure ≥ 0, monthly_charges > 0, total_charges ≥ 0)
    - Validate categorical values against expected sets
    - Return ValidationResult with errors and warnings
    - _Requirements: 1.4, 15.5, 14.4_
  
  - [x] 2.3 Implement preprocessing pipeline
    - Implement null value imputation for numerical and categorical features
    - Implement categorical encoding (label encoding and one-hot encoding)
    - Implement numerical feature standardization using StandardScaler
    - Implement train-test split with 80-20 ratio
    - _Requirements: 2.1, 2.2, 2.3, 2.4_
  
  - [x] 2.4 Implement pipeline persistence
    - Implement `DataProcessor.save_pipeline()` to serialize sklearn Pipeline
    - Implement `DataProcessor.load_pipeline()` to deserialize pipeline
    - Save pipelines to `/models/pipelines/` directory
    - _Requirements: 2.5, 1.2_
  
  - [x] 2.5 Write property test for data loading
    - **Property 1: Data Loading Succeeds for Valid CSV Files**
    - **Validates: Requirements 1.1, 1.4**
  
  - [x] 2.6 Write property test for data persistence
    - **Property 2: Data Persistence Round-Trip**
    - **Validates: Requirements 1.2**
  
  - [x] 2.7 Write property test for null value imputation
    - **Property 3: Null Value Imputation Completeness**
    - **Validates: Requirements 2.1**
  
  - [x] 2.8 Write property test for categorical encoding
    - **Property 4: Categorical Encoding Produces Numerical Values**
    - **Validates: Requirements 2.2**
  
  - [x] 2.9 Write property test for numerical standardization
    - **Property 5: Numerical Standardization Properties**
    - **Validates: Requirements 2.3**
  
  - [x] 2.10 Write property test for train-test split
    - **Property 6: Train-Test Split Ratio**
    - **Validates: Requirements 2.4**
  
  - [x] 2.11 Write property test for pipeline round-trip
    - **Property 7: Preprocessing Pipeline Round-Trip**
    - **Validates: Requirements 2.5**
  
  - [x] 2.12 Write property test for schema validation
    - **Property 31: Schema Validation Before Preprocessing**
    - **Validates: Requirements 15.5**

- [x] 3. Checkpoint - Verify data processing layer
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement training service with MLflow integration
  - [x] 4.1 Create training service module
    - Implement `TrainingService.__init__()` to configure MLflow tracking URI
    - Implement `TrainingService.train()` to train RandomForestClassifier
    - Support configurable hyperparameters (n_estimators, max_depth, random_state)
    - Add progress logging during training
    - _Requirements: 3.1, 3.4, 14.3_
  
  - [x] 4.2 Create model evaluator module
    - Implement `TrainingService.evaluate()` to compute accuracy, F1, and ROC-AUC
    - Generate confusion matrix and feature importance
    - Return TrainingMetrics dataclass with all metrics
    - _Requirements: 3.2_
  
  - [x] 4.3 Implement MLflow experiment logging
    - Implement `TrainingService.log_experiment()` to log hyperparameters to MLflow
    - Log all evaluation metrics (accuracy, F1, ROC-AUC)
    - Log model artifact and preprocessing pipeline
    - Log feature importance plots and confusion matrix
    - Add timestamp and run identifier tags
    - _Requirements: 4.1, 4.2, 4.3, 4.4_
  
  - [x] 4.4 Implement model registry operations
    - Implement `TrainingService.register_model()` to register model with version
    - Store model metadata (training date, metrics, hyperparameters)
    - Implement `TrainingService.promote_to_production()` for stage transitions
    - Implement promotion logic: promote if new ROC-AUC ≥ current + 0.01
    - _Requirements: 5.1, 5.2, 5.3, 5.4_
  
  - [x] 4.5 Implement model persistence
    - Save trained models to `/models/` directory
    - Use consistent naming convention with version numbers
    - _Requirements: 3.3_
  
  - [x] 4.6 Write property test for training produces fitted model
    - **Property 8: Training Produces Fitted Model**
    - **Validates: Requirements 3.1**
  
  - [x] 4.7 Write property test for evaluation metrics
    - **Property 9: Model Evaluation Returns Required Metrics**
    - **Validates: Requirements 3.2**
  
  - [x] 4.8 Write property test for model persistence
    - **Property 10: Model Persistence After Training**
    - **Validates: Requirements 3.3**
  
  - [x] 4.9 Write property test for hyperparameter configuration
    - **Property 11: Hyperparameter Configuration Respected**
    - **Validates: Requirements 3.4**
  
  - [x] 4.10 Write property test for complete experiment logging
    - **Property 12: Complete Experiment Logging**
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4**
  
  - [x] 4.11 Write property test for model registration
    - **Property 13: Model Registration with Metadata**
    - **Validates: Requirements 5.1, 5.2**
  
  - [x] 4.12 Write property test for model stage transitions
    - **Property 14: Model Stage Transitions**
    - **Validates: Requirements 5.3**
  
  - [x] 4.13 Write property test for performance-based promotion
    - **Property 15: Performance-Based Model Promotion**
    - **Validates: Requirements 5.4**

- [x] 5. Checkpoint - Verify training service
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement FastAPI prediction service
  - [x] 6.1 Create Pydantic models for API
    - Create `CustomerInput` model with all 19 customer attributes
    - Create `PredictionResponse` model with churn_probability, risk_label, model_version, timestamp
    - Create `HealthResponse` model with status, model_loaded, model_version
    - Create `ErrorResponse` model with error, detail, timestamp, path
    - _Requirements: 6.1_
  
  - [x] 6.2 Create input validators
    - Implement validation for required fields presence
    - Implement validation for numerical field types and ranges
    - Implement validation for categorical field values
    - Return descriptive error messages for validation failures
    - _Requirements: 15.1, 15.2, 15.3, 15.4_
  
  - [x] 6.3 Implement predictor module
    - Implement model loading from MLflow Model Registry on startup
    - Implement preprocessing pipeline loading
    - Implement `predict()` function to apply pipeline and generate predictions
    - Implement `classify_risk()` function for risk label assignment (Low: 0-0.33, Medium: 0.33-0.66, High: 0.66-1.0)
    - Add error handling for model not loaded state
    - _Requirements: 6.4, 7.4_
  
  - [x] 6.4 Create FastAPI application
    - Implement POST `/predict` endpoint with input validation
    - Implement GET `/health` endpoint returning 200 when healthy, 503 when model not loaded
    - Implement GET `/model-info` endpoint returning model metadata
    - Add exception handlers for ValidationError, ModelNotLoadedError
    - Add CORS middleware configuration
    - Add logging for all requests and errors
    - _Requirements: 6.1, 6.2, 6.3, 6.5, 10.1, 10.2, 10.3, 14.1, 14.2_
  
  - [x] 6.5 Write property test for valid prediction returns probability
    - **Property 16: Valid Prediction Returns Probability**
    - **Validates: Requirements 6.2**
  
  - [x] 6.6 Write property test for invalid input returns 400
    - **Property 17: Invalid Input Returns 400 Status**
    - **Validates: Requirements 6.3**
  
  - [x] 6.7 Write property test for risk label classification
    - **Property 18: Risk Label Classification**
    - **Validates: Requirements 7.4**
  
  - [x] 6.8 Write property test for model loading before predictions
    - **Property 22: Model Loading Before Predictions**
    - **Validates: Requirements 10.3**
  
  - [x] 6.9 Write property test for comprehensive input validation
    - **Property 30: Comprehensive Input Validation**
    - **Validates: Requirements 15.1, 15.2, 15.3, 15.4**
  
  - [x] 6.10 Write unit tests for API endpoints
    - Test health endpoint returns 200 when healthy
    - Test health endpoint returns 503 when model not loaded
    - Test predict endpoint with valid input
    - Test predict endpoint with missing fields
    - Test predict endpoint with invalid types
    - Test model-info endpoint

- [x] 7. Checkpoint - Verify prediction API
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement Streamlit UI application
  - [x] 8.1 Create UI components module
    - Implement `render_input_form()` to create form with all customer attributes
    - Group form fields by category (Demographics, Services, Account)
    - Use appropriate Streamlit widgets (selectbox for categorical, number_input for numerical)
    - _Requirements: 7.1_
  
  - [x] 8.2 Implement API integration
    - Implement `call_prediction_api()` to POST data to `/predict` endpoint
    - Add timeout configuration (5 seconds)
    - Add error handling for API unavailability
    - Add retry logic with exponential backoff
    - _Requirements: 7.2_
  
  - [x] 8.3 Create result display components
    - Implement `display_prediction()` to show churn probability as percentage
    - Display risk label with color coding (green=Low, yellow=Medium, red=High)
    - Display model version and prediction timestamp
    - Add visual indicators (progress bar, emoji icons)
    - _Requirements: 7.3, 7.4_
  
  - [x] 8.4 Create error display components
    - Implement `display_error()` to show user-friendly error messages
    - Handle API unavailability gracefully
    - Handle validation errors with field-specific messages
    - _Requirements: 7.5_
  
  - [x] 8.5 Create main Streamlit application
    - Wire together form, API call, and result display
    - Add application title and description
    - Add "Predict Churn" button
    - Configure page layout and styling
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 9. Checkpoint - Verify Streamlit UI
  - Ensure all tests pass, ask the user if questions arise.

- [~] 10. Implement Airflow retraining DAG
  - [~] 10.1 Create DAG utility functions
    - Implement `load_data_task()` function to load fresh training data
    - Implement `preprocess_task()` function to preprocess data
    - Implement `train_task()` function to train new model
    - Implement `evaluate_task()` function to evaluate new model
    - _Requirements: 8.2_
  
  - [~] 10.2 Implement model comparison logic
    - Implement `compare_models()` function to compare new vs production ROC-AUC
    - Implement `should_promote_model()` function with 0.01 threshold
    - _Requirements: 8.4_
  
  - [~] 10.3 Implement promotion logic
    - Implement `promote_if_better()` function to promote model to Production
    - Add logging for promotion decisions
    - _Requirements: 8.4_
  
  - [~] 10.4 Implement alert notifications
    - Implement `send_alert()` function for training failures
    - Configure trigger_rule='one_failed' for alert task
    - Add email or Slack notification integration
    - _Requirements: 8.5_
  
  - [~] 10.5 Create retraining DAG
    - Create `retraining_dag.py` with weekly schedule
    - Wire tasks: load_data >> preprocess >> train >> evaluate >> compare >> promote
    - Add alert task with failure trigger
    - Configure retry logic (2 retries, 5 minute delay)
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_
  
  - [~] 10.6 Write property test for DAG triggers training
    - **Property 19: DAG Triggers Training**
    - **Validates: Requirements 8.2, 8.3**
  
  - [~] 10.7 Write property test for conditional model promotion
    - **Property 20: Conditional Model Promotion in DAG**
    - **Validates: Requirements 8.4**
  
  - [~] 10.8 Write property test for retraining failure alerts
    - **Property 21: Retraining Failure Alerts**
    - **Validates: Requirements 8.5**

- [~] 11. Checkpoint - Verify Airflow DAG
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Implement Docker containerization
  - [x] 12.1 Create Dockerfile for Prediction API
    - Use python:3.9-slim base image
    - Copy requirements and install dependencies
    - Copy source code and models
    - Expose port 8000
    - Set CMD to run uvicorn server
    - _Requirements: 9.1_
  
  - [x] 12.2 Create Dockerfile for Streamlit UI
    - Use python:3.9-slim base image
    - Copy requirements and install dependencies
    - Copy UI source code
    - Expose port 8501
    - Set CMD to run streamlit app
    - _Requirements: 9.2_
  
  - [x] 12.3 Create Dockerfile for MLflow server
    - Use python:3.9-slim base image
    - Install mlflow and psycopg2-binary
    - Expose port 5000
    - Set CMD to run mlflow server with backend store and artifact root
    - _Requirements: 9.3_
  
  - [x] 12.4 Create docker-compose.yml
    - Define services: mlflow, prediction-api, streamlit-ui, airflow-webserver, airflow-scheduler, postgres
    - Configure port mappings for all services
    - Configure volume mounts for data, models, logs, mlflow artifacts
    - Configure environment variables for all services
    - Configure service dependencies and health checks
    - Define networks and volumes
    - _Requirements: 9.3, 9.4, 9.5, 9.6_

- [x] 13. Implement logging and error handling
  - [x] 13.1 Create logging configuration module
    - Implement `setup_logging()` function with console and file handlers
    - Configure rotating file handler (10MB max, 5 backups)
    - Configure log format with timestamp, component, level, function, line number
    - Create separate log files per component
    - _Requirements: 14.5_
  
  - [x] 13.2 Add logging to data processor
    - Log successful data loads with record count
    - Log warnings for data quality issues
    - Log errors with stack traces
    - _Requirements: 14.1, 14.4_
  
  - [x] 13.3 Add logging to training service
    - Log training progress and metric updates
    - Log experiment logging to MLflow
    - Log model registration and promotion
    - Log errors with stack traces
    - _Requirements: 14.1, 14.3_
  
  - [x] 13.4 Add logging to prediction API
    - Log all incoming requests
    - Log prediction results
    - Log validation errors
    - Log errors with stack traces
    - _Requirements: 14.1, 14.2_
  
  - [x] 13.5 Implement custom exception classes
    - Create DataProcessorError, DataLoadError, SchemaValidationError, PreprocessingError
    - Create TrainingError, ModelEvaluationError, MLflowError
    - Create ModelNotLoadedError for API
    - _Requirements: 14.1, 14.2_
  
  - [x] 13.6 Write property test for error logging completeness
    - **Property 25: Error Logging Completeness**
    - **Validates: Requirements 14.1**
  
  - [x] 13.7 Write property test for structured error responses
    - **Property 26: Structured Error Responses**
    - **Validates: Requirements 14.2**
  
  - [x] 13.8 Write property test for training progress logging
    - **Property 27: Training Progress Logging**
    - **Validates: Requirements 14.3**
  
  - [x] 13.9 Write property test for data quality warning logging
    - **Property 28: Data Quality Warning Logging**
    - **Validates: Requirements 14.4**
  
  - [x] 13.10 Write property test for dual logging output
    - **Property 29: Dual Logging Output**
    - **Validates: Requirements 14.5**

- [x] 14. Implement configuration management
  - [x] 14.1 Write property test for configuration from environment variables
    - **Property 23: Configuration from Environment Variables**
    - **Validates: Requirements 11.2, 11.3**
  
  - [x] 14.2 Write property test for default configuration values
    - **Property 24: Default Configuration Values**
    - **Validates: Requirements 11.4**

- [x] 15. Create Hypothesis test strategies
  - Create `tests/fixtures/strategies.py` file
  - Implement `customer_strategy()` to generate valid customer data
  - Implement `invalid_customer_strategy()` to generate invalid customer data
  - Implement `dataset_strategy()` to generate customer datasets
  - Configure Hypothesis settings (max_examples=100)

- [~] 16. Create integration tests
  - [~] 16.1 Write end-to-end integration test
    - Test complete workflow: load data → preprocess → train → register → predict
    - Verify all components work together
    - _Requirements: All_
  
  - [~] 16.2 Write MLflow integration test
    - Test experiment logging, model registration, stage transitions
    - Verify MLflow server connectivity
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3_
  
  - [~] 16.3 Write Airflow DAG integration test
    - Test DAG execution with all tasks
    - Verify model promotion logic
    - Verify alert notifications
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 17. Create project documentation
  - [x] 17.1 Create README.md
    - Add project overview and architecture diagram
    - Add prerequisites and dependencies
    - Add step-by-step setup instructions
    - Add instructions for running with Docker Compose
    - Add API endpoint documentation with examples
    - Add directory structure explanation
    - Add troubleshooting section
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_
  
  - [x] 17.2 Add inline code comments
    - Add docstrings to all classes and functions
    - Add comments explaining complex logic
    - Add type hints to all function signatures
    - _Requirements: 12.6_
  
  - [x] 17.3 Create sample dataset
    - Add sample `telco_churn.csv` to `/data/raw/` directory
    - Ensure dataset has all required columns
    - _Requirements: 1.1, 1.4_

- [~] 18. Final checkpoint - End-to-end verification
  - Run all unit tests and verify 80%+ coverage
  - Run all property tests and verify all 31 properties pass
  - Run all integration tests
  - Start all services with docker-compose and verify health
  - Test complete workflow: train model → make prediction via API → view in UI
  - Verify Airflow DAG can be triggered manually
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at key milestones
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- Integration tests validate component interactions
- The implementation uses Python with FastAPI, Streamlit, MLflow, Airflow, and Docker
- All code should include comprehensive error handling and logging
- Configuration should be externalized via environment variables
- The system should be production-ready with health checks and monitoring
