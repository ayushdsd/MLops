"""
FastAPI application for Customer Churn Prediction API.

This module provides REST endpoints for churn prediction, health checks,
and model information. It integrates with the predictor, validators, and
models modules to provide a complete prediction service.
"""

import os
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.api.models import (
    CustomerInput,
    PredictionResponse,
    HealthResponse,
    ErrorResponse
)
from src.api.predictor import Predictor, ModelNotLoadedError, PredictorError
from src.api.validators import validate_customer_input
from src.logging_config import get_logger

# Configure logger
logger = get_logger('api')

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="REST API for predicting customer churn probability",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor: Predictor = None


@app.on_event("startup")
async def startup_event():
    """
    Initialize the predictor on application startup.
    
    Loads the model from MLflow Model Registry and preprocessing pipeline.
    """
    global predictor
    
    try:
        # Get configuration from environment variables
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        model_name = os.getenv("MODEL_NAME", "churn_model")
        model_stage = os.getenv("MODEL_STAGE", "Production")
        
        logger.info(f"Initializing predictor with MLflow URI: {mlflow_uri}")
        predictor = Predictor(mlflow_uri=mlflow_uri)
        
        # Load model from MLflow
        logger.info(f"Loading model '{model_name}' from stage '{model_stage}'")
        predictor.load_model(model_name=model_name, stage=model_stage)
        
        # Optionally load preprocessing pipeline
        pipeline_run_id = os.getenv("PREPROCESSING_RUN_ID")
        pipeline_path = os.getenv("PREPROCESSING_PIPELINE_PATH")
        
        if pipeline_run_id:
            logger.info(f"Loading preprocessing pipeline from run {pipeline_run_id}")
            predictor.load_preprocessing_pipeline(run_id=pipeline_run_id)
        elif pipeline_path:
            logger.info(f"Loading preprocessing pipeline from {pipeline_path}")
            predictor.load_preprocessing_pipeline(pipeline_path=pipeline_path)
        else:
            logger.warning("No preprocessing pipeline configured")
        
        logger.info("Predictor initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {str(e)}", exc_info=True)
        # Don't fail startup - allow health endpoint to report unhealthy state
        predictor = None


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log all incoming requests and responses.
    
    Logs request method, path, and response status code.
    """
    logger.info(f"Request: {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        logger.info(f"Response: {request.method} {request.url.path} - Status {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request failed: {request.method} {request.url.path} - {str(e)}", exc_info=True)
        raise


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(customer: CustomerInput, request: Request) -> PredictionResponse:
    """
    Generate churn prediction for a customer.
    
    This endpoint accepts customer data, validates it, applies preprocessing,
    and returns the churn probability and risk classification.
    
    Args:
        customer: Customer data with all required attributes
        request: FastAPI request object
        
    Returns:
        PredictionResponse with churn probability, risk label, model version, and timestamp
        
    Raises:
        HTTPException 400: If input validation fails
        HTTPException 503: If model is not loaded
        HTTPException 500: If prediction fails
        
    Validates: Requirements 6.1, 6.2, 6.3, 10.1, 10.2, 10.3, 14.1, 14.2
    """
    try:
        logger.info("Received prediction request")
        
        # Check if predictor is initialized
        if predictor is None:
            logger.error("Predictor not initialized")
            raise ModelNotLoadedError("Prediction service not initialized")
        
        # Convert Pydantic model to dict for validation
        customer_data = customer.model_dump()
        
        # Validate input using validators module
        validation_result = validate_customer_input(customer_data)
        if not validation_result.is_valid:
            error_messages = validation_result.get_error_messages()
            logger.warning(f"Input validation failed: {error_messages}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ErrorResponse(
                    error="Validation Error",
                    detail=validation_result.get_error_summary(),
                    timestamp=datetime.utcnow().isoformat(),
                    path=request.url.path
                ).model_dump()
            )
        
        # Generate prediction
        logger.info("Generating prediction")
        result = predictor.predict(customer_data)
        
        # Get model info
        model_info = predictor.get_model_info()
        model_version = model_info.get("model_version", "unknown")
        
        # Create response
        response = PredictionResponse(
            churn_probability=result.churn_probability,
            risk_label=result.risk_label,
            model_version=model_version,
            timestamp=datetime.utcnow().isoformat()
        )
        
        logger.info(f"Prediction successful: probability={result.churn_probability:.4f}, risk={result.risk_label}")
        return response
        
    except ModelNotLoadedError as e:
        logger.error(f"Model not loaded: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                error="Service Unavailable",
                detail="Model not loaded. Please try again later.",
                timestamp=datetime.utcnow().isoformat(),
                path=request.url.path
            ).model_dump()
        )
    
    except PredictorError as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="Prediction Error",
                detail=str(e),
                timestamp=datetime.utcnow().isoformat(),
                path=request.url.path
            ).model_dump()
        )
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="Internal Server Error",
                detail="An unexpected error occurred",
                timestamp=datetime.utcnow().isoformat(),
                path=request.url.path
            ).model_dump()
        )


@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns service health status and model loading state.
    Returns 200 when healthy (model loaded), 503 when unhealthy (model not loaded).
    
    Returns:
        HealthResponse with status, model_loaded flag, and model version
        
    Validates: Requirements 6.5, 10.1, 10.2
    """
    try:
        if predictor is None:
            logger.warning("Health check: predictor not initialized")
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=HealthResponse(
                    status="unhealthy",
                    model_loaded=False,
                    model_version=None
                ).model_dump()
            )
        
        is_ready = predictor.is_ready()
        model_info = predictor.get_model_info()
        model_version = model_info.get("model_version")
        
        if is_ready:
            logger.debug("Health check: healthy")
            return HealthResponse(
                status="healthy",
                model_loaded=True,
                model_version=model_version
            )
        else:
            logger.warning("Health check: model not loaded")
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=HealthResponse(
                    status="unhealthy",
                    model_loaded=False,
                    model_version=None
                ).model_dump()
            )
    
    except Exception as e:
        logger.error(f"Health check error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=HealthResponse(
                status="unhealthy",
                model_loaded=False,
                model_version=None
            ).model_dump()
        )


@app.get("/model-info")
async def model_info() -> Dict[str, Any]:
    """
    Get information about the loaded model.
    
    Returns model metadata including name, version, and loading status.
    
    Returns:
        Dictionary with model information
        
    Validates: Requirements 6.5
    """
    try:
        if predictor is None:
            logger.warning("Model info requested but predictor not initialized")
            return {
                "model_name": None,
                "model_version": None,
                "model_loaded": False,
                "preprocessing_pipeline_loaded": False
            }
        
        info = predictor.get_model_info()
        logger.debug(f"Model info: {info}")
        return info
    
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}", exc_info=True)
        return {
            "model_name": None,
            "model_version": None,
            "model_loaded": False,
            "preprocessing_pipeline_loaded": False,
            "error": str(e)
        }


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle Pydantic validation errors.
    
    Returns structured error response with validation details.
    
    Validates: Requirements 14.2
    """
    logger.warning(f"Validation error: {exc.errors()}")
    
    # Format validation errors
    error_details = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"])
        message = error["msg"]
        error_details.append(f"{field}: {message}")
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="Validation Error",
            detail="; ".join(error_details),
            timestamp=datetime.utcnow().isoformat(),
            path=request.url.path
        ).model_dump()
    )


@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    """
    Handle Pydantic ValidationError exceptions.
    
    Returns structured error response with validation details.
    
    Validates: Requirements 14.2
    """
    logger.warning(f"Pydantic validation error: {exc.errors()}")
    
    # Format validation errors
    error_details = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"])
        message = error["msg"]
        error_details.append(f"{field}: {message}")
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="Validation Error",
            detail="; ".join(error_details),
            timestamp=datetime.utcnow().isoformat(),
            path=request.url.path
        ).model_dump()
    )


@app.exception_handler(ModelNotLoadedError)
async def model_not_loaded_exception_handler(request: Request, exc: ModelNotLoadedError):
    """
    Handle ModelNotLoadedError exceptions.
    
    Returns 503 Service Unavailable when model is not loaded.
    
    Validates: Requirements 10.2, 14.2
    """
    logger.error(f"Model not loaded error: {str(exc)}")
    
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            error="Service Unavailable",
            detail="Model not loaded. Please try again later.",
            timestamp=datetime.utcnow().isoformat(),
            path=request.url.path
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle all other exceptions.
    
    Returns 500 Internal Server Error for unexpected exceptions.
    
    Validates: Requirements 14.1, 14.2
    """
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred",
            timestamp=datetime.utcnow().isoformat(),
            path=request.url.path
        ).model_dump()
    )


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv("API_PORT", "8000"))
    
    logger.info(f"Starting FastAPI server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
