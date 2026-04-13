"""
Predictor module for the Customer Churn MLOps Pipeline.

This module provides the Predictor class that handles model loading from MLflow,
preprocessing pipeline application, and churn prediction generation.
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Import centralized logging configuration
from ..logging_config import get_logger

# Configure logger
logger = get_logger('prediction_api')


class PredictorError(Exception):
    """Base exception for predictor errors."""
    pass


class ModelNotLoadedError(PredictorError):
    """Raised when attempting to predict without a loaded model."""
    pass


class PreprocessingError(PredictorError):
    """Raised when preprocessing fails during prediction."""
    pass


@dataclass
class PredictionResult:
    """
    Result of a churn prediction.
    
    Attributes:
        churn_probability: Probability of customer churn (0-1)
        risk_label: Risk classification (Low, Medium, or High)
    """
    churn_probability: float
    risk_label: str


class Predictor:
    """
    Predictor for customer churn with MLflow model loading and preprocessing.
    
    This class manages:
    - Loading models from MLflow Model Registry
    - Loading preprocessing pipelines
    - Applying preprocessing to input data
    - Generating churn predictions
    - Classifying risk levels
    
    The predictor ensures models and pipelines are loaded before accepting
    prediction requests and provides comprehensive error handling.
    """
    
    def __init__(self, mlflow_uri: str):
        """
        Initialize the Predictor with MLflow configuration.
        
        Args:
            mlflow_uri: MLflow tracking server URI (e.g., "http://localhost:5000")
            
        Examples:
            >>> predictor = Predictor(mlflow_uri="http://localhost:5000")
            >>> predictor.load_model(model_name="churn_model", stage="Production")
            >>> result = predictor.predict(customer_data)
        """
        self.mlflow_uri = mlflow_uri
        self.model = None
        self.preprocessing_pipeline = None
        self.model_version = None
        self.model_name = None
        
        # Configure MLflow tracking URI
        mlflow.set_tracking_uri(self.mlflow_uri)
        logger.info(f"Predictor initialized with MLflow URI: {self.mlflow_uri}")
    
    def load_model(self, model_name: str, stage: str = "Production") -> None:
        """
        Load a model from MLflow Model Registry.
        
        This method loads a model from the MLflow Model Registry based on the
        model name and stage (None, Staging, or Production). The model is cached
        in memory for efficient prediction serving.
        
        Args:
            model_name: Name of the registered model in MLflow
            stage: Model stage to load (default: "Production")
            
        Raises:
            PredictorError: If model loading fails
            
        Validates: Requirements 6.4
            
        Examples:
            >>> predictor = Predictor(mlflow_uri="http://localhost:5000")
            >>> predictor.load_model(model_name="churn_model", stage="Production")
        """
        try:
            logger.info(f"Loading model '{model_name}' from stage '{stage}'")
            
            # Construct model URI for the specified stage
            model_uri = f"models:/{model_name}/{stage}"
            
            # Load the model from MLflow
            self.model = mlflow.sklearn.load_model(model_uri)
            self.model_name = model_name
            
            # Get model version information
            client = mlflow.tracking.MlflowClient()
            model_versions = client.get_latest_versions(model_name, stages=[stage])
            
            if model_versions:
                self.model_version = model_versions[0].version
                logger.info(f"Loaded model version {self.model_version}")
            else:
                self.model_version = "unknown"
                logger.warning(f"Could not determine model version for {model_name}/{stage}")
            
            logger.info(f"Successfully loaded model '{model_name}' (version {self.model_version}) from stage '{stage}'")
            
        except Exception as e:
            error_msg = f"Failed to load model '{model_name}' from stage '{stage}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise PredictorError(error_msg) from e
    
    def load_preprocessing_pipeline(self, run_id: Optional[str] = None, pipeline_path: Optional[str] = None) -> None:
        """
        Load preprocessing pipeline from MLflow or local file.
        
        This method loads the preprocessing pipeline that was saved during training.
        The pipeline can be loaded either from an MLflow run artifact or from a
        local file path.
        
        Args:
            run_id: MLflow run ID containing the preprocessing pipeline artifact
            pipeline_path: Local file path to the preprocessing pipeline
            
        Raises:
            PredictorError: If pipeline loading fails
            ValueError: If neither run_id nor pipeline_path is provided
            
        Validates: Requirements 6.4
            
        Examples:
            >>> predictor = Predictor(mlflow_uri="http://localhost:5000")
            >>> # Load from MLflow run
            >>> predictor.load_preprocessing_pipeline(run_id="abc123")
            >>> # Or load from local file
            >>> predictor.load_preprocessing_pipeline(pipeline_path="models/pipelines/preprocessing.pkl")
        """
        try:
            if run_id is not None:
                logger.info(f"Loading preprocessing pipeline from MLflow run {run_id}")
                
                # Construct artifact URI
                artifact_uri = f"runs:/{run_id}/preprocessing_pipeline"
                
                # Load the pipeline from MLflow
                self.preprocessing_pipeline = mlflow.sklearn.load_model(artifact_uri)
                logger.info(f"Successfully loaded preprocessing pipeline from run {run_id}")
                
            elif pipeline_path is not None:
                logger.info(f"Loading preprocessing pipeline from local file {pipeline_path}")
                
                # Load from local file using joblib
                import joblib
                self.preprocessing_pipeline = joblib.load(pipeline_path)
                logger.info(f"Successfully loaded preprocessing pipeline from {pipeline_path}")
                
            else:
                raise ValueError("Either run_id or pipeline_path must be provided")
                
        except ValueError:
            raise
        except Exception as e:
            error_msg = f"Failed to load preprocessing pipeline: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise PredictorError(error_msg) from e
    
    def _check_model_loaded(self) -> None:
        """
        Check if model is loaded, raise error if not.
        
        Raises:
            ModelNotLoadedError: If model is not loaded
            
        Validates: Requirements 6.4
        """
        if self.model is None:
            error_msg = "Model not loaded. Call load_model() before making predictions."
            logger.error(error_msg)
            raise ModelNotLoadedError(error_msg)
    
    def _preprocess_input(self, customer_data: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess customer input data for prediction.
        
        This method applies the preprocessing pipeline to transform raw customer
        data into the format expected by the model. If no preprocessing pipeline
        is loaded, it converts the data to a DataFrame and extracts values.
        
        Args:
            customer_data: Dictionary containing customer attributes
            
        Returns:
            Preprocessed feature array ready for model prediction
            
        Raises:
            PreprocessingError: If preprocessing fails
            
        Validates: Requirements 6.4
        """
        try:
            logger.debug("Preprocessing customer input data")
            
            # Convert to DataFrame for consistent processing
            df = pd.DataFrame([customer_data])
            
            # Apply preprocessing pipeline if available
            if self.preprocessing_pipeline is not None:
                logger.debug("Applying preprocessing pipeline")
                preprocessed = self.preprocessing_pipeline.transform(df)
            else:
                logger.warning("No preprocessing pipeline loaded, using raw values")
                # Extract values in the expected order
                # This assumes the model was trained with the same feature order
                preprocessed = df.values
            
            logger.debug(f"Preprocessed data shape: {preprocessed.shape}")
            return preprocessed
            
        except Exception as e:
            error_msg = f"Preprocessing failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise PreprocessingError(error_msg) from e
    
    def predict(self, customer_data: Dict[str, Any]) -> PredictionResult:
        """
        Generate churn prediction for a customer.
        
        This method applies the complete prediction pipeline:
        1. Validates that model is loaded
        2. Preprocesses the input data
        3. Generates probability prediction
        4. Classifies risk level
        
        Args:
            customer_data: Dictionary containing customer attributes
            
        Returns:
            PredictionResult containing churn probability and risk label
            
        Raises:
            ModelNotLoadedError: If model is not loaded
            PreprocessingError: If preprocessing fails
            PredictorError: If prediction fails
            
        Validates: Requirements 6.2, 6.4, 7.4
            
        Examples:
            >>> predictor = Predictor(mlflow_uri="http://localhost:5000")
            >>> predictor.load_model(model_name="churn_model")
            >>> customer = {
            ...     "gender": "Female",
            ...     "senior_citizen": 0,
            ...     "partner": "Yes",
            ...     "tenure": 12,
            ...     # ... other fields
            ... }
            >>> result = predictor.predict(customer)
            >>> print(f"Churn probability: {result.churn_probability:.2%}")
            >>> print(f"Risk level: {result.risk_label}")
        """
        try:
            # Check if model is loaded
            self._check_model_loaded()
            
            logger.info("Generating churn prediction")
            
            # Preprocess input data
            preprocessed_data = self._preprocess_input(customer_data)
            
            # Generate prediction probabilities
            # For binary classification, predict_proba returns [prob_class_0, prob_class_1]
            # We want the probability of churn (class 1)
            probabilities = self.model.predict_proba(preprocessed_data)
            churn_probability = float(probabilities[0][1])
            
            logger.info(f"Predicted churn probability: {churn_probability:.4f}")
            
            # Classify risk level
            risk_label = self.classify_risk(churn_probability)
            logger.info(f"Risk classification: {risk_label}")
            
            return PredictionResult(
                churn_probability=churn_probability,
                risk_label=risk_label
            )
            
        except (ModelNotLoadedError, PreprocessingError):
            # Re-raise these specific errors as-is
            raise
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise PredictorError(error_msg) from e
    
    @staticmethod
    def classify_risk(churn_probability: float) -> str:
        """
        Classify churn risk level based on probability.
        
        Risk levels are classified as:
        - Low: 0.0 ≤ probability < 0.33
        - Medium: 0.33 ≤ probability < 0.66
        - High: 0.66 ≤ probability ≤ 1.0
        
        Args:
            churn_probability: Churn probability value (0-1)
            
        Returns:
            Risk label: "Low", "Medium", or "High"
            
        Validates: Requirements 7.4
            
        Examples:
            >>> Predictor.classify_risk(0.15)
            'Low'
            >>> Predictor.classify_risk(0.50)
            'Medium'
            >>> Predictor.classify_risk(0.85)
            'High'
        """
        if churn_probability < 0.33:
            return "Low"
        elif churn_probability < 0.66:
            return "Medium"
        else:
            return "High"
    
    def is_ready(self) -> bool:
        """
        Check if predictor is ready to make predictions.
        
        Returns:
            True if model is loaded, False otherwise
            
        Examples:
            >>> predictor = Predictor(mlflow_uri="http://localhost:5000")
            >>> predictor.is_ready()
            False
            >>> predictor.load_model(model_name="churn_model")
            >>> predictor.is_ready()
            True
        """
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model name, version, and status
            
        Examples:
            >>> predictor = Predictor(mlflow_uri="http://localhost:5000")
            >>> predictor.load_model(model_name="churn_model")
            >>> info = predictor.get_model_info()
            >>> print(info)
            {'model_name': 'churn_model', 'model_version': '3', 'model_loaded': True}
        """
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_loaded": self.model is not None,
            "preprocessing_pipeline_loaded": self.preprocessing_pipeline is not None
        }
