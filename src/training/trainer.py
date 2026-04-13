"""
Training service module for the Customer Churn MLOps Pipeline.

This module provides the TrainingService class that handles model training
with MLflow integration, including experiment tracking, model evaluation,
and model registry operations.
"""

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Import centralized logging configuration
from ..logging_config import get_logger

# Configure logger
logger = get_logger('training_service')


class TrainingError(Exception):
    """Base exception for training errors."""
    pass


class ModelEvaluationError(TrainingError):
    """Raised when model evaluation fails."""
    pass


class MLflowError(TrainingError):
    """Raised when MLflow operations fail."""
    pass


@dataclass
class TrainingConfig:
    """
    Configuration for model training.
    
    Attributes:
        n_estimators: Number of trees in the random forest (default: 100)
        max_depth: Maximum depth of trees (default: None for unlimited)
        random_state: Random seed for reproducibility (default: 42)
        test_size: Proportion of data for testing (default: 0.2)
    """
    n_estimators: int = 100
    max_depth: Optional[int] = None
    random_state: int = 42
    test_size: float = 0.2


@dataclass
class TrainingMetrics:
    """
    Metrics computed during model evaluation.
    
    Attributes:
        accuracy: Classification accuracy (0-1)
        precision: Precision score (0-1)
        recall: Recall score (0-1)
        f1_score: F1 score (0-1)
        roc_auc: ROC-AUC score (0-1)
        confusion_matrix: Confusion matrix as numpy array
        feature_importance: Dictionary mapping feature indices to importance values
    """
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: np.ndarray
    feature_importance: Dict[str, float]


class TrainingService:
    """
    Training service for Random Forest classifier with MLflow integration.
    
    This class manages the complete training workflow including:
    - Model training with configurable hyperparameters
    - Progress logging during training
    - MLflow experiment tracking
    - Model evaluation and metrics computation
    - Model registration in MLflow Model Registry
    
    The service integrates with MLflow for experiment tracking and model
    versioning, enabling reproducible ML workflows.
    """
    
    def __init__(self, mlflow_uri: str, config: Optional[TrainingConfig] = None):
        """
        Initialize the TrainingService with MLflow configuration.
        
        This method sets up the MLflow tracking URI and initializes the
        training configuration. The MLflow tracking server must be running
        and accessible at the specified URI.
        
        Args:
            mlflow_uri: MLflow tracking server URI (e.g., "http://localhost:5000")
            config: Training configuration object. If None, uses default configuration.
            
        Examples:
            >>> service = TrainingService(mlflow_uri="http://localhost:5000")
            >>> # Or with custom configuration
            >>> config = TrainingConfig(n_estimators=200, max_depth=15)
            >>> service = TrainingService(mlflow_uri="http://localhost:5000", config=config)
        """
        self.mlflow_uri = mlflow_uri
        self.config = config if config is not None else TrainingConfig()
        
        # Configure MLflow tracking URI
        mlflow.set_tracking_uri(self.mlflow_uri)
        logger.info(f"TrainingService initialized with MLflow URI: {self.mlflow_uri}")
        logger.info(f"Training configuration: n_estimators={self.config.n_estimators}, "
                   f"max_depth={self.config.max_depth}, random_state={self.config.random_state}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """
        Train a Random Forest classifier on the provided training data.
        
        This method trains a RandomForestClassifier using the hyperparameters
        specified in the training configuration. Progress is logged during
        training to provide visibility into the training process.
        
        The method supports configurable hyperparameters:
        - n_estimators: Number of trees in the forest
        - max_depth: Maximum depth of each tree
        - random_state: Random seed for reproducibility
        
        Args:
            X_train: Training features as numpy array of shape (n_samples, n_features)
            y_train: Training labels as numpy array of shape (n_samples,)
            
        Returns:
            RandomForestClassifier: Trained model ready for predictions
            
        Raises:
            TrainingError: If training fails due to invalid data or configuration
            
        Examples:
            >>> service = TrainingService(mlflow_uri="http://localhost:5000")
            >>> model = service.train(X_train, y_train)
            >>> predictions = model.predict(X_test)
        """
        try:
            logger.info("Starting model training")
            logger.info(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
            
            # Validate input data
            if len(X_train) == 0 or len(y_train) == 0:
                raise TrainingError("Training data is empty")
            
            if len(X_train) != len(y_train):
                raise TrainingError(
                    f"Mismatch between features and labels: "
                    f"X_train has {len(X_train)} samples, y_train has {len(y_train)} samples"
                )
            
            # Log training start
            logger.info(f"Training Random Forest with {self.config.n_estimators} estimators")
            if self.config.max_depth is not None:
                logger.info(f"Maximum tree depth: {self.config.max_depth}")
            else:
                logger.info("Maximum tree depth: unlimited")
            
            # Create and train the model
            model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
                verbose=1,  # Enable verbose output for progress logging
                n_jobs=-1   # Use all available CPU cores
            )
            
            # Log progress message
            logger.info("Fitting Random Forest model...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Log completion
            logger.info("Model training completed successfully")
            logger.info(f"Model trained on {len(X_train)} samples with {X_train.shape[1]} features")
            
            # Log feature importance information
            if hasattr(model, 'feature_importances_'):
                top_features = np.argsort(model.feature_importances_)[-5:][::-1]
                logger.info(f"Top 5 most important feature indices: {top_features.tolist()}")
            
            return model
            
        except TrainingError:
            # Re-raise TrainingError as-is
            raise
        except Exception as e:
            error_msg = f"Model training failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise TrainingError(error_msg) from e
    
    def evaluate(self, model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray) -> TrainingMetrics:
        """
        Evaluate a trained model on test data and compute comprehensive metrics.
        
        This method computes all required evaluation metrics including:
        - Accuracy: Overall classification accuracy
        - Precision: Positive predictive value
        - Recall: True positive rate (sensitivity)
        - F1 Score: Harmonic mean of precision and recall
        - ROC-AUC: Area under the ROC curve
        - Confusion Matrix: True/false positives/negatives
        - Feature Importance: Relative importance of each feature
        
        Args:
            model: Trained RandomForestClassifier to evaluate
            X_test: Test features as numpy array of shape (n_samples, n_features)
            y_test: Test labels as numpy array of shape (n_samples,)
            
        Returns:
            TrainingMetrics: Dataclass containing all computed metrics
            
        Raises:
            ModelEvaluationError: If evaluation fails due to invalid data or model
            
        Examples:
            >>> service = TrainingService(mlflow_uri="http://localhost:5000")
            >>> model = service.train(X_train, y_train)
            >>> metrics = service.evaluate(model, X_test, y_test)
            >>> print(f"Accuracy: {metrics.accuracy:.3f}")
            >>> print(f"F1 Score: {metrics.f1_score:.3f}")
            >>> print(f"ROC-AUC: {metrics.roc_auc:.3f}")
        """
        try:
            logger.info("Starting model evaluation")
            logger.info(f"Test data shape: X_test={X_test.shape}, y_test={y_test.shape}")
            
            # Validate input data
            if len(X_test) == 0 or len(y_test) == 0:
                raise ModelEvaluationError("Test data is empty")
            
            if len(X_test) != len(y_test):
                raise ModelEvaluationError(
                    f"Mismatch between features and labels: "
                    f"X_test has {len(X_test)} samples, y_test has {len(y_test)} samples"
                )
            
            # Validate model is fitted
            if not hasattr(model, 'estimators_'):
                raise ModelEvaluationError("Model is not fitted. Call train() first.")
            
            # Generate predictions
            logger.info("Generating predictions on test set...")
            y_pred = model.predict(X_test)
            
            # For ROC-AUC, we need probability predictions
            # Check if model supports predict_proba (it should for RandomForest)
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
                # For binary classification, use positive class probabilities
                # For multiclass, use one-vs-rest approach
                if y_pred_proba.shape[1] == 2:
                    y_pred_proba_positive = y_pred_proba[:, 1]
                else:
                    # For multiclass, compute ROC-AUC using ovr (one-vs-rest)
                    y_pred_proba_positive = y_pred_proba
            else:
                raise ModelEvaluationError("Model does not support probability predictions")
            
            # Compute metrics
            logger.info("Computing evaluation metrics...")
            
            # Accuracy
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Accuracy: {accuracy:.4f}")
            
            # F1 Score
            # For binary classification, use 'binary' average
            # For multiclass, use 'weighted' average
            n_classes = len(np.unique(y_test))
            if n_classes == 2:
                f1 = f1_score(y_test, y_pred, average='binary')
            else:
                f1 = f1_score(y_test, y_pred, average='weighted')
            logger.info(f"F1 Score: {f1:.4f}")
            
            # ROC-AUC
            # For binary classification, use binary mode
            # For multiclass, use ovr (one-vs-rest)
            if n_classes == 2:
                roc_auc = roc_auc_score(y_test, y_pred_proba_positive)
            else:
                roc_auc = roc_auc_score(y_test, y_pred_proba_positive, multi_class='ovr')
            logger.info(f"ROC-AUC: {roc_auc:.4f}")
            
            # Confusion Matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            logger.info(f"Confusion Matrix:\n{conf_matrix}")
            
            # Feature Importance
            feature_importances = model.feature_importances_
            # Create dictionary mapping feature index to importance
            feature_importance_dict = {
                f"feature_{i}": float(importance) 
                for i, importance in enumerate(feature_importances)
            }
            
            # Log top 5 most important features
            top_features_idx = np.argsort(feature_importances)[-5:][::-1]
            logger.info("Top 5 most important features:")
            for idx in top_features_idx:
                logger.info(f"  feature_{idx}: {feature_importances[idx]:.4f}")
            
            # For precision and recall, compute them separately
            # We'll use sklearn's precision_score and recall_score
            from sklearn.metrics import precision_score, recall_score
            
            if n_classes == 2:
                precision = precision_score(y_test, y_pred, average='binary')
                recall = recall_score(y_test, y_pred, average='binary')
            else:
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
            
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            
            # Create and return TrainingMetrics dataclass
            metrics = TrainingMetrics(
                accuracy=float(accuracy),
                precision=float(precision),
                recall=float(recall),
                f1_score=float(f1),
                roc_auc=float(roc_auc),
                confusion_matrix=conf_matrix,
                feature_importance=feature_importance_dict
            )
            
            logger.info("Model evaluation completed successfully")
            return metrics
            
        except ModelEvaluationError:
            # Re-raise ModelEvaluationError as-is
            raise
        except Exception as e:
            error_msg = f"Model evaluation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ModelEvaluationError(error_msg) from e
    
    def log_experiment(
        self, 
        model: RandomForestClassifier, 
        metrics: TrainingMetrics, 
        params: Dict[str, Any],
        preprocessing_pipeline: Optional[Any] = None
    ) -> str:
        """
        Log a training experiment to MLflow with all artifacts and metadata.
        
        This method logs a complete training experiment to MLflow, including:
        - Hyperparameters used for training
        - All evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
        - Trained model artifact
        - Preprocessing pipeline (if provided)
        - Feature importance visualization
        - Confusion matrix visualization
        - Timestamp and run identifier tags
        
        The method creates visualizations for feature importance and confusion matrix,
        saves them as artifacts, and ensures all metadata is properly tracked for
        experiment reproducibility and comparison.
        
        Args:
            model: Trained RandomForestClassifier to log
            metrics: TrainingMetrics containing all evaluation metrics
            params: Dictionary of hyperparameters used for training
            preprocessing_pipeline: Optional preprocessing pipeline to log
            
        Returns:
            str: MLflow run ID for the logged experiment
            
        Raises:
            MLflowError: If logging to MLflow fails
            
        Examples:
            >>> service = TrainingService(mlflow_uri="http://localhost:5000")
            >>> model = service.train(X_train, y_train)
            >>> metrics = service.evaluate(model, X_test, y_test)
            >>> params = {"n_estimators": 100, "max_depth": 10}
            >>> run_id = service.log_experiment(model, metrics, params)
            >>> print(f"Experiment logged with run ID: {run_id}")
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import seaborn as sns
            from datetime import datetime
            import tempfile
            import os
            
            logger.info("Starting MLflow experiment logging")
            
            # Start MLflow run
            with mlflow.start_run() as run:
                run_id = run.info.run_id
                logger.info(f"MLflow run started with ID: {run_id}")
                
                # Log hyperparameters
                logger.info("Logging hyperparameters to MLflow")
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
                logger.info(f"Logged {len(params)} hyperparameters")
                
                # Log evaluation metrics
                logger.info("Logging evaluation metrics to MLflow")
                mlflow.log_metric("accuracy", metrics.accuracy)
                mlflow.log_metric("precision", metrics.precision)
                mlflow.log_metric("recall", metrics.recall)
                mlflow.log_metric("f1_score", metrics.f1_score)
                mlflow.log_metric("roc_auc", metrics.roc_auc)
                logger.info("Logged 5 evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)")
                
                # Log model artifact
                logger.info("Logging model artifact to MLflow")
                mlflow.sklearn.log_model(model, "model")
                logger.info("Model artifact logged successfully")
                
                # Log preprocessing pipeline if provided
                if preprocessing_pipeline is not None:
                    logger.info("Logging preprocessing pipeline to MLflow")
                    mlflow.sklearn.log_model(preprocessing_pipeline, "preprocessing_pipeline")
                    logger.info("Preprocessing pipeline logged successfully")
                
                # Create and log feature importance plot
                logger.info("Creating feature importance visualization")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Extract feature names and importances
                feature_names = list(metrics.feature_importance.keys())
                importances = list(metrics.feature_importance.values())
                
                # Sort by importance
                sorted_indices = np.argsort(importances)[::-1]
                sorted_features = [feature_names[i] for i in sorted_indices]
                sorted_importances = [importances[i] for i in sorted_indices]
                
                # Plot top 20 features (or all if less than 20)
                n_features_to_plot = min(20, len(sorted_features))
                ax.barh(range(n_features_to_plot), sorted_importances[:n_features_to_plot])
                ax.set_yticks(range(n_features_to_plot))
                ax.set_yticklabels(sorted_features[:n_features_to_plot])
                ax.set_xlabel('Importance')
                ax.set_ylabel('Feature')
                ax.set_title('Feature Importance (Top 20)')
                ax.invert_yaxis()  # Highest importance at top
                plt.tight_layout()
                
                # Save and log feature importance plot
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    feature_importance_path = tmp_file.name
                plt.savefig(feature_importance_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                mlflow.log_artifact(feature_importance_path, "plots")
                try:
                    os.unlink(feature_importance_path)
                except (PermissionError, OSError):
                    # On Windows, file may still be locked - log warning but continue
                    logger.warning(f"Could not delete temporary file: {feature_importance_path}")
                
                logger.info("Feature importance plot logged successfully")
                
                # Create and log confusion matrix plot
                logger.info("Creating confusion matrix visualization")
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Use seaborn for better visualization
                sns.heatmap(
                    metrics.confusion_matrix,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    ax=ax,
                    cbar=True,
                    square=True
                )
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
                ax.set_title('Confusion Matrix')
                plt.tight_layout()
                
                # Save and log confusion matrix plot
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    confusion_matrix_path = tmp_file.name
                plt.savefig(confusion_matrix_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                mlflow.log_artifact(confusion_matrix_path, "plots")
                try:
                    os.unlink(confusion_matrix_path)
                except (PermissionError, OSError):
                    # On Windows, file may still be locked - log warning but continue
                    logger.warning(f"Could not delete temporary file: {confusion_matrix_path}")
                
                logger.info("Confusion matrix plot logged successfully")
                
                # Add timestamp and run identifier tags
                logger.info("Adding tags to MLflow run")
                timestamp = datetime.utcnow().isoformat()
                mlflow.set_tag("timestamp", timestamp)
                mlflow.set_tag("run_identifier", run_id)
                mlflow.set_tag("model_type", "RandomForestClassifier")
                mlflow.set_tag("framework", "scikit-learn")
                logger.info(f"Added tags: timestamp={timestamp}, run_identifier={run_id}")
                
                logger.info(f"MLflow experiment logging completed successfully. Run ID: {run_id}")
                return run_id
                
        except Exception as e:
            error_msg = f"MLflow experiment logging failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise MLflowError(error_msg) from e


    def log_experiment(
        self,
        model: RandomForestClassifier,
        metrics: TrainingMetrics,
        params: Dict[str, Any],
        preprocessing_pipeline: Optional[Any] = None
    ) -> str:
        """
        Log a training experiment to MLflow with all artifacts and metadata.

        This method logs a complete training experiment to MLflow, including:
        - Hyperparameters used for training
        - All evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
        - Trained model artifact
        - Preprocessing pipeline (if provided)
        - Feature importance visualization
        - Confusion matrix visualization
        - Timestamp and run identifier tags

        The method creates visualizations for feature importance and confusion matrix,
        saves them as artifacts, and ensures all metadata is properly tracked for
        experiment reproducibility and comparison.

        Args:
            model: Trained RandomForestClassifier to log
            metrics: TrainingMetrics containing all evaluation metrics
            params: Dictionary of hyperparameters used for training
            preprocessing_pipeline: Optional preprocessing pipeline to log

        Returns:
            str: MLflow run ID for the logged experiment

        Raises:
            MLflowError: If logging to MLflow fails

        Examples:
            >>> service = TrainingService(mlflow_uri="http://localhost:5000")
            >>> model = service.train(X_train, y_train)
            >>> metrics = service.evaluate(model, X_test, y_test)
            >>> params = {"n_estimators": 100, "max_depth": 10}
            >>> run_id = service.log_experiment(model, metrics, params)
            >>> print(f"Experiment logged with run ID: {run_id}")
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import seaborn as sns
            from datetime import datetime
            import tempfile
            import os

            logger.info("Starting MLflow experiment logging")

            # Start MLflow run
            with mlflow.start_run() as run:
                run_id = run.info.run_id
                logger.info(f"MLflow run started with ID: {run_id}")

                # Log hyperparameters
                logger.info("Logging hyperparameters to MLflow")
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
                logger.info(f"Logged {len(params)} hyperparameters")

                # Log evaluation metrics
                logger.info("Logging evaluation metrics to MLflow")
                mlflow.log_metric("accuracy", metrics.accuracy)
                mlflow.log_metric("precision", metrics.precision)
                mlflow.log_metric("recall", metrics.recall)
                mlflow.log_metric("f1_score", metrics.f1_score)
                mlflow.log_metric("roc_auc", metrics.roc_auc)
                logger.info("Logged 5 evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)")

                # Log model artifact
                logger.info("Logging model artifact to MLflow")
                mlflow.sklearn.log_model(model, "model")
                logger.info("Model artifact logged successfully")

                # Log preprocessing pipeline if provided
                if preprocessing_pipeline is not None:
                    logger.info("Logging preprocessing pipeline to MLflow")
                    mlflow.sklearn.log_model(preprocessing_pipeline, "preprocessing_pipeline")
                    logger.info("Preprocessing pipeline logged successfully")

                # Create and log feature importance plot
                logger.info("Creating feature importance visualization")
                fig, ax = plt.subplots(figsize=(10, 6))

                # Extract feature names and importances
                feature_names = list(metrics.feature_importance.keys())
                importances = list(metrics.feature_importance.values())

                # Sort by importance
                sorted_indices = np.argsort(importances)[::-1]
                sorted_features = [feature_names[i] for i in sorted_indices]
                sorted_importances = [importances[i] for i in sorted_indices]

                # Plot top 20 features (or all if less than 20)
                n_features_to_plot = min(20, len(sorted_features))
                ax.barh(range(n_features_to_plot), sorted_importances[:n_features_to_plot])
                ax.set_yticks(range(n_features_to_plot))
                ax.set_yticklabels(sorted_features[:n_features_to_plot])
                ax.set_xlabel('Importance')
                ax.set_ylabel('Feature')
                ax.set_title('Feature Importance (Top 20)')
                ax.invert_yaxis()  # Highest importance at top
                plt.tight_layout()

                # Save and log feature importance plot
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    feature_importance_path = tmp_file.name
                plt.savefig(feature_importance_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                mlflow.log_artifact(feature_importance_path, "plots")
                try:
                    os.unlink(feature_importance_path)
                except (PermissionError, OSError):
                    # On Windows, file may still be locked - log warning but continue
                    logger.warning(f"Could not delete temporary file: {feature_importance_path}")

                logger.info("Feature importance plot logged successfully")

                # Create and log confusion matrix plot
                logger.info("Creating confusion matrix visualization")
                fig, ax = plt.subplots(figsize=(8, 6))

                # Use seaborn for better visualization
                sns.heatmap(
                    metrics.confusion_matrix,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    ax=ax,
                    cbar=True,
                    square=True
                )
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
                ax.set_title('Confusion Matrix')
                plt.tight_layout()

                # Save and log confusion matrix plot
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    confusion_matrix_path = tmp_file.name
                plt.savefig(confusion_matrix_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                mlflow.log_artifact(confusion_matrix_path, "plots")
                try:
                    os.unlink(confusion_matrix_path)
                except (PermissionError, OSError):
                    # On Windows, file may still be locked - log warning but continue
                    logger.warning(f"Could not delete temporary file: {confusion_matrix_path}")

                logger.info("Confusion matrix plot logged successfully")

                # Add timestamp and run identifier tags
                logger.info("Adding tags to MLflow run")
                timestamp = datetime.utcnow().isoformat()
                mlflow.set_tag("timestamp", timestamp)
                mlflow.set_tag("run_identifier", run_id)
                mlflow.set_tag("model_type", "RandomForestClassifier")
                mlflow.set_tag("framework", "scikit-learn")
                logger.info(f"Added tags: timestamp={timestamp}, run_identifier={run_id}")

                logger.info(f"MLflow experiment logging completed successfully. Run ID: {run_id}")
                return run_id

        except Exception as e:
            error_msg = f"MLflow experiment logging failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise MLflowError(error_msg) from e
    
    def register_model(self, run_id: str, model_name: str, metrics: Optional[TrainingMetrics] = None) -> Any:
        """
        Register a trained model in the MLflow Model Registry with versioning.
        
        This method registers a model from a completed MLflow run into the Model Registry,
        enabling version tracking and lifecycle management. The model is registered with
        complete metadata including training date, evaluation metrics, and hyperparameters.
        
        The Model Registry provides:
        - Unique version numbers for each registered model
        - Metadata storage (training date, metrics, hyperparameters)
        - Stage management (None, Staging, Production)
        - Model lineage tracking
        
        Args:
            run_id: MLflow run ID containing the model artifact to register
            model_name: Name for the registered model (e.g., "churn_model")
            metrics: Optional TrainingMetrics to store as model metadata
            
        Returns:
            ModelVersion: MLflow ModelVersion object with version number and metadata
            
        Raises:
            MLflowError: If model registration fails
            
        Examples:
            >>> service = TrainingService(mlflow_uri="http://localhost:5000")
            >>> model = service.train(X_train, y_train)
            >>> metrics = service.evaluate(model, X_test, y_test)
            >>> run_id = service.log_experiment(model, metrics, params)
            >>> model_version = service.register_model(run_id, "churn_model", metrics)
            >>> print(f"Registered model version: {model_version.version}")
        """
        try:
            from datetime import datetime
            
            logger.info(f"Registering model '{model_name}' from run {run_id}")
            
            # Construct model URI from run_id
            model_uri = f"runs:/{run_id}/model"
            
            # Register the model in MLflow Model Registry
            # This creates a new version if the model name already exists
            logger.info(f"Registering model from URI: {model_uri}")
            model_version = mlflow.register_model(model_uri, model_name)
            
            logger.info(f"Model registered successfully: {model_name} version {model_version.version}")
            
            # Get the MLflow client to add metadata
            client = mlflow.tracking.MlflowClient()
            
            # Add training date as a tag
            training_date = datetime.utcnow().isoformat()
            client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="training_date",
                value=training_date
            )
            logger.info(f"Added training_date tag: {training_date}")
            
            # Add metrics as tags if provided
            if metrics is not None:
                logger.info("Adding evaluation metrics as model version tags")
                client.set_model_version_tag(
                    name=model_name,
                    version=model_version.version,
                    key="accuracy",
                    value=str(metrics.accuracy)
                )
                client.set_model_version_tag(
                    name=model_name,
                    version=model_version.version,
                    key="f1_score",
                    value=str(metrics.f1_score)
                )
                client.set_model_version_tag(
                    name=model_name,
                    version=model_version.version,
                    key="roc_auc",
                    value=str(metrics.roc_auc)
                )
                logger.info(f"Added metrics tags: accuracy={metrics.accuracy:.4f}, "
                          f"f1_score={metrics.f1_score:.4f}, roc_auc={metrics.roc_auc:.4f}")
            
            # Add hyperparameters as tags
            logger.info("Adding hyperparameters as model version tags")
            client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="n_estimators",
                value=str(self.config.n_estimators)
            )
            client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="max_depth",
                value=str(self.config.max_depth) if self.config.max_depth is not None else "None"
            )
            client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="random_state",
                value=str(self.config.random_state)
            )
            logger.info(f"Added hyperparameter tags: n_estimators={self.config.n_estimators}, "
                       f"max_depth={self.config.max_depth}, random_state={self.config.random_state}")
            
            # Add description
            description = (
                f"Random Forest model trained on {training_date}. "
                f"Metrics: accuracy={metrics.accuracy:.4f}, f1={metrics.f1_score:.4f}, "
                f"roc_auc={metrics.roc_auc:.4f}" if metrics else "Random Forest model"
            )
            client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=description
            )
            logger.info(f"Added model description")
            
            logger.info(f"Model registration completed: {model_name} v{model_version.version}")
            return model_version
            
        except Exception as e:
            error_msg = f"Model registration failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise MLflowError(error_msg) from e
    
    def promote_to_production(self, model_name: str, version: int, archive_existing: bool = True) -> None:
        """
        Promote a model version to Production stage in the Model Registry.
        
        This method transitions a model version to the Production stage, making it
        the active model for serving predictions. The method supports automatic
        archiving of existing Production models to prevent conflicts.
        
        Stage Lifecycle:
        - None: Initial state after registration
        - Staging: Testing/validation stage
        - Production: Active serving stage
        - Archived: Retired models
        
        The promotion logic implements a performance-based strategy:
        - New model is promoted if ROC-AUC ≥ current Production model + 0.01
        - This threshold ensures meaningful performance improvements
        - Prevents unnecessary model churn from minor fluctuations
        
        Args:
            model_name: Name of the registered model
            version: Version number to promote to Production
            archive_existing: If True, archive existing Production models (default: True)
            
        Raises:
            MLflowError: If promotion fails
            
        Examples:
            >>> service = TrainingService(mlflow_uri="http://localhost:5000")
            >>> # After training and registering a model
            >>> service.promote_to_production("churn_model", version=3)
            >>> # Model version 3 is now in Production stage
        """
        try:
            logger.info(f"Promoting model '{model_name}' version {version} to Production")
            
            # Get the MLflow client
            client = mlflow.tracking.MlflowClient()
            
            # Archive existing Production models if requested
            if archive_existing:
                logger.info("Checking for existing Production models to archive")
                try:
                    # Get all versions of this model
                    model_versions = client.search_model_versions(f"name='{model_name}'")
                    
                    # Find and archive any existing Production versions
                    production_versions = [
                        mv for mv in model_versions 
                        if mv.current_stage == "Production"
                    ]
                    
                    if production_versions:
                        logger.info(f"Found {len(production_versions)} existing Production model(s)")
                        for prod_version in production_versions:
                            logger.info(f"Archiving model version {prod_version.version}")
                            client.transition_model_version_stage(
                                name=model_name,
                                version=prod_version.version,
                                stage="Archived"
                            )
                            logger.info(f"Model version {prod_version.version} archived successfully")
                    else:
                        logger.info("No existing Production models found")
                        
                except Exception as e:
                    logger.warning(f"Could not archive existing Production models: {str(e)}")
                    # Continue with promotion even if archiving fails
            
            # Promote the specified version to Production
            logger.info(f"Transitioning model version {version} to Production stage")
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production"
            )
            
            logger.info(f"Model '{model_name}' version {version} promoted to Production successfully")
            
        except Exception as e:
            error_msg = f"Model promotion failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise MLflowError(error_msg) from e
    
    def should_promote_model(self, new_roc_auc: float, model_name: str) -> bool:
        """
        Determine if a new model should be promoted to Production based on performance.
        
        This method implements the promotion logic by comparing the new model's ROC-AUC
        score against the current Production model. Promotion is recommended if:
        
        new_roc_auc >= current_production_roc_auc + 0.01
        
        The 0.01 threshold ensures that only meaningful improvements trigger promotion,
        preventing unnecessary model churn from minor performance fluctuations.
        
        Args:
            new_roc_auc: ROC-AUC score of the new model
            model_name: Name of the registered model to check
            
        Returns:
            bool: True if new model should be promoted, False otherwise
            
        Examples:
            >>> service = TrainingService(mlflow_uri="http://localhost:5000")
            >>> new_metrics = service.evaluate(new_model, X_test, y_test)
            >>> should_promote = service.should_promote_model(new_metrics.roc_auc, "churn_model")
            >>> if should_promote:
            >>>     service.promote_to_production("churn_model", new_version)
        """
        try:
            logger.info(f"Checking if new model should be promoted (new ROC-AUC: {new_roc_auc:.4f})")
            
            # Get the MLflow client
            client = mlflow.tracking.MlflowClient()
            
            # Find current Production model
            try:
                model_versions = client.search_model_versions(f"name='{model_name}'")
                production_versions = [
                    mv for mv in model_versions 
                    if mv.current_stage == "Production"
                ]
                
                if not production_versions:
                    logger.info("No Production model found. Recommending promotion.")
                    return True
                
                # Get the first (should be only one) Production version
                production_version = production_versions[0]
                logger.info(f"Found Production model version {production_version.version}")
                
                # Get ROC-AUC from tags
                production_tags = {tag.key: tag.value for tag in production_version.tags}
                
                if "roc_auc" not in production_tags:
                    logger.warning("Production model has no roc_auc tag. Recommending promotion.")
                    return True
                
                current_roc_auc = float(production_tags["roc_auc"])
                logger.info(f"Current Production ROC-AUC: {current_roc_auc:.4f}")
                
                # Check if new model meets promotion threshold
                improvement = new_roc_auc - current_roc_auc
                threshold = 0.01
                
                should_promote = new_roc_auc >= current_roc_auc + threshold
                
                logger.info(f"ROC-AUC improvement: {improvement:+.4f}")
                logger.info(f"Promotion threshold: {threshold:.4f}")
                logger.info(f"Should promote: {should_promote}")
                
                return should_promote
                
            except Exception as e:
                logger.warning(f"Could not retrieve Production model: {str(e)}")
                logger.info("Recommending promotion due to error retrieving current model")
                return True
                
        except Exception as e:
            error_msg = f"Error checking promotion criteria: {str(e)}"
            logger.error(error_msg, exc_info=True)
            # In case of error, be conservative and don't promote
            return False
    
    def save_model(self, model: RandomForestClassifier, model_name: str, version: int, models_dir: str = "models") -> str:
        """
        Save a trained model to the local filesystem with version-based naming.
        
        This method persists a trained model to disk using a consistent naming convention
        that includes the model name and version number. The model is saved in pickle format
        for efficient serialization and deserialization.
        
        Naming Convention:
        - Format: {model_name}_v{version}.pkl
        - Example: churn_model_v1.pkl, churn_model_v2.pkl
        
        The method creates the models directory if it doesn't exist and logs the
        save operation for audit trail purposes.
        
        Args:
            model: Trained RandomForestClassifier to save
            model_name: Base name for the model file (e.g., "churn_model")
            version: Version number for the model (e.g., 1, 2, 3)
            models_dir: Directory path where models should be saved (default: "models")
            
        Returns:
            str: Full path to the saved model file
            
        Raises:
            TrainingError: If model saving fails
            
        Examples:
            >>> service = TrainingService(mlflow_uri="http://localhost:5000")
            >>> model = service.train(X_train, y_train)
            >>> model_path = service.save_model(model, "churn_model", version=1)
            >>> print(f"Model saved to: {model_path}")
            Model saved to: models/churn_model_v1.pkl
        """
        try:
            import os
            import pickle
            
            logger.info(f"Saving model '{model_name}' version {version} to local filesystem")
            
            # Create models directory if it doesn't exist
            os.makedirs(models_dir, exist_ok=True)
            logger.info(f"Ensured models directory exists: {models_dir}")
            
            # Construct filename with version-based naming convention
            filename = f"{model_name}_v{version}.pkl"
            filepath = os.path.join(models_dir, filename)
            
            logger.info(f"Saving model to: {filepath}")
            
            # Save model using pickle
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            
            # Verify file was created
            if not os.path.exists(filepath):
                raise TrainingError(f"Model file was not created: {filepath}")
            
            file_size = os.path.getsize(filepath)
            logger.info(f"Model saved successfully: {filepath} ({file_size} bytes)")
            
            return filepath
            
        except Exception as e:
            error_msg = f"Failed to save model: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise TrainingError(error_msg) from e
    
    def load_model(self, model_name: str, version: int, models_dir: str = "models") -> RandomForestClassifier:
        """
        Load a trained model from the local filesystem using version-based naming.
        
        This method loads a previously saved model from disk using the consistent
        naming convention. The model is deserialized from pickle format and returned
        ready for making predictions.
        
        Naming Convention:
        - Format: {model_name}_v{version}.pkl
        - Example: churn_model_v1.pkl, churn_model_v2.pkl
        
        The method validates that the file exists and logs the load operation for
        audit trail purposes.
        
        Args:
            model_name: Base name of the model file (e.g., "churn_model")
            version: Version number of the model to load (e.g., 1, 2, 3)
            models_dir: Directory path where models are stored (default: "models")
            
        Returns:
            RandomForestClassifier: Loaded model ready for predictions
            
        Raises:
            TrainingError: If model file doesn't exist or loading fails
            
        Examples:
            >>> service = TrainingService(mlflow_uri="http://localhost:5000")
            >>> model = service.load_model("churn_model", version=1)
            >>> predictions = model.predict(X_test)
        """
        try:
            import os
            import pickle
            
            logger.info(f"Loading model '{model_name}' version {version} from local filesystem")
            
            # Construct filename with version-based naming convention
            filename = f"{model_name}_v{version}.pkl"
            filepath = os.path.join(models_dir, filename)
            
            logger.info(f"Loading model from: {filepath}")
            
            # Check if file exists
            if not os.path.exists(filepath):
                raise TrainingError(f"Model file not found: {filepath}")
            
            # Load model using pickle
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            
            # Validate loaded object is a RandomForestClassifier
            if not isinstance(model, RandomForestClassifier):
                raise TrainingError(
                    f"Loaded object is not a RandomForestClassifier: {type(model)}"
                )
            
            # Validate model is fitted
            if not hasattr(model, 'estimators_'):
                raise TrainingError("Loaded model is not fitted")
            
            file_size = os.path.getsize(filepath)
            logger.info(f"Model loaded successfully: {filepath} ({file_size} bytes)")
            logger.info(f"Model has {model.n_estimators} estimators and {model.n_features_in_} features")
            
            return model
            
        except TrainingError:
            # Re-raise TrainingError as-is
            raise
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise TrainingError(error_msg) from e


