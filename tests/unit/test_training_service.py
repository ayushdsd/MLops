"""
Unit tests for the TrainingService module.

This module tests the core functionality of the TrainingService class,
including initialization, model training, and error handling.
"""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.training.trainer import TrainingService, TrainingConfig, TrainingError, TrainingMetrics, ModelEvaluationError


class TestTrainingServiceInit:
    """Test TrainingService initialization."""
    
    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        service = TrainingService(mlflow_uri="http://localhost:5000")
        
        assert service.mlflow_uri == "http://localhost:5000"
        assert service.config is not None
        assert service.config.n_estimators == 100
        assert service.config.max_depth is None
        assert service.config.random_state == 42
        assert service.config.test_size == 0.2
    
    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = TrainingConfig(
            n_estimators=200,
            max_depth=15,
            random_state=123,
            test_size=0.3
        )
        service = TrainingService(mlflow_uri="http://localhost:5000", config=config)
        
        assert service.mlflow_uri == "http://localhost:5000"
        assert service.config.n_estimators == 200
        assert service.config.max_depth == 15
        assert service.config.random_state == 123
        assert service.config.test_size == 0.3


class TestTrainingServiceTrain:
    """Test TrainingService.train() method."""
    
    def test_train_with_valid_data(self):
        """Test training with valid data produces a fitted model."""
        # Create sample training data
        np.random.seed(42)
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 2, 100)
        
        # Train model
        config = TrainingConfig(n_estimators=10, random_state=42)
        service = TrainingService(mlflow_uri="http://localhost:5000", config=config)
        model = service.train(X_train, y_train)
        
        # Verify model is trained
        assert isinstance(model, RandomForestClassifier)
        assert hasattr(model, 'estimators_')
        assert len(model.estimators_) == 10
        
        # Verify model can make predictions
        predictions = model.predict(X_train)
        assert len(predictions) == len(y_train)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_train_respects_hyperparameters(self):
        """Test that trained model has configured hyperparameters."""
        X_train = np.random.rand(50, 5)
        y_train = np.random.randint(0, 2, 50)
        
        config = TrainingConfig(n_estimators=25, max_depth=8, random_state=99)
        service = TrainingService(mlflow_uri="http://localhost:5000", config=config)
        model = service.train(X_train, y_train)
        
        # Verify hyperparameters
        assert model.n_estimators == 25
        assert model.max_depth == 8
        assert model.random_state == 99
    
    def test_train_with_empty_data_raises_error(self):
        """Test that training with empty data raises TrainingError."""
        X_train = np.array([])
        y_train = np.array([])
        
        service = TrainingService(mlflow_uri="http://localhost:5000")
        
        with pytest.raises(TrainingError) as exc_info:
            service.train(X_train, y_train)
        
        assert "empty" in str(exc_info.value).lower()
    
    def test_train_with_mismatched_shapes_raises_error(self):
        """Test that training with mismatched X and y shapes raises TrainingError."""
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 2, 50)  # Different length
        
        service = TrainingService(mlflow_uri="http://localhost:5000")
        
        with pytest.raises(TrainingError) as exc_info:
            service.train(X_train, y_train)
        
        assert "mismatch" in str(exc_info.value).lower()
    
    def test_train_with_small_dataset(self):
        """Test training with a small dataset."""
        X_train = np.random.rand(10, 3)
        y_train = np.random.randint(0, 2, 10)
        
        config = TrainingConfig(n_estimators=5, random_state=42)
        service = TrainingService(mlflow_uri="http://localhost:5000", config=config)
        model = service.train(X_train, y_train)
        
        assert isinstance(model, RandomForestClassifier)
        assert hasattr(model, 'estimators_')
    
    def test_train_with_multiclass_labels(self):
        """Test training with multiclass labels (more than 2 classes)."""
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 3, 100)  # 3 classes
        
        service = TrainingService(mlflow_uri="http://localhost:5000")
        model = service.train(X_train, y_train)
        
        assert isinstance(model, RandomForestClassifier)
        predictions = model.predict(X_train)
        assert all(pred in [0, 1, 2] for pred in predictions)
    
    def test_train_produces_feature_importances(self):
        """Test that trained model has feature importances."""
        X_train = np.random.rand(50, 8)
        y_train = np.random.randint(0, 2, 50)
        
        service = TrainingService(mlflow_uri="http://localhost:5000")
        model = service.train(X_train, y_train)
        
        assert hasattr(model, 'feature_importances_')
        assert len(model.feature_importances_) == 8
        assert all(importance >= 0 for importance in model.feature_importances_)
        # Feature importances should sum to approximately 1
        assert abs(sum(model.feature_importances_) - 1.0) < 0.01


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""
    
    def test_default_config_values(self):
        """Test default configuration values."""
        config = TrainingConfig()
        
        assert config.n_estimators == 100
        assert config.max_depth is None
        assert config.random_state == 42
        assert config.test_size == 0.2
    
    def test_custom_config_values(self):
        """Test custom configuration values."""
        config = TrainingConfig(
            n_estimators=150,
            max_depth=20,
            random_state=999,
            test_size=0.25
        )
        
        assert config.n_estimators == 150
        assert config.max_depth == 20
        assert config.random_state == 999
        assert config.test_size == 0.25



class TestTrainingServiceEvaluate:
    """Test TrainingService.evaluate() method."""
    
    def test_evaluate_with_valid_binary_classification(self):
        """Test evaluation with valid binary classification data."""
        # Create and train a model
        np.random.seed(42)
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.rand(50, 10)
        y_test = np.random.randint(0, 2, 50)
        
        config = TrainingConfig(n_estimators=10, random_state=42)
        service = TrainingService(mlflow_uri="http://localhost:5000", config=config)
        model = service.train(X_train, y_train)
        
        # Evaluate model
        metrics = service.evaluate(model, X_test, y_test)
        
        # Verify metrics are returned
        assert isinstance(metrics, TrainingMetrics)
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.precision <= 1.0
        assert 0.0 <= metrics.recall <= 1.0
        assert 0.0 <= metrics.f1_score <= 1.0
        assert 0.0 <= metrics.roc_auc <= 1.0
        
        # Verify confusion matrix
        assert metrics.confusion_matrix is not None
        assert metrics.confusion_matrix.shape == (2, 2)  # Binary classification
        
        # Verify feature importance
        assert isinstance(metrics.feature_importance, dict)
        assert len(metrics.feature_importance) == 10  # 10 features
        assert all(0.0 <= v <= 1.0 for v in metrics.feature_importance.values())
    
    def test_evaluate_returns_all_required_metrics(self):
        """Test that evaluate returns all required metrics (accuracy, F1, ROC-AUC)."""
        # Create and train a model
        np.random.seed(42)
        X_train = np.random.rand(80, 5)
        y_train = np.random.randint(0, 2, 80)
        X_test = np.random.rand(20, 5)
        y_test = np.random.randint(0, 2, 20)
        
        service = TrainingService(mlflow_uri="http://localhost:5000")
        model = service.train(X_train, y_train)
        metrics = service.evaluate(model, X_test, y_test)
        
        # Verify all three required metrics are present
        assert hasattr(metrics, 'accuracy')
        assert hasattr(metrics, 'f1_score')
        assert hasattr(metrics, 'roc_auc')
        assert metrics.accuracy is not None
        assert metrics.f1_score is not None
        assert metrics.roc_auc is not None
    
    def test_evaluate_with_multiclass_classification(self):
        """Test evaluation with multiclass classification data."""
        # Create and train a model with 3 classes
        np.random.seed(42)
        X_train = np.random.rand(150, 8)
        y_train = np.random.randint(0, 3, 150)
        X_test = np.random.rand(50, 8)
        y_test = np.random.randint(0, 3, 50)
        
        config = TrainingConfig(n_estimators=10, random_state=42)
        service = TrainingService(mlflow_uri="http://localhost:5000", config=config)
        model = service.train(X_train, y_train)
        
        # Evaluate model
        metrics = service.evaluate(model, X_test, y_test)
        
        # Verify metrics are returned
        assert isinstance(metrics, TrainingMetrics)
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.f1_score <= 1.0
        assert 0.0 <= metrics.roc_auc <= 1.0
        
        # Verify confusion matrix for 3 classes
        assert metrics.confusion_matrix.shape == (3, 3)
    
    def test_evaluate_with_empty_test_data_raises_error(self):
        """Test that evaluation with empty test data raises ModelEvaluationError."""
        # Create and train a model
        X_train = np.random.rand(50, 5)
        y_train = np.random.randint(0, 2, 50)
        
        service = TrainingService(mlflow_uri="http://localhost:5000")
        model = service.train(X_train, y_train)
        
        # Try to evaluate with empty data
        X_test = np.array([])
        y_test = np.array([])
        
        with pytest.raises(ModelEvaluationError) as exc_info:
            service.evaluate(model, X_test, y_test)
        
        assert "empty" in str(exc_info.value).lower()
    
    def test_evaluate_with_mismatched_shapes_raises_error(self):
        """Test that evaluation with mismatched X and y shapes raises ModelEvaluationError."""
        # Create and train a model
        X_train = np.random.rand(50, 5)
        y_train = np.random.randint(0, 2, 50)
        
        service = TrainingService(mlflow_uri="http://localhost:5000")
        model = service.train(X_train, y_train)
        
        # Try to evaluate with mismatched shapes
        X_test = np.random.rand(30, 5)
        y_test = np.random.randint(0, 2, 20)  # Different length
        
        with pytest.raises(ModelEvaluationError) as exc_info:
            service.evaluate(model, X_test, y_test)
        
        assert "mismatch" in str(exc_info.value).lower()
    
    def test_evaluate_with_unfitted_model_raises_error(self):
        """Test that evaluation with an unfitted model raises ModelEvaluationError."""
        # Create an unfitted model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        X_test = np.random.rand(30, 5)
        y_test = np.random.randint(0, 2, 30)
        
        service = TrainingService(mlflow_uri="http://localhost:5000")
        
        with pytest.raises(ModelEvaluationError) as exc_info:
            service.evaluate(model, X_test, y_test)
        
        assert "not fitted" in str(exc_info.value).lower()
    
    def test_evaluate_confusion_matrix_shape(self):
        """Test that confusion matrix has correct shape."""
        # Create and train a model
        np.random.seed(42)
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.rand(40, 5)
        y_test = np.random.randint(0, 2, 40)
        
        service = TrainingService(mlflow_uri="http://localhost:5000")
        model = service.train(X_train, y_train)
        metrics = service.evaluate(model, X_test, y_test)
        
        # For binary classification, confusion matrix should be 2x2
        assert metrics.confusion_matrix.shape == (2, 2)
        # Sum of confusion matrix should equal number of test samples
        assert metrics.confusion_matrix.sum() == len(y_test)
    
    def test_evaluate_feature_importance_completeness(self):
        """Test that feature importance is computed for all features."""
        # Create and train a model
        np.random.seed(42)
        n_features = 12
        X_train = np.random.rand(100, n_features)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.rand(40, n_features)
        y_test = np.random.randint(0, 2, 40)
        
        service = TrainingService(mlflow_uri="http://localhost:5000")
        model = service.train(X_train, y_train)
        metrics = service.evaluate(model, X_test, y_test)
        
        # Verify feature importance for all features
        assert len(metrics.feature_importance) == n_features
        # All importance values should be non-negative
        assert all(v >= 0 for v in metrics.feature_importance.values())
        # Sum of importances should be approximately 1
        total_importance = sum(metrics.feature_importance.values())
        assert abs(total_importance - 1.0) < 0.01
    
    def test_evaluate_metrics_are_floats(self):
        """Test that all metric values are Python floats (not numpy types)."""
        # Create and train a model
        np.random.seed(42)
        X_train = np.random.rand(80, 5)
        y_train = np.random.randint(0, 2, 80)
        X_test = np.random.rand(20, 5)
        y_test = np.random.randint(0, 2, 20)
        
        service = TrainingService(mlflow_uri="http://localhost:5000")
        model = service.train(X_train, y_train)
        metrics = service.evaluate(model, X_test, y_test)
        
        # Verify all metrics are Python floats (for JSON serialization)
        assert isinstance(metrics.accuracy, float)
        assert isinstance(metrics.precision, float)
        assert isinstance(metrics.recall, float)
        assert isinstance(metrics.f1_score, float)
        assert isinstance(metrics.roc_auc, float)


class TestTrainingMetrics:
    """Test TrainingMetrics dataclass."""
    
    def test_training_metrics_creation(self):
        """Test creating a TrainingMetrics instance."""
        conf_matrix = np.array([[10, 2], [3, 15]])
        feature_importance = {"feature_0": 0.3, "feature_1": 0.7}
        
        metrics = TrainingMetrics(
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            roc_auc=0.90,
            confusion_matrix=conf_matrix,
            feature_importance=feature_importance
        )
        
        assert metrics.accuracy == 0.85
        assert metrics.precision == 0.82
        assert metrics.recall == 0.88
        assert metrics.f1_score == 0.85
        assert metrics.roc_auc == 0.90
        assert np.array_equal(metrics.confusion_matrix, conf_matrix)
        assert metrics.feature_importance == feature_importance


class TestTrainingServiceLogExperiment:
    """Test TrainingService.log_experiment() method."""
    
    def test_log_experiment_returns_run_id(self, monkeypatch):
        """Test that log_experiment returns a valid MLflow run ID."""
        import mlflow
        from unittest.mock import MagicMock, patch
        
        # Create and train a model
        np.random.seed(42)
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.rand(50, 10)
        y_test = np.random.randint(0, 2, 50)
        
        config = TrainingConfig(n_estimators=10, random_state=42)
        service = TrainingService(mlflow_uri="http://localhost:5000", config=config)
        model = service.train(X_train, y_train)
        metrics = service.evaluate(model, X_test, y_test)
        
        # Mock MLflow operations
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id_12345"
        
        with patch('mlflow.start_run', return_value=MagicMock(__enter__=lambda self: mock_run, __exit__=lambda *args: None)):
            with patch('mlflow.log_param'):
                with patch('mlflow.log_metric'):
                    with patch('mlflow.sklearn.log_model'):
                        with patch('mlflow.log_artifact'):
                            with patch('mlflow.set_tag'):
                                # Log experiment
                                params = {
                                    "n_estimators": config.n_estimators,
                                    "max_depth": config.max_depth,
                                    "random_state": config.random_state
                                }
                                
                                run_id = service.log_experiment(model, metrics, params)
                                
                                # Verify run_id is returned and is a non-empty string
                                assert isinstance(run_id, str)
                                assert len(run_id) > 0
                                assert run_id == "test_run_id_12345"
    
    def test_log_experiment_with_preprocessing_pipeline(self, monkeypatch):
        """Test that log_experiment can log preprocessing pipeline."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from unittest.mock import MagicMock, patch
        
        # Create and train a model
        np.random.seed(42)
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.rand(50, 5)
        y_test = np.random.randint(0, 2, 50)
        
        service = TrainingService(mlflow_uri="http://localhost:5000")
        model = service.train(X_train, y_train)
        metrics = service.evaluate(model, X_test, y_test)
        
        # Create a preprocessing pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])
        pipeline.fit(X_train)
        
        # Mock MLflow operations
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id_67890"
        
        with patch('mlflow.start_run', return_value=MagicMock(__enter__=lambda self: mock_run, __exit__=lambda *args: None)):
            with patch('mlflow.log_param'):
                with patch('mlflow.log_metric'):
                    with patch('mlflow.sklearn.log_model') as mock_log_model:
                        with patch('mlflow.log_artifact'):
                            with patch('mlflow.set_tag'):
                                # Log experiment with pipeline
                                params = {"n_estimators": 100}
                                run_id = service.log_experiment(model, metrics, params, preprocessing_pipeline=pipeline)
                                
                                # Verify run_id is returned
                                assert isinstance(run_id, str)
                                assert len(run_id) > 0
                                
                                # Verify log_model was called twice (model + pipeline)
                                assert mock_log_model.call_count == 2
    
    def test_log_experiment_with_empty_params(self, monkeypatch):
        """Test that log_experiment works with empty params dictionary."""
        from unittest.mock import MagicMock, patch
        
        # Create and train a model
        np.random.seed(42)
        X_train = np.random.rand(50, 5)
        y_train = np.random.randint(0, 2, 50)
        X_test = np.random.rand(20, 5)
        y_test = np.random.randint(0, 2, 20)
        
        service = TrainingService(mlflow_uri="http://localhost:5000")
        model = service.train(X_train, y_train)
        metrics = service.evaluate(model, X_test, y_test)
        
        # Mock MLflow operations
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id_empty"
        
        with patch('mlflow.start_run', return_value=MagicMock(__enter__=lambda self: mock_run, __exit__=lambda *args: None)):
            with patch('mlflow.log_param') as mock_log_param:
                with patch('mlflow.log_metric'):
                    with patch('mlflow.sklearn.log_model'):
                        with patch('mlflow.log_artifact'):
                            with patch('mlflow.set_tag'):
                                # Log experiment with empty params
                                params = {}
                                run_id = service.log_experiment(model, metrics, params)
                                
                                # Verify run_id is returned
                                assert isinstance(run_id, str)
                                assert len(run_id) > 0
                                
                                # Verify log_param was not called (empty params)
                                assert mock_log_param.call_count == 0
    
    def test_log_experiment_with_multiple_params(self, monkeypatch):
        """Test that log_experiment logs multiple hyperparameters."""
        from unittest.mock import MagicMock, patch
        
        # Create and train a model
        np.random.seed(42)
        X_train = np.random.rand(80, 8)
        y_train = np.random.randint(0, 2, 80)
        X_test = np.random.rand(20, 8)
        y_test = np.random.randint(0, 2, 20)
        
        config = TrainingConfig(n_estimators=50, max_depth=15, random_state=123)
        service = TrainingService(mlflow_uri="http://localhost:5000", config=config)
        model = service.train(X_train, y_train)
        metrics = service.evaluate(model, X_test, y_test)
        
        # Mock MLflow operations
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id_multi"
        
        with patch('mlflow.start_run', return_value=MagicMock(__enter__=lambda self: mock_run, __exit__=lambda *args: None)):
            with patch('mlflow.log_param') as mock_log_param:
                with patch('mlflow.log_metric') as mock_log_metric:
                    with patch('mlflow.sklearn.log_model'):
                        with patch('mlflow.log_artifact'):
                            with patch('mlflow.set_tag') as mock_set_tag:
                                # Log experiment with multiple params
                                params = {
                                    "n_estimators": 50,
                                    "max_depth": 15,
                                    "random_state": 123,
                                    "min_samples_split": 2,
                                    "min_samples_leaf": 1
                                }
                                
                                run_id = service.log_experiment(model, metrics, params)
                                
                                # Verify run_id is returned
                                assert isinstance(run_id, str)
                                assert len(run_id) > 0
                                
                                # Verify log_param was called 5 times (5 params)
                                assert mock_log_param.call_count == 5
                                
                                # Verify log_metric was called 5 times (accuracy, precision, recall, f1, roc_auc)
                                assert mock_log_metric.call_count == 5
                                
                                # Verify set_tag was called 4 times (timestamp, run_identifier, model_type, framework)
                                assert mock_set_tag.call_count == 4



class TestTrainingServiceRegisterModel:
    """Test TrainingService.register_model() method."""
    
    def test_register_model_returns_model_version(self, monkeypatch):
        """Test that register_model returns a ModelVersion object."""
        from unittest.mock import MagicMock, patch
        
        # Create service
        service = TrainingService(mlflow_uri="http://localhost:5000")
        
        # Mock MLflow operations
        mock_model_version = MagicMock()
        mock_model_version.version = 1
        mock_model_version.name = "test_model"
        
        mock_client = MagicMock()
        
        with patch('mlflow.register_model', return_value=mock_model_version):
            with patch('mlflow.tracking.MlflowClient', return_value=mock_client):
                # Register model
                run_id = "test_run_123"
                model_name = "test_model"
                
                result = service.register_model(run_id, model_name)
                
                # Verify ModelVersion is returned
                assert result is not None
                assert result.version == 1
                assert result.name == "test_model"
    
    def test_register_model_with_metrics(self, monkeypatch):
        """Test that register_model stores metrics as tags."""
        from unittest.mock import MagicMock, patch, call
        
        # Create service
        service = TrainingService(mlflow_uri="http://localhost:5000")
        
        # Create metrics
        conf_matrix = np.array([[10, 2], [3, 15]])
        feature_importance = {"feature_0": 0.3, "feature_1": 0.7}
        metrics = TrainingMetrics(
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            roc_auc=0.90,
            confusion_matrix=conf_matrix,
            feature_importance=feature_importance
        )
        
        # Mock MLflow operations
        mock_model_version = MagicMock()
        mock_model_version.version = 2
        mock_model_version.name = "churn_model"
        
        mock_client = MagicMock()
        
        with patch('mlflow.register_model', return_value=mock_model_version):
            with patch('mlflow.tracking.MlflowClient', return_value=mock_client):
                # Register model with metrics
                run_id = "test_run_456"
                model_name = "churn_model"
                
                result = service.register_model(run_id, model_name, metrics)
                
                # Verify set_model_version_tag was called for metrics
                tag_calls = mock_client.set_model_version_tag.call_args_list
                
                # Should have tags for: training_date, accuracy, f1_score, roc_auc, 
                # n_estimators, max_depth, random_state
                assert len(tag_calls) >= 7
                
                # Verify specific metric tags
                tag_keys = [call[1]['key'] for call in tag_calls]
                assert 'accuracy' in tag_keys
                assert 'f1_score' in tag_keys
                assert 'roc_auc' in tag_keys
                assert 'training_date' in tag_keys
    
    def test_register_model_stores_hyperparameters(self, monkeypatch):
        """Test that register_model stores hyperparameters as tags."""
        from unittest.mock import MagicMock, patch
        
        # Create service with custom config
        config = TrainingConfig(n_estimators=200, max_depth=15, random_state=999)
        service = TrainingService(mlflow_uri="http://localhost:5000", config=config)
        
        # Mock MLflow operations
        mock_model_version = MagicMock()
        mock_model_version.version = 3
        
        mock_client = MagicMock()
        
        with patch('mlflow.register_model', return_value=mock_model_version):
            with patch('mlflow.tracking.MlflowClient', return_value=mock_client):
                # Register model
                run_id = "test_run_789"
                model_name = "test_model"
                
                service.register_model(run_id, model_name)
                
                # Verify set_model_version_tag was called for hyperparameters
                tag_calls = mock_client.set_model_version_tag.call_args_list
                tag_dict = {call[1]['key']: call[1]['value'] for call in tag_calls}
                
                assert 'n_estimators' in tag_dict
                assert tag_dict['n_estimators'] == '200'
                assert 'max_depth' in tag_dict
                assert tag_dict['max_depth'] == '15'
                assert 'random_state' in tag_dict
                assert tag_dict['random_state'] == '999'
    
    def test_register_model_without_metrics(self, monkeypatch):
        """Test that register_model works without metrics."""
        from unittest.mock import MagicMock, patch
        
        # Create service
        service = TrainingService(mlflow_uri="http://localhost:5000")
        
        # Mock MLflow operations
        mock_model_version = MagicMock()
        mock_model_version.version = 1
        
        mock_client = MagicMock()
        
        with patch('mlflow.register_model', return_value=mock_model_version):
            with patch('mlflow.tracking.MlflowClient', return_value=mock_client):
                # Register model without metrics
                run_id = "test_run_no_metrics"
                model_name = "test_model"
                
                result = service.register_model(run_id, model_name, metrics=None)
                
                # Verify ModelVersion is returned
                assert result is not None
                assert result.version == 1
                
                # Verify tags were still set (training_date, hyperparameters)
                tag_calls = mock_client.set_model_version_tag.call_args_list
                assert len(tag_calls) >= 4  # training_date + 3 hyperparameters


class TestTrainingServicePromoteToProduction:
    """Test TrainingService.promote_to_production() method."""
    
    def test_promote_to_production_transitions_stage(self, monkeypatch):
        """Test that promote_to_production transitions model to Production stage."""
        from unittest.mock import MagicMock, patch
        
        # Create service
        service = TrainingService(mlflow_uri="http://localhost:5000")
        
        # Mock MLflow client
        mock_client = MagicMock()
        mock_client.search_model_versions.return_value = []  # No existing Production models
        
        with patch('mlflow.tracking.MlflowClient', return_value=mock_client):
            # Promote model
            model_name = "test_model"
            version = 2
            
            service.promote_to_production(model_name, version)
            
            # Verify transition_model_version_stage was called
            mock_client.transition_model_version_stage.assert_called_once_with(
                name=model_name,
                version=version,
                stage="Production"
            )
    
    def test_promote_to_production_archives_existing(self, monkeypatch):
        """Test that promote_to_production archives existing Production models."""
        from unittest.mock import MagicMock, patch, call
        
        # Create service
        service = TrainingService(mlflow_uri="http://localhost:5000")
        
        # Mock existing Production model
        mock_existing_version = MagicMock()
        mock_existing_version.version = 1
        mock_existing_version.current_stage = "Production"
        
        mock_client = MagicMock()
        mock_client.search_model_versions.return_value = [mock_existing_version]
        
        with patch('mlflow.tracking.MlflowClient', return_value=mock_client):
            # Promote new model
            model_name = "test_model"
            version = 2
            
            service.promote_to_production(model_name, version, archive_existing=True)
            
            # Verify existing model was archived
            calls = mock_client.transition_model_version_stage.call_args_list
            assert len(calls) == 2  # Archive old + promote new
            
            # First call should archive version 1
            assert calls[0] == call(name=model_name, version=1, stage="Archived")
            # Second call should promote version 2
            assert calls[1] == call(name=model_name, version=version, stage="Production")
    
    def test_promote_to_production_without_archiving(self, monkeypatch):
        """Test that promote_to_production can skip archiving."""
        from unittest.mock import MagicMock, patch
        
        # Create service
        service = TrainingService(mlflow_uri="http://localhost:5000")
        
        # Mock MLflow client
        mock_client = MagicMock()
        
        with patch('mlflow.tracking.MlflowClient', return_value=mock_client):
            # Promote model without archiving
            model_name = "test_model"
            version = 3
            
            service.promote_to_production(model_name, version, archive_existing=False)
            
            # Verify search_model_versions was not called (no archiving)
            mock_client.search_model_versions.assert_not_called()
            
            # Verify only promotion was called
            mock_client.transition_model_version_stage.assert_called_once_with(
                name=model_name,
                version=version,
                stage="Production"
            )
    
    def test_promote_to_production_with_multiple_existing(self, monkeypatch):
        """Test that promote_to_production archives multiple existing Production models."""
        from unittest.mock import MagicMock, patch
        
        # Create service
        service = TrainingService(mlflow_uri="http://localhost:5000")
        
        # Mock multiple existing Production models
        mock_version_1 = MagicMock()
        mock_version_1.version = 1
        mock_version_1.current_stage = "Production"
        
        mock_version_2 = MagicMock()
        mock_version_2.version = 2
        mock_version_2.current_stage = "Production"
        
        mock_client = MagicMock()
        mock_client.search_model_versions.return_value = [mock_version_1, mock_version_2]
        
        with patch('mlflow.tracking.MlflowClient', return_value=mock_client):
            # Promote new model
            model_name = "test_model"
            version = 3
            
            service.promote_to_production(model_name, version, archive_existing=True)
            
            # Verify both existing models were archived
            calls = mock_client.transition_model_version_stage.call_args_list
            assert len(calls) == 3  # Archive 2 old + promote 1 new


class TestTrainingServiceShouldPromoteModel:
    """Test TrainingService.should_promote_model() method."""
    
    def test_should_promote_when_no_production_model(self, monkeypatch):
        """Test that should_promote_model returns True when no Production model exists."""
        from unittest.mock import MagicMock, patch
        
        # Create service
        service = TrainingService(mlflow_uri="http://localhost:5000")
        
        # Mock MLflow client with no Production models
        mock_client = MagicMock()
        mock_client.search_model_versions.return_value = []
        
        with patch('mlflow.tracking.MlflowClient', return_value=mock_client):
            # Check promotion
            new_roc_auc = 0.85
            model_name = "test_model"
            
            result = service.should_promote_model(new_roc_auc, model_name)
            
            # Should recommend promotion when no Production model exists
            assert result is True
    
    def test_should_promote_when_improvement_exceeds_threshold(self, monkeypatch):
        """Test that should_promote_model returns True when improvement >= 0.01."""
        from unittest.mock import MagicMock, patch
        
        # Create service
        service = TrainingService(mlflow_uri="http://localhost:5000")
        
        # Mock existing Production model with ROC-AUC = 0.80
        mock_tag = MagicMock()
        mock_tag.key = "roc_auc"
        mock_tag.value = "0.80"
        
        mock_version = MagicMock()
        mock_version.version = 1
        mock_version.current_stage = "Production"
        mock_version.tags = [mock_tag]
        
        mock_client = MagicMock()
        mock_client.search_model_versions.return_value = [mock_version]
        
        with patch('mlflow.tracking.MlflowClient', return_value=mock_client):
            # Check promotion with new ROC-AUC = 0.81 (improvement = 0.01)
            new_roc_auc = 0.81
            model_name = "test_model"
            
            result = service.should_promote_model(new_roc_auc, model_name)
            
            # Should recommend promotion (0.81 >= 0.80 + 0.01)
            assert result is True
    
    def test_should_not_promote_when_improvement_below_threshold(self, monkeypatch):
        """Test that should_promote_model returns False when improvement < 0.01."""
        from unittest.mock import MagicMock, patch
        
        # Create service
        service = TrainingService(mlflow_uri="http://localhost:5000")
        
        # Mock existing Production model with ROC-AUC = 0.80
        mock_tag = MagicMock()
        mock_tag.key = "roc_auc"
        mock_tag.value = "0.80"
        
        mock_version = MagicMock()
        mock_version.version = 1
        mock_version.current_stage = "Production"
        mock_version.tags = [mock_tag]
        
        mock_client = MagicMock()
        mock_client.search_model_versions.return_value = [mock_version]
        
        with patch('mlflow.tracking.MlflowClient', return_value=mock_client):
            # Check promotion with new ROC-AUC = 0.805 (improvement = 0.005)
            new_roc_auc = 0.805
            model_name = "test_model"
            
            result = service.should_promote_model(new_roc_auc, model_name)
            
            # Should not recommend promotion (0.805 < 0.80 + 0.01)
            assert result is False
    
    def test_should_promote_when_significant_improvement(self, monkeypatch):
        """Test that should_promote_model returns True for significant improvements."""
        from unittest.mock import MagicMock, patch
        
        # Create service
        service = TrainingService(mlflow_uri="http://localhost:5000")
        
        # Mock existing Production model with ROC-AUC = 0.75
        mock_tag = MagicMock()
        mock_tag.key = "roc_auc"
        mock_tag.value = "0.75"
        
        mock_version = MagicMock()
        mock_version.version = 1
        mock_version.current_stage = "Production"
        mock_version.tags = [mock_tag]
        
        mock_client = MagicMock()
        mock_client.search_model_versions.return_value = [mock_version]
        
        with patch('mlflow.tracking.MlflowClient', return_value=mock_client):
            # Check promotion with new ROC-AUC = 0.85 (improvement = 0.10)
            new_roc_auc = 0.85
            model_name = "test_model"
            
            result = service.should_promote_model(new_roc_auc, model_name)
            
            # Should recommend promotion (0.85 >= 0.75 + 0.01)
            assert result is True
    
    def test_should_promote_when_production_model_has_no_roc_auc_tag(self, monkeypatch):
        """Test that should_promote_model returns True when Production model has no roc_auc tag."""
        from unittest.mock import MagicMock, patch
        
        # Create service
        service = TrainingService(mlflow_uri="http://localhost:5000")
        
        # Mock existing Production model without roc_auc tag
        mock_version = MagicMock()
        mock_version.version = 1
        mock_version.current_stage = "Production"
        mock_version.tags = []  # No tags
        
        mock_client = MagicMock()
        mock_client.search_model_versions.return_value = [mock_version]
        
        with patch('mlflow.tracking.MlflowClient', return_value=mock_client):
            # Check promotion
            new_roc_auc = 0.85
            model_name = "test_model"
            
            result = service.should_promote_model(new_roc_auc, model_name)
            
            # Should recommend promotion when current model has no metrics
            assert result is True
    
    def test_should_not_promote_when_performance_degrades(self, monkeypatch):
        """Test that should_promote_model returns False when new model is worse."""
        from unittest.mock import MagicMock, patch
        
        # Create service
        service = TrainingService(mlflow_uri="http://localhost:5000")
        
        # Mock existing Production model with ROC-AUC = 0.90
        mock_tag = MagicMock()
        mock_tag.key = "roc_auc"
        mock_tag.value = "0.90"
        
        mock_version = MagicMock()
        mock_version.version = 1
        mock_version.current_stage = "Production"
        mock_version.tags = [mock_tag]
        
        mock_client = MagicMock()
        mock_client.search_model_versions.return_value = [mock_version]
        
        with patch('mlflow.tracking.MlflowClient', return_value=mock_client):
            # Check promotion with new ROC-AUC = 0.85 (worse than current)
            new_roc_auc = 0.85
            model_name = "test_model"
            
            result = service.should_promote_model(new_roc_auc, model_name)
            
            # Should not recommend promotion (0.85 < 0.90 + 0.01)
            assert result is False
    
    def test_should_promote_at_exact_threshold(self, monkeypatch):
        """Test that should_promote_model returns True at exact threshold."""
        from unittest.mock import MagicMock, patch
        
        # Create service
        service = TrainingService(mlflow_uri="http://localhost:5000")
        
        # Mock existing Production model with ROC-AUC = 0.8500
        mock_tag = MagicMock()
        mock_tag.key = "roc_auc"
        mock_tag.value = "0.8500"
        
        mock_version = MagicMock()
        mock_version.version = 1
        mock_version.current_stage = "Production"
        mock_version.tags = [mock_tag]
        
        mock_client = MagicMock()
        mock_client.search_model_versions.return_value = [mock_version]
        
        with patch('mlflow.tracking.MlflowClient', return_value=mock_client):
            # Check promotion with new ROC-AUC = 0.8600 (exactly 0.01 improvement)
            new_roc_auc = 0.8600
            model_name = "test_model"
            
            result = service.should_promote_model(new_roc_auc, model_name)
            
            # Should recommend promotion (0.8600 >= 0.8500 + 0.01)
            assert result is True


class TestTrainingServiceModelPersistence:
    """Test TrainingService model persistence methods (save_model and load_model)."""
    
    def test_save_model_creates_file_with_version(self, tmp_path):
        """Test that save_model creates a file with version-based naming."""
        # Create and train a model
        np.random.seed(42)
        X_train = np.random.rand(50, 5)
        y_train = np.random.randint(0, 2, 50)
        
        service = TrainingService(mlflow_uri="http://localhost:5000")
        model = service.train(X_train, y_train)
        
        # Save model to temporary directory
        models_dir = str(tmp_path / "models")
        model_path = service.save_model(model, "churn_model", version=1, models_dir=models_dir)
        
        # Verify file was created with correct naming convention
        import os
        assert os.path.exists(model_path)
        assert model_path.endswith("churn_model_v1.pkl")
        assert "models" in model_path
        
        # Verify file is not empty
        assert os.path.getsize(model_path) > 0
    
    def test_save_model_with_different_versions(self, tmp_path):
        """Test saving multiple versions of the same model."""
        # Create and train a model
        np.random.seed(42)
        X_train = np.random.rand(50, 5)
        y_train = np.random.randint(0, 2, 50)
        
        service = TrainingService(mlflow_uri="http://localhost:5000")
        model = service.train(X_train, y_train)
        
        # Save multiple versions
        models_dir = str(tmp_path / "models")
        path_v1 = service.save_model(model, "churn_model", version=1, models_dir=models_dir)
        path_v2 = service.save_model(model, "churn_model", version=2, models_dir=models_dir)
        path_v3 = service.save_model(model, "churn_model", version=3, models_dir=models_dir)
        
        # Verify all versions exist
        import os
        assert os.path.exists(path_v1)
        assert os.path.exists(path_v2)
        assert os.path.exists(path_v3)
        
        # Verify correct naming
        assert path_v1.endswith("churn_model_v1.pkl")
        assert path_v2.endswith("churn_model_v2.pkl")
        assert path_v3.endswith("churn_model_v3.pkl")
    
    def test_save_model_creates_directory_if_not_exists(self, tmp_path):
        """Test that save_model creates the models directory if it doesn't exist."""
        # Create and train a model
        np.random.seed(42)
        X_train = np.random.rand(50, 5)
        y_train = np.random.randint(0, 2, 50)
        
        service = TrainingService(mlflow_uri="http://localhost:5000")
        model = service.train(X_train, y_train)
        
        # Save to non-existent directory
        models_dir = str(tmp_path / "new_models_dir")
        import os
        assert not os.path.exists(models_dir)
        
        model_path = service.save_model(model, "test_model", version=1, models_dir=models_dir)
        
        # Verify directory was created
        assert os.path.exists(models_dir)
        assert os.path.exists(model_path)
    
    def test_save_model_returns_full_path(self, tmp_path):
        """Test that save_model returns the full path to the saved file."""
        # Create and train a model
        np.random.seed(42)
        X_train = np.random.rand(50, 5)
        y_train = np.random.randint(0, 2, 50)
        
        service = TrainingService(mlflow_uri="http://localhost:5000")
        model = service.train(X_train, y_train)
        
        # Save model
        models_dir = str(tmp_path / "models")
        model_path = service.save_model(model, "my_model", version=5, models_dir=models_dir)
        
        # Verify returned path is correct
        import os
        assert isinstance(model_path, str)
        assert os.path.isabs(model_path) or models_dir in model_path
        assert "my_model_v5.pkl" in model_path
    
    def test_load_model_returns_fitted_model(self, tmp_path):
        """Test that load_model returns a fitted RandomForestClassifier."""
        # Create, train, and save a model
        np.random.seed(42)
        X_train = np.random.rand(50, 5)
        y_train = np.random.randint(0, 2, 50)
        
        service = TrainingService(mlflow_uri="http://localhost:5000")
        original_model = service.train(X_train, y_train)
        
        models_dir = str(tmp_path / "models")
        service.save_model(original_model, "churn_model", version=1, models_dir=models_dir)
        
        # Load the model
        loaded_model = service.load_model("churn_model", version=1, models_dir=models_dir)
        
        # Verify loaded model is a fitted RandomForestClassifier
        assert isinstance(loaded_model, RandomForestClassifier)
        assert hasattr(loaded_model, 'estimators_')
        
        # Verify model can make predictions
        predictions = loaded_model.predict(X_train)
        assert len(predictions) == len(y_train)
    
    def test_load_model_preserves_model_properties(self, tmp_path):
        """Test that load_model preserves model hyperparameters and properties."""
        # Create and train a model with specific hyperparameters
        np.random.seed(42)
        X_train = np.random.rand(50, 5)
        y_train = np.random.randint(0, 2, 50)
        
        config = TrainingConfig(n_estimators=25, max_depth=8, random_state=99)
        service = TrainingService(mlflow_uri="http://localhost:5000", config=config)
        original_model = service.train(X_train, y_train)
        
        # Save and load model
        models_dir = str(tmp_path / "models")
        service.save_model(original_model, "test_model", version=1, models_dir=models_dir)
        loaded_model = service.load_model("test_model", version=1, models_dir=models_dir)
        
        # Verify hyperparameters are preserved
        assert loaded_model.n_estimators == 25
        assert loaded_model.max_depth == 8
        assert loaded_model.random_state == 99
        assert loaded_model.n_features_in_ == 5
    
    def test_load_model_produces_same_predictions(self, tmp_path):
        """Test that loaded model produces same predictions as original."""
        # Create, train, and save a model
        np.random.seed(42)
        X_train = np.random.rand(50, 5)
        y_train = np.random.randint(0, 2, 50)
        X_test = np.random.rand(20, 5)
        
        service = TrainingService(mlflow_uri="http://localhost:5000")
        original_model = service.train(X_train, y_train)
        
        # Get predictions from original model
        original_predictions = original_model.predict(X_test)
        original_probabilities = original_model.predict_proba(X_test)
        
        # Save and load model
        models_dir = str(tmp_path / "models")
        service.save_model(original_model, "churn_model", version=1, models_dir=models_dir)
        loaded_model = service.load_model("churn_model", version=1, models_dir=models_dir)
        
        # Get predictions from loaded model
        loaded_predictions = loaded_model.predict(X_test)
        loaded_probabilities = loaded_model.predict_proba(X_test)
        
        # Verify predictions are identical
        assert np.array_equal(original_predictions, loaded_predictions)
        assert np.allclose(original_probabilities, loaded_probabilities)
    
    def test_load_model_with_nonexistent_file_raises_error(self, tmp_path):
        """Test that load_model raises TrainingError when file doesn't exist."""
        service = TrainingService(mlflow_uri="http://localhost:5000")
        
        models_dir = str(tmp_path / "models")
        
        with pytest.raises(TrainingError) as exc_info:
            service.load_model("nonexistent_model", version=1, models_dir=models_dir)
        
        assert "not found" in str(exc_info.value).lower()
    
    def test_load_model_with_different_versions(self, tmp_path):
        """Test loading different versions of the same model."""
        # Create and train models with different configurations
        np.random.seed(42)
        X_train = np.random.rand(50, 5)
        y_train = np.random.randint(0, 2, 50)
        
        # Train model v1 with 10 estimators
        config_v1 = TrainingConfig(n_estimators=10, random_state=42)
        service_v1 = TrainingService(mlflow_uri="http://localhost:5000", config=config_v1)
        model_v1 = service_v1.train(X_train, y_train)
        
        # Train model v2 with 20 estimators
        config_v2 = TrainingConfig(n_estimators=20, random_state=42)
        service_v2 = TrainingService(mlflow_uri="http://localhost:5000", config=config_v2)
        model_v2 = service_v2.train(X_train, y_train)
        
        # Save both versions
        models_dir = str(tmp_path / "models")
        service_v1.save_model(model_v1, "churn_model", version=1, models_dir=models_dir)
        service_v2.save_model(model_v2, "churn_model", version=2, models_dir=models_dir)
        
        # Load both versions
        service = TrainingService(mlflow_uri="http://localhost:5000")
        loaded_v1 = service.load_model("churn_model", version=1, models_dir=models_dir)
        loaded_v2 = service.load_model("churn_model", version=2, models_dir=models_dir)
        
        # Verify correct versions were loaded
        assert loaded_v1.n_estimators == 10
        assert loaded_v2.n_estimators == 20
    
    def test_save_and_load_model_round_trip(self, tmp_path):
        """Test complete round-trip: train -> save -> load -> predict."""
        # Create and train a model
        np.random.seed(42)
        X_train = np.random.rand(100, 8)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.rand(30, 8)
        
        config = TrainingConfig(n_estimators=15, max_depth=10, random_state=42)
        service = TrainingService(mlflow_uri="http://localhost:5000", config=config)
        model = service.train(X_train, y_train)
        
        # Save model
        models_dir = str(tmp_path / "models")
        model_path = service.save_model(model, "churn_model", version=1, models_dir=models_dir)
        
        # Verify file exists
        import os
        assert os.path.exists(model_path)
        
        # Load model
        loaded_model = service.load_model("churn_model", version=1, models_dir=models_dir)
        
        # Make predictions with loaded model
        predictions = loaded_model.predict(X_test)
        probabilities = loaded_model.predict_proba(X_test)
        
        # Verify predictions are valid
        assert len(predictions) == len(X_test)
        assert all(pred in [0, 1] for pred in predictions)
        assert probabilities.shape == (len(X_test), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_model_persistence_with_default_models_dir(self, tmp_path, monkeypatch):
        """Test model persistence using default 'models' directory."""
        # Change to temporary directory
        monkeypatch.chdir(tmp_path)
        
        # Create and train a model
        np.random.seed(42)
        X_train = np.random.rand(50, 5)
        y_train = np.random.randint(0, 2, 50)
        
        service = TrainingService(mlflow_uri="http://localhost:5000")
        model = service.train(X_train, y_train)
        
        # Save model using default directory
        model_path = service.save_model(model, "churn_model", version=1)
        
        # Verify file was created in default 'models' directory
        import os
        assert os.path.exists(model_path)
        assert "models" in model_path
        assert model_path.endswith("churn_model_v1.pkl")
        
        # Load model using default directory
        loaded_model = service.load_model("churn_model", version=1)
        
        # Verify model was loaded successfully
        assert isinstance(loaded_model, RandomForestClassifier)
        assert hasattr(loaded_model, 'estimators_')
