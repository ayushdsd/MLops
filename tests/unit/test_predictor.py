"""
Unit tests for the Predictor module.

Tests cover model loading, preprocessing, prediction generation,
and risk classification functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.api.predictor import (
    Predictor,
    PredictorError,
    ModelNotLoadedError,
    PreprocessingError,
    PredictionResult
)


@pytest.fixture
def predictor():
    """Create a Predictor instance for testing."""
    with patch('mlflow.set_tracking_uri'):
        return Predictor(mlflow_uri="http://localhost:5000")


@pytest.fixture
def sample_customer_data():
    """Sample customer data for testing."""
    return {
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


class TestPredictorInitialization:
    """Tests for Predictor initialization."""
    
    def test_predictor_initialization(self):
        """Test that predictor initializes correctly."""
        with patch('mlflow.set_tracking_uri') as mock_set_uri:
            predictor = Predictor(mlflow_uri="http://localhost:5000")
            
            assert predictor.mlflow_uri == "http://localhost:5000"
            assert predictor.model is None
            assert predictor.preprocessing_pipeline is None
            assert predictor.model_version is None
            assert predictor.model_name is None
            mock_set_uri.assert_called_once_with("http://localhost:5000")
    
    def test_is_ready_returns_false_initially(self, predictor):
        """Test that is_ready returns False when model not loaded."""
        assert predictor.is_ready() is False


class TestModelLoading:
    """Tests for model loading functionality."""
    
    def test_load_model_success(self, predictor):
        """Test successful model loading from MLflow."""
        mock_model = Mock()
        mock_client = Mock()
        mock_version = Mock()
        mock_version.version = "3"
        mock_client.get_latest_versions.return_value = [mock_version]
        
        with patch('mlflow.sklearn.load_model', return_value=mock_model):
            with patch('mlflow.tracking.MlflowClient', return_value=mock_client):
                predictor.load_model(model_name="churn_model", stage="Production")
        
        assert predictor.model == mock_model
        assert predictor.model_name == "churn_model"
        assert predictor.model_version == "3"
        assert predictor.is_ready() is True
    
    def test_load_model_with_staging_stage(self, predictor):
        """Test loading model from Staging stage."""
        mock_model = Mock()
        mock_client = Mock()
        mock_version = Mock()
        mock_version.version = "2"
        mock_client.get_latest_versions.return_value = [mock_version]
        
        with patch('mlflow.sklearn.load_model', return_value=mock_model) as mock_load:
            with patch('mlflow.tracking.MlflowClient', return_value=mock_client):
                predictor.load_model(model_name="churn_model", stage="Staging")
        
        mock_load.assert_called_once_with("models:/churn_model/Staging")
        assert predictor.model_version == "2"
    
    def test_load_model_failure(self, predictor):
        """Test model loading failure handling."""
        with patch('mlflow.sklearn.load_model', side_effect=Exception("MLflow error")):
            with pytest.raises(PredictorError) as exc_info:
                predictor.load_model(model_name="churn_model", stage="Production")
            
            assert "Failed to load model" in str(exc_info.value)
            assert predictor.model is None
            assert predictor.is_ready() is False
    
    def test_load_model_no_version_info(self, predictor):
        """Test model loading when version info is unavailable."""
        mock_model = Mock()
        mock_client = Mock()
        mock_client.get_latest_versions.return_value = []
        
        with patch('mlflow.sklearn.load_model', return_value=mock_model):
            with patch('mlflow.tracking.MlflowClient', return_value=mock_client):
                predictor.load_model(model_name="churn_model", stage="Production")
        
        assert predictor.model == mock_model
        assert predictor.model_version == "unknown"


class TestPreprocessingPipelineLoading:
    """Tests for preprocessing pipeline loading."""
    
    def test_load_pipeline_from_mlflow_run(self, predictor):
        """Test loading preprocessing pipeline from MLflow run."""
        mock_pipeline = Mock()
        
        with patch('mlflow.sklearn.load_model', return_value=mock_pipeline) as mock_load:
            predictor.load_preprocessing_pipeline(run_id="abc123")
        
        mock_load.assert_called_once_with("runs:/abc123/preprocessing_pipeline")
        assert predictor.preprocessing_pipeline == mock_pipeline
    
    def test_load_pipeline_from_local_file(self, predictor):
        """Test loading preprocessing pipeline from local file."""
        mock_pipeline = Mock()
        
        with patch('joblib.load', return_value=mock_pipeline) as mock_load:
            predictor.load_preprocessing_pipeline(pipeline_path="models/pipelines/preprocessing.pkl")
        
        mock_load.assert_called_once_with("models/pipelines/preprocessing.pkl")
        assert predictor.preprocessing_pipeline == mock_pipeline
    
    def test_load_pipeline_no_source_provided(self, predictor):
        """Test that ValueError is raised when no source is provided."""
        with pytest.raises(ValueError) as exc_info:
            predictor.load_preprocessing_pipeline()
        
        assert "Either run_id or pipeline_path must be provided" in str(exc_info.value)
    
    def test_load_pipeline_failure(self, predictor):
        """Test preprocessing pipeline loading failure."""
        with patch('mlflow.sklearn.load_model', side_effect=Exception("Load error")):
            with pytest.raises(PredictorError) as exc_info:
                predictor.load_preprocessing_pipeline(run_id="abc123")
            
            assert "Failed to load preprocessing pipeline" in str(exc_info.value)


class TestRiskClassification:
    """Tests for risk classification functionality."""
    
    def test_classify_risk_low(self):
        """Test risk classification for low probability."""
        assert Predictor.classify_risk(0.0) == "Low"
        assert Predictor.classify_risk(0.15) == "Low"
        assert Predictor.classify_risk(0.32) == "Low"
    
    def test_classify_risk_medium(self):
        """Test risk classification for medium probability."""
        assert Predictor.classify_risk(0.33) == "Medium"
        assert Predictor.classify_risk(0.50) == "Medium"
        assert Predictor.classify_risk(0.65) == "Medium"
    
    def test_classify_risk_high(self):
        """Test risk classification for high probability."""
        assert Predictor.classify_risk(0.66) == "High"
        assert Predictor.classify_risk(0.85) == "High"
        assert Predictor.classify_risk(1.0) == "High"
    
    def test_classify_risk_boundary_values(self):
        """Test risk classification at boundary values."""
        # Test exact boundaries
        assert Predictor.classify_risk(0.33) == "Medium"
        assert Predictor.classify_risk(0.66) == "High"
        
        # Test just below boundaries
        assert Predictor.classify_risk(0.3299999) == "Low"
        assert Predictor.classify_risk(0.6599999) == "Medium"


class TestPrediction:
    """Tests for prediction functionality."""
    
    def test_predict_without_model_raises_error(self, predictor, sample_customer_data):
        """Test that prediction fails when model is not loaded."""
        with pytest.raises(ModelNotLoadedError) as exc_info:
            predictor.predict(sample_customer_data)
        
        assert "Model not loaded" in str(exc_info.value)
    
    def test_predict_success_with_low_risk(self, predictor, sample_customer_data):
        """Test successful prediction with low churn probability."""
        # Setup mock model
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.85, 0.15]])  # 15% churn probability
        predictor.model = mock_model
        
        # Mock preprocessing
        with patch.object(predictor, '_preprocess_input', return_value=np.array([[1, 2, 3]])):
            result = predictor.predict(sample_customer_data)
        
        assert isinstance(result, PredictionResult)
        assert result.churn_probability == 0.15
        assert result.risk_label == "Low"
    
    def test_predict_success_with_medium_risk(self, predictor, sample_customer_data):
        """Test successful prediction with medium churn probability."""
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.5, 0.5]])  # 50% churn probability
        predictor.model = mock_model
        
        with patch.object(predictor, '_preprocess_input', return_value=np.array([[1, 2, 3]])):
            result = predictor.predict(sample_customer_data)
        
        assert result.churn_probability == 0.5
        assert result.risk_label == "Medium"
    
    def test_predict_success_with_high_risk(self, predictor, sample_customer_data):
        """Test successful prediction with high churn probability."""
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.27, 0.73]])  # 73% churn probability
        predictor.model = mock_model
        
        with patch.object(predictor, '_preprocess_input', return_value=np.array([[1, 2, 3]])):
            result = predictor.predict(sample_customer_data)
        
        assert result.churn_probability == 0.73
        assert result.risk_label == "High"
    
    def test_predict_with_preprocessing_pipeline(self, predictor, sample_customer_data):
        """Test prediction with preprocessing pipeline applied."""
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.6, 0.4]])
        predictor.model = mock_model
        
        mock_pipeline = Mock()
        mock_pipeline.transform.return_value = np.array([[1, 2, 3, 4, 5]])
        predictor.preprocessing_pipeline = mock_pipeline
        
        result = predictor.predict(sample_customer_data)
        
        # Verify pipeline was called
        mock_pipeline.transform.assert_called_once()
        
        # Verify prediction was made with preprocessed data
        mock_model.predict_proba.assert_called_once()
        
        assert result.churn_probability == 0.4
        assert result.risk_label == "Medium"
    
    def test_predict_preprocessing_failure(self, predictor, sample_customer_data):
        """Test prediction when preprocessing fails."""
        predictor.model = Mock()
        
        # Simulate preprocessing error by raising PreprocessingError directly
        with patch.object(predictor, '_preprocess_input', side_effect=PreprocessingError("Preprocessing failed: test error")):
            with pytest.raises(PreprocessingError) as exc_info:
                predictor.predict(sample_customer_data)
            
            assert "Preprocessing failed" in str(exc_info.value)
    
    def test_predict_model_failure(self, predictor, sample_customer_data):
        """Test prediction when model prediction fails."""
        mock_model = Mock()
        mock_model.predict_proba.side_effect = Exception("Model error")
        predictor.model = mock_model
        
        with patch.object(predictor, '_preprocess_input', return_value=np.array([[1, 2, 3]])):
            with pytest.raises(PredictorError) as exc_info:
                predictor.predict(sample_customer_data)
            
            assert "Prediction failed" in str(exc_info.value)


class TestModelInfo:
    """Tests for model information retrieval."""
    
    def test_get_model_info_no_model_loaded(self, predictor):
        """Test getting model info when no model is loaded."""
        info = predictor.get_model_info()
        
        assert info["model_name"] is None
        assert info["model_version"] is None
        assert info["model_loaded"] is False
        assert info["preprocessing_pipeline_loaded"] is False
    
    def test_get_model_info_with_model_loaded(self, predictor):
        """Test getting model info when model is loaded."""
        predictor.model = Mock()
        predictor.model_name = "churn_model"
        predictor.model_version = "3"
        
        info = predictor.get_model_info()
        
        assert info["model_name"] == "churn_model"
        assert info["model_version"] == "3"
        assert info["model_loaded"] is True
        assert info["preprocessing_pipeline_loaded"] is False
    
    def test_get_model_info_with_pipeline_loaded(self, predictor):
        """Test getting model info when both model and pipeline are loaded."""
        predictor.model = Mock()
        predictor.model_name = "churn_model"
        predictor.model_version = "3"
        predictor.preprocessing_pipeline = Mock()
        
        info = predictor.get_model_info()
        
        assert info["model_name"] == "churn_model"
        assert info["model_version"] == "3"
        assert info["model_loaded"] is True
        assert info["preprocessing_pipeline_loaded"] is True


class TestPreprocessInput:
    """Tests for input preprocessing."""
    
    def test_preprocess_without_pipeline(self, predictor, sample_customer_data):
        """Test preprocessing without a pipeline loaded."""
        with patch('pandas.DataFrame') as mock_df_class:
            mock_df = Mock()
            mock_df.values = np.array([[1, 2, 3]])
            mock_df_class.return_value = mock_df
            
            result = predictor._preprocess_input(sample_customer_data)
        
        assert isinstance(result, np.ndarray)
    
    def test_preprocess_with_pipeline(self, predictor, sample_customer_data):
        """Test preprocessing with a pipeline loaded."""
        mock_pipeline = Mock()
        mock_pipeline.transform.return_value = np.array([[1, 2, 3, 4, 5]])
        predictor.preprocessing_pipeline = mock_pipeline
        
        result = predictor._preprocess_input(sample_customer_data)
        
        mock_pipeline.transform.assert_called_once()
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 5)
    
    def test_preprocess_failure(self, predictor, sample_customer_data):
        """Test preprocessing failure handling."""
        mock_pipeline = Mock()
        mock_pipeline.transform.side_effect = Exception("Transform error")
        predictor.preprocessing_pipeline = mock_pipeline
        
        with pytest.raises(PreprocessingError) as exc_info:
            predictor._preprocess_input(sample_customer_data)
        
        assert "Preprocessing failed" in str(exc_info.value)
