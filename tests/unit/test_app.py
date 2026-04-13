"""
Unit tests for the FastAPI application.

Tests all API endpoints including predict, health, and model-info,
as well as error handling and middleware functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.predictor import PredictionResult, ModelNotLoadedError, PredictorError


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_predictor():
    """Create a mock predictor instance."""
    predictor = Mock()
    predictor.is_ready.return_value = True
    predictor.get_model_info.return_value = {
        "model_name": "churn_model",
        "model_version": "3",
        "model_loaded": True,
        "preprocessing_pipeline_loaded": True
    }
    return predictor


@pytest.fixture
def valid_customer_data():
    """Create valid customer data for testing."""
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
        "streaming_movies": "Yes",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 70.35,
        "total_charges": 844.20
    }


class TestHealthEndpoint:
    """Tests for the /health endpoint."""
    
    def test_health_endpoint_returns_200_when_healthy(self, client, mock_predictor):
        """
        Test health endpoint returns 200 when healthy.
        
        Validates: Requirements 10.1
        """
        with patch('src.api.app.predictor', mock_predictor):
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True
            assert data["model_version"] == "3"
    
    def test_health_endpoint_returns_503_when_model_not_loaded(self, client):
        """
        Test health endpoint returns 503 when model not loaded.
        
        Validates: Requirements 10.2
        """
        mock_predictor = Mock()
        mock_predictor.is_ready.return_value = False
        mock_predictor.get_model_info.return_value = {
            "model_name": "churn_model",
            "model_version": None,
            "model_loaded": False,
            "preprocessing_pipeline_loaded": False
        }
        
        with patch('src.api.app.predictor', mock_predictor):
            response = client.get("/health")
            
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["model_loaded"] is False
            assert data["model_version"] is None
    
    def test_health_endpoint_returns_503_when_predictor_none(self, client):
        """
        Test health endpoint returns 503 when predictor is None.
        
        Validates: Requirements 10.2
        """
        with patch('src.api.app.predictor', None):
            response = client.get("/health")
            
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["model_loaded"] is False


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""
    
    def test_predict_endpoint_with_valid_input(self, client, mock_predictor, valid_customer_data):
        """
        Test predict endpoint with valid input.
        
        Validates: Requirements 6.1, 6.2
        """
        # Mock prediction result
        mock_predictor.predict.return_value = PredictionResult(
            churn_probability=0.75,
            risk_label="High"
        )
        
        with patch('src.api.app.predictor', mock_predictor):
            response = client.post("/predict", json=valid_customer_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "churn_probability" in data
            assert "risk_label" in data
            assert "model_version" in data
            assert "timestamp" in data
            assert data["churn_probability"] == 0.75
            assert data["risk_label"] == "High"
            assert data["model_version"] == "3"
    
    def test_predict_endpoint_with_missing_fields(self, client, mock_predictor):
        """
        Test predict endpoint with missing fields.
        
        Validates: Requirements 6.3, 15.1
        """
        incomplete_data = {
            "gender": "Female",
            "senior_citizen": 0
            # Missing many required fields
        }
        
        with patch('src.api.app.predictor', mock_predictor):
            response = client.post("/predict", json=incomplete_data)
            
            assert response.status_code == 400
            data = response.json()
            assert "error" in data
            assert data["error"] == "Validation Error"
    
    def test_predict_endpoint_with_invalid_types(self, client, mock_predictor, valid_customer_data):
        """
        Test predict endpoint with invalid types.
        
        Validates: Requirements 6.3, 15.2
        """
        invalid_data = valid_customer_data.copy()
        invalid_data["tenure"] = "not_a_number"  # Should be int
        
        with patch('src.api.app.predictor', mock_predictor):
            response = client.post("/predict", json=invalid_data)
            
            assert response.status_code == 400
            data = response.json()
            assert "error" in data
    
    def test_predict_endpoint_with_invalid_categorical_values(self, client, mock_predictor, valid_customer_data):
        """
        Test predict endpoint with invalid categorical values.
        
        Validates: Requirements 6.3, 15.3
        """
        invalid_data = valid_customer_data.copy()
        invalid_data["gender"] = "Other"  # Invalid value
        
        with patch('src.api.app.predictor', mock_predictor):
            response = client.post("/predict", json=invalid_data)
            
            assert response.status_code == 400
            data = response.json()
            assert "error" in data
    
    def test_predict_endpoint_with_negative_tenure(self, client, mock_predictor, valid_customer_data):
        """
        Test predict endpoint with negative tenure.
        
        Validates: Requirements 15.2
        """
        invalid_data = valid_customer_data.copy()
        invalid_data["tenure"] = -5
        
        with patch('src.api.app.predictor', mock_predictor):
            response = client.post("/predict", json=invalid_data)
            
            assert response.status_code == 400
            data = response.json()
            assert "error" in data
    
    def test_predict_endpoint_with_zero_monthly_charges(self, client, mock_predictor, valid_customer_data):
        """
        Test predict endpoint with zero monthly charges.
        
        Validates: Requirements 15.2
        """
        invalid_data = valid_customer_data.copy()
        invalid_data["monthly_charges"] = 0.0  # Should be > 0
        
        with patch('src.api.app.predictor', mock_predictor):
            response = client.post("/predict", json=invalid_data)
            
            assert response.status_code == 400
            data = response.json()
            assert "error" in data
    
    def test_predict_endpoint_when_model_not_loaded(self, client, valid_customer_data):
        """
        Test predict endpoint when model not loaded.
        
        Validates: Requirements 10.3
        """
        mock_predictor = Mock()
        mock_predictor.predict.side_effect = ModelNotLoadedError("Model not loaded")
        
        with patch('src.api.app.predictor', mock_predictor):
            response = client.post("/predict", json=valid_customer_data)
            
            assert response.status_code == 503
            data = response.json()
            assert "error" in data
            assert data["error"] == "Service Unavailable"
    
    def test_predict_endpoint_when_predictor_none(self, client, valid_customer_data):
        """
        Test predict endpoint when predictor is None.
        
        Validates: Requirements 10.3
        """
        with patch('src.api.app.predictor', None):
            response = client.post("/predict", json=valid_customer_data)
            
            assert response.status_code == 503
            data = response.json()
            assert "error" in data
    
    def test_predict_endpoint_with_predictor_error(self, client, mock_predictor, valid_customer_data):
        """
        Test predict endpoint with predictor error.
        
        Validates: Requirements 14.2
        """
        mock_predictor.predict.side_effect = PredictorError("Prediction failed")
        
        with patch('src.api.app.predictor', mock_predictor):
            response = client.post("/predict", json=valid_customer_data)
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert data["error"] == "Prediction Error"


class TestModelInfoEndpoint:
    """Tests for the /model-info endpoint."""
    
    def test_model_info_endpoint(self, client, mock_predictor):
        """
        Test model-info endpoint.
        
        Validates: Requirements 6.5
        """
        with patch('src.api.app.predictor', mock_predictor):
            response = client.get("/model-info")
            
            assert response.status_code == 200
            data = response.json()
            assert "model_name" in data
            assert "model_version" in data
            assert "model_loaded" in data
            assert data["model_name"] == "churn_model"
            assert data["model_version"] == "3"
            assert data["model_loaded"] is True
    
    def test_model_info_endpoint_when_predictor_none(self, client):
        """
        Test model-info endpoint when predictor is None.
        
        Validates: Requirements 6.5
        """
        with patch('src.api.app.predictor', None):
            response = client.get("/model-info")
            
            assert response.status_code == 200
            data = response.json()
            assert data["model_loaded"] is False
            assert data["model_name"] is None


class TestErrorHandling:
    """Tests for error handling and exception handlers."""
    
    def test_validation_error_handler(self, client):
        """
        Test validation error handler.
        
        Validates: Requirements 14.2
        """
        # Send request with completely invalid data
        response = client.post("/predict", json={"invalid": "data"})
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "detail" in data
        assert "timestamp" in data
        assert "path" in data
    
    def test_error_response_structure(self, client, valid_customer_data):
        """
        Test error response structure.
        
        Validates: Requirements 14.2
        """
        mock_predictor = Mock()
        mock_predictor.predict.side_effect = Exception("Unexpected error")
        
        with patch('src.api.app.predictor', mock_predictor):
            response = client.post("/predict", json=valid_customer_data)
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert "detail" in data
            assert "timestamp" in data
            assert "path" in data
            assert data["path"] == "/predict"


class TestCORSMiddleware:
    """Tests for CORS middleware configuration."""
    
    def test_cors_middleware_configured(self, client):
        """
        Test CORS middleware is configured.
        
        Validates: Requirements 6.5
        """
        # Verify CORS middleware is configured by checking app middleware stack
        from src.api.app import app
        from fastapi.middleware.cors import CORSMiddleware
        
        # Check if any middleware is CORSMiddleware
        has_cors = False
        for middleware in app.user_middleware:
            if middleware.cls == CORSMiddleware:
                has_cors = True
                break
        
        assert has_cors, "CORS middleware not configured"


class TestLoggingMiddleware:
    """Tests for logging middleware."""
    
    def test_request_logging(self, client, mock_predictor):
        """
        Test that requests are logged.
        
        Validates: Requirements 14.1
        """
        with patch('src.api.app.predictor', mock_predictor):
            with patch('src.api.app.logger') as mock_logger:
                response = client.get("/health")
                
                # Verify logging was called
                assert mock_logger.info.called
                
                # Check that request was logged
                call_args = [str(call) for call in mock_logger.info.call_args_list]
                assert any("Request:" in str(call) for call in call_args)


class TestRiskClassification:
    """Tests for risk label classification."""
    
    def test_low_risk_classification(self, client, mock_predictor, valid_customer_data):
        """
        Test low risk classification.
        
        Validates: Requirements 7.4
        """
        mock_predictor.predict.return_value = PredictionResult(
            churn_probability=0.15,
            risk_label="Low"
        )
        
        with patch('src.api.app.predictor', mock_predictor):
            response = client.post("/predict", json=valid_customer_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["risk_label"] == "Low"
    
    def test_medium_risk_classification(self, client, mock_predictor, valid_customer_data):
        """
        Test medium risk classification.
        
        Validates: Requirements 7.4
        """
        mock_predictor.predict.return_value = PredictionResult(
            churn_probability=0.50,
            risk_label="Medium"
        )
        
        with patch('src.api.app.predictor', mock_predictor):
            response = client.post("/predict", json=valid_customer_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["risk_label"] == "Medium"
    
    def test_high_risk_classification(self, client, mock_predictor, valid_customer_data):
        """
        Test high risk classification.
        
        Validates: Requirements 7.4
        """
        mock_predictor.predict.return_value = PredictionResult(
            churn_probability=0.85,
            risk_label="High"
        )
        
        with patch('src.api.app.predictor', mock_predictor):
            response = client.post("/predict", json=valid_customer_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["risk_label"] == "High"
