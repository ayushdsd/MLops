"""
Property-based tests for the Prediction API.

These tests use Hypothesis to verify universal properties that should hold
for all valid inputs to the API functionality.
"""

import pytest
from unittest.mock import Mock, patch
from hypothesis import given, settings, HealthCheck
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.predictor import PredictionResult, ModelNotLoadedError, Predictor
from tests.fixtures.strategies import (
    valid_customer_strategy,
    invalid_customer_strategy,
    churn_probability_strategy,
    low_risk_probability_strategy,
    medium_risk_probability_strategy,
    high_risk_probability_strategy
)


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_predictor():
    """Create a mock predictor instance."""
    predictor = Mock(spec=Predictor)
    predictor.is_ready.return_value = True
    predictor.get_model_info.return_value = {
        "model_name": "churn_model",
        "model_version": "3",
        "model_loaded": True,
        "preprocessing_pipeline_loaded": True
    }
    return predictor


class TestAPIProperties:
    """Property-based tests for API functionality."""
    
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(customer_data=valid_customer_strategy(), probability=churn_probability_strategy())
    def test_property_16_valid_prediction_returns_probability(self, client, mock_predictor, customer_data, probability):
        """
        **Validates: Requirements 6.2**
        
        Feature: customer-churn-mlops-pipeline, Property 16: Valid Prediction Returns Probability
        
        For any valid customer data submitted to the /predict endpoint, the response should
        contain a churn_probability value between 0.0 and 1.0.
        """
        # Mock prediction result with the generated probability
        risk_label = Predictor.classify_risk(probability)
        mock_predictor.predict.return_value = PredictionResult(
            churn_probability=probability,
            risk_label=risk_label
        )
        
        with patch('src.api.app.predictor', mock_predictor):
            response = client.post("/predict", json=customer_data)
            
            # Verify successful response
            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.json()}"
            
            data = response.json()
            
            # Verify response contains churn_probability
            assert "churn_probability" in data, "Response missing churn_probability field"
            
            # Verify probability is between 0.0 and 1.0
            churn_prob = data["churn_probability"]
            assert isinstance(churn_prob, (int, float)), f"churn_probability must be numeric, got {type(churn_prob)}"
            assert 0.0 <= churn_prob <= 1.0, f"churn_probability must be in [0.0, 1.0], got {churn_prob}"
            
            # Verify other required fields are present
            assert "risk_label" in data, "Response missing risk_label field"
            assert "model_version" in data, "Response missing model_version field"
            assert "timestamp" in data, "Response missing timestamp field"
    
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    @given(customer_data=invalid_customer_strategy())
    def test_property_17_invalid_input_returns_400(self, client, mock_predictor, customer_data):
        """
        **Validates: Requirements 6.3**
        
        Feature: customer-churn-mlops-pipeline, Property 17: Invalid Input Returns 400 Status
        
        For any invalid customer data (missing fields, wrong types, out-of-range values),
        the /predict endpoint should return a 400 status code with error details.
        """
        with patch('src.api.app.predictor', mock_predictor):
            response = client.post("/predict", json=customer_data)
            
            # Verify 400 status code for invalid input
            assert response.status_code == 400, \
                f"Expected 400 for invalid input, got {response.status_code}"
            
            data = response.json()
            
            # Verify error response structure
            assert "error" in data, "Error response missing 'error' field"
            assert "detail" in data, "Error response missing 'detail' field"
            assert "timestamp" in data, "Error response missing 'timestamp' field"
            assert "path" in data, "Error response missing 'path' field"
            
            # Verify error field is not empty
            assert len(data["error"]) > 0, "Error field should not be empty"
            assert len(data["detail"]) > 0, "Detail field should not be empty"
    
    @settings(max_examples=50)
    @given(probability=low_risk_probability_strategy())
    def test_property_18_risk_label_classification_low(self, probability):
        """
        **Validates: Requirements 7.4**
        
        Feature: customer-churn-mlops-pipeline, Property 18: Risk Label Classification
        
        For any churn probability value in [0.0, 0.33), the assigned risk label should be "Low".
        """
        risk_label = Predictor.classify_risk(probability)
        
        assert risk_label == "Low", \
            f"Probability {probability:.4f} in [0.0, 0.33) should be classified as 'Low', got '{risk_label}'"
    
    @settings(max_examples=50)
    @given(probability=medium_risk_probability_strategy())
    def test_property_18_risk_label_classification_medium(self, probability):
        """
        **Validates: Requirements 7.4**
        
        Feature: customer-churn-mlops-pipeline, Property 18: Risk Label Classification
        
        For any churn probability value in [0.33, 0.66), the assigned risk label should be "Medium".
        """
        risk_label = Predictor.classify_risk(probability)
        
        assert risk_label == "Medium", \
            f"Probability {probability:.4f} in [0.33, 0.66) should be classified as 'Medium', got '{risk_label}'"
    
    @settings(max_examples=50)
    @given(probability=high_risk_probability_strategy())
    def test_property_18_risk_label_classification_high(self, probability):
        """
        **Validates: Requirements 7.4**
        
        Feature: customer-churn-mlops-pipeline, Property 18: Risk Label Classification
        
        For any churn probability value in [0.66, 1.0], the assigned risk label should be "High".
        """
        risk_label = Predictor.classify_risk(probability)
        
        assert risk_label == "High", \
            f"Probability {probability:.4f} in [0.66, 1.0] should be classified as 'High', got '{risk_label}'"
    
    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(customer_data=valid_customer_strategy())
    def test_property_22_model_loading_before_predictions(self, client, customer_data):
        """
        **Validates: Requirements 10.3**
        
        Feature: customer-churn-mlops-pipeline, Property 22: Model Loading Before Predictions
        
        For any Prediction_API instance, prediction requests should fail or be rejected
        until model loading is complete.
        """
        # Test with predictor not loaded (None)
        with patch('src.api.app.predictor', None):
            response = client.post("/predict", json=customer_data)
            
            # Verify request is rejected with 503 status
            assert response.status_code == 503, \
                f"Expected 503 when model not loaded, got {response.status_code}"
            
            data = response.json()
            assert "error" in data, "Error response missing 'error' field"
            assert data["error"] == "Service Unavailable", \
                f"Expected 'Service Unavailable' error, got '{data['error']}'"
        
        # Test with predictor that raises ModelNotLoadedError
        mock_predictor = Mock(spec=Predictor)
        mock_predictor.predict.side_effect = ModelNotLoadedError("Model not loaded")
        
        with patch('src.api.app.predictor', mock_predictor):
            response = client.post("/predict", json=customer_data)
            
            # Verify request is rejected with 503 status
            assert response.status_code == 503, \
                f"Expected 503 when model not loaded, got {response.status_code}"
            
            data = response.json()
            assert "error" in data, "Error response missing 'error' field"
    
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(customer_data=valid_customer_strategy())
    def test_property_30_comprehensive_input_validation_valid(self, client, mock_predictor, customer_data):
        """
        **Validates: Requirements 15.1, 15.2, 15.3, 15.4**
        
        Feature: customer-churn-mlops-pipeline, Property 30: Comprehensive Input Validation
        
        For any prediction request with valid data, the Prediction_API should validate that
        all required fields are present, numerical fields contain valid numbers, and
        categorical fields contain expected values, accepting the request.
        """
        # Mock successful prediction
        mock_predictor.predict.return_value = PredictionResult(
            churn_probability=0.5,
            risk_label="Medium"
        )
        
        with patch('src.api.app.predictor', mock_predictor):
            response = client.post("/predict", json=customer_data)
            
            # Valid data should be accepted (200 status)
            assert response.status_code == 200, \
                f"Valid data should be accepted with 200, got {response.status_code}: {response.json()}"
            
            # Verify predictor.predict was called (validation passed)
            assert mock_predictor.predict.called, \
                "Predictor should be called for valid input"
    
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    @given(customer_data=invalid_customer_strategy())
    def test_property_30_comprehensive_input_validation_invalid(self, client, mock_predictor, customer_data):
        """
        **Validates: Requirements 15.1, 15.2, 15.3, 15.4**
        
        Feature: customer-churn-mlops-pipeline, Property 30: Comprehensive Input Validation
        
        For any prediction request with invalid data, the Prediction_API should validate
        and return descriptive error messages indicating which fields are invalid.
        """
        with patch('src.api.app.predictor', mock_predictor):
            response = client.post("/predict", json=customer_data)
            
            # Invalid data should be rejected (400 status)
            assert response.status_code == 400, \
                f"Invalid data should be rejected with 400, got {response.status_code}"
            
            data = response.json()
            
            # Verify error response structure
            assert "error" in data, "Error response missing 'error' field"
            assert "detail" in data, "Error response missing 'detail' field"
            
            # Verify error details are descriptive (not empty)
            assert len(data["detail"]) > 0, \
                "Error detail should provide descriptive message"
            
            # Verify predictor.predict was NOT called (validation failed before prediction)
            assert not mock_predictor.predict.called, \
                "Predictor should not be called for invalid input"
    
    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(
        customer_data=valid_customer_strategy(),
        probability=churn_probability_strategy()
    )
    def test_property_risk_label_matches_probability(self, client, mock_predictor, customer_data, probability):
        """
        Property: Risk label should always match the probability range.
        
        For any prediction, the risk_label in the response should correctly correspond
        to the churn_probability value according to the classification rules.
        """
        # Determine expected risk label
        if probability < 0.33:
            expected_risk = "Low"
        elif probability < 0.66:
            expected_risk = "Medium"
        else:
            expected_risk = "High"
        
        # Mock prediction with the generated probability
        mock_predictor.predict.return_value = PredictionResult(
            churn_probability=probability,
            risk_label=expected_risk
        )
        
        with patch('src.api.app.predictor', mock_predictor):
            response = client.post("/predict", json=customer_data)
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify risk label matches probability
            actual_risk = data["risk_label"]
            actual_prob = data["churn_probability"]
            
            if actual_prob < 0.33:
                assert actual_risk == "Low", \
                    f"Probability {actual_prob:.4f} < 0.33 should have risk 'Low', got '{actual_risk}'"
            elif actual_prob < 0.66:
                assert actual_risk == "Medium", \
                    f"Probability {actual_prob:.4f} in [0.33, 0.66) should have risk 'Medium', got '{actual_risk}'"
            else:
                assert actual_risk == "High", \
                    f"Probability {actual_prob:.4f} >= 0.66 should have risk 'High', got '{actual_risk}'"
    
    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(customer_data=valid_customer_strategy())
    def test_property_prediction_response_structure(self, client, mock_predictor, customer_data):
        """
        Property: All prediction responses should have consistent structure.
        
        For any successful prediction, the response should contain all required fields
        with correct types.
        """
        # Mock successful prediction
        mock_predictor.predict.return_value = PredictionResult(
            churn_probability=0.5,
            risk_label="Medium"
        )
        
        with patch('src.api.app.predictor', mock_predictor):
            response = client.post("/predict", json=customer_data)
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify all required fields are present
            required_fields = ["churn_probability", "risk_label", "model_version", "timestamp"]
            for field in required_fields:
                assert field in data, f"Response missing required field '{field}'"
            
            # Verify field types
            assert isinstance(data["churn_probability"], (int, float)), \
                "churn_probability must be numeric"
            assert isinstance(data["risk_label"], str), \
                "risk_label must be string"
            assert isinstance(data["model_version"], str), \
                "model_version must be string"
            assert isinstance(data["timestamp"], str), \
                "timestamp must be string"
            
            # Verify risk_label is valid
            assert data["risk_label"] in ["Low", "Medium", "High"], \
                f"risk_label must be Low/Medium/High, got '{data['risk_label']}'"
    
    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    @given(customer_data=invalid_customer_strategy())
    def test_property_error_response_structure(self, client, mock_predictor, customer_data):
        """
        Property: All error responses should have consistent structure.
        
        For any request that fails validation, the error response should contain
        all required fields with descriptive information.
        """
        with patch('src.api.app.predictor', mock_predictor):
            response = client.post("/predict", json=customer_data)
            
            # Should be an error response
            assert response.status_code >= 400
            data = response.json()
            
            # Verify all required error fields are present
            required_fields = ["error", "detail", "timestamp", "path"]
            for field in required_fields:
                assert field in data, f"Error response missing required field '{field}'"
            
            # Verify field types
            assert isinstance(data["error"], str), "error must be string"
            assert isinstance(data["detail"], str), "detail must be string"
            assert isinstance(data["timestamp"], str), "timestamp must be string"
            assert isinstance(data["path"], str), "path must be string"
            
            # Verify fields are not empty
            assert len(data["error"]) > 0, "error field should not be empty"
            assert len(data["detail"]) > 0, "detail field should not be empty"
