"""
Integration tests for Streamlit UI with Prediction API.

Tests the complete flow from UI components to API interaction.
"""

import pytest
from unittest.mock import Mock, patch
import requests

from src.ui.components import call_prediction_api


class TestUIAPIIntegration:
    """Integration tests for UI and API interaction."""
    
    def test_end_to_end_prediction_flow(self):
        """Test complete prediction flow from UI to API."""
        # Arrange
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
            "streaming_movies": "No",
            "contract": "Month-to-month",
            "paperless_billing": "Yes",
            "payment_method": "Electronic check",
            "monthly_charges": 70.35,
            "total_charges": 844.20
        }
        
        expected_response = {
            "churn_probability": 0.73,
            "risk_label": "High",
            "model_version": "3",
            "timestamp": "2024-01-15T10:30:00Z"
        }
        
        # Mock API response
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response
            mock_post.return_value = mock_response
            
            # Act
            result = call_prediction_api(customer_data)
            
            # Assert
            assert result["churn_probability"] == 0.73
            assert result["risk_label"] == "High"
            assert result["model_version"] == "3"
            assert "timestamp" in result
    
    def test_ui_handles_api_validation_error(self):
        """Test UI properly handles API validation errors."""
        # Arrange
        invalid_customer_data = {
            "gender": "Invalid",  # Invalid value
            "senior_citizen": 0,
            "tenure": -5  # Invalid negative value
        }
        
        error_response = {
            "error": "Validation Error",
            "detail": "gender must be 'Male' or 'Female'; tenure must be non-negative",
            "timestamp": "2024-01-15T10:30:00Z",
            "path": "/predict"
        }
        
        # Mock API response
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.json.return_value = error_response
            mock_post.return_value = mock_response
            
            # Act & Assert
            with pytest.raises(ValueError) as exc_info:
                call_prediction_api(invalid_customer_data)
            
            assert "Validation Error" in str(exc_info.value)
    
    def test_ui_handles_api_unavailable(self):
        """Test UI properly handles API unavailability."""
        # Arrange
        customer_data = {"gender": "Male", "senior_citizen": 0}
        
        # Mock connection error
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.ConnectionError("Connection refused")
            
            with patch('time.sleep'):
                # Act & Assert
                with pytest.raises(requests.RequestException) as exc_info:
                    call_prediction_api(customer_data, max_retries=2)
                
                assert "Cannot connect" in str(exc_info.value)
    
    def test_ui_handles_api_timeout(self):
        """Test UI properly handles API timeout."""
        # Arrange
        customer_data = {"gender": "Male", "senior_citizen": 0}
        
        # Mock timeout
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.Timeout("Request timeout")
            
            with patch('time.sleep'):
                # Act & Assert
                with pytest.raises(requests.RequestException) as exc_info:
                    call_prediction_api(customer_data, timeout=1, max_retries=2)
                
                assert "timeout" in str(exc_info.value).lower()
    
    def test_ui_retry_logic_with_transient_failures(self):
        """Test UI retry logic handles transient failures."""
        # Arrange
        customer_data = {"gender": "Female", "senior_citizen": 1}
        
        expected_response = {
            "churn_probability": 0.45,
            "risk_label": "Medium",
            "model_version": "2",
            "timestamp": "2024-01-15T10:30:00Z"
        }
        
        # Mock API to fail twice then succeed
        with patch('requests.post') as mock_post:
            mock_fail_response = Mock()
            mock_fail_response.status_code = 503
            
            mock_success_response = Mock()
            mock_success_response.status_code = 200
            mock_success_response.json.return_value = expected_response
            
            mock_post.side_effect = [
                mock_fail_response,
                mock_fail_response,
                mock_success_response
            ]
            
            with patch('time.sleep'):
                # Act
                result = call_prediction_api(customer_data, max_retries=3)
                
                # Assert
                assert result == expected_response
                assert mock_post.call_count == 3
    
    def test_ui_respects_custom_api_url(self):
        """Test UI uses custom API URL from configuration."""
        # Arrange
        customer_data = {"gender": "Male"}
        custom_api_url = "http://custom-api:9000"
        
        expected_response = {
            "churn_probability": 0.2,
            "risk_label": "Low",
            "model_version": "1",
            "timestamp": "2024-01-15T10:30:00Z"
        }
        
        # Mock API response
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response
            mock_post.return_value = mock_response
            
            # Act
            call_prediction_api(customer_data, api_url=custom_api_url)
            
            # Assert
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == f"{custom_api_url}/predict"
    
    def test_ui_respects_custom_timeout(self):
        """Test UI uses custom timeout from configuration."""
        # Arrange
        customer_data = {"gender": "Female"}
        custom_timeout = 15
        
        expected_response = {
            "churn_probability": 0.6,
            "risk_label": "Medium",
            "model_version": "2",
            "timestamp": "2024-01-15T10:30:00Z"
        }
        
        # Mock API response
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response
            mock_post.return_value = mock_response
            
            # Act
            call_prediction_api(customer_data, timeout=custom_timeout)
            
            # Assert
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args[1]
            assert call_kwargs['timeout'] == custom_timeout
