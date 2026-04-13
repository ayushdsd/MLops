"""
Unit tests for Streamlit UI components.

Tests the UI components including form rendering, API calls, and display functions.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
from datetime import datetime

from src.ui.components import (
    call_prediction_api,
    display_prediction,
    display_error
)


class TestCallPredictionAPI:
    """Test suite for call_prediction_api function."""
    
    def test_successful_api_call(self):
        """Test successful API call returns response data."""
        # Arrange
        customer_data = {
            "gender": "Female",
            "senior_citizen": 0,
            "partner": "Yes",
            "tenure": 12,
            "monthly_charges": 70.0,
            "total_charges": 840.0
        }
        
        expected_response = {
            "churn_probability": 0.73,
            "risk_label": "High",
            "model_version": "3",
            "timestamp": "2024-01-15T10:30:00Z"
        }
        
        # Mock requests.post
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response
            mock_post.return_value = mock_response
            
            # Act
            result = call_prediction_api(customer_data, api_url="http://test:8000")
            
            # Assert
            assert result == expected_response
            mock_post.assert_called_once_with(
                "http://test:8000/predict",
                json=customer_data,
                timeout=5
            )
    
    def test_validation_error_raises_value_error(self):
        """Test that 400 status code raises ValueError without retry."""
        # Arrange
        customer_data = {"invalid": "data"}
        
        error_response = {
            "error": "Validation Error",
            "detail": "Missing required field: gender"
        }
        
        # Mock requests.post
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.json.return_value = error_response
            mock_post.return_value = mock_response
            
            # Act & Assert
            with pytest.raises(ValueError) as exc_info:
                call_prediction_api(customer_data, api_url="http://test:8000")
            
            assert "Validation Error" in str(exc_info.value)
            # Should not retry on validation errors
            assert mock_post.call_count == 1
    
    def test_service_unavailable_retries_with_backoff(self):
        """Test that 503 status code triggers retry with exponential backoff."""
        # Arrange
        customer_data = {"gender": "Male"}
        
        # Mock requests.post to return 503 on all attempts
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 503
            mock_post.return_value = mock_response
            
            with patch('time.sleep') as mock_sleep:
                # Act & Assert
                with pytest.raises(requests.RequestException) as exc_info:
                    call_prediction_api(
                        customer_data,
                        api_url="http://test:8000",
                        max_retries=3
                    )
                
                assert "unavailable" in str(exc_info.value).lower()
                # Should retry 3 times
                assert mock_post.call_count == 3
                # Should sleep with exponential backoff: 1s, 2s
                assert mock_sleep.call_count == 2
    
    def test_connection_error_retries(self):
        """Test that connection errors trigger retry."""
        # Arrange
        customer_data = {"gender": "Male"}
        
        # Mock requests.post to raise ConnectionError
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.ConnectionError("Connection refused")
            
            with patch('time.sleep') as mock_sleep:
                # Act & Assert
                with pytest.raises(requests.RequestException) as exc_info:
                    call_prediction_api(
                        customer_data,
                        api_url="http://test:8000",
                        max_retries=3
                    )
                
                assert "Cannot connect" in str(exc_info.value)
                # Should retry 3 times
                assert mock_post.call_count == 3
                # Should sleep with exponential backoff
                assert mock_sleep.call_count == 2
    
    def test_timeout_retries(self):
        """Test that timeout errors trigger retry."""
        # Arrange
        customer_data = {"gender": "Male"}
        
        # Mock requests.post to raise Timeout
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.Timeout("Request timeout")
            
            with patch('time.sleep') as mock_sleep:
                # Act & Assert
                with pytest.raises(requests.RequestException) as exc_info:
                    call_prediction_api(
                        customer_data,
                        api_url="http://test:8000",
                        timeout=5,
                        max_retries=3
                    )
                
                assert "timeout" in str(exc_info.value).lower()
                # Should retry 3 times
                assert mock_post.call_count == 3
    
    def test_successful_retry_after_failure(self):
        """Test that API call succeeds after initial failures."""
        # Arrange
        customer_data = {"gender": "Male"}
        expected_response = {
            "churn_probability": 0.5,
            "risk_label": "Medium",
            "model_version": "2",
            "timestamp": "2024-01-15T10:30:00Z"
        }
        
        # Mock requests.post to fail twice then succeed
        with patch('requests.post') as mock_post:
            mock_response_fail = Mock()
            mock_response_fail.status_code = 503
            
            mock_response_success = Mock()
            mock_response_success.status_code = 200
            mock_response_success.json.return_value = expected_response
            
            mock_post.side_effect = [
                mock_response_fail,
                mock_response_fail,
                mock_response_success
            ]
            
            with patch('time.sleep'):
                # Act
                result = call_prediction_api(
                    customer_data,
                    api_url="http://test:8000",
                    max_retries=3
                )
                
                # Assert
                assert result == expected_response
                assert mock_post.call_count == 3
    
    def test_custom_timeout_parameter(self):
        """Test that custom timeout is used in API call."""
        # Arrange
        customer_data = {"gender": "Male"}
        custom_timeout = 10
        
        expected_response = {
            "churn_probability": 0.3,
            "risk_label": "Low",
            "model_version": "1",
            "timestamp": "2024-01-15T10:30:00Z"
        }
        
        # Mock requests.post
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response
            mock_post.return_value = mock_response
            
            # Act
            call_prediction_api(
                customer_data,
                api_url="http://test:8000",
                timeout=custom_timeout
            )
            
            # Assert
            mock_post.assert_called_once_with(
                "http://test:8000/predict",
                json=customer_data,
                timeout=custom_timeout
            )


class TestDisplayPrediction:
    """Test suite for display_prediction function."""
    
    @patch('streamlit.markdown')
    @patch('streamlit.subheader')
    @patch('streamlit.progress')
    @patch('streamlit.success')
    @patch('streamlit.columns')
    def test_display_low_risk_prediction(
        self,
        mock_columns,
        mock_success,
        mock_progress,
        mock_subheader,
        mock_markdown
    ):
        """Test displaying low risk prediction."""
        # Arrange
        response = {
            "churn_probability": 0.25,
            "risk_label": "Low",
            "model_version": "3",
            "timestamp": "2024-01-15T10:30:00Z"
        }
        
        # Mock columns
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_columns.return_value = [mock_col1, mock_col2]
        
        # Act
        display_prediction(response)
        
        # Assert
        mock_progress.assert_called_once_with(0.25)
        mock_success.assert_called_once()
    
    @patch('streamlit.markdown')
    @patch('streamlit.subheader')
    @patch('streamlit.progress')
    @patch('streamlit.warning')
    @patch('streamlit.columns')
    def test_display_medium_risk_prediction(
        self,
        mock_columns,
        mock_warning,
        mock_progress,
        mock_subheader,
        mock_markdown
    ):
        """Test displaying medium risk prediction."""
        # Arrange
        response = {
            "churn_probability": 0.50,
            "risk_label": "Medium",
            "model_version": "3",
            "timestamp": "2024-01-15T10:30:00Z"
        }
        
        # Mock columns
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_columns.return_value = [mock_col1, mock_col2]
        
        # Act
        display_prediction(response)
        
        # Assert
        mock_progress.assert_called_once_with(0.50)
        mock_warning.assert_called_once()
    
    @patch('streamlit.markdown')
    @patch('streamlit.subheader')
    @patch('streamlit.progress')
    @patch('streamlit.error')
    @patch('streamlit.columns')
    def test_display_high_risk_prediction(
        self,
        mock_columns,
        mock_error,
        mock_progress,
        mock_subheader,
        mock_markdown
    ):
        """Test displaying high risk prediction."""
        # Arrange
        response = {
            "churn_probability": 0.85,
            "risk_label": "High",
            "model_version": "3",
            "timestamp": "2024-01-15T10:30:00Z"
        }
        
        # Mock columns
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_columns.return_value = [mock_col1, mock_col2]
        
        # Act
        display_prediction(response)
        
        # Assert
        mock_progress.assert_called_once_with(0.85)
        mock_error.assert_called_once()


class TestDisplayError:
    """Test suite for display_error function."""
    
    @patch('streamlit.markdown')
    @patch('streamlit.error')
    @patch('streamlit.info')
    def test_display_service_unavailable_error(
        self,
        mock_info,
        mock_error,
        mock_markdown
    ):
        """Test displaying service unavailable error."""
        # Arrange
        error_message = "Prediction service unavailable. Please try again later."
        
        # Act
        display_error(error_message)
        
        # Assert
        mock_error.assert_called_once()
        assert mock_markdown.call_count >= 2  # Multiple markdown calls for styling
    
    @patch('streamlit.markdown')
    @patch('streamlit.warning')
    @patch('streamlit.info')
    def test_display_validation_error(
        self,
        mock_info,
        mock_warning,
        mock_markdown
    ):
        """Test displaying validation error."""
        # Arrange
        error_message = "Validation Error: Missing required field: gender"
        
        # Act
        display_error(error_message)
        
        # Assert
        mock_warning.assert_called_once()
        assert mock_markdown.call_count >= 2
    
    @patch('streamlit.markdown')
    @patch('streamlit.error')
    @patch('streamlit.info')
    def test_display_timeout_error(
        self,
        mock_info,
        mock_error,
        mock_markdown
    ):
        """Test displaying timeout error."""
        # Arrange
        error_message = "Request timeout after 3 attempts"
        
        # Act
        display_error(error_message)
        
        # Assert
        mock_error.assert_called_once()
        assert mock_markdown.call_count >= 2
    
    @patch('streamlit.markdown')
    @patch('streamlit.error')
    @patch('streamlit.info')
    def test_display_connection_error(
        self,
        mock_info,
        mock_error,
        mock_markdown
    ):
        """Test displaying connection error."""
        # Arrange
        error_message = "Cannot connect to prediction API at http://localhost:8000"
        
        # Act
        display_error(error_message)
        
        # Assert
        mock_error.assert_called_once()
        assert mock_markdown.call_count >= 2
    
    @patch('streamlit.markdown')
    @patch('streamlit.warning')
    @patch('streamlit.info')
    def test_display_warning_type_error(
        self,
        mock_info,
        mock_warning,
        mock_markdown
    ):
        """Test displaying error with warning type."""
        # Arrange
        error_message = "Some warning message"
        
        # Act
        display_error(error_message, error_type="warning")
        
        # Assert
        mock_warning.assert_called_once()
    
    @patch('streamlit.markdown')
    @patch('streamlit.info')
    def test_display_info_type_error(
        self,
        mock_info,
        mock_markdown
    ):
        """Test displaying error with info type."""
        # Arrange
        error_message = "Some informational message"
        
        # Act
        display_error(error_message, error_type="info")
        
        # Assert
        # info is called twice: once for the message, once for the tip
        assert mock_info.call_count == 2
