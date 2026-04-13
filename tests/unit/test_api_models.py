"""
Unit tests for API Pydantic models.

Tests validation logic, field constraints, and model instantiation.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError
from src.api.models import (
    CustomerInput,
    PredictionResponse,
    HealthResponse,
    ErrorResponse
)


class TestCustomerInput:
    """Test cases for CustomerInput model."""
    
    def test_valid_customer_input(self):
        """Test creating CustomerInput with valid data."""
        customer = CustomerInput(
            gender="Male",
            senior_citizen=0,
            partner="Yes",
            dependents="No",
            tenure=12,
            contract="Month-to-month",
            paperless_billing="Yes",
            payment_method="Electronic check",
            monthly_charges=50.0,
            total_charges=600.0,
            phone_service="Yes",
            multiple_lines="No",
            internet_service="DSL",
            online_security="Yes",
            online_backup="No",
            device_protection="Yes",
            tech_support="No",
            streaming_tv="Yes",
            streaming_movies="No"
        )
        
        assert customer.gender == "Male"
        assert customer.tenure == 12
        assert customer.monthly_charges == 50.0
    
    def test_invalid_gender(self):
        """Test that invalid gender raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CustomerInput(
                gender="Other",
                senior_citizen=0,
                partner="Yes",
                dependents="No",
                tenure=12,
                contract="Month-to-month",
                paperless_billing="Yes",
                payment_method="Electronic check",
                monthly_charges=50.0,
                total_charges=600.0,
                phone_service="Yes",
                multiple_lines="No",
                internet_service="DSL",
                online_security="Yes",
                online_backup="No",
                device_protection="Yes",
                tech_support="No",
                streaming_tv="Yes",
                streaming_movies="No"
            )
        
        assert 'gender must be "Male" or "Female"' in str(exc_info.value)
    
    def test_invalid_senior_citizen(self):
        """Test that invalid senior_citizen raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CustomerInput(
                gender="Male",
                senior_citizen=2,
                partner="Yes",
                dependents="No",
                tenure=12,
                contract="Month-to-month",
                paperless_billing="Yes",
                payment_method="Electronic check",
                monthly_charges=50.0,
                total_charges=600.0,
                phone_service="Yes",
                multiple_lines="No",
                internet_service="DSL",
                online_security="Yes",
                online_backup="No",
                device_protection="Yes",
                tech_support="No",
                streaming_tv="Yes",
                streaming_movies="No"
            )
        
        assert 'senior_citizen must be 0 or 1' in str(exc_info.value)
    
    def test_negative_tenure(self):
        """Test that negative tenure raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CustomerInput(
                gender="Male",
                senior_citizen=0,
                partner="Yes",
                dependents="No",
                tenure=-5,
                contract="Month-to-month",
                paperless_billing="Yes",
                payment_method="Electronic check",
                monthly_charges=50.0,
                total_charges=600.0,
                phone_service="Yes",
                multiple_lines="No",
                internet_service="DSL",
                online_security="Yes",
                online_backup="No",
                device_protection="Yes",
                tech_support="No",
                streaming_tv="Yes",
                streaming_movies="No"
            )
        
        assert 'greater than or equal to 0' in str(exc_info.value).lower()
    
    def test_invalid_monthly_charges(self):
        """Test that non-positive monthly_charges raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CustomerInput(
                gender="Male",
                senior_citizen=0,
                partner="Yes",
                dependents="No",
                tenure=12,
                contract="Month-to-month",
                paperless_billing="Yes",
                payment_method="Electronic check",
                monthly_charges=0.0,
                total_charges=600.0,
                phone_service="Yes",
                multiple_lines="No",
                internet_service="DSL",
                online_security="Yes",
                online_backup="No",
                device_protection="Yes",
                tech_support="No",
                streaming_tv="Yes",
                streaming_movies="No"
            )
        
        assert 'greater than 0' in str(exc_info.value).lower()
    
    def test_invalid_contract(self):
        """Test that invalid contract raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CustomerInput(
                gender="Male",
                senior_citizen=0,
                partner="Yes",
                dependents="No",
                tenure=12,
                contract="Three year",
                paperless_billing="Yes",
                payment_method="Electronic check",
                monthly_charges=50.0,
                total_charges=600.0,
                phone_service="Yes",
                multiple_lines="No",
                internet_service="DSL",
                online_security="Yes",
                online_backup="No",
                device_protection="Yes",
                tech_support="No",
                streaming_tv="Yes",
                streaming_movies="No"
            )
        
        assert 'contract must be' in str(exc_info.value)
    
    def test_missing_required_field(self):
        """Test that missing required field raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CustomerInput(
                gender="Male",
                senior_citizen=0,
                partner="Yes",
                dependents="No",
                tenure=12,
                contract="Month-to-month",
                paperless_billing="Yes",
                payment_method="Electronic check",
                monthly_charges=50.0,
                # total_charges missing
                phone_service="Yes",
                multiple_lines="No",
                internet_service="DSL",
                online_security="Yes",
                online_backup="No",
                device_protection="Yes",
                tech_support="No",
                streaming_tv="Yes",
                streaming_movies="No"
            )
        
        assert 'total_charges' in str(exc_info.value).lower()


class TestPredictionResponse:
    """Test cases for PredictionResponse model."""
    
    def test_valid_prediction_response(self):
        """Test creating PredictionResponse with valid data."""
        response = PredictionResponse(
            churn_probability=0.75,
            risk_label="High",
            model_version="v1.0.0",
            timestamp=datetime.utcnow().isoformat()
        )
        
        assert response.churn_probability == 0.75
        assert response.risk_label == "High"
        assert response.model_version == "v1.0.0"
    
    def test_probability_out_of_range_high(self):
        """Test that probability > 1.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(
                churn_probability=1.5,
                risk_label="High",
                model_version="v1.0.0",
                timestamp=datetime.utcnow().isoformat()
            )
        
        assert 'less than or equal to 1' in str(exc_info.value).lower()
    
    def test_probability_out_of_range_low(self):
        """Test that probability < 0.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(
                churn_probability=-0.1,
                risk_label="Low",
                model_version="v1.0.0",
                timestamp=datetime.utcnow().isoformat()
            )
        
        assert 'greater than or equal to 0' in str(exc_info.value).lower()
    
    def test_invalid_risk_label(self):
        """Test that invalid risk_label raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(
                churn_probability=0.5,
                risk_label="Critical",
                model_version="v1.0.0",
                timestamp=datetime.utcnow().isoformat()
            )
        
        assert 'risk_label must be' in str(exc_info.value)
    
    def test_all_risk_labels(self):
        """Test that all valid risk labels are accepted."""
        for risk_label in ["Low", "Medium", "High"]:
            response = PredictionResponse(
                churn_probability=0.5,
                risk_label=risk_label,
                model_version="v1.0.0",
                timestamp=datetime.utcnow().isoformat()
            )
            assert response.risk_label == risk_label


class TestHealthResponse:
    """Test cases for HealthResponse model."""
    
    def test_healthy_with_model(self):
        """Test creating HealthResponse for healthy service with model loaded."""
        response = HealthResponse(
            status="healthy",
            model_loaded=True,
            model_version="v1.0.0"
        )
        
        assert response.status == "healthy"
        assert response.model_loaded is True
        assert response.model_version == "v1.0.0"
    
    def test_unhealthy_without_model(self):
        """Test creating HealthResponse for unhealthy service without model."""
        response = HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_version=None
        )
        
        assert response.status == "unhealthy"
        assert response.model_loaded is False
        assert response.model_version is None
    
    def test_invalid_status(self):
        """Test that invalid status raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            HealthResponse(
                status="degraded",
                model_loaded=True,
                model_version="v1.0.0"
            )
        
        assert 'status must be' in str(exc_info.value)


class TestErrorResponse:
    """Test cases for ErrorResponse model."""
    
    def test_valid_error_response(self):
        """Test creating ErrorResponse with valid data."""
        response = ErrorResponse(
            error="ValidationError",
            detail="Invalid customer data provided",
            timestamp=datetime.utcnow().isoformat(),
            path="/predict"
        )
        
        assert response.error == "ValidationError"
        assert response.detail == "Invalid customer data provided"
        assert response.path == "/predict"
    
    def test_all_fields_required(self):
        """Test that all fields are required."""
        with pytest.raises(ValidationError) as exc_info:
            ErrorResponse(
                error="ValidationError",
                detail="Invalid customer data provided"
                # timestamp and path missing
            )
        
        errors = str(exc_info.value).lower()
        assert 'timestamp' in errors
        assert 'path' in errors
