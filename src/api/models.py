"""
Pydantic models for the Prediction API.

This module defines request and response models for the FastAPI prediction service.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class CustomerInput(BaseModel):
    """
    Customer data input model for churn prediction.
    
    Contains all 19 customer attributes including demographics, account information,
    and service subscriptions.
    """
    # Demographics
    gender: str = Field(..., description="Customer gender: Male or Female")
    senior_citizen: int = Field(..., description="Whether customer is a senior citizen: 0 or 1")
    partner: str = Field(..., description="Whether customer has a partner: Yes or No")
    dependents: str = Field(..., description="Whether customer has dependents: Yes or No")
    
    # Account Information
    tenure: int = Field(..., ge=0, description="Number of months with the company")
    contract: str = Field(..., description="Contract type: Month-to-month, One year, or Two year")
    paperless_billing: str = Field(..., description="Whether customer uses paperless billing: Yes or No")
    payment_method: str = Field(..., description="Payment method: Electronic check, Mailed check, Bank transfer, or Credit card")
    monthly_charges: float = Field(..., gt=0, description="Monthly bill amount")
    total_charges: float = Field(..., ge=0, description="Total amount charged to customer")
    
    # Services
    phone_service: str = Field(..., description="Whether customer has phone service: Yes or No")
    multiple_lines: str = Field(..., description="Whether customer has multiple lines: Yes, No, or No phone service")
    internet_service: str = Field(..., description="Internet service type: DSL, Fiber optic, or No")
    online_security: str = Field(..., description="Whether customer has online security: Yes, No, or No internet service")
    online_backup: str = Field(..., description="Whether customer has online backup: Yes, No, or No internet service")
    device_protection: str = Field(..., description="Whether customer has device protection: Yes, No, or No internet service")
    tech_support: str = Field(..., description="Whether customer has tech support: Yes, No, or No internet service")
    streaming_tv: str = Field(..., description="Whether customer has streaming TV: Yes, No, or No internet service")
    streaming_movies: str = Field(..., description="Whether customer has streaming movies: Yes, No, or No internet service")
    
    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v: str) -> str:
        """Validate gender field."""
        if v not in ['Male', 'Female']:
            raise ValueError('gender must be "Male" or "Female"')
        return v
    
    @field_validator('senior_citizen')
    @classmethod
    def validate_senior_citizen(cls, v: int) -> int:
        """Validate senior_citizen field."""
        if v not in [0, 1]:
            raise ValueError('senior_citizen must be 0 or 1')
        return v
    
    @field_validator('partner', 'dependents', 'paperless_billing', 'phone_service')
    @classmethod
    def validate_yes_no(cls, v: str) -> str:
        """Validate Yes/No fields."""
        if v not in ['Yes', 'No']:
            raise ValueError(f'field must be "Yes" or "No"')
        return v
    
    @field_validator('multiple_lines')
    @classmethod
    def validate_multiple_lines(cls, v: str) -> str:
        """Validate multiple_lines field."""
        if v not in ['Yes', 'No', 'No phone service']:
            raise ValueError('multiple_lines must be "Yes", "No", or "No phone service"')
        return v
    
    @field_validator('internet_service')
    @classmethod
    def validate_internet_service(cls, v: str) -> str:
        """Validate internet_service field."""
        if v not in ['DSL', 'Fiber optic', 'No']:
            raise ValueError('internet_service must be "DSL", "Fiber optic", or "No"')
        return v
    
    @field_validator('online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies')
    @classmethod
    def validate_internet_dependent(cls, v: str) -> str:
        """Validate internet-dependent service fields."""
        if v not in ['Yes', 'No', 'No internet service']:
            raise ValueError(f'field must be "Yes", "No", or "No internet service"')
        return v
    
    @field_validator('contract')
    @classmethod
    def validate_contract(cls, v: str) -> str:
        """Validate contract field."""
        if v not in ['Month-to-month', 'One year', 'Two year']:
            raise ValueError('contract must be "Month-to-month", "One year", or "Two year"')
        return v
    
    @field_validator('payment_method')
    @classmethod
    def validate_payment_method(cls, v: str) -> str:
        """Validate payment_method field."""
        if v not in ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']:
            raise ValueError('payment_method must be "Electronic check", "Mailed check", "Bank transfer (automatic)", or "Credit card (automatic)"')
        return v


class PredictionResponse(BaseModel):
    """
    Response model for churn prediction.
    
    Contains the predicted churn probability, risk classification, model version,
    and prediction timestamp.
    """
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of customer churn (0-1)")
    risk_label: str = Field(..., description="Risk classification: Low, Medium, or High")
    model_version: str = Field(..., description="Version of the model used for prediction")
    timestamp: str = Field(..., description="ISO format timestamp of prediction")
    
    @field_validator('risk_label')
    @classmethod
    def validate_risk_label(cls, v: str) -> str:
        """Validate risk_label field."""
        if v not in ['Low', 'Medium', 'High']:
            raise ValueError('risk_label must be "Low", "Medium", or "High"')
        return v


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.
    
    Indicates service status, whether the model is loaded, and the current model version.
    """
    status: str = Field(..., description="Service status: healthy or unhealthy")
    model_loaded: bool = Field(..., description="Whether the prediction model is loaded")
    model_version: Optional[str] = Field(None, description="Current model version if loaded")
    
    @field_validator('status')
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate status field."""
        if v not in ['healthy', 'unhealthy']:
            raise ValueError('status must be "healthy" or "unhealthy"')
        return v


class ErrorResponse(BaseModel):
    """
    Response model for error responses.
    
    Provides structured error information including error type, detail message,
    timestamp, and request path.
    """
    error: str = Field(..., description="Error type or category")
    detail: str = Field(..., description="Detailed error message")
    timestamp: str = Field(..., description="ISO format timestamp of error")
    path: str = Field(..., description="Request path where error occurred")
