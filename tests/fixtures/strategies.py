"""
Hypothesis strategies for property-based testing.

This module provides reusable strategies for generating test data
for the Customer Churn MLOps Pipeline.
"""

import hypothesis.strategies as st
from hypothesis import assume
from typing import Dict, Any


@st.composite
def valid_customer_strategy(draw) -> Dict[str, Any]:
    """
    Generate random valid customer data for testing.
    
    This strategy generates customer data that passes all validation rules:
    - All required fields present
    - Numerical fields within valid ranges
    - Categorical fields with valid values
    
    Returns:
        Dictionary containing valid customer attributes
    """
    return {
        # Demographics
        "gender": draw(st.sampled_from(["Male", "Female"])),
        "senior_citizen": draw(st.integers(min_value=0, max_value=1)),
        "partner": draw(st.sampled_from(["Yes", "No"])),
        "dependents": draw(st.sampled_from(["Yes", "No"])),
        
        # Account Information
        "tenure": draw(st.integers(min_value=0, max_value=72)),
        "contract": draw(st.sampled_from(["Month-to-month", "One year", "Two year"])),
        "paperless_billing": draw(st.sampled_from(["Yes", "No"])),
        "payment_method": draw(st.sampled_from([
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ])),
        "monthly_charges": draw(st.floats(min_value=0.01, max_value=200.0, allow_nan=False, allow_infinity=False)),
        "total_charges": draw(st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False)),
        
        # Services
        "phone_service": draw(st.sampled_from(["Yes", "No"])),
        "multiple_lines": draw(st.sampled_from(["Yes", "No", "No phone service"])),
        "internet_service": draw(st.sampled_from(["DSL", "Fiber optic", "No"])),
        "online_security": draw(st.sampled_from(["Yes", "No", "No internet service"])),
        "online_backup": draw(st.sampled_from(["Yes", "No", "No internet service"])),
        "device_protection": draw(st.sampled_from(["Yes", "No", "No internet service"])),
        "tech_support": draw(st.sampled_from(["Yes", "No", "No internet service"])),
        "streaming_tv": draw(st.sampled_from(["Yes", "No", "No internet service"])),
        "streaming_movies": draw(st.sampled_from(["Yes", "No", "No internet service"]))
    }


@st.composite
def invalid_customer_strategy(draw) -> Dict[str, Any]:
    """
    Generate random invalid customer data for validation testing.
    
    This strategy generates customer data with various validation errors:
    - Missing required fields
    - Invalid numerical values (negative, zero where not allowed)
    - Invalid categorical values
    - Wrong data types
    
    Returns:
        Dictionary containing invalid customer attributes
    """
    # Start with valid data
    customer = draw(valid_customer_strategy())
    
    # Choose what kind of validation error to introduce
    error_type = draw(st.sampled_from([
        "missing_field",
        "invalid_categorical",
        "negative_tenure",
        "zero_monthly_charges",
        "invalid_type",
        "invalid_gender",
        "invalid_contract"
    ]))
    
    if error_type == "missing_field":
        # Remove a required field
        field_to_remove = draw(st.sampled_from([
            "gender", "tenure", "monthly_charges", "contract", "internet_service"
        ]))
        del customer[field_to_remove]
    
    elif error_type == "invalid_categorical":
        # Use invalid categorical value
        customer["gender"] = "Other"
    
    elif error_type == "negative_tenure":
        # Use negative tenure
        customer["tenure"] = draw(st.integers(min_value=-100, max_value=-1))
    
    elif error_type == "zero_monthly_charges":
        # Use zero or negative monthly charges
        customer["monthly_charges"] = draw(st.floats(min_value=-100.0, max_value=0.0, allow_nan=False, allow_infinity=False))
    
    elif error_type == "invalid_type":
        # Use wrong type for numerical field
        customer["tenure"] = "not_a_number"
    
    elif error_type == "invalid_gender":
        # Use invalid gender value - use simple fixed values instead of filtering
        customer["gender"] = draw(st.sampled_from(["Other", "Unknown", "NonBinary", "X"]))
    
    elif error_type == "invalid_contract":
        # Use invalid contract value - use simple fixed values instead of filtering
        customer["contract"] = draw(st.sampled_from(["Invalid", "Weekly", "Lifetime", "X"]))
    
    return customer


@st.composite
def churn_probability_strategy(draw) -> float:
    """
    Generate random churn probability values.
    
    Returns:
        Float between 0.0 and 1.0 representing churn probability
    """
    return draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))


@st.composite
def low_risk_probability_strategy(draw) -> float:
    """
    Generate churn probability in the low risk range.
    
    Returns:
        Float between 0.0 and 0.33 (exclusive)
    """
    return draw(st.floats(min_value=0.0, max_value=0.33, exclude_max=True, allow_nan=False, allow_infinity=False))


@st.composite
def medium_risk_probability_strategy(draw) -> float:
    """
    Generate churn probability in the medium risk range.
    
    Returns:
        Float between 0.33 and 0.66 (exclusive)
    """
    return draw(st.floats(min_value=0.33, max_value=0.66, exclude_max=True, allow_nan=False, allow_infinity=False))


@st.composite
def high_risk_probability_strategy(draw) -> float:
    """
    Generate churn probability in the high risk range.
    
    Returns:
        Float between 0.66 and 1.0 (inclusive)
    """
    return draw(st.floats(min_value=0.66, max_value=1.0, allow_nan=False, allow_infinity=False))
