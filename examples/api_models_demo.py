"""
Demo script showing usage of API Pydantic models.

This script demonstrates how to create and validate API request/response models.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from src.api.models import (
    CustomerInput,
    PredictionResponse,
    HealthResponse,
    ErrorResponse
)


def demo_customer_input():
    """Demonstrate CustomerInput model creation and validation."""
    print("=" * 60)
    print("CustomerInput Model Demo")
    print("=" * 60)
    
    # Valid customer data
    customer = CustomerInput(
        gender="Female",
        senior_citizen=0,
        partner="Yes",
        dependents="No",
        tenure=24,
        contract="One year",
        paperless_billing="Yes",
        payment_method="Credit card (automatic)",
        monthly_charges=70.35,
        total_charges=1688.40,
        phone_service="Yes",
        multiple_lines="No",
        internet_service="Fiber optic",
        online_security="Yes",
        online_backup="Yes",
        device_protection="No",
        tech_support="Yes",
        streaming_tv="No",
        streaming_movies="Yes"
    )
    
    print(f"✓ Created valid CustomerInput")
    print(f"  Gender: {customer.gender}")
    print(f"  Tenure: {customer.tenure} months")
    print(f"  Monthly Charges: ${customer.monthly_charges}")
    print(f"  Internet Service: {customer.internet_service}")
    print()


def demo_prediction_response():
    """Demonstrate PredictionResponse model creation."""
    print("=" * 60)
    print("PredictionResponse Model Demo")
    print("=" * 60)
    
    # Low risk prediction
    low_risk = PredictionResponse(
        churn_probability=0.15,
        risk_label="Low",
        model_version="v1.0.0",
        timestamp=datetime.utcnow().isoformat()
    )
    print(f"✓ Low Risk Prediction:")
    print(f"  Probability: {low_risk.churn_probability:.2%}")
    print(f"  Risk: {low_risk.risk_label}")
    print()
    
    # Medium risk prediction
    medium_risk = PredictionResponse(
        churn_probability=0.52,
        risk_label="Medium",
        model_version="v1.0.0",
        timestamp=datetime.utcnow().isoformat()
    )
    print(f"✓ Medium Risk Prediction:")
    print(f"  Probability: {medium_risk.churn_probability:.2%}")
    print(f"  Risk: {medium_risk.risk_label}")
    print()
    
    # High risk prediction
    high_risk = PredictionResponse(
        churn_probability=0.87,
        risk_label="High",
        model_version="v1.0.0",
        timestamp=datetime.utcnow().isoformat()
    )
    print(f"✓ High Risk Prediction:")
    print(f"  Probability: {high_risk.churn_probability:.2%}")
    print(f"  Risk: {high_risk.risk_label}")
    print()


def demo_health_response():
    """Demonstrate HealthResponse model creation."""
    print("=" * 60)
    print("HealthResponse Model Demo")
    print("=" * 60)
    
    # Healthy service
    healthy = HealthResponse(
        status="healthy",
        model_loaded=True,
        model_version="v1.0.0"
    )
    print(f"✓ Healthy Service:")
    print(f"  Status: {healthy.status}")
    print(f"  Model Loaded: {healthy.model_loaded}")
    print(f"  Model Version: {healthy.model_version}")
    print()
    
    # Unhealthy service
    unhealthy = HealthResponse(
        status="unhealthy",
        model_loaded=False,
        model_version=None
    )
    print(f"✓ Unhealthy Service:")
    print(f"  Status: {unhealthy.status}")
    print(f"  Model Loaded: {unhealthy.model_loaded}")
    print(f"  Model Version: {unhealthy.model_version}")
    print()


def demo_error_response():
    """Demonstrate ErrorResponse model creation."""
    print("=" * 60)
    print("ErrorResponse Model Demo")
    print("=" * 60)
    
    error = ErrorResponse(
        error="ValidationError",
        detail="Invalid customer data: gender must be 'Male' or 'Female'",
        timestamp=datetime.utcnow().isoformat(),
        path="/predict"
    )
    print(f"✓ Error Response:")
    print(f"  Error: {error.error}")
    print(f"  Detail: {error.detail}")
    print(f"  Path: {error.path}")
    print()


def demo_validation_errors():
    """Demonstrate validation error handling."""
    print("=" * 60)
    print("Validation Error Demo")
    print("=" * 60)
    
    try:
        # Invalid gender
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
    except Exception as e:
        print(f"✗ Caught validation error for invalid gender:")
        print(f"  {str(e)[:100]}...")
        print()
    
    try:
        # Negative monthly charges
        CustomerInput(
            gender="Male",
            senior_citizen=0,
            partner="Yes",
            dependents="No",
            tenure=12,
            contract="Month-to-month",
            paperless_billing="Yes",
            payment_method="Electronic check",
            monthly_charges=-10.0,
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
    except Exception as e:
        print(f"✗ Caught validation error for negative monthly_charges:")
        print(f"  {str(e)[:100]}...")
        print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "API Pydantic Models Demonstration" + " " * 15 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    demo_customer_input()
    demo_prediction_response()
    demo_health_response()
    demo_error_response()
    demo_validation_errors()
    
    print("=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
