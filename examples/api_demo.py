"""
Demo script for the Customer Churn Prediction API.

This script demonstrates how to:
1. Start the FastAPI server
2. Make prediction requests
3. Check health status
4. Get model information

To run the API server:
    python -m uvicorn src.api.app:app --reload --port 8000

Or run this demo (which will make requests to a running server):
    python examples/api_demo.py
"""

import requests
import json
from typing import Dict, Any


# API configuration
API_BASE_URL = "http://localhost:8000"


def check_health() -> Dict[str, Any]:
    """
    Check the health status of the API.
    
    Returns:
        Health status response
    """
    print("\n=== Checking API Health ===")
    response = requests.get(f"{API_BASE_URL}/health")
    
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")
    
    return data


def get_model_info() -> Dict[str, Any]:
    """
    Get information about the loaded model.
    
    Returns:
        Model information response
    """
    print("\n=== Getting Model Info ===")
    response = requests.get(f"{API_BASE_URL}/model-info")
    
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")
    
    return data


def make_prediction(customer_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make a churn prediction for a customer.
    
    Args:
        customer_data: Dictionary containing customer attributes
        
    Returns:
        Prediction response
    """
    print("\n=== Making Prediction ===")
    print(f"Customer Data: {json.dumps(customer_data, indent=2)}")
    
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json=customer_data
    )
    
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")
    
    return data


def demo_valid_prediction():
    """Demonstrate a valid prediction request."""
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
        "streaming_movies": "Yes",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 70.35,
        "total_charges": 844.20
    }
    
    result = make_prediction(customer_data)
    
    if "churn_probability" in result:
        print(f"\n✓ Prediction successful!")
        print(f"  Churn Probability: {result['churn_probability']:.2%}")
        print(f"  Risk Level: {result['risk_label']}")
        print(f"  Model Version: {result['model_version']}")
    
    return result


def demo_invalid_prediction():
    """Demonstrate an invalid prediction request (missing fields)."""
    print("\n=== Testing Invalid Request (Missing Fields) ===")
    
    incomplete_data = {
        "gender": "Female",
        "senior_citizen": 0
        # Missing many required fields
    }
    
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json=incomplete_data
    )
    
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")
    
    if response.status_code == 400:
        print("\n✓ Validation error correctly returned!")
    
    return data


def demo_high_risk_customer():
    """Demonstrate prediction for a high-risk customer."""
    print("\n=== High Risk Customer Example ===")
    
    # Customer with characteristics indicating high churn risk
    high_risk_customer = {
        "gender": "Male",
        "senior_citizen": 1,
        "partner": "No",
        "dependents": "No",
        "tenure": 1,  # Very short tenure
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Month-to-month",  # Short-term contract
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 85.00,  # High charges
        "total_charges": 85.00  # Low total (new customer)
    }
    
    result = make_prediction(high_risk_customer)
    
    if "churn_probability" in result:
        print(f"\n✓ High risk customer identified!")
        print(f"  Churn Probability: {result['churn_probability']:.2%}")
        print(f"  Risk Level: {result['risk_label']}")
    
    return result


def demo_low_risk_customer():
    """Demonstrate prediction for a low-risk customer."""
    print("\n=== Low Risk Customer Example ===")
    
    # Customer with characteristics indicating low churn risk
    low_risk_customer = {
        "gender": "Female",
        "senior_citizen": 0,
        "partner": "Yes",
        "dependents": "Yes",
        "tenure": 60,  # Long tenure
        "phone_service": "Yes",
        "multiple_lines": "Yes",
        "internet_service": "Fiber optic",
        "online_security": "Yes",
        "online_backup": "Yes",
        "device_protection": "Yes",
        "tech_support": "Yes",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "contract": "Two year",  # Long-term contract
        "paperless_billing": "Yes",
        "payment_method": "Credit card (automatic)",
        "monthly_charges": 95.00,
        "total_charges": 5700.00  # High total (loyal customer)
    }
    
    result = make_prediction(low_risk_customer)
    
    if "churn_probability" in result:
        print(f"\n✓ Low risk customer identified!")
        print(f"  Churn Probability: {result['churn_probability']:.2%}")
        print(f"  Risk Level: {result['risk_label']}")
    
    return result


def main():
    """Run all demo examples."""
    print("=" * 60)
    print("Customer Churn Prediction API Demo")
    print("=" * 60)
    
    try:
        # Check API health
        health = check_health()
        
        if health.get("status") != "healthy":
            print("\n⚠ Warning: API is not healthy. Model may not be loaded.")
            print("Make sure the API server is running and the model is loaded.")
            return
        
        # Get model information
        get_model_info()
        
        # Demo valid prediction
        demo_valid_prediction()
        
        # Demo invalid prediction
        demo_invalid_prediction()
        
        # Demo high risk customer
        demo_high_risk_customer()
        
        # Demo low risk customer
        demo_low_risk_customer()
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API server.")
        print("Make sure the server is running:")
        print("  python -m uvicorn src.api.app:app --reload --port 8000")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
