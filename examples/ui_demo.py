"""
Demo script for Streamlit UI components.

This script demonstrates how to use the UI components programmatically
for testing or integration purposes.
"""

from src.ui.components import call_prediction_api


def demo_api_call():
    """
    Demonstrate calling the prediction API with sample customer data.
    """
    print("=" * 60)
    print("Streamlit UI Component Demo")
    print("=" * 60)
    
    # Sample customer data
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
    
    print("\n1. Sample Customer Data:")
    print("-" * 60)
    for key, value in customer_data.items():
        print(f"  {key:20s}: {value}")
    
    print("\n2. Calling Prediction API...")
    print("-" * 60)
    
    try:
        # Call API with retry logic
        response = call_prediction_api(
            customer_data=customer_data,
            api_url="http://localhost:8000",
            timeout=5,
            max_retries=3
        )
        
        print("✓ API call successful!")
        print("\n3. Prediction Results:")
        print("-" * 60)
        print(f"  Churn Probability: {response['churn_probability']:.2%}")
        print(f"  Risk Label:        {response['risk_label']}")
        print(f"  Model Version:     {response['model_version']}")
        print(f"  Timestamp:         {response['timestamp']}")
        
        # Interpret risk level
        print("\n4. Risk Interpretation:")
        print("-" * 60)
        risk_label = response['risk_label']
        probability = response['churn_probability']
        
        if risk_label == "Low":
            print("  🟢 Low Risk: Customer is unlikely to churn")
            print(f"     Probability: {probability:.1%} (0-33% range)")
        elif risk_label == "Medium":
            print("  🟡 Medium Risk: Customer may churn - consider retention actions")
            print(f"     Probability: {probability:.1%} (33-66% range)")
        elif risk_label == "High":
            print("  🔴 High Risk: Customer is likely to churn - immediate action recommended")
            print(f"     Probability: {probability:.1%} (66-100% range)")
        
    except ValueError as e:
        print(f"✗ Validation Error: {e}")
        print("\nTroubleshooting:")
        print("  - Check that all required fields are present")
        print("  - Verify numerical values are within valid ranges")
        print("  - Ensure categorical values match expected options")
    
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nTroubleshooting:")
        print("  - Ensure the prediction API is running at http://localhost:8000")
        print("  - Check network connectivity")
        print("  - Verify API health: curl http://localhost:8000/health")
    
    print("\n" + "=" * 60)


def demo_error_handling():
    """
    Demonstrate error handling with invalid data.
    """
    print("\n" + "=" * 60)
    print("Error Handling Demo")
    print("=" * 60)
    
    # Invalid customer data (missing required fields)
    invalid_data = {
        "gender": "Male",
        "senior_citizen": 0
        # Missing many required fields
    }
    
    print("\n1. Invalid Customer Data (missing fields):")
    print("-" * 60)
    for key, value in invalid_data.items():
        print(f"  {key:20s}: {value}")
    
    print("\n2. Calling Prediction API...")
    print("-" * 60)
    
    try:
        response = call_prediction_api(
            customer_data=invalid_data,
            api_url="http://localhost:8000",
            timeout=5,
            max_retries=1  # Fewer retries for demo
        )
        print("✓ API call successful (unexpected)")
        
    except ValueError as e:
        print(f"✓ Validation Error caught (expected): {e}")
        print("\nThis demonstrates proper validation error handling.")
    
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    print("\n" + "=" * 60)


def demo_retry_logic():
    """
    Demonstrate retry logic with custom configuration.
    """
    print("\n" + "=" * 60)
    print("Retry Logic Demo")
    print("=" * 60)
    
    customer_data = {
        "gender": "Male",
        "senior_citizen": 1,
        "partner": "No",
        "dependents": "No",
        "tenure": 24,
        "phone_service": "Yes",
        "multiple_lines": "Yes",
        "internet_service": "DSL",
        "online_security": "Yes",
        "online_backup": "No",
        "device_protection": "Yes",
        "tech_support": "Yes",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Two year",
        "paperless_billing": "No",
        "payment_method": "Bank transfer (automatic)",
        "monthly_charges": 85.50,
        "total_charges": 2052.00
    }
    
    print("\n1. Configuration:")
    print("-" * 60)
    print("  API URL:      http://localhost:8000")
    print("  Timeout:      10 seconds")
    print("  Max Retries:  5")
    
    print("\n2. Calling API with retry logic...")
    print("-" * 60)
    
    try:
        response = call_prediction_api(
            customer_data=customer_data,
            api_url="http://localhost:8000",
            timeout=10,
            max_retries=5
        )
        
        print("✓ API call successful!")
        print(f"\n  Churn Probability: {response['churn_probability']:.2%}")
        print(f"  Risk Label:        {response['risk_label']}")
        
    except Exception as e:
        print(f"✗ Error after retries: {e}")
        print("\nRetry logic with exponential backoff:")
        print("  Attempt 1: Wait 1 second")
        print("  Attempt 2: Wait 2 seconds")
        print("  Attempt 3: Wait 4 seconds")
        print("  Attempt 4: Wait 8 seconds")
        print("  Attempt 5: Wait 16 seconds")
    
    print("\n" + "=" * 60)


def main():
    """
    Run all demos.
    """
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "Streamlit UI Component Demos" + " " * 20 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # Demo 1: Normal API call
    demo_api_call()
    
    # Demo 2: Error handling
    demo_error_handling()
    
    # Demo 3: Retry logic
    demo_retry_logic()
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nTo run the Streamlit UI application:")
    print("  streamlit run src/ui/app.py")
    print("\nTo access the UI:")
    print("  http://localhost:8501")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
