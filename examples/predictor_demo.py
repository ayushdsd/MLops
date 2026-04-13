"""
Demo script for the Predictor module.

This script demonstrates how to use the Predictor class to:
1. Load a model from MLflow Model Registry
2. Load a preprocessing pipeline
3. Make predictions on customer data
4. Classify risk levels

Note: This is a demonstration script. In production, the predictor would be
integrated into the FastAPI application.
"""

import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.predictor import Predictor, ModelNotLoadedError, PredictorError


def demo_basic_usage():
    """Demonstrate basic predictor usage."""
    print("=" * 60)
    print("Demo 1: Basic Predictor Usage")
    print("=" * 60)
    
    # Initialize predictor
    predictor = Predictor(mlflow_uri="http://localhost:5000")
    print(f"✓ Predictor initialized")
    print(f"  Ready: {predictor.is_ready()}")
    
    # Load model from MLflow Model Registry
    try:
        predictor.load_model(model_name="churn_model", stage="Production")
        print(f"✓ Model loaded successfully")
        print(f"  Ready: {predictor.is_ready()}")
    except PredictorError as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Get model info
    info = predictor.get_model_info()
    print(f"\nModel Information:")
    print(f"  Name: {info['model_name']}")
    print(f"  Version: {info['model_version']}")
    print(f"  Model Loaded: {info['model_loaded']}")
    print(f"  Pipeline Loaded: {info['preprocessing_pipeline_loaded']}")
    
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
    
    # Make prediction
    try:
        result = predictor.predict(customer_data)
        print(f"\nPrediction Result:")
        print(f"  Churn Probability: {result.churn_probability:.2%}")
        print(f"  Risk Level: {result.risk_label}")
    except (ModelNotLoadedError, PredictorError) as e:
        print(f"✗ Prediction failed: {e}")


def demo_risk_classification():
    """Demonstrate risk classification."""
    print("\n" + "=" * 60)
    print("Demo 2: Risk Classification")
    print("=" * 60)
    
    test_probabilities = [0.05, 0.15, 0.32, 0.33, 0.50, 0.65, 0.66, 0.85, 0.95]
    
    print("\nChurn Probability → Risk Level:")
    print("-" * 40)
    for prob in test_probabilities:
        risk = Predictor.classify_risk(prob)
        print(f"  {prob:.2f} ({prob:.0%}) → {risk}")
    
    print("\nRisk Level Thresholds:")
    print("  Low:    0.00 ≤ probability < 0.33")
    print("  Medium: 0.33 ≤ probability < 0.66")
    print("  High:   0.66 ≤ probability ≤ 1.00")


def demo_error_handling():
    """Demonstrate error handling."""
    print("\n" + "=" * 60)
    print("Demo 3: Error Handling")
    print("=" * 60)
    
    predictor = Predictor(mlflow_uri="http://localhost:5000")
    
    # Try to predict without loading model
    print("\nAttempting prediction without loading model...")
    try:
        customer_data = {"gender": "Male", "tenure": 24}
        result = predictor.predict(customer_data)
        print(f"✓ Prediction succeeded: {result.churn_probability}")
    except ModelNotLoadedError as e:
        print(f"✗ Expected error caught: {e}")
    
    # Try to load non-existent model
    print("\nAttempting to load non-existent model...")
    try:
        predictor.load_model(model_name="non_existent_model", stage="Production")
        print(f"✓ Model loaded")
    except PredictorError as e:
        print(f"✗ Expected error caught: Failed to load model")


def demo_with_preprocessing_pipeline():
    """Demonstrate usage with preprocessing pipeline."""
    print("\n" + "=" * 60)
    print("Demo 4: With Preprocessing Pipeline")
    print("=" * 60)
    
    predictor = Predictor(mlflow_uri="http://localhost:5000")
    
    # Load model
    try:
        predictor.load_model(model_name="churn_model", stage="Production")
        print(f"✓ Model loaded")
    except PredictorError as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Load preprocessing pipeline from MLflow run
    print("\nLoading preprocessing pipeline from MLflow run...")
    try:
        predictor.load_preprocessing_pipeline(run_id="abc123def456")
        print(f"✓ Pipeline loaded from MLflow run")
    except PredictorError as e:
        print(f"✗ Failed to load pipeline from MLflow: {e}")
    
    # Alternative: Load from local file
    print("\nLoading preprocessing pipeline from local file...")
    try:
        predictor.load_preprocessing_pipeline(pipeline_path="models/pipelines/preprocessing.pkl")
        print(f"✓ Pipeline loaded from local file")
    except PredictorError as e:
        print(f"✗ Failed to load pipeline from file: {e}")
    
    # Get updated model info
    info = predictor.get_model_info()
    print(f"\nModel Information:")
    print(f"  Model Loaded: {info['model_loaded']}")
    print(f"  Pipeline Loaded: {info['preprocessing_pipeline_loaded']}")


def demo_multiple_predictions():
    """Demonstrate making multiple predictions."""
    print("\n" + "=" * 60)
    print("Demo 5: Multiple Predictions")
    print("=" * 60)
    
    predictor = Predictor(mlflow_uri="http://localhost:5000")
    
    # Load model
    try:
        predictor.load_model(model_name="churn_model", stage="Production")
        print(f"✓ Model loaded")
    except PredictorError as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Multiple customer scenarios
    customers = [
        {
            "name": "Low Risk Customer",
            "data": {
                "gender": "Male",
                "senior_citizen": 0,
                "partner": "Yes",
                "dependents": "Yes",
                "tenure": 60,
                "contract": "Two year",
                "monthly_charges": 50.0,
                "total_charges": 3000.0,
                # ... other fields
            }
        },
        {
            "name": "Medium Risk Customer",
            "data": {
                "gender": "Female",
                "senior_citizen": 0,
                "partner": "No",
                "dependents": "No",
                "tenure": 24,
                "contract": "One year",
                "monthly_charges": 65.0,
                "total_charges": 1560.0,
                # ... other fields
            }
        },
        {
            "name": "High Risk Customer",
            "data": {
                "gender": "Female",
                "senior_citizen": 1,
                "partner": "No",
                "dependents": "No",
                "tenure": 3,
                "contract": "Month-to-month",
                "monthly_charges": 85.0,
                "total_charges": 255.0,
                # ... other fields
            }
        }
    ]
    
    print("\nMaking predictions for multiple customers:")
    print("-" * 60)
    
    for customer in customers:
        try:
            # Note: In real usage, all required fields must be present
            # This is just a demonstration
            print(f"\n{customer['name']}:")
            print(f"  Tenure: {customer['data']['tenure']} months")
            print(f"  Contract: {customer['data']['contract']}")
            print(f"  Monthly Charges: ${customer['data']['monthly_charges']:.2f}")
            # result = predictor.predict(customer['data'])
            # print(f"  Churn Probability: {result.churn_probability:.2%}")
            # print(f"  Risk Level: {result.risk_label}")
            print(f"  (Prediction skipped - incomplete data)")
        except (ModelNotLoadedError, PredictorError) as e:
            print(f"  ✗ Prediction failed: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PREDICTOR MODULE DEMONSTRATION")
    print("=" * 60)
    print("\nThis demo shows how to use the Predictor class for")
    print("customer churn prediction with MLflow integration.")
    print("\nNote: Some demos require a running MLflow server and")
    print("trained models in the Model Registry.")
    
    # Run demos
    demo_risk_classification()
    demo_error_handling()
    
    # These demos require MLflow server and models
    print("\n" + "=" * 60)
    print("The following demos require MLflow server and models:")
    print("=" * 60)
    print("- demo_basic_usage()")
    print("- demo_with_preprocessing_pipeline()")
    print("- demo_multiple_predictions()")
    print("\nTo run these demos, ensure:")
    print("1. MLflow server is running at http://localhost:5000")
    print("2. A model named 'churn_model' is registered in Production stage")
    print("3. Preprocessing pipeline is available")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
