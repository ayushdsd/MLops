"""
Demonstration of model persistence functionality in TrainingService.

This script demonstrates how to:
1. Train a Random Forest model
2. Save the model to disk with version-based naming
3. Load the model from disk
4. Verify the loaded model produces the same predictions
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.training.trainer import TrainingService, TrainingConfig

def main():
    print("=" * 60)
    print("Model Persistence Demonstration")
    print("=" * 60)
    
    # Create sample training data
    print("\n1. Creating sample training data...")
    np.random.seed(42)
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(20, 10)
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")
    
    # Initialize training service
    print("\n2. Initializing TrainingService...")
    config = TrainingConfig(n_estimators=50, max_depth=10, random_state=42)
    service = TrainingService(mlflow_uri="http://localhost:5000", config=config)
    print(f"   Configuration: n_estimators={config.n_estimators}, max_depth={config.max_depth}")
    
    # Train model
    print("\n3. Training Random Forest model...")
    model = service.train(X_train, y_train)
    print(f"   Model trained successfully")
    print(f"   Number of trees: {model.n_estimators}")
    
    # Get predictions from original model
    print("\n4. Getting predictions from original model...")
    original_predictions = model.predict(X_test)
    original_probabilities = model.predict_proba(X_test)
    print(f"   Sample predictions: {original_predictions[:5]}")
    print(f"   Sample probabilities: {original_probabilities[0]}")
    
    # Save model with version 1
    print("\n5. Saving model to disk (version 1)...")
    model_path_v1 = service.save_model(model, "churn_model", version=1)
    print(f"   Model saved to: {model_path_v1}")
    
    # Save model with version 2 (same model, different version)
    print("\n6. Saving model to disk (version 2)...")
    model_path_v2 = service.save_model(model, "churn_model", version=2)
    print(f"   Model saved to: {model_path_v2}")
    
    # Load model version 1
    print("\n7. Loading model from disk (version 1)...")
    loaded_model_v1 = service.load_model("churn_model", version=1)
    print(f"   Model loaded successfully")
    print(f"   Number of trees: {loaded_model_v1.n_estimators}")
    print(f"   Number of features: {loaded_model_v1.n_features_in_}")
    
    # Verify loaded model produces same predictions
    print("\n8. Verifying loaded model predictions...")
    loaded_predictions = loaded_model_v1.predict(X_test)
    loaded_probabilities = loaded_model_v1.predict_proba(X_test)
    
    predictions_match = np.array_equal(original_predictions, loaded_predictions)
    probabilities_match = np.allclose(original_probabilities, loaded_probabilities)
    
    print(f"   Predictions match: {predictions_match}")
    print(f"   Probabilities match: {probabilities_match}")
    
    if predictions_match and probabilities_match:
        print("\n✓ SUCCESS: Loaded model produces identical predictions!")
    else:
        print("\n✗ ERROR: Loaded model predictions differ from original!")
    
    # Load model version 2
    print("\n9. Loading model from disk (version 2)...")
    loaded_model_v2 = service.load_model("churn_model", version=2)
    print(f"   Model loaded successfully")
    
    # Demonstrate version management
    print("\n10. Version Management Summary:")
    print(f"    - Version 1 saved to: {model_path_v1}")
    print(f"    - Version 2 saved to: {model_path_v2}")
    print(f"    - Both versions can be loaded independently")
    print(f"    - Naming convention: {{model_name}}_v{{version}}.pkl")
    
    print("\n" + "=" * 60)
    print("Model Persistence Demonstration Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
