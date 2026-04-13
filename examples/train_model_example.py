"""
Example script demonstrating the TrainingService usage.

This script shows how to use the TrainingService to train a Random Forest
classifier with MLflow integration.
"""

import numpy as np
from src.training.trainer import TrainingService, TrainingConfig

# Create sample training data
print("Creating sample training data...")
np.random.seed(42)
X_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, 100)

print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")

# Configure training with custom hyperparameters
print("\nConfiguring training service...")
config = TrainingConfig(
    n_estimators=50,
    max_depth=10,
    random_state=42
)

# Initialize training service with MLflow
service = TrainingService(
    mlflow_uri="http://localhost:5000",
    config=config
)

# Train the model
print("\nTraining Random Forest model...")
model = service.train(X_train, y_train)

print("\nTraining completed successfully!")
print(f"Model has {len(model.estimators_)} trees")
print(f"Model hyperparameters:")
print(f"  - n_estimators: {model.n_estimators}")
print(f"  - max_depth: {model.max_depth}")
print(f"  - random_state: {model.random_state}")

# Make predictions
print("\nMaking predictions on training data...")
predictions = model.predict(X_train)
accuracy = (predictions == y_train).mean()
print(f"Training accuracy: {accuracy:.2%}")

print("\nExample completed successfully!")

