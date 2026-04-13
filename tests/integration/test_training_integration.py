"""
Integration tests for the TrainingService with data processing.

This module tests the integration between DataProcessor and TrainingService
to ensure the complete training workflow works end-to-end.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.data_processing.data_loader import DataProcessor
from src.training.trainer import TrainingService, TrainingConfig


class TestTrainingIntegration:
    """Integration tests for training workflow."""
    
    def test_train_with_preprocessed_data(self, tmp_path):
        """Test complete workflow: load data -> preprocess -> train."""
        # Create sample dataset
        np.random.seed(42)
        data = {
            'gender': np.random.choice(['Male', 'Female'], 100),
            'SeniorCitizen': np.random.choice([0, 1], 100),
            'Partner': np.random.choice(['Yes', 'No'], 100),
            'Dependents': np.random.choice(['Yes', 'No'], 100),
            'tenure': np.random.randint(0, 72, 100),
            'PhoneService': np.random.choice(['Yes', 'No'], 100),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], 100),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], 100),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], 100),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], 100),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], 100),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], 100),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], 100),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], 100),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 100),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], 100),
            'PaymentMethod': np.random.choice([
                'Electronic check', 'Mailed check',
                'Bank transfer (automatic)', 'Credit card (automatic)'
            ], 100),
            'MonthlyCharges': np.random.uniform(20, 120, 100),
            'TotalCharges': np.random.uniform(20, 8000, 100),
            'Churn': np.random.choice(['Yes', 'No'], 100)
        }
        df = pd.DataFrame(data)
        
        # Save to CSV
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)
        
        # Step 1: Load and preprocess data
        processor = DataProcessor()
        loaded_df = processor.load_data(str(csv_path))
        preprocessed = processor.preprocess(loaded_df)
        
        # Step 2: Train model
        config = TrainingConfig(n_estimators=10, random_state=42)
        service = TrainingService(mlflow_uri="http://localhost:5000", config=config)
        model = service.train(preprocessed.X_train, preprocessed.y_train)
        
        # Verify model is trained and can make predictions
        assert model is not None
        assert hasattr(model, 'estimators_')
        
        # Test predictions on training data
        train_predictions = model.predict(preprocessed.X_train)
        assert len(train_predictions) == len(preprocessed.y_train)
        
        # Test predictions on test data
        test_predictions = model.predict(preprocessed.X_test)
        assert len(test_predictions) == len(preprocessed.y_test)
        
        # Verify predictions are valid (0 or 1)
        assert all(pred in [0, 1] for pred in train_predictions)
        assert all(pred in [0, 1] for pred in test_predictions)
    
    def test_train_with_different_hyperparameters(self, tmp_path):
        """Test training with various hyperparameter configurations."""
        # Create minimal dataset
        np.random.seed(42)
        X_train = np.random.rand(50, 10)
        y_train = np.random.randint(0, 2, 50)
        
        # Test different configurations
        configs = [
            TrainingConfig(n_estimators=5, max_depth=3, random_state=42),
            TrainingConfig(n_estimators=10, max_depth=5, random_state=42),
            TrainingConfig(n_estimators=15, max_depth=None, random_state=42),
        ]
        
        for config in configs:
            service = TrainingService(mlflow_uri="http://localhost:5000", config=config)
            model = service.train(X_train, y_train)
            
            # Verify hyperparameters are respected
            assert model.n_estimators == config.n_estimators
            assert model.max_depth == config.max_depth
            assert model.random_state == config.random_state
            
            # Verify model can make predictions
            predictions = model.predict(X_train)
            assert len(predictions) == len(y_train)

