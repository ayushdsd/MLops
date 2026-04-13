"""
Integration tests for the preprocessing pipeline.

Tests the complete workflow from loading data to preprocessing.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_processing import DataProcessor


class TestPreprocessingIntegration:
    """Integration tests for preprocessing pipeline."""
    
    @pytest.fixture
    def processor(self):
        """Create a DataProcessor instance for testing."""
        return DataProcessor()
    
    @pytest.fixture
    def sample_dataset_file(self, tmp_path):
        """Create a sample dataset file for integration testing."""
        data = {
            'gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
            'SeniorCitizen': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'Partner': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
            'Dependents': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes'],
            'tenure': [12, 24, 6, 36, 18, 48, 3, 60, 9, 72],
            'PhoneService': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes'],
            'MultipleLines': ['No', 'No phone service', 'Yes', 'No', 'No phone service', 'Yes', 'No', 'No phone service', 'Yes', 'No'],
            'InternetService': ['DSL', 'Fiber optic', 'No', 'DSL', 'Fiber optic', 'No', 'DSL', 'Fiber optic', 'No', 'DSL'],
            'OnlineSecurity': ['Yes', 'No', 'No internet service', 'Yes', 'No', 'No internet service', 'Yes', 'No', 'No internet service', 'Yes'],
            'OnlineBackup': ['No', 'Yes', 'No internet service', 'No', 'Yes', 'No internet service', 'No', 'Yes', 'No internet service', 'No'],
            'DeviceProtection': ['Yes', 'No', 'No internet service', 'Yes', 'No', 'No internet service', 'Yes', 'No', 'No internet service', 'Yes'],
            'TechSupport': ['No', 'Yes', 'No internet service', 'No', 'Yes', 'No internet service', 'No', 'Yes', 'No internet service', 'No'],
            'StreamingTV': ['Yes', 'No', 'No internet service', 'Yes', 'No', 'No internet service', 'Yes', 'No', 'No internet service', 'Yes'],
            'StreamingMovies': ['No', 'Yes', 'No internet service', 'No', 'Yes', 'No internet service', 'No', 'Yes', 'No internet service', 'No'],
            'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'One year', 'Two year', 'Month-to-month', 'One year', 'Two year', 'Month-to-month'],
            'PaperlessBilling': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)', 
                            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)',
                            'Electronic check', 'Mailed check'],
            'MonthlyCharges': [50.0, 75.5, 30.25, 85.0, 60.0, 90.5, 45.0, 100.0, 55.0, 70.0],
            'TotalCharges': [600.0, 1812.0, 181.5, 3060.0, 1080.0, 4344.0, 135.0, 6000.0, 495.0, 5040.0],
            'Churn': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes']
        }
        df = pd.DataFrame(data)
        
        csv_path = tmp_path / "test_churn_data.csv"
        df.to_csv(csv_path, index=False)
        
        return csv_path
    
    def test_end_to_end_preprocessing_workflow(self, processor, sample_dataset_file):
        """Test complete workflow: load -> validate -> preprocess."""
        # Step 1: Load data
        df = processor.load_data(str(sample_dataset_file))
        assert df is not None
        assert len(df) == 10
        
        # Step 2: Validate schema
        validation_result = processor.validate_schema(df)
        assert validation_result.is_valid is True
        
        # Step 3: Preprocess data
        preprocessed = processor.preprocess(df)
        
        # Verify preprocessing results
        assert preprocessed.X_train is not None
        assert preprocessed.X_test is not None
        assert preprocessed.y_train is not None
        assert preprocessed.y_test is not None
        
        # Verify no null values
        assert not np.isnan(preprocessed.X_train).any()
        assert not np.isnan(preprocessed.X_test).any()
        
        # Verify all values are numerical
        assert preprocessed.X_train.dtype in [np.float64, np.float32, np.int64, np.int32]
        assert preprocessed.X_test.dtype in [np.float64, np.float32, np.int64, np.int32]
        
        # Verify train-test split
        total_samples = len(df)
        assert len(preprocessed.X_train) + len(preprocessed.X_test) == total_samples
        
        # Verify 80-20 split (approximately)
        train_ratio = len(preprocessed.X_train) / total_samples
        assert 0.7 < train_ratio < 0.9  # Allow some tolerance
        
        # Verify transformers were fitted
        assert preprocessed.scaler is not None
        assert len(preprocessed.label_encoders) > 0
        assert len(preprocessed.feature_names) > 0
    
    def test_preprocessing_with_null_values(self, processor, tmp_path):
        """Test preprocessing handles null values correctly."""
        # Create dataset with null values
        data = {
            'gender': ['Male', 'Female', None, 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
            'SeniorCitizen': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'Partner': ['Yes', None, 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
            'Dependents': ['No', 'Yes', 'No', None, 'No', 'Yes', 'No', 'Yes', 'No', 'Yes'],
            'tenure': [12, None, 6, 36, 18, 48, 3, 60, 9, 72],
            'PhoneService': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes'],
            'MultipleLines': ['No', 'No phone service', 'Yes', 'No', 'No phone service', 'Yes', 'No', 'No phone service', 'Yes', 'No'],
            'InternetService': ['DSL', 'Fiber optic', 'No', 'DSL', 'Fiber optic', 'No', 'DSL', 'Fiber optic', 'No', 'DSL'],
            'OnlineSecurity': ['Yes', 'No', 'No internet service', 'Yes', 'No', 'No internet service', 'Yes', 'No', 'No internet service', 'Yes'],
            'OnlineBackup': ['No', 'Yes', 'No internet service', 'No', 'Yes', 'No internet service', 'No', 'Yes', 'No internet service', 'No'],
            'DeviceProtection': ['Yes', 'No', 'No internet service', 'Yes', 'No', 'No internet service', 'Yes', 'No', 'No internet service', 'Yes'],
            'TechSupport': ['No', 'Yes', 'No internet service', 'No', 'Yes', 'No internet service', 'No', 'Yes', 'No internet service', 'No'],
            'StreamingTV': ['Yes', 'No', 'No internet service', 'Yes', 'No', 'No internet service', 'Yes', 'No', 'No internet service', 'Yes'],
            'StreamingMovies': ['No', 'Yes', 'No internet service', 'No', 'Yes', 'No internet service', 'No', 'Yes', 'No internet service', 'No'],
            'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'One year', 'Two year', 'Month-to-month', 'One year', 'Two year', 'Month-to-month'],
            'PaperlessBilling': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)', 
                            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)',
                            'Electronic check', 'Mailed check'],
            'MonthlyCharges': [50.0, None, 30.25, 85.0, 60.0, 90.5, 45.0, 100.0, 55.0, 70.0],
            'TotalCharges': [600.0, 1812.0, None, 3060.0, 1080.0, 4344.0, 135.0, 6000.0, 495.0, 5040.0],
            'Churn': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes']
        }
        df = pd.DataFrame(data)
        
        csv_path = tmp_path / "data_with_nulls.csv"
        df.to_csv(csv_path, index=False)
        
        # Load and preprocess
        df_loaded = processor.load_data(str(csv_path))
        preprocessed = processor.preprocess(df_loaded)
        
        # Verify no null values remain after preprocessing
        assert not np.isnan(preprocessed.X_train).any()
        assert not np.isnan(preprocessed.X_test).any()
    
    def test_preprocessing_produces_consistent_feature_count(self, processor, sample_dataset_file):
        """Test that preprocessing produces consistent feature count."""
        df = processor.load_data(str(sample_dataset_file))
        preprocessed = processor.preprocess(df)
        
        # All samples should have same number of features
        n_features = preprocessed.X_train.shape[1]
        assert preprocessed.X_test.shape[1] == n_features
        assert len(preprocessed.feature_names) == n_features
        
        # Feature count should match original columns minus target
        expected_features = len(df.columns) - 1  # Minus 'Churn'
        assert n_features == expected_features
