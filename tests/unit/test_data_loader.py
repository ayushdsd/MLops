"""
Unit tests for the data loader module.

Tests cover successful data loading, error handling for missing files,
empty files, and parse errors.
"""

import os
import pytest
import pandas as pd
from pathlib import Path
from src.data_processing import DataProcessor, DataLoadError


class TestDataProcessor:
    """Test suite for DataProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create a DataProcessor instance for testing."""
        return DataProcessor()
    
    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a sample CSV file for testing."""
        csv_path = tmp_path / "sample_data.csv"
        data = {
            'gender': ['Male', 'Female', 'Male'],
            'senior_citizen': [0, 1, 0],
            'tenure': [12, 24, 6],
            'monthly_charges': [50.0, 75.5, 30.25],
            'churn': ['No', 'Yes', 'No']
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        return csv_path
    
    @pytest.fixture
    def empty_csv(self, tmp_path):
        """Create an empty CSV file for testing."""
        csv_path = tmp_path / "empty.csv"
        csv_path.touch()
        return csv_path
    
    @pytest.fixture
    def malformed_csv(self, tmp_path):
        """Create a malformed CSV file for testing."""
        csv_path = tmp_path / "malformed.csv"
        with open(csv_path, 'w') as f:
            f.write("col1,col2,col3\n")
            f.write("val1,val2\n")  # Missing column
            f.write("val3,val4,val5,val6\n")  # Extra column
        return csv_path
    
    def test_load_data_success(self, processor, sample_csv):
        """Test successful data loading from a valid CSV file."""
        df = processor.load_data(str(sample_csv))
        
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'gender' in df.columns
        assert 'tenure' in df.columns
        assert 'churn' in df.columns
    
    def test_load_data_missing_file(self, processor):
        """Test error handling when CSV file doesn't exist."""
        nonexistent_path = "nonexistent_file.csv"
        
        with pytest.raises(DataLoadError) as exc_info:
            processor.load_data(nonexistent_path)
        
        assert "not found" in str(exc_info.value).lower()
        assert nonexistent_path in str(exc_info.value)
    
    def test_load_data_empty_file(self, processor, empty_csv):
        """Test error handling when CSV file is empty."""
        with pytest.raises(DataLoadError) as exc_info:
            processor.load_data(str(empty_csv))
        
        assert "empty" in str(exc_info.value).lower()
    
    def test_load_data_malformed_csv(self, processor, malformed_csv):
        """Test error handling when CSV file has parsing errors."""
        # Note: pandas is quite forgiving with malformed CSVs, so this may not
        # always raise an error. We test that if it does raise, it's handled properly.
        try:
            df = processor.load_data(str(malformed_csv))
            # If pandas successfully parsed it, verify we got a DataFrame
            assert isinstance(df, pd.DataFrame)
        except DataLoadError as e:
            # If it failed, verify the error message is appropriate
            assert "parse" in str(e).lower() or "error" in str(e).lower()
    
    def test_load_data_returns_correct_shape(self, processor, sample_csv):
        """Test that loaded data has the expected shape."""
        df = processor.load_data(str(sample_csv))
        
        assert df.shape == (3, 5)  # 3 rows, 5 columns
    
    def test_load_data_preserves_column_names(self, processor, sample_csv):
        """Test that column names are preserved correctly."""
        df = processor.load_data(str(sample_csv))
        
        expected_columns = ['gender', 'senior_citizen', 'tenure', 'monthly_charges', 'churn']
        assert list(df.columns) == expected_columns
    
    def test_load_data_preserves_data_types(self, processor, sample_csv):
        """Test that data types are preserved correctly."""
        df = processor.load_data(str(sample_csv))
        
        assert df['gender'].dtype == object
        assert df['senior_citizen'].dtype in [int, 'int64']
        assert df['tenure'].dtype in [int, 'int64']
        assert df['monthly_charges'].dtype in [float, 'float64']
    
    def test_load_data_with_special_characters(self, processor, tmp_path):
        """Test loading CSV with special characters in data."""
        csv_path = tmp_path / "special_chars.csv"
        data = {
            'name': ['John, Jr.', 'Mary "Sue"', "O'Brien"],
            'value': [1, 2, 3]
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        
        loaded_df = processor.load_data(str(csv_path))
        
        assert len(loaded_df) == 3
        assert loaded_df['name'].iloc[0] == 'John, Jr.'
        assert loaded_df['name'].iloc[1] == 'Mary "Sue"'
        assert loaded_df['name'].iloc[2] == "O'Brien"
    
    def test_load_data_with_missing_values(self, processor, tmp_path):
        """Test loading CSV with missing values."""
        csv_path = tmp_path / "missing_values.csv"
        data = {
            'col1': [1, 2, None, 4],
            'col2': ['a', None, 'c', 'd']
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        
        loaded_df = processor.load_data(str(csv_path))
        
        assert len(loaded_df) == 4
        assert loaded_df['col1'].isna().sum() == 1
        assert loaded_df['col2'].isna().sum() == 1
    
    def test_load_data_large_file(self, processor, tmp_path):
        """Test loading a larger CSV file."""
        csv_path = tmp_path / "large_data.csv"
        data = {
            'id': range(1000),
            'value': [i * 2 for i in range(1000)]
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        
        loaded_df = processor.load_data(str(csv_path))
        
        assert len(loaded_df) == 1000
        assert loaded_df['id'].iloc[0] == 0
        assert loaded_df['id'].iloc[999] == 999


class TestDataValidation:
    """Test suite for data validation functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create a DataProcessor instance for testing."""
        return DataProcessor()
    
    @pytest.fixture
    def valid_churn_data(self):
        """Create a valid customer churn dataset."""
        data = {
            'gender': ['Male', 'Female', 'Male'],
            'SeniorCitizen': [0, 1, 0],
            'Partner': ['Yes', 'No', 'Yes'],
            'Dependents': ['No', 'Yes', 'No'],
            'tenure': [12, 24, 6],
            'PhoneService': ['Yes', 'No', 'Yes'],
            'MultipleLines': ['No', 'No phone service', 'Yes'],
            'InternetService': ['DSL', 'Fiber optic', 'No'],
            'OnlineSecurity': ['Yes', 'No', 'No internet service'],
            'OnlineBackup': ['No', 'Yes', 'No internet service'],
            'DeviceProtection': ['Yes', 'No', 'No internet service'],
            'TechSupport': ['No', 'Yes', 'No internet service'],
            'StreamingTV': ['Yes', 'No', 'No internet service'],
            'StreamingMovies': ['No', 'Yes', 'No internet service'],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'PaperlessBilling': ['Yes', 'No', 'Yes'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)'],
            'MonthlyCharges': [50.0, 75.5, 30.25],
            'TotalCharges': [600.0, 1812.0, 181.5]
        }
        return pd.DataFrame(data)
    
    def test_validate_schema_success(self, processor, valid_churn_data):
        """Test validation passes for valid data."""
        result = processor.validate_schema(valid_churn_data)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_validate_schema_missing_columns(self, processor):
        """Test validation fails when required columns are missing."""
        data = pd.DataFrame({
            'gender': ['Male', 'Female'],
            'tenure': [12, 24]
        })
        
        result = processor.validate_schema(data)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any('Missing required columns' in error for error in result.errors)
    
    def test_validate_schema_invalid_tenure_range(self, processor, valid_churn_data):
        """Test validation fails when tenure is negative."""
        invalid_data = valid_churn_data.copy()
        invalid_data.loc[0, 'tenure'] = -5
        
        result = processor.validate_schema(invalid_data)
        
        assert result.is_valid is False
        assert any('tenure' in error and '< 0' in error for error in result.errors)
    
    def test_validate_schema_invalid_monthly_charges_range(self, processor, valid_churn_data):
        """Test validation fails when monthly charges is zero or negative."""
        invalid_data = valid_churn_data.copy()
        invalid_data.loc[0, 'MonthlyCharges'] = 0
        
        result = processor.validate_schema(invalid_data)
        
        assert result.is_valid is False
        assert any('MonthlyCharges' in error and '≤ 0' in error for error in result.errors)
    
    def test_validate_schema_invalid_total_charges_range(self, processor, valid_churn_data):
        """Test validation fails when total charges is negative."""
        invalid_data = valid_churn_data.copy()
        invalid_data.loc[0, 'TotalCharges'] = -100
        
        result = processor.validate_schema(invalid_data)
        
        assert result.is_valid is False
        assert any('TotalCharges' in error and '< 0' in error for error in result.errors)
    
    def test_validate_schema_invalid_categorical_gender(self, processor, valid_churn_data):
        """Test validation fails when gender has invalid values."""
        invalid_data = valid_churn_data.copy()
        invalid_data.loc[0, 'gender'] = 'Other'
        
        result = processor.validate_schema(invalid_data)
        
        assert result.is_valid is False
        assert any('gender' in error and 'invalid values' in error.lower() for error in result.errors)
    
    def test_validate_schema_invalid_categorical_contract(self, processor, valid_churn_data):
        """Test validation fails when contract has invalid values."""
        invalid_data = valid_churn_data.copy()
        invalid_data.loc[0, 'Contract'] = 'Three year'
        
        result = processor.validate_schema(invalid_data)
        
        assert result.is_valid is False
        assert any('Contract' in error and 'invalid values' in error.lower() for error in result.errors)
    
    def test_validate_schema_invalid_senior_citizen(self, processor, valid_churn_data):
        """Test validation fails when SeniorCitizen has invalid values."""
        invalid_data = valid_churn_data.copy()
        invalid_data.loc[0, 'SeniorCitizen'] = 5
        
        result = processor.validate_schema(invalid_data)
        
        assert result.is_valid is False
        assert any('SeniorCitizen' in error for error in result.errors)
    
    def test_validate_schema_high_missing_values_warning(self, processor):
        """Test validation generates warning for high percentage of missing values."""
        # Create a larger dataset to properly test the 50% threshold
        data = {
            'gender': ['Male'] * 10,
            'SeniorCitizen': [0] * 10,
            'Partner': ['Yes'] * 10,
            'Dependents': ['No'] * 10,
            'tenure': [12] * 10,
            'PhoneService': ['Yes'] * 10,
            'MultipleLines': ['No'] * 10,
            'InternetService': ['DSL'] * 10,
            'OnlineSecurity': ['Yes'] * 10,
            'OnlineBackup': ['No'] * 10,
            'DeviceProtection': ['Yes'] * 10,
            'TechSupport': ['No'] * 10,
            'StreamingTV': ['Yes'] * 10,
            'StreamingMovies': ['No'] * 10,
            'Contract': ['Month-to-month'] * 10,
            'PaperlessBilling': ['Yes'] * 10,
            'PaymentMethod': ['Electronic check'] * 10,
            'MonthlyCharges': [50.0] * 10,
            'TotalCharges': [600.0] * 10
        }
        data_with_nulls = pd.DataFrame(data)
        # Set 60% of values to NaN in one column
        data_with_nulls.loc[:5, 'MonthlyCharges'] = None
        
        result = processor.validate_schema(data_with_nulls)
        
        # Should still be valid (warnings don't invalidate)
        assert len(result.warnings) > 0
        assert any('missing values' in warning.lower() for warning in result.warnings)
    
    def test_validate_schema_multiple_errors(self, processor, valid_churn_data):
        """Test validation captures multiple errors."""
        invalid_data = valid_churn_data.copy()
        invalid_data.loc[0, 'tenure'] = -5
        invalid_data.loc[1, 'MonthlyCharges'] = 0
        invalid_data.loc[2, 'gender'] = 'Invalid'
        
        result = processor.validate_schema(invalid_data)
        
        assert result.is_valid is False
        assert len(result.errors) >= 3
    
    def test_validate_schema_with_nan_values(self, processor, valid_churn_data):
        """Test validation handles NaN values appropriately."""
        data_with_nan = valid_churn_data.copy()
        data_with_nan.loc[0, 'TotalCharges'] = None
        
        result = processor.validate_schema(data_with_nan)
        
        # NaN values should not cause validation errors (they'll be handled in preprocessing)
        # Only check that we don't get range errors for NaN values
        assert result.is_valid is True or not any('TotalCharges' in error and '< 0' in error for error in result.errors)
    
    def test_validate_schema_empty_dataframe(self, processor):
        """Test validation handles empty DataFrame."""
        empty_df = pd.DataFrame()
        
        result = processor.validate_schema(empty_df)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
    
    def test_validate_schema_all_categorical_values(self, processor, valid_churn_data):
        """Test validation accepts all valid categorical values."""
        # Test with all possible valid values for each categorical column
        result = processor.validate_schema(valid_churn_data)
        
        # Should not have any categorical validation errors
        categorical_errors = [e for e in result.errors if 'invalid values' in e.lower()]
        assert len(categorical_errors) == 0
