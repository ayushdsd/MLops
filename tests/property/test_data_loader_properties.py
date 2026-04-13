"""
Property-based tests for the data loader module.

These tests use Hypothesis to verify universal properties that should hold
for all valid inputs to the data loading functionality.
"""

import pandas as pd
import pytest
import tempfile
from hypothesis import given, settings, strategies as st, HealthCheck
from pathlib import Path
from src.data_processing import DataProcessor, DataLoadError


# Hypothesis strategies for generating test data
@st.composite
def valid_dataframe_strategy(draw, min_rows=1, max_rows=100):
    """
    Generate a valid customer-like DataFrame for testing.
    
    Args:
        draw: Hypothesis draw function
        min_rows: Minimum number of rows
        max_rows: Maximum number of rows
        
    Returns:
        pd.DataFrame: Generated DataFrame
    """
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    
    data = {
        'gender': draw(st.lists(
            st.sampled_from(['Male', 'Female']),
            min_size=n_rows,
            max_size=n_rows
        )),
        'senior_citizen': draw(st.lists(
            st.integers(min_value=0, max_value=1),
            min_size=n_rows,
            max_size=n_rows
        )),
        'tenure': draw(st.lists(
            st.integers(min_value=0, max_value=72),
            min_size=n_rows,
            max_size=n_rows
        )),
        'monthly_charges': draw(st.lists(
            st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False),
            min_size=n_rows,
            max_size=n_rows
        )),
        'churn': draw(st.lists(
            st.sampled_from(['Yes', 'No']),
            min_size=n_rows,
            max_size=n_rows
        ))
    }
    
    return pd.DataFrame(data)


class TestDataLoaderProperties:
    """Property-based tests for data loading functionality."""
    
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(df=valid_dataframe_strategy(min_rows=1, max_rows=100))
    def test_property_1_data_loading_succeeds(self, df, tmp_path):
        """
        **Validates: Requirements 1.1, 1.4**
        
        Feature: customer-churn-mlops-pipeline, Property 1: Data Loading Succeeds for Valid CSV Files
        
        For any valid CSV file containing customer churn data with the required schema,
        the DataProcessor should successfully load it into a DataFrame without errors.
        """
        # Create processor
        processor = DataProcessor()
        
        # Save DataFrame to temporary CSV
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)
        
        # Load data
        loaded_df = processor.load_data(str(csv_path))
        
        # Verify successful loading
        assert loaded_df is not None
        assert isinstance(loaded_df, pd.DataFrame)
        assert len(loaded_df) == len(df)
        assert list(loaded_df.columns) == list(df.columns)
    
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(df=valid_dataframe_strategy(min_rows=1, max_rows=100))
    def test_property_2_data_persistence_round_trip(self, df, tmp_path):
        """
        **Validates: Requirements 1.2**
        
        Feature: customer-churn-mlops-pipeline, Property 2: Data Persistence Round-Trip
        
        For any loaded customer dataset, storing it to the /data directory and then
        loading it back should produce an equivalent dataset.
        """
        # Create processor
        processor = DataProcessor()
        
        # Save original DataFrame
        csv_path = tmp_path / "round_trip_data.csv"
        df.to_csv(csv_path, index=False)
        
        # Load data
        loaded_df = processor.load_data(str(csv_path))
        
        # Save loaded data to a new file
        csv_path_2 = tmp_path / "round_trip_data_2.csv"
        loaded_df.to_csv(csv_path_2, index=False)
        
        # Load again
        reloaded_df = processor.load_data(str(csv_path_2))
        
        # Verify equivalence
        assert len(loaded_df) == len(reloaded_df)
        assert list(loaded_df.columns) == list(reloaded_df.columns)
        
        # Compare data values (allowing for floating point precision)
        for col in loaded_df.columns:
            if loaded_df[col].dtype in ['float64', 'float32']:
                pd.testing.assert_series_equal(
                    loaded_df[col],
                    reloaded_df[col],
                    check_exact=False,
                    rtol=1e-5
                )
            else:
                pd.testing.assert_series_equal(loaded_df[col], reloaded_df[col])
    
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(
        df=valid_dataframe_strategy(min_rows=1, max_rows=100),
        extra_cols=st.integers(min_value=1, max_value=5)
    )
    def test_property_column_preservation(self, df, extra_cols, tmp_path):
        """
        Property: Column names and count are preserved during loading.
        
        For any DataFrame with N columns, loading it from CSV should produce
        a DataFrame with the same N columns and same column names.
        """
        # Create processor
        processor = DataProcessor()
        
        # Add extra columns to test with varying column counts
        for i in range(extra_cols):
            df[f'extra_col_{i}'] = range(len(df))
        
        # Save and load
        csv_path = tmp_path / "column_test.csv"
        df.to_csv(csv_path, index=False)
        loaded_df = processor.load_data(str(csv_path))
        
        # Verify column preservation
        assert len(loaded_df.columns) == len(df.columns)
        assert list(loaded_df.columns) == list(df.columns)
    
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(df=valid_dataframe_strategy(min_rows=1, max_rows=100))
    def test_property_row_count_preservation(self, df, tmp_path):
        """
        Property: Row count is preserved during loading.
        
        For any DataFrame with N rows, loading it from CSV should produce
        a DataFrame with exactly N rows.
        """
        # Create processor
        processor = DataProcessor()
        
        # Save and load
        csv_path = tmp_path / "row_test.csv"
        df.to_csv(csv_path, index=False)
        loaded_df = processor.load_data(str(csv_path))
        
        # Verify row count preservation
        assert len(loaded_df) == len(df)
    
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(
        n_rows=st.integers(min_value=1, max_value=1000),
        n_cols=st.integers(min_value=1, max_value=20)
    )
    def test_property_shape_preservation(self, n_rows, n_cols, tmp_path):
        """
        Property: DataFrame shape is preserved during loading.
        
        For any DataFrame with shape (N, M), loading it from CSV should produce
        a DataFrame with the same shape (N, M).
        """
        # Create processor
        processor = DataProcessor()
        
        # Create DataFrame with specified shape
        data = {f'col_{i}': range(n_rows) for i in range(n_cols)}
        df = pd.DataFrame(data)
        
        # Save and load
        csv_path = tmp_path / "shape_test.csv"
        df.to_csv(csv_path, index=False)
        loaded_df = processor.load_data(str(csv_path))
        
        # Verify shape preservation
        assert loaded_df.shape == df.shape
        assert loaded_df.shape == (n_rows, n_cols)
    
    @settings(max_examples=20)
    @given(filename=st.text(
        alphabet=st.characters(blacklist_categories=('Cs', 'Cc'), blacklist_characters='/\\:*?"<>|'),
        min_size=1,
        max_size=50
    ))
    def test_property_missing_file_raises_error(self, filename):
        """
        Property: Loading a non-existent file always raises DataLoadError.
        
        For any file path that doesn't exist, load_data should raise DataLoadError
        with a descriptive message.
        """
        # Create processor
        processor = DataProcessor()
        
        # Ensure file doesn't exist
        nonexistent_path = f"nonexistent_{filename}.csv"
        
        # Verify error is raised
        with pytest.raises(DataLoadError) as exc_info:
            processor.load_data(nonexistent_path)
        
        # Verify error message is descriptive
        error_msg = str(exc_info.value).lower()
        assert "not found" in error_msg or "does not exist" in error_msg
    
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(df=valid_dataframe_strategy(min_rows=1, max_rows=100))
    def test_property_data_type_categories_preserved(self, df, tmp_path):
        """
        Property: Data type categories are preserved during loading.
        
        For any DataFrame, numeric columns should remain numeric and
        string columns should remain string after loading.
        """
        # Create processor
        processor = DataProcessor()
        
        # Save and load
        csv_path = tmp_path / "dtype_test.csv"
        df.to_csv(csv_path, index=False)
        loaded_df = processor.load_data(str(csv_path))
        
        # Verify numeric columns remain numeric
        for col in ['senior_citizen', 'tenure']:
            assert pd.api.types.is_numeric_dtype(loaded_df[col])
        
        # Verify string columns remain string/object
        for col in ['gender', 'churn']:
            assert pd.api.types.is_object_dtype(loaded_df[col]) or pd.api.types.is_string_dtype(loaded_df[col])
    
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(df=valid_dataframe_strategy(min_rows=10, max_rows=100))
    def test_property_non_empty_result(self, df, tmp_path):
        """
        Property: Loading a non-empty CSV always produces a non-empty DataFrame.
        
        For any CSV file with at least one row of data, load_data should return
        a DataFrame with at least one row.
        """
        # Create processor
        processor = DataProcessor()
        
        # Save and load
        csv_path = tmp_path / "non_empty_test.csv"
        df.to_csv(csv_path, index=False)
        loaded_df = processor.load_data(str(csv_path))
        
        # Verify non-empty result
        assert not loaded_df.empty
        assert len(loaded_df) > 0
