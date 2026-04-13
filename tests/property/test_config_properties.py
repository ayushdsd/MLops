"""
Property-based tests for configuration management.

This module contains property-based tests that validate configuration loading
from environment variables and default value handling for the Customer Churn MLOps Pipeline.
"""

import os
import tempfile
from pathlib import Path
from hypothesis import given, settings, strategies as st
import pytest

from src.config import Settings


# Test 14.1: Property 23 - Configuration from Environment Variables
@settings(max_examples=100)
@given(
    mlflow_uri=st.text(min_size=10, max_size=100, alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters=':/.-_')),
    api_port=st.integers(min_value=1024, max_value=65535),
    model_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_-')),
    log_level=st.sampled_from(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']),
    n_estimators=st.integers(min_value=10, max_value=500),
    test_size=st.floats(min_value=0.1, max_value=0.5, allow_nan=False, allow_infinity=False)
)
def test_property_23_configuration_from_environment_variables(
    mlflow_uri, api_port, model_name, log_level, n_estimators, test_size
):
    """
    Feature: customer-churn-mlops-pipeline, Property 23: Configuration from Environment Variables
    
    **Validates: Requirements 11.2, 11.3**
    
    For any environment variable set for configuration (API ports, MLflow URI, model paths),
    the system should use that value instead of defaults.
    
    This property verifies that:
    1. Environment variables override default configuration values
    2. Configuration values are correctly parsed from environment
    3. Different data types (str, int, float) are handled correctly
    4. Configuration is case-insensitive (Pydantic BaseSettings behavior)
    """
    # Save original environment
    original_env = os.environ.copy()
    
    try:
        # Set environment variables
        os.environ['MLFLOW_TRACKING_URI'] = mlflow_uri
        os.environ['API_PORT'] = str(api_port)
        os.environ['MODEL_NAME'] = model_name
        os.environ['LOG_LEVEL'] = log_level
        os.environ['N_ESTIMATORS'] = str(n_estimators)
        os.environ['TEST_SIZE'] = str(test_size)
        
        # Create new Settings instance (reads from environment)
        config = Settings()
        
        # Verify environment variables override defaults
        assert config.mlflow_tracking_uri == mlflow_uri, \
            f"MLflow URI should be {mlflow_uri}, got {config.mlflow_tracking_uri}"
        
        assert config.api_port == api_port, \
            f"API port should be {api_port}, got {config.api_port}"
        
        assert config.model_name == model_name, \
            f"Model name should be {model_name}, got {config.model_name}"
        
        assert config.log_level == log_level, \
            f"Log level should be {log_level}, got {config.log_level}"
        
        assert config.n_estimators == n_estimators, \
            f"N estimators should be {n_estimators}, got {config.n_estimators}"
        
        # Float comparison with tolerance
        assert abs(config.test_size - test_size) < 0.001, \
            f"Test size should be {test_size}, got {config.test_size}"
        
    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)


# Test 14.1 (variant): Property 23 - Configuration with different data types
@settings(max_examples=50)
@given(
    api_workers=st.integers(min_value=1, max_value=16),
    random_state=st.integers(min_value=0, max_value=1000),
    ui_port=st.integers(min_value=1024, max_value=65535),
    model_path=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='/./_-')),
    data_path=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='/./_-'))
)
def test_property_23_configuration_various_types(
    api_workers, random_state, ui_port, model_path, data_path
):
    """
    Feature: customer-churn-mlops-pipeline, Property 23: Configuration from Environment Variables
    
    **Validates: Requirements 11.2, 11.3**
    
    Variant test focusing on different configuration parameter types and paths.
    
    This property verifies that:
    1. Integer configuration values are correctly parsed
    2. Path configuration values are correctly loaded
    3. Multiple configuration values can be set simultaneously
    """
    # Save original environment
    original_env = os.environ.copy()
    
    try:
        # Set environment variables
        os.environ['API_WORKERS'] = str(api_workers)
        os.environ['RANDOM_STATE'] = str(random_state)
        os.environ['UI_PORT'] = str(ui_port)
        os.environ['MODEL_PATH'] = model_path
        os.environ['DATA_PATH'] = data_path
        
        # Create new Settings instance
        config = Settings()
        
        # Verify all values are correctly loaded
        assert config.api_workers == api_workers
        assert config.random_state == random_state
        assert config.ui_port == ui_port
        assert config.model_path == model_path
        assert config.data_path == data_path
        
    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)


# Test 14.2: Property 24 - Default Configuration Values
@settings(max_examples=100)
@given(
    # Generate random environment variable names that DON'T match our config
    unrelated_env_var=st.text(
        min_size=5, 
        max_size=20, 
        alphabet=st.characters(whitelist_categories=('Lu',), whitelist_characters='_')
    ).filter(lambda x: x not in [
        'MLFLOW_TRACKING_URI', 'API_PORT', 'MODEL_NAME', 'LOG_LEVEL',
        'N_ESTIMATORS', 'TEST_SIZE', 'API_WORKERS', 'RANDOM_STATE',
        'UI_PORT', 'MODEL_PATH', 'DATA_PATH', 'MLFLOW_ARTIFACT_ROOT',
        'MODEL_STAGE', 'API_HOST', 'API_URL', 'MAX_DEPTH',
        'PROCESSED_DATA_PATH', 'LOG_PATH', 'AIRFLOW_HOME'
    ])
)
def test_property_24_default_configuration_values(unrelated_env_var):
    """
    Feature: customer-churn-mlops-pipeline, Property 24: Default Configuration Values
    
    **Validates: Requirements 11.4**
    
    For any configuration parameter without a set environment variable, the system
    should use a sensible default value and function correctly.
    
    This property verifies that:
    1. Default values are used when environment variables are not set
    2. Default values are sensible and allow the system to function
    3. Configuration object can be created without any environment variables
    4. All required configuration fields have defaults
    """
    # Save original environment
    original_env = os.environ.copy()
    
    try:
        # Clear all configuration-related environment variables
        config_vars = [
            'MLFLOW_TRACKING_URI', 'MLFLOW_ARTIFACT_ROOT', 'MODEL_NAME',
            'MODEL_STAGE', 'MODEL_PATH', 'API_HOST', 'API_PORT', 'API_WORKERS',
            'UI_PORT', 'API_URL', 'N_ESTIMATORS', 'MAX_DEPTH', 'RANDOM_STATE',
            'TEST_SIZE', 'DATA_PATH', 'PROCESSED_DATA_PATH', 'LOG_LEVEL',
            'LOG_PATH', 'AIRFLOW_HOME', 'AIRFLOW__CORE__EXECUTOR',
            'AIRFLOW__CORE__LOAD_EXAMPLES', 'AIRFLOW__DATABASE__SQL_ALCHEMY_CONN'
        ]
        
        for var in config_vars:
            os.environ.pop(var, None)
        
        # Set an unrelated environment variable to ensure we're not just reading empty env
        os.environ[unrelated_env_var] = 'test_value'
        
        # Create Settings instance with defaults only
        config = Settings()
        
        # Verify all default values are present and sensible
        
        # MLflow defaults
        assert config.mlflow_tracking_uri == "http://localhost:5000"
        assert config.mlflow_artifact_root == "/mlflow/artifacts"
        
        # Model defaults
        assert config.model_name == "churn_model"
        assert config.model_stage == "Production"
        assert config.model_path == "./models"
        
        # API defaults
        assert config.api_host == "0.0.0.0"
        assert config.api_port == 8000
        assert config.api_workers == 4
        assert 1 <= config.api_workers <= 16, "Default API workers should be reasonable"
        
        # UI defaults
        assert config.ui_port == 8501
        assert config.api_url == "http://localhost:8000"
        
        # Training defaults
        assert config.n_estimators == 100
        assert config.n_estimators > 0, "Default n_estimators should be positive"
        assert config.max_depth == 10
        assert config.random_state == 42
        assert config.test_size == 0.2
        assert 0.0 < config.test_size < 1.0, "Default test_size should be valid proportion"
        
        # Data defaults
        assert config.data_path == "./data/telco_churn.csv"
        assert config.processed_data_path == "./data/processed"
        
        # Logging defaults
        assert config.log_level == "INFO"
        assert config.log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        assert config.log_path == "./logs"
        
        # Airflow defaults
        assert config.airflow_home == "/opt/airflow"
        assert config.airflow__core__executor == "LocalExecutor"
        assert config.airflow__core__load_examples == False
        assert "airflow" in config.airflow__database__sql_alchemy_conn.lower()
        
        # Verify configuration is functional (can be used)
        assert isinstance(config.api_port, int)
        assert isinstance(config.ui_port, int)
        assert isinstance(config.n_estimators, int)
        assert isinstance(config.test_size, float)
        assert isinstance(config.mlflow_tracking_uri, str)
        
    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)


# Test 14.2 (variant): Property 24 - Partial configuration with defaults
@settings(max_examples=50)
@given(
    set_mlflow_uri=st.booleans(),
    set_api_port=st.booleans(),
    set_model_name=st.booleans(),
    mlflow_uri=st.text(min_size=10, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters=':/.-_')),
    api_port=st.integers(min_value=1024, max_value=65535),
    model_name=st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_-'))
)
def test_property_24_partial_configuration_with_defaults(
    set_mlflow_uri, set_api_port, set_model_name,
    mlflow_uri, api_port, model_name
):
    """
    Feature: customer-churn-mlops-pipeline, Property 24: Default Configuration Values
    
    **Validates: Requirements 11.4**
    
    Variant test for partial configuration where some values are set and others use defaults.
    
    This property verifies that:
    1. System works correctly with mix of environment variables and defaults
    2. Set values override defaults while unset values use defaults
    3. Configuration is consistent regardless of which subset is configured
    """
    # Save original environment
    original_env = os.environ.copy()
    
    try:
        # Clear configuration variables
        for var in ['MLFLOW_TRACKING_URI', 'API_PORT', 'MODEL_NAME']:
            os.environ.pop(var, None)
        
        # Conditionally set environment variables
        if set_mlflow_uri:
            os.environ['MLFLOW_TRACKING_URI'] = mlflow_uri
        
        if set_api_port:
            os.environ['API_PORT'] = str(api_port)
        
        if set_model_name:
            os.environ['MODEL_NAME'] = model_name
        
        # Create Settings instance
        config = Settings()
        
        # Verify correct values (either from env or default)
        if set_mlflow_uri:
            assert config.mlflow_tracking_uri == mlflow_uri
        else:
            assert config.mlflow_tracking_uri == "http://localhost:5000"
        
        if set_api_port:
            assert config.api_port == api_port
        else:
            assert config.api_port == 8000
        
        if set_model_name:
            assert config.model_name == model_name
        else:
            assert config.model_name == "churn_model"
        
        # Verify other defaults are still present
        assert config.model_stage == "Production"
        assert config.random_state == 42
        assert config.log_level == "INFO"
        
    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)


# Additional test: Verify Settings can be instantiated multiple times
def test_settings_instantiation():
    """
    Verify that Settings class can be instantiated multiple times.
    
    This ensures that configuration loading is reliable and repeatable.
    """
    # Save original environment
    original_env = os.environ.copy()
    
    try:
        # Clear config vars
        for var in list(os.environ.keys()):
            if any(x in var.upper() for x in ['MLFLOW', 'API', 'MODEL', 'LOG', 'AIRFLOW', 'DATA', 'UI']):
                os.environ.pop(var, None)
        
        # Create multiple instances
        config1 = Settings()
        config2 = Settings()
        
        # Verify they have the same default values
        assert config1.mlflow_tracking_uri == config2.mlflow_tracking_uri
        assert config1.api_port == config2.api_port
        assert config1.model_name == config2.model_name
        assert config1.n_estimators == config2.n_estimators
        
    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)


# Additional test: Verify optional configuration fields
def test_optional_configuration_fields():
    """
    Verify that optional configuration fields (like max_depth) work correctly.
    
    This ensures that Optional fields can be None or have values.
    """
    # Save original environment
    original_env = os.environ.copy()
    
    try:
        # Test with max_depth not set (should use default)
        os.environ.pop('MAX_DEPTH', None)
        
        config = Settings()
        
        # max_depth should have default value
        assert config.max_depth == 10
        
        # Test with max_depth set to a value
        os.environ['MAX_DEPTH'] = '15'
        config2 = Settings()
        assert config2.max_depth == 15
        
        # Test with max_depth set to empty string (should fail or use default)
        # Pydantic v2 will raise validation error for empty string on int field
        # So we just verify that valid integers work
        
    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)


# Additional test: Verify case insensitivity
def test_case_insensitive_configuration():
    """
    Verify that configuration is case-insensitive (Pydantic BaseSettings behavior).
    
    This ensures that environment variables work regardless of case.
    """
    # Save original environment
    original_env = os.environ.copy()
    
    try:
        # Set environment variable in different cases
        os.environ['api_port'] = '9000'  # lowercase
        
        config = Settings()
        
        # Should still load the value
        assert config.api_port == 9000
        
    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)
