"""
Centralized configuration management for the Customer Churn MLOps Pipeline.

This module uses Pydantic BaseSettings to manage configuration from environment
variables with sensible defaults. Configuration can be overridden via .env file
or environment variables.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings can be overridden via environment variables or .env file.
    Default values are provided for local development.
    """
    
    # MLflow Configuration
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_artifact_root: str = "/mlflow/artifacts"
    
    # Model Configuration
    model_name: str = "churn_model"
    model_stage: str = "Production"
    model_path: str = "./models"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    # UI Configuration
    ui_port: int = 8501
    api_url: str = "http://localhost:8000"
    
    # Training Configuration
    n_estimators: int = 100
    max_depth: Optional[int] = 10
    random_state: int = 42
    test_size: float = 0.2
    
    # Data Configuration
    data_path: str = "./data/telco_churn.csv"
    processed_data_path: str = "./data/processed"
    
    # Logging Configuration
    log_level: str = "INFO"
    log_path: str = "./logs"
    
    # Airflow Configuration
    airflow_home: str = "/opt/airflow"
    airflow__core__executor: str = "LocalExecutor"
    airflow__core__load_examples: bool = False
    airflow__database__sql_alchemy_conn: str = "postgresql+psycopg2://airflow:airflow@localhost/airflow"
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "allow"  # Allow extra fields from environment
    }


# Global settings instance
settings = Settings()
