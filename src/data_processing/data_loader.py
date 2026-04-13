"""
Data loading module for the Customer Churn MLOps Pipeline.

This module provides functionality to load customer churn data from CSV files
with comprehensive error handling and logging.
"""

import os
import pandas as pd
import numpy as np
import joblib
from typing import Optional, List, Any, Tuple
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Import centralized logging configuration
from ..logging_config import get_logger

# Configure logger
logger = get_logger('data_processor')


class DataLoadError(Exception):
    """Raised when data cannot be loaded."""
    pass


class SchemaValidationError(Exception):
    """Raised when data schema is invalid."""
    pass


class PreprocessingError(Exception):
    """Raised when preprocessing fails."""
    pass


@dataclass
class PreprocessedData:
    """
    Container for preprocessed training and testing data.
    
    Attributes:
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
        feature_names: List of feature names after preprocessing
        scaler: Fitted StandardScaler for numerical features
        label_encoders: Dictionary of fitted LabelEncoders for categorical features
    """
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    scaler: StandardScaler
    label_encoders: dict


@dataclass
class ValidationResult:
    """
    Result of data validation containing validation status and messages.
    
    Attributes:
        is_valid: True if all validations passed, False otherwise
        errors: List of error messages for critical validation failures
        warnings: List of warning messages for non-critical issues
    """
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class DataProcessor:
    """
    Data processor for loading and preprocessing customer churn data.
    
    This class handles data ingestion from CSV files with robust error handling
    for common failure scenarios including missing files, empty files, and
    parsing errors.
    """
    
    def __init__(self):
        """Initialize the DataProcessor."""
        logger.info("DataProcessor initialized")
        
        # Define expected schema
        self.required_columns = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
        ]
        
        # Define expected categorical values
        self.categorical_values = {
            'gender': ['Male', 'Female'],
            'Partner': ['Yes', 'No'],
            'Dependents': ['Yes', 'No'],
            'PhoneService': ['Yes', 'No'],
            'MultipleLines': ['Yes', 'No', 'No phone service'],
            'InternetService': ['DSL', 'Fiber optic', 'No'],
            'OnlineSecurity': ['Yes', 'No', 'No internet service'],
            'OnlineBackup': ['Yes', 'No', 'No internet service'],
            'DeviceProtection': ['Yes', 'No', 'No internet service'],
            'TechSupport': ['Yes', 'No', 'No internet service'],
            'StreamingTV': ['Yes', 'No', 'No internet service'],
            'StreamingMovies': ['Yes', 'No', 'No internet service'],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'PaperlessBilling': ['Yes', 'No'],
            'PaymentMethod': [
                'Electronic check', 'Mailed check',
                'Bank transfer (automatic)', 'Credit card (automatic)'
            ]
        }
        
        # Define numerical columns
        self.numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
        self.integer_columns = ['SeniorCitizen', 'tenure']
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load customer churn data from a CSV file.
        
        This method reads a CSV file and returns a pandas DataFrame. It includes
        comprehensive error handling for missing files, empty files, and CSV
        parsing errors.
        
        Args:
            file_path: Path to the CSV file to load
            
        Returns:
            pd.DataFrame: Loaded customer data
            
        Raises:
            DataLoadError: If the file doesn't exist, is empty, or cannot be parsed
            
        Examples:
            >>> processor = DataProcessor()
            >>> df = processor.load_data("data/telco_churn.csv")
            >>> print(f"Loaded {len(df)} records")
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                error_msg = f"Dataset file not found: {file_path}"
                logger.error(error_msg)
                raise DataLoadError(error_msg)
            
            # Check if file is empty
            if os.path.getsize(file_path) == 0:
                error_msg = f"Dataset file is empty: {file_path}"
                logger.error(error_msg)
                raise DataLoadError(error_msg)
            
            # Load CSV file
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            
            # Verify data was loaded
            if df.empty:
                error_msg = f"Dataset file contains no data: {file_path}"
                logger.warning(error_msg)
                raise DataLoadError(error_msg)
            
            # Log success
            logger.info(f"Successfully loaded {len(df)} records with {len(df.columns)} columns from {file_path}")
            
            return df
            
        except pd.errors.EmptyDataError as e:
            error_msg = f"Dataset file is empty or has no data: {file_path}"
            logger.error(error_msg, exc_info=True)
            raise DataLoadError(error_msg) from e
            
        except pd.errors.ParserError as e:
            error_msg = f"Failed to parse CSV file: {file_path}. Error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise DataLoadError(error_msg) from e
            
        except PermissionError as e:
            error_msg = f"Permission denied when accessing file: {file_path}"
            logger.error(error_msg, exc_info=True)
            raise DataLoadError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Unexpected error loading data from {file_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise DataLoadError(error_msg) from e

    def validate_schema(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate the schema and data quality of a customer churn dataset.
        
        This method performs comprehensive validation including:
        - Checking for required columns
        - Validating data types
        - Validating numerical ranges (tenure ≥ 0, monthly_charges > 0, total_charges ≥ 0)
        - Validating categorical values against expected sets
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult: Object containing validation status, errors, and warnings
            
        Examples:
            >>> processor = DataProcessor()
            >>> df = processor.load_data("data/telco_churn.csv")
            >>> result = processor.validate_schema(df)
            >>> if not result.is_valid:
            ...     print(f"Validation errors: {result.errors}")
        """
        errors = []
        warnings = []
        
        # Check for required columns
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")
            logger.error(f"Schema validation failed: missing columns {missing_columns}")
        
        # If critical columns are missing, return early
        if errors:
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        # Validate data types for numerical columns
        for col in self.numerical_columns:
            if col in df.columns:
                # Check if column can be converted to numeric
                try:
                    # Handle TotalCharges which might have spaces or empty strings
                    if col == 'TotalCharges':
                        # Convert to numeric, coercing errors to NaN
                        numeric_values = pd.to_numeric(df[col], errors='coerce')
                        non_numeric_count = numeric_values.isna().sum() - df[col].isna().sum()
                        if non_numeric_count > 0:
                            warnings.append(
                                f"Column '{col}' contains {non_numeric_count} non-numeric values that will need cleaning"
                            )
                            logger.warning(f"Column '{col}' has {non_numeric_count} non-numeric values")
                    else:
                        # For other numerical columns, check if they're numeric
                        if not pd.api.types.is_numeric_dtype(df[col]):
                            numeric_values = pd.to_numeric(df[col], errors='coerce')
                            non_numeric_count = numeric_values.isna().sum() - df[col].isna().sum()
                            if non_numeric_count > 0:
                                errors.append(
                                    f"Column '{col}' should be numeric but contains {non_numeric_count} non-numeric values"
                                )
                                logger.error(f"Column '{col}' has invalid data type")
                except Exception as e:
                    errors.append(f"Error validating data type for column '{col}': {str(e)}")
                    logger.error(f"Data type validation error for '{col}': {str(e)}")
        
        # Validate SeniorCitizen is integer (0 or 1)
        if 'SeniorCitizen' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['SeniorCitizen']):
                errors.append("Column 'SeniorCitizen' should be numeric (0 or 1)")
            else:
                invalid_values = df[~df['SeniorCitizen'].isin([0, 1])]['SeniorCitizen'].unique()
                if len(invalid_values) > 0:
                    errors.append(
                        f"Column 'SeniorCitizen' contains invalid values: {invalid_values.tolist()}. Expected 0 or 1."
                    )
                    logger.error(f"SeniorCitizen has invalid values: {invalid_values.tolist()}")
        
        # Validate numerical ranges
        if 'tenure' in df.columns:
            try:
                numeric_tenure = pd.to_numeric(df['tenure'], errors='coerce')
                invalid_tenure = numeric_tenure[numeric_tenure < 0]
                if len(invalid_tenure) > 0:
                    errors.append(
                        f"Column 'tenure' contains {len(invalid_tenure)} values < 0. Tenure must be ≥ 0."
                    )
                    logger.error(f"Found {len(invalid_tenure)} invalid tenure values < 0")
            except Exception as e:
                errors.append(f"Error validating tenure range: {str(e)}")
        
        if 'MonthlyCharges' in df.columns:
            try:
                numeric_charges = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
                invalid_charges = numeric_charges[numeric_charges <= 0]
                if len(invalid_charges) > 0:
                    errors.append(
                        f"Column 'MonthlyCharges' contains {len(invalid_charges)} values ≤ 0. Monthly charges must be > 0."
                    )
                    logger.error(f"Found {len(invalid_charges)} invalid MonthlyCharges values ≤ 0")
            except Exception as e:
                errors.append(f"Error validating MonthlyCharges range: {str(e)}")
        
        if 'TotalCharges' in df.columns:
            try:
                numeric_total = pd.to_numeric(df['TotalCharges'], errors='coerce')
                # Filter out NaN values before checking range
                valid_total = numeric_total.dropna()
                invalid_total = valid_total[valid_total < 0]
                if len(invalid_total) > 0:
                    errors.append(
                        f"Column 'TotalCharges' contains {len(invalid_total)} values < 0. Total charges must be ≥ 0."
                    )
                    logger.error(f"Found {len(invalid_total)} invalid TotalCharges values < 0")
            except Exception as e:
                errors.append(f"Error validating TotalCharges range: {str(e)}")
        
        # Validate categorical values
        for col, expected_values in self.categorical_values.items():
            if col in df.columns:
                # Get unique values in the column (excluding NaN)
                unique_values = df[col].dropna().unique()
                invalid_values = [val for val in unique_values if val not in expected_values]
                
                if invalid_values:
                    errors.append(
                        f"Column '{col}' contains invalid values: {invalid_values}. "
                        f"Expected values: {expected_values}"
                    )
                    logger.error(f"Column '{col}' has invalid categorical values: {invalid_values}")
        
        # Check for high percentage of missing values (warning only)
        for col in df.columns:
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            if missing_pct > 50:
                warnings.append(
                    f"Column '{col}' has {missing_pct:.1f}% missing values"
                )
                logger.warning(f"High missing value percentage in '{col}': {missing_pct:.1f}%")
        
        # Determine overall validation status
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info(f"Schema validation passed with {len(warnings)} warnings")
        else:
            logger.error(f"Schema validation failed with {len(errors)} errors and {len(warnings)} warnings")
        
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)

    def preprocess(self, df: pd.DataFrame, target_column: str = 'Churn', test_size: float = 0.2, random_state: int = 42) -> PreprocessedData:
        """
        Preprocess customer churn data with imputation, encoding, and scaling.
        
        This method performs the complete preprocessing pipeline:
        1. Handles null values via imputation (median for numerical, mode for categorical)
        2. Encodes categorical variables using label encoding
        3. Standardizes numerical features using StandardScaler
        4. Splits data into train/test sets (80-20 by default)
        
        Args:
            df: DataFrame to preprocess
            target_column: Name of the target column (default: 'Churn')
            test_size: Proportion of data for testing (default: 0.2)
            random_state: Random seed for reproducibility (default: 42)
            
        Returns:
            PreprocessedData: Object containing train/test splits and fitted transformers
            
        Raises:
            PreprocessingError: If preprocessing fails
            
        Examples:
            >>> processor = DataProcessor()
            >>> df = processor.load_data("data/telco_churn.csv")
            >>> preprocessed = processor.preprocess(df)
            >>> print(f"Training samples: {len(preprocessed.X_train)}")
        """
        try:
            logger.info("Starting data preprocessing")
            
            # Make a copy to avoid modifying original data
            df_processed = df.copy()
            
            # Separate features and target
            if target_column in df_processed.columns:
                y = df_processed[target_column]
                X = df_processed.drop(columns=[target_column])
                logger.info(f"Separated target column '{target_column}' from features")
            else:
                # If no target column, use all columns as features
                X = df_processed
                y = None
                logger.warning(f"Target column '{target_column}' not found, processing features only")
            
            # Identify numerical and categorical columns
            numerical_cols = []
            categorical_cols = []
            
            for col in X.columns:
                if col in self.numerical_columns:
                    numerical_cols.append(col)
                else:
                    categorical_cols.append(col)
            
            logger.info(f"Identified {len(numerical_cols)} numerical and {len(categorical_cols)} categorical columns")
            
            # Step 1: Handle null values via imputation
            logger.info("Step 1: Imputing null values")
            
            # Impute numerical columns with median
            for col in numerical_cols:
                # Handle TotalCharges which might be stored as string with spaces
                if col == 'TotalCharges' and X[col].dtype == 'object':
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                
                if X[col].isna().sum() > 0:
                    median_value = X[col].median()
                    null_count = X[col].isna().sum()
                    X.loc[:, col] = X[col].fillna(median_value)
                    logger.info(f"Imputed {null_count} null values in '{col}' with median: {median_value}")
            
            # Impute categorical columns with mode
            for col in categorical_cols:
                if X[col].isna().sum() > 0:
                    mode_value = X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown'
                    null_count = X[col].isna().sum()
                    X.loc[:, col] = X[col].fillna(mode_value)
                    logger.info(f"Imputed {null_count} null values in '{col}' with mode: {mode_value}")
            
            # Verify no null values remain
            remaining_nulls = X.isna().sum().sum()
            if remaining_nulls > 0:
                logger.warning(f"Warning: {remaining_nulls} null values remain after imputation")
            else:
                logger.info("All null values successfully imputed")
            
            # Step 2: Encode categorical variables
            logger.info("Step 2: Encoding categorical variables")
            label_encoders = {}
            
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
                logger.info(f"Encoded '{col}' with {len(le.classes_)} unique values")
            
            # Step 3: Standardize numerical features
            logger.info("Step 3: Standardizing numerical features")
            scaler = StandardScaler()
            
            if len(numerical_cols) > 0:
                X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
                logger.info(f"Standardized {len(numerical_cols)} numerical features")
            else:
                logger.warning("No numerical columns to standardize")
            
            # Get feature names
            feature_names = list(X.columns)
            
            # Convert to numpy arrays
            X_array = X.values
            
            # Step 4: Train-test split
            if y is not None:
                # Encode target variable if it's categorical
                if y.dtype == 'object':
                    y_encoder = LabelEncoder()
                    y_encoded = y_encoder.fit_transform(y)
                    logger.info(f"Encoded target variable with classes: {y_encoder.classes_}")
                else:
                    y_encoded = y.values
                
                logger.info(f"Step 4: Splitting data with test_size={test_size}, random_state={random_state}")
                
                # Check if dataset is large enough for stratified split
                # Need at least 2 samples per class in test set
                min_class_count = np.min(np.bincount(y_encoded))
                n_test_samples = int(len(X_array) * test_size)
                
                if n_test_samples < len(np.unique(y_encoded)) or min_class_count < 2:
                    # Dataset too small for stratified split, use regular split
                    logger.warning(f"Dataset too small for stratified split (min_class_count={min_class_count}, n_test={n_test_samples}). Using non-stratified split.")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_array, y_encoded, test_size=test_size, random_state=random_state
                    )
                else:
                    # Use stratified split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_array, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
                    )
                
                logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
                logger.info(f"Train/Test ratio: {len(X_train)/len(X_array):.2%}/{len(X_test)/len(X_array):.2%}")
            else:
                # No target variable, return all data as training set
                X_train = X_array
                X_test = np.array([])
                y_train = np.array([])
                y_test = np.array([])
                logger.info("No target variable provided, returning all data as training set")
            
            logger.info("Preprocessing completed successfully")
            
            return PreprocessedData(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                feature_names=feature_names,
                scaler=scaler,
                label_encoders=label_encoders
            )
            
        except Exception as e:
            error_msg = f"Preprocessing failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise PreprocessingError(error_msg) from e

    def save_pipeline(self, pipeline: Pipeline, path: str) -> None:
        """
        Save a preprocessing pipeline to disk.
        
        This method serializes an sklearn Pipeline object to a file using joblib,
        which is optimized for storing large numpy arrays and scikit-learn models.
        The pipeline can later be loaded for inference to ensure consistent
        preprocessing transformations.
        
        Args:
            pipeline: sklearn Pipeline object to serialize
            path: File path where the pipeline should be saved
            
        Raises:
            PreprocessingError: If pipeline cannot be saved
            
        Examples:
            >>> processor = DataProcessor()
            >>> pipeline = Pipeline([('scaler', StandardScaler())])
            >>> processor.save_pipeline(pipeline, "models/pipelines/preprocessing.pkl")
        """
        try:
            # Ensure the directory exists
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")
            
            # Save the pipeline using joblib
            joblib.dump(pipeline, path)
            logger.info(f"Successfully saved pipeline to {path}")
            
        except Exception as e:
            error_msg = f"Failed to save pipeline to {path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise PreprocessingError(error_msg) from e

    def load_pipeline(self, path: str) -> Pipeline:
        """
        Load a preprocessing pipeline from disk.
        
        This method deserializes an sklearn Pipeline object from a file that was
        previously saved using save_pipeline(). The loaded pipeline can be used
        to apply the same preprocessing transformations to new data.
        
        Args:
            path: File path from which to load the pipeline
            
        Returns:
            Pipeline: Deserialized sklearn Pipeline object
            
        Raises:
            PreprocessingError: If pipeline file doesn't exist or cannot be loaded
            
        Examples:
            >>> processor = DataProcessor()
            >>> pipeline = processor.load_pipeline("models/pipelines/preprocessing.pkl")
            >>> transformed_data = pipeline.transform(new_data)
        """
        try:
            # Check if file exists
            if not os.path.exists(path):
                error_msg = f"Pipeline file not found: {path}"
                logger.error(error_msg)
                raise PreprocessingError(error_msg)
            
            # Load the pipeline using joblib
            pipeline = joblib.load(path)
            logger.info(f"Successfully loaded pipeline from {path}")
            
            return pipeline
            
        except PreprocessingError:
            # Re-raise PreprocessingError as-is
            raise
        except Exception as e:
            error_msg = f"Failed to load pipeline from {path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise PreprocessingError(error_msg) from e
