"""
Input validation utilities for the Prediction API.

This module provides validation functions for customer input data,
ensuring data quality and providing descriptive error messages.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ValidationError:
    """Represents a validation error for a specific field."""
    field: str
    message: str
    value: Any


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    errors: List[ValidationError]
    
    def add_error(self, field: str, message: str, value: Any = None) -> None:
        """Add a validation error."""
        self.errors.append(ValidationError(field=field, message=message, value=value))
        self.is_valid = False
    
    def get_error_messages(self) -> Dict[str, str]:
        """Get error messages as a dictionary."""
        return {error.field: error.message for error in self.errors}
    
    def get_error_summary(self) -> str:
        """Get a summary of all validation errors."""
        if self.is_valid:
            return "Validation passed"
        
        error_lines = [f"- {error.field}: {error.message}" for error in self.errors]
        return f"Validation failed with {len(self.errors)} error(s):\n" + "\n".join(error_lines)


# Required fields for customer input
REQUIRED_FIELDS = [
    'gender', 'senior_citizen', 'partner', 'dependents', 'tenure',
    'contract', 'paperless_billing', 'payment_method', 'monthly_charges',
    'total_charges', 'phone_service', 'multiple_lines', 'internet_service',
    'online_security', 'online_backup', 'device_protection', 'tech_support',
    'streaming_tv', 'streaming_movies'
]

# Valid categorical values
VALID_CATEGORICAL_VALUES = {
    'gender': ['Male', 'Female'],
    'senior_citizen': [0, 1],
    'partner': ['Yes', 'No'],
    'dependents': ['Yes', 'No'],
    'contract': ['Month-to-month', 'One year', 'Two year'],
    'paperless_billing': ['Yes', 'No'],
    'payment_method': [
        'Electronic check',
        'Mailed check',
        'Bank transfer (automatic)',
        'Credit card (automatic)'
    ],
    'phone_service': ['Yes', 'No'],
    'multiple_lines': ['Yes', 'No', 'No phone service'],
    'internet_service': ['DSL', 'Fiber optic', 'No'],
    'online_security': ['Yes', 'No', 'No internet service'],
    'online_backup': ['Yes', 'No', 'No internet service'],
    'device_protection': ['Yes', 'No', 'No internet service'],
    'tech_support': ['Yes', 'No', 'No internet service'],
    'streaming_tv': ['Yes', 'No', 'No internet service'],
    'streaming_movies': ['Yes', 'No', 'No internet service']
}

# Numerical field constraints
NUMERICAL_FIELD_CONSTRAINTS = {
    'tenure': {'type': int, 'min': 0, 'max': None},
    'monthly_charges': {'type': float, 'min': 0.0, 'max': None, 'exclusive_min': True},
    'total_charges': {'type': float, 'min': 0.0, 'max': None}
}


def validate_required_fields(data: Dict[str, Any]) -> ValidationResult:
    """
    Validate that all required fields are present in the input data.
    
    Args:
        data: Dictionary containing customer input data
        
    Returns:
        ValidationResult indicating whether all required fields are present
        
    Validates: Requirements 15.1
    """
    result = ValidationResult(is_valid=True, errors=[])
    
    for field in REQUIRED_FIELDS:
        if field not in data:
            result.add_error(
                field=field,
                message=f"Required field '{field}' is missing",
                value=None
            )
        elif data[field] is None:
            result.add_error(
                field=field,
                message=f"Required field '{field}' cannot be null",
                value=None
            )
    
    return result


def validate_numerical_fields(data: Dict[str, Any]) -> ValidationResult:
    """
    Validate numerical field types and ranges.
    
    Args:
        data: Dictionary containing customer input data
        
    Returns:
        ValidationResult indicating whether numerical fields are valid
        
    Validates: Requirements 15.2
    """
    result = ValidationResult(is_valid=True, errors=[])
    
    for field, constraints in NUMERICAL_FIELD_CONSTRAINTS.items():
        if field not in data:
            continue
            
        value = data[field]
        expected_type = constraints['type']
        min_value = constraints.get('min')
        max_value = constraints.get('max')
        exclusive_min = constraints.get('exclusive_min', False)
        
        # Type validation
        if not isinstance(value, (int, float)):
            result.add_error(
                field=field,
                message=f"Field '{field}' must be a number, got {type(value).__name__}",
                value=value
            )
            continue
        
        # Range validation
        if min_value is not None:
            if exclusive_min:
                if value <= min_value:
                    result.add_error(
                        field=field,
                        message=f"Field '{field}' must be greater than {min_value}, got {value}",
                        value=value
                    )
            else:
                if value < min_value:
                    result.add_error(
                        field=field,
                        message=f"Field '{field}' must be greater than or equal to {min_value}, got {value}",
                        value=value
                    )
        
        if max_value is not None and value > max_value:
            result.add_error(
                field=field,
                message=f"Field '{field}' must be less than or equal to {max_value}, got {value}",
                value=value
            )
    
    return result


def validate_categorical_fields(data: Dict[str, Any]) -> ValidationResult:
    """
    Validate that categorical fields contain expected values.
    
    Args:
        data: Dictionary containing customer input data
        
    Returns:
        ValidationResult indicating whether categorical fields are valid
        
    Validates: Requirements 15.3
    """
    result = ValidationResult(is_valid=True, errors=[])
    
    for field, valid_values in VALID_CATEGORICAL_VALUES.items():
        if field not in data:
            continue
            
        value = data[field]
        
        if value not in valid_values:
            valid_values_str = ', '.join([f"'{v}'" for v in valid_values])
            result.add_error(
                field=field,
                message=f"Field '{field}' must be one of [{valid_values_str}], got '{value}'",
                value=value
            )
    
    return result


def validate_customer_input(data: Dict[str, Any]) -> ValidationResult:
    """
    Perform comprehensive validation of customer input data.
    
    This function validates:
    - All required fields are present
    - Numerical fields have correct types and ranges
    - Categorical fields contain expected values
    
    Args:
        data: Dictionary containing customer input data
        
    Returns:
        ValidationResult with all validation errors
        
    Validates: Requirements 15.1, 15.2, 15.3, 15.4
    """
    # Combine all validation results
    combined_result = ValidationResult(is_valid=True, errors=[])
    
    # Validate required fields
    required_result = validate_required_fields(data)
    if not required_result.is_valid:
        combined_result.errors.extend(required_result.errors)
        combined_result.is_valid = False
    
    # Validate numerical fields (only if present)
    numerical_result = validate_numerical_fields(data)
    if not numerical_result.is_valid:
        combined_result.errors.extend(numerical_result.errors)
        combined_result.is_valid = False
    
    # Validate categorical fields (only if present)
    categorical_result = validate_categorical_fields(data)
    if not categorical_result.is_valid:
        combined_result.errors.extend(categorical_result.errors)
        combined_result.is_valid = False
    
    return combined_result


def get_field_description(field: str) -> str:
    """
    Get a human-readable description of a field.
    
    Args:
        field: Field name
        
    Returns:
        Description of the field
    """
    descriptions = {
        'gender': 'Customer gender (Male or Female)',
        'senior_citizen': 'Whether customer is a senior citizen (0 or 1)',
        'partner': 'Whether customer has a partner (Yes or No)',
        'dependents': 'Whether customer has dependents (Yes or No)',
        'tenure': 'Number of months with the company (0 or greater)',
        'contract': 'Contract type (Month-to-month, One year, or Two year)',
        'paperless_billing': 'Whether customer uses paperless billing (Yes or No)',
        'payment_method': 'Payment method (Electronic check, Mailed check, Bank transfer, or Credit card)',
        'monthly_charges': 'Monthly bill amount (greater than 0)',
        'total_charges': 'Total amount charged to customer (0 or greater)',
        'phone_service': 'Whether customer has phone service (Yes or No)',
        'multiple_lines': 'Whether customer has multiple lines (Yes, No, or No phone service)',
        'internet_service': 'Internet service type (DSL, Fiber optic, or No)',
        'online_security': 'Whether customer has online security (Yes, No, or No internet service)',
        'online_backup': 'Whether customer has online backup (Yes, No, or No internet service)',
        'device_protection': 'Whether customer has device protection (Yes, No, or No internet service)',
        'tech_support': 'Whether customer has tech support (Yes, No, or No internet service)',
        'streaming_tv': 'Whether customer has streaming TV (Yes, No, or No internet service)',
        'streaming_movies': 'Whether customer has streaming movies (Yes, No, or No internet service)'
    }
    
    return descriptions.get(field, f'Field: {field}')
