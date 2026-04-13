"""
Unit tests for API input validators.

Tests validation logic for required fields, numerical fields, and categorical fields.
"""

import pytest
from src.api.validators import (
    validate_required_fields,
    validate_numerical_fields,
    validate_categorical_fields,
    validate_customer_input,
    get_field_description,
    ValidationResult,
    ValidationError,
    REQUIRED_FIELDS,
    VALID_CATEGORICAL_VALUES,
    NUMERICAL_FIELD_CONSTRAINTS
)


class TestValidationResult:
    """Test cases for ValidationResult class."""
    
    def test_validation_result_initialization(self):
        """Test creating a ValidationResult."""
        result = ValidationResult(is_valid=True, errors=[])
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_add_error(self):
        """Test adding an error to ValidationResult."""
        result = ValidationResult(is_valid=True, errors=[])
        result.add_error("field1", "Error message", "invalid_value")
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field == "field1"
        assert result.errors[0].message == "Error message"
        assert result.errors[0].value == "invalid_value"
    
    def test_get_error_messages(self):
        """Test getting error messages as dictionary."""
        result = ValidationResult(is_valid=True, errors=[])
        result.add_error("field1", "Error 1")
        result.add_error("field2", "Error 2")
        
        messages = result.get_error_messages()
        assert messages == {"field1": "Error 1", "field2": "Error 2"}
    
    def test_get_error_summary_valid(self):
        """Test error summary for valid result."""
        result = ValidationResult(is_valid=True, errors=[])
        summary = result.get_error_summary()
        assert summary == "Validation passed"
    
    def test_get_error_summary_invalid(self):
        """Test error summary for invalid result."""
        result = ValidationResult(is_valid=True, errors=[])
        result.add_error("field1", "Error 1")
        result.add_error("field2", "Error 2")
        
        summary = result.get_error_summary()
        assert "Validation failed with 2 error(s)" in summary
        assert "field1: Error 1" in summary
        assert "field2: Error 2" in summary


class TestValidateRequiredFields:
    """Test cases for validate_required_fields function."""
    
    def test_all_required_fields_present(self):
        """Test validation passes when all required fields are present."""
        data = {field: "value" for field in REQUIRED_FIELDS}
        result = validate_required_fields(data)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_missing_required_field(self):
        """Test validation fails when a required field is missing."""
        data = {field: "value" for field in REQUIRED_FIELDS}
        del data['gender']
        
        result = validate_required_fields(data)
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field == 'gender'
        assert "missing" in result.errors[0].message.lower()
    
    def test_multiple_missing_fields(self):
        """Test validation fails when multiple required fields are missing."""
        data = {'gender': 'Male', 'tenure': 12}
        
        result = validate_required_fields(data)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        missing_fields = [error.field for error in result.errors]
        assert 'senior_citizen' in missing_fields
        assert 'partner' in missing_fields
    
    def test_null_required_field(self):
        """Test validation fails when a required field is null."""
        data = {field: "value" for field in REQUIRED_FIELDS}
        data['gender'] = None
        
        result = validate_required_fields(data)
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field == 'gender'
        assert "cannot be null" in result.errors[0].message.lower()


class TestValidateNumericalFields:
    """Test cases for validate_numerical_fields function."""
    
    def test_valid_numerical_fields(self):
        """Test validation passes for valid numerical values."""
        data = {
            'tenure': 12,
            'monthly_charges': 50.0,
            'total_charges': 600.0
        }
        
        result = validate_numerical_fields(data)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_tenure_negative(self):
        """Test validation fails for negative tenure."""
        data = {'tenure': -5}
        
        result = validate_numerical_fields(data)
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field == 'tenure'
        assert "greater than or equal to 0" in result.errors[0].message
    
    def test_monthly_charges_zero(self):
        """Test validation fails for zero monthly_charges (exclusive minimum)."""
        data = {'monthly_charges': 0.0}
        
        result = validate_numerical_fields(data)
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field == 'monthly_charges'
        assert "greater than 0" in result.errors[0].message
    
    def test_monthly_charges_negative(self):
        """Test validation fails for negative monthly_charges."""
        data = {'monthly_charges': -10.0}
        
        result = validate_numerical_fields(data)
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field == 'monthly_charges'
    
    def test_total_charges_zero_valid(self):
        """Test validation passes for zero total_charges (inclusive minimum)."""
        data = {'total_charges': 0.0}
        
        result = validate_numerical_fields(data)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_total_charges_negative(self):
        """Test validation fails for negative total_charges."""
        data = {'total_charges': -100.0}
        
        result = validate_numerical_fields(data)
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field == 'total_charges'
    
    def test_invalid_type_string(self):
        """Test validation fails for string instead of number."""
        data = {'tenure': "twelve"}
        
        result = validate_numerical_fields(data)
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field == 'tenure'
        assert "must be a number" in result.errors[0].message
    
    def test_invalid_type_none(self):
        """Test validation fails for None value."""
        data = {'tenure': None}
        
        result = validate_numerical_fields(data)
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field == 'tenure'
    
    def test_missing_numerical_field_skipped(self):
        """Test validation skips missing numerical fields."""
        data = {}
        
        result = validate_numerical_fields(data)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_multiple_numerical_errors(self):
        """Test validation catches multiple numerical errors."""
        data = {
            'tenure': -5,
            'monthly_charges': 0.0,
            'total_charges': -100.0
        }
        
        result = validate_numerical_fields(data)
        
        assert result.is_valid is False
        assert len(result.errors) == 3


class TestValidateCategoricalFields:
    """Test cases for validate_categorical_fields function."""
    
    def test_valid_categorical_fields(self):
        """Test validation passes for valid categorical values."""
        data = {
            'gender': 'Male',
            'senior_citizen': 0,
            'partner': 'Yes',
            'contract': 'Month-to-month',
            'internet_service': 'DSL'
        }
        
        result = validate_categorical_fields(data)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_invalid_gender(self):
        """Test validation fails for invalid gender."""
        data = {'gender': 'Other'}
        
        result = validate_categorical_fields(data)
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field == 'gender'
        assert "'Male'" in result.errors[0].message
        assert "'Female'" in result.errors[0].message
    
    def test_invalid_senior_citizen(self):
        """Test validation fails for invalid senior_citizen."""
        data = {'senior_citizen': 2}
        
        result = validate_categorical_fields(data)
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field == 'senior_citizen'
    
    def test_invalid_contract(self):
        """Test validation fails for invalid contract."""
        data = {'contract': 'Three year'}
        
        result = validate_categorical_fields(data)
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field == 'contract'
        assert "Month-to-month" in result.errors[0].message
    
    def test_invalid_internet_service(self):
        """Test validation fails for invalid internet_service."""
        data = {'internet_service': 'Cable'}
        
        result = validate_categorical_fields(data)
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field == 'internet_service'
    
    def test_missing_categorical_field_skipped(self):
        """Test validation skips missing categorical fields."""
        data = {}
        
        result = validate_categorical_fields(data)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_multiple_categorical_errors(self):
        """Test validation catches multiple categorical errors."""
        data = {
            'gender': 'Other',
            'senior_citizen': 3,
            'contract': 'Invalid',
            'internet_service': 'Cable'
        }
        
        result = validate_categorical_fields(data)
        
        assert result.is_valid is False
        assert len(result.errors) == 4
    
    def test_all_valid_categorical_values(self):
        """Test all valid values for each categorical field."""
        for field, valid_values in VALID_CATEGORICAL_VALUES.items():
            for value in valid_values:
                data = {field: value}
                result = validate_categorical_fields(data)
                assert result.is_valid is True, f"Failed for {field}={value}"


class TestValidateCustomerInput:
    """Test cases for validate_customer_input function."""
    
    def test_fully_valid_input(self):
        """Test validation passes for fully valid customer input."""
        data = {
            'gender': 'Male',
            'senior_citizen': 0,
            'partner': 'Yes',
            'dependents': 'No',
            'tenure': 12,
            'contract': 'Month-to-month',
            'paperless_billing': 'Yes',
            'payment_method': 'Electronic check',
            'monthly_charges': 50.0,
            'total_charges': 600.0,
            'phone_service': 'Yes',
            'multiple_lines': 'No',
            'internet_service': 'DSL',
            'online_security': 'Yes',
            'online_backup': 'No',
            'device_protection': 'Yes',
            'tech_support': 'No',
            'streaming_tv': 'Yes',
            'streaming_movies': 'No'
        }
        
        result = validate_customer_input(data)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_missing_field_error(self):
        """Test validation fails for missing required field."""
        data = {
            'gender': 'Male',
            'senior_citizen': 0,
            # Missing other required fields
        }
        
        result = validate_customer_input(data)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
    
    def test_invalid_numerical_field_error(self):
        """Test validation fails for invalid numerical field."""
        data = {field: "value" for field in REQUIRED_FIELDS}
        data['tenure'] = -5
        data['monthly_charges'] = 50.0
        data['total_charges'] = 600.0
        
        result = validate_customer_input(data)
        
        assert result.is_valid is False
        assert any(error.field == 'tenure' for error in result.errors)
    
    def test_invalid_categorical_field_error(self):
        """Test validation fails for invalid categorical field."""
        data = {field: "value" for field in REQUIRED_FIELDS}
        data['gender'] = 'Other'
        data['tenure'] = 12
        data['monthly_charges'] = 50.0
        data['total_charges'] = 600.0
        
        result = validate_customer_input(data)
        
        assert result.is_valid is False
        assert any(error.field == 'gender' for error in result.errors)
    
    def test_multiple_validation_errors(self):
        """Test validation catches multiple types of errors."""
        data = {
            'gender': 'Other',  # Invalid categorical
            'senior_citizen': 0,
            'tenure': -5,  # Invalid numerical
            'monthly_charges': 0.0,  # Invalid numerical
            # Missing other required fields
        }
        
        result = validate_customer_input(data)
        
        assert result.is_valid is False
        assert len(result.errors) > 3  # At least 3 errors (gender, tenure, monthly_charges)
    
    def test_empty_input(self):
        """Test validation fails for empty input."""
        data = {}
        
        result = validate_customer_input(data)
        
        assert result.is_valid is False
        assert len(result.errors) == len(REQUIRED_FIELDS)


class TestGetFieldDescription:
    """Test cases for get_field_description function."""
    
    def test_known_field_descriptions(self):
        """Test getting descriptions for known fields."""
        assert "gender" in get_field_description('gender').lower()
        assert "months" in get_field_description('tenure').lower()
        assert "monthly" in get_field_description('monthly_charges').lower()
    
    def test_unknown_field_description(self):
        """Test getting description for unknown field."""
        description = get_field_description('unknown_field')
        assert 'unknown_field' in description
    
    def test_all_required_fields_have_descriptions(self):
        """Test that all required fields have descriptions."""
        for field in REQUIRED_FIELDS:
            description = get_field_description(field)
            assert len(description) > 0
            # Description should contain meaningful content, not just the field name
            assert len(description) > len(field)


class TestValidationConstants:
    """Test cases for validation constants."""
    
    def test_required_fields_count(self):
        """Test that all 19 required fields are defined."""
        assert len(REQUIRED_FIELDS) == 19
    
    def test_categorical_values_defined(self):
        """Test that categorical values are defined for all categorical fields."""
        categorical_fields = [
            'gender', 'senior_citizen', 'partner', 'dependents', 'contract',
            'paperless_billing', 'payment_method', 'phone_service', 'multiple_lines',
            'internet_service', 'online_security', 'online_backup', 'device_protection',
            'tech_support', 'streaming_tv', 'streaming_movies'
        ]
        
        for field in categorical_fields:
            assert field in VALID_CATEGORICAL_VALUES
            assert len(VALID_CATEGORICAL_VALUES[field]) > 0
    
    def test_numerical_constraints_defined(self):
        """Test that numerical constraints are defined for all numerical fields."""
        numerical_fields = ['tenure', 'monthly_charges', 'total_charges']
        
        for field in numerical_fields:
            assert field in NUMERICAL_FIELD_CONSTRAINTS
            assert 'type' in NUMERICAL_FIELD_CONSTRAINTS[field]
            assert 'min' in NUMERICAL_FIELD_CONSTRAINTS[field]
