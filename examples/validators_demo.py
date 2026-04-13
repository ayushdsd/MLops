"""
Demo script showing how to use the input validators.

This script demonstrates the validation utilities for customer input data.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.api.validators import (
    validate_customer_input,
    validate_required_fields,
    validate_numerical_fields,
    validate_categorical_fields,
    get_field_description
)


def print_validation_result(result, title):
    """Print validation result in a formatted way."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Valid: {result.is_valid}")
    
    if not result.is_valid:
        print(f"\nErrors ({len(result.errors)}):")
        for error in result.errors:
            print(f"  - {error.field}: {error.message}")
            if error.value is not None:
                print(f"    Value: {error.value}")
    else:
        print("✓ All validations passed!")


def demo_valid_input():
    """Demonstrate validation with valid input."""
    print("\n" + "="*60)
    print("DEMO 1: Valid Customer Input")
    print("="*60)
    
    valid_data = {
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
    
    result = validate_customer_input(valid_data)
    print_validation_result(result, "Validation Result")


def demo_missing_fields():
    """Demonstrate validation with missing required fields."""
    print("\n" + "="*60)
    print("DEMO 2: Missing Required Fields")
    print("="*60)
    
    incomplete_data = {
        'gender': 'Female',
        'senior_citizen': 1,
        'tenure': 24
        # Missing many required fields
    }
    
    result = validate_required_fields(incomplete_data)
    print_validation_result(result, "Required Fields Validation")


def demo_invalid_numerical():
    """Demonstrate validation with invalid numerical values."""
    print("\n" + "="*60)
    print("DEMO 3: Invalid Numerical Values")
    print("="*60)
    
    invalid_numerical = {
        'tenure': -5,  # Negative tenure
        'monthly_charges': 0.0,  # Zero monthly charges (must be > 0)
        'total_charges': -100.0  # Negative total charges
    }
    
    result = validate_numerical_fields(invalid_numerical)
    print_validation_result(result, "Numerical Fields Validation")


def demo_invalid_categorical():
    """Demonstrate validation with invalid categorical values."""
    print("\n" + "="*60)
    print("DEMO 4: Invalid Categorical Values")
    print("="*60)
    
    invalid_categorical = {
        'gender': 'Other',  # Invalid gender
        'senior_citizen': 2,  # Invalid senior_citizen
        'contract': 'Three year',  # Invalid contract
        'internet_service': 'Cable'  # Invalid internet service
    }
    
    result = validate_categorical_fields(invalid_categorical)
    print_validation_result(result, "Categorical Fields Validation")


def demo_comprehensive_validation():
    """Demonstrate comprehensive validation with multiple errors."""
    print("\n" + "="*60)
    print("DEMO 5: Comprehensive Validation (Multiple Errors)")
    print("="*60)
    
    bad_data = {
        'gender': 'Unknown',  # Invalid categorical
        'senior_citizen': 0,
        'tenure': -10,  # Invalid numerical
        'monthly_charges': 0.0,  # Invalid numerical
        'contract': 'Invalid'  # Invalid categorical
        # Missing many required fields
    }
    
    result = validate_customer_input(bad_data)
    print_validation_result(result, "Comprehensive Validation")
    
    print("\n" + "-"*60)
    print("Error Summary:")
    print("-"*60)
    print(result.get_error_summary())


def demo_field_descriptions():
    """Demonstrate field descriptions."""
    print("\n" + "="*60)
    print("DEMO 6: Field Descriptions")
    print("="*60)
    
    fields = ['gender', 'tenure', 'monthly_charges', 'contract', 'internet_service']
    
    for field in fields:
        description = get_field_description(field)
        print(f"\n{field}:")
        print(f"  {description}")


def main():
    """Run all demos."""
    print("\n" + "#"*60)
    print("# Customer Input Validators Demo")
    print("#"*60)
    
    demo_valid_input()
    demo_missing_fields()
    demo_invalid_numerical()
    demo_invalid_categorical()
    demo_comprehensive_validation()
    demo_field_descriptions()
    
    print("\n" + "#"*60)
    print("# Demo Complete!")
    print("#"*60)


if __name__ == "__main__":
    main()
