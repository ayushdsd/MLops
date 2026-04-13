"""
UI components for the Streamlit Customer Churn Prediction application.

This module provides reusable components for rendering input forms,
displaying predictions, and handling errors.

Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5
"""

import streamlit as st
from typing import Dict, Any, Optional
import requests
from datetime import datetime
import time


def render_input_form() -> Dict[str, Any]:
    """
    Render customer data input form.
    
    Creates a comprehensive form with all 19 customer attributes organized
    into logical sections: Demographics, Account Information, and Services.
    
    Returns:
        Dictionary containing all customer input data
        
    Validates: Requirements 7.1
    """
    st.subheader("📋 Customer Information")
    
    customer_data = {}
    
    # Demographics Section
    st.markdown("### Demographics")
    col1, col2 = st.columns(2)
    
    with col1:
        customer_data["gender"] = st.selectbox(
            "Gender",
            options=["Male", "Female"],
            help="Customer gender"
        )
        
        customer_data["senior_citizen"] = st.selectbox(
            "Senior Citizen",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="Whether customer is a senior citizen (65+)"
        )
    
    with col2:
        customer_data["partner"] = st.selectbox(
            "Partner",
            options=["Yes", "No"],
            help="Whether customer has a partner"
        )
        
        customer_data["dependents"] = st.selectbox(
            "Dependents",
            options=["Yes", "No"],
            help="Whether customer has dependents"
        )
    
    # Account Information Section
    st.markdown("### Account Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        customer_data["tenure"] = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12,
            help="Number of months with the company"
        )
    
    with col2:
        customer_data["monthly_charges"] = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=0.01,
            help="Monthly bill amount"
        )
    
    with col3:
        customer_data["total_charges"] = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=840.0,
            step=0.01,
            help="Total amount charged to customer"
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        customer_data["contract"] = st.selectbox(
            "Contract Type",
            options=["Month-to-month", "One year", "Two year"],
            help="Contract duration"
        )
        
        customer_data["paperless_billing"] = st.selectbox(
            "Paperless Billing",
            options=["Yes", "No"],
            help="Whether customer uses paperless billing"
        )
    
    with col2:
        customer_data["payment_method"] = st.selectbox(
            "Payment Method",
            options=[
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ],
            help="Payment method used by customer"
        )
    
    # Services Section
    st.markdown("### Services")
    col1, col2 = st.columns(2)
    
    with col1:
        customer_data["phone_service"] = st.selectbox(
            "Phone Service",
            options=["Yes", "No"],
            help="Whether customer has phone service"
        )
        
        customer_data["multiple_lines"] = st.selectbox(
            "Multiple Lines",
            options=["Yes", "No", "No phone service"],
            help="Whether customer has multiple phone lines"
        )
        
        customer_data["internet_service"] = st.selectbox(
            "Internet Service",
            options=["DSL", "Fiber optic", "No"],
            help="Type of internet service"
        )
        
        customer_data["online_security"] = st.selectbox(
            "Online Security",
            options=["Yes", "No", "No internet service"],
            help="Whether customer has online security service"
        )
        
        customer_data["online_backup"] = st.selectbox(
            "Online Backup",
            options=["Yes", "No", "No internet service"],
            help="Whether customer has online backup service"
        )
    
    with col2:
        customer_data["device_protection"] = st.selectbox(
            "Device Protection",
            options=["Yes", "No", "No internet service"],
            help="Whether customer has device protection service"
        )
        
        customer_data["tech_support"] = st.selectbox(
            "Tech Support",
            options=["Yes", "No", "No internet service"],
            help="Whether customer has tech support service"
        )
        
        customer_data["streaming_tv"] = st.selectbox(
            "Streaming TV",
            options=["Yes", "No", "No internet service"],
            help="Whether customer has streaming TV service"
        )
        
        customer_data["streaming_movies"] = st.selectbox(
            "Streaming Movies",
            options=["Yes", "No", "No internet service"],
            help="Whether customer has streaming movies service"
        )
    
    return customer_data


def call_prediction_api(
    customer_data: Dict[str, Any],
    api_url: str = "http://localhost:8000",
    timeout: int = 5,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Call the prediction API with retry logic and exponential backoff.
    
    Makes a POST request to the /predict endpoint with customer data.
    Implements retry logic with exponential backoff for transient failures.
    
    Args:
        customer_data: Dictionary containing customer attributes
        api_url: Base URL of the prediction API
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        
    Returns:
        Dictionary containing prediction response
        
    Raises:
        requests.RequestException: If API call fails after all retries
        
    Validates: Requirements 7.2
    """
    endpoint = f"{api_url}/predict"
    
    for attempt in range(max_retries):
        try:
            # Make API request with timeout
            response = requests.post(
                endpoint,
                json=customer_data,
                timeout=timeout
            )
            
            # Check for successful response
            if response.status_code == 200:
                return response.json()
            
            # Handle error responses
            elif response.status_code == 400:
                # Validation error - don't retry
                error_data = response.json()
                raise ValueError(f"Validation Error: {error_data.get('detail', 'Invalid input')}")
            
            elif response.status_code == 503:
                # Service unavailable - retry with backoff
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    time.sleep(wait_time)
                    continue
                else:
                    raise requests.RequestException("Prediction service unavailable. Please try again later.")
            
            else:
                # Other errors
                response.raise_for_status()
        
        except requests.Timeout:
            # Timeout - retry with backoff
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            else:
                raise requests.RequestException(f"Request timeout after {max_retries} attempts")
        
        except requests.ConnectionError:
            # Connection error - retry with backoff
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            else:
                raise requests.RequestException(
                    f"Cannot connect to prediction API at {api_url}. "
                    "Please ensure the API service is running."
                )
        
        except ValueError:
            # Validation error - don't retry
            raise
        
        except Exception as e:
            # Unexpected error
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            else:
                raise requests.RequestException(f"Unexpected error: {str(e)}")
    
    # Should not reach here, but just in case
    raise requests.RequestException(f"Failed after {max_retries} attempts")


def display_prediction(response: Dict[str, Any]) -> None:
    """
    Display prediction results with visual indicators.
    
    Shows churn probability as percentage, risk label with color coding,
    model version, and prediction timestamp. Uses progress bars and emoji
    icons for visual appeal.
    
    Args:
        response: Dictionary containing prediction response from API
        
    Validates: Requirements 7.3, 7.4
    """
    st.markdown("---")
    st.subheader("🎯 Prediction Results")
    
    # Extract response data
    churn_probability = response.get("churn_probability", 0.0)
    risk_label = response.get("risk_label", "Unknown")
    model_version = response.get("model_version", "unknown")
    timestamp = response.get("timestamp", datetime.utcnow().isoformat())
    
    # Convert probability to percentage
    probability_percent = churn_probability * 100
    
    # Display churn probability with progress bar
    st.markdown("### Churn Probability")
    st.progress(churn_probability)
    st.markdown(f"<h2 style='text-align: center;'>{probability_percent:.1f}%</h2>", unsafe_allow_html=True)
    
    # Display risk label with color coding
    st.markdown("### Risk Level")
    
    if risk_label == "Low":
        st.success(f"✅ **{risk_label} Risk**")
        st.markdown(
            """
            <div style='background-color: #d4edda; padding: 20px; border-radius: 10px; text-align: center;'>
                <h3 style='color: #155724; margin: 0;'>✅ Low Risk</h3>
                <p style='color: #155724; margin: 10px 0 0 0;'>Customer is unlikely to churn</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif risk_label == "Medium":
        st.warning(f"⚠️ **{risk_label} Risk**")
        st.markdown(
            """
            <div style='background-color: #fff3cd; padding: 20px; border-radius: 10px; text-align: center;'>
                <h3 style='color: #856404; margin: 0;'>⚠️ Medium Risk</h3>
                <p style='color: #856404; margin: 10px 0 0 0;'>Customer may churn - consider retention actions</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif risk_label == "High":
        st.error(f"🚨 **{risk_label} Risk**")
        st.markdown(
            """
            <div style='background-color: #f8d7da; padding: 20px; border-radius: 10px; text-align: center;'>
                <h3 style='color: #721c24; margin: 0;'>🚨 High Risk</h3>
                <p style='color: #721c24; margin: 10px 0 0 0;'>Customer is likely to churn - immediate action recommended</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info(f"ℹ️ **{risk_label} Risk**")
    
    # Display metadata
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Model Version:**")
        st.code(model_version)
    
    with col2:
        st.markdown("**Prediction Time:**")
        # Format timestamp for display
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except:
            formatted_time = timestamp
        st.code(formatted_time)


def display_error(error: str, error_type: str = "error") -> None:
    """
    Display user-friendly error messages.
    
    Shows error messages with appropriate styling based on error type.
    Handles API unavailability, validation errors, and other failures gracefully.
    
    Args:
        error: Error message to display
        error_type: Type of error ("error", "warning", "info")
        
    Validates: Requirements 7.5
    """
    st.markdown("---")
    
    # Determine error icon and styling based on type
    if "unavailable" in error.lower() or "connect" in error.lower():
        st.error("🔌 **Service Unavailable**")
        st.markdown(
            f"""
            <div style='background-color: #f8d7da; padding: 20px; border-radius: 10px;'>
                <p style='color: #721c24; margin: 0;'>{error}</p>
                <p style='color: #721c24; margin: 10px 0 0 0;'><strong>Troubleshooting:</strong></p>
                <ul style='color: #721c24; margin: 5px 0 0 20px;'>
                    <li>Ensure the prediction API is running</li>
                    <li>Check that the API URL is correct</li>
                    <li>Verify network connectivity</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    elif "validation" in error.lower() or "invalid" in error.lower():
        st.warning("⚠️ **Validation Error**")
        st.markdown(
            f"""
            <div style='background-color: #fff3cd; padding: 20px; border-radius: 10px;'>
                <p style='color: #856404; margin: 0;'>{error}</p>
                <p style='color: #856404; margin: 10px 0 0 0;'><strong>Please check:</strong></p>
                <ul style='color: #856404; margin: 5px 0 0 20px;'>
                    <li>All required fields are filled</li>
                    <li>Numerical values are within valid ranges</li>
                    <li>Categorical values match expected options</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    elif "timeout" in error.lower():
        st.error("⏱️ **Request Timeout**")
        st.markdown(
            f"""
            <div style='background-color: #f8d7da; padding: 20px; border-radius: 10px;'>
                <p style='color: #721c24; margin: 0;'>{error}</p>
                <p style='color: #721c24; margin: 10px 0 0 0;'>The prediction service is taking longer than expected. Please try again.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    else:
        # Generic error
        if error_type == "warning":
            st.warning(f"⚠️ **Warning**")
        elif error_type == "info":
            st.info(f"ℹ️ **Information**")
        else:
            st.error(f"❌ **Error**")
        
        st.markdown(
            f"""
            <div style='background-color: #f8d7da; padding: 20px; border-radius: 10px;'>
                <p style='color: #721c24; margin: 0;'>{error}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Add retry suggestion
    st.markdown("---")
    st.info("💡 **Tip:** Try submitting the form again or contact support if the issue persists.")
