"""
Streamlit application for Customer Churn Prediction.

This is the main entry point for the Streamlit UI. It provides a user-friendly
interface for business users to predict customer churn without technical knowledge.

Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5
"""

import streamlit as st
import os
from typing import Optional

from src.ui.components import (
    render_input_form,
    call_prediction_api,
    display_prediction,
    display_error
)


def configure_page():
    """
    Configure Streamlit page settings.
    
    Sets page title, icon, layout, and initial sidebar state.
    """
    st.set_page_config(
        page_title="Customer Churn Prediction",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )


def render_header():
    """
    Render application header with title and description.
    
    Validates: Requirements 7.1
    """
    st.title("📊 Customer Churn Prediction System")
    
    st.markdown(
        """
        <div style='background-color: #e7f3ff; padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
            <p style='color: #004085; margin: 0; font-size: 16px;'>
                Predict the likelihood of customer churn using machine learning. 
                Enter customer information below to get a real-time churn risk assessment.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_sidebar():
    """
    Render sidebar with application information and settings.
    """
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown(
            """
            This application uses a Random Forest machine learning model 
            to predict customer churn probability based on customer attributes.
            
            **Risk Levels:**
            - 🟢 **Low**: 0-33% probability
            - 🟡 **Medium**: 33-66% probability
            - 🔴 **High**: 66-100% probability
            """
        )
        
        st.markdown("---")
        
        st.header("⚙️ Settings")
        
        # API URL configuration
        default_api_url = os.getenv("API_URL", "http://localhost:8000")
        api_url = st.text_input(
            "API URL",
            value=default_api_url,
            help="URL of the prediction API service"
        )
        
        # Store in session state
        st.session_state["api_url"] = api_url
        
        # API timeout configuration
        timeout = st.slider(
            "Request Timeout (seconds)",
            min_value=1,
            max_value=30,
            value=5,
            help="Maximum time to wait for API response"
        )
        st.session_state["timeout"] = timeout
        
        # Max retries configuration
        max_retries = st.slider(
            "Max Retries",
            min_value=1,
            max_value=5,
            value=3,
            help="Maximum number of retry attempts"
        )
        st.session_state["max_retries"] = max_retries
        
        st.markdown("---")
        
        st.header("📚 Documentation")
        st.markdown(
            """
            For more information, visit:
            - [API Documentation](http://localhost:8000/docs)
            - [MLflow UI](http://localhost:5000)
            """
        )


def main():
    """
    Main application entry point.
    
    Orchestrates the UI flow: form rendering, API calls, and result display.
    
    Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5
    """
    # Configure page
    configure_page()
    
    # Render header
    render_header()
    
    # Render sidebar
    render_sidebar()
    
    # Initialize session state
    if "api_url" not in st.session_state:
        st.session_state["api_url"] = os.getenv("API_URL", "http://localhost:8000")
    if "timeout" not in st.session_state:
        st.session_state["timeout"] = 5
    if "max_retries" not in st.session_state:
        st.session_state["max_retries"] = 3
    
    # Render input form
    customer_data = render_input_form()
    
    # Add predict button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        predict_button = st.button(
            "🔮 Predict Churn",
            type="primary",
            use_container_width=True,
            help="Click to generate churn prediction"
        )
    
    # Handle prediction
    if predict_button:
        # Show loading spinner
        with st.spinner("🔄 Generating prediction..."):
            try:
                # Get API configuration from session state
                api_url = st.session_state.get("api_url", "http://localhost:8000")
                timeout = st.session_state.get("timeout", 5)
                max_retries = st.session_state.get("max_retries", 3)
                
                # Call prediction API
                response = call_prediction_api(
                    customer_data=customer_data,
                    api_url=api_url,
                    timeout=timeout,
                    max_retries=max_retries
                )
                
                # Display prediction results
                display_prediction(response)
                
                # Show success message
                st.balloons()
            
            except ValueError as e:
                # Validation error
                display_error(str(e), error_type="warning")
            
            except Exception as e:
                # Other errors (connection, timeout, etc.)
                display_error(str(e), error_type="error")
    
    # Add footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>Customer Churn MLOps Pipeline | Powered by FastAPI, MLflow, and Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
