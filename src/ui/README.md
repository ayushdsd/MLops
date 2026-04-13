# Streamlit UI Application

This directory contains the Streamlit-based user interface for the Customer Churn Prediction system.

## Overview

The UI provides a user-friendly web interface for business users to predict customer churn without technical knowledge. It integrates with the FastAPI prediction service to generate real-time churn risk assessments.

## Components

### `app.py`
Main Streamlit application entry point. Orchestrates the UI flow including:
- Page configuration and layout
- Header and sidebar rendering
- Form rendering and prediction handling
- Error handling and result display

### `components.py`
Reusable UI components:
- `render_input_form()`: Renders customer data input form with 19 attributes
- `call_prediction_api()`: Calls prediction API with retry logic and exponential backoff
- `display_prediction()`: Displays prediction results with color-coded risk labels
- `display_error()`: Displays user-friendly error messages

## Features

### Input Form
- Organized into logical sections: Demographics, Account Information, Services
- All 19 customer attributes with appropriate input types
- Helpful tooltips and validation
- Responsive layout with columns

### API Integration
- Configurable API URL and timeout
- Retry logic with exponential backoff (default: 3 retries)
- Timeout configuration (default: 5 seconds)
- Graceful error handling for:
  - Connection errors
  - Timeouts
  - Validation errors
  - Service unavailability

### Result Display
- Churn probability as percentage with progress bar
- Color-coded risk labels:
  - 🟢 Low (0-33%): Green
  - 🟡 Medium (33-66%): Yellow
  - 🔴 High (66-100%): Red
- Model version and prediction timestamp
- Visual indicators and emoji icons

### Error Handling
- User-friendly error messages
- Specific handling for:
  - Service unavailable
  - Validation errors
  - Timeouts
  - Connection errors
- Troubleshooting tips and suggestions

## Running the Application

### Prerequisites
- Python 3.9+
- Streamlit installed (`pip install streamlit`)
- Prediction API running at `http://localhost:8000`

### Start the Application

```bash
# From project root
streamlit run src/ui/app.py

# Or with custom port
streamlit run src/ui/app.py --server.port=8501
```

### Environment Variables

```bash
# API URL (default: http://localhost:8000)
export API_URL=http://localhost:8000

# Or use .env file
API_URL=http://localhost:8000
```

### Configuration

The application supports runtime configuration via the sidebar:
- **API URL**: Base URL of the prediction API
- **Request Timeout**: Maximum time to wait for API response (1-30 seconds)
- **Max Retries**: Maximum number of retry attempts (1-5)

## Usage

1. **Enter Customer Information**
   - Fill in all required fields in the form
   - Use the tooltips for guidance on each field
   - Ensure numerical values are within valid ranges

2. **Generate Prediction**
   - Click the "🔮 Predict Churn" button
   - Wait for the prediction to be generated
   - View the results with risk assessment

3. **Interpret Results**
   - **Churn Probability**: Likelihood of customer churn (0-100%)
   - **Risk Level**: Color-coded risk classification
   - **Model Version**: Version of the model used
   - **Prediction Time**: Timestamp of the prediction

## Testing

### Unit Tests
```bash
# Run UI component tests
pytest tests/unit/test_ui_components.py -v
```

### Integration Tests
```bash
# Run UI integration tests
pytest tests/integration/test_ui_integration.py -v
```

## Architecture

```
┌─────────────────────────────────────┐
│     Streamlit UI (Port 8501)        │
│  ┌───────────────────────────────┐  │
│  │   app.py (Main Application)   │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │  components.py (UI Components)│  │
│  │  - render_input_form()        │  │
│  │  - call_prediction_api()      │  │
│  │  - display_prediction()       │  │
│  │  - display_error()            │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
                 │
                 │ HTTP POST /predict
                 ▼
┌─────────────────────────────────────┐
│   Prediction API (Port 8000)        │
│  ┌───────────────────────────────┐  │
│  │   FastAPI Application         │  │
│  │   - /predict endpoint         │  │
│  │   - /health endpoint          │  │
│  │   - /model-info endpoint      │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

## Error Handling

The UI implements comprehensive error handling:

### Connection Errors
- Displays "Service Unavailable" message
- Provides troubleshooting steps
- Suggests checking API status

### Validation Errors
- Displays "Validation Error" message
- Shows specific field errors
- Provides guidance on fixing issues

### Timeout Errors
- Displays "Request Timeout" message
- Suggests retrying the request
- Configurable timeout duration

### Retry Logic
- Automatic retry with exponential backoff
- Configurable max retries (default: 3)
- Backoff intervals: 1s, 2s, 4s

## Customization

### Styling
The UI uses custom HTML/CSS for enhanced visual appeal:
- Color-coded risk indicators
- Responsive layout
- Custom progress bars
- Emoji icons

### Theming
Streamlit supports custom themes via `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

## Deployment

### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ui/ ./ui/

EXPOSE 8501

CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Docker Compose
```yaml
streamlit-ui:
  build:
    context: .
    dockerfile: Dockerfile.ui
  ports:
    - "8501:8501"
  environment:
    - API_URL=http://prediction-api:8000
  depends_on:
    - prediction-api
```

## Troubleshooting

### UI Won't Start
- Check Python version (3.9+)
- Verify Streamlit is installed: `pip install streamlit`
- Check port 8501 is available

### Cannot Connect to API
- Verify API is running: `curl http://localhost:8000/health`
- Check API URL in sidebar settings
- Verify network connectivity

### Prediction Fails
- Check API logs for errors
- Verify all form fields are filled correctly
- Check API health endpoint
- Review error message for specific issues

## Requirements Validation

This UI implementation validates the following requirements:

- **Requirement 7.1**: Provides form for entering customer attributes ✓
- **Requirement 7.2**: Calls Prediction API /predict endpoint ✓
- **Requirement 7.3**: Displays churn probability as percentage ✓
- **Requirement 7.4**: Displays risk label with color coding ✓
- **Requirement 7.5**: Displays error messages when API unavailable ✓

## Future Enhancements

- Batch prediction support
- Prediction history tracking
- Export results to CSV/PDF
- Model comparison view
- Real-time model performance metrics
- User authentication
- Role-based access control
