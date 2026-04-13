# Streamlit UI Usage Guide

## Quick Start

### 1. Start the Prediction API

First, ensure the FastAPI prediction service is running:

```bash
# Start the API
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Or if using Docker
docker-compose up prediction-api
```

Verify the API is running:
```bash
curl http://localhost:8000/health
```

### 2. Start the Streamlit UI

```bash
# From project root
streamlit run src/ui/app.py

# The UI will open automatically in your browser at:
# http://localhost:8501
```

## Using the UI

### Step 1: Enter Customer Information

The form is organized into three sections:

#### Demographics
- **Gender**: Male or Female
- **Senior Citizen**: Yes (1) or No (0)
- **Partner**: Whether customer has a partner
- **Dependents**: Whether customer has dependents

#### Account Information
- **Tenure**: Number of months with the company (0-72)
- **Monthly Charges**: Monthly bill amount ($)
- **Total Charges**: Total amount charged ($)
- **Contract Type**: Month-to-month, One year, or Two year
- **Paperless Billing**: Yes or No
- **Payment Method**: Electronic check, Mailed check, Bank transfer, or Credit card

#### Services
- **Phone Service**: Yes or No
- **Multiple Lines**: Yes, No, or No phone service
- **Internet Service**: DSL, Fiber optic, or No
- **Online Security**: Yes, No, or No internet service
- **Online Backup**: Yes, No, or No internet service
- **Device Protection**: Yes, No, or No internet service
- **Tech Support**: Yes, No, or No internet service
- **Streaming TV**: Yes, No, or No internet service
- **Streaming Movies**: Yes, No, or No internet service

### Step 2: Generate Prediction

Click the **"🔮 Predict Churn"** button to generate a prediction.

The UI will:
1. Validate the input data
2. Call the prediction API
3. Display the results

### Step 3: Interpret Results

The results display includes:

#### Churn Probability
- Shown as a percentage (0-100%)
- Visual progress bar
- Large, easy-to-read number

#### Risk Level
Color-coded risk classification:
- **🟢 Low Risk** (0-33%): Customer is unlikely to churn
- **🟡 Medium Risk** (33-66%): Customer may churn - consider retention actions
- **🔴 High Risk** (66-100%): Customer is likely to churn - immediate action recommended

#### Metadata
- **Model Version**: Version of the ML model used
- **Prediction Time**: Timestamp of when the prediction was made

## Configuration

### Sidebar Settings

Access settings via the sidebar (click the arrow in the top-left):

#### API URL
- Default: `http://localhost:8000`
- Change if your API is running on a different host/port
- Example: `http://api.example.com:8000`

#### Request Timeout
- Default: 5 seconds
- Range: 1-30 seconds
- Increase if API responses are slow

#### Max Retries
- Default: 3 attempts
- Range: 1-5 attempts
- Number of times to retry failed requests

### Environment Variables

Set environment variables before starting the UI:

```bash
# Linux/Mac
export API_URL=http://localhost:8000

# Windows
set API_URL=http://localhost:8000

# Or use .env file
echo "API_URL=http://localhost:8000" > .env
```

## Error Handling

### Service Unavailable

**Error Message**: "Prediction service unavailable"

**Causes**:
- API is not running
- Wrong API URL
- Network connectivity issues

**Solutions**:
1. Check API is running: `curl http://localhost:8000/health`
2. Verify API URL in sidebar settings
3. Check network connectivity
4. Review API logs for errors

### Validation Error

**Error Message**: "Validation Error: [details]"

**Causes**:
- Missing required fields
- Invalid field values
- Out-of-range numbers

**Solutions**:
1. Ensure all fields are filled
2. Check numerical values are positive
3. Verify categorical values match options
4. Review error message for specific field issues

### Request Timeout

**Error Message**: "Request timeout after X attempts"

**Causes**:
- API is slow to respond
- Network latency
- Model loading issues

**Solutions**:
1. Increase timeout in sidebar settings
2. Check API performance
3. Verify model is loaded: `curl http://localhost:8000/model-info`
4. Review API logs for performance issues

### Connection Error

**Error Message**: "Cannot connect to prediction API"

**Causes**:
- API is not running
- Firewall blocking connection
- Wrong host/port

**Solutions**:
1. Start the API service
2. Check firewall settings
3. Verify API URL is correct
4. Test connection: `curl http://localhost:8000/health`

## Example Scenarios

### Scenario 1: High-Risk Customer

**Input**:
- Gender: Female
- Senior Citizen: No
- Partner: Yes
- Dependents: No
- Tenure: 12 months
- Contract: Month-to-month
- Internet Service: Fiber optic
- Monthly Charges: $70.35
- Total Charges: $844.20
- Paperless Billing: Yes
- Payment Method: Electronic check
- Most services: No

**Expected Result**:
- Churn Probability: ~70-80%
- Risk Label: High (Red)
- Recommendation: Immediate retention action

### Scenario 2: Low-Risk Customer

**Input**:
- Gender: Male
- Senior Citizen: Yes
- Partner: Yes
- Dependents: Yes
- Tenure: 60 months
- Contract: Two year
- Internet Service: DSL
- Monthly Charges: $85.50
- Total Charges: $5,130.00
- Paperless Billing: No
- Payment Method: Bank transfer (automatic)
- Most services: Yes

**Expected Result**:
- Churn Probability: ~10-20%
- Risk Label: Low (Green)
- Recommendation: Continue current engagement

## Troubleshooting

### UI Won't Start

**Problem**: `streamlit: command not found`

**Solution**:
```bash
pip install streamlit
```

**Problem**: Port 8501 already in use

**Solution**:
```bash
# Use a different port
streamlit run src/ui/app.py --server.port=8502
```

### Predictions Not Working

**Problem**: All predictions fail

**Solution**:
1. Check API health:
   ```bash
   curl http://localhost:8000/health
   ```
2. Review API logs
3. Verify model is loaded
4. Check API URL in sidebar

**Problem**: Slow predictions

**Solution**:
1. Increase timeout in sidebar
2. Check API performance
3. Review API logs for bottlenecks
4. Consider scaling API instances

### Display Issues

**Problem**: UI looks broken

**Solution**:
1. Clear browser cache
2. Refresh the page (Ctrl+R or Cmd+R)
3. Try a different browser
4. Check Streamlit version: `streamlit --version`

**Problem**: Form fields not showing

**Solution**:
1. Refresh the page
2. Check browser console for errors
3. Verify Streamlit version compatibility

## Advanced Usage

### Batch Predictions

For multiple predictions, use the API directly:

```python
import requests

customers = [
    {...},  # Customer 1
    {...},  # Customer 2
    {...},  # Customer 3
]

for customer in customers:
    response = requests.post(
        "http://localhost:8000/predict",
        json=customer
    )
    print(response.json())
```

### Programmatic Access

Use the UI components programmatically:

```python
from src.ui.components import call_prediction_api

customer_data = {...}

response = call_prediction_api(
    customer_data=customer_data,
    api_url="http://localhost:8000",
    timeout=10,
    max_retries=3
)

print(f"Churn Probability: {response['churn_probability']:.2%}")
print(f"Risk Label: {response['risk_label']}")
```

### Custom Styling

Modify the UI appearance by editing `src/ui/components.py`:

```python
# Change risk level colors
if risk_label == "Low":
    st.markdown(
        """
        <div style='background-color: #your-color; ...'>
        ...
        </div>
        """,
        unsafe_allow_html=True
    )
```

## Best Practices

### Data Entry
1. Double-check numerical values
2. Ensure consistency (e.g., if no phone service, multiple lines should be "No phone service")
3. Use realistic values for charges and tenure

### Performance
1. Keep timeout reasonable (5-10 seconds)
2. Use 2-3 retries for production
3. Monitor API health regularly

### Security
1. Don't expose API publicly without authentication
2. Use HTTPS in production
3. Validate all inputs
4. Don't log sensitive customer data

## Support

### Documentation
- [API Documentation](http://localhost:8000/docs)
- [MLflow UI](http://localhost:5000)
- [Streamlit Documentation](https://docs.streamlit.io)

### Logs
- UI logs: Check terminal where Streamlit is running
- API logs: Check API terminal or log files
- MLflow logs: Check MLflow server logs

### Common Issues
- Check GitHub Issues for known problems
- Review API and UI logs
- Verify all services are running
- Test with example data first
