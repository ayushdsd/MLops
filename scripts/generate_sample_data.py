"""
Generate sample telco customer churn dataset for testing.
Creates a realistic dataset with 200 rows and all required columns.
"""
import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 200

# Generate data
data = {
    # Demographics
    'gender': np.random.choice(['Male', 'Female'], n_samples),
    'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.84, 0.16]),
    'Partner': np.random.choice(['Yes', 'No'], n_samples, p=[0.52, 0.48]),
    'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70]),
    
    # Account Information
    'tenure': np.random.randint(0, 73, n_samples),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.21, 0.24]),
    'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples, p=[0.59, 0.41]),
    'PaymentMethod': np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        n_samples,
        p=[0.34, 0.16, 0.22, 0.28]
    ),
}

# Generate monthly charges (correlated with services)
base_charges = np.random.uniform(18.0, 40.0, n_samples)
data['MonthlyCharges'] = np.round(base_charges, 2)

# Generate total charges (correlated with tenure and monthly charges)
data['TotalCharges'] = np.round(data['tenure'] * data['MonthlyCharges'] + np.random.normal(0, 100, n_samples), 2)
data['TotalCharges'] = np.maximum(data['TotalCharges'], 0)  # Ensure non-negative

# Services
data['PhoneService'] = np.random.choice(['Yes', 'No'], n_samples, p=[0.90, 0.10])

# Multiple lines depends on phone service
data['MultipleLines'] = []
for has_phone in data['PhoneService']:
    if has_phone == 'No':
        data['MultipleLines'].append('No phone service')
    else:
        data['MultipleLines'].append(np.random.choice(['Yes', 'No'], p=[0.42, 0.58]))

data['InternetService'] = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.34, 0.44, 0.22])

# Internet-dependent services
internet_dependent_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

for service in internet_dependent_services:
    data[service] = []
    for internet in data['InternetService']:
        if internet == 'No':
            data[service].append('No internet service')
        else:
            data[service].append(np.random.choice(['Yes', 'No'], p=[0.35, 0.65]))

# Generate churn (target variable)
# Higher churn probability for: month-to-month contracts, high monthly charges, low tenure
churn_prob = []
for i in range(n_samples):
    prob = 0.2  # Base probability
    
    # Contract type influence
    if data['Contract'][i] == 'Month-to-month':
        prob += 0.25
    elif data['Contract'][i] == 'One year':
        prob += 0.05
    
    # Tenure influence (lower tenure = higher churn)
    if data['tenure'][i] < 12:
        prob += 0.20
    elif data['tenure'][i] < 24:
        prob += 0.10
    
    # Monthly charges influence
    if data['MonthlyCharges'][i] > 70:
        prob += 0.15
    
    # Payment method influence
    if data['PaymentMethod'][i] == 'Electronic check':
        prob += 0.10
    
    churn_prob.append(min(prob, 0.85))  # Cap at 85%

data['Churn'] = np.random.binomial(1, churn_prob)
data['Churn'] = ['Yes' if x == 1 else 'No' for x in data['Churn']]

# Create DataFrame
df = pd.DataFrame(data)

# Ensure output directory exists
os.makedirs('data/raw', exist_ok=True)

# Save to CSV
output_path = 'data/raw/telco_churn.csv'
df.to_csv(output_path, index=False)

print(f"Sample dataset created successfully!")
print(f"Location: {output_path}")
print(f"Shape: {df.shape}")
print(f"\nColumn names:")
print(df.columns.tolist())
print(f"\nFirst few rows:")
print(df.head())
print(f"\nChurn distribution:")
print(df['Churn'].value_counts())
