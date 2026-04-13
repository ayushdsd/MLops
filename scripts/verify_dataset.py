"""Verify the sample dataset."""
import pandas as pd

df = pd.read_csv('data/raw/telco_churn.csv')

print('=== Sample Dataset Summary ===')
print(f'\nTotal rows: {len(df)}')
print(f'Total columns: {len(df.columns)}')

print(f'\nColumns ({len(df.columns)}):')
for i, col in enumerate(df.columns, 1):
    print(f'  {i}. {col} ({df[col].dtype})')

print(f'\nChurn distribution:')
print(df['Churn'].value_counts())

print(f'\nSample statistics:')
print(f'  Tenure range: {df["tenure"].min()}-{df["tenure"].max()} months')
print(f'  Monthly charges range: ${df["MonthlyCharges"].min():.2f}-${df["MonthlyCharges"].max():.2f}')
print(f'  Total charges range: ${df["TotalCharges"].min():.2f}-${df["TotalCharges"].max():.2f}')
print(f'  Senior citizens: {(df["SeniorCitizen"] == 1).sum()} ({(df["SeniorCitizen"] == 1).sum()/len(df)*100:.1f}%)')

print(f'\nNull values: {df.isnull().sum().sum()}')
print('\nDataset is ready for use!')
