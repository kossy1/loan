import pandas as pd
import numpy as np

# Create sample data
sample_data = pd.DataFrame({
    'ApplicantIncome': [750000, 350000, 1200000, 200000, 850000, 450000, 1500000, 300000, 950000, 550000],
    'CoapplicantIncome': [300000, 50000, 200000, 0, 150000, 100000, 500000, 0, 250000, 0],
    'LoanAmount': [2000000, 500000, 3000000, 800000, 1500000, 1200000, 4000000, 300000, 2500000, 1000000],
    'LoanAmountTerm': [36, 24, 48, 36, 24, 48, 60, 36, 36, 24],
    'CreditHistory': [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female', 'Male'],
    'Married': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes'],
    'Dependents': [2, 0, 1, 3, 0, 2, 1, 4, 2, 0],
    'Education': ['Graduate', 'Not Graduate', 'Post Graduate', 'Graduate', 'Graduate', 'Not Graduate', 'Post Graduate', 'Not Graduate', 'Graduate', 'Graduate'],
    'SelfEmployed': ['No', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No'],
    'PropertyArea': ['Lagos', 'Rural', 'Abuja', 'Kano', 'Port Harcourt', 'Ibadan', 'Abuja', 'Rural', 'Lagos', 'Other Urban'],
    'EmploymentType': ['Private Sector', 'Unemployed', 'Public Sector', 'Self Employed', 'Private Sector', 'Unemployed', 'Public Sector', 'Self Employed', 'Private Sector', 'Business Owner'],
    'YearsAtJob': [5, 1, 8, 2, 4, 0, 10, 3, 6, 7],
    'Age': [32, 45, 28, 51, 34, 29, 38, 42, 30, 36]
})

# Save to CSV
sample_data.to_csv('sample_loan_applications.csv', index=False)
print("✅ Sample CSV created: sample_loan_applications.csv")
print(f"📊 Contains {len(sample_data)} applications")