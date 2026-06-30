"""
Quick test script for loan approval prediction
Run this after training the models
"""

import pandas as pd
import numpy as np
import joblib

def test_prediction():
    # Load the trained models
    model = joblib.load('logistic_regression_model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('encoders.pkl')
    
    # Create a sample application
    sample = pd.DataFrame({
        'ApplicantIncome': [750000],
        'CoapplicantIncome': [300000],
        'LoanAmount': [2000000],
        'LoanAmountTerm': [240],
        'CreditHistory': [1.0],
        'Gender': ['Male'],
        'Married': ['Yes'],
        'Dependents': [2],
        'Education': ['Graduate'],
        'SelfEmployed': ['No'],
        'PropertyArea': ['Urban'],
        'Age': [32]
    })
    
    # Preprocess the sample
    categorical_cols = ['Gender', 'Married', 'Education', 'SelfEmployed', 'PropertyArea']
    for col in categorical_cols:
        sample[col] = encoders[col].transform(sample[col])
    
    # Feature engineering
    sample['TotalIncome'] = sample['ApplicantIncome'] + sample['CoapplicantIncome']
    sample['IncomeToLoanRatio'] = sample['TotalIncome'] / sample['LoanAmount']
    sample['LogApplicantIncome'] = np.log1p(sample['ApplicantIncome'])
    sample['LogCoapplicantIncome'] = np.log1p(sample['CoapplicantIncome'])
    sample['LogLoanAmount'] = np.log1p(sample['LoanAmount'])
    sample['LogTotalIncome'] = np.log1p(sample['TotalIncome'])
    
    sample = sample.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'TotalIncome'], axis=1)
    
    numerical_cols = ['Age', 'LoanAmountTerm', 'IncomeToLoanRatio',
                     'LogApplicantIncome', 'LogCoapplicantIncome', 'LogLoanAmount', 'LogTotalIncome']
    sample[numerical_cols] = scaler.transform(sample[numerical_cols])
    
    # Make prediction
    prob = model.predict_proba(sample)[0, 1]
    pred = model.predict(sample)[0]
    
    print("="*60)
    print("LOAN APPROVAL PREDICTION TEST")
    print("="*60)
    print(f"Approval Probability: {prob:.2%}")
    print(f"Decision: {'APPROVED ✅' if pred == 1 else 'REJECTED ❌'}")
    print("="*60)

if __name__ == "__main__":
    test_prediction()