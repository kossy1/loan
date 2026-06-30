"""
Loan Approval Prediction API
Load trained models and make predictions on new applications
Works with models from the no-XGBoost version
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

class LoanApprovalPredictor:
    """
    A class for loading trained models and making predictions
    """
    
    def __init__(self, model_path='logistic_regression_model.pkl', 
                 scaler_path='scaler.pkl',
                 encoder_path='encoders.pkl'):
        """
        Initialize the predictor with trained models
        """
        # Check if model files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file '{model_path}' not found. "
                f"Please run 'python loan_approval_prediction_no_xgboost.py' first to train and save the models."
            )
        
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Scaler file '{scaler_path}' not found. "
                f"Please run 'python loan_approval_prediction_no_xgboost.py' first to train and save the models."
            )
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.encoders = joblib.load(encoder_path)
        print(f"✓ Models loaded successfully from {model_path}")
        
    def preprocess_input(self, data):
        """
        Preprocess raw input data (single application or batch)
        """
        # Convert to DataFrame if dict
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        X = data.copy()
        
        # Encode categorical variables
        categorical_cols = ['Gender', 'Married', 'Education', 'SelfEmployed', 'PropertyArea']
        for col in categorical_cols:
            if col in X.columns:
                # Check if encoder exists
                if col not in self.encoders:
                    raise ValueError(f"Encoder for '{col}' not found in saved encoders.")
                X[col] = self.encoders[col].transform(X[col])
        
        # Feature engineering (same as in training)
        X['TotalIncome'] = X['ApplicantIncome'] + X['CoapplicantIncome']
        X['IncomeToLoanRatio'] = X['TotalIncome'] / X['LoanAmount']
        
        # Log transform
        X['LogApplicantIncome'] = np.log1p(X['ApplicantIncome'])
        X['LogCoapplicantIncome'] = np.log1p(X['CoapplicantIncome'])
        X['LogLoanAmount'] = np.log1p(X['LoanAmount'])
        X['LogTotalIncome'] = np.log1p(X['TotalIncome'])
        
        # Drop original columns
        X = X.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'TotalIncome'], axis=1)
        
        # Scale numerical features
        numerical_cols = ['Age', 'LoanAmountTerm', 'IncomeToLoanRatio',
                         'LogApplicantIncome', 'LogCoapplicantIncome', 
                         'LogLoanAmount', 'LogTotalIncome']
        
        # Ensure all columns exist
        for col in numerical_cols:
            if col not in X.columns:
                raise ValueError(f"Required column '{col}' not found in input data")
        
        X[numerical_cols] = self.scaler.transform(X[numerical_cols])
        
        return X
    
    def predict(self, data):
        """
        Make predictions on new applications
        """
        X_processed = self.preprocess_input(data)
        probabilities = self.model.predict_proba(X_processed)[:, 1]
        predictions = self.model.predict(X_processed)
        
        return predictions, probabilities
    
    def predict_with_details(self, data):
        """
        Make predictions with detailed explanation
        """
        X_processed = self.preprocess_input(data)
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)[:, 1]
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Application_ID': range(1, len(predictions) + 1),
            'Prediction': ['Approved' if p == 1 else 'Rejected' for p in predictions],
            'Approval_Probability': probabilities,
            'Confidence': np.where(probabilities > 0.5, probabilities, 1 - probabilities)
        })
        
        return results

def example_usage():
    """
    Demonstrate how to use the predictor
    """
    try:
        # Create sample applications
        new_applications = pd.DataFrame({
            'ApplicantIncome': [750000, 350000, 1200000, 200000],
            'CoapplicantIncome': [300000, 50000, 200000, 100000],
            'LoanAmount': [2000000, 500000, 3000000, 800000],
            'LoanAmountTerm': [240, 300, 360, 180],
            'CreditHistory': [1.0, 0.0, 1.0, 1.0],
            'Gender': ['Male', 'Female', 'Male', 'Female'],
            'Married': ['Yes', 'No', 'Yes', 'Yes'],
            'Dependents': [2, 0, 1, 3],
            'Education': ['Graduate', 'Not Graduate', 'Graduate', 'Graduate'],
            'SelfEmployed': ['No', 'Yes', 'No', 'No'],
            'PropertyArea': ['Urban', 'Rural', 'Semiurban', 'Urban'],
            'Age': [32, 45, 28, 51]
        })
        
        # Load predictor
        print("Loading trained models...")
        predictor = LoanApprovalPredictor()
        
        # Make predictions
        results = predictor.predict_with_details(new_applications)
        
        print("\n" + "="*60)
        print("LOAN APPROVAL PREDICTIONS")
        print("="*60)
        print(results.to_string(index=False))
        
        # Also show individual probabilities
        print("\n" + "="*60)
        print("DETAILED ANALYSIS")
        print("="*60)
        for idx, row in results.iterrows():
            print(f"\nApplication #{row['Application_ID']}:")
            print(f"  Prediction: {row['Prediction']}")
            print(f"  Approval Probability: {row['Approval_Probability']:.2%}")
            print(f"  Confidence: {row['Confidence']:.2%}")
            if row['Approval_Probability'] > 0.7:
                print("  Status: ✓ High confidence decision")
            elif row['Approval_Probability'] > 0.5:
                print("  Status: ⚠ Moderate confidence - may need manual review")
            else:
                print("  Status: ⚠ Low confidence - recommend manual review")
        
        return results
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease follow these steps:")
        print("1. Run: python loan_approval_prediction_no_xgboost.py")
        print("2. Wait for training to complete")
        print("3. Then run: python prediction_api.py")
        return None
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def batch_predict_from_csv(input_file='new_applications.csv', output_file='predictions.csv'):
    """
    Batch predict from CSV file
    """
    try:
        # Load data
        data = pd.read_csv(input_file)
        print(f"✓ Loaded {len(data)} applications from {input_file}")
        
        # Load predictor
        predictor = LoanApprovalPredictor()
        
        # Make predictions
        results = predictor.predict_with_details(data)
        
        # Save results
        results.to_csv(output_file, index=False)
        print(f"✓ Predictions saved to {output_file}")
        
        # Display summary
        print("\nPrediction Summary:")
        print(f"  Total Applications: {len(results)}")
        approved = results['Prediction'].value_counts().get('Approved', 0)
        rejected = results['Prediction'].value_counts().get('Rejected', 0)
        print(f"  Approved: {approved}")
        print(f"  Rejected: {rejected}")
        
        return results
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        if 'predictor' in str(e):
            print("\nPlease run 'python loan_approval_prediction_no_xgboost.py' first to train models.")
        else:
            print(f"\nInput file '{input_file}' not found.")
        return None

if __name__ == "__main__":
    print("="*60)
    print("LOAN APPROVAL PREDICTION API")
    print("="*60)
    
    # Check if models exist
    if not os.path.exists('logistic_regression_model.pkl'):
        print("\n⚠ Models not found. Training models first...")
        print("\nPlease run: python loan_approval_prediction_no_xgboost.py")
        print("Then run this script again.")
    else:
        print("\n✓ Models found! Making predictions...")
        example_usage()
        
        # Optional: Batch prediction from CSV
        print("\n" + "="*60)
        print("BATCH PREDICTION (Optional)")
        print("="*60)
        print("\nTo run batch prediction on a CSV file:")
        print("1. Create a CSV file with the required columns")
        print("2. Call: batch_predict_from_csv('your_file.csv')")