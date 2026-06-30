"""
Unit tests for loan approval prediction
"""

import unittest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from loan_approval_prediction import generate_loan_data, preprocess_data

class TestLoanApproval(unittest.TestCase):
    
    def test_data_generation(self):
        """Test that data generation works correctly"""
        data = generate_loan_data(1000)
        self.assertEqual(len(data), 1000)
        self.assertIn('LoanStatus', data.columns)
        self.assertIn('CreditHistory', data.columns)
    
    def test_preprocessing(self):
        """Test preprocessing pipeline"""
        data = generate_loan_data(100)
        train_data = data.sample(80, random_state=42)
        test_data = data.drop(train_data.index)
        
        X_train, y_train, scaler, encoders = preprocess_data(train_data, is_training=True)
        X_test, y_test, _, _ = preprocess_data(test_data, is_training=False, 
                                                scaler=scaler, encoders=encoders)
        
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        self.assertGreater(len(X_train.columns), 10)  # Should have engineered features
    
    def test_model_accuracy(self):
        """Test that models achieve reasonable accuracy"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        
        data = generate_loan_data(2000)
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        
        X_train, y_train, scaler, encoders = preprocess_data(train_data, is_training=True)
        X_test, y_test, _, _ = preprocess_data(test_data, is_training=False,
                                                scaler=scaler, encoders=encoders)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        self.assertGreater(accuracy, 0.70)  # Should be at least 70% accurate
        
if __name__ == '__main__':
    unittest.main()