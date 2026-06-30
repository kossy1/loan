"""
Loan Approval Prediction using Logistic Regression and XGBoost
Complete end-to-end implementation with data generation, preprocessing, model training, and evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================
# PART 1: DATA GENERATION (Synthetic Loan Dataset)
# ============================================

def generate_loan_data(n_samples=10000):
    """
    Generate synthetic loan application data with realistic patterns
    """
    np.random.seed(42)
    
    # Demographic features
    age = np.random.randint(18, 70, n_samples)
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.55, 0.45])
    married = np.random.choice(['Yes', 'No'], n_samples, p=[0.65, 0.35])
    dependents = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.25, 0.30, 0.25, 0.15, 0.05])
    education = np.random.choice(['Graduate', 'Not Graduate'], n_samples, p=[0.75, 0.25])
    self_employed = np.random.choice(['Yes', 'No'], n_samples, p=[0.15, 0.85])
    
    # Financial features - FIXED: using proper parameter ordering
    applicant_income = np.random.lognormal(mean=10.8, sigma=0.5, size=n_samples).astype(int)
    applicant_income = np.clip(applicant_income, 50000, 5000000)  # INR
    
    coapplicant_income = np.random.lognormal(mean=10.2, sigma=0.7, size=n_samples).astype(int)
    coapplicant_income = np.clip(coapplicant_income, 0, 3000000)  # INR
    
    loan_amount = np.random.lognormal(mean=11.5, sigma=0.6, size=n_samples).astype(int)
    loan_amount = np.clip(loan_amount, 50000, 10000000)  # INR
    
    loan_amount_term = np.random.choice([180, 240, 300, 360, 480], n_samples, 
                                        p=[0.2, 0.3, 0.3, 0.15, 0.05])
    
    credit_history = np.random.choice([0.0, 1.0], n_samples, p=[0.25, 0.75])
    
    property_area = np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples, 
                                      p=[0.35, 0.40, 0.25])
    
    # Generate target variable based on realistic rules
    # Higher chance of approval with: good credit history, higher income, lower loan amount relative to income
    approval_prob = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Base probability
        prob = 0.3
        
        # Credit history is the strongest predictor
        if credit_history[i] == 1.0:
            prob += 0.35
        
        # Income effect (higher income = higher approval)
        if applicant_income[i] > 500000:
            prob += 0.10
        if applicant_income[i] > 1000000:
            prob += 0.10
            
        # Co-applicant income boost
        if coapplicant_income[i] > 200000:
            prob += 0.05
            
        # Loan amount relative to income
        income_ratio = (applicant_income[i] + coapplicant_income[i]) / loan_amount[i]
        if income_ratio > 0.5:
            prob += 0.10
        if income_ratio > 1.0:
            prob += 0.10
            
        # Employment status
        if self_employed[i] == 'Yes':
            prob -= 0.05
            
        # Education
        if education[i] == 'Graduate':
            prob += 0.05
            
        # Dependents
        if dependents[i] >= 3:
            prob -= 0.05
            
        # Property area
        if property_area[i] == 'Urban':
            prob += 0.05
        elif property_area[i] == 'Rural':
            prob -= 0.05
            
        approval_prob[i] = np.clip(prob, 0, 0.95)
    
    # Generate target with probabilities
    loan_status = np.random.binomial(1, approval_prob)
    
    # Create dataframe
    data = pd.DataFrame({
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'LoanAmountTerm': loan_amount_term,
        'CreditHistory': credit_history,
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'SelfEmployed': self_employed,
        'PropertyArea': property_area,
        'Age': age,
        'LoanStatus': loan_status
    })
    
    return data

# ============================================
# PART 2: DATA PREPROCESSING
# ============================================

def preprocess_data(df, is_training=True, scaler=None, encoders=None):
    """
    Complete preprocessing pipeline: handle missing values, encode, and scale
    """
    df_copy = df.copy()
    
    # Separate features and target
    if is_training:
        X = df_copy.drop('LoanStatus', axis=1)
        y = df_copy['LoanStatus']
    else:
        X = df_copy
        y = None
    
    # --- Handle Missing Values ---
    # In real data, we'd handle missing values. For our synthetic data, we'll add some for demonstration
    # For this example, we'll assume data is clean
    
    # --- Encode Categorical Variables ---
    categorical_cols = ['Gender', 'Married', 'Education', 'SelfEmployed', 'PropertyArea']
    
    # For training: fit encoders
    if is_training:
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le
    else:
        # For testing: use existing encoders
        for col in categorical_cols:
            X[col] = encoders[col].transform(X[col])
    
    # 'Dependents' is already numeric
    
    # --- Feature Engineering ---
    # Total Income
    X['TotalIncome'] = X['ApplicantIncome'] + X['CoapplicantIncome']
    
    # Income to Loan Ratio
    X['IncomeToLoanRatio'] = X['TotalIncome'] / X['LoanAmount']
    
    # Log transform skewed features
    X['LogApplicantIncome'] = np.log1p(X['ApplicantIncome'])
    X['LogCoapplicantIncome'] = np.log1p(X['CoapplicantIncome'])
    X['LogLoanAmount'] = np.log1p(X['LoanAmount'])
    X['LogTotalIncome'] = np.log1p(X['TotalIncome'])
    
    # Drop original skewed features (keep for interpretation)
    X = X.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'TotalIncome'], axis=1)
    
    # --- Scale Numerical Features ---
    numerical_cols = ['Age', 'LoanAmountTerm', 'IncomeToLoanRatio', 
                      'LogApplicantIncome', 'LogCoapplicantIncome', 'LogLoanAmount', 'LogTotalIncome']
    
    if is_training:
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    else:
        X[numerical_cols] = scaler.transform(X[numerical_cols])
    
    return X, y, scaler, encoders

# ============================================
# PART 3: MODEL TRAINING
# ============================================

def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression model with hyperparameter tuning
    """
    print("\n" + "="*60)
    print("TRAINING LOGISTIC REGRESSION")
    print("="*60)
    
    # Define hyperparameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'class_weight': ['balanced', None],
        'max_iter': [1000]
    }
    
    # Base model
    lr = LogisticRegression(random_state=42)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='roc_auc', 
                               n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation ROC-AUC: {grid_search.best_score_:.4f}")
    
    return best_model

def train_xgboost(X_train, y_train):
    """
    Train XGBoost model with hyperparameter tuning
    """
    print("\n" + "="*60)
    print("TRAINING XGBOOST")
    print("="*60)
    
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'scale_pos_weight': [1, 2, 3]  # Handle class imbalance
    }
    
    # Base model
    xgb = XGBClassifier(random_state=42, eval_metric='logloss')
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='roc_auc',
                               n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation ROC-AUC: {grid_search.best_score_:.4f}")
    
    return best_model

def train_with_smote(X_train, y_train):
    """
    Train models using SMOTE for handling class imbalance
    """
    print("\n" + "="*60)
    print("TRAINING WITH SMOTE (IMBALANCE HANDLING)")
    print("="*60)
    
    # Logistic Regression with SMOTE
    lr_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    lr_param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['lbfgs', 'liblinear'],
        'smote__k_neighbors': [3, 5, 7]
    }
    
    lr_grid = GridSearchCV(lr_pipeline, lr_param_grid, cv=5, 
                           scoring='roc_auc', n_jobs=-1, verbose=1)
    lr_grid.fit(X_train, y_train)
    
    # XGBoost with SMOTE
    xgb_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))
    ])
    
    xgb_param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.01, 0.1],
        'smote__k_neighbors': [3, 5]
    }
    
    xgb_grid = GridSearchCV(xgb_pipeline, xgb_param_grid, cv=5,
                            scoring='roc_auc', n_jobs=-1, verbose=1)
    xgb_grid.fit(X_train, y_train)
    
    print(f"\nBest LR with SMOTE ROC-AUC: {lr_grid.best_score_:.4f}")
    print(f"Best XGB with SMOTE ROC-AUC: {xgb_grid.best_score_:.4f}")
    
    return lr_grid.best_estimator_, xgb_grid.best_estimator_

# ============================================
# PART 4: MODEL EVALUATION
# ============================================

def evaluate_model(model, X_test, y_test, model_name):
    """
    Comprehensive model evaluation with multiple metrics
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n{'='*50}")
    print(f"EVALUATION: {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"[[{cm[0,0]:4d} {cm[0,1]:4d}]")
    print(f" [{cm[1,0]:4d} {cm[1,1]:4d}]]")
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 
            'f1': f1, 'roc_auc': roc_auc}, cm

def plot_roc_curves(models, X_test, y_test):
    """
    Plot ROC curves for all models
    """
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Loan Approval Prediction')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300)
    plt.show()

def plot_confusion_matrices(models, X_test, y_test):
    """
    Plot confusion matrices for all models
    """
    fig, axes = plt.subplots(1, len(models), figsize=(15, 5))
    
    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'{name}\nConfusion Matrix')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xticklabels(['Rejected', 'Approved'])
        axes[idx].set_yticklabels(['Rejected', 'Approved'])
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300)
    plt.show()

def plot_feature_importance(models, X):
    """
    Plot feature importance for models that support it
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Logistic Regression coefficients
    lr_model = models.get('Logistic Regression')
    if lr_model and hasattr(lr_model, 'coef_'):
        coefs = lr_model.coef_[0]
        feature_names = X.columns
        sorted_idx = np.argsort(np.abs(coefs))[-10:]  # Top 10
        
        axes[0].barh(feature_names[sorted_idx], coefs[sorted_idx])
        axes[0].set_title('Logistic Regression - Feature Coefficients')
        axes[0].set_xlabel('Coefficient Value')
        axes[0].axvline(0, color='black', linestyle='-', linewidth=0.5)
    
    # XGBoost feature importance
    xgb_model = models.get('XGBoost')
    if xgb_model and hasattr(xgb_model, 'feature_importances_'):
        importances = xgb_model.feature_importances_
        feature_names = X.columns
        sorted_idx = np.argsort(importances)[-10:]  # Top 10
        
        axes[1].barh(feature_names[sorted_idx], importances[sorted_idx])
        axes[1].set_title('XGBoost - Feature Importance')
        axes[1].set_xlabel('Importance Score')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.show()

# ============================================
# PART 5: MAIN EXECUTION
# ============================================

def main():
    print("="*60)
    print("LOAN APPROVAL PREDICTION")
    print("Logistic Regression vs XGBoost")
    print("="*60)
    
    # --- Step 1: Generate Data ---
    print("\n[1] Generating synthetic loan dataset...")
    data = generate_loan_data(n_samples=10000)
    print(f"Dataset shape: {data.shape}")
    print(f"Approval rate: {data['LoanStatus'].mean()*100:.1f}%")
    
    # --- Step 2: Train-Test Split ---
    print("\n[2] Splitting data into train/test sets...")
    train_data, test_data = train_test_split(data, test_size=0.2, 
                                             random_state=42, stratify=data['LoanStatus'])
    print(f"Training samples: {len(train_data)}, Test samples: {len(test_data)}")
    
    # --- Step 3: Preprocess Data ---
    print("\n[3] Preprocessing data...")
    X_train, y_train, scaler, encoders = preprocess_data(train_data, is_training=True)
    X_test, y_test, _, _ = preprocess_data(test_data, is_training=False, 
                                            scaler=scaler, encoders=encoders)
    print(f"Features after preprocessing: {X_train.shape[1]}")
    
    # --- Step 4: Train Models ---
    print("\n[4] Training models...")
    
    # Standard approach (without SMOTE)
    lr_model = train_logistic_regression(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    
    # SMOTE approach
    print("\n[5] Training with SMOTE for imbalance handling...")
    lr_smote_model, xgb_smote_model = train_with_smote(X_train, y_train)
    
    # --- Step 5: Evaluate Models ---
    print("\n[6] Evaluating models...")
    
    models = {
        'Logistic Regression': lr_model,
        'XGBoost': xgb_model,
        'Logistic Regression + SMOTE': lr_smote_model,
        'XGBoost + SMOTE': xgb_smote_model
    }
    
    results = {}
    for name, model in models.items():
        metrics, cm = evaluate_model(model, X_test, y_test, name)
        results[name] = metrics
    
    # --- Step 6: Visualizations ---
    print("\n[7] Generating visualizations...")
    
    plot_roc_curves(models, X_test, y_test)
    plot_confusion_matrices(models, X_test, y_test)
    plot_feature_importance(models, X_test)
    
    # --- Step 7: Feature Importance Comparison ---
    print("\n[8] Analyzing feature importance...")
    
    # Create feature importance dataframe for logistic regression
    if hasattr(lr_model, 'coef_'):
        lr_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Coefficient': lr_model.coef_[0]
        }).sort_values('Coefficient', ascending=False)
        print("\nTop 5 Important Features (Logistic Regression):")
        print(lr_importance.head(5))
    
    # Feature importance for XGBoost
    if hasattr(xgb_model, 'feature_importances_'):
        xgb_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': xgb_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        print("\nTop 5 Important Features (XGBoost):")
        print(xgb_importance.head(5))
    
    # --- Step 8: Save Models ---
    print("\n[9] Saving models...")
    joblib.dump(lr_model, 'logistic_regression_model.pkl')
    joblib.dump(xgb_model, 'xgboost_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(encoders, 'encoders.pkl')
    print("✓ Models saved successfully!")
    
    # --- Step 9: Business Impact Summary ---
    print("\n" + "="*60)
    print("BUSINESS IMPACT SUMMARY")
    print("="*60)
    
    print(f"\nDataset Statistics:")
    print(f"  Total Applications: {len(data)}")
    print(f"  Approval Rate: {data['LoanStatus'].mean()*100:.1f}%")
    
    print(f"\nModel Performance Comparison:")
    results_df = pd.DataFrame(results).T
    print(results_df.round(4))
    
    # Best performing model
    best_model = results_df['roc_auc'].idxmax()
    best_auc = results_df.loc[best_model, 'roc_auc']
    print(f"\n✓ Best Model: {best_model} (ROC-AUC: {best_auc:.4f})")
    
    # --- Step 10: Example Predictions ---
    print("\n[10] Making sample predictions on new applications...")
    
    # Get a few samples from test set
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    sample_features = X_test.iloc[sample_indices]
    sample_true = y_test.iloc[sample_indices]
    
    for idx in range(len(sample_features)):
        features = sample_features.iloc[idx:idx+1]
        true_label = sample_true.iloc[idx]
        
        lr_prob = lr_model.predict_proba(features)[0, 1]
        xgb_prob = xgb_model.predict_proba(features)[0, 1]
        
        print(f"\nApplication {idx+1}:")
        print(f"  True Label: {'Approved' if true_label == 1 else 'Rejected'}")
        print(f"  LR Approval Probability: {lr_prob:.2%}")
        print(f"  XGB Approval Probability: {xgb_prob:.2%}")
        print(f"  LR Prediction: {'Approved' if lr_prob >= 0.5 else 'Rejected'}")
        print(f"  XGB Prediction: {'Approved' if xgb_prob >= 0.5 else 'Rejected'}")
    
    print("\n" + "="*60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()