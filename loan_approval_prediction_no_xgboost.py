"""
Loan Approval Prediction - NO XGBOOST VERSION
Uses Logistic Regression, Random Forest, and Gradient Boosting
Works perfectly on Windows without XGBoost issues
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================
# DATA GENERATION
# ============================================

def generate_loan_data(n_samples=5000):
    """Generate synthetic loan application data"""
    np.random.seed(42)
    
    age = np.random.randint(18, 70, n_samples)
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.55, 0.45])
    married = np.random.choice(['Yes', 'No'], n_samples, p=[0.65, 0.35])
    dependents = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.25, 0.30, 0.25, 0.15, 0.05])
    education = np.random.choice(['Graduate', 'Not Graduate'], n_samples, p=[0.75, 0.25])
    self_employed = np.random.choice(['Yes', 'No'], n_samples, p=[0.15, 0.85])
    
    applicant_income = np.random.lognormal(mean=10.8, sigma=0.5, size=n_samples).astype(int)
    applicant_income = np.clip(applicant_income, 50000, 5000000)
    
    coapplicant_income = np.random.lognormal(mean=10.2, sigma=0.7, size=n_samples).astype(int)
    coapplicant_income = np.clip(coapplicant_income, 0, 3000000)
    
    loan_amount = np.random.lognormal(mean=11.5, sigma=0.6, size=n_samples).astype(int)
    loan_amount = np.clip(loan_amount, 50000, 10000000)
    
    loan_amount_term = np.random.choice([180, 240, 300, 360, 480], n_samples, 
                                        p=[0.2, 0.3, 0.3, 0.15, 0.05])
    
    credit_history = np.random.choice([0.0, 1.0], n_samples, p=[0.25, 0.75])
    property_area = np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples, 
                                      p=[0.35, 0.40, 0.25])
    
    # Generate target
    approval_prob = np.zeros(n_samples)
    for i in range(n_samples):
        prob = 0.3
        if credit_history[i] == 1.0:
            prob += 0.35
        if applicant_income[i] > 500000:
            prob += 0.10
        if applicant_income[i] > 1000000:
            prob += 0.10
        if coapplicant_income[i] > 200000:
            prob += 0.05
        income_ratio = (applicant_income[i] + coapplicant_income[i]) / loan_amount[i]
        if income_ratio > 0.5:
            prob += 0.10
        if income_ratio > 1.0:
            prob += 0.10
        if self_employed[i] == 'Yes':
            prob -= 0.05
        if education[i] == 'Graduate':
            prob += 0.05
        if dependents[i] >= 3:
            prob -= 0.05
        if property_area[i] == 'Urban':
            prob += 0.05
        elif property_area[i] == 'Rural':
            prob -= 0.05
        approval_prob[i] = np.clip(prob, 0, 0.95)
    
    loan_status = np.random.binomial(1, approval_prob)
    
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
# DATA PREPROCESSING - FIXED
# ============================================

def preprocess_data(df, is_training=True, scaler=None, encoders=None):
    """
    Complete preprocessing pipeline - FIXED: properly separates features and target
    """
    df_copy = df.copy()
    
    # Separate features and target for training data
    if is_training:
        if 'LoanStatus' in df_copy.columns:
            y = df_copy['LoanStatus']
            X = df_copy.drop('LoanStatus', axis=1)
        else:
            raise ValueError("Training data must contain 'LoanStatus' column")
    else:
        # For test data, we might have LoanStatus or not
        if 'LoanStatus' in df_copy.columns:
            y = df_copy['LoanStatus']
            X = df_copy.drop('LoanStatus', axis=1)
        else:
            y = None
            X = df_copy
    
    # Encode categorical variables
    categorical_cols = ['Gender', 'Married', 'Education', 'SelfEmployed', 'PropertyArea']
    
    if is_training:
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le
    else:
        for col in categorical_cols:
            if col in X.columns:
                X[col] = encoders[col].transform(X[col])
    
    # Feature engineering
    X['TotalIncome'] = X['ApplicantIncome'] + X['CoapplicantIncome']
    X['IncomeToLoanRatio'] = X['TotalIncome'] / X['LoanAmount']
    X['LogApplicantIncome'] = np.log1p(X['ApplicantIncome'])
    X['LogCoapplicantIncome'] = np.log1p(X['CoapplicantIncome'])
    X['LogLoanAmount'] = np.log1p(X['LoanAmount'])
    X['LogTotalIncome'] = np.log1p(X['TotalIncome'])
    
    # Drop original columns (keep only engineered features)
    X = X.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'TotalIncome'], axis=1)
    
    # Scale numerical features
    numerical_cols = ['Age', 'LoanAmountTerm', 'IncomeToLoanRatio', 
                      'LogApplicantIncome', 'LogCoapplicantIncome', 'LogLoanAmount', 'LogTotalIncome']
    
    if is_training:
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    else:
        X[numerical_cols] = scaler.transform(X[numerical_cols])
    
    return X, y, scaler, encoders

# ============================================
# MODEL TRAINING (NO XGBOOST)
# ============================================

def train_models(X_train, y_train):
    """Train all models without XGBoost"""
    
    print("\n" + "="*60)
    print("TRAINING MODELS (No XGBoost)")
    print("="*60)
    
    # 1. Logistic Regression
    print("\n[1/3] Training Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr_param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear'],
        'class_weight': ['balanced', None]
    }
    lr_grid = GridSearchCV(lr, lr_param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
    lr_grid.fit(X_train, y_train)
    lr_best = lr_grid.best_estimator_
    print(f"✓ Best LR ROC-AUC: {lr_grid.best_score_:.4f}")
    
    # 2. Random Forest
    print("\n[2/3] Training Random Forest...")
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced', None]
    }
    rf_grid = GridSearchCV(rf, rf_param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
    rf_grid.fit(X_train, y_train)
    rf_best = rf_grid.best_estimator_
    print(f"✓ Best RF ROC-AUC: {rf_grid.best_score_:.4f}")
    
    # 3. Gradient Boosting (XGBoost alternative)
    print("\n[3/3] Training Gradient Boosting...")
    gb = GradientBoostingClassifier(random_state=42)
    gb_param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    }
    gb_grid = GridSearchCV(gb, gb_param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
    gb_grid.fit(X_train, y_train)
    gb_best = gb_grid.best_estimator_
    print(f"✓ Best GB ROC-AUC: {gb_grid.best_score_:.4f}")
    
    return lr_best, rf_best, gb_best

def train_with_smote(X_train, y_train):
    """Train models with SMOTE"""
    
    print("\n" + "="*60)
    print("TRAINING WITH SMOTE")
    print("="*60)
    
    # Logistic Regression with SMOTE
    print("\n[1/3] LR with SMOTE...")
    lr_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    lr_param_grid = {
        'classifier__C': [0.01, 0.1, 1],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['lbfgs', 'liblinear'],
        'smote__k_neighbors': [3, 5]
    }
    lr_grid = GridSearchCV(lr_pipeline, lr_param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
    lr_grid.fit(X_train, y_train)
    
    # Random Forest with SMOTE
    print("\n[2/3] RF with SMOTE...")
    rf_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    rf_param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [5, 10],
        'smote__k_neighbors': [3, 5]
    }
    rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
    rf_grid.fit(X_train, y_train)
    
    # Gradient Boosting with SMOTE
    print("\n[3/3] GB with SMOTE...")
    gb_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ])
    gb_param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.05, 0.1],
        'smote__k_neighbors': [3, 5]
    }
    gb_grid = GridSearchCV(gb_pipeline, gb_param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
    gb_grid.fit(X_train, y_train)
    
    print(f"\n✓ Best LR+SMOTE ROC-AUC: {lr_grid.best_score_:.4f}")
    print(f"✓ Best RF+SMOTE ROC-AUC: {rf_grid.best_score_:.4f}")
    print(f"✓ Best GB+SMOTE ROC-AUC: {gb_grid.best_score_:.4f}")
    
    return lr_grid.best_estimator_, rf_grid.best_estimator_, gb_grid.best_estimator_

# ============================================
# EVALUATION
# ============================================

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n{'='*50}")
    print(f"{model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {cm[0,0]:4d}")
    print(f"  False Positives: {cm[0,1]:4d}")
    print(f"  False Negatives: {cm[1,0]:4d}")
    print(f"  True Positives:  {cm[1,1]:4d}")
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 
            'f1': f1, 'roc_auc': roc_auc}

def plot_comparison(models, X_test, y_test):
    """Plot ROC curves"""
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})')
        except Exception as e:
            print(f"Could not plot {name}: {e}")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Loan Approval Prediction')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300)
    plt.show()

def plot_confusion_matrices(models, X_test, y_test):
    """Plot confusion matrices for all models"""
    n_models = len(models)
    fig, axes = plt.subplots(1, min(n_models, 3), figsize=(15, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, model) in enumerate(list(models.items())[:3]):  # Plot top 3
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

# ============================================
# MAIN
# ============================================

def main():
    print("="*60)
    print("LOAN APPROVAL PREDICTION (No XGBoost)")
    print("="*60)
    
    # Generate data
    print("\n[1] Generating data...")
    data = generate_loan_data(n_samples=5000)
    print(f"Dataset: {data.shape}, Approval rate: {data['LoanStatus'].mean()*100:.1f}%")
    
    # Split
    print("\n[2] Splitting data...")
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['LoanStatus'])
    print(f"Training: {len(train_data)}, Test: {len(test_data)}")
    
    # Preprocess training data
    print("\n[3] Preprocessing training data...")
    X_train, y_train, scaler, encoders = preprocess_data(train_data, is_training=True)
    print(f"Training features: {X_train.shape[1]}")
    
    # Preprocess test data
    print("\n[4] Preprocessing test data...")
    X_test, y_test, _, _ = preprocess_data(test_data, is_training=False, scaler=scaler, encoders=encoders)
    print(f"Test features: {X_test.shape[1]}")
    
    # Train models
    print("\n[5] Training models...")
    lr_model, rf_model, gb_model = train_models(X_train, y_train)
    
    # Train with SMOTE
    print("\n[6] Training with SMOTE...")
    lr_smote, rf_smote, gb_smote = train_with_smote(X_train, y_train)
    
    # Evaluate
    print("\n[7] Evaluating models...")
    models = {
        'Logistic Regression': lr_model,
        'Random Forest': rf_model,
        'Gradient Boosting': gb_model,
        'LR + SMOTE': lr_smote,
        'RF + SMOTE': rf_smote,
        'GB + SMOTE': gb_smote
    }
    
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_test, y_test, name)
    
    # Plot
    print("\n[8] Generating plots...")
    try:
        plot_comparison(models, X_test, y_test)
        plot_confusion_matrices(models, X_test, y_test)
    except Exception as e:
        print(f"Plot generation warning: {e}")
    
    # Save models
    print("\n[9] Saving models...")
    joblib.dump(lr_model, 'logistic_regression_model.pkl')
    joblib.dump(rf_model, 'random_forest_model.pkl')
    joblib.dump(gb_model, 'gradient_boosting_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(encoders, 'encoders.pkl')
    print("✓ Models saved successfully!")
    
    # Summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    results_df = pd.DataFrame(results).T
    print(results_df.round(4))
    
    best = results_df['roc_auc'].idxmax()
    print(f"\n✓ Best Model: {best} (ROC-AUC: {results_df.loc[best, 'roc_auc']:.4f})")
    
    # Sample predictions
    print("\n[10] Sample Predictions...")
    sample_indices = np.random.choice(len(X_test), 3, replace=False)
    for idx in sample_indices:
        features = X_test.iloc[idx:idx+1]
        true_label = y_test.iloc[idx]
        
        lr_prob = lr_model.predict_proba(features)[0, 1]
        rf_prob = rf_model.predict_proba(features)[0, 1]
        gb_prob = gb_model.predict_proba(features)[0, 1]
        
        print(f"\n  True Label: {'Approved' if true_label == 1 else 'Rejected'}")
        print(f"  LR Probability: {lr_prob:.2%}")
        print(f"  RF Probability: {rf_prob:.2%}")
        print(f"  GB Probability: {gb_prob:.2%}")
    
    print("\n" + "="*60)
    print("✓ PROJECT COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()