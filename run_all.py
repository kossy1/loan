"""
One-click runner for the entire loan approval prediction project
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and print output"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {command}")
    print(f"{'='*60}")
    result = subprocess.run(command, shell=True, capture_output=False, text=True)
    return result.returncode

def main():
    print("="*60)
    print("LOAN APPROVAL PREDICTION - COMPLETE PIPELINE")
    print("="*60)
    
    # Step 1: Train models
    print("\n[STEP 1] Training and saving models...")
    if run_command("python loan_approval_prediction.py") != 0:
        print("❌ Training failed!")
        return
    
    # Step 2: Make predictions
    print("\n[STEP 2] Making predictions with saved models...")
    if run_command("python prediction_api.py") != 0:
        print("❌ Prediction failed!")
        return
    
    print("\n" + "="*60)
    print("✓ COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated files:")
    print("  - logistic_regression_model.pkl (Logistic Regression model)")
    print("  - xgboost_model.pkl (XGBoost model)")
    print("  - scaler.pkl (StandardScaler for preprocessing)")
    print("  - encoders.pkl (LabelEncoders for categorical variables)")
    print("  - roc_curves.png (ROC curves visualization)")
    print("  - confusion_matrices.png (Confusion matrices)")
    print("  - feature_importance.png (Feature importance plots)")

if __name__ == "__main__":
    main()