#!/bin/bash

echo "=========================================="
echo "LOAN APPROVAL PREDICTION PIPELINE"
echo "=========================================="

# Create virtual environment (optional)
# python -m venv venv
# source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the main script
echo "Running loan approval prediction..."
python loan_approval_prediction.py

# Test the API
echo "Testing prediction API..."
python prediction_api.py

echo "=========================================="
echo "COMPLETE! Check the generated plots and models."
echo "=========================================="