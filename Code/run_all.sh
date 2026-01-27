#!/bin/bash

# Abort on error
set -e

echo "========================================================"
echo " Starting HIRA AI Analysis Pipeline"
echo "========================================================"

# Files 02, 03, 04, 05 are modules imported by 06.
# Running them individually only executes their internal unit tests.
# We will skip them to focus on the actual analysis execution.

echo ""
echo "[Step 1/2] Running Analysis (TDA Pipeline)..."
echo "Executing: python 06_run_analysis.py"
echo "--------------------------------------------------------"
python 06_run_analysis.py

echo ""
echo "[Step 2/2] Running AI Modeling (LSTM & Escape Recommender)..."
echo "Executing: python 07_ai_modeling.py"
echo "--------------------------------------------------------"
python 07_ai_modeling.py

echo ""
echo "========================================================"
echo " Pipeline Finished Successfully."
echo " Check 'result/' directory for outputs."
echo "========================================================"
