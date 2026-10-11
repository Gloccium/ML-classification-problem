#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="titanic-ml"
ENV_FILE="environment.yml"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda is not installed or is not available in PATH."
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: $ENV_FILE not found. Run this script from the project root."
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Environment '$ENV_NAME' already exists. Updating it..."
  conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
  echo "Creating environment '$ENV_NAME'..."
  conda env create -f "$ENV_FILE"
fi

echo "Running import smoke test..."
conda run -n "$ENV_NAME" python - <<'PY'
import numpy
import pandas
import sklearn
import torch
import xgboost
import catboost
import optuna
import shap
import omegaconf
import joblib

print("Environment smoke test passed.")
print(f"Python/Torch ready; MPS available: {torch.backends.mps.is_available()}")
PY

echo
echo "Setup complete."
echo "Activate manually with: conda activate $ENV_NAME"
