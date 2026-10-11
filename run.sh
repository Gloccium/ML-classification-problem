#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="titanic-ml"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda is not installed or is not available in PATH."
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Environment '$ENV_NAME' does not exist. Running setup.sh..."
  bash setup.sh
fi

if [[ ! -f "data/train.csv" || ! -f "data/test.csv" ]]; then
  echo "Error: data/train.csv and data/test.csv are required."
  echo "Download the Titanic competition data first, for example:"
  echo "  kaggle competitions download -c titanic -p data"
  echo "  unzip data/titanic.zip -d data"
  echo "  rm data/titanic.zip"
  exit 1
fi

echo "Training best public-leaderboard model: MLP baseline..."
conda run -n "$ENV_NAME" python -m deep_learning_pipeline.main

echo
echo "Running inference..."
conda run -n "$ENV_NAME" python -m deep_learning_pipeline.inference --experiment mlp_baseline

echo
echo "Done. Submission: submissions/mlp_baseline.csv"
