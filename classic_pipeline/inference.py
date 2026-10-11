import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from omegaconf import OmegaConf

from classic_pipeline.config import config
from classic_pipeline.data import load_test_data
from classic_pipeline.utils import create_submission


def run_inference(model_name, experiment_name, config):
    run_config = OmegaConf.create(
        OmegaConf.to_container(config, resolve=True)
    )

    run_config.general.experiment_name = experiment_name

    model_dir = (
        Path(run_config.paths.checkpoints)
        / experiment_name
        / model_name
    )

    metadata_path = model_dir / "metadata.json"

    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    threshold = metadata["threshold"]

    X_test, test_ids = load_test_data(run_config)

    fold_probabilities = []

    for fold in range(1, run_config.data.n_splits + 1):
        print(f"Inference fold {fold}/{run_config.data.n_splits}")

        pipeline_path = model_dir / f"fold_{fold}.joblib"
        estimator = joblib.load(pipeline_path)

        probabilities = estimator.predict_proba(X_test)[:, 1]

        fold_probabilities.append(probabilities)

    fold_probabilities = np.stack(fold_probabilities)

    ensemble_probabilities = np.mean(
        fold_probabilities,
        axis=0,
    )

    print(f"Experiment: {experiment_name}")
    print(f"Model: {model_name}")
    print(f"Threshold: {threshold:.2f}")

    submission_path = create_submission(
        test_ids=test_ids,
        test_probabilities=ensemble_probabilities,
        config=run_config,
        model_name=model_name,
        threshold=threshold,
    )

    return submission_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "logistic_regression",
            "random_forest",
            "xgboost",
            "catboost",
        ],
    )

    parser.add_argument(
        "--experiment",
        default="classic_baselines",
    )

    args = parser.parse_args()

    run_inference(
        model_name=args.model,
        experiment_name=args.experiment,
        config=config,
    )