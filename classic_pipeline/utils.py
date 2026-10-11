from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def calculate_metrics(probabilities, targets, threshold=0.5):
    probabilities = np.asarray(probabilities)
    targets = np.asarray(targets).astype(int)

    predictions = (probabilities >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision_score(targets, predictions, zero_division=0),
        "recall": recall_score(targets, predictions, zero_division=0),
        "f1": f1_score(targets, predictions, zero_division=0),
        "roc_auc": roc_auc_score(targets, probabilities),
    }


def find_best_threshold(probabilities, targets, metric="accuracy"):
    best_threshold = 0.5
    best_score = -np.inf

    for threshold in np.arange(0.30, 0.71, 0.01):
        metrics = calculate_metrics(
            probabilities=probabilities,
            targets=targets,
            threshold=threshold,
        )

        score = metrics[metric]

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def save_fold_pipeline(pipeline, config, model_name, fold):
    experiment_dir = (
        Path(config.paths.checkpoints)
        / config.general.experiment_name
        / model_name
    )

    experiment_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = experiment_dir / f"fold_{fold}.joblib"

    joblib.dump(pipeline, pipeline_path)

    return pipeline_path


def create_submission(test_ids, test_probabilities, config, model_name, threshold):
    submission_dir = Path(config.paths.submissions)
    submission_dir.mkdir(parents=True, exist_ok=True)

    predictions = (np.asarray(test_probabilities) >= threshold).astype(int)

    submission = pd.DataFrame({
        config.data.id_column: test_ids.to_numpy(),
        config.data.target: predictions,
    })

    submission_path = (
        submission_dir
        / f"{config.general.experiment_name}_{model_name}.csv"
    )

    submission.to_csv(submission_path, index=False)

    print(f"Submission saved to: {submission_path}")

    return submission_path


def log_model(model_name, model, config):
    print(f"Experiment: {config.general.experiment_name}")
    print(f"Model: {model_name}")
    print(f"Estimator: {model.__class__.__name__}")
    print(f"CV folds: {config.data.n_splits}")
    print(f"Primary metric: {config.training.metric.upper()}")


def save_model_metadata(config, model_name, threshold, oof_metrics):
    model_dir = (
        Path(config.paths.checkpoints)
        / config.general.experiment_name
        / model_name
    )

    model_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "threshold": float(threshold),
        "oof_metrics": {
            key: float(value)
            for key, value in oof_metrics.items()
        },
    }

    metadata_path = model_dir / "metadata.json"

    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=4)

    return metadata_path