import argparse
import json
from pathlib import Path

import numpy as np
import optuna

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from classic_pipeline.config import config
from classic_pipeline.data import load_train_data, create_preprocessor
from classic_pipeline.models import get_model
from classic_pipeline.utils import calculate_metrics


def get_trial_params(trial, model_name):
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

    if model_name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 200, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 3, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        }

    raise ValueError(f"Unsupported model for tuning: {model_name}")


def objective(trial, model_name, X, y, config):
    params = get_trial_params(trial, model_name)

    skf = StratifiedKFold(
        n_splits=config.data.n_splits,
        shuffle=True,
        random_state=config.general.seed,
    )

    oof_probabilities = np.zeros(len(X), dtype=np.float64)

    for train_idx, val_idx in skf.split(X, y):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]

        model = get_model(
            model_name=model_name,
            config=config,
            params=params,
        )

        if model_name == "catboost":
            estimator = model
        else:
            preprocessor = create_preprocessor(X_train)

            estimator = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model),
            ])

        estimator.fit(X_train, y_train)

        val_probabilities = estimator.predict_proba(X_val)[:, 1]
        oof_probabilities[val_idx] = val_probabilities

    metrics = calculate_metrics(
        probabilities=oof_probabilities,
        targets=y,
        threshold=config.training.threshold,
    )

    trial.set_user_attr("f1", metrics["f1"])
    trial.set_user_attr("roc_auc", metrics["roc_auc"])

    return metrics["accuracy"]


def tune(model_name, config, n_trials):
    X, y = load_train_data(config)

    study = optuna.create_study(
        direction="maximize",
        study_name=f"{model_name}_tuning",
    )

    study.optimize(
        lambda trial: objective(
            trial=trial,
            model_name=model_name,
            X=X,
            y=y,
            config=config,
        ),
        n_trials=n_trials,
    )

    print("\nOPTUNA RESULTS")
    print(f"Model: {model_name}")
    print(f"Best accuracy: {study.best_value:.4f}")
    print(f"Best trial: {study.best_trial.number}")

    print("\nBest parameters:")

    for key, value in study.best_params.items():
        print(f"{key}: {value}")

    print(f"\nF1: {study.best_trial.user_attrs['f1']:.4f}")
    print(f"ROC-AUC: {study.best_trial.user_attrs['roc_auc']:.4f}")

    output_dir = (
        Path(config.paths.checkpoints)
        / "optuna"
        / model_name
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "best_params.json"

    with open(output_path, "w") as file:
        json.dump(study.best_params, file, indent=4)

    print(f"\nBest parameters saved to: {output_path}")

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "xgboost",
            "catboost",
        ],
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    tune(
        model_name=args.model,
        config=config,
        n_trials=args.trials,
    )