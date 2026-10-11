import argparse
import json
from pathlib import Path

import numpy as np

from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedKFold

from classic_pipeline.config import config
from classic_pipeline.data import load_train_data
from classic_pipeline.models import get_model
from classic_pipeline.train import train_fold
from classic_pipeline.utils import (
    calculate_metrics,
    log_model,
    save_model_metadata,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def fit(model_name, config, params=None):
    X, y = load_train_data(config)

    skf = StratifiedKFold(
        n_splits=config.data.n_splits,
        shuffle=True,
        random_state=config.general.seed,
    )

    oof_probabilities = np.zeros(len(X), dtype=np.float64)
    fold_metrics = []

    model = get_model(model_name, config, params=params)
    log_model(model_name, model, config)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\nFold {fold}/{config.data.n_splits}")

        val_probabilities, metrics = train_fold(
            X=X,
            y=y,
            train_idx=train_idx,
            val_idx=val_idx,
            model_name=model_name,
            fold=fold,
            config=config,
            params=params,
        )

        oof_probabilities[val_idx] = val_probabilities
        fold_metrics.append(metrics)

        print(
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"ROC-AUC: {metrics['roc_auc']:.4f}"
        )

    oof_metrics = calculate_metrics(
        probabilities=oof_probabilities,
        targets=y,
        threshold=config.training.threshold,
    )

    print("\n=== OOF RESULTS ===")
    print(f"Threshold: {config.training.threshold:.2f}")
    print(f"Accuracy:  {oof_metrics['accuracy']:.4f}")
    print(f"Precision: {oof_metrics['precision']:.4f}")
    print(f"Recall:    {oof_metrics['recall']:.4f}")
    print(f"F1:        {oof_metrics['f1']:.4f}")
    print(f"ROC-AUC:   {oof_metrics['roc_auc']:.4f}")

    save_model_metadata(
        config=config,
        model_name=model_name,
        threshold=config.training.threshold,
        oof_metrics=oof_metrics,
    )

    return {
        "oof_probabilities": oof_probabilities,
        "oof_metrics": oof_metrics,
        "fold_metrics": fold_metrics,
    }


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
        "--tuned",
        action="store_true",
    )

    args = parser.parse_args()

    run_config = OmegaConf.create(
        OmegaConf.to_container(config, resolve=True)
    )

    params = None

    if args.tuned:
        params_path = (
            PROJECT_ROOT
            / "checkpoints"
            / "optuna"
            / args.model
            / "best_params.json"
        )

        with open(params_path, "r") as file:
            params = json.load(file)

        run_config.general.experiment_name = "classic_tuned"

        print(f"Using tuned parameters from: {params_path}")

    results = fit(
        model_name=args.model,
        config=run_config,
        params=params,
    )