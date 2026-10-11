import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

from classic_pipeline.config import config as classic_config
from classic_pipeline.data import (
    load_train_data as load_classic_train,
    load_test_data as load_classic_test,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLEND_DIR = PROJECT_ROOT / "artifacts" / "blend"


def generate_mlp_predictions():
    import joblib
    import torch

    from omegaconf import OmegaConf
    from sklearn.model_selection import StratifiedKFold

    from deep_learning_pipeline.data import create_inference_loader
    from deep_learning_pipeline.model import MLPClassifier
    from deep_learning_pipeline.predict import predict

    experiment_dir = PROJECT_ROOT / "checkpoints" / "mlp_baseline"
    config = OmegaConf.load(experiment_dir / "config.yaml")

    X_train, y = load_classic_train(classic_config)
    X_test, _ = load_classic_test(classic_config)

    skf = StratifiedKFold(
        n_splits=config.data.n_splits,
        shuffle=True,
        random_state=config.general.seed,
    )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    oof_probabilities = np.zeros(len(X_train), dtype=np.float64)
    test_fold_probabilities = []

    for fold, (_, val_idx) in enumerate(skf.split(X_train, y), start=1):
        print(f"MLP fold {fold}/{config.data.n_splits}")

        fold_dir = experiment_dir / f"fold_{fold}"

        checkpoint = torch.load(
            fold_dir / "model.pt",
            map_location="cpu",
        )

        preprocessor = joblib.load(
            fold_dir / "preprocessor.joblib",
        )

        model = MLPClassifier(
            input_size=checkpoint["input_size"],
            hidden_dims=checkpoint["hidden_dims"],
            dropout=checkpoint["dropout"],
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        val_loader = create_inference_loader(
            X=X_train.iloc[val_idx],
            preprocessor=preprocessor,
            batch_size=config.training.batch_size,
        )

        test_loader = create_inference_loader(
            X=X_test,
            preprocessor=preprocessor,
            batch_size=config.training.batch_size,
        )

        oof_probabilities[val_idx] = predict(
            model=model,
            data_loader=val_loader,
            device=device,
        )

        test_fold_probabilities.append(
            predict(
                model=model,
                data_loader=test_loader,
                device=device,
            )
        )

        model.to("cpu")

    test_probabilities = np.mean(
        np.stack(test_fold_probabilities),
        axis=0,
    )

    BLEND_DIR.mkdir(parents=True, exist_ok=True)

    np.savez(
        BLEND_DIR / "mlp_predictions.npz",
        oof=oof_probabilities,
        test=test_probabilities,
    )


def generate_xgboost_predictions():
    import joblib

    from sklearn.model_selection import StratifiedKFold

    X_train, y = load_classic_train(classic_config)
    X_test, _ = load_classic_test(classic_config)

    model_dir = (
        PROJECT_ROOT
        / "checkpoints"
        / "classic_tuned"
        / "xgboost"
    )

    skf = StratifiedKFold(
        n_splits=classic_config.data.n_splits,
        shuffle=True,
        random_state=classic_config.general.seed,
    )

    oof_probabilities = np.zeros(len(X_train), dtype=np.float64)
    test_fold_probabilities = []

    for fold, (_, val_idx) in enumerate(skf.split(X_train, y), start=1):
        print(f"XGBoost fold {fold}/{classic_config.data.n_splits}")

        pipeline = joblib.load(
            model_dir / f"fold_{fold}.joblib"
        )

        oof_probabilities[val_idx] = (
            pipeline.predict_proba(X_train.iloc[val_idx])[:, 1]
        )

        test_fold_probabilities.append(
            pipeline.predict_proba(X_test)[:, 1]
        )

    test_probabilities = np.mean(
        np.stack(test_fold_probabilities),
        axis=0,
    )

    BLEND_DIR.mkdir(parents=True, exist_ok=True)

    np.savez(
        BLEND_DIR / "xgboost_predictions.npz",
        oof=oof_probabilities,
        test=test_probabilities,
    )


def find_best_blend(mlp_oof, xgb_oof, y):
    best_accuracy = -1
    best_weight = None
    best_threshold = None

    for mlp_weight in np.arange(0.0, 1.01, 0.05):
        xgb_weight = 1.0 - mlp_weight

        probabilities = (
            mlp_weight * mlp_oof
            + xgb_weight * xgb_oof
        )

        for threshold in np.arange(0.40, 0.66, 0.01):
            predictions = (
                probabilities >= threshold
            ).astype(int)

            accuracy = accuracy_score(
                y,
                predictions,
            )

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weight = mlp_weight
                best_threshold = threshold

    return best_weight, best_threshold, best_accuracy


def create_blend():
    BLEND_DIR.mkdir(parents=True, exist_ok=True)

    print("\nGenerating MLP predictions in separate process...")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "classic_pipeline.blend",
            "--generate-mlp",
        ],
        check=True,
    )

    print("\nGenerating XGBoost predictions in separate process...")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "classic_pipeline.blend",
            "--generate-xgboost",
        ],
        check=True,
    )

    mlp = np.load(BLEND_DIR / "mlp_predictions.npz")
    xgb = np.load(BLEND_DIR / "xgboost_predictions.npz")

    mlp_oof = mlp["oof"]
    mlp_test = mlp["test"]

    xgb_oof = xgb["oof"]
    xgb_test = xgb["test"]

    _, y = load_classic_train(classic_config)
    _, test_ids = load_classic_test(classic_config)

    mlp_weight, threshold, accuracy = find_best_blend(
        mlp_oof=mlp_oof,
        xgb_oof=xgb_oof,
        y=y,
    )

    xgb_weight = 1.0 - mlp_weight

    print("\nBEST BLEND")
    print(f"MLP weight:     {mlp_weight:.2f}")
    print(f"XGBoost weight: {xgb_weight:.2f}")
    print(f"Threshold:      {threshold:.2f}")
    print(f"OOF Accuracy:   {accuracy:.4f}")

    test_probabilities = (
        mlp_weight * mlp_test
        + xgb_weight * xgb_test
    )

    test_predictions = (
        test_probabilities >= threshold
    ).astype(int)

    submission = pd.DataFrame({
        classic_config.data.id_column: test_ids.to_numpy(),
        classic_config.data.target: test_predictions,
    })

    submission_dir = PROJECT_ROOT / "submissions"
    submission_dir.mkdir(parents=True, exist_ok=True)

    submission_path = (
        submission_dir
        / "blend_mlp_xgboost.csv"
    )

    submission.to_csv(
        submission_path,
        index=False,
    )

    print(f"\nSubmission saved to: {submission_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--generate-mlp",
        action="store_true",
    )

    parser.add_argument(
        "--generate-xgboost",
        action="store_true",
    )

    args = parser.parse_args()

    if args.generate_mlp:
        generate_mlp_predictions()

    elif args.generate_xgboost:
        generate_xgboost_predictions()

    else:
        create_blend()