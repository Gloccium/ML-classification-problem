import argparse
from pathlib import Path

import joblib
import numpy as np
import torch

from omegaconf import OmegaConf

from deep_learning_pipeline.data import load_test_data, create_inference_loader
from deep_learning_pipeline.model import MLPClassifier
from deep_learning_pipeline.predict import predict
from deep_learning_pipeline.utils import get_device, create_submission


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_inference(experiment_name):
    experiment_dir = PROJECT_ROOT / "checkpoints" / experiment_name
    config = OmegaConf.load(experiment_dir / "config.yaml")

    config.paths.test_data = str(PROJECT_ROOT / "data" / "test.csv")
    config.paths.submissions = str(PROJECT_ROOT / "submissions")

    device = get_device(config)

    X_test, test_ids = load_test_data(config)

    fold_probabilities = []

    for fold in range(1, config.data.n_splits + 1):
        print(f"Inference fold {fold}/{config.data.n_splits}")

        fold_dir = experiment_dir / f"fold_{fold}"

        checkpoint = torch.load(fold_dir / "model.pt", map_location="cpu")
        preprocessor = joblib.load(fold_dir / "preprocessor.joblib")

        model = MLPClassifier(
            input_size=checkpoint["input_size"],
            hidden_dims=checkpoint["hidden_dims"],
            dropout=checkpoint["dropout"],
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        test_loader = create_inference_loader(
            X=X_test,
            preprocessor=preprocessor,
            batch_size=config.training.batch_size,
        )

        probabilities = predict(
            model=model,
            data_loader=test_loader,
            device=device,
        )

        fold_probabilities.append(probabilities)

    fold_probabilities = np.stack(fold_probabilities)

    ensemble_probabilities = np.mean(fold_probabilities, axis=0)

    submission_path = create_submission(
        test_ids=test_ids,
        test_probabilities=ensemble_probabilities,
        config=config,
    )

    return submission_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)

    args = parser.parse_args()

    run_inference(args.experiment)