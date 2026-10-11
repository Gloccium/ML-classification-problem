import os
import random
import joblib
from pathlib import Path
from omegaconf import OmegaConf

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(config):
    if config.training.device != "auto":
        return torch.device(config.training.device)

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def get_loss(config):
    loss_name = config.training.loss.lower()

    if loss_name == "bce":
        return torch.nn.BCEWithLogitsLoss()

    raise ValueError(f"Unknown loss: {loss_name}")


def calculate_metrics(probabilities, targets, threshold=0.5):
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.detach().cpu().numpy()

    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    predictions = (probabilities >= threshold).astype(int)
    targets = targets.astype(int)

    return {
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision_score(targets, predictions, zero_division=0),
        "recall": recall_score(targets, predictions, zero_division=0),
        "f1": f1_score(targets, predictions, zero_division=0),
        "roc_auc": roc_auc_score(targets, probabilities),
    }


def get_optimizer(model, config):
    optimizer_name = config.training.optimizer.lower()

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )

    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(optimizer, config):
    scheduler_name = config.training.scheduler.lower()

    if scheduler_name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.training.scheduler_factor,
            patience=config.training.scheduler_patience,
        )

    if scheduler_name == "none":
        return None

    raise ValueError(f"Unknown scheduler: {scheduler_name}")


def save_fold_artifacts(model, preprocessor, input_size, config, fold):
    experiment_dir = Path(config.paths.checkpoints) / config.general.experiment_name
    fold_dir = experiment_dir / f"fold_{fold}"

    fold_dir.mkdir(parents=True, exist_ok=True)

    model_state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }

    torch.save({
        "model_state_dict": model_state_dict,
        "input_size": input_size,
        "hidden_dims": list(config.model.hidden_dims),
        "dropout": float(config.model.dropout),
        "threshold": float(config.training.threshold),
    }, fold_dir / "model.pt")

    joblib.dump(preprocessor, fold_dir / "preprocessor.joblib")

    OmegaConf.save(
        config=config,
        f=experiment_dir / "config.yaml",
    )


def log_experiment(config, device, input_size, loss_func, optimizer, scheduler):
    print(f"Experiment: {config.general.experiment_name}")
    print(f"Device: {device}")
    print(f"Input size: {input_size}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Loss: {loss_func.__class__.__name__}")
    print(f"Primary metric: {config.training.metric.upper()}")
    print(f"Threshold: {config.training.threshold}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Scheduler: {scheduler.__class__.__name__}")


def create_submission(test_ids, test_probabilities, config):
    submission_dir = Path(config.paths.submissions)
    submission_dir.mkdir(parents=True, exist_ok=True)

    test_predictions = (test_probabilities >= config.training.threshold).astype(int)

    submission = pd.DataFrame({
        config.data.id_column: test_ids.to_numpy(),
        config.data.target: test_predictions,
    })

    submission_path = submission_dir / f"{config.general.experiment_name}.csv"
    submission.to_csv(submission_path, index=False)

    print(f"Submission saved to: {submission_path}")

    return submission_path