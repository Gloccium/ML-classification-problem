from pathlib import Path
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]

config = OmegaConf.create({
    "general": {
            "experiment_name": "mlp_baseline",
            "seed": 42,
    },

    "paths": {
        "train_data": str(PROJECT_ROOT / "data" / "train.csv"),
        "test_data": str(PROJECT_ROOT / "data" / "test.csv"),
        "checkpoints": str(PROJECT_ROOT / "checkpoints"),
        "submissions": str(PROJECT_ROOT / "submissions"),
    },

    "data": {
        "target": "Survived",
        "id_column": "PassengerId",
        "n_splits": 5,
    },

    "model": {
        "hidden_dims": [64, 32],
        "dropout": 0.2
    },

    "training": {
        "epochs": 100,
        "batch_size": 32,

        "loss": "bce",
        "metric": "accuracy",
        "threshold": 0.6,

        "optimizer": "adamw",
        "lr": 1e-3,
        "weight_decay": 1e-4,

        "scheduler": "reduce_on_plateau",
        "scheduler_factor": 0.5,
        "scheduler_patience": 5,

        "early_stopping_patience": 15,
        "device": "auto"
    },
})