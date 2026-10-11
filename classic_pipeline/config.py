from pathlib import Path

from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]


config = OmegaConf.create({
    "general": {
        "experiment_name": "classic_baselines",
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

    "training": {
        "metric": "accuracy",
        "threshold": 0.5,
    },
})