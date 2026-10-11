from sklearn.pipeline import Pipeline

from classic_pipeline.data import create_preprocessor
from classic_pipeline.models import get_model
from classic_pipeline.utils import calculate_metrics, save_fold_pipeline


def train_fold(X, y, train_idx, val_idx, model_name, fold, config, params=None):
    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    model = get_model(model_name, config, params=params)

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

    metrics = calculate_metrics(
        probabilities=val_probabilities,
        targets=y_val,
        threshold=config.training.threshold,
    )

    save_fold_pipeline(
        pipeline=estimator,
        config=config,
        model_name=model_name,
        fold=fold,
    )

    return val_probabilities, metrics