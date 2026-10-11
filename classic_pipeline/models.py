import pandas as pd

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# Just a wrapper to use catboost as a sklearn model
class NativeCatBoostClassifier:
    def __init__(self, seed, params=None):
        self.seed = seed
        self.params = params or {}

    def fit(self, X, y):
        self.numerical_columns = X.select_dtypes(include=["number"]).columns.tolist()
        self.categorical_columns = X.select_dtypes(exclude=["number"]).columns.tolist()

        self.numerical_medians = X[self.numerical_columns].median()

        self.categorical_fill_values = {}

        for column in self.categorical_columns:
            mode = X[column].mode(dropna=True)
            self.categorical_fill_values[column] = mode.iloc[0] if len(mode) else "Unknown"

        X = self._transform(X)

        self.model = CatBoostClassifier(
            loss_function="Logloss",
            random_seed=self.seed,
            verbose=False,
            allow_writing_files=False,
            **self.params,
        )

        self.model.fit(
            X,
            y,
            cat_features=self.categorical_columns,
        )

        return self

    def _transform(self, X):
        X = X.copy()

        X[self.numerical_columns] = X[self.numerical_columns].fillna(self.numerical_medians)

        for column in self.categorical_columns:
            X[column] = (
                X[column]
                .fillna(self.categorical_fill_values[column])
                .astype(str)
            )

        return X

    def predict_proba(self, X):
        X = self._transform(X)

        return self.model.predict_proba(X)


def get_model(model_name, config, params=None):
    seed = config.general.seed
    params = params or {}

    if model_name == "logistic_regression":
        return LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=seed,
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        )

    if model_name == "xgboost":
        default_params = {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }

        default_params.update(params)

        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
            **default_params,
        )

    if model_name == "catboost":
        default_params = {
            "iterations": 500,
            "learning_rate": 0.03,
            "depth": 5,
        }

        default_params.update(params)

        return NativeCatBoostClassifier(
            seed=seed,
            params=default_params,
        )

    raise ValueError(f"Unknown model: {model_name}")