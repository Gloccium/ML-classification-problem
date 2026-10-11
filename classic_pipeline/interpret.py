import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import StratifiedKFold

from classic_pipeline.config import config
from classic_pipeline.data import load_train_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_fold_indices(X, y, config):
    skf = StratifiedKFold(
        n_splits=config.data.n_splits,
        shuffle=True,
        random_state=config.general.seed,
    )

    return list(skf.split(X, y))


def calculate_feature_importance(X, y, experiment_name, model_name, config):
    model_dir = (
        Path(config.paths.checkpoints)
        / experiment_name
        / model_name
    )

    fold_indices = get_fold_indices(X, y, config)

    fold_importances = []

    for fold, (_, val_idx) in enumerate(fold_indices, start=1):
        pipeline_path = model_dir / f"fold_{fold}.joblib"
        pipeline = joblib.load(pipeline_path)

        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]

        feature_names = preprocessor.get_feature_names_out()

        importance = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_,
        })

        fold_importances.append(importance)

    importance_df = pd.concat(fold_importances)

    importance_df = (
        importance_df
        .groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
    )

    return importance_df


def calculate_shap_values(X, y, experiment_name, model_name, config):
    model_dir = (
        Path(config.paths.checkpoints)
        / experiment_name
        / model_name
    )

    fold_indices = get_fold_indices(X, y, config)

    all_shap_values = []
    all_X = []
    all_feature_names = []

    for fold, (_, val_idx) in enumerate(fold_indices, start=1):
        print(f"SHAP fold {fold}/{config.data.n_splits}")

        pipeline_path = model_dir / f"fold_{fold}.joblib"
        pipeline = joblib.load(pipeline_path)

        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]

        X_val = X.iloc[val_idx]

        X_val_transformed = preprocessor.transform(X_val)
        feature_names = preprocessor.get_feature_names_out()

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val_transformed)

        shap_df = pd.DataFrame(
            shap_values,
            columns=feature_names,
        )

        X_df = pd.DataFrame(
            X_val_transformed,
            columns=feature_names,
        )

        all_shap_values.append(shap_df)
        all_X.append(X_df)
        all_feature_names.extend(feature_names)

    common_features = sorted(set.intersection(
        *[set(df.columns) for df in all_shap_values]
    ))

    shap_values = pd.concat(
        [df[common_features] for df in all_shap_values],
        ignore_index=True,
    )

    X_transformed = pd.concat(
        [df[common_features] for df in all_X],
        ignore_index=True,
    )

    return shap_values, X_transformed


def save_feature_importance_plot(importance_df, output_dir, top_n=20):
    top_features = importance_df.head(top_n).sort_values("importance")

    plt.figure(figsize=(10, 7))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.xlabel("Importance")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()

    output_path = output_dir / "feature_importance.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def save_shap_plots(shap_values, X_transformed, output_dir, top_n=20):
    shap.summary_plot(
        shap_values.to_numpy(),
        X_transformed,
        feature_names=shap_values.columns,
        plot_type="bar",
        max_display=top_n,
        show=False,
    )

    plt.tight_layout()
    bar_path = output_dir / "shap_importance.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()

    shap.summary_plot(
        shap_values.to_numpy(),
        X_transformed,
        feature_names=shap_values.columns,
        max_display=top_n,
        show=False,
    )

    plt.tight_layout()
    beeswarm_path = output_dir / "shap_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()

    return bar_path, beeswarm_path


def interpret(experiment_name, model_name, config):
    if model_name != "xgboost":
        raise ValueError("This interpretation script currently supports XGBoost only.")

    X, y = load_train_data(config)

    output_dir = (
        PROJECT_ROOT
        / "artifacts"
        / "interpretability"
        / experiment_name
        / model_name
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    importance_df = calculate_feature_importance(
        X=X,
        y=y,
        experiment_name=experiment_name,
        model_name=model_name,
        config=config,
    )

    importance_path = output_dir / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    save_feature_importance_plot(
        importance_df=importance_df,
        output_dir=output_dir,
    )

    shap_values, X_transformed = calculate_shap_values(
        X=X,
        y=y,
        experiment_name=experiment_name,
        model_name=model_name,
        config=config,
    )

    shap_importance = pd.DataFrame({
        "feature": shap_values.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    shap_importance.to_csv(
        output_dir / "shap_importance.csv",
        index=False,
    )

    save_shap_plots(
        shap_values=shap_values,
        X_transformed=X_transformed,
        output_dir=output_dir,
    )

    print("\n=== TOP FEATURE IMPORTANCE ===")
    print(importance_df.head(15).to_string(index=False))

    print("\n=== TOP SHAP FEATURES ===")
    print(shap_importance.head(15).to_string(index=False))

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="xgboost",
        choices=["xgboost"],
    )

    parser.add_argument(
        "--experiment",
        default="classic_tuned",
    )

    args = parser.parse_args()

    interpret(
        experiment_name=args.experiment,
        model_name=args.model,
        config=config,
    )