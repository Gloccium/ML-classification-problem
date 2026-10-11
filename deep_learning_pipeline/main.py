import numpy as np

from sklearn.model_selection import StratifiedKFold

from deep_learning_pipeline.config import config
from deep_learning_pipeline.data import load_train_data, create_fold_dataloaders
from deep_learning_pipeline.model import MLPClassifier
from deep_learning_pipeline.predict import predict
from deep_learning_pipeline.train import train, validate
from deep_learning_pipeline.utils import (
    set_seed,
    get_device,
    get_loss,
    get_optimizer,
    get_scheduler,
    calculate_metrics,
    log_experiment,
    save_fold_artifacts,
)


def fit(config):
    device = get_device(config)

    X, y = load_train_data(config)

    loss_func = get_loss(config)

    skf = StratifiedKFold(
        n_splits=config.data.n_splits,
        shuffle=True,
        random_state=config.general.seed,
    )

    fold_histories = []
    fold_metrics = []

    oof_probabilities = np.zeros(len(X), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\nFold {fold}/{config.data.n_splits}")

        set_seed(config.general.seed + fold - 1)

        train_loader, val_loader, preprocessor, input_size = create_fold_dataloaders(
            X=X,
            y=y,
            train_idx=train_idx,
            val_idx=val_idx,
            config=config,
        )

        model = MLPClassifier(
            input_size=input_size,
            hidden_dims=config.model.hidden_dims,
            dropout=config.model.dropout,
        ).to(device)

        optimizer = get_optimizer(model, config)
        scheduler = get_scheduler(optimizer, config)

        log_experiment(
            config=config,
            device=device,
            input_size=input_size,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
        )

        save_fold_artifacts(
            model=model,
            preprocessor=preprocessor,
            input_size=input_size,
            config=config,
            fold=fold,
        )

        val_probabilities = predict(
            model=model,
            data_loader=val_loader,
            device=device,
        )

        oof_probabilities[val_idx] = val_probabilities

        val_loss, metrics = validate(
            model=model,
            val_loader=val_loader,
            loss_func=loss_func,
            device=device,
            threshold=config.training.threshold,
        )

        print(
            f"Fold {fold} result | "
            f"Val loss: {val_loss:.4f} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"ROC-AUC: {metrics['roc_auc']:.4f}"
        )

        fold_histories.append(history)
        fold_metrics.append(metrics)

        model.to("cpu")

    oof_metrics = calculate_metrics(
        probabilities=oof_probabilities,
        targets=y,
        threshold=config.training.threshold,
    )

    print("\nOOF RESULTS")
    print(f"Accuracy:  {oof_metrics['accuracy']:.4f}")
    print(f"Precision: {oof_metrics['precision']:.4f}")
    print(f"Recall:    {oof_metrics['recall']:.4f}")
    print(f"F1:        {oof_metrics['f1']:.4f}")
    print(f"ROC-AUC:   {oof_metrics['roc_auc']:.4f}")

    return {
        "histories": fold_histories,
        "fold_metrics": fold_metrics,
        "oof_metrics": oof_metrics,
        "oof_probabilities": oof_probabilities,
        "targets": y,
    }


if __name__ == "__main__":
    results = fit(config)

    for threshold in np.arange(0.30, 0.71, 0.05):
        metrics = calculate_metrics(
            probabilities=results["oof_probabilities"],
            targets=results["targets"],
            threshold=threshold,
        )

        print(
            f"Threshold: {threshold:.2f} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1']:.4f}"
        )