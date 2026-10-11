import copy
import torch

from deep_learning_pipeline.utils import calculate_metrics


def validate(model, val_loader, loss_func, device, threshold=0.5):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_probabilities = []
    all_targets = []

    with torch.inference_mode():
        for X, y in val_loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            loss = loss_func(logits, y)

            batch_size = X.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            probabilities = torch.sigmoid(logits)

            all_probabilities.append(probabilities.cpu())
            all_targets.append(y.cpu())

    val_loss = total_loss / total_samples

    all_probabilities = torch.cat(all_probabilities)
    all_targets = torch.cat(all_targets)

    metrics = calculate_metrics(
        probabilities=all_probabilities,
        targets=all_targets,
        threshold=threshold,
    )

    return val_loss, metrics


def train(model, train_loader, val_loader, loss_func, optimizer, scheduler, device, config):
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None

    history = {
        "train_loss": [],
        "val_loss": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
        "learning_rate": [],
    }

    for epoch in range(1, config.training.epochs + 1):
        model.train()

        total_train_loss = 0.0
        total_samples = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(X)
            loss = loss_func(logits, y)

            loss.backward()
            optimizer.step()

            batch_size = X.size(0)

            total_train_loss += loss.item() * batch_size
            total_samples += batch_size

        train_loss = total_train_loss / total_samples

        val_loss, metrics = validate(
            model=model,
            val_loader=val_loader,
            loss_func=loss_func,
            device=device,
            threshold=config.training.threshold,
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["accuracy"].append(metrics["accuracy"])
        history["precision"].append(metrics["precision"])
        history["recall"].append(metrics["recall"])
        history["f1"].append(metrics["f1"])
        history["roc_auc"].append(metrics["roc_auc"])
        history["learning_rate"].append(current_lr)

        print(
            f"Epoch {epoch}/{config.training.epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"ROC-AUC: {metrics['roc_auc']:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.training.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch} | "
                f"Best val loss: {best_val_loss:.4f}"
            )
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history