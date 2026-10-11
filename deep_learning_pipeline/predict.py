import numpy as np
import torch


def predict(model, data_loader, device):
    model.eval()

    all_probabilities = []

    with torch.inference_mode():
        for batch in data_loader:
            if isinstance(batch, (tuple, list)):
                X = batch[0]
            else:
                X = batch

            X = X.to(device)

            logits = model(X)
            probabilities = torch.sigmoid(logits)

            all_probabilities.append(probabilities.cpu().numpy())

    return np.concatenate(all_probabilities)