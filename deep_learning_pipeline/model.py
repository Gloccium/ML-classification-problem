import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_dims, dropout):
        super().__init__()

        layers = []
        in_features = input_size

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])

            in_features = hidden_dim

        layers.append(nn.Linear(in_features, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)
