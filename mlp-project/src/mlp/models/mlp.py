import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Fully-connected MLP with optional BatchNorm and Dropout.

    Parameters
    ----------
    input_dim  : number of input features
    model_cfg  : dict with keys hidden_dims, dropout_p, batch_norm
    output_dim : 1 for binary/regression, num_classes for multiclass
    """

    def __init__(self, input_dim: int, model_cfg: dict, output_dim: int = 1):
        super().__init__()
        dims       = [input_dim] + model_cfg["hidden_dims"]
        dropout_p  = model_cfg["dropout_p"]
        batch_norm = model_cfg["batch_norm"]

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))

        layers.append(nn.Linear(dims[-1], output_dim))
        self.network = nn.Sequential(*layers)
        self._output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        if self._output_dim == 1:
            out = out.squeeze(1)   # (B, 1) → (B,)  for binary / regression
        return out


def model_summary(model: nn.Module, input_dim: int) -> None:
    total  = sum(p.numel() for p in model.parameters())
    train_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'─' * 44}")
    print(f"  Architecture:     {model.__class__.__name__}")
    print(f"  Input dim:        {input_dim}")
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {train_:,}")
    print(f"{'─' * 44}\n")
