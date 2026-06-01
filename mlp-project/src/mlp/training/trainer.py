import os

import torch
import torch.nn as nn

from src.mlp.utils.utils import EarlyStopping, timer


class Trainer:
    """
    Task-aware training loop.

    Handles binary classification, multiclass classification, and regression
    from a single class — behaviour is driven by cfg['task'].
    """

    def __init__(self, model: nn.Module, cfg: dict, device: torch.device):
        self.model  = model
        self.task   = cfg["task"]
        self.cfg    = cfg["training"]
        self.device = device

        # ── loss ──────────────────────────────────────────────────────────────
        if self.task == "binary":
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.task == "multiclass":
            self.criterion = nn.CrossEntropyLoss()
        elif self.task == "regression":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown task: {self.task!r}")

        # ── optimizer + scheduler + early stopping ────────────────────────────
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.cfg["lr"],
            weight_decay=self.cfg["weight_decay"],
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=self.cfg["lr_patience"],
            factor=self.cfg["lr_factor"],
        )
        self.early_stopping = EarlyStopping(patience=self.cfg["patience"])

        # ── history ───────────────────────────────────────────────────────────
        self.history: dict = {"train_loss": [], "val_loss": []}
        if self.task != "regression":
            self.history["train_acc"] = []
            self.history["val_acc"]   = []

    # ─────────────────────────────────────────────────────────────────────────

    def _run_epoch(self, loader, train: bool):
        self.model.train() if train else self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                logits = self.model(Xb)
                loss   = self.criterion(logits, yb)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * len(Xb)
                total      += len(Xb)

                if self.task == "binary":
                    preds    = (torch.sigmoid(logits) >= 0.5).float()
                    correct += (preds == yb).sum().item()
                elif self.task == "multiclass":
                    correct += (logits.argmax(dim=1) == yb).sum().item()

        avg_loss = total_loss / total
        acc      = correct / total if self.task != "regression" else None
        return avg_loss, acc

    # ─────────────────────────────────────────────────────────────────────────

    @timer
    def fit(self, train_loader, val_loader, checkpoint_path: str) -> dict:
        print(f"\nTraining [{self.task}] for up to {self.cfg['epochs']} epochs ...\n")

        for epoch in range(1, self.cfg["epochs"] + 1):
            tl, ta = self._run_epoch(train_loader, train=True)
            vl, va = self._run_epoch(val_loader,   train=False)

            self.history["train_loss"].append(tl)
            self.history["val_loss"].append(vl)
            if self.task != "regression":
                self.history["train_acc"].append(ta)
                self.history["val_acc"].append(va)

            self.scheduler.step(vl)
            self.early_stopping.step(vl)

            if epoch % 10 == 0 or epoch == 1:
                lr = self.optimizer.param_groups[0]["lr"]
                if self.task != "regression":
                    print(f"Ep {epoch:>4}  TrLoss:{tl:.4f} Acc:{ta:.4f}  "
                          f"VlLoss:{vl:.4f} Acc:{va:.4f}  LR:{lr:.2e}")
                else:
                    print(f"Ep {epoch:>4}  TrMSE:{tl:.4f}  VlMSE:{vl:.4f}  LR:{lr:.2e}")

            if self.early_stopping.stop:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(
            {"model_state": self.model.state_dict(), "history": self.history},
            checkpoint_path,
        )
        print(f"\nCheckpoint saved → {checkpoint_path}")
        return self.history
