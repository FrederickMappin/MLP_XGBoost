import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
)


def evaluate(model, loader, device, task: str, scaler_y=None) -> dict:
    """
    Run inference and compute metrics.

    Parameters
    ----------
    scaler_y : sklearn scaler (regression only) — used to inverse-transform
               predictions back to original scale before computing metrics.
    """
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for Xb, yb in loader:
            logits = model(Xb.to(device)).cpu()
            all_logits.append(logits)
            all_labels.append(yb)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()

    if task == "binary":
        return _eval_binary(logits, labels)
    elif task == "multiclass":
        return _eval_multiclass(logits, labels)
    elif task == "regression":
        return _eval_regression(logits, labels, scaler_y)
    else:
        raise ValueError(f"Unknown task: {task!r}")


# ─── task-specific helpers ────────────────────────────────────────────────────

def _eval_binary(logits, labels):
    probs = 1 / (1 + np.exp(-logits))          # sigmoid
    preds = (probs >= 0.5).astype(int)
    labels_int = labels.astype(int)

    metrics = {
        "accuracy":  accuracy_score(labels_int, preds),
        "precision": precision_score(labels_int, preds, zero_division=0),
        "recall":    recall_score(labels_int, preds, zero_division=0),
        "f1":        f1_score(labels_int, preds, zero_division=0),
        "roc_auc":   roc_auc_score(labels_int, probs),
    }
    _print_metrics(metrics)
    print(classification_report(labels_int, preds))
    _plot_confusion(labels_int, preds, "Binary Classification")
    return metrics


def _eval_multiclass(logits, labels):
    preds      = logits.argmax(axis=1)
    labels_int = labels.astype(int)

    metrics = {
        "accuracy":       accuracy_score(labels_int, preds),
        "f1_macro":       f1_score(labels_int, preds, average="macro",    zero_division=0),
        "f1_weighted":    f1_score(labels_int, preds, average="weighted", zero_division=0),
        "precision_macro":precision_score(labels_int, preds, average="macro", zero_division=0),
        "recall_macro":   recall_score(labels_int, preds, average="macro", zero_division=0),
    }
    _print_metrics(metrics)
    print(classification_report(labels_int, preds))
    _plot_confusion(labels_int, preds, "Multiclass Classification")
    return metrics


def _eval_regression(logits, labels, scaler_y):
    preds = logits.copy()
    true  = labels.copy()

    if scaler_y is not None:
        preds = scaler_y.inverse_transform(preds.reshape(-1, 1)).ravel()
        true  = scaler_y.inverse_transform(true.reshape(-1, 1)).ravel()

    mse  = mean_squared_error(true, preds)
    metrics = {
        "mse":  mse,
        "rmse": np.sqrt(mse),
        "mae":  mean_absolute_error(true, preds),
        "r2":   r2_score(true, preds),
    }
    _print_metrics(metrics)
    _plot_regression(true, preds)
    return metrics


# ─── plot helpers ─────────────────────────────────────────────────────────────

def _print_metrics(metrics: dict) -> None:
    print("\n── Metrics ──────────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<18}: {v:.4f}")
    print()


def _plot_confusion(labels, preds, title: str) -> None:
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {title}")
    plt.tight_layout(); plt.show()


def _plot_regression(true, preds) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(true, preds, alpha=0.4, s=15)
    lim = [min(true.min(), preds.min()), max(true.max(), preds.max())]
    axes[0].plot(lim, lim, "r--", linewidth=1)
    axes[0].set_xlabel("True"); axes[0].set_ylabel("Predicted")
    axes[0].set_title("True vs Predicted")
    residuals = true - preds
    axes[1].hist(residuals, bins=40, edgecolor="white")
    axes[1].axvline(0, color="red", linewidth=1)
    axes[1].set_title("Residual Distribution")
    plt.tight_layout(); plt.show()


def plot_history(history: dict) -> None:
    """Plot train/val loss (and accuracy if present)."""
    has_acc = "train_acc" in history
    n_cols  = 2 if has_acc else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    if has_acc:
        axes[1].plot(history["train_acc"], label="Train")
        axes[1].plot(history["val_acc"],   label="Val")
        axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle("Training History", fontsize=13)
    plt.tight_layout(); plt.show()
