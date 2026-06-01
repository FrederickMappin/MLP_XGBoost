"""
train.py — unified training entry point for all three tasks.

Usage
-----
# Train with defaults:
    python scripts/train.py --config configs/binary.yaml

# Run Optuna tuning first, then train:
    python scripts/train.py --config configs/binary.yaml --tune

# View results in MLflow:
    mlflow ui          # → http://localhost:5000
"""

import argparse
import sys
sys.path.insert(0, ".")

import yaml
import torch
import mlflow
import mlflow.pytorch

from src.mlp.utils.utils     import set_seed
from src.mlp.data.dataset    import load_dataset, build_loaders
from src.mlp.models.mlp      import MLP, model_summary
from src.mlp.training.trainer import Trainer
from src.mlp.training.tuner   import run_study
from src.mlp.evaluation.evaluate import evaluate, plot_history


def main():
    parser = argparse.ArgumentParser(description="Train an MLP with MLflow + optional Optuna tuning.")
    parser.add_argument("--config", default="configs/binary.yaml", help="Path to YAML config file.")
    parser.add_argument("--tune",   action="store_true",           help="Run Optuna before training.")
    args = parser.parse_args()

    # ── load config ───────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"\nTask:   {cfg['task']}")
    print(f"Config: {args.config}")
    print(f"Tune:   {args.tune}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    set_seed(cfg["data"]["random_state"])

    # ── data ─────────────────────────────────────────────────────────────────
    X_raw, y_raw = load_dataset(cfg["task"])
    input_dim    = X_raw.shape[1]

    # ── optional Optuna tuning ────────────────────────────────────────────────
    if args.tune:
        cfg = run_study(X_raw, y_raw, input_dim, cfg, device)

    # ── build loaders + model ─────────────────────────────────────────────────
    result     = build_loaders(X_raw, y_raw, cfg)
    train_loader, val_loader, test_loader = result[0], result[1], result[2]
    scaler_y   = result[4] if cfg["task"] == "regression" else None

    output_dim = cfg["data"].get("num_classes", 1)
    model      = MLP(input_dim, cfg["model"], output_dim).to(device)
    model_summary(model, input_dim)

    # ── MLflow run ────────────────────────────────────────────────────────────
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    with mlflow.start_run(run_name=cfg["mlflow"]["run_name"]):
        mlflow.log_params({
            "task":         cfg["task"],
            "hidden_dims":  str(cfg["model"]["hidden_dims"]),
            "dropout_p":    cfg["model"]["dropout_p"],
            "batch_norm":   cfg["model"]["batch_norm"],
            "lr":           cfg["training"]["lr"],
            "batch_size":   cfg["training"]["batch_size"],
            "weight_decay": cfg["training"]["weight_decay"],
            "epochs":       cfg["training"]["epochs"],
            "tuned":        args.tune,
        })

        # ── train ─────────────────────────────────────────────────────────────
        trainer = Trainer(model, cfg, device)
        history = trainer.fit(train_loader, val_loader, cfg["paths"]["checkpoint"])

        # ── log per-epoch metrics ─────────────────────────────────────────────
        keys   = list(history.keys())
        values = list(zip(*history.values()))
        for epoch, vals in enumerate(values, 1):
            mlflow.log_metrics(dict(zip(keys, vals)), step=epoch)

        # ── evaluate + log test metrics ───────────────────────────────────────
        plot_history(history)
        test_metrics = evaluate(model, test_loader, device, cfg["task"], scaler_y)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # ── log model artifact ────────────────────────────────────────────────
        mlflow.pytorch.log_model(model, artifact_path="model")

    print(f'\nRun logged → experiment: "{cfg["mlflow"]["experiment"]}"')
    print("View results: mlflow ui  →  http://localhost:5000")


if __name__ == "__main__":
    main()
