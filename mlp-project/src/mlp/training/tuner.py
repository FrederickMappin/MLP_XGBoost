import optuna
from copy import deepcopy

from src.mlp.data.dataset import build_loaders
from src.mlp.models.mlp import MLP
from src.mlp.training.trainer import Trainer


def run_study(X_raw, y_raw, input_dim: int, cfg: dict, device) -> dict:
    """
    Run an Optuna hyperparameter search and return a patched copy of cfg.

    The search is always capped at cfg['optuna']['n_trials'] trials and
    uses MedianPruner to cut unpromising trials early.
    """

    def objective(trial):
        n_layers     = trial.suggest_int("n_layers", 1, 4)
        hidden_dim   = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
        dropout_p    = trial.suggest_float("dropout_p", 0.1, 0.5)
        lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        batch_size   = trial.suggest_categorical("batch_size", [16, 32, 64])

        trial_cfg = deepcopy(cfg)
        trial_cfg["model"]["hidden_dims"] = [hidden_dim] * n_layers
        trial_cfg["model"]["dropout_p"]   = dropout_p
        trial_cfg["training"]["lr"]           = lr
        trial_cfg["training"]["weight_decay"] = weight_decay
        trial_cfg["training"]["batch_size"]   = batch_size
        trial_cfg["training"]["epochs"]       = 40
        trial_cfg["training"]["patience"]     = 8
        trial_cfg["training"]["lr_patience"]  = 4

        result    = build_loaders(X_raw, y_raw, trial_cfg)
        t_train, t_val = result[0], result[1]

        output_dim = trial_cfg["data"].get("num_classes", 1)
        t_model    = MLP(input_dim, trial_cfg["model"], output_dim).to(device)
        t_trainer  = Trainer(t_model, trial_cfg, device)

        for epoch in range(1, 41):
            t_trainer._run_epoch(t_train, train=True)
            vl, _ = t_trainer._run_epoch(t_val, train=False)
            trial.report(vl, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return vl

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction=cfg["optuna"]["direction"],
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    study.optimize(objective, n_trials=cfg["optuna"]["n_trials"], show_progress_bar=True)

    best = study.best_params
    print(f"\nOptuna best — val_loss: {study.best_value:.5f}")
    print(f"Best params: {best}")

    # ── patch cfg with best values ────────────────────────────────────────────
    tuned = deepcopy(cfg)
    tuned["model"]["hidden_dims"]     = [best["hidden_dim"]] * best["n_layers"]
    tuned["model"]["dropout_p"]       = best["dropout_p"]
    tuned["training"]["lr"]           = best["lr"]
    tuned["training"]["weight_decay"] = best["weight_decay"]
    tuned["training"]["batch_size"]   = best["batch_size"]

    return tuned
