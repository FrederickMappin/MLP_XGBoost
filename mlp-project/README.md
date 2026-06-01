# MLP Harness

PyTorch MLP template supporting **binary classification**, **multiclass classification**, and **regression** — with Optuna hyperparameter tuning and MLflow experiment tracking built in.

---

## Project Structure

```
mlp-project/
├── .devcontainer/
│   └── devcontainer.json       ← GitHub Codespaces config
├── configs/
│   ├── binary.yaml             ← Breast Cancer dataset
│   ├── multiclass.yaml         ← Iris dataset
│   └── regression.yaml         ← California Housing dataset
├── src/
│   └── mlp/
│       ├── data/dataset.py     ← TabularDataset, build_loaders, load_dataset
│       ├── models/mlp.py       ← MLP architecture
│       ├── training/
│       │   ├── trainer.py      ← Task-aware training loop
│       │   └── tuner.py        ← Optuna search + CONFIG patching
│       ├── evaluation/
│       │   └── evaluate.py     ← Metrics + plots for all 3 tasks
│       └── utils/utils.py      ← set_seed, timer, EarlyStopping
├── scripts/
│   └── train.py                ← Unified training entry point
├── outputs/                    ← Saved model checkpoints (.gitignored)
├── requirements.txt
└── .gitignore
```

---

## Quickstart

### 1 — Open in GitHub Codespaces
Click **Code → Codespaces → Create codespace**. Dependencies install automatically via `postCreateCommand`.

### 2 — Train (defaults)
```bash
python scripts/train.py --config configs/binary.yaml
python scripts/train.py --config configs/multiclass.yaml
python scripts/train.py --config configs/regression.yaml
```

### 3 — Train with Optuna tuning
```bash
python scripts/train.py --config configs/binary.yaml --tune
```
Runs 40 trials (configurable via `n_trials` in the YAML), prunes bad ones with `MedianPruner`, then trains the final model with the best hyperparams found.

### 4 — View experiments in MLflow
```bash
mlflow ui
```
Open **http://localhost:5000** (port is auto-forwarded in Codespaces).

---

## Configuration

All hyperparameters live in the YAML configs — nothing is hardcoded.

| Key | Description |
|-----|-------------|
| `task` | `binary` / `multiclass` / `regression` |
| `data.val_size` / `test_size` | Train/val/test split ratios |
| `model.hidden_dims` | List of hidden layer sizes |
| `model.dropout_p` | Dropout probability |
| `model.batch_norm` | Enable BatchNorm after each layer |
| `training.epochs` / `lr` / `batch_size` | Standard training params |
| `training.patience` | Early stopping patience |
| `optuna.n_trials` | Number of tuning trials |
| `mlflow.experiment` / `run_name` | MLflow tracking labels |

---

## Architecture

```
Input → [Linear → BN → ReLU → Dropout] × N → Linear → Output
```

- **Binary**: output dim = 1, `BCEWithLogitsLoss`, sigmoid ≥ 0.5 threshold  
- **Multiclass**: output dim = num_classes, `CrossEntropyLoss`, argmax  
- **Regression**: output dim = 1, `MSELoss`, target is z-scored during training  
