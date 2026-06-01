import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ─── Dataset ──────────────────────────────────────────────────────────────────

class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, y_dtype=torch.float32):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=y_dtype)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─── Dataset loader ───────────────────────────────────────────────────────────

def load_dataset(task: str):
    """Return (X, y) numpy arrays for the built-in demo dataset for each task."""
    if task == "binary":
        from sklearn.datasets import load_breast_cancer
        raw = load_breast_cancer()
        return raw.data, raw.target.astype(np.float32)

    elif task == "multiclass":
        from sklearn.datasets import load_iris
        raw = load_iris()
        return raw.data, raw.target.astype(np.int64)

    elif task == "regression":
        from sklearn.datasets import fetch_california_housing
        raw = fetch_california_housing()
        return raw.data, raw.target.astype(np.float32)

    else:
        raise ValueError(f"Unknown task: {task!r}. Choose binary | multiclass | regression.")


# ─── DataLoader factory ───────────────────────────────────────────────────────

def build_loaders(X: np.ndarray, y: np.ndarray, cfg: dict):
    """
    Split, scale, and wrap data into DataLoaders.

    Returns
    -------
    classification : (train_loader, val_loader, test_loader, scaler_X)
    regression     : (train_loader, val_loader, test_loader, scaler_X, scaler_y)
    """
    task = cfg["task"]
    d, t = cfg["data"], cfg["training"]

    # ── split ──────────────────────────────────────────────────────────────────
    val_test = d["val_size"] + d["test_size"]
    stratify = y if task in ("binary", "multiclass") else None

    X_tr, X_vt, y_tr, y_vt = train_test_split(
        X, y, test_size=val_test, random_state=d["random_state"], stratify=stratify
    )
    rel_test = d["test_size"] / val_test
    stratify_vt = y_vt if task in ("binary", "multiclass") else None

    X_v, X_te, y_v, y_te = train_test_split(
        X_vt, y_vt, test_size=rel_test,
        random_state=d["random_state"], stratify=stratify_vt
    )

    # ── scale features ─────────────────────────────────────────────────────────
    scaler_X = StandardScaler()
    X_tr = scaler_X.fit_transform(X_tr)
    X_v  = scaler_X.transform(X_v)
    X_te = scaler_X.transform(X_te)

    # ── scale targets (regression only) ────────────────────────────────────────
    scaler_y = None
    if task == "regression":
        scaler_y = StandardScaler()
        y_tr = scaler_y.fit_transform(y_tr.reshape(-1, 1)).ravel()
        y_v  = scaler_y.transform(y_v.reshape(-1, 1)).ravel()
        y_te = scaler_y.transform(y_te.reshape(-1, 1)).ravel()

    # ── y dtype ────────────────────────────────────────────────────────────────
    y_dtype = torch.int64 if task == "multiclass" else torch.float32

    bs = t["batch_size"]
    tr_l = DataLoader(TabularDataset(X_tr, y_tr, y_dtype), batch_size=bs, shuffle=True)
    v_l  = DataLoader(TabularDataset(X_v,  y_v,  y_dtype), batch_size=bs, shuffle=False)
    te_l = DataLoader(TabularDataset(X_te, y_te, y_dtype), batch_size=bs, shuffle=False)

    print(f"Split — Train: {len(X_tr)} | Val: {len(X_v)} | Test: {len(X_te)}")

    if task == "regression":
        return tr_l, v_l, te_l, scaler_X, scaler_y
    return tr_l, v_l, te_l, scaler_X
