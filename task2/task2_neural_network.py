"""
COMP30027 Project 2 — Task 2: Fine-Grained Bird Species Classification
Neural Network classifier with ResNet-50 feature extraction, hyperparameter tuning,
and validation accuracy optimisation.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from transformers import AutoImageProcessor, AutoModel

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'task1'))
from _model_utils import evaluate as mu_evaluate

# ─────────────────────────────────────────────────────────────────────────────
# 0.  PATHS
# ─────────────────────────────────────────────────────────────────────────────

TASK2_DIR  = "task2_data/"
TRAIN_META = os.path.join(TASK2_DIR, "train_metadata.csv")
TEST_META  = os.path.join(TASK2_DIR, "test_metadata.csv")

# Pre-extracted feature CSVs provided with the assignment (unified shared files)
HOG_PCA_FILE  = os.path.join(TASK2_DIR, "hog_pca.csv")
ADD_FEAT_FILE = os.path.join(TASK2_DIR, "additional_features.csv")

# ResNet features already extracted and saved
RESNET_FEATURES_TRAIN = "resnet_features_train2.csv"
RESNET_FEATURES_TEST  = "resnet_features_test2.csv"

# Output predictions file for Kaggle
KAGGLE_OUTPUT = "submission_results/task2_neural_predictions.csv"

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  RESNET-50 FEATURE EXTRACTION  (mirrors task 1 approach)
# ─────────────────────────────────────────────────────────────────────────────

def extract_resnet_features(metadata: pd.DataFrame, img_dir: str, out_path: str) -> pd.DataFrame:
    """
    Extracts 2048-dim ResNet-50 pooler features for every image in `metadata`.
    Saves to `out_path` and also returns the DataFrame.
    If `out_path` already exists the function loads and returns it directly
    (so re-running the script is fast).
    """
    if os.path.exists(out_path):
        print(f"Found cached features: {out_path}")
        return pd.read_csv(out_path)

    model_name = "microsoft/resnet-50"
    processor  = AutoImageProcessor.from_pretrained(model_name)
    model      = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    features = []
    print(f"Extracting ResNet-50 features → {out_path}")
    for idx, row in metadata.iterrows():
        img_path = os.path.join(img_dir, row["image_path"])
        image    = Image.open(img_path).convert("RGB")
        inputs   = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        features.append(outputs.pooler_output.squeeze().cpu().numpy())

    df = pd.DataFrame(features)
    df.insert(0, "image_id", metadata["image_id"])
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} feature vectors to {out_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE ASSEMBLY
#     We combine three sources:
#       (a) ResNet-50 deep features          — rich semantic representation
#       (b) HOG-PCA features                 — structural / shape information
#       (c) Additional features              — colour statistics, edge density
#     Colour histograms are intentionally omitted for task 2: fine-grained
#     bird species share very similar colour distributions, so they add noise.
# ─────────────────────────────────────────────────────────────────────────────

def load_and_merge_features(
    meta_path: str,
    resnet_path: str,
    hog_path: str,
    add_path: str,
    has_labels: bool = True,
) -> tuple:
    """
    Returns (X, y, image_ids, label_encoder) where y is None when has_labels=False.
    X is a numpy float32 array of shape (n_samples, n_features).
    Rows are perfectly aligned and indexed by image_id across all files.
    """
    meta = pd.read_csv(meta_path)

    # ── 1. ResNet features ── (Drop both image_id and image_path text columns safely)
    resnet_df   = pd.read_csv(resnet_path)
    resnet_feat = resnet_df.drop(columns=["image_id", "image_path"], errors="ignore").values.astype(np.float32)

    # ── 2. HOG-PCA features ── (Extract only rows matching active metadata image_ids)
    hog_combined = pd.read_csv(hog_path)
    hog_df       = pd.merge(meta[['image_id']], hog_combined, on='image_id', how='left')
    hog_feat     = hog_df.drop(columns=["image_id", "image_path"], errors="ignore").values.astype(np.float32)

    # ── 3. Additional features ── (Extract only rows matching active metadata image_ids)
    add_combined = pd.read_csv(add_path)
    add_df       = pd.merge(meta[['image_id']], add_combined, on='image_id', how='left')
    add_feat     = add_df.drop(columns=["image_id", "image_path"], errors="ignore").values.astype(np.float32)

    # Concatenate columns side-by-side
    X = np.concatenate([resnet_feat, hog_feat, add_feat], axis=1)

    le = None
    y  = None
    if has_labels:
        le = LabelEncoder()
        y  = le.fit_transform(meta["class_id"].values)
        print(f"Classes ({len(le.classes_)}): {list(le.classes_)}")

    return X, y, meta["image_id"].values, le


# ─────────────────────────────────────────────────────────────────────────────
# 3.  NEURAL NETWORK ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

class BirdClassifierMLP(nn.Module):
    """
    Multi-Layer Perceptron for fine-grained bird classification.
    Architecture:  input → BN → FC → GELU → Dropout
                              → FC → GELU → Dropout
                              → FC → (logits)
    """

    def __init__(self, input_dim: int, hidden_dims: list, n_classes: int, dropout: float):
        super().__init__()
        layers = [nn.BatchNorm1d(input_dim)]
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TRAINING & EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimiser, scheduler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimiser.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimiser.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            preds       = logits.argmax(1)
            correct    += (preds == y_batch).sum().item()
            total      += len(y_batch)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def make_loaders(X_tr, y_tr, X_val, y_val, batch_size):
    def to_tensor_ds(X, y):
        return TensorDataset(torch.tensor(X, dtype=torch.float32),
                             torch.tensor(y, dtype=torch.long))
    tr_loader  = DataLoader(to_tensor_ds(X_tr, y_tr),   batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader = DataLoader(to_tensor_ds(X_val, y_val), batch_size=batch_size, shuffle=False)
    return tr_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# 5.  HYPERPARAMETER SEARCH  (grid search over a small manual grid,
#     evaluated via stratified k-fold cross-validation)
# ─────────────────────────────────────────────────────────────────────────────

PARAM_GRID = {
    "hidden_dims" : [[1024, 512], [512, 256], [1024, 512, 256]],
    "dropout"     : [0.3, 0.5],
    "lr"          : [3e-4, 1e-4],
    "weight_decay": [1e-4, 1e-3],
    "batch_size"  : [32, 64],
    "epochs"      : [80],   # inner-loop epochs (kept shorter during search)
}

N_CV_FOLDS = 5   # stratified k-fold folds for hyperparameter evaluation


def cross_val_score_config(X, y, config: dict, n_classes: int) -> float:
    """Returns mean validation accuracy across folds for a single config."""
    skf    = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scores = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        # Scale per fold (fit on train, apply to val)
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_val  = scaler.transform(X_val)

        model = BirdClassifierMLP(
            input_dim   = X_tr.shape[1],
            hidden_dims = config["hidden_dims"],
            n_classes   = n_classes,
            dropout     = config["dropout"],
        ).to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimiser = optim.AdamW(model.parameters(),
                                lr=config["lr"],
                                weight_decay=config["weight_decay"])
        total_steps = config["epochs"] * max(1, len(X_tr) // config["batch_size"])
        scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=total_steps)

        tr_loader, val_loader = make_loaders(X_tr, y_tr, X_val, y_val, config["batch_size"])

        best_val_acc = 0.0
        patience_cnt = 0
        PATIENCE = 15  # early-stopping patience (in epochs)

        for epoch in range(config["epochs"]):
            train_epoch(model, tr_loader, criterion, optimiser, scheduler)
            _, val_acc, _, _ = eval_epoch(model, val_loader, criterion)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_cnt = 0
            else:
                patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

        scores.append(best_val_acc)

    mean_acc = float(np.mean(scores))
    print(f"  Config {config}  →  CV acc = {mean_acc:.4f}  (folds: {[f'{s:.3f}' for s in scores]})")
    return mean_acc


def hyperparameter_search(X, y, n_classes: int) -> dict:
    """Exhaustive grid search; returns the best config dict."""
    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(itertools.product(*values))

    best_acc, best_cfg = 0.0, None
    for combo in combos:
        cfg = dict(zip(keys, combo))
        acc = cross_val_score_config(X, y, cfg, n_classes)
        if acc > best_acc:
            best_acc, best_cfg = acc, cfg

    print(f"\n✓ Best config (val acc = {best_acc:.4f}):\n  {best_cfg}")
    return best_cfg


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FINAL TRAINING ON ALL TRAINING DATA  (using best config)
# ─────────────────────────────────────────────────────────────────────────────

def train_final_model(X, y, config: dict, n_classes: int, final_epochs: int = 150):
    """Train on the entire training set with best hyperparameters."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = BirdClassifierMLP(
        input_dim   = X_scaled.shape[1],
        hidden_dims = config["hidden_dims"],
        n_classes   = n_classes,
        dropout     = config["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimiser = optim.AdamW(model.parameters(),
                            lr=config["lr"],
                            weight_decay=config["weight_decay"])
    total_steps = final_epochs * max(1, len(X) // config["batch_size"])
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=total_steps)

    full_loader = DataLoader(
        TensorDataset(torch.tensor(X_scaled, dtype=torch.float32),
                      torch.tensor(y,        dtype=torch.long)),
        batch_size=config["batch_size"], shuffle=True, drop_last=True,
    )

    print(f"\nTraining final model for {final_epochs} epochs …")
    for epoch in range(1, final_epochs + 1):
        loss, acc = train_epoch(model, full_loader, criterion, optimiser, scheduler)
        if epoch % 25 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}  loss={loss:.4f}  train_acc={acc:.4f}")

    return model, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 7.  REPORTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.tight_layout()
    plt.savefig("task2_confusion_matrix.png", dpi=150)
    print("Saved confusion matrix → task2_confusion_matrix.png")
    plt.show()


def final_cv_report(X, y, best_cfg: dict, n_classes: int, class_names, train_meta):
    """
    Runs one more stratified k-fold with the best config to get a reliable
    validation accuracy estimate for the report, and plots a confusion matrix.
    """
    _train_meta_for_report = train_meta
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    all_preds, all_labels = [], []

    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_val  = scaler.transform(X_val)

        model = BirdClassifierMLP(X_tr.shape[1], best_cfg["hidden_dims"], n_classes, best_cfg["dropout"]).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimiser = optim.AdamW(model.parameters(), lr=best_cfg["lr"], weight_decay=best_cfg["weight_decay"])
        total_steps = best_cfg["epochs"] * max(1, len(X_tr) // best_cfg["batch_size"])
        scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=total_steps)
        tr_loader, val_loader = make_loaders(X_tr, y_tr, X_val, y_val, best_cfg["batch_size"])

        best_state, best_val = None, 0.0
        patience_cnt = 0
        for epoch in range(best_cfg["epochs"]):
            train_epoch(model, tr_loader, criterion, optimiser, scheduler)
            _, val_acc, preds, labels = eval_epoch(model, val_loader, criterion)
            if val_acc > best_val:
                best_val   = val_acc
                best_state = (preds.copy(), labels.copy())
                patience_cnt = 0
            else:
                patience_cnt += 1
            if patience_cnt >= 15:
                break

        all_preds.extend(best_state[0])
        all_labels.extend(best_state[1])

    # Remap encoded indices (0-9) back to original class_ids so _model_utils
    # can look up class names from train_metadata correctly
    le_tmp = LabelEncoder().fit(_train_meta_for_report["class_id"].values)
    all_labels_mapped = le_tmp.inverse_transform(all_labels)
    all_preds_mapped  = le_tmp.inverse_transform(all_preds)
    overall_acc = mu_evaluate(
        "Neural Network (Ensemble)",
        all_labels_mapped, all_preds_mapped,
        train_meta=_train_meta_for_report,
    )
    return overall_acc


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── 8a. Load metadata ──
    train_meta = pd.read_csv(TRAIN_META)
    test_meta  = pd.read_csv(TEST_META)

    # ── 8b. ResNet features already extracted; this is a no-op if files exist ──
    extract_resnet_features(train_meta, TASK2_DIR, RESNET_FEATURES_TRAIN)
    extract_resnet_features(test_meta,  TASK2_DIR, RESNET_FEATURES_TEST)

    # ── 8c. Assemble combined feature matrix ──
    # Both train and test function calls pass the exact same shared feature databases
    X_train, y_train, train_ids, le = load_and_merge_features(
        TRAIN_META, RESNET_FEATURES_TRAIN, HOG_PCA_FILE, ADD_FEAT_FILE,
        has_labels=True,
    )

    X_test, _, test_ids, _ = load_and_merge_features(
        TEST_META, RESNET_FEATURES_TEST, HOG_PCA_FILE, ADD_FEAT_FILE,
        has_labels=False,
    )

    n_classes   = len(np.unique(y_train))
    class_names = le.classes_
    print(f"\nFeature matrix shape — train: {X_train.shape}, test: {X_test.shape}")

    # ── 8d. Hyperparameter search ──
    print("\n=== Hyperparameter search (grid search + 5-fold CV) ===")
    best_cfg = hyperparameter_search(X_train, y_train, n_classes)

    # ── 8e. Cross-val report with best config (for the written report) ──
    print("\n=== Final CV accuracy report ===")
    final_cv_report(X_train, y_train, best_cfg, n_classes, class_names, train_meta)

    # ── 8f. Train final model on all training data ──
    final_model, final_scaler = train_final_model(X_train, y_train, best_cfg, n_classes)

    # ── 8g. Predict on test set ──
    final_model.eval()
    X_test_scaled = final_scaler.transform(X_test)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits     = final_model(X_test_tensor)
        test_preds = logits.argmax(1).cpu().numpy()

    # le was fitted on class_id integers, so inverse_transform gives back class_ids
    pred_class_ids = le.inverse_transform(test_preds)

    # ── 8h. Save Kaggle submission (image_id, class_id only) ──
    submission = pd.DataFrame({
        "image_id": test_ids,
        "class_id": pred_class_ids,
    })
    submission.to_csv(KAGGLE_OUTPUT, index=False)
    print(f"\n✓ Kaggle submission saved → {KAGGLE_OUTPUT}")
    print(submission.head())


if __name__ == "__main__":
    main()