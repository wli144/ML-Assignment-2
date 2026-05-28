import os
import copy
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from transformers import AutoImageProcessor, AutoModel

TASK2_DIR  = "task2_data/"
TRAIN_META = os.path.join(TASK2_DIR, "train_metadata.csv")
TEST_META  = os.path.join(TASK2_DIR, "test_metadata.csv")

# ResNet features already extracted and saved
RESNET_FEATURES_TRAIN = "resnet_features_train2.csv"
RESNET_FEATURES_TEST  = "resnet_features_test2.csv"

# Output predictions file for Kaggle
KAGGLE_OUTPUT = "task2_predictions.csv"

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def extract_resnet_features(metadata: pd.DataFrame, img_dir: str, out_path: str) -> pd.DataFrame:
    """
    Extracts 2048-dim ResNet-50 pooler features. Cached — skips if file exists.
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
    for _, row in metadata.iterrows():
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


def load_and_merge_features(
    meta_path: str,
    resnet_path: str,
    has_labels: bool = True,
) -> tuple:
    """
    Returns (X, y, image_ids, label_encoder).
    X = ResNet-50 pooler features only (2048-dim).
    """
    meta = pd.read_csv(meta_path)

    resnet_df   = pd.read_csv(resnet_path)
    resnet_feat = resnet_df.drop(columns=["image_id", "image_path"], errors="ignore").values.astype(np.float32)

    le = None
    y  = None
    if has_labels:
        le = LabelEncoder()
        y  = le.fit_transform(meta["class_id"].values)
        print(f"Classes ({len(le.classes_)}): {list(le.classes_)}")

    return resnet_feat, y, meta["image_id"].values, le


class BirdClassifierMLP(nn.Module):
    """
    MLP:  input → BN → [FC → GELU → Dropout] × n_layers → FC (logits)
    """
    def __init__(self, input_dim: int, hidden_dims: list, n_classes: int, dropout: float):
        super().__init__()
        layers = [nn.BatchNorm1d(input_dim)]
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers += [nn.Linear(in_dim, h_dim), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def mixup_batch(X, y, n_classes, alpha=0.3):
    """
    Mixup data augmentation: interpolates pairs of training samples.
    Returns mixed features and soft (one-hot blended) labels.
    Works in feature space, so no image pipeline needed.
    """
    lam   = np.random.beta(alpha, alpha)
    idx   = torch.randperm(X.size(0), device=X.device)
    X_mix = lam * X + (1 - lam) * X[idx]
    # one-hot blend for soft cross-entropy
    y_a   = torch.zeros(X.size(0), n_classes, device=X.device).scatter_(1, y.unsqueeze(1), 1.0)
    y_b   = torch.zeros(X.size(0), n_classes, device=X.device).scatter_(1, y[idx].unsqueeze(1), 1.0)
    y_mix = lam * y_a + (1 - lam) * y_b
    return X_mix, y_mix


def train_epoch(model, loader, criterion, optimiser, n_classes, scheduler=None, use_mixup=True):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        if use_mixup and model.training:
            X_batch, y_soft = mixup_batch(X_batch, y_batch, n_classes)
            optimiser.zero_grad()
            logits = model(X_batch)
            # soft cross-entropy: -sum(y_soft * log_softmax(logits))
            log_p  = torch.log_softmax(logits, dim=1)
            loss   = -(y_soft * log_p).sum(dim=1).mean()
        else:
            optimiser.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch) + PAIR_PENALTY * pairwise_confusion_loss(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            probs  = torch.softmax(logits, dim=1)
            total_loss += loss.item() * len(y_batch)
            preds       = logits.argmax(1)
            correct    += (preds == y_batch).sum().item()
            total      += len(y_batch)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    return (total_loss / total, correct / total,
            np.array(all_preds), np.array(all_labels),
            np.concatenate(all_probs, axis=0))


def make_loaders(X_tr, y_tr, X_val, y_val, batch_size):
    def ds(X, y):
        return TensorDataset(torch.tensor(X, dtype=torch.float32),
                             torch.tensor(y, dtype=torch.long))
    tr_loader  = DataLoader(ds(X_tr, y_tr),   batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader = DataLoader(ds(X_val, y_val), batch_size=batch_size, shuffle=False)
    return tr_loader, val_loader


def build_criterion(n_classes):
    """
    Weighted cross-entropy loss: hard classes receive HARD_CLASS_WEIGHT,
    all other classes receive 1.0. This directly addresses the Herring/Ring-billed
    Gull and Yellow/Wilson Warbler confusion pairs seen in the confusion matrix.
    """
    weights = torch.ones(n_classes, device=device)
    for c in HARD_CLASSES:
        if c < n_classes:
            weights[c] = HARD_CLASS_WEIGHT
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)


def pairwise_confusion_loss(logits, y_true):
    """
    Auxiliary penalty: for each confused pair (a, b), when the true label is a,
    penalise the model for assigning high probability to b (and vice versa).
    This is a soft constraint that discourages the specific misclassifications
    observed in the confusion matrix.
    """
    probs = torch.softmax(logits, dim=1)
    loss  = torch.tensor(0.0, device=logits.device)
    for a, b in CONFUSED_PAIRS:
        if a < logits.size(1) and b < logits.size(1):
            mask_a = (y_true == a)
            mask_b = (y_true == b)
            if mask_a.any():
                loss = loss + probs[mask_a, b].mean()
            if mask_b.any():
                loss = loss + probs[mask_b, a].mean()
    return loss


def build_model_and_optimiser(input_dim, n_classes, config):
    model     = BirdClassifierMLP(input_dim, config["hidden_dims"], n_classes, config["dropout"]).to(device)
    criterion = build_criterion(n_classes)
    optimiser = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    return model, criterion, optimiser



PARAM_GRID = {
    "hidden_dims" : [[512, 256], [1024, 512], [512, 256, 128]],
    "dropout"     : [0.4, 0.5, 0.6],
    "lr"          : [3e-4, 2e-4],
    "weight_decay": [5e-4, 1e-3, 2e-3],
    "batch_size"  : [32],
    "epochs"      : [120],  
}

N_CV_FOLDS = 5
PATIENCE   = 20 
HARD_CLASS_WEIGHT = 2.0   
HARD_CLASSES      = [3, 6, 8, 9]   
CONFUSED_PAIRS    = [(3, 6), (8, 9)] 
PAIR_PENALTY      = 0.5  


def train_fold(X_tr, y_tr, X_val, y_val, config, n_classes):
    """
    Trains one fold, returns (best_val_acc, best_model_state, scaler).
    Saves model weights at best validation epoch — not last epoch.
    """
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_val  = scaler.transform(X_val)

    model, criterion, optimiser = build_model_and_optimiser(X_tr.shape[1], n_classes, config)
    total_steps = config["epochs"] * max(1, len(X_tr) // config["batch_size"])
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=total_steps)
    tr_loader, val_loader = make_loaders(X_tr, y_tr, X_val, y_val, config["batch_size"])

    best_val_acc   = 0.0
    best_state     = None
    patience_cnt   = 0

    for epoch in range(config["epochs"]):
        train_epoch(model, tr_loader, criterion, optimiser, n_classes, scheduler, use_mixup=True)
        _, _, val_preds, val_labels, _ = eval_epoch(model, val_loader, criterion)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        if val_f1 > best_val_acc:
            best_val_acc = val_f1
            best_state   = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
        if patience_cnt >= PATIENCE:
            break

    return best_val_acc, best_state, scaler


def cross_val_score_config(X, y, config, n_classes):
    skf    = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scores = []
    for tr_idx, val_idx in skf.split(X, y):
        f1, _, _ = train_fold(X[tr_idx], y[tr_idx], X[val_idx], y[val_idx], config, n_classes)
        scores.append(f1)
    mean_f1 = float(np.mean(scores))
    std_f1  = float(np.std(scores))
    print(f"  {config}  →  CV macro-F1 = {mean_f1:.4f} ± {std_f1:.4f}  "
          f"(folds: {[f'{s:.3f}' for s in scores]})")
    return mean_f1, std_f1


def hyperparameter_search(X, y, n_classes):
    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))

    results = []
    for combo in combos:
        cfg          = dict(zip(keys, combo))
        mean, std    = cross_val_score_config(X, y, cfg, n_classes)
        results.append((mean, std, cfg))

    # Sort: highest mean first; break ties by lowest std
    results.sort(key=lambda x: (-x[0], x[1]))
    best_mean, best_std, best_cfg = results[0]
    print(f"\n✓ Best config (macro-F1 = {best_mean:.4f} ± {best_std:.4f}):\n  {best_cfg}")
    return best_cfg


def train_fold_ensemble(X, y, config, n_classes):
    """
    Trains N_CV_FOLDS models with best-epoch checkpointing.
    Returns list of (model, scaler) pairs, one per fold.
    Also returns (all_preds, all_labels) for a CV accuracy estimate.
    """
    skf         = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_models = []
    all_preds, all_labels = [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        acc, best_state, scaler = train_fold(
            X[tr_idx], y[tr_idx], X[val_idx], y[val_idx], config, n_classes
        )
        print(f"  Fold {fold+1}/{N_CV_FOLDS}  best val acc = {acc:.4f}")

        # Rebuild model and load best weights
        scaler_tr = StandardScaler().fit(X[tr_idx])
        model = BirdClassifierMLP(
            scaler_tr.transform(X[tr_idx]).shape[1],
            config["hidden_dims"], n_classes, config["dropout"]
        ).to(device)
        model.load_state_dict(best_state)
        model.eval()
        fold_models.append((model, scaler_tr))

        # Collect val predictions for the final CV report
        X_val_scaled = scaler_tr.transform(X[val_idx])
        val_loader   = DataLoader(
            TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32),
                          torch.tensor(y[val_idx], dtype=torch.long)),
            batch_size=64, shuffle=False,
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
        _, _, preds, labels, _ = eval_epoch(model, val_loader, criterion)
        all_preds.extend(preds)
        all_labels.extend(labels)

    return fold_models, np.array(all_preds), np.array(all_labels)


ENSEMBLE_TEMPERATURE = 0.7 

def ensemble_predict(fold_models, X_test):
    """
    Averages temperature-scaled softmax probabilities across all fold models.
    Temperature < 1.0 sharpens the distribution, reducing the chance that a
    hard-pair confusion (e.g. 8↔9) is selected by a slim margin.
    """
    all_probs = []
    for model, scaler in fold_models:
        X_scaled = scaler.transform(X_test)
        loader   = DataLoader(
            TensorDataset(torch.tensor(X_scaled, dtype=torch.float32)),
            batch_size=64, shuffle=False,
        )
        model.eval()
        probs = []
        with torch.no_grad():
            for (X_batch,) in loader:
                logits = model(X_batch.to(device))
                # Apply temperature scaling before softmax
                scaled_probs = torch.softmax(logits / ENSEMBLE_TEMPERATURE, dim=1)
                probs.append(scaled_probs.cpu().numpy())
        all_probs.append(np.concatenate(probs, axis=0))

    mean_probs = np.mean(all_probs, axis=0)
    return mean_probs.argmax(axis=1)



def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.tight_layout()
    plt.savefig("task2_confusion_matrix.png", dpi=150)
    print("Saved confusion matrix → task2_confusion_matrix.png")
    plt.show()


def print_per_class_accuracy(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    print("\nPer-class accuracy:")
    for i, name in enumerate(class_names):
        row_sum = cm[i].sum()
        acc     = cm[i, i] / row_sum if row_sum > 0 else 0.0
        print(f"  {str(name):30s}  {acc:.3f}  ({cm[i,i]}/{row_sum})")



def main():
    # ── Load metadata ──
    train_meta = pd.read_csv(TRAIN_META)
    test_meta  = pd.read_csv(TEST_META)

    # ── ResNet features — no-op if files already exist ──
    extract_resnet_features(train_meta, TASK2_DIR, RESNET_FEATURES_TRAIN)
    extract_resnet_features(test_meta,  TASK2_DIR, RESNET_FEATURES_TEST)

    # ── Assemble feature matrices ──
    X_train, y_train, train_ids, le = load_and_merge_features(
        TRAIN_META, RESNET_FEATURES_TRAIN,
        has_labels=True,
    )
    X_test, _, test_ids, _ = load_and_merge_features(
        TEST_META, RESNET_FEATURES_TEST,
        has_labels=False,
    )

    n_classes   = len(np.unique(y_train))
    class_names = le.classes_
    print(f"\nFeature matrix shape — train: {X_train.shape}, test: {X_test.shape}")

    # ── Hyperparameter search ──
    print("\n=== Hyperparameter search (grid search + 5-fold CV) ===")
    best_cfg = hyperparameter_search(X_train, y_train, n_classes)

    # ── Train fold ensemble with best config ──
    print(f"\n=== Training fold ensemble ({N_CV_FOLDS} models) ===")
    fold_models, cv_preds, cv_labels = train_fold_ensemble(X_train, y_train, best_cfg, n_classes)

    # ── CV report (uses ensemble fold predictions collected during training) ──
    cv_acc = accuracy_score(cv_labels, cv_preds)
    cv_f1  = f1_score(cv_labels, cv_preds, average="macro", zero_division=0)
    print(f"\n✓ Ensemble CV accuracy: {cv_acc:.4f}  |  macro-F1: {cv_f1:.4f}")
    print_per_class_accuracy(cv_labels, cv_preds, class_names)
    plot_confusion_matrix(cv_labels, cv_preds, class_names,
                          title=f"Task 2 Ensemble CV Confusion Matrix (acc={cv_acc:.3f}, F1={cv_f1:.3f})")

    # ── Predict on test set using ensemble soft-voting ──
    print("\nGenerating test predictions via ensemble soft-voting …")
    test_preds     = ensemble_predict(fold_models, X_test)
    pred_class_ids = le.inverse_transform(test_preds)

    # ── Save Kaggle submission ──
    submission = pd.DataFrame({
        "image_id": test_ids,
        "class_id": pred_class_ids,
    })
    submission.to_csv(KAGGLE_OUTPUT, index=False)
    print(f"\n✓ Kaggle submission saved → {KAGGLE_OUTPUT}")
    print(submission.head())


if __name__ == "__main__":
    main()