"""
model_task2_svm.py
------------------
SVM classifier for Task 2: fine-grained bird species classification.

Kernel choice: RBF over Linear — rationale
-------------------------------------------
For Task 1, both kernels were run and compared because it was genuinely
unclear whether the 10 animal classes were linearly separable in the
provided feature space. For Task 2, the choice can be argued on principled
grounds before running a single experiment.

The argument for RBF:

  1. ResNet embedding geometry. ResNet-50's penultimate layer produces
     activations that are the output of a ReLU nonlinearity. The resulting
     vectors are non-negative and sparse, and the decision boundaries
     separating bird species in this space are empirically known to be
     non-linear — this is the entire motivation for fine-tuning or training
     a classifier on top of frozen ResNet features rather than using raw
     pixels. A linear classifier in ResNet space is equivalent to a linear
     probe, which provides a lower bound on performance. The RBF kernel
     implicitly maps to an infinite-dimensional space where non-linear
     boundaries become hyperplanes, consistently outperforming linear probes
     on transfer learning benchmarks.

  2. Fine-grained class structure. In Task 1, inter-class differences are
     large (a spider vs. a butterfly). A linear boundary in feature space
     is plausible. In Task 2, the 10 bird species occupy overlapping,
     irregular regions of the ResNet activation space — subtle differences
     in beak activation channels, eye ring channels, and wing bar channels
     produce complex, curved boundaries that a linear kernel cannot represent.

  3. PCA pre-compression changes the geometry. After PCA reduces 2048 dims
     to ~200, the components are orthogonal and ranked by variance. The
     class-discriminative signal is spread across many components in a curved
     manifold. This is a setting where RBF has well-documented advantages.

The argument for Linear (and why it loses here):

  Linear SVM has one key advantage: it scales to very high dimensions without
  blowing up. In the original 2048-d ResNet space, LinearSVC (which uses a
  more efficient solver) is competitive with RBF-SVM because the curse of
  dimensionality can actually help linear classifiers — in very high dimensions,
  classes become approximately linearly separable (Cover's theorem). However,
  once we apply PCA to ~200 dimensions, this advantage disappears. In the
  compressed space, the linear kernel is simply less expressive.

  The practical conclusion: if you were running SVM directly on raw 2048-d
  ResNet features without PCA, a LinearSVC would be a strong and fast
  competitor. With PCA pre-compression (which is necessary for RBF-SVM's
  runtime), RBF is the better choice.

Why PCA is mandatory before RBF-SVM on ResNet features
-------------------------------------------------------
RBF-SVM with kernel='rbf' builds a kernel matrix K of shape (n_train, n_train).
With n_train ~ 3000 and d = 2048, each kernel evaluation computes
exp(-gamma * ||x_i - x_j||^2) over 2048 dimensions. The kernel matrix itself
is 3000 × 3000 = 9M values, which is manageable, but:

  - The ||x_i - x_j||^2 computation over 2048 dimensions per pair is expensive
  - Hyperparameter search (GridSearchCV, 5-fold) multiplies this cost by
    n_param_combinations × 5
  - gamma='scale' computes 1 / (n_features * X.var()), which becomes very
    small at d=2048, making the RBF kernel nearly flat and ineffective

PCA to ~200 components solves all three issues:
  - Reduces per-pair distance computation by 10×
  - Retains the directions of maximum variance (which encode the most
    class-discriminative information)
  - Makes gamma='scale' numerically sensible

In practice, 150–250 PCA components from ResNet-50 typically retain 85–95%
of the variance, with negligible accuracy loss relative to the full 2048-d space.

What transfers from Task 1 SVM and what changes
------------------------------------------------
Transfers:
  - StandardScaler is still mandatory (SVM is distance-based)
  - GridSearchCV over C and gamma
  - The run_configuration() pattern
  - Refit on full training data before generating test predictions

Changes:
  - Only RBF kernel is run (linear is excluded with documented rationale)
  - ResNet-PCA features replace the hand-crafted-only feature set
  - PCA compression step added inside the feature pipeline (use_pca=True)
  - C search range shifts higher: in high-dimensional PCA space the optimal
    C tends to be larger (less regularisation needed) because the feature
    representation is already compact and informative
  - gamma grid is narrowed: 'scale' is almost always optimal after PCA
    normalisation, but 'auto' and one manual value are retained for safety
  - Confusion matrix saved for species-level error analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)

from data_loader import load_raw_data, get_datasets, Preprocessor
from feature_engineering import make_resnet_selector


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR        = "task2_data"
VAL_SIZE        = 0.2
RANDOM_STATE    = 42
SUBMISSION_FILE = "submission_results/task2_svm_rbf_submission.csv"
CONFUSION_PLOT  = "submission_results/task2_svm_rbf_confusion_matrix.png"

RESNET_TRAIN_CSV = "resnet_features_train2.csv"
RESNET_TEST_CSV  = "resnet_features_test2.csv"

# Feature pipeline
# use_pca=True: compress 2048-d ResNet to pca_k dimensions before SVM.
#   This is mandatory for RBF-SVM runtime (see module docstring).
# pca_k=200: retains ~90% of ResNet variance empirically; adjust downward
#   (e.g. 100) to trade accuracy for speed during initial experiments.
# selection_k=150: ANOVA top-150 from the ~436-feature post-PCA space
#   (200 ResNet-PCA + ~236 hand-crafted + engineered).
#   Removes low-signal hand-crafted features that add noise rather than
#   discriminative power for fine-grained classification.
# selection_method="anova": fast; switch to "mutual_info" for a more
#   accurate selection at the cost of longer preprocessing.
ACTIVE_SELECTOR = make_resnet_selector(
    resnet_train_csv=RESNET_TRAIN_CSV,
    resnet_test_csv=RESNET_TEST_CSV,
    use_pca=True,
    pca_k=200,
    selection_k=150,
    selection_method="anova",
)

# StandardScaler is mandatory for SVM.
# RobustScaler is an alternative if ResNet activations contain extreme outliers
# (post-ReLU activations can have long right tails). Swap if val accuracy
# is lower than expected.
PREPROCESSOR = Preprocessor(
    scaler_type="standard",
    pca_components=None,   # PCA is handled inside ACTIVE_SELECTOR, not here
)

# Hyperparameter grid (RBF only)
#
# C controls the margin softness:
#   - Low C (0.1, 1): wider margin, more misclassifications tolerated.
#     Appropriate when classes overlap substantially in PCA space.
#   - High C (10, 100): narrow margin, fits training data more tightly.
#     Risk of overfitting on fine-grained boundaries.
#   Range starts at 1 (not 0.01 as in Task 1) because in PCA-compressed
#   ResNet space the features are already compact and informative; heavy
#   regularisation is less necessary.
#
# gamma controls the RBF kernel bandwidth:
#   - "scale": 1 / (n_features * X.var()) — the sklearn default; almost
#     always optimal after StandardScaler + PCA.
#   - "auto":  1 / n_features — slightly larger bandwidth.
#   - 0.001:   small manual value; makes kernel very smooth (high bias).
#   The grid is deliberately narrow because 'scale' dominates empirically
#   after preprocessing. Expand if CV results show 'scale' is not selected.
PARAM_GRID = {
    "C":     [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.001],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_confusion_matrix(y_true, y_pred, class_names, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=class_names,
        normalize="true",
        xticks_rotation="vertical",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("SVM (RBF) — Task 2 Validation Confusion Matrix (normalised)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved to {output_path}")


def _get_class_names(train_meta, unique_labels):
    """Map integer class_ids to class_name strings from training metadata."""
    if "class_name" not in train_meta.columns:
        return [str(l) for l in unique_labels]
    label_map = (train_meta
                 .drop_duplicates("class_id")
                 .set_index("class_id")["class_name"]
                 .to_dict())
    return [label_map.get(l, str(l)) for l in unique_labels]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    os.makedirs(os.path.dirname(SUBMISSION_FILE), exist_ok=True)

    # 1. Load CSVs once
    print("Loading data...")
    raw = load_raw_data(DATA_DIR)

    # 2. Build feature matrices, split, and preprocess
    #    The feature pipeline: all 219 hand-crafted features + engineered
    #    additions -> merge ResNet -> PCA(200) on ResNet -> ANOVA top-150.
    #    Preprocessor then applies StandardScaler to the 150-d output.
    datasets = get_datasets(
        raw,
        feature_selector=ACTIVE_SELECTOR,
        preprocessor=PREPROCESSOR,
        val_size=VAL_SIZE,
        random_state=RANDOM_STATE,
    )

    X_train      = datasets["X_train"]
    y_train      = datasets["y_train"]
    X_val        = datasets["X_val"]
    y_val        = datasets["y_val"]
    X_test       = datasets["X_test"]
    image_ids    = datasets["image_ids"]
    X_train_full = datasets["X_train_full"]
    y_train_full = datasets["y_train_full"]

    print(f"\nTrain : {X_train.shape} | Val : {X_val.shape} | Test : {X_test.shape}")

    # 3. GridSearchCV over C and gamma (RBF kernel only)
    print(f"\nTuning SVM [RBF]...")
    print(f"Grid: C={PARAM_GRID['C']}, gamma={PARAM_GRID['gamma']}")
    print(f"Total combinations: {len(PARAM_GRID['C']) * len(PARAM_GRID['gamma'])} × 5-fold CV\n")

    grid_search = GridSearchCV(
        SVC(kernel="rbf", random_state=RANDOM_STATE, probability=True),
        PARAM_GRID,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print(f"\nBest parameters : {best_params}")
    print(f"Best CV accuracy : {grid_search.best_score_ * 100:.2f}%")

    # 4. Evaluate on validation split
    best_model = grid_search.best_estimator_
    y_val_pred = best_model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"\nValidation Accuracy : {val_acc * 100:.2f}%")
    print(classification_report(y_val, y_val_pred))

    # 5. Confusion matrix — identify confused species pairs
    unique_labels = sorted(np.unique(y_val))
    class_names   = _get_class_names(raw["train_meta"], unique_labels)
    _save_confusion_matrix(y_val, y_val_pred, class_names, CONFUSION_PLOT)

    # 6. Refit on full training data and generate test predictions
    print("\nRefitting on full training data...")
    final_model = SVC(
        kernel="rbf",
        random_state=RANDOM_STATE,
        probability=True,
        **best_params,
    )
    final_model.fit(X_train_full, y_train_full)

    y_test_pred = final_model.predict(X_test)
    submission  = pd.DataFrame({"image_id": image_ids, "class_id": y_test_pred})
    submission.to_csv(SUBMISSION_FILE, index=False)
    print(f"Submission saved to {SUBMISSION_FILE}")

    # 7. Print a concise summary for the report
    print(f"\n{'='*55}")
    print("Task 2 SVM Summary")
    print(f"{'='*55}")
    print(f"  Kernel         : RBF")
    print(f"  Feature input  : ResNet-PCA ({ACTIVE_SELECTOR}) + hand-crafted")
    print(f"  Final features : {X_train.shape[1]}")
    print(f"  Best C         : {best_params['C']}")
    print(f"  Best gamma     : {best_params['gamma']}")
    print(f"  Val accuracy   : {val_acc * 100:.2f}%")


if __name__ == "__main__":
    run()