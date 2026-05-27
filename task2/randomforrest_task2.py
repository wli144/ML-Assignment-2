"""
model_task2_random_forest.py
----------------------------
Random Forest classifier for Task 2: fine-grained bird species classification.

Task 2 differences from Task 1
--------------------------------
Task 1 separates 10 visually distinct animal classes (spider vs. butterfly vs.
squirrel, etc.). Inter-class differences are large — colour and HOG alone
are sufficient discriminators for several class pairs.

Task 2 classifies 10 bird species. The inter-class visual differences are
substantially smaller: all classes share a roughly similar body silhouette,
similar colour distributions (many birds are brown/grey), and similar textures.
HOG-PCA and colour histograms lose much of their discriminative power here
because they are not fine-grained enough to capture subtle differences in
beak shape, plumage pattern, or wing bar configuration.

What transfers from Task 1
--------------------------
- The pipeline structure (data_loader, feature_engineering) transfers directly.
- Engineered features (colour ratios, HOG stats, histogram stats) still add
  marginal signal, but their contribution shrinks relative to Task 1.
- Feature selection (SelectKBest) is still valuable for reducing noise.

What does not transfer
----------------------
- Low-level HOG/colour features are insufficient for fine-grained classification.
  A crow and a raven have near-identical colour histograms and HOG profiles.
- The same hyperparameter ranges that worked for Task 1 are unlikely to be
  optimal here; the boundary structure is fundamentally different.

Why ResNet features are essential for Task 2
--------------------------------------------
ResNet-50 (pretrained on ImageNet) learns hierarchical feature representations:
early layers capture edges and colours (similar to HOG/histogram), but deeper
layers capture part-level patterns — feathers, eye rings, wing bars, beak
shapes. These are exactly the cues needed for fine-grained species classification.
The 2048-d penultimate layer activations encode a rich, semantically meaningful
representation of each image.

Why Random Forest for Task 2
-----------------------------
1. Robustness to irrelevant features: with 2048+ ResNet dimensions, many
   channels are class-irrelevant. Random Forest's random feature subsampling
   at each split (max_features="sqrt") provides implicit feature selection,
   making it more robust than a single Decision Tree.

2. Variance reduction through bagging: fine-grained classification boundaries
   are complex and a single tree overfits badly. Averaging across 200–500
   trees trained on bootstrap samples dramatically reduces variance.

3. Axis-aligned splits on raw ResNet: ResNet activation channels correspond
   to specific learned filters. Trees can identify the subset of channels
   that fire differently for (e.g.) a goldfinch vs. a house sparrow.
   This is why use_pca=False is recommended here — PCA rotations destroy the
   channel-level interpretability that makes Random Forest splits meaningful.

4. No scaling required: trees are threshold-based and scale-invariant.

5. Feature importance: Random Forest provides permutation/Gini importances,
   which are useful for understanding which ResNet channels (or hand-crafted
   features) drive species predictions.

Suggested models for Task 2 (for report discussion)
-----------------------------------------------------
1. Random Forest (this file) — strong baseline with ResNet; interpretable
2. Gradient Boosting (XGBoost / LightGBM) — sequential correction of errors;
   typically outperforms Random Forest on tabular features at the cost of
   longer training and more hyperparameters
3. SVM with RBF kernel + ResNet-PCA — SVMs generalise well in high-dimensional
   spaces; ResNet-PCA reduces to ~200d first; suitable when training set is small
4. k-NN with cosine distance on ResNet features — ResNet embedding space is
   approximately metric; cosine similarity is more appropriate than Euclidean
   for high-dimensional dense activations
5. MLP (sklearn MLPClassifier) — can learn non-linear combinations of ResNet
   channels; more flexible than SVM but needs careful regularisation
6. Soft Voting Ensemble (RF + SVM + kNN) — averages class probability
   estimates; often outperforms any single model when base models have
   complementary error patterns
7. Stacking Ensemble — trains a meta-classifier on out-of-fold predictions
   from base classifiers; more powerful than voting but risks overfitting
   on small datasets
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from scipy.stats import randint

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_raw_data, get_datasets, Preprocessor
from feature_engineering import make_resnet_selector


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR        = "task2_data"
VAL_SIZE        = 0.2
RANDOM_STATE    = 42
SUBMISSION_FILE = "submission_results/task2_random_forest_submission.csv"
CONFUSION_PLOT  = "submission_results/task2_rf_confusion_matrix.png"

# ResNet CSV paths — use the same files for both tasks since they cover all images
RESNET_TRAIN_CSV = "resnet_features_train2.csv"
RESNET_TEST_CSV  = "resnet_features_test2.csv"

# Feature pipeline
# use_pca=False: preserve raw ResNet channel axes for tree splits.
# selection_k=300: retain top-300 features from ~2267+engineering dims.
#   This is higher than Task 1's 150 because fine-grained classification needs
#   more channels to distinguish subtle inter-species differences.
# selection_method="anova": fast; mutual_info is more accurate but ~10× slower
#   on 2267 features. Switch to "mutual_info" if runtime permits.
ACTIVE_SELECTOR = make_resnet_selector(
    resnet_train_csv=RESNET_TRAIN_CSV,
    resnet_test_csv=RESNET_TEST_CSV,
    use_pca=False,              # keep raw ResNet axes for tree splits
    selection_k=300,            # wider net than Task 1 for fine-grained classes
    selection_method="anova",
)

# Trees are scale-invariant; scaler has no effect on RF behaviour.
# Kept for pipeline consistency. Set scaler_type=None to skip explicitly.
PREPROCESSOR = Preprocessor(
    scaler_type=None,
    pca_components=None,
)

# RandomizedSearchCV is used instead of GridSearchCV because the Random Forest
# parameter space is large. Sampling 60 random combinations across 5-fold CV
# covers the space more efficiently than an exhaustive grid.
#
# Key parameters:
#   n_estimators     : more trees = lower variance; diminishing returns above ~500
#   max_depth        : None = fully grown (RF relies on bagging, not pruning)
#   min_samples_leaf : smooths decision boundaries; important for fine-grained tasks
#   max_features     : "sqrt" is the RF default; "log2" for higher-dimensional tasks
#   class_weight     : "balanced" compensates for any class imbalance in bird data
PARAM_DIST = {
    "n_estimators":      randint(100, 600),
    "max_depth":         [None, 20, 40],
    "min_samples_split": randint(2, 12),
    "min_samples_leaf":  randint(1, 8),
    "max_features":      ["sqrt", "log2"],
    "class_weight":      [None, "balanced"],
}

N_ITER_SEARCH = 60   # number of random combinations to evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_confusion_matrix(y_true, y_pred, class_names, output_path):
    """Save a normalised confusion matrix plot."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=class_names,
        normalize="true",
        xticks_rotation="vertical",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Random Forest — Task 2 Validation Confusion Matrix (normalised)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    os.makedirs(os.path.dirname(SUBMISSION_FILE), exist_ok=True)

    # 1. Load CSVs once
    print("Loading data...")
    raw = load_raw_data(DATA_DIR)

    # 2. Build feature matrices, split, and preprocess
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

    # 3. Randomised hyperparameter search on train split
    print(f"\nRunning RandomizedSearchCV ({N_ITER_SEARCH} iterations, 5-fold CV)...")
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        PARAM_DIST,
        n_iter=N_ITER_SEARCH,
        cv=5,
        scoring="accuracy",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best_params = search.best_params_
    print(f"\nBest parameters : {best_params}")
    print(f"Best CV accuracy : {search.best_score_ * 100:.2f}%")

    # 4. Evaluate on validation split
    best_model = search.best_estimator_
    y_val_pred = best_model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"\nValidation Accuracy : {val_acc * 100:.2f}%")
    print(classification_report(y_val, y_val_pred))

    # 5. Confusion matrix — identify which bird species are confused
    # Retrieve class names from training metadata if available
    class_names = None
    if "class_name" in raw["train_meta"].columns:
        label_map = (raw["train_meta"]
                     .drop_duplicates("class_id")
                     .set_index("class_id")["class_name"]
                     .to_dict())
        unique_labels = sorted(np.unique(y_val))
        class_names = [label_map.get(l, str(l)) for l in unique_labels]

    _save_confusion_matrix(y_val, y_val_pred, class_names, CONFUSION_PLOT)

    # 6. Feature importances — top-30 most discriminative features
    importances = best_model.feature_importances_
    feat_cols   = datasets["feature_cols"]
    top_idx     = np.argsort(importances)[::-1][:30]
    print("\nTop-30 features by Gini importance:")
    for rank, idx in enumerate(top_idx, 1):
        col = feat_cols[idx] if idx < len(feat_cols) else f"feature_{idx}"
        print(f"  {rank:2d}. {col:40s}  {importances[idx]:.4f}")

    # 7. Refit on full training data and generate test predictions
    print("\nRefitting on full training data...")
    final_model = RandomForestClassifier(
        **best_params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    final_model.fit(X_train_full, y_train_full)

    y_test_pred = final_model.predict(X_test)
    submission  = pd.DataFrame({"image_id": image_ids, "class_id": y_test_pred})
    submission.to_csv(SUBMISSION_FILE, index=False)
    print(f"Submission saved to {SUBMISSION_FILE}")


if __name__ == "__main__":
    run()
