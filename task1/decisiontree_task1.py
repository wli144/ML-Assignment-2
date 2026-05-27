"""
decisiontree_task1.py
---------------------
Decision Tree classifier for Task 1.

Depends on:
  data_loader.py          - data loading, merging, splitting, preprocessing
  feature_engineering.py  - feature selectors / engineering pipelines

Feature selection rationale for Decision Trees
------------------------------------------------
Decision trees are scale-invariant and axis-aligned: each split tests a single
feature against a threshold.  This has two consequences for feature selection:

  1. PCA-compressed features are LESS useful here than for kNN/SVM.
     PCA rotates the feature space into principal components — axes that are
     linear combinations of the originals.  A tree split on a PCA component
     has no semantic meaning, and the tree loses the ability to isolate a
     single discriminative original dimension cleanly.

  2. Original ResNet dimensions (pre-PCA) are preferable.  Each of the 2,048
     raw ResNet activation values corresponds to a specific learned filter
     response.  Trees can isolate the most discriminative individual channels.

Strategy: ResNet raw features + hand-crafted features, reduced with ANOVA
SelectKBest (resnet_anova_selector).
  - Raw resnet_* columns are merged (no PCA rotation).
  - All 219 hand-crafted features (HOG, colour, additional) are included first.
  - ANOVA F-test (SelectKBest) then retains the top-150 most individually
    discriminative features from the combined ~2,267-d space.
  - This gives the tree a manageable, high-signal feature set while preserving
    the axis-aligned interpretability that makes trees effective.

Note on scaling: trees are split-threshold based and are completely unaffected
by monotonic feature transformations (scaling).  StandardScaler is still applied
here for pipeline consistency, but has zero effect on tree behaviour or accuracy.
"""

import os
import sys

# --- FIX: Tell Python to look in the parent folder for imports ---
# This grabs the folder above the current script and adds it to Python's search path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

from data_loader import load_raw_data, get_datasets, Preprocessor
from feature_engineering import make_resnet_selector


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR        = "task1_data"
VAL_SIZE        = 0.2
RANDOM_STATE    = 42
SUBMISSION_FILE = "submission_results/decision_tree_submission.csv"

# ResNet CSV paths
RESNET_TRAIN_CSV = "resnet_features_hf.csv"
RESNET_TEST_CSV  = "resnet_features_test.csv"

# Feature selector: ResNet raw features (no PCA) + hand-crafted features,
# filtered to top-150 by ANOVA F-score.
# Rationale: trees benefit from original feature axes rather than PCA rotations.
# include_handcrafted=True adds HOG/colour/additional before ANOVA selection.
# Swap include_handcrafted=False to ablate the hand-crafted contribution.
#
# Note: make_resnet_selector with include_handcrafted=True uses add_resnet_pca
# internally. To keep raw ResNet features for the tree (no PCA), we build the
# selector manually here using compose + merge_resnet_features + select_k_best_anova.
from feature_engineering import (
    compose,
    all_features,
    add_color_channel_ratios,
    add_color_histogram_stats,
    add_hog_statistics,
    add_feat_statistics,
    merge_resnet_features,
    select_k_best_anova,
)

def _merge_raw_resnet(train_csv=RESNET_TRAIN_CSV, test_csv=RESNET_TEST_CSV):
    """
    Stateless wrapper that merges raw ResNet features (no PCA) using the
    correct CSV for train vs test, matching the is_train flag convention.
    """
    def _apply(merged_df, feature_cols, is_train=True):
        csv = train_csv if is_train else test_csv
        resnet_cols_present = [c for c in feature_cols if c.startswith("resnet_")]
        if not resnet_cols_present:
            print(f"  [merge_raw_resnet] Merging raw ResNet features from '{csv}'")
            merged_df, feature_cols = merge_resnet_features(
                merged_df, feature_cols, resnet_csv=csv
            )
        return merged_df, feature_cols
    return _apply

ACTIVE_SELECTOR = compose(
    all_features,                   # keep all 219 hand-crafted features
    add_color_channel_ratios,       # engineered: R/G, R/B, G/B ratios
    add_color_histogram_stats,      # engineered: mean/std/skew per colour channel
    add_hog_statistics,             # engineered: mean/std/energy of HOG features
    add_feat_statistics,            # engineered: stats on additional_features cols
    _merge_raw_resnet(              # merge raw 2048-d ResNet (no PCA rotation)
        train_csv=RESNET_TRAIN_CSV,
        test_csv=RESNET_TEST_CSV,
    ),
    select_k_best_anova(k=150),     # ANOVA F-test: top-150 most discriminative dims
)

# Scaling has no effect on tree splits but is kept for pipeline consistency.
PREPROCESSOR = Preprocessor(
    scaler_type="standard",
    pca_components=None,
)

# Hyperparameter grid
# criterion:         gini (default, faster) vs entropy (information gain)
# max_depth:         None = fully grown; limiting depth controls overfitting
# min_samples_split: minimum samples to attempt a split (higher = smoother tree)
# min_samples_leaf:  minimum samples in a leaf (higher = more regularisation)
# max_features:      features considered per split; sqrt/log2 add randomness
#                    similar to Random Forest — helps generalisation
PARAM_GRID = {
    "criterion":         ["gini", "entropy"],
    "max_depth":         [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      [None, "sqrt", "log2"],
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    # 1. Load CSVs once
    raw = load_raw_data(DATA_DIR)

    # 2. Build feature matrices, split, and preprocess.
    #    Training matrix is built first so the stateful SelectKBest selector
    #    is fitted before it is applied to the test set.
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

    print(f"\nTrain split : {X_train.shape}  |  Val : {X_val.shape}  |  Test : {X_test.shape}")

    # 3. Hyperparameter search on train split
    print("\nTuning Decision Tree...")
    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        PARAM_GRID,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print(f"Best parameters : {best_params}")

    # 4. Evaluate on validation split
    best_model = grid_search.best_estimator_
    y_val_pred = best_model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"\nValidation Accuracy : {val_acc * 100:.2f}%")
    print(classification_report(y_val, y_val_pred))

    # 5. Refit on full training data and generate test predictions
    print("Refitting on full training data...")
    final_model = DecisionTreeClassifier(**best_params, random_state=RANDOM_STATE)
    final_model.fit(X_train_full, y_train_full)

    y_test_pred = final_model.predict(X_test)
    submission  = pd.DataFrame({"image_id": image_ids, "class_id": y_test_pred})
    submission.to_csv(SUBMISSION_FILE, index=False)
    print(f"Submission saved to {SUBMISSION_FILE}")


if __name__ == "__main__":
    run()