"""
kNN_task1.py
------------
k-Nearest Neighbours classifier for Task 1.

Depends on:
  data_loader.py          - data loading, merging, splitting, preprocessing
  feature_engineering.py  - feature selectors / engineering pipelines

Feature selection rationale for kNN
-------------------------------------
kNN is a distance-based algorithm — every feature dimension contributes to the
distance calculation equally (after scaling).  This makes it uniquely sensitive
to two problems:

  1. Curse of dimensionality: in high-dimensional spaces, distance measures
     become uninformative because all points become approximately equidistant.
     Raw ResNet embeddings are 2,048-d; the provided features are 219-d.
     Combining them naively gives ~2,267 dimensions, which degrades kNN.

  2. Irrelevant / correlated features: features that carry no class signal
     still add noise to every distance computation.

Strategy: use ResNet PCA only (resnet_only_pca).
  - PCA decorrelates the embedding space, removing redundant axes.
  - 95% variance retention collapses 2,048 dims to ~250-400 orthogonal components.
  - Hand-crafted features (HOG, colour, additional) are deliberately excluded:
    they are largely redundant with the ResNet embedding and add noisy dimensions.
  - The Preprocessor StandardScaler normalises the PCA components so that no
    single component dominates the Euclidean distance.

If you want to experiment with including hand-crafted features, swap
ACTIVE_SELECTOR to resnet_anova_selector (ResNet PCA + ANOVA top-150), which
at least filters out the weakest hand-crafted dimensions before combining.
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

from task2.data_loader import load_raw_data, get_datasets, Preprocessor
from task2.feature_engineering import make_resnet_selector


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR        = "task1_data"
VAL_SIZE        = 0.2
RANDOM_STATE    = 42
SUBMISSION_FILE = "knn_submission.csv"

# ResNet CSV paths
RESNET_TRAIN_CSV = "resnet_features_hf.csv"
RESNET_TEST_CSV  = "resnet_features_test.csv"

# Feature selector: ResNet PCA only — no hand-crafted features.
# Rationale: kNN degrades with irrelevant/correlated dimensions.  PCA-compressed
# ResNet embeddings give a compact, decorrelated, semantically rich feature space
# that is ideal for distance-based methods.
# Swap to include_handcrafted=True to ablate the effect of adding hand-crafted features.
ACTIVE_SELECTOR = make_resnet_selector(
    train_csv=RESNET_TRAIN_CSV,
    test_csv=RESNET_TEST_CSV,
    n_components=0.95,        # keep 95% of ResNet embedding variance (~250-400 dims)
    k_best=200,               # only used when include_handcrafted=True
    include_handcrafted=False,
)

# StandardScaler is mandatory for kNN — distances are not scale-invariant.
# pca_components=None: dimensionality reduction is already handled by the
# feature selector's add_resnet_pca step; a second PCA here would be redundant.
PREPROCESSOR = Preprocessor(
    scaler_type="standard",
    pca_components=None,
)

# Hyperparameter grid
# n_neighbors: odd values avoid ties; range covers local (3) to more global (15)
# weights:     "distance" downweights far neighbours — tends to help on image data
# metric:      euclidean is standard in PCA space (orthonormal axes);
#              manhattan can outperform on high-d data (less sensitive to outliers)
PARAM_GRID = {
    "n_neighbors": [3, 5, 7, 11, 15],
    "weights":     ["uniform", "distance"],
    "metric":      ["euclidean", "manhattan"],
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    # 1. Load CSVs once
    raw = load_raw_data(DATA_DIR)

    # 2. Build feature matrices, split, and preprocess.
    #    Training matrix is built first so the stateful ResNet PCA selector
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
    print("\nTuning kNN...")
    grid_search = GridSearchCV(
        KNeighborsClassifier(),
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
    final_model = KNeighborsClassifier(**best_params)
    final_model.fit(X_train_full, y_train_full)

    y_test_pred = final_model.predict(X_test)
    submission  = pd.DataFrame({"image_id": image_ids, "class_id": y_test_pred})
    submission.to_csv(SUBMISSION_FILE, index=False)
    print(f"Submission saved to {SUBMISSION_FILE}")


if __name__ == "__main__":
    run()