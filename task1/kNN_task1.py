"""
model_knn.py
------------
k-Nearest Neighbours classifier for Task 1.

Depends on:
  data_feature_pre.data_loader.py          - data loading, merging, splitting, preprocessing
  data_feature_pre.feature_engineering.py  - feature selectors / engineering pipelines

To change the feature set, swap out the `feature_selector` argument in
get_datasets(). All available selectors are defined in feature_engineering.py.
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

from data_loader import load_raw_data, get_datasets, Preprocessor
from feature_engineering import combined_selector 
from feature_engineering import resnet_combined_selector   # swap this to experiment



# ---------------------------------------------------------------------------
# Configuration — change these to experiment
# ---------------------------------------------------------------------------

DATA_DIR          = "task1_data"
VAL_SIZE          = 0.2
RANDOM_STATE      = 42
SUBMISSION_FILE   = "knn_submission.csv"

# Preprocessor: StandardScaler is required for kNN (distance-based).
# pca_components=None keeps the full feature set after selection;
# set to e.g. 0.95 to retain 95% variance via PCA.
PREPROCESSOR = Preprocessor(
    scaler_type="standard",
    pca_components=None,
)

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

    # 2. Build feature matrices, split, and preprocess
    datasets = get_datasets(
        raw,
        feature_selector=   resnet_combined_selector,   # swap this to experiment
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