"""
model_decision_tree.py
----------------------
Decision Tree classifier for Task 1.

Depends on:
  data_loader.py          - data loading, merging, splitting, preprocessing
  feature_engineering.py  - feature selectors / engineering pipelines

Note on scaling: Decision trees are scale-invariant (splits on individual
feature thresholds are unaffected by monotonic transformations). Scaling is
still applied here for pipeline consistency — it does not affect tree behaviour.
"""
import os

print("SCRIPT DIR:", os.path.dirname(__file__))
print("FILES:", os.listdir(os.path.dirname(__file__)))

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

from data_loader import load_raw_data, get_datasets, Preprocessor
from feature_engineering import combined_selector   # swap this to experiment


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR        = "task1_data"
VAL_SIZE        = 0.2
RANDOM_STATE    = 42
SUBMISSION_FILE = "decision_tree_submission.csv"

# Decision trees do not require scaling, but standard preprocessing is applied
# for consistency. Set scaler_type=None to skip scaling explicitly.
PREPROCESSOR = Preprocessor(
    scaler_type="standard",
    pca_components=None,
)

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

    # 2. Build feature matrices, split, and preprocess
    datasets = get_datasets(
        raw,
        feature_selector=combined_selector,
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