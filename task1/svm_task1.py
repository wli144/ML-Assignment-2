"""
svm_task1.py
------------
SVM classifier for Task 1 — runs both Linear and RBF configurations,
reports each independently, and writes separate submission files.

Depends on:
  data_loader.py          - data loading, merging, splitting, preprocessing
  feature_engineering.py  - feature selectors / engineering pipelines

SVM is sensitive to feature scale; StandardScaler is mandatory here.

ResNet features
---------------
Training features : resnet_features_hf.csv   (produced by resnet_feature_creation.py
                    run against train_metadata.csv)
Test features     : resnet_features_test.csv  (same script, run against
                    test_metadata.csv)

How the stateful selector works
--------------------------------
add_resnet_pca() inside the selector pipeline holds a closure with a scaler
and PCA object.  get_datasets() always calls the selector on training data
first (is_train=True), which fits those objects.  When the selector is then
called on test data (is_train=False) it:
  - detects that resnet_* columns are absent from the test DataFrame
  - loads resnet_features_test.csv and merges it on image_path
  - applies (does NOT refit) the already-fitted scaler + PCA
This produces a test matrix with the same column count as the training matrix.
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

from data_loader import load_raw_data, get_datasets, Preprocessor
from feature_engineering import combined_selector, make_resnet_selector


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR     = "task1_data"
VAL_SIZE     = 0.2
RANDOM_STATE = 42

# Paths to ResNet feature CSVs — adjust if they live in a subdirectory.
RESNET_TRAIN_CSV = "resnet_features_hf.csv"
RESNET_TEST_CSV  = "resnet_features_test.csv"

# Build the combined selector once with both CSV paths.
# Swap ACTIVE_SELECTOR to `combined_selector` to run without ResNet features.
resnet_combined_selector = make_resnet_selector(
    train_csv=RESNET_TRAIN_CSV,
    test_csv=RESNET_TEST_CSV,
    n_components=0.95,
    k_best=200,
    include_handcrafted=True,
)

ACTIVE_SELECTOR = resnet_combined_selector   # swap to combined_selector to ablate

# StandardScaler is required for SVM — do not set scaler_type=None here.
PREPROCESSOR = Preprocessor(
    scaler_type="standard",
    pca_components=None,
)

# Each entry: (config_name, kernel, param_grid, submission_filename)
SVM_CONFIGURATIONS = [
    (
        "Linear",
        "linear",
        {"C": [0.01, 0.1, 1, 10, 100]},
        "svm_linear_submission.csv",
    ),
    (
        "RBF",
        "rbf",
        {
            "C":     [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.001, 0.01],
        },
        "svm_rbf_submission.csv",
    ),
]


# ---------------------------------------------------------------------------
# Single-configuration runner
# ---------------------------------------------------------------------------

def run_configuration(name, kernel, param_grid,
                      X_train, y_train, X_val, y_val,
                      X_train_full, y_train_full, X_test, image_ids,
                      submission_file, random_state):

    print(f"\n{'='*55}")
    print(f"SVM [{name}] — kernel='{kernel}'")
    print(f"{'='*55}")

    grid_search = GridSearchCV(
        SVC(kernel=kernel, random_state=random_state),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print(f"Best parameters : {best_params}")

    best_model = grid_search.best_estimator_
    y_val_pred = best_model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy : {val_acc * 100:.2f}%")
    print(classification_report(y_val, y_val_pred))

    # Refit on full training data for the final submission predictions
    print("Refitting on full training data...")
    final_model = SVC(kernel=kernel, random_state=random_state, **best_params)
    final_model.fit(X_train_full, y_train_full)

    y_test_pred = final_model.predict(X_test)
    submission  = pd.DataFrame({"image_id": image_ids, "class_id": y_test_pred})
    submission.to_csv(submission_file, index=False)
    print(f"Submission saved to {submission_file}")

    return val_acc, best_params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    # 1. Load CSVs once
    raw = load_raw_data(DATA_DIR)

    # 2. Build feature matrices, split, and preprocess.
    #    get_datasets() always builds the training matrix first so the stateful
    #    selector is fitted before it is applied to the test set.
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

    # 3. Run each SVM configuration
    results = {}
    for name, kernel, param_grid, submission_file in SVM_CONFIGURATIONS:
        val_acc, best_params = run_configuration(
            name=name,
            kernel=kernel,
            param_grid=param_grid,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_train_full=X_train_full,
            y_train_full=y_train_full,
            X_test=X_test,
            image_ids=image_ids,
            submission_file=submission_file,
            random_state=RANDOM_STATE,
        )
        results[name] = {"val_accuracy": val_acc, "best_params": best_params}

    # 4. Summary
    print(f"\n{'='*55}")
    print("SVM Configuration Summary")
    print(f"{'='*55}")
    for name, info in results.items():
        print(f"  {name:10s} — Val Accuracy: {info['val_accuracy'] * 100:.2f}%  "
              f"| Params: {info['best_params']}")


if __name__ == "__main__":
    run()