from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import numpy as np

from data_loader import load_raw_data, get_datasets, Preprocessor
from feature_engineering import build_selector
from _model_utils import evaluate, save_submission

DATA_DIR         = "task2_data"
RESNET_TRAIN_CSV = "resnet_features_train2.csv"
RESNET_TEST_CSV  = "resnet_features_test2.csv"
SUBMISSION_FILE  = "randomforest_task2.csv"
RANDOM_STATE     = 42
N_ITER           = 60

# Combined raw ResNet + handcrafted, no PCA. Random Forest's random feature
# subsampling at each split (max_features="sqrt") provides implicit selection,
# making it robust to the 2048 ResNet dimensions. No PCA preserves axis-aligned
# splits on individual ResNet filter channels.
SELECTOR = build_selector(
    mode="combined",
    resnet_train_csv=RESNET_TRAIN_CSV,
    resnet_test_csv=RESNET_TEST_CSV,
    use_pca=False,
    selection_k=300,
)

PREPROCESSOR = Preprocessor(scaler_type=None)

PARAM_DIST = {
    "n_estimators":      randint(100, 600),
    "max_depth":         [None, 20, 40],
    "min_samples_split": randint(2, 12),
    "min_samples_leaf":  randint(1, 8),
    "max_features":      ["sqrt", "log2"],
    "class_weight":      [None, "balanced"],
}


def run():
    raw = load_raw_data(DATA_DIR)
    ds  = get_datasets(raw, feature_selector=SELECTOR, preprocessor=PREPROCESSOR,
                       random_state=RANDOM_STATE)

    print(f"\nTrain: {ds['X_train'].shape} | Val: {ds['X_val'].shape} | Test: {ds['X_test'].shape}")

    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        PARAM_DIST, n_iter=N_ITER, cv=5, scoring="accuracy",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1,
    )
    search.fit(ds["X_train"], ds["y_train"])
    print(f"Best params: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_*100:.2f}%")

    y_val_pred = search.best_estimator_.predict(ds["X_val"])
    evaluate("Random Forest Task2", ds["y_val"], y_val_pred, raw["train_meta"])

    importances = search.best_estimator_.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]
    print("\nTop-20 features by Gini importance:")
    for rank, idx in enumerate(top_idx, 1):
        col = ds["feature_cols"][idx] if idx < len(ds["feature_cols"]) else f"feature_{idx}"
        print(f"  {rank:2d}. {col:40s}  {importances[idx]:.4f}")

    final = RandomForestClassifier(**search.best_params_, random_state=RANDOM_STATE, n_jobs=-1)
    final.fit(ds["X_train_full"], ds["y_train_full"])
    save_submission(ds["image_ids"], final.predict(ds["X_test"]), SUBMISSION_FILE)


if __name__ == "__main__":
    run()
