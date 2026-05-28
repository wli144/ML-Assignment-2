from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from data_loader import load_raw_data, get_datasets, Preprocessor
from feature_engineering import build_selector
from _model_utils import evaluate, save_submission

DATA_DIR         = "task1_data"
RESNET_TRAIN_CSV = "resnet_features_train1.csv"
RESNET_TEST_CSV  = "resnet_features_test1.csv"
SUBMISSION_FILE  = "knn_task1.csv"
RANDOM_STATE     = 42

# ResNet-PCA only: kNN degrades in high dimensions, so raw 2048-d ResNet is
# compressed to ~200 orthogonal components. Handcrafted features are excluded
# because they add noisy correlated dimensions that inflate distances.
# use_pca=True is mandatory here.
SELECTOR = build_selector(
    mode="resnet",
    resnet_train_csv=RESNET_TRAIN_CSV,
    resnet_test_csv=RESNET_TEST_CSV,
    use_pca=True,
    pca_k=200,
    selection_k=200,
)

PREPROCESSOR = Preprocessor(scaler_type="standard")

PARAM_GRID = {
    "n_neighbors": [3, 5, 7, 11, 15],
    "weights":     ["uniform", "distance"],
    "metric":      ["euclidean", "manhattan"],
}


def run():
    raw = load_raw_data(DATA_DIR)
    ds  = get_datasets(raw, feature_selector=SELECTOR, preprocessor=PREPROCESSOR,
                       random_state=RANDOM_STATE)

    print(f"\nTrain: {ds['X_train'].shape} | Val: {ds['X_val'].shape} | Test: {ds['X_test'].shape}")

    search = GridSearchCV(KNeighborsClassifier(), PARAM_GRID,
                          cv=5, scoring="accuracy", n_jobs=-1, verbose=1)
    search.fit(ds["X_train"], ds["y_train"])
    print(f"Best params: {search.best_params_}")

    y_val_pred = search.best_estimator_.predict(ds["X_val"])
    evaluate("kNN Task1", ds["y_val"], y_val_pred, raw["train_meta"])

    final = KNeighborsClassifier(**search.best_params_)
    final.fit(ds["X_train_full"], ds["y_train_full"])
    save_submission(ds["image_ids"], final.predict(ds["X_test"]), SUBMISSION_FILE)


if __name__ == "__main__":
    run()
