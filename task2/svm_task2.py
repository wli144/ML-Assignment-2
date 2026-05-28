from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from data_loader import load_raw_data, get_datasets, Preprocessor
from feature_engineering import build_selector
from _model_utils import evaluate, save_submission

DATA_DIR         = "task2_data"
RESNET_TRAIN_CSV = "resnet_features_train2.csv"
RESNET_TEST_CSV  = "resnet_features_test2.csv"
SUBMISSION_FILE  = "svm_task2.csv"
RANDOM_STATE     = 42

SELECTOR = build_selector(
    mode="combined",
    resnet_train_csv=RESNET_TRAIN_CSV,
    resnet_test_csv=RESNET_TEST_CSV,
    use_pca=True,
    pca_k=150,
    selection_k=250,
)

PREPROCESSOR = Preprocessor(scaler_type="standard")

PARAM_GRID = {
    "C": [0.1, 1, 10, 50, 100, 500, 1000],
    "gamma": ["scale", "auto", 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
}


def run():
    raw = load_raw_data(DATA_DIR)
    ds  = get_datasets(raw, feature_selector=SELECTOR, preprocessor=PREPROCESSOR,
                       random_state=RANDOM_STATE)

    print(f"\nTrain: {ds['X_train'].shape} | Val: {ds['X_val'].shape} | Test: {ds['X_test'].shape}")

    search = GridSearchCV(
        SVC(kernel="rbf", random_state=RANDOM_STATE, probability=True),
        PARAM_GRID, cv=5, scoring="f1_macro", n_jobs=-1, verbose=1,
    )
    search.fit(ds["X_train"], ds["y_train"])
    print(f"Best params: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_*100:.2f}%")

    y_val_pred = search.best_estimator_.predict(ds["X_val"])
    evaluate("SVM RBF Task2", ds["y_val"], y_val_pred, raw["train_meta"])

    final = SVC(kernel="rbf", random_state=RANDOM_STATE,
                probability=True, **search.best_params_)
    final.fit(ds["X_train_full"], ds["y_train_full"])
    save_submission(ds["image_ids"], final.predict(ds["X_test"]), SUBMISSION_FILE)


if __name__ == "__main__":
    run()
