from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from data_loader import load_raw_data, get_datasets, Preprocessor
from feature_engineering import build_selector
from _model_utils import evaluate, save_submission

DATA_DIR         = "task1_data"
RESNET_TRAIN_CSV = "resnet_features_train1.csv"
RESNET_TEST_CSV  = "resnet_features_test1.csv"
SUBMISSION_FILE  = "decisiontree_task1.csv"
RANDOM_STATE     = 42

# Combined raw ResNet + handcrafted, no PCA. 
SELECTOR = build_selector(
    mode="combined",
    resnet_train_csv=RESNET_TRAIN_CSV,
    resnet_test_csv=RESNET_TEST_CSV,
    use_pca=False,
    selection_k=300,
)

PREPROCESSOR = Preprocessor(scaler_type=None)

PARAM_GRID = {
    "criterion":         ["gini", "entropy"],
    "max_depth":         [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      [None, "sqrt", "log2"],
}


def run():
    raw = load_raw_data(DATA_DIR)
    ds  = get_datasets(raw, feature_selector=SELECTOR, preprocessor=PREPROCESSOR,
                       random_state=RANDOM_STATE)

    print(f"\nTrain: {ds['X_train'].shape} | Val: {ds['X_val'].shape} | Test: {ds['X_test'].shape}")

    search = GridSearchCV(DecisionTreeClassifier(random_state=RANDOM_STATE),
                          PARAM_GRID, cv=5, scoring="accuracy", n_jobs=-1, verbose=1)
    search.fit(ds["X_train"], ds["y_train"])
    print(f"Best params: {search.best_params_}")

    y_val_pred = search.best_estimator_.predict(ds["X_val"])
    evaluate("Decision Tree Task1", ds["y_val"], y_val_pred, raw["train_meta"])

    final = DecisionTreeClassifier(**search.best_params_, random_state=RANDOM_STATE)
    final.fit(ds["X_train_full"], ds["y_train_full"])
    save_submission(ds["image_ids"], final.predict(ds["X_test"]), SUBMISSION_FILE)


if __name__ == "__main__":
    run()
