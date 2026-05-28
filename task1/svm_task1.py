from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from data_loader import load_raw_data, get_datasets, Preprocessor
from feature_engineering import build_selector
from _model_utils import evaluate, save_submission

DATA_DIR         = "task1_data"
RESNET_TRAIN_CSV = "resnet_features_train1.csv"
RESNET_TEST_CSV  = "resnet_features_test1.csv"
RANDOM_STATE     = 42

# ResNet-PCA + handcrafted, scaled. StandardScaler is mandatory for SVM.
# use_pca=True compresses 2048-d ResNet before combining with handcrafted
SELECTOR = build_selector(
    mode="combined",
    resnet_train_csv=RESNET_TRAIN_CSV,
    resnet_test_csv=RESNET_TEST_CSV,
    use_pca=True,
    pca_k= 45,
    selection_k=100,
)

PREPROCESSOR = Preprocessor(scaler_type="standard")

CONFIGS = [
    ("Linear", "linear", {"C": [0.01, 0.1, 1, 10, 100]},
     "svm_linear_task1.csv"),
    ("RBF",    "rbf",    {"C": [0.01, 0.1, 1, 5, 10, 50, 100], "gamma": [0.0005, 0.001, 0.003, 0.005, 0.01]},
     "svm_rbf_task1.csv"),
]


def run():
    raw = load_raw_data(DATA_DIR)
    ds  = get_datasets(raw, feature_selector=SELECTOR, preprocessor=PREPROCESSOR,
                       random_state=RANDOM_STATE)

    print(f"\nTrain: {ds['X_train'].shape} | Val: {ds['X_val'].shape} | Test: {ds['X_test'].shape}")

    results = {}
    for name, kernel, param_grid, submission_file in CONFIGS:
        print(f"\n{'='*50}\nSVM [{name}]\n{'='*50}")
        search = GridSearchCV(SVC(kernel=kernel, random_state=RANDOM_STATE),
                              param_grid, cv=5, scoring="f1_macro", n_jobs=-1, verbose=1)
        search.fit(ds["X_train"], ds["y_train"])
        print(f"Best params: {search.best_params_}")

        y_val_pred = search.best_estimator_.predict(ds["X_val"])
        acc = evaluate(f"SVM {name} Task1", ds["y_val"], y_val_pred, raw["train_meta"])

        final = SVC(kernel=kernel, random_state=RANDOM_STATE, **search.best_params_)
        final.fit(ds["X_train_full"], ds["y_train_full"])
        save_submission(ds["image_ids"], final.predict(ds["X_test"]), submission_file)
        results[name] = acc

    print(f"\n{'='*50}\nSummary")
    for name, acc in results.items():
        print(f"  {name:8s}: {acc*100:.2f}%")


if __name__ == "__main__":
    run()
