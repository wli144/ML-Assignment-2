import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


def prepare_feature_matrix(metadata_df, additional_df, hog_df, color_df, is_train=True):
    features_df = pd.merge(additional_df, hog_df, on="image_id", how="inner")
    features_df = pd.merge(features_df, color_df, on="image_id", how="inner")
    merged_data = pd.merge(metadata_df, features_df, on="image_id", how="left")

    n_missing = merged_data.isnull().any(axis=1).sum()
    if n_missing > 0:
        print(f"Warning: {n_missing} rows have missing feature values after merge. Dropping.")
        merged_data = merged_data.dropna()

    if is_train:
        drop_columns = ["image_id", "image_path", "class_id", "class_name"]
        X = merged_data.drop(columns=drop_columns).values
        y = merged_data["class_id"].values
        return X, y
    else:
        drop_columns = ["image_id", "image_path"]
        X = merged_data.drop(columns=drop_columns).values
        image_ids = merged_data["image_id"].values
        return X, image_ids


def run_svm_configuration(name, estimator, param_grid, X_train_scaled, y_train,
                           X_val_scaled, y_val):
    """Tune, evaluate, and return the best model and its validation accuracy."""
    print(f"\n{'='*50}")
    print(f"Tuning SVM [{name}]...")
    grid_search = GridSearchCV(
        estimator, param_grid, cv=5,
        scoring="accuracy", n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train_scaled, y_train)

    best_params = grid_search.best_params_
    print(f"Best Hyperparameters: {best_params}")

    best_model = grid_search.best_estimator_
    y_val_pred = best_model.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    print(classification_report(y_val, y_val_pred))

    return best_model, best_params, val_accuracy


def run_svm_classifier():
    data_dir = "task1_data"

    train_meta = pd.read_csv(os.path.join(data_dir, "train_metadata.csv"))
    test_meta  = pd.read_csv(os.path.join(data_dir, "test_metadata.csv"))
    additional = pd.read_csv(os.path.join(data_dir, "additional_features.csv"))
    hog        = pd.read_csv(os.path.join(data_dir, "hog_pca.csv"))
    color      = pd.read_csv(os.path.join(data_dir, "color_histogram.csv"))

    X_train_full, y_train_full = prepare_feature_matrix(
        train_meta, additional, hog, color, is_train=True
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    # SVM is sensitive to feature scale; standardisation is required
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)

    # --- Configuration 1: Linear kernel ---
    linear_param_grid = {
        "C": [0.01, 0.1, 1, 10, 100]
    }
    linear_model, linear_params, linear_acc = run_svm_configuration(
        name="Linear",
        estimator=SVC(kernel="linear", random_state=42),
        param_grid=linear_param_grid,
        X_train_scaled=X_train_scaled,
        y_train=y_train,
        X_val_scaled=X_val_scaled,
        y_val=y_val
    )

    # --- Configuration 2: RBF kernel ---
    rbf_param_grid = {
        "C":     [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.001, 0.01]
    }
    rbf_model, rbf_params, rbf_acc = run_svm_configuration(
        name="RBF",
        estimator=SVC(kernel="rbf", random_state=42),
        param_grid=rbf_param_grid,
        X_train_scaled=X_train_scaled,
        y_train=y_train,
        X_val_scaled=X_val_scaled,
        y_val=y_val
    )

    # --- Summary ---
    print(f"\n{'='*50}")
    print(f"Linear SVM Validation Accuracy : {linear_acc * 100:.2f}%")
    print(f"RBF SVM    Validation Accuracy : {rbf_acc * 100:.2f}%")

    # --- Generate test predictions for both configurations ---
    scaler_full = StandardScaler()
    X_train_full_scaled = scaler_full.fit_transform(X_train_full)

    X_test, image_ids = prepare_feature_matrix(
        test_meta, additional, hog, color, is_train=False
    )
    X_test_scaled = scaler_full.transform(X_test)

    for name, kernel, params in [
        ("linear", "linear", linear_params),
        ("rbf",    "rbf",    rbf_params)
    ]:
        final_model = SVC(kernel=kernel, random_state=42, **params)
        final_model.fit(X_train_full_scaled, y_train_full)
        y_test_pred = final_model.predict(X_test_scaled)

        out_file = f"svm_{name}_submission.csv"
        submission = pd.DataFrame({"image_id": image_ids, "class_id": y_test_pred})
        submission.to_csv(out_file, index=False)
        print(f"Test predictions saved to {out_file}")


if __name__ == "__main__":
    run_svm_classifier()