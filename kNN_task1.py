import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def prepare_feature_matrix(metadata_df, additional_df, hog_df, color_df, is_train=True):
    features_df = pd.merge(additional_df, hog_df, on="image_id", how="inner")
    features_df = pd.merge(features_df, color_df, on="image_id", how="inner")
    merged_data = pd.merge(metadata_df, features_df, on="image_id", how="left")

    # Explicit NaN check after join
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


def run_knn_classifier():
    data_dir = "task1_data"

    train_meta = pd.read_csv(os.path.join(data_dir, "train_metadata.csv"))
    test_meta  = pd.read_csv(os.path.join(data_dir, "test_metadata.csv"))
    additional = pd.read_csv(os.path.join(data_dir, "additional_features.csv"))
    hog        = pd.read_csv(os.path.join(data_dir, "hog_pca.csv"))
    color      = pd.read_csv(os.path.join(data_dir, "color_histogram.csv"))

    X_train_full, y_train_full = prepare_feature_matrix(
        train_meta, additional, hog, color, is_train=True
    )

    # Validation split for local accuracy estimate
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    # Fit scaler on training split only; transform val separately
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)

    param_grid = {
        "n_neighbors": [3, 5, 7, 11, 15],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    }

    print("Tuning kNN model...")
    grid_search = GridSearchCV(
        KNeighborsClassifier(), param_grid, cv=5,
        scoring="accuracy", n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train_scaled, y_train)

    best_params = grid_search.best_params_
    print(f"Best Hyperparameters: {best_params}")

    best_model = grid_search.best_estimator_
    y_val_pred = best_model.predict(X_val_scaled)
    print(f"\nValidation Accuracy: {accuracy_score(y_val, y_val_pred) * 100:.2f}%")
    print(classification_report(y_val, y_val_pred))

    # Refit on full training data for test predictions
    scaler_full = StandardScaler()
    X_train_full_scaled = scaler_full.fit_transform(X_train_full)

    final_model = KNeighborsClassifier(**best_params)
    final_model.fit(X_train_full_scaled, y_train_full)

    # Generate test predictions
    X_test, image_ids = prepare_feature_matrix(
        test_meta, additional, hog, color, is_train=False
    )
    X_test_scaled = scaler_full.transform(X_test)
    y_test_pred = final_model.predict(X_test_scaled)

    submission = pd.DataFrame({"image_id": image_ids, "class_id": y_test_pred})
    submission.to_csv("knn_submission.csv", index=False)
    print("\nTest predictions saved to knn_submission.csv")


if __name__ == "__main__":
    run_knn_classifier()