"""
data_loader.py
--------------
Responsible for:
  - Reading all raw CSV files from disk (done once)
  - Merging feature files with metadata
  - Applying preprocessing (scaling, imputation, dimensionality reduction)
  - Returning clean (X_train, y_train), (X_val, y_val), (X_test, image_ids) splits

All model scripts import from here. CSVs are never read directly in model files.

Task 1 vs Task 2
----------------
Task 1: 10-class animal classification. Pass data_dir="task1_data".
Task 2: 10-class fine-grained bird species classification.
        Pass data_dir="task2_data". The metadata CSVs are different (bird species
        labels), but the feature files have the same structure.
        The ResNet features are loaded separately via feature_engineering.py
        (merge_resnet_features), since they join on image_path rather than image_id.
"""

import os
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


# ---------------------------------------------------------------------------
# 1. Raw data loading
# ---------------------------------------------------------------------------

def load_raw_data(data_dir: str = "task1_data") -> dict:
    """
    Read all CSV files from disk and return them as a dict of DataFrames.
    Call this once and pass the result to get_datasets().

    Returns
    -------
    dict with keys:
        "train_meta"  - train_metadata.csv   (image_id, image_path, class_id, class_name)
        "test_meta"   - test_metadata.csv    (image_id, image_path)
        "additional"  - additional_features.csv  (feat_0 .. feat_22)
        "hog"         - hog_pca.csv              (hog_pca_0 .. hog_pca_99)
        "color"       - color_histogram.csv      (color_0 .. color_95)
    """
    paths = {
        "train_meta": "train_metadata.csv",
        "test_meta":  "test_metadata.csv",
        "additional": "additional_features.csv",
        "hog":        "hog_pca.csv",
        "color":      "color_histogram.csv",
    }
    raw = {}
    for key, filename in paths.items():
        full_path = os.path.join(data_dir, filename)
        raw[key] = pd.read_csv(full_path)
        print(f"  Loaded {key:12s}: {raw[key].shape}")
    return raw


# ---------------------------------------------------------------------------
# 2. Merging
# ---------------------------------------------------------------------------

def _merge_features(metadata_df: pd.DataFrame,
                    additional_df: pd.DataFrame,
                    hog_df: pd.DataFrame,
                    color_df: pd.DataFrame,
                    is_train: bool = True):
    """
    Inner-join the three feature files on image_id, then left-join onto
    metadata so that metadata ordering is preserved.

    Returns (merged_df, feature_cols).
    feature_cols is the list of columns that are purely numeric features
    (i.e. everything except image_id, image_path, class_id, class_name).
    """
    features_df = pd.merge(additional_df, hog_df,   on="image_id", how="inner")
    features_df = pd.merge(features_df,   color_df, on="image_id", how="inner")
    merged = pd.merge(metadata_df, features_df, on="image_id", how="left")

    n_missing = merged.isnull().any(axis=1).sum()
    if n_missing > 0:
        print(f"  Warning: dropping {n_missing} rows with missing feature values.")
        merged = merged.dropna()

    # Derive feature column list dynamically — works for both task 1 and task 2
    non_feature = {"image_id", "image_path", "class_id", "class_name"}
    feature_cols = [c for c in merged.columns if c not in non_feature]

    return merged, feature_cols


def build_feature_matrix(raw: dict,
                         feature_selector=None,
                         is_train: bool = True):
    """
    Construct X (and y or image_ids) from the raw dict.

    Parameters
    ----------
    raw              : dict from load_raw_data()
    feature_selector : optional callable from feature_engineering.py
    is_train         : True for training data (returns labels), False for test

    Returns
    -------
    is_train=True  -> (X, y, feature_cols)
    is_train=False -> (X, image_ids, feature_cols)
    """
    metadata_df = raw["train_meta"] if is_train else raw["test_meta"]
    merged, feature_cols = _merge_features(
        metadata_df,
        raw["additional"],
        raw["hog"],
        raw["color"],
        is_train=is_train,
    )

    if feature_selector is not None:
        merged, feature_cols = feature_selector(merged, feature_cols, is_train=is_train)

    X = merged[feature_cols].values.astype(float)

    if is_train:
        y = merged["class_id"].values
        return X, y, feature_cols
    else:
        image_ids = merged["image_id"].values
        return X, image_ids, feature_cols


# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------

class Preprocessor:
    """
    Stateful preprocessing: imputation -> scaling -> optional PCA.
    Fit parameters are learned on training data and applied consistently
    to validation and test sets.

    Parameters
    ----------
    scaler_type : "standard" | "minmax" | "robust" | None
        "standard" is required for kNN and SVM (distance-sensitive).
        "robust"   is preferable when outliers are present (e.g. ResNet
                   activations can have large positive outliers post-ReLU).
        None       skips scaling; acceptable for tree-based models if desired.
    imputer_strategy : "mean" | "median" | "most_frequent"
    pca_components : int | float | None
        int   -> retain exactly that many components
        float -> retain enough components to explain that fraction of variance
        None  -> no PCA
    """

    def __init__(self,
                 scaler_type: str = "standard",
                 imputer_strategy: str = "mean",
                 pca_components=None):
        self.scaler_type      = scaler_type
        self.imputer_strategy = imputer_strategy
        self.pca_components   = pca_components

        self._imputer = SimpleImputer(strategy=imputer_strategy)
        self._scaler  = self._build_scaler(scaler_type)
        self._pca     = PCA(n_components=pca_components) if pca_components else None
        self._fitted  = False

    @staticmethod
    def _build_scaler(scaler_type):
        return {
            "standard": StandardScaler(),
            "minmax":   MinMaxScaler(),
            "robust":   RobustScaler(),
            None:       None,
        }.get(scaler_type)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = self._imputer.fit_transform(X)
        if self._scaler is not None:
            X = self._scaler.fit_transform(X)
        if self._pca is not None:
            X = self._pca.fit_transform(X)
            explained = self._pca.explained_variance_ratio_.sum()
            print(f"  PCA: retained {self._pca.n_components_} components "
                  f"({explained * 100:.1f}% variance explained)")
        self._fitted = True
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Preprocessor.fit_transform() must be called before transform().")
        X = self._imputer.transform(X)
        if self._scaler is not None:
            X = self._scaler.transform(X)
        if self._pca is not None:
            X = self._pca.transform(X)
        return X


# ---------------------------------------------------------------------------
# 4. Full pipeline: load -> build -> split -> preprocess
# ---------------------------------------------------------------------------

def get_datasets(raw: dict,
                 feature_selector=None,
                 preprocessor: Preprocessor = None,
                 val_size: float = 0.2,
                 random_state: int = 42) -> dict:
    """
    Top-level convenience function used by all model scripts.

    Produces two preprocessors:
      pp_val  : fitted on the 80% train split  -> used to evaluate val accuracy
      pp_full : fitted on all training data     -> used for final test predictions

    Parameters
    ----------
    raw              : dict from load_raw_data()
    feature_selector : optional callable from feature_engineering.py
    preprocessor     : Preprocessor instance; defaults to StandardScaler, no PCA
    val_size         : fraction of training data for validation holdout
    random_state     : reproducibility seed

    Returns
    -------
    dict with keys:
        X_train, y_train          - preprocessed train split
        X_val,   y_val            - preprocessed val split
        X_test,  image_ids        - preprocessed test set
        X_train_full, y_train_full- full training data (for final model refit)
        feature_cols              - feature column names before preprocessing
        preprocessor_val          - Preprocessor fitted on train split
        preprocessor_full         - Preprocessor fitted on full training data
    """
    if preprocessor is None:
        preprocessor = Preprocessor(scaler_type="standard")

    print("\n--- Building training feature matrix ---")
    X_full, y_full, feature_cols = build_feature_matrix(
        raw, feature_selector=feature_selector, is_train=True
    )
    print(f"  Training matrix: {X_full.shape}")

    print("\n--- Building test feature matrix ---")
    X_test_raw, image_ids, _ = build_feature_matrix(
        raw, feature_selector=feature_selector, is_train=False
    )
    print(f"  Test matrix: {X_test_raw.shape}")

    # Train / val split
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_full, y_full,
        test_size=val_size,
        random_state=random_state,
        stratify=y_full,
    )

    # Preprocessor for validation accuracy estimates (fit on train split only)
    pp_val = deepcopy(preprocessor)
    X_train = pp_val.fit_transform(X_train_raw)
    X_val   = pp_val.transform(X_val_raw)

    # Preprocessor for final submission (fit on all training data)
    pp_full = deepcopy(preprocessor)
    X_train_full = pp_full.fit_transform(X_full)
    X_test       = pp_full.transform(X_test_raw)

    return {
        "X_train":           X_train,
        "y_train":           y_train,
        "X_val":             X_val,
        "y_val":             y_val,
        "X_test":            X_test,
        "image_ids":         image_ids,
        "X_train_full":      X_train_full,
        "y_train_full":      y_full,
        "feature_cols":      feature_cols,
        "preprocessor_val":  pp_val,
        "preprocessor_full": pp_full,
    }
