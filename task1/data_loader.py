"""
data_loader.py
--------------
Responsible for:
  - Reading all raw CSV files from disk (done once)
  - Merging feature files with metadata
  - Applying preprocessing (scaling, imputation, dimensionality reduction)
  - Returning clean (X_train, y_train), (X_val, y_val), (X_test, image_ids) splits

All other modules import from here. CSVs are never read directly in model files.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


# ---------------------------------------------------------------------------
# 1. Raw data loading
# ---------------------------------------------------------------------------

def load_raw_data(data_dir: str = "task1_data") -> dict:
    """
    Read all CSV files from disk and return them as a dict of DataFrames.
    Call this once and pass the result around rather than re-reading files.

    Returns
    -------
    dict with keys:
        "train_meta", "test_meta",
        "additional", "hog", "color"
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
        print(f"Loaded {key}: {raw[key].shape}")
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

    Returns a single merged DataFrame plus the list of pure feature columns.
    """
    features_df = pd.merge(additional_df, hog_df,   on="image_id", how="inner")
    features_df = pd.merge(features_df,   color_df, on="image_id", how="inner")
    merged = pd.merge(metadata_df, features_df, on="image_id", how="left")

    # Report and drop rows where any feature is missing (gap in feature files)
    n_missing = merged.isnull().any(axis=1).sum()
    if n_missing > 0:
        print(f"  Warning: dropping {n_missing} rows with missing feature values.")
        merged = merged.dropna()

    # Identify which columns are purely numeric features
    meta_cols = ["image_id", "image_path"]
    if is_train:
        meta_cols += ["class_id", "class_name"]
    feature_cols = [c for c in merged.columns if c not in meta_cols]

    return merged, feature_cols


def build_feature_matrix(raw: dict,
                         feature_selector=None,
                         is_train: bool = True):
    """
    Construct X (and y / image_ids) from the raw dict returned by load_raw_data.

    Parameters
    ----------
    raw            : dict returned by load_raw_data()
    feature_selector : optional callable from feature_engineering.py;
                       if provided, it receives the merged DataFrame and
                       feature_cols and returns a modified (df, feature_cols)
    is_train       : whether to extract labels

    Returns
    -------
    is_train=True  -> (X: np.ndarray, y: np.ndarray, feature_cols: list)
    is_train=False -> (X: np.ndarray, image_ids: np.ndarray, feature_cols: list)
    """
    metadata_df = raw["train_meta"] if is_train else raw["test_meta"]
    merged, feature_cols = _merge_features(
        metadata_df,
        raw["additional"],
        raw["hog"],
        raw["color"],
        is_train=is_train
    )

    # Optionally apply feature engineering / selection
    if feature_selector is not None:
        merged, feature_cols = feature_selector(merged, feature_cols, is_train=is_train)

    X = merged[feature_cols].values

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
    Encapsulates all stateful preprocessing steps so that fit parameters
    are learned on training data only and consistently applied to val/test.

    Parameters
    ----------
    scaler_type : "standard" | "minmax" | "robust" | None
        StandardScaler is appropriate for SVM and kNN (distance-based).
        RobustScaler is preferable when outliers are present.
        Decision trees do not require scaling, but it is applied here for
        pipeline consistency; it does not affect tree splits.
    imputer_strategy : "mean" | "median" | "most_frequent"
        Strategy for any residual NaN values that survive the merge step.
    pca_components : int | float | None
        If set, apply PCA after scaling.
        int   -> retain that many components
        float -> retain enough components to explain that fraction of variance
        None  -> no PCA
    """

    def __init__(self,
                 scaler_type: str = "standard",
                 imputer_strategy: str = "mean",
                 pca_components=None):
        self.scaler_type       = scaler_type
        self.imputer_strategy  = imputer_strategy
        self.pca_components    = pca_components

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
        }.get(scaler_type, StandardScaler())

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on training data and return transformed array."""
        X = self._imputer.fit_transform(X)
        if self._scaler is not None:
            X = self._scaler.fit_transform(X)
        if self._pca is not None:
            X = self._pca.fit_transform(X)
            explained = self._pca.explained_variance_ratio_.sum()
            print(f"  PCA: {self._pca.n_components_} components, "
                  f"{explained * 100:.1f}% variance explained")
        self._fitted = True
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply previously fitted transformations to val or test data."""
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fit before calling transform().")
        X = self._imputer.transform(X)
        if self._scaler is not None:
            X = self._scaler.transform(X)
        if self._pca is not None:
            X = self._pca.transform(X)
        return X


# ---------------------------------------------------------------------------
# 4. Full pipeline: load -> split -> preprocess
# ---------------------------------------------------------------------------

def get_datasets(raw: dict,
                 feature_selector=None,
                 preprocessor: Preprocessor = None,
                 val_size: float = 0.2,
                 random_state: int = 42):
    """
    Top-level convenience function used by all model scripts.

    Workflow
    --------
    1. Build feature matrices from raw data (train + test)
    2. Train/val split on training data
    3. Fit preprocessor on train split; apply to val and test
    4. Refit preprocessor on full training data for final test predictions

    Parameters
    ----------
    raw              : dict from load_raw_data()
    feature_selector : optional callable from feature_engineering.py
    preprocessor     : Preprocessor instance; defaults to StandardScaler, no PCA
    val_size         : fraction of training data reserved for validation
    random_state     : random seed for reproducibility

    Returns
    -------
    dict with keys:
        "X_train", "y_train"           -> scaled train split
        "X_val",   "y_val"             -> scaled val split
        "X_test",  "image_ids"         -> scaled test set
        "X_train_full", "y_train_full" -> full training data scaled on itself
                                          (use to fit final model before submission)
        "feature_cols"                 -> list of feature column names pre-scaling
        "preprocessor_val"             -> Preprocessor fitted on train split
        "preprocessor_full"            -> Preprocessor fitted on full train data
    """
    if preprocessor is None:
        preprocessor = Preprocessor(scaler_type="standard")

    # Build matrices
    print("\n--- Building training feature matrix ---")
    X_full, y_full, feature_cols = build_feature_matrix(
        raw, feature_selector=feature_selector, is_train=True
    )
    print(f"  Training matrix shape: {X_full.shape}")

    print("\n--- Building test feature matrix ---")
    X_test_raw, image_ids, _ = build_feature_matrix(
        raw, feature_selector=feature_selector, is_train=False
    )
    print(f"  Test matrix shape: {X_test_raw.shape}")

    # Train / val split
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_full, y_full,
        test_size=val_size,
        random_state=random_state,
        stratify=y_full
    )

    # Preprocessor fitted on train split (for validation accuracy estimates)
    from copy import deepcopy
    pp_val = deepcopy(preprocessor)
    X_train = pp_val.fit_transform(X_train_raw)
    X_val   = pp_val.transform(X_val_raw)

    # Preprocessor fitted on full training data (for final submission)
    pp_full = deepcopy(preprocessor)
    X_train_full = pp_full.fit_transform(X_full)
    X_test       = pp_full.transform(X_test_raw)

    return {
        "X_train":          X_train,
        "y_train":          y_train,
        "X_val":            X_val,
        "y_val":            y_val,
        "X_test":           X_test,
        "image_ids":        image_ids,
        "X_train_full":     X_train_full,
        "y_train_full":     y_full,
        "feature_cols":     feature_cols,
        "preprocessor_val": pp_val,
        "preprocessor_full":pp_full,
    }