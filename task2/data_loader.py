import os
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def load_raw_data(data_dir):
    keys = ["train_meta", "test_meta", "additional", "hog", "color"]
    files = ["train_metadata.csv", "test_metadata.csv",
             "additional_features.csv", "hog_pca.csv", "color_histogram.csv"]
    raw = {}
    for key, fname in zip(keys, files):
        raw[key] = pd.read_csv(os.path.join(data_dir, fname))
        print(f"  Loaded {key:12s}: {raw[key].shape}")
    return raw


def _merge_all(metadata_df, additional_df, hog_df, color_df):
    feats = pd.merge(additional_df, hog_df,   on="image_id", how="inner")
    feats = pd.merge(feats,         color_df, on="image_id", how="inner")
    merged = pd.merge(metadata_df, feats, on="image_id", how="left")
    n_missing = merged.isnull().any(axis=1).sum()
    if n_missing:
        print(f"  Warning: dropping {n_missing} rows with missing values.")
        merged = merged.dropna()
    non_feat = {"image_id", "image_path", "class_id", "class_name"}
    feature_cols = [c for c in merged.columns if c not in non_feat]
    return merged, feature_cols


def build_feature_matrix(raw, feature_selector=None, is_train=True):
    meta = raw["train_meta"] if is_train else raw["test_meta"]
    merged, feature_cols = _merge_all(meta, raw["additional"], raw["hog"], raw["color"])

    if feature_selector is not None:
        merged, feature_cols = feature_selector(merged, feature_cols, is_train=is_train)

    X = merged[feature_cols].values.astype(float)
    if is_train:
        return X, merged["class_id"].values, feature_cols
    else:
        return X, merged["image_id"].values, feature_cols


class Preprocessor:
    """
    Stateful: imputation -> optional scaling -> optional per-group PCA.

    pca_groups allows PCA to be applied independently to named feature groups
    rather than globally across all features. This is important because applying
    a single PCA to the combined feature matrix conflates very different feature
    types (e.g. colour histograms and HOG components) and loses the ability to
    control compression per group.

    pca_groups : dict mapping column prefix -> n_components, e.g.
                 {"color_": 20, "hog_pca_": 30}
                 PCA is fitted on each group separately on training data
                 and applied consistently to val/test.
    """

    def __init__(self, scaler_type="standard", pca_groups=None):
        self.scaler_type = scaler_type
        self.pca_groups  = pca_groups or {}
        self._imputer    = SimpleImputer(strategy="mean")
        self._scaler     = {"standard": StandardScaler(),
                            "minmax":   MinMaxScaler(),
                            "robust":   RobustScaler(),
                            None:       None}.get(scaler_type)
        self._pcas       = {}   # prefix -> fitted PCA
        self._col_order  = None
        self._fitted     = False

    def fit_transform(self, X, feature_cols=None):
        X = self._imputer.fit_transform(X)
        if self._scaler is not None:
            X = self._scaler.fit_transform(X)

        if self.pca_groups and feature_cols is not None:
            X, self._col_order = self._apply_group_pca(X, feature_cols, fit=True)

        self._fitted = True
        return X

    def transform(self, X, feature_cols=None):
        if not self._fitted:
            raise RuntimeError("Call fit_transform() before transform().")
        X = self._imputer.transform(X)
        if self._scaler is not None:
            X = self._scaler.transform(X)
        if self.pca_groups and feature_cols is not None:
            X, _ = self._apply_group_pca(X, feature_cols, fit=False)
        return X

    def _apply_group_pca(self, X, feature_cols, fit):
        result_blocks = []
        used = set()

        for prefix, n_comp in self.pca_groups.items():
            idx = [i for i, c in enumerate(feature_cols) if c.startswith(prefix)]
            if not idx:
                continue
            block = X[:, idx]
            k = min(n_comp, block.shape[1], block.shape[0] - 1)
            if fit:
                pca = PCA(n_components=k)
                block = pca.fit_transform(block)
                self._pcas[prefix] = pca
                print(f"  PCA [{prefix}]: {k} components, "
                      f"{pca.explained_variance_ratio_.sum()*100:.1f}% variance")
            else:
                block = self._pcas[prefix].transform(block)
            result_blocks.append(block)
            used.update(idx)

        # Append remaining columns that were not part of any PCA group
        remaining_idx = [i for i in range(X.shape[1]) if i not in used]
        if remaining_idx:
            result_blocks.append(X[:, remaining_idx])

        return np.hstack(result_blocks), None


def get_datasets(raw, feature_selector=None, preprocessor=None,
                 val_size=0.2, random_state=42):

    if preprocessor is None:
        preprocessor = Preprocessor(scaler_type="standard")

    print("\n--- Building training feature matrix ---")
    X_full, y_full, feature_cols = build_feature_matrix(
        raw, feature_selector=feature_selector, is_train=True)
    print(f"  Shape: {X_full.shape}")

    print("\n--- Building test feature matrix ---")
    X_test_raw, image_ids, _ = build_feature_matrix(
        raw, feature_selector=feature_selector, is_train=False)
    print(f"  Shape: {X_test_raw.shape}")

    if X_full.shape[1] != X_test_raw.shape[1]:
        raise ValueError(
            f"Column mismatch: train={X_full.shape[1]}, test={X_test_raw.shape[1]}. "
            "Check that feature CSVs are present and selectors produce consistent output.")

    X_tr_raw, X_val_raw, y_train, y_val = train_test_split(
        X_full, y_full, test_size=val_size, random_state=random_state, stratify=y_full)

    pp_val = deepcopy(preprocessor)
    X_train = pp_val.fit_transform(X_tr_raw, feature_cols)
    X_val   = pp_val.transform(X_val_raw, feature_cols)

    pp_full = deepcopy(preprocessor)
    X_train_full = pp_full.fit_transform(X_full, feature_cols)
    X_test       = pp_full.transform(X_test_raw, feature_cols)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "image_ids": image_ids,
        "X_train_full": X_train_full, "y_train_full": y_full,
        "feature_cols": feature_cols,
    }
