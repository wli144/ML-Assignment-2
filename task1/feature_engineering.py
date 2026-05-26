"""
feature_engineering.py
-----------------------
Defines feature selectors and transformations that can be passed as the
`feature_selector` argument to data_loader.get_datasets().

A feature selector is any callable with the signature:

    def my_selector(merged_df, feature_cols, is_train=True):
        -> (modified_df, modified_feature_cols)

The merged_df is the full DataFrame after the metadata/feature join.
feature_cols is the list of column names that constitute the feature matrix.
Return the (possibly modified) df and updated feature_cols list.

Usage example
-------------
from data_loader import load_raw_data, get_datasets, Preprocessor
from feature_engineering import combined_selector

raw = load_raw_data("task1_data")
datasets = get_datasets(raw, feature_selector=combined_selector)
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_cols(df, feature_cols):
    """Drop any feature_cols that are no longer in df (safety check)."""
    return [c for c in feature_cols if c in df.columns]


# ---------------------------------------------------------------------------
# 1. Feature group selectors
#    Use these to restrict the model to one feature type, for ablation studies.
# ---------------------------------------------------------------------------

def hog_only(merged_df, feature_cols, is_train=True):
    """
    Retain only HOG-PCA features.
    HOG captures edge orientation and shape; good for silhouette-based classes.
    """
    selected = [c for c in feature_cols if c.startswith("hog") or c.startswith("pca")]
    if not selected:
        # Fallback: try to infer by exclusion of known prefixes
        known_other = {"edge_density", "texture_variance",
                       "avg_r", "avg_g", "avg_b"}
        selected = [c for c in feature_cols
                    if c not in known_other and not c.startswith("color")]
    print(f"  [hog_only] {len(selected)} features selected")
    return merged_df, selected


def color_only(merged_df, feature_cols, is_train=True):
    """
    Retain only colour histogram features.
    Colour is highly discriminative for classes like butterfly or frog.
    """
    selected = [c for c in feature_cols if c.startswith("color") or c.startswith("hist")]
    if not selected:
        selected = [c for c in feature_cols
                    if not (c.startswith("hog") or c.startswith("pca"))
                    and c not in {"edge_density", "texture_variance",
                                  "avg_r", "avg_g", "avg_b"}]
    print(f"  [color_only] {len(selected)} features selected")
    return merged_df, selected


def additional_only(merged_df, feature_cols, is_train=True):
    """
    Retain only the additional hand-crafted features:
    edge density, texture variance, average RGB channel values.
    Low dimensionality; useful as a weak baseline.
    """
    known_additional = {"edge_density", "texture_variance", "avg_r", "avg_g", "avg_b"}
    selected = [c for c in feature_cols if c in known_additional]
    print(f"  [additional_only] {len(selected)} features selected")
    return merged_df, selected


def all_features(merged_df, feature_cols, is_train=True):
    """
    No filtering — use all available provided features.
    This is the default when no feature_selector is passed.
    """
    print(f"  [all_features] {len(feature_cols)} features selected")
    return merged_df, feature_cols


# ---------------------------------------------------------------------------
# 2. Engineered feature additions
#    These add new columns to merged_df derived from existing ones.
# ---------------------------------------------------------------------------

def add_color_moments(merged_df, feature_cols, is_train=True):
    """
    Compute per-channel mean, variance, and skewness from available
    colour histogram bins (approximation).

    If avg_r / avg_g / avg_b are present, derive additional ratio features:
      - Hue proxy: r / (g + 1e-6)
      - Saturation proxy: max(r,g,b) - min(r,g,b)
      - Brightness proxy: (r + g + b) / 3  (redundant if avg already present,
        but the ratio features are novel)
    These ratios are illumination-sensitive discriminators for animal colour.
    """
    new_cols = []
    required = {"avg_r", "avg_g", "avg_b"}
    if required.issubset(set(merged_df.columns)):
        r = merged_df["avg_r"]
        g = merged_df["avg_g"]
        b = merged_df["avg_b"]

        merged_df = merged_df.copy()
        merged_df["feat_rg_ratio"]    = r / (g + 1e-6)
        merged_df["feat_rb_ratio"]    = r / (b + 1e-6)
        merged_df["feat_gb_ratio"]    = g / (b + 1e-6)
        merged_df["feat_saturation"]  = (
            merged_df[["avg_r", "avg_g", "avg_b"]].max(axis=1) -
            merged_df[["avg_r", "avg_g", "avg_b"]].min(axis=1)
        )
        new_cols = ["feat_rg_ratio", "feat_rb_ratio",
                    "feat_gb_ratio", "feat_saturation"]
        print(f"  [add_color_moments] Added {len(new_cols)} colour ratio features")
    else:
        print("  [add_color_moments] avg_r/g/b not found; skipping")

    return merged_df, feature_cols + new_cols


def add_texture_edge_interaction(merged_df, feature_cols, is_train=True):
    """
    Multiply edge_density by texture_variance to create a joint feature
    capturing images that are both highly textured and edge-rich
    (e.g., spider webs, butterfly wing patterns).

    Also adds log(edge_density + 1) to compress the right-skewed distribution
    of edge density values.
    """
    new_cols = []
    merged_df = merged_df.copy()

    if "edge_density" in merged_df.columns and "texture_variance" in merged_df.columns:
        merged_df["feat_edge_texture"] = (
            merged_df["edge_density"] * merged_df["texture_variance"]
        )
        merged_df["feat_log_edge"]     = np.log1p(merged_df["edge_density"])
        new_cols += ["feat_edge_texture", "feat_log_edge"]
        print(f"  [add_texture_edge_interaction] Added {len(new_cols)} features")
    else:
        print("  [add_texture_edge_interaction] edge_density/texture_variance not found; skipping")

    return merged_df, feature_cols + new_cols


def add_hog_color_cross(merged_df, feature_cols, is_train=True):
    """
    Create pairwise products between the first few HOG-PCA components and
    the average colour channels. This captures animals that are both a
    particular colour AND have a particular edge-orientation profile
    (e.g., a bright red bird vs. a bright red flower in background).

    Only uses the first 3 PCA components to keep dimensionality controlled.
    """
    new_cols = []
    merged_df = merged_df.copy()

    pca_cols = [c for c in feature_cols if c.startswith("pca") or c.startswith("hog_pca")][:3]
    avg_cols = [c for c in ["avg_r", "avg_g", "avg_b"] if c in merged_df.columns]

    for pc in pca_cols:
        for ac in avg_cols:
            col_name = f"feat_cross_{pc}_{ac}"
            merged_df[col_name] = merged_df[pc] * merged_df[ac]
            new_cols.append(col_name)

    if new_cols:
        print(f"  [add_hog_color_cross] Added {len(new_cols)} cross features")
    else:
        print("  [add_hog_color_cross] Insufficient PCA/colour columns; skipping")

    return merged_df, feature_cols + new_cols


# ---------------------------------------------------------------------------
# 3. Statistical feature selection
#    These reduce dimensionality by scoring features against the target.
#    Only valid during training (when labels are available).
# ---------------------------------------------------------------------------

def select_k_best_anova(k: int = 50):
    """
    Return a selector that keeps the top-k features by ANOVA F-score.
    ANOVA F-score measures how much a feature's mean differs across classes
    relative to within-class variance.

    Suitable for: continuous features with approximately normal within-class
    distributions (colour histograms approximate this reasonably).

    Parameters
    ----------
    k : number of features to retain

    Usage
    -----
    datasets = get_datasets(raw, feature_selector=select_k_best_anova(k=60))
    """
    selector_state = {"support": None, "k": k}

    def _select(merged_df, feature_cols, is_train=True):
        feature_cols = _validate_cols(merged_df, feature_cols)
        X = merged_df[feature_cols].values

        if is_train:
            if "class_id" not in merged_df.columns:
                print("  [select_k_best_anova] No labels available; skipping selection")
                return merged_df, feature_cols
            y = merged_df["class_id"].values
            actual_k = min(k, X.shape[1])
            skb = SelectKBest(f_classif, k=actual_k)
            skb.fit(X, y)
            selector_state["support"] = skb.get_support()
            print(f"  [select_k_best_anova] Selected {actual_k} / {X.shape[1]} features")
        else:
            if selector_state["support"] is None:
                print("  [select_k_best_anova] Not yet fitted; returning all features")
                return merged_df, feature_cols

        selected_cols = [c for c, keep in
                         zip(feature_cols, selector_state["support"]) if keep]
        return merged_df, selected_cols

    return _select


def select_k_best_mutual_info(k: int = 50):
    """
    Return a selector that keeps the top-k features by mutual information.
    Mutual information is non-parametric and captures non-linear associations,
    making it preferable to ANOVA when feature-class relationships are
    not linear (e.g., HOG components whose class signal is non-monotonic).

    Parameters
    ----------
    k : number of features to retain
    """
    selector_state = {"support": None, "k": k}

    def _select(merged_df, feature_cols, is_train=True):
        feature_cols = _validate_cols(merged_df, feature_cols)
        X = merged_df[feature_cols].values

        if is_train:
            if "class_id" not in merged_df.columns:
                return merged_df, feature_cols
            y = merged_df["class_id"].values
            actual_k = min(k, X.shape[1])
            skb = SelectKBest(mutual_info_classif, k=actual_k)
            skb.fit(X, y)
            selector_state["support"] = skb.get_support()
            print(f"  [select_k_best_mutual_info] Selected {actual_k} / {X.shape[1]} features")
        else:
            if selector_state["support"] is None:
                return merged_df, feature_cols

        selected_cols = [c for c, keep in
                         zip(feature_cols, selector_state["support"]) if keep]
        return merged_df, selected_cols

    return _select


# ---------------------------------------------------------------------------
# 4. Composite selectors
#    Chain multiple steps together.
# ---------------------------------------------------------------------------

def compose(*selectors):
    """
    Chain multiple feature selector callables left-to-right.

    Usage
    -----
    pipeline = compose(
        add_color_moments,
        add_texture_edge_interaction,
        select_k_best_mutual_info(k=80)
    )
    datasets = get_datasets(raw, feature_selector=pipeline)
    """
    def _composed(merged_df, feature_cols, is_train=True):
        for sel in selectors:
            merged_df, feature_cols = sel(merged_df, feature_cols, is_train=is_train)
        return merged_df, feature_cols
    return _composed


# ---------------------------------------------------------------------------
# 5. Ready-to-use named configurations
#    Import these directly for quick experimentation.
# ---------------------------------------------------------------------------

# Baseline: all provided features, no additions
baseline = all_features

# Add engineered features on top of the full set; no selection step
with_engineered = compose(
    all_features,
    add_color_moments,
    add_texture_edge_interaction,
)

# Full engineering + cross features + mutual-info selection (top 80)
# Recommended starting point for SVM and kNN
combined_selector = compose(
    all_features,
    add_color_moments,
    add_texture_edge_interaction,
    add_hog_color_cross,
    select_k_best_mutual_info(k=80),
)

# HOG only, ANOVA selection (top 30); useful for shape-dominant classes
hog_anova_30 = compose(
    hog_only,
    select_k_best_anova(k=30),
)

# Colour + additional features only; fast, low-dimensional
color_plus_additional = compose(
    add_color_moments,
    select_k_best_anova(k=40),
)