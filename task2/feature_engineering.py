"""
feature_engineering.py
-----------------------
Defines feature selectors and transformations passed as the
`feature_selector` argument to data_loader.get_datasets().

Column prefixes in the provided feature files
----------------------------------------------
  additional_features.csv  ->  feat_0    .. feat_22      (23 columns)
  hog_pca.csv              ->  hog_pca_0 .. hog_pca_99   (100 columns)
  color_histogram.csv      ->  color_0   .. color_95      (96 columns)
  resnet_features_*.csv    ->  0, 1, .. 2047             (2048 columns, renamed
                                                           to resnet_0..resnet_2047
                                                           on merge)
                                                   Base total: 219 features
                                               With ResNet:  2267 features

A feature selector is any callable with this signature:

    def my_selector(merged_df, feature_cols, is_train=True):
        -> (merged_df, feature_cols)

The merged_df is the full DataFrame after the metadata/feature join.
feature_cols is the current list of active feature column names.
Return the (possibly modified) df and the updated feature_cols list.

Why feature engineering is modular and shared across models
-----------------------------------------------------------
A common point of confusion: if feature engineering is done in a shared module,
are we not leaking information or applying the same transformations to all models
unfairly?

The answer is no, for two separate reasons:

1. STATELESS transformations (add_color_channel_ratios, add_hog_statistics, etc.)
   are pure mathematical functions of the existing columns. They compute ratios,
   norms, entropy, etc. These are identical to writing them inline in each model
   script — sharing them is purely a code hygiene decision. There is no fitting
   step and therefore no risk of data leakage or inconsistency across models.

2. STATEFUL transformations (select_k_best_anova, select_k_best_mutual_info)
   DO have a fitting step (which features to keep is learned from training labels).
   These are implemented as closures with private state dicts. Each call to
   e.g. select_k_best_anova(k=150) creates a FRESH closure with its own empty
   state. When you pass it into get_datasets(), data_loader calls it on training
   data first (is_train=True), which fits the selector and stores the mask.
   Subsequent calls with is_train=False apply the stored mask. Crucially,
   get_datasets() deepcopies the Preprocessor before fitting, so two separate
   calls to get_datasets() — one for kNN, one for SVM — each get their own
   independent fitted selector state, as long as they each receive a fresh
   selector callable (which they do, because make_resnet_selector() or compose()
   constructs a new closure each time it is called).

   The key rule: never reuse the same stateful selector instance across two
   different get_datasets() calls. Always call compose(...) or make_resnet_selector()
   at the top of each model script to get a fresh instance.

Usage example
-------------
    from data_loader import load_raw_data, get_datasets
    from feature_engineering import make_resnet_selector

    raw      = load_raw_data("task2_data")
    selector = make_resnet_selector(
        resnet_train_csv="resnet_features_hf.csv",
        resnet_test_csv="resnet_features_test.csv",
        use_pca=True, pca_k=200, selection_k=150,
    )
    datasets = get_datasets(raw, feature_selector=selector)
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cols_by_prefix(feature_cols, *prefixes):
    """Return elements of feature_cols that start with any of the given prefixes."""
    return [c for c in feature_cols if any(c.startswith(p) for p in prefixes)]


# ---------------------------------------------------------------------------
# 1. Feature group isolators (stateless)
# ---------------------------------------------------------------------------

def hog_only(merged_df, feature_cols, is_train=True):
    """
    Retain only HOG-PCA features (hog_pca_0 .. hog_pca_99).
    100 PCA-compressed components of the HOG descriptor.
    Encodes local edge orientations — captures body shape and texture profile.
    Use for ablation: isolates the contribution of shape information.
    """
    selected = _cols_by_prefix(feature_cols, "hog_pca_")
    print(f"  [hog_only] {len(selected)} / {len(feature_cols)} features retained")
    return merged_df, selected


def color_only(merged_df, feature_cols, is_train=True):
    """
    Retain only colour histogram features (color_0 .. color_95).
    96 bins across three channels (32 bins each).
    Use for ablation: isolates the contribution of colour information.
    """
    selected = _cols_by_prefix(feature_cols, "color_")
    print(f"  [color_only] {len(selected)} / {len(feature_cols)} features retained")
    return merged_df, selected


def additional_only(merged_df, feature_cols, is_train=True):
    """
    Retain only the 23 hand-crafted additional features (feat_0 .. feat_22).
    feat_1 is on a different scale (~70–160); all others are in [0, 1].
    Use for ablation: the weakest single-group baseline.
    """
    selected = _cols_by_prefix(feature_cols, "feat_")
    print(f"  [additional_only] {len(selected)} / {len(feature_cols)} features retained")
    return merged_df, selected


def resnet_only(merged_df, feature_cols, is_train=True):
    """
    Retain only ResNet features (resnet_0 .. resnet_2047).
    Only valid after merge_resnet_features() has been applied upstream.
    """
    selected = _cols_by_prefix(feature_cols, "resnet_")
    print(f"  [resnet_only] {len(selected)} / {len(feature_cols)} features retained")
    return merged_df, selected


def all_features(merged_df, feature_cols, is_train=True):
    """
    No filtering — retain all features currently in feature_cols.
    This is the default when no feature_selector is passed to get_datasets().
    """
    print(f"  [all_features] {len(feature_cols)} features retained")
    return merged_df, feature_cols


# ---------------------------------------------------------------------------
# 2. ResNet feature merging
#    ResNet CSVs join on image_path (not image_id) and have integer column
#    names (0..2047). This function handles the join and renaming.
# ---------------------------------------------------------------------------

def merge_resnet_features(merged_df: pd.DataFrame,
                          feature_cols: list,
                          resnet_csv: str) -> tuple:
    """
    Load a ResNet feature CSV and left-join it onto merged_df via image_path.

    The ResNet CSVs use integer column names (0, 1, ..., 2047). These are
    renamed to resnet_0, resnet_1, ..., resnet_2047 to avoid collisions with
    any integer-indexed columns and to make prefix-based selection unambiguous.

    Parameters
    ----------
    merged_df    : DataFrame already containing metadata + hand-crafted features
    feature_cols : current feature column list
    resnet_csv   : path to the ResNet CSV file (train or test, matched externally)

    Returns
    -------
    (merged_df, feature_cols) with resnet_* columns appended
    """
    resnet_df = pd.read_csv(resnet_csv)

    # Rename integer columns to resnet_*
    rename_map = {c: f"resnet_{c}" for c in resnet_df.columns if c != "image_path"}
    resnet_df = resnet_df.rename(columns=rename_map)

    resnet_cols = [c for c in resnet_df.columns if c.startswith("resnet_")]

    # Left join on image_path; image_path must exist in merged_df
    if "image_path" not in merged_df.columns:
        raise ValueError("merged_df does not contain 'image_path'; cannot join ResNet features.")

    merged_df = pd.merge(merged_df, resnet_df, on="image_path", how="left")

    n_missing = merged_df[resnet_cols].isnull().any(axis=1).sum()
    if n_missing > 0:
        print(f"  Warning: {n_missing} rows have no ResNet features after join.")

    print(f"  [merge_resnet_features] Added {len(resnet_cols)} ResNet columns "
          f"from '{resnet_csv}'")
    return merged_df, feature_cols + resnet_cols


def make_resnet_merger(resnet_train_csv: str, resnet_test_csv: str):
    """
    Return a stateless selector that merges the correct ResNet CSV depending
    on whether is_train is True or False.

    This is the safe way to add ResNet features inside a compose() pipeline,
    since compose() forwards is_train to each step.

    Parameters
    ----------
    resnet_train_csv : path to ResNet features for the training set
    resnet_test_csv  : path to ResNet features for the test set
    """
    def _merge(merged_df, feature_cols, is_train=True):
        # Skip if ResNet columns already present (idempotent)
        if any(c.startswith("resnet_") for c in feature_cols):
            return merged_df, feature_cols
        csv = resnet_train_csv if is_train else resnet_test_csv
        return merge_resnet_features(merged_df, feature_cols, resnet_csv=csv)
    return _merge


# ---------------------------------------------------------------------------
# 3. Engineered feature additions (all stateless)
# ---------------------------------------------------------------------------

def add_color_channel_ratios(merged_df, feature_cols, is_train=True):
    """
    Derive per-channel means from the 96 colour histogram bins
    (assumed layout: 32 bins per channel, channels 0/1/2) and compute
    pairwise channel ratios and a saturation proxy.

    New columns: feat_eng_ch{i}_mean (×3), feat_eng_ch{ij}_ratio (×3),
                 feat_eng_saturation

    Motivation: raw bin values are lighting-sensitive; ratios between channel
    means are a hue proxy that is more robust to global illumination changes.
    """
    ch0 = [f"color_{i}" for i in range( 0, 32) if f"color_{i}" in merged_df.columns]
    ch1 = [f"color_{i}" for i in range(32, 64) if f"color_{i}" in merged_df.columns]
    ch2 = [f"color_{i}" for i in range(64, 96) if f"color_{i}" in merged_df.columns]

    if not (ch0 and ch1 and ch2):
        print("  [add_color_channel_ratios] Insufficient colour columns; skipping")
        return merged_df, feature_cols

    merged_df = merged_df.copy()
    c0 = merged_df[ch0].mean(axis=1)
    c1 = merged_df[ch1].mean(axis=1)
    c2 = merged_df[ch2].mean(axis=1)

    merged_df["feat_eng_ch0_mean"]   = c0
    merged_df["feat_eng_ch1_mean"]   = c1
    merged_df["feat_eng_ch2_mean"]   = c2
    merged_df["feat_eng_ch01_ratio"] = c0 / (c1 + 1e-6)
    merged_df["feat_eng_ch02_ratio"] = c0 / (c2 + 1e-6)
    merged_df["feat_eng_ch12_ratio"] = c1 / (c2 + 1e-6)
    merged_df["feat_eng_saturation"] = (
        pd.concat([c0, c1, c2], axis=1).max(axis=1) -
        pd.concat([c0, c1, c2], axis=1).min(axis=1)
    )
    new_cols = ["feat_eng_ch0_mean", "feat_eng_ch1_mean", "feat_eng_ch2_mean",
                "feat_eng_ch01_ratio", "feat_eng_ch02_ratio", "feat_eng_ch12_ratio",
                "feat_eng_saturation"]
    print(f"  [add_color_channel_ratios] Added {len(new_cols)} features")
    return merged_df, feature_cols + new_cols


def add_color_histogram_stats(merged_df, feature_cols, is_train=True):
    """
    Compute distributional statistics across all 96 colour histogram bins:
    variance, skewness, Shannon entropy, and dominant bin index.

    New columns: feat_eng_color_var, feat_eng_color_skew,
                 feat_eng_color_entropy, feat_eng_color_peak

    These scalars summarise the shape of the full colour distribution without
    adding 96 dimensions. Entropy distinguishes animals with pure uniform colour
    (low entropy) from those with complex multicoloured patterns (high entropy).
    """
    color_cols = _cols_by_prefix(feature_cols, "color_")
    if not color_cols:
        print("  [add_color_histogram_stats] No color_ columns; skipping")
        return merged_df, feature_cols

    merged_df = merged_df.copy()
    H = merged_df[color_cols].values.astype(float)
    eps = 1e-9

    merged_df["feat_eng_color_var"]     = H.var(axis=1)
    merged_df["feat_eng_color_skew"]    = pd.DataFrame(H).skew(axis=1).values
    merged_df["feat_eng_color_entropy"] = -(H * np.log(H + eps)).sum(axis=1)
    merged_df["feat_eng_color_peak"]    = H.argmax(axis=1).astype(float)

    new_cols = ["feat_eng_color_var", "feat_eng_color_skew",
                "feat_eng_color_entropy", "feat_eng_color_peak"]
    print(f"  [add_color_histogram_stats] Added {len(new_cols)} features")
    return merged_df, feature_cols + new_cols


def add_hog_statistics(merged_df, feature_cols, is_train=True):
    """
    Compute summary statistics across the 100 HOG-PCA components:
    L2 norm (overall edge energy), variance, and skewness.

    New columns: feat_eng_hog_l2norm, feat_eng_hog_var, feat_eng_hog_skew

    The L2 norm correlates with how edge-rich (textured) an image is.
    Animals with high-contrast patterns (butterfly wings, cat fur) have
    systematically higher norms than smooth-skinned animals.
    """
    hog_cols = _cols_by_prefix(feature_cols, "hog_pca_")
    if not hog_cols:
        print("  [add_hog_statistics] No hog_pca_ columns; skipping")
        return merged_df, feature_cols

    merged_df = merged_df.copy()
    H = merged_df[hog_cols].values.astype(float)

    merged_df["feat_eng_hog_l2norm"] = np.linalg.norm(H, axis=1)
    merged_df["feat_eng_hog_var"]    = H.var(axis=1)
    merged_df["feat_eng_hog_skew"]   = pd.DataFrame(H).skew(axis=1).values

    new_cols = ["feat_eng_hog_l2norm", "feat_eng_hog_var", "feat_eng_hog_skew"]
    print(f"  [add_hog_statistics] Added {len(new_cols)} features")
    return merged_df, feature_cols + new_cols


def add_feat_statistics(merged_df, feature_cols, is_train=True):
    """
    Compute summary statistics over the 23 additional features.
    feat_1 is excluded from the mean/std computation due to its different
    scale (~70–160 vs [0, 1] for all other feat_* columns), and instead
    added as log(feat_1 + 1) to compress it to a comparable scale.

    New columns: feat_eng_additional_mean, feat_eng_additional_std,
                 feat_eng_feat1_log
    """
    feat_cols = _cols_by_prefix(feature_cols, "feat_")
    if not feat_cols:
        print("  [add_feat_statistics] No feat_ columns; skipping")
        return merged_df, feature_cols

    merged_df = merged_df.copy()
    new_cols  = []

    norm_cols = [c for c in feat_cols if c != "feat_1"]
    if norm_cols:
        F = merged_df[norm_cols].values.astype(float)
        merged_df["feat_eng_additional_mean"] = F.mean(axis=1)
        merged_df["feat_eng_additional_std"]  = F.std(axis=1)
        new_cols += ["feat_eng_additional_mean", "feat_eng_additional_std"]

    if "feat_1" in merged_df.columns:
        merged_df["feat_eng_feat1_log"] = np.log1p(merged_df["feat_1"])
        new_cols.append("feat_eng_feat1_log")

    print(f"  [add_feat_statistics] Added {len(new_cols)} features")
    return merged_df, feature_cols + new_cols


def add_hog_color_cross_features(n_hog: int = 5, n_color: int = 5):
    """
    Return a selector that creates pairwise products between the leading
    HOG-PCA components and the leading colour histogram bins.

    Parameters
    ----------
    n_hog   : leading HOG components to use (keep ≤ 5 to control dimensionality)
    n_color : leading colour bins to use

    New columns: feat_eng_cross_hog_pca_{i}_color_{j}

    Motivation: cross features encode animals that are simultaneously a
    particular shape AND a particular colour — a conjunction that neither
    HOG nor colour features alone can represent.
    """
    def _cross(merged_df, feature_cols, is_train=True):
        hog_cols   = _cols_by_prefix(feature_cols, "hog_pca_")[:n_hog]
        color_cols = _cols_by_prefix(feature_cols, "color_")[:n_color]

        if not hog_cols or not color_cols:
            print("  [add_hog_color_cross] Insufficient columns; skipping")
            return merged_df, feature_cols

        merged_df_out = merged_df.copy()
        new_cols = []
        for hc in hog_cols:
            for cc in color_cols:
                name = f"feat_eng_cross_{hc}_{cc}"
                merged_df_out[name] = merged_df_out[hc] * merged_df_out[cc]
                new_cols.append(name)

        print(f"  [add_hog_color_cross] Added {len(new_cols)} cross features "
              f"({n_hog} HOG × {n_color} colour)")
        return merged_df_out, feature_cols + new_cols

    return _cross


def add_resnet_pca(n_components: int = 200):
    """
    Return a stateful selector that applies PCA to the ResNet features
    (resnet_0..resnet_2047) in-place and replaces them with n_components
    principal components.

    This is appropriate for kNN and SVM where the full 2048-d ResNet space
    is expensive and redundant. For tree-based models, prefer raw ResNet
    features (no PCA) since PCA-rotated axes lose axis-aligned interpretability.

    Parameters
    ----------
    n_components : number of PCA components to retain from the 2048-d ResNet space

    The PCA is fitted on training data only (is_train=True) and applied
    consistently to test data (is_train=False).
    """
    state = {"pca": None}

    def _apply(merged_df, feature_cols, is_train=True):
        resnet_cols = _cols_by_prefix(feature_cols, "resnet_")
        if not resnet_cols:
            print("  [add_resnet_pca] No resnet_ columns found; skipping")
            return merged_df, feature_cols

        merged_df = merged_df.copy()
        X_resnet = merged_df[resnet_cols].values.astype(float)
        actual_k = min(n_components, X_resnet.shape[1], X_resnet.shape[0] - 1)

        if is_train:
            pca = PCA(n_components=actual_k)
            X_pca = pca.fit_transform(X_resnet)
            state["pca"] = pca
            explained = pca.explained_variance_ratio_.sum()
            print(f"  [add_resnet_pca] PCA: {actual_k} components, "
                  f"{explained * 100:.1f}% variance explained")
        else:
            if state["pca"] is None:
                raise RuntimeError("[add_resnet_pca] Must be called with is_train=True first.")
            X_pca = state["pca"].transform(X_resnet)
            print(f"  [add_resnet_pca] Applied fitted PCA ({actual_k} components) to test data")

        pca_col_names = [f"resnet_pca_{i}" for i in range(X_pca.shape[1])]
        pca_df = pd.DataFrame(X_pca, columns=pca_col_names, index=merged_df.index)
        merged_df = pd.concat([merged_df.drop(columns=resnet_cols), pca_df], axis=1)

        # Update feature_cols: remove raw resnet_* and add resnet_pca_*
        non_resnet = [c for c in feature_cols if not c.startswith("resnet_")]
        return merged_df, non_resnet + pca_col_names

    return _apply


# ---------------------------------------------------------------------------
# 4. Statistical feature selection (stateful closures)
# ---------------------------------------------------------------------------

def select_k_best_anova(k: int = 100):
    """
    Retain the top-k features by ANOVA F-score (between-class variance /
    within-class variance). Fast and appropriate for continuous features
    with approximately normal within-class distributions.

    The mask is fitted on training data (is_train=True) and applied to
    all subsequent calls (is_train=False) using the same closure instance.

    Do not reuse a closure instance across separate get_datasets() calls.
    """
    state = {"mask": None}

    def _select(merged_df, feature_cols, is_train=True):
        if is_train:
            if "class_id" not in merged_df.columns:
                print("  [select_k_best_anova] No labels available; skipping")
                return merged_df, feature_cols
            X = merged_df[feature_cols].values.astype(float)
            y = merged_df["class_id"].values
            actual_k = min(k, X.shape[1])
            skb = SelectKBest(f_classif, k=actual_k)
            skb.fit(X, y)
            state["mask"] = skb.get_support()
            print(f"  [select_k_best_anova] {actual_k} / {X.shape[1]} features selected")
        else:
            if state["mask"] is None:
                print("  [select_k_best_anova] Not yet fitted; returning all features")
                return merged_df, feature_cols

        selected = [c for c, keep in zip(feature_cols, state["mask"]) if keep]
        return merged_df, selected

    return _select


def select_k_best_mutual_info(k: int = 100):
    """
    Retain the top-k features by mutual information with the class label.
    Non-parametric — captures non-linear feature-class associations.
    Slower than ANOVA but more reliable for HOG-PCA components and ResNet
    activations, whose class signals are typically non-monotonic.

    The mask is fitted on training data and re-applied to test data.
    """
    state = {"mask": None}

    def _select(merged_df, feature_cols, is_train=True):
        if is_train:
            if "class_id" not in merged_df.columns:
                return merged_df, feature_cols
            X = merged_df[feature_cols].values.astype(float)
            y = merged_df["class_id"].values
            actual_k = min(k, X.shape[1])
            skb = SelectKBest(mutual_info_classif, k=actual_k)
            skb.fit(X, y)
            state["mask"] = skb.get_support()
            print(f"  [select_k_best_mutual_info] {actual_k} / {X.shape[1]} features selected")
        else:
            if state["mask"] is None:
                return merged_df, feature_cols

        selected = [c for c, keep in zip(feature_cols, state["mask"]) if keep]
        return merged_df, selected

    return _select


# ---------------------------------------------------------------------------
# 5. Composition utility
# ---------------------------------------------------------------------------

def compose(*selectors):
    """
    Chain multiple feature selector callables left-to-right.
    Each selector receives the (merged_df, feature_cols) output of the previous.
    is_train is forwarded to every step so stateful selectors fit correctly.

    Always call compose() fresh in each model script to get independent
    closure state. Do not share a composed selector across model scripts.
    """
    def _composed(merged_df, feature_cols, is_train=True):
        for sel in selectors:
            merged_df, feature_cols = sel(merged_df, feature_cols, is_train=is_train)
        return merged_df, feature_cols
    return _composed


# ---------------------------------------------------------------------------
# 6. Named factory functions
#    These return fresh selector instances. Call them in each model script.
# ---------------------------------------------------------------------------

def make_resnet_selector(resnet_train_csv: str,
                         resnet_test_csv: str,
                         use_pca: bool = True,
                         pca_k: int = 200,
                         selection_k: int = 150,
                         selection_method: str = "anova"):
    """
    Factory: build a complete feature pipeline that includes ResNet features.

    Workflow
    --------
    1. Start with all 219 hand-crafted features
    2. Add engineered features (colour ratios, histogram stats, HOG stats, feat stats)
    3. Merge ResNet features (2048-d) from the appropriate CSV
    4. Optionally compress ResNet to pca_k dimensions via PCA
    5. Select top selection_k features by ANOVA or mutual info

    Parameters
    ----------
    resnet_train_csv  : path to the ResNet CSV for training images
    resnet_test_csv   : path to the ResNet CSV for test images
    use_pca           : if True, compress ResNet with PCA before selection
                        Set False for tree-based models (preserves axis alignment)
    pca_k             : PCA components to retain from ResNet (if use_pca=True)
    selection_k       : final number of features after SelectKBest
    selection_method  : "anova" or "mutual_info"

    Returns a fresh compose() pipeline (new closure state each call).
    """
    resnet_merger = make_resnet_merger(resnet_train_csv, resnet_test_csv)
    selector = select_k_best_anova(k=selection_k) if selection_method == "anova" \
               else select_k_best_mutual_info(k=selection_k)

    steps = [
        all_features,
        add_color_channel_ratios,
        add_color_histogram_stats,
        add_hog_statistics,
        add_feat_statistics,
        resnet_merger,
    ]

    if use_pca:
        steps.append(add_resnet_pca(n_components=pca_k))

    steps.append(selector)
    return compose(*steps)


# ---------------------------------------------------------------------------
# 7. Ready-to-use named configurations (no ResNet)
# ---------------------------------------------------------------------------

# All 219 provided features, no modifications
baseline = all_features

# All features + engineered additions (~236 features, no selection)
with_engineering = compose(
    all_features,
    add_color_channel_ratios,
    add_color_histogram_stats,
    add_hog_statistics,
    add_feat_statistics,
)

# Full engineering + cross features + mutual-info top-120
# Good default for kNN and SVM (no ResNet)
combined_selector = compose(
    all_features,
    add_color_channel_ratios,
    add_color_histogram_stats,
    add_hog_statistics,
    add_feat_statistics,
    add_hog_color_cross_features(n_hog=5, n_color=5),
    select_k_best_mutual_info(k=120),
)

# HOG + stats + ANOVA-60; shape-focused, tree-friendly
hog_with_stats = compose(
    hog_only,
    add_hog_statistics,
    select_k_best_anova(k=60),
)

# Colour + channel ratios + distributional stats + ANOVA-50
color_with_stats = compose(
    color_only,
    add_color_channel_ratios,
    add_color_histogram_stats,
    select_k_best_anova(k=50),
)
