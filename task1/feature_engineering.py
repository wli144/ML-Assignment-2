"""
feature_engineering.py
-----------------------
Defines feature selectors and transformations passed as the
`feature_selector` argument to data_loader.get_datasets().

Column prefixes in the actual data
-----------------------------------
  additional_features.csv  ->  feat_0  .. feat_22       (23 columns)
  hog_pca.csv              ->  hog_pca_0 .. hog_pca_99  (100 columns)
  color_histogram.csv      ->  color_0 .. color_95       (96 columns)
                                                   Total: 219 features

A feature selector is any callable with this signature:

    def my_selector(merged_df, feature_cols, is_train=True):
        -> (merged_df, feature_cols)

The merged_df is the full DataFrame after the metadata/feature join.
feature_cols is the current list of active feature column names.
Return the (possibly modified) df and the updated feature_cols list.

Usage example
-------------
    from data_loader import load_raw_data, get_datasets
    from feature_engineering import combined_selector

    raw      = load_raw_data("task1_data")
    datasets = get_datasets(raw, feature_selector=combined_selector)
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cols_by_prefix(feature_cols, *prefixes):
    """Return feature_cols that start with any of the given prefixes."""
    return [c for c in feature_cols if any(c.startswith(p) for p in prefixes)]


# ---------------------------------------------------------------------------
# 1. Feature group isolators
#    Use for ablation: find out which feature type contributes most.
# ---------------------------------------------------------------------------

def hog_only(merged_df, feature_cols, is_train=True):
    """
    Retain only HOG-PCA features (hog_pca_0 .. hog_pca_99).

    HOG encodes local edge orientations. These 100 PCA-compressed components
    capture the shape and texture profile of the animal silhouette, which
    tends to be stable across lighting conditions.

    When to use: classes separated primarily by body shape
    (e.g., spider vs. butterfly vs. squirrel).
    """
    selected = _cols_by_prefix(feature_cols, "hog_pca_")
    print(f"  [hog_only] {len(selected)} / {len(feature_cols)} features retained")
    return merged_df, selected


def color_only(merged_df, feature_cols, is_train=True):
    """
    Retain only colour histogram features (color_0 .. color_95).

    The 96 bins cover three channels (likely R, G, B or HSV), 32 bins each.
    Colour distributions are highly discriminative for animals with distinctive
    colouration (e.g., butterfly wing colours, frog skin tones).

    When to use: colour is the dominant inter-class discriminator;
    expect lower performance on classes that differ only in shape.
    """
    selected = _cols_by_prefix(feature_cols, "color_")
    print(f"  [color_only] {len(selected)} / {len(feature_cols)} features retained")
    return merged_df, selected


def additional_only(merged_df, feature_cols, is_train=True):
    """
    Retain only the additional hand-crafted features (feat_0 .. feat_22).

    These 23 features are compact and domain-specific. Their exact semantics
    are unspecified, but the range [0, 1] for most columns and the one
    column with values in [70, 160] suggests a mix of ratio-based descriptors
    and a global statistic (possibly mean brightness or contrast).

    When to use: as a lightweight low-dimensional baseline; useful to
    quantify what the hand-crafted features add over raw HOG/colour.
    """
    selected = _cols_by_prefix(feature_cols, "feat_")
    print(f"  [additional_only] {len(selected)} / {len(feature_cols)} features retained")
    return merged_df, selected


def all_features(merged_df, feature_cols, is_train=True):
    """
    No filtering — use all 219 provided features.
    This is the default when no feature_selector is passed to get_datasets().
    """
    print(f"  [all_features] {len(feature_cols)} features retained")
    return merged_df, feature_cols


# ---------------------------------------------------------------------------
# 2. Engineered feature additions
#    These compute new columns from existing ones and append them.
#    They do not remove any existing features.
# ---------------------------------------------------------------------------

def add_color_channel_ratios(merged_df, feature_cols, is_train=True):
    """
    Derive per-channel summary statistics from the colour histogram bins
    and compute inter-channel ratios.

    The 96 colour bins are assumed to be ordered as three 32-bin channel
    histograms (indices 0-31, 32-63, 64-95). Channel means approximate the
    average intensity per channel and their ratios are a proxy for hue —
    illumination-robust relative to raw bin values.

    New columns
    -----------
    feat_eng_ch0_mean   : mean of bins 0-31  (channel 0)
    feat_eng_ch1_mean   : mean of bins 32-63 (channel 1)
    feat_eng_ch2_mean   : mean of bins 64-95 (channel 2)
    feat_eng_ch01_ratio : ch0_mean / (ch1_mean + 1e-6)
    feat_eng_ch02_ratio : ch0_mean / (ch2_mean + 1e-6)
    feat_eng_ch12_ratio : ch1_mean / (ch2_mean + 1e-6)
    feat_eng_saturation : max(ch_means) - min(ch_means)  — colour purity proxy

    Motivation: a yellow butterfly and a green frog may overlap in individual
    bin values, but their channel mean ratios will be clearly distinct.
    """
    ch0_cols = [f"color_{i}" for i in range(0,  32) if f"color_{i}" in merged_df.columns]
    ch1_cols = [f"color_{i}" for i in range(32, 64) if f"color_{i}" in merged_df.columns]
    ch2_cols = [f"color_{i}" for i in range(64, 96) if f"color_{i}" in merged_df.columns]

    if not (ch0_cols and ch1_cols and ch2_cols):
        print("  [add_color_channel_ratios] Insufficient colour columns; skipping")
        return merged_df, feature_cols

    merged_df = merged_df.copy()
    c0 = merged_df[ch0_cols].mean(axis=1)
    c1 = merged_df[ch1_cols].mean(axis=1)
    c2 = merged_df[ch2_cols].mean(axis=1)

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

    new_cols = [
        "feat_eng_ch0_mean", "feat_eng_ch1_mean", "feat_eng_ch2_mean",
        "feat_eng_ch01_ratio", "feat_eng_ch02_ratio", "feat_eng_ch12_ratio",
        "feat_eng_saturation",
    ]
    print(f"  [add_color_channel_ratios] Added {len(new_cols)} features")
    return merged_df, feature_cols + new_cols


def add_color_histogram_stats(merged_df, feature_cols, is_train=True):
    """
    Compute distributional statistics across all 96 colour histogram bins:
    variance, skewness (third standardised moment), and the index of the
    dominant bin (argmax).

    New columns
    -----------
    feat_eng_color_var     : variance across all 96 bins — measures how
                             concentrated the colour distribution is
    feat_eng_color_skew    : skewness of the 96-bin distribution — positive
                             skew indicates most mass in darker bins
    feat_eng_color_entropy : Shannon entropy of the histogram — low entropy
                             means a narrow/pure colour profile
    feat_eng_color_peak    : bin index of the global maximum — coarse dominant
                             colour location

    Motivation: a plain grey squirrel has low colour variance and low entropy;
    a multicoloured butterfly has high variance and high entropy. These scalars
    summarise the shape of the full histogram without adding 96 dimensions.
    """
    color_cols = _cols_by_prefix(feature_cols, "color_")
    if not color_cols:
        print("  [add_color_histogram_stats] No color columns found; skipping")
        return merged_df, feature_cols

    merged_df = merged_df.copy()
    H = merged_df[color_cols].values.astype(float)

    merged_df["feat_eng_color_var"]     = H.var(axis=1)
    merged_df["feat_eng_color_skew"]    = (
        pd.DataFrame(H).skew(axis=1).values
    )
    # Shannon entropy: -sum(p * log(p+eps))
    eps = 1e-9
    merged_df["feat_eng_color_entropy"] = -(H * np.log(H + eps)).sum(axis=1)
    merged_df["feat_eng_color_peak"]    = H.argmax(axis=1).astype(float)

    new_cols = [
        "feat_eng_color_var", "feat_eng_color_skew",
        "feat_eng_color_entropy", "feat_eng_color_peak",
    ]
    print(f"  [add_color_histogram_stats] Added {len(new_cols)} features")
    return merged_df, feature_cols + new_cols


def add_hog_statistics(merged_df, feature_cols, is_train=True):
    """
    Compute summary statistics across the 100 HOG-PCA components:
    L2 norm, variance, and the index of the dominant component.

    New columns
    -----------
    feat_eng_hog_l2norm   : L2 norm of the HOG-PCA vector — overall edge energy
    feat_eng_hog_var      : variance across components — spread of edge activity
    feat_eng_hog_skew     : skewness of PCA component values

    Motivation: the L2 norm of the HOG vector correlates with how
    edge-rich (textured) an image is. Animals with high-contrast fur or
    wing patterns (cats, butterflies) will have systematically higher norms
    than smooth-skinned animals (frogs). These are cheap to compute and
    add a global shape signal complementary to the per-component values.
    """
    hog_cols = _cols_by_prefix(feature_cols, "hog_pca_")
    if not hog_cols:
        print("  [add_hog_statistics] No hog_pca columns found; skipping")
        return merged_df, feature_cols

    merged_df = merged_df.copy()
    H = merged_df[hog_cols].values.astype(float)

    merged_df["feat_eng_hog_l2norm"] = np.linalg.norm(H, axis=1)
    merged_df["feat_eng_hog_var"]    = H.var(axis=1)
    merged_df["feat_eng_hog_skew"]   = pd.DataFrame(H).skew(axis=1).values

    new_cols = ["feat_eng_hog_l2norm", "feat_eng_hog_var", "feat_eng_hog_skew"]
    print(f"  [add_hog_statistics] Added {len(new_cols)} features")
    return merged_df, feature_cols + new_cols


def add_hog_color_cross_features(n_hog: int = 5, n_color: int = 5):
    """
    Return a selector that creates pairwise products between the leading
    HOG-PCA components and the leading colour histogram bins.

    Parameters
    ----------
    n_hog   : number of leading hog_pca_* components to use (default 5)
    n_color : number of leading color_* bins to use (default 5)

    New columns: feat_eng_cross_hog{i}_color{j}  (n_hog × n_color total)

    Motivation: a cross feature between hog_pca_0 (dominant shape mode) and
    color_48 (a mid-range bin) encodes animals that are simultaneously a
    particular shape AND a particular colour. For example, a bird might have
    a distinctive combination of a streamlined silhouette and bright plumage.
    Pure shape or colour features cannot represent this conjunction.

    Note: cross features grow quadratically — keep n_hog and n_color small
    (≤5 each) to avoid adding noise features that harm generalisation.
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


def add_feat_statistics(merged_df, feature_cols, is_train=True):
    """
    Compute summary statistics across the 23 additional features (feat_0..feat_22).

    feat_0 appears to be edge density (range ~0.04–0.12).
    feat_1 appears to be a global contrast or mean intensity statistic
    (range ~70–160, on a different scale from the rest).
    feat_2 onwards are normalised descriptors in [0, 1].

    New columns
    -----------
    feat_eng_additional_mean : mean of feat_2..feat_22 (excluding the outlier
                               scale of feat_1)
    feat_eng_additional_std  : standard deviation of feat_2..feat_22
    feat_eng_feat1_log       : log(feat_1) to compress its scale to match others

    Motivation: normalising feat_1 prevents it from dominating distance-based
    models (kNN, RBF-SVM) even after StandardScaler, since its raw variance
    is much larger than the other feat_* columns.
    """
    feat_cols = _cols_by_prefix(feature_cols, "feat_")
    if not feat_cols:
        print("  [add_feat_statistics] No feat_ columns found; skipping")
        return merged_df, feature_cols

    merged_df = merged_df.copy()
    new_cols  = []

    # Summary stats over the normalised subset (exclude feat_1 outlier scale)
    norm_cols = [c for c in feat_cols if c != "feat_1"]
    if norm_cols:
        F = merged_df[norm_cols].values.astype(float)
        merged_df["feat_eng_additional_mean"] = F.mean(axis=1)
        merged_df["feat_eng_additional_std"]  = F.std(axis=1)
        new_cols += ["feat_eng_additional_mean", "feat_eng_additional_std"]

    # Log-compress the large-scale feature
    if "feat_1" in merged_df.columns:
        merged_df["feat_eng_feat1_log"] = np.log1p(merged_df["feat_1"])
        new_cols.append("feat_eng_feat1_log")

    print(f"  [add_feat_statistics] Added {len(new_cols)} features")
    return merged_df, feature_cols + new_cols


# ---------------------------------------------------------------------------
# 3. Statistical feature selection
#    These reduce the feature set by scoring against training labels.
#    The selector state (which columns to keep) is fitted on train and
#    re-applied identically on val/test.
# ---------------------------------------------------------------------------

def select_k_best_anova(k: int = 100):
    """
    Retain the top-k features ranked by ANOVA F-score.

    ANOVA F-score measures the ratio of between-class variance to
    within-class variance for each feature independently. It is fast
    and appropriate for continuous features with roughly normal
    within-class distributions.

    Suitable for: colour histogram bins (approximately normal within a class),
    and normalised feat_* descriptors.
    Less suitable for: HOG-PCA components, which can be bimodal.

    Parameters
    ----------
    k : number of features to retain; set to "all" to rank without truncating
    """
    state = {"mask": None}

    def _select(merged_df, feature_cols, is_train=True):
        if is_train:
            if "class_id" not in merged_df.columns:
                print("  [select_k_best_anova] No labels in training data; skipping")
                return merged_df, feature_cols
            X = merged_df[feature_cols].values
            y = merged_df["class_id"].values
            actual_k = min(k, X.shape[1])
            skb = SelectKBest(f_classif, k=actual_k)
            skb.fit(X, y)
            state["mask"] = skb.get_support()
            print(f"  [select_k_best_anova] {actual_k} / {X.shape[1]} features selected")
        else:
            if state["mask"] is None:
                print("  [select_k_best_anova] Selector not yet fitted; returning all")
                return merged_df, feature_cols

        selected = [c for c, keep in zip(feature_cols, state["mask"]) if keep]
        return merged_df, selected

    return _select


def select_k_best_mutual_info(k: int = 100):
    """
    Retain the top-k features ranked by mutual information with the class label.

    Mutual information is non-parametric: it captures non-linear associations
    without assuming a particular distribution. It is more appropriate than
    ANOVA for HOG-PCA components, which may have non-monotonic class signals.

    Slower than ANOVA (uses nearest-neighbour estimation) but more reliable
    when the feature-class relationship is non-linear.

    Parameters
    ----------
    k : number of features to retain
    """
    state = {"mask": None}

    def _select(merged_df, feature_cols, is_train=True):
        if is_train:
            if "class_id" not in merged_df.columns:
                return merged_df, feature_cols
            X = merged_df[feature_cols].values
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
# 4. Composition utility
# ---------------------------------------------------------------------------

def compose(*selectors):
    """
    Chain multiple feature selector callables left-to-right.

    Each selector receives the output (merged_df, feature_cols) of the
    previous one. is_train is forwarded to every selector so that stateful
    ones (select_k_best_*) fit only during training.

    Usage
    -----
    pipeline = compose(
        all_features,
        add_color_channel_ratios,
        add_hog_statistics,
        select_k_best_mutual_info(k=120),
    )
    datasets = get_datasets(raw, feature_selector=pipeline)
    """
    def _composed(merged_df, feature_cols, is_train=True):
        for sel in selectors:
            merged_df, feature_cols = sel(merged_df, feature_cols, is_train=is_train)
        return merged_df, feature_cols
    return _composed



# ---------------------------------------------------------------------------
# 5. Named ready-to-use configurations
# ---------------------------------------------------------------------------

# --- Baselines (no engineering) ---

# All 219 provided features, no modifications
baseline = all_features

# Single feature-type baselines for ablation
hog_baseline   = hog_only
color_baseline = color_only
feat_baseline  = additional_only


# --- Engineered, no selection ---
# Adds 14 new features on top of all 219; total ~233 features.
# Use this to verify that engineered features improve over baseline before
# adding a selection step.
with_engineering = compose(
    all_features,
    add_color_channel_ratios,
    add_color_histogram_stats,
    add_hog_statistics,
    add_feat_statistics,
)

# --- Recommended default ---
# Full engineering + 5×5 cross features + mutual-info selection to top 120.
# Balances expressiveness with dimensionality. Good starting point for
# SVM (RBF) and kNN where feature count affects runtime.
combined_selector = compose(
    #all_features,
    add_color_channel_ratios,
    add_color_histogram_stats,
    add_hog_statistics,
    add_feat_statistics,
    add_hog_color_cross_features(n_hog=5, n_color=5),
    select_k_best_mutual_info(k=120),
)

# --- Shape-focused (for tree-based models) ---
# HOG + engineered HOG stats; ANOVA selection.
# Trees benefit from fewer, highly discriminative features.
hog_with_stats = compose(
    hog_only,
    add_hog_statistics,
    select_k_best_anova(k=60),
)

# --- Colour-focused ---
# Colour histograms + channel ratios + distributional stats.
# Useful if classes are primarily colour-differentiated.
color_with_stats = compose(
    color_only,
    add_color_channel_ratios,
    add_color_histogram_stats,
    select_k_best_anova(k=50),
)
