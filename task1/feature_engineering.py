import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif


def _cols(feature_cols, *prefixes):
    return [c for c in feature_cols if any(c.startswith(p) for p in prefixes)]


# ---------------------------------------------------------------------------
# ResNet merging
# ---------------------------------------------------------------------------

def _merge_resnet(merged_df, feature_cols, resnet_csv):
    if any(c.startswith("resnet_") for c in feature_cols):
        return merged_df, feature_cols
    df = pd.read_csv(resnet_csv)
    df = df.rename(columns={c: f"resnet_{c}" for c in df.columns if c != "image_path"})
    resnet_cols = [c for c in df.columns if c.startswith("resnet_")]
    merged_df = pd.merge(merged_df, df, on="image_path", how="left")
    missing = merged_df[resnet_cols].isnull().any(axis=1).sum()
    if missing:
        print(f"  Warning: {missing} rows missing ResNet features after join.")
    return merged_df, feature_cols + resnet_cols


def make_resnet_merger(train_csv, test_csv):
    def _apply(merged_df, feature_cols, is_train=True):
        return _merge_resnet(merged_df, feature_cols, train_csv if is_train else test_csv)
    return _apply


# ---------------------------------------------------------------------------
# ResNet PCA  (stateful — fits on train, applies to test)
# ---------------------------------------------------------------------------

def make_resnet_pca(n_components=200):
    state = {}

    def _apply(merged_df, feature_cols, is_train=True):
        resnet_cols = _cols(feature_cols, "resnet_")
        if not resnet_cols:
            return merged_df, feature_cols

        merged_df = merged_df.copy()
        X = merged_df[resnet_cols].values.astype(float)
        k = min(n_components, X.shape[1], X.shape[0] - 1)

        if is_train:
            pca = PCA(n_components=k)
            X_pca = pca.fit_transform(X)
            state["pca"] = pca
            print(f"  ResNet PCA: {k} components, {pca.explained_variance_ratio_.sum()*100:.1f}% variance")
        else:
            X_pca = state["pca"].transform(X)

        pca_names = [f"resnet_pca_{i}" for i in range(X_pca.shape[1])]
        pca_df = pd.DataFrame(X_pca, columns=pca_names, index=merged_df.index)
        merged_df = pd.concat([merged_df.drop(columns=resnet_cols), pca_df], axis=1)
        non_resnet = [c for c in feature_cols if not c.startswith("resnet_")]
        return merged_df, non_resnet + pca_names

    return _apply


# ---------------------------------------------------------------------------
# ANOVA feature selection  (stateful — fits on train, applies to test)
# ---------------------------------------------------------------------------

def make_anova_selector(k=150):
    state = {}

    def _apply(merged_df, feature_cols, is_train=True):
        if is_train:
            X = merged_df[feature_cols].values.astype(float)
            y = merged_df["class_id"].values
            k_actual = min(k, X.shape[1])
            skb = SelectKBest(f_classif, k=k_actual)
            skb.fit(X, y)
            state["mask"] = skb.get_support()
            print(f"  ANOVA: {k_actual} / {X.shape[1]} features selected")
        selected = [c for c, keep in zip(feature_cols, state["mask"]) if keep]
        return merged_df, selected

    return _apply


# ---------------------------------------------------------------------------
# Handcrafted feature group isolator
# ---------------------------------------------------------------------------

def _keep_handcrafted(merged_df, feature_cols, is_train=True):
    selected = _cols(feature_cols, "feat_", "hog_pca_", "color_")
    return merged_df, selected


# ---------------------------------------------------------------------------
# Pipeline composition
# ---------------------------------------------------------------------------

def _compose(*steps):
    def _run(merged_df, feature_cols, is_train=True):
        for step in steps:
            merged_df, feature_cols = step(merged_df, feature_cols, is_train=is_train)
        return merged_df, feature_cols
    return _run


# ---------------------------------------------------------------------------
# Public factory  — the only function model scripts need to call
#
# mode        : "resnet"       — ResNet features only (with PCA if use_pca=True)
#             : "handcrafted"  — feat_*, hog_pca_*, color_* only
#             : "combined"     — both, merged before ANOVA selection
# use_pca     : compress ResNet to pca_k dims before selection
#               True  → kNN, SVM  (required for RBF-SVM runtime)
#               False → Decision Tree, Random Forest  (preserves axis alignment)
# pca_k       : PCA components kept from the 2048-d ResNet space
# selection_k : top-k features kept by ANOVA after all other steps
# ---------------------------------------------------------------------------

def build_selector(mode, resnet_train_csv, resnet_test_csv,
                   use_pca=True, pca_k=200, selection_k=150):

    merger   = make_resnet_merger(resnet_train_csv, resnet_test_csv)
    resnet_pca = make_resnet_pca(n_components=pca_k)
    selector = make_anova_selector(k=selection_k)

    if mode == "resnet":
        steps = [merger]
        if use_pca:
            steps.append(resnet_pca)
        steps.append(selector)

    elif mode == "handcrafted":
        steps = [_keep_handcrafted, selector]

    elif mode == "combined":
        steps = [merger]
        if use_pca:
            steps.append(resnet_pca)
        steps += [_keep_handcrafted_and_resnet, selector]

    else:
        raise ValueError(f"mode must be 'resnet', 'handcrafted', or 'combined'. Got: {mode!r}")

    return _compose(*steps)


def _keep_handcrafted_and_resnet(merged_df, feature_cols, is_train=True):
    # After ResNet merger (and optional PCA), keep both handcrafted + resnet columns
    selected = _cols(feature_cols, "feat_", "hog_pca_", "color_", "resnet_")
    return merged_df, selected
