import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ==========================================================
# 1. LOAD AND PREPARE DATA
# ==========================================================
print("Loading datasets...")
# Load the original labels and your newly engineered ResNet features
metadata = pd.read_csv('train_metadata.csv')
features_df = pd.read_csv('resnet_features_train.csv')

# Merge them together on the image path to ensure perfect alignment
merged_df = pd.merge(metadata, features_df, on='image_path')

# Separate into features (X) and target labels (y)
# Adjust 'label' or 'class_id' to match your metadata column names
y = merged_df['class_id'] 
X = merged_df.drop(columns=['image_path', 'class_name', 'class_id'], errors='ignore')

# Split into Training and Validation sets (80% train, 20% val)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (Crucial before performing PCA!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print(f"Original Feature Matrix Shape: {X_train_scaled.shape} (2048 features)")

# ==========================================================
# 2. METHOD A: PRINCIPAL COMPONENT ANALYSIS (PCA)
# ==========================================================
print("\n--- Running PCA Reduction ---")
# n_components=0.95 means: "Keep enough components to capture 95% of the data's variance"
# This automatically drops redundant noise features.
pca = PCA(n_components=0.95, random_state=42)

X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)

print(f"PCA Reduced Matrix Shape: {X_train_pca.shape}")

# Test PCA features with a baseline Support Vector Machine (SVM)
svm_pca = SVC(kernel='linear')
svm_pca.fit(X_train_pca, y_train)
pca_preds = svm_pca.predict(X_val_pca)
print(f"SVM Accuracy with PCA features: {accuracy_score(y_val, pca_preds):.4f}")


# ==========================================================
# 3. METHOD B: SELECT K BEST (Feature Selection)
# ==========================================================
print("\n--- Running SelectKBest Selection ---")
# f_classif uses an ANOVA F-test to see which features correlate strongest with the labels.
# Let's select the top 100 most useful features out of the 2,048.
k_features = 100
selector = SelectKBest(score_func=f_classif, k=k_features)

X_train_kbest = selector.fit_transform(X_train_scaled, y_train)
X_val_kbest = selector.transform(X_val_scaled)

print(f"SelectKBest Matrix Shape: {X_train_kbest.shape}")

# Test SelectKBest features with the same baseline SVM
svm_kbest = SVC(kernel='linear')
svm_kbest.fit(X_train_kbest, y_train)
kbest_preds = svm_kbest.predict(X_val_kbest)
print(f"SVM Accuracy with SelectKBest ({k_features} features): {accuracy_score(y_val, kbest_preds):.4f}")