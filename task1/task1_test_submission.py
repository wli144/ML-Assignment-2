import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ==========================================================
# 1. LOAD TRAINING DATA (To train the model)
# ==========================================================
print("Loading training datasets...")
train_metadata = pd.read_csv('task1_data/train_metadata.csv')
train_features = pd.read_csv('resnet_features_hf.csv') # Your training features

# Merge training data
train_merged = pd.merge(train_metadata, train_features, on='image_path')
y_train = train_merged['class_id']
X_train = train_merged.drop(columns=['image_id', 'image_path', 'class_id', 'class_name'])

# ==========================================================
# 2. LOAD TEST DATA (To generate Kaggle predictions)
# ==========================================================
print("Loading test datasets...")
# NOTE: Make sure these filenames match your actual test feature/metadata file names!
test_metadata = pd.read_csv('task1_data/test_metadata.csv') 
test_features = pd.read_csv('resnet_features_test.csv') # Your test features

# Merge test data on image_path
test_merged = pd.merge(test_metadata, test_features, on='image_path')

# Extract test features (Drop metadata columns so it matches X_train perfectly)
# Note: Test metadata won't have 'class_id' or 'class_name', so we just drop what exists
X_test = test_merged.drop(columns=['image_id', 'image_path'], errors='ignore')

# ==========================================================
# 3. STANDARDIZE THE FEATURES SAFELY
# ==========================================================
print("Standardizing features...")
scaler = StandardScaler()

# Crucial: Fit the scaler on TRAIN data, then transform both TRAIN and TEST
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================================
# 4. CONFIGURE & TRAIN THE WINNING PIPELINE
# ==========================================================
print("\nTraining the optimized model pipeline...")

# Replace these numbers with the optimal values from your Grid Search
BEST_PCA_COMPONENTS = 700  
BEST_C_PENALTY = 1.0       

optimized_pipeline = Pipeline([
    ('pca', PCA(n_components=BEST_PCA_COMPONENTS, random_state=42)),
    ('lr', LogisticRegression(C=BEST_C_PENALTY, max_iter=1000, random_state=42))
])

# Train the model on your full, scaled training set
optimized_pipeline.fit(X_train_scaled, y_train)

# ==========================================================
# 5. GENERATE TEST PREDICTIONS & EXPORT FOR KAGGLE
# ==========================================================
print("\nGenerating predictions on the test dataset...")
# Run the test data through the pipeline to get your submission labels
test_predictions = optimized_pipeline.predict(X_test_scaled)

# Create the clean submission DataFrame using the test set's image_id
submission_df = pd.DataFrame({
    'image_id': test_merged['image_id'],
    'predicted_class_id': test_predictions
})

# Save to CSV
output_filename = 'task1_kaggle_submission.csv'
submission_df.to_csv(output_filename, index=False)

print(f"\nSuccess! Kaggle submission file created: '{output_filename}'")
print(submission_df.head(10))