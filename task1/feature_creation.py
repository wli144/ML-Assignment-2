import torch
import pandas as pd
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import os

# 1. Load the metadata and setup paths
metadata = pd.read_csv('task1_data/train_metadata.csv')
img_dir = 'task1_data/'
# 2. Load a pre-trained model and its matching image processor from online
# This automatically downloads a powerful ResNet-50 model trained on ImageNet
model_name = "microsoft/resnet-50"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

features = []

print(f"Extracting features using Hugging Face on {device}...")

# 3. Extract features image-by-image (very simple loop)
for idx, row in metadata.iterrows():
    # Open the image
    img_path = os.path.join(img_dir, row['image_path'])
    image = Image.open(img_path).convert("RGB")
    
    # Let Hugging Face handle all resizing and normalization automatically
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # Pass through the model to get the feature vector
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Pooler_output gives a clean, flattened feature vector representing the image
    image_embedding = outputs.pooler_output.squeeze().cpu().numpy()
    features.append(image_embedding)

# 4. Save to CSV directly
df_features = pd.DataFrame(features)
df_features.insert(0, 'image_path', metadata['image_path'])
df_features.to_csv('resnet_features_hf.csv', index=False)

print("Done! Features saved to resnet_features_hf.csv")