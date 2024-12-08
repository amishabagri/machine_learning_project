import os
import random
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from torchvision.models import resnet50, ResNet50_Weights


# --- Configuration ---
lfw_dataset_path = r"C:\Users\NAU_SICCS_CyberLab\PycharmProjects\ml_final _project\lfw-deepfunneled\lfw-deepfunneled"  # Path to LFW dataset
pairs_file = r"C:\Users\NAU_SICCS_CyberLab\PycharmProjects\ml_final _project\pairs.csv"  # Path to your pairs dataset
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)

# --- Step 1: Parse Positive Pairs from Pairs File ---
def parse_positive_pairs(pairs_file, lfw_dataset_path):
    pairs = []
    with open(pairs_file, 'r') as f:
        for line in f.readlines()[1:]:  # Skip header
            # Remove any trailing commas and split the line
            fields = [field.strip() for field in line.split(',') if field.strip()]
            if len(fields) != 3:  # Skip lines that don't have exactly 3 fields
                continue
            name, img1, img2 = fields
            img1_path = os.path.join(lfw_dataset_path, name, f"{name}_{int(img1):04d}.jpg")
            img2_path = os.path.join(lfw_dataset_path, name, f"{name}_{int(img2):04d}.jpg")
            pairs.append((img1_path, img2_path))
    return pairs



positive_pairs = parse_positive_pairs(pairs_file, lfw_dataset_path)

# --- Step 2: Generate Negative Pairs ---
def generate_negative_pairs(positive_pairs, lfw_dataset_path):
    people = os.listdir(lfw_dataset_path)
    negative_pairs = []
    for _ in range(len(positive_pairs)):
        person1, person2 = random.sample(people, 2)
        img1 = random.choice(os.listdir(os.path.join(lfw_dataset_path, person1)))
        img2 = random.choice(os.listdir(os.path.join(lfw_dataset_path, person2)))
        img1_path = os.path.join(lfw_dataset_path, person1, img1)
        img2_path = os.path.join(lfw_dataset_path, person2, img2)
        negative_pairs.append((img1_path, img2_path))
    return negative_pairs

negative_pairs = generate_negative_pairs(positive_pairs, lfw_dataset_path)

# --- Step 3: Create a Dataset Class ---
class LFWPairsDataset(Dataset):
    def __init__(self, pairs, labels, transform=None):
        self.pairs = pairs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.pairs[idx]
        label = self.labels[idx]

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

# Combine positive and negative pairs
all_pairs = positive_pairs + negative_pairs
all_labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create Dataset and DataLoader
dataset = LFWPairsDataset(all_pairs, all_labels, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- Step 4: Load Pre-trained Model ---
# Load pre-trained ResNet50 model with recommended weights argument
# Load the ResNet50 model with default weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

# Remove the final classification layer
model.fc = nn.Identity()

# Move the model to the appropriate device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# --- Step 5: Feature Extraction ---
def extract_features(model, images):
    with torch.no_grad():
        features = model(images)
    return features

# --- Step 6: Train/Evaluate on the Dataset ---
model.eval()
similarities = []
true_labels = []

for img1, img2, label in data_loader:
    img1, img2, label = img1.to(device), img2.to(device), label.to(device)

    # Extract features
    features1 = extract_features(model, img1)
    features2 = extract_features(model, img2)

    # Compute cosine similarity
    similarity = nn.functional.cosine_similarity(features1, features2)
    similarities.extend(similarity.cpu().numpy())
    true_labels.extend(label.cpu().numpy())

# --- Step 7: Evaluate Metrics ---
threshold = 0.30  # Cosine similarity threshold
predicted_labels = [1 if sim > threshold else 0 for sim in similarities]

accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
cm = confusion_matrix(true_labels, predicted_labels)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print("Confusion Matrix:")
print(cm)
