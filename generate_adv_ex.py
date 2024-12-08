import os
import random
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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

# --- Step 3: Creating a Dataset Class ---
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

# --- Step 6: FGSM Attack Definition ---
def fgsm_attack(model, images, labels, epsilon=0.1):
    # Set the model to evaluation mode
    model.eval()

    # Make the image tensor require gradients
    images.requires_grad = True

    # Forward pass the images through the model
    outputs = model(images)

    # Calculate the loss (using CrossEntropyLoss)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)

    # Backward pass to compute gradients
    model.zero_grad()
    loss.backward()

    # Get the sign of the gradients to create the perturbation
    grad_sign = images.grad.sign()

    # Apply the perturbation to the image
    perturbed_images = images + epsilon * grad_sign

    # Clip the perturbed images to ensure they are still valid image pixels
    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    return perturbed_images

# --- Step 7: Create Adversarial Examples and Save ---
# --- Step 7: Create Adversarial Examples and Save ---
def test_adversarial(model, test_loader, epsilon):
    model.eval()
    correct = 0
    total = 0
    all_true_labels = []
    all_pred_labels = []

    # Create folders to save clean and adversarial images
    clean_folder = "all_images_clean"
    adv_folder = "all_images_adv"
    os.makedirs(clean_folder, exist_ok=True)
    os.makedirs(adv_folder, exist_ok=True)

    for i, (img1, img2, labels) in enumerate(test_loader):  # Process all samples in DataLoader
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

        # Generate adversarial examples for both images
        adv_img1 = fgsm_attack(model, img1, labels, epsilon)
        adv_img2 = fgsm_attack(model, img2, labels, epsilon)

        # Save clean and adversarial images
        for j in range(labels.size(0)):  # Save all images in the batch
            index = i * test_loader.batch_size + j + 1  # Unique index for each image
            clean_image_path_1 = os.path.join(clean_folder, f"clean_img1_{index}.png")
            clean_image_path_2 = os.path.join(clean_folder, f"clean_img2_{index}.png")
            adv_image_path_1 = os.path.join(adv_folder, f"adv_img1_{index}.png")
            adv_image_path_2 = os.path.join(adv_folder, f"adv_img2_{index}.png")

            save_image(img1[j], clean_image_path_1)
            save_image(img2[j], clean_image_path_2)
            save_image(adv_img1[j], adv_image_path_1)
            save_image(adv_img2[j], adv_image_path_2)

        total += labels.size(0)

        # Re-classify the adversarial examples
        outputs1 = model(adv_img1)
        outputs2 = model(adv_img2)

        _, predicted1 = outputs1.max(1)
        _, predicted2 = outputs2.max(1)
        predicted = (predicted1 + predicted2) // 2

        correct += (predicted == labels).sum().item()

        all_true_labels.extend(labels.cpu().numpy())
        all_pred_labels.extend(predicted.cpu().numpy())

    adv_accuracy = 100 * correct / total
    f1 = f1_score(all_true_labels, all_pred_labels, average="macro")  # Updated for multiclass
    cm = confusion_matrix(all_true_labels, all_pred_labels)
    print(f"Epsilon: {epsilon}\tAdversarial Accuracy: {adv_accuracy:.2f}%\tF1-Score: {f1:.2f}")
    return adv_accuracy, f1, cm


# Test with all pairs
epsilon_values = [0.01]
for epsilon in epsilon_values:
    print(f"Performing FGSM attack with epsilon = {epsilon}")
    adv_accuracy, f1, cm = test_adversarial(model, data_loader, epsilon)
    print(f"Confusion Matrix:\n{cm}")
    print(f"F1-Score for epsilon {epsilon}: {f1:.2f}")


