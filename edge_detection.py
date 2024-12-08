import os
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


# Function to load a single image from a file path
def load_image(file_path):
    """
    Load a single image from the given file path and convert it to a PyTorch tensor.
    """
    img = Image.open(file_path).convert('RGB')  # Ensure the image is in RGB
    img_tensor = transforms.ToTensor()(img)  # Convert image to Tensor
    return img_tensor


# Edge detection function (unchanged)
def detect_edges(image, index, label):
    """
    Detect edges using the Canny edge detection algorithm.
    """
    # Convert tensor to NumPy and scale to [0, 255]
    img_np = image.permute(1, 2, 0).cpu().numpy() * 255  # (H x W x C)
    img_np = img_np.astype(np.uint8)

    # If the image is RGB, convert it to grayscale
    if img_np.shape[-1] == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Log the image being used
    print(f"Processing edge detection for: {label} image from path {index}")

    # Detect edges using Canny edge detector
    edges = cv2.Canny(img_np, threshold1=100, threshold2=200)

    # Save and display the detected edges
    os.makedirs("analysis_results", exist_ok=True)  # Ensure output directory exists
    plt.figure(figsize=(6, 6))
    plt.imshow(edges, cmap='gray')
    plt.title(f"Edges Detected ({label} Image)")
    plt.axis('off')
    plt.savefig(f"analysis_results/edges_{label}_image_{os.path.basename(index)}.png", bbox_inches='tight')
    plt.show()

    return edges


# Direct paths for clean and adversarial images
clean_image_path = r"C:\Users\NAU_SICCS_CyberLab\PycharmProjects\ml_final _project\all_images_clean\clean_img1_2.png"
adversarial_image_path = r"C:\Users\NAU_SICCS_CyberLab\PycharmProjects\ml_final _project\all_images_adv\adv_img1_2.png"

# Load images directly from paths
clean_image = load_image(clean_image_path)
adversarial_image = load_image(adversarial_image_path)

# Detect edges for the specific clean and adversarial images
edge_clean = detect_edges(clean_image, clean_image_path, "Clean")
edge_adv = detect_edges(adversarial_image, adversarial_image_path, "Adversarial")

