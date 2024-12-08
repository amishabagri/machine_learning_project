
import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Function to load images from a folder
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('RGB')  # Ensure all images are in RGB
            img_tensor = transforms.ToTensor()(img)  # Convert image to Tensor
            images.append(img_tensor)
    return images

def visualize_fft(image, index, label):
    # Convert to grayscale and numpy
    img_np = image.mean(dim=0).cpu().numpy()  # Average over RGB channels
    fft_result = np.fft.fft2(img_np)
    fft_shift = np.fft.fftshift(fft_result)
    magnitude_spectrum = np.log(np.abs(fft_shift) + 1)

    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(f"FFT Spectrum ({label} Image {index + 1})")
    plt.savefig(f"analysis_results/fft_spectrum_{label}_image_{index + 1}.png")
    plt.show()

def compute_change_statistics(clean_images, adversarial_images):
    # Compute absolute pixel differences
    diff = torch.abs(clean_images - adversarial_images)

    # Mean, Std, and Max change
    mean_change = diff.mean().item()
    std_change = diff.std().item()
    max_change = diff.max().item()

    print(f"Mean Pixel Change: {mean_change}")
    print(f"Standard Deviation of Change: {std_change}")
    print(f"Max Pixel Change: {max_change}")

    return mean_change, std_change, max_change


# Paths for clean and adversarial images
clean_images_path = r"C:\Users\NAU_SICCS_CyberLab\PycharmProjects\ml_final _project\all_images_clean"
adversarial_images_path = r"C:\Users\NAU_SICCS_CyberLab\PycharmProjects\ml_final _project\all_images_adv"

# Load clean and adversarial images
clean_images = load_images_from_folder(clean_images_path)
adversarial_images = load_images_from_folder(adversarial_images_path)

# Visualize FFT for clean and adversarial images
visualize_fft(clean_images[1], 1, "Clean")
visualize_fft(adversarial_images[1], 1, "Adversarial")