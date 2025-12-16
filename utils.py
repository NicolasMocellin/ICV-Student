import os
import shutil
import random
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import math
import numpy as np

def create_subset_dataset(source_dir, dest_dir, percentage, random_seed=42):
    """
    Creates a dataset subset by keeping a percentage of images from each class.

    Args:
        source_dir (str): Source folder path containing class subfolders
        dest_dir (str): Destination folder path for the subset
        percentage (float): Percentage of images to keep (between 0 and 100)
        random_seed (int): Seed for reproducibility (default: 42)

    Returns:
        dict: Creation statistics (number of images per class)

    Example:
        >>> create_subset_dataset('./images/Train', './images/Train_light', 10)
        {'0': 50, '1': 45, ...}
    """
    if not os.path.exists(source_dir):
        raise ValueError(f"Source folder '{source_dir}' does not exist.")

    if percentage <= 0 or percentage > 100:
        raise ValueError(f"Percentage must be between 0 and 100 (received: {percentage})")

    # Initialize random generator for reproducibility
    random.seed(random_seed)

    # Create destination folder if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    stats = {}
    total_copied = 0

    # List all subfolders (classes)
    class_folders = [f for f in os.listdir(source_dir)
                     if os.path.isdir(os.path.join(source_dir, f))]


    # Process each class
    for class_name in tqdm(sorted(class_folders), desc="Classes"):
        source_class_dir = os.path.join(source_dir, class_name)
        dest_class_dir = os.path.join(dest_dir, class_name)

        # Create class folder in destination
        os.makedirs(dest_class_dir, exist_ok=True)

        # List all images in the class
        image_files = [f for f in os.listdir(source_class_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm', '.bmp'))]

        # Calculate number of images to keep
        num_to_keep = max(1, int(len(image_files) * percentage / 100))

        # Randomly select images
        selected_images = random.sample(image_files, num_to_keep)

        # Copy selected images
        for img_name in selected_images:
            source_path = os.path.join(source_class_dir, img_name)
            dest_path = os.path.join(dest_class_dir, img_name)
            shutil.copy2(source_path, dest_path)

        stats[class_name] = num_to_keep
        total_copied += num_to_keep

    return stats


def print_dataset_summary(dataset_dir):
    """
    Displays a dataset summary (number of images per class).

    Args:
        dataset_dir (str): Dataset folder path

    Returns:
        dict: Number of images per class
    """
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Folder '{dataset_dir}' does not exist.")

    class_folders = [f for f in os.listdir(dataset_dir)
                     if os.path.isdir(os.path.join(dataset_dir, f))]

    summary = {}
    total = 0

    print(f"\nDataset summary: {dataset_dir}")
    print(f"{'Class':<10} {'Number of images':<20}")
    print("-" * 30)

    for class_name in sorted(class_folders):
        class_dir = os.path.join(dataset_dir, class_name)
        image_files = [f for f in os.listdir(class_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm', '.bmp'))]
        count = len(image_files)
        summary[class_name] = count
        total += count
        print(f"{class_name:<10} {count:<20}")

    print("-" * 30)
    print(f"{'TOTAL':<10} {total:<20}")

    return summary

def show(images = [None], titles = [None], nb_lines = None, nb_cols = None):
    """Display a list of images using matplotlib with automatic BGR to RGB conversion.

    Args:
        images: List of images (in BGR format, as loaded by OpenCV)
        titles: List of titles corresponding to each image
        nb_lines: Number of rows in the subplot grid (auto-calculated if None)
        nb_cols: Number of columns in the subplot grid (auto-calculated if None)

    Raises:
        ValueError: If number of images doesn't match number of titles

    Note:
        Automatically converts BGR (OpenCV format) to RGB (matplotlib format)
    """
    if len(images) != len(titles):
        raise ValueError("Number of images and titles must match")

    nb_images = len(images)
    if nb_cols is None:
        nb_cols = math.ceil(math.sqrt(nb_images))  # Square-ish layout by default
    if nb_lines is None:
        nb_lines = math.ceil(nb_images / nb_cols)

    fig, axs = plt.subplots(nb_lines, nb_cols)
    for i, (img, title) in enumerate(zip(images, titles)):
        if img is None:
            continue
        # Handle the case where axs is 1D if there's only one row or one column of subplots
        if nb_lines == 1 and nb_cols == 1:
            ax = axs
        elif nb_lines == 1:
            ax = axs[i % nb_cols]
        elif nb_cols == 1:
            ax = axs[i // nb_cols]
        else:
            ax = axs[i // nb_cols, i % nb_cols]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
        ax.set_title(title)
        ax.axis("off")
    plt.show()

def add_noise(img, noise_type="gaussian", noise_mean=0, noise_std=25, noise_amount=0.002):
    """Ajoute du bruit synthétique à une image pour tester les algorithmes de débruitage.
    
    Args:
        img: Image d'entrée (format BGR)
        noise_type: Type de bruit - "gaussian", "poisson", ou "salt_and_pepper"
        noise_mean: Moyenne de la distribution du bruit
        noise_std: Écart-type du bruit (gaussien uniquement)
        noise_amount: Proportion de pixels affectés par le bruit sel et poivre
    
    Returns:
        Image bruitée
    """
    if noise_type == "gaussian":
        noise = np.random.normal(noise_mean, noise_std, img.shape).astype(np.float32)
        noisy = cv2.add(img.astype(np.float32), noise)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    elif noise_type == "salt_and_pepper":
        noisy = img.copy()
        num_pixels = img.size
        
        # Bruit "sel" (pixels blancs)
        num_salt = int(noise_amount * num_pixels / 2)
        coords = [np.random.randint(0, i, num_salt) for i in img.shape]
        noisy[coords[0], coords[1], :] = 2
        
        # Bruit "poivre" (pixels noirs)
        num_pepper = int(noise_amount * num_pixels / 2)
        coords = [np.random.randint(0, i, num_pepper) for i in img.shape]
        noisy[coords[0], coords[1], :] = 0
    else:
        raise ValueError(f"Type de bruit non reconnu: {noise_type}")
    
    return noisy


if __name__ == "__main__":
    # Example usage: Create a 10% subset for faster experimentation
    source = r".\images\Train"
    destination = r".\images\Train_small"

    # Create a subset with 10% of images (stratified sampling per class)
    stats = create_subset_dataset(source, destination, percentage=10)

    # Display summary statistics
    print_dataset_summary(destination)

