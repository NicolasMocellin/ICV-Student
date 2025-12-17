import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
import random  # Pour le tirage aléatoire
import cv2
import json
from utils import show

def compute_mean_std(directory):
    mean = []
    std = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(root, file)
                img = cv2.imread(src_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_norm = img.astype(np.float32) / 255.0
                    mean.append(np.mean(img_norm, axis=(0,1)))
                    std.append(np.std(img_norm, axis=(0,1)))
    return np.mean(mean, axis=0), np.mean(std, axis=0)

def create_model(train_path : str, mean : np.ndarray = None, std : np.ndarray = None) -> tuple[tf.keras.Model, dict]:
    # ========================
    # Load and prepare training data
    # ========================
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        #preprocessing_function=lambda x: (x - mean) / std if mean is not None and std is not None else None
        )
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(64, 64),      # Resize all images to 64×64 pixels
        batch_size=32,              # Process 32 images at a time
        class_mode='categorical'    # Labels in one-hot encoding format (e.g., [0,0,1,0...])
    )

    # Retrieve class indices from training generator
    class_indices = train_generator.class_indices

    # ========================
    # Build the CNN model
    # ========================
    n_classes = len(class_indices)  # Number of classes to predict

    # CNN Architecture: Hierarchical feature extraction
    #
    # Feature hierarchy (progressively more abstract representations):
    #   Level 1 (Conv2D 32):  Low-level features - edges, colors, simple textures
    #   Level 2 (Conv2D 64):  Mid-level features - geometric shapes, patterns
    #   Level 3 (Conv2D 128): High-level features - complex objects (digits, symbols)
    #
    model = Sequential([
        # === BLOCK 1: Low-level features ===
        Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
        # Conv2D: Applies 32 convolutional filters (3×3 kernel) to detect patterns
        # activation='relu': Introduces non-linearity (ReLU = max(0,x))
        #                    Allows the network to learn complex, non-linear relationships

        MaxPooling2D(pool_size=(2,2)),
        # MaxPooling: Reduces spatial dimensions by factor of 2 (64×64 → 32×32)
        # Benefits: - Reduces number of parameters (computational efficiency)
        #           - Provides translation invariance (small shifts don't affect detection)
        #           - Focuses on most prominent features in each region

        # === BLOCK 2: Mid-level features ===
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),  # 32×32 → 16×16

        # === BLOCK 3: High-level features ===
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),  # 16×16 → 8×8

        # === FULLY-CONNECTED LAYERS ===
        Flatten(),
        # Flatten: Transforms 3D volume (8×8×128=8192) into 1D feature vector
        #          Required transition from convolutional to dense layers

        Dense(128, activation='relu'),
        # Dense: Fully-connected layer to combine extracted features
        #        Learns high-level patterns from all spatial features

        Dropout(0.5),
        # Dropout: Randomly deactivates 50% of neurons during training
        # Purpose: Regularization technique to prevent overfitting
        #          Forces network to learn robust, redundant feature representations
        #          Note: Only active during training, disabled during inference

        Dense(n_classes, activation='softmax')
        # Output layer: n_classes neurons (one per traffic sign class)
        # softmax: Converts raw scores into probability distribution [0,1] that sums to 1
        #          Example: [0.05, 0.85, 0.10] → class 1 has 85% confidence
    ])

    # Model compilation
    model.compile(
        optimizer='adam',                    # Adam: Adaptive learning rate optimizer (variant of gradient descent)
        loss='categorical_crossentropy',     # Loss function for multi-class classification
        metrics=['accuracy']                 # Metrics to track during training
    )

    # ========================
    # Train the model
    # ========================
    history = model.fit(
        train_generator,
        epochs=20,                           # Number of complete passes through the training dataset
        steps_per_epoch=len(train_generator),  # Number of batches per epoch
        verbose=1                             # Display training progress
    )

    return model, class_indices


def create_test_generator(test_path : str, csv_path : str, class_indices : dict, mean : np.ndarray = None, std : np.ndarray = None) -> tf.keras.Model:
    # ========================
    # Load and prepare test data
    # ========================
    df_test = pd.read_csv(csv_path)

    # Fix test image paths (adjust relative paths)
    df_test["Path"] = df_test["Path"].apply(lambda x: os.path.join(test_path, x.replace("Test/", "")))
    df_test = df_test[df_test["Path"].apply(lambda x: os.path.exists(x))]

    # Convert labels to string format (required for flow_from_dataframe)
    df_test["ClassId"] = df_test["ClassId"].astype(str)

    # Filter to keep only classes present in training set
    # This handles cases where test set might contain classes not seen during training
    df_test = df_test[df_test["ClassId"].isin(class_indices.keys())]

    # Create test image generator with same preprocessing as training
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=lambda x: (x - mean) / std if mean is not None and std is not None else None)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=df_test,
        x_col="Path",
        y_col="ClassId",
        target_size=(64, 64),
        batch_size=32,
        class_mode="categorical",
        shuffle=False                         # Don't shuffle test data to maintain order
    )

    return test_generator

def create_model_and_evaluate(train_path: str, test_path: str):
    models_path = "./models"
    os.makedirs(models_path, exist_ok=True)

    mean, std = compute_mean_std(train_path)

    model_name = os.path.basename(train_path)        # estrae "Train_small"
    model_file = os.path.join(models_path, f"cnn_{model_name}.h5")
    class_file = os.path.join(models_path, f"class_indices_{model_name}.json")

    if not os.path.exists(model_file):
        print("Creating model...")
        model, class_indices = create_model(train_path, mean, std)

        model.summary()

        model.save(model_file)

        with open(class_file, "w") as f:
            json.dump(class_indices, f)

    else:
        print("Model already exists.")

        model = load_model(model_file)

        with open(class_file, "r") as f:
            class_indices = json.load(f)

        model.summary()

    csv_path = r"./images/Test.csv"  # CSV file with test set labels
    test_generator = create_test_generator(test_path, csv_path, class_indices, mean, std)
    test_loss, test_acc = model.evaluate(test_generator)

    print(f" Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f" Loss: {test_loss:.4f}")

    # Make predictions on the test set
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)  # Predicted class indices
    true_classes = test_generator.classes  # True class indices
    class_labels = list(test_generator.class_indices.keys())  # Class label names

    # Find indices of correctly and incorrectly classified images
    correct_idx = np.where(predicted_classes == true_classes)[0]
    incorrect_idx = np.where(predicted_classes != true_classes)[0]

    print(f" Number of correctly classified images: {len(correct_idx)}/{len(true_classes)} ({100*len(correct_idx)/len(true_classes):.1f}%)")
    print(f" Number of incorrectly classified images: {len(incorrect_idx)}/{len(true_classes)} ({100*len(incorrect_idx)/len(true_classes):.1f}%)")

    # Display one correctly and one incorrectly classified image (random selection)
    if len(correct_idx) > 0 and len(incorrect_idx) > 0:
        idx_correct = random.choice(correct_idx)  # Random selection
        im_correct = cv2.imread(test_generator.filepaths[idx_correct])
        idx_incorrect = random.choice(incorrect_idx)  # Random selection
        im_incorrect = cv2.imread(test_generator.filepaths[idx_incorrect])
        show([im_correct, im_incorrect],
            [f" Correct: {class_labels[true_classes[idx_correct]]} (Predicted: {class_labels[predicted_classes[idx_correct]]})",
            f" Incorrect: {class_labels[true_classes[idx_incorrect]]} (Predicted: {class_labels[predicted_classes[idx_incorrect]]})"])

if __name__ == "__main__":
    create_model_and_evaluate(r"./images/Train_small", r"./images/Test")