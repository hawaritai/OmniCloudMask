import rasterio
import os
import cv2
import numpy as np
from skimage.filters import laplace, sobel, roberts
from keras.models import Sequential
from keras.layers import Dense, Dropout
from pathlib import Path

def load_image(path):
    path = Path(path)
    ext = path.suffix.lower()

    if ext in [".tif", ".tiff", ".pix"]:  # Raster formats
        with rasterio.open(path) as src:
            image = src.read([1, 2, 3]) if src.count >= 3 else src.read()
            image = np.transpose(image, (1, 2, 0))  # Channels last
            image = image.astype(np.uint8)  # Optional: normalize/cast
    elif ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return image


def build_model(input_shape=(9,), weights_path=None):
    """
    Builds and compiles the neural network model for blur detection.
    
    Args:
        input_shape (tuple): Shape of input features (default: 9 features).
        weights_path (str): Path to pre-trained model weights.
    
    Returns:
        keras.Sequential: Compiled neural network model.
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(current_dir, weights_path)
    
    if weights_path and os.path.exists(weights_path):
        model.load_weights(weights_path)
    else:
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")
    
    return model


def extract_features(image_gray):
    """
    Extracts edge-based features from a grayscale image using Laplace, Sobel, and Roberts filters.
    
    Args:
        image_gray (np.ndarray): Grayscale image array.
    
    Returns:
        list: Features [mean, variance, max] for each filter (9 features total).
    """
    lap_feat = laplace(image_gray)
    sob_feat = sobel(image_gray)
    rob_feat = roberts(image_gray)
    
    return [
        lap_feat.mean(), lap_feat.var(), np.max(lap_feat),
        sob_feat.mean(), sob_feat.var(), np.max(sob_feat),
        rob_feat.mean(), rob_feat.var(), np.max(rob_feat)
    ]