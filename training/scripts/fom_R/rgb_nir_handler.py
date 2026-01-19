"""
RGB + Synthetic NIR Band Handler for OmniCloudMask

This module provides utilities for preparing RGB images with synthetic NIR bands
for cloud/shadow detection using OmniCloudMask.

OmniCloudMask expects input in the format: (bands, height, width)
where bands are arranged as [RED, GREEN, NIR] in that order.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from pathlib import Path


class RGBNIRHandler:
    """
    Handles preparation of RGB + synthetic NIR data for OmniCloudMask detection.
    
    OmniCloudMask Requirements:
    - Input shape: (3, height, width) - exactly 3 bands
    - Band order: [RED, GREEN, NIR]
    - Data type: float32 or float64
    - Value range: typically 0-1 (normalized) or 0-10000 (DN values)
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize handler.
        
        Args:
            device: torch device (cuda/cpu). If None, uses cuda if available.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def normalize_band(
        self, 
        band: np.ndarray,
        method: str = "minmax",
        percentile: Tuple[float, float] = (2, 98)
    ) -> np.ndarray:
        """
        Normalize a single band to [0, 1] range.
        
        Args:
            band: Input band array (H, W)
            method: Normalization method
                - "minmax": (x - min) / (max - min)
                - "percentile": Robust to outliers using percentiles
                - "stddev": (x - mean) / std
            percentile: Percentile range for robust normalization (min_p, max_p)
        
        Returns:
            Normalized band in [0, 1] range
        """
        band = band.astype(np.float32)
        
        if method == "minmax":
            vmin, vmax = band.min(), band.max()
        elif method == "percentile":
            vmin, vmax = np.percentile(band, percentile)
        elif method == "stddev":
            mean, std = band.mean(), band.std()
            return np.clip((band - mean) / (std + 1e-8), -3, 3) / 6.0 + 0.5
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Avoid division by zero
        if vmax - vmin < 1e-8:
            return np.ones_like(band) * 0.5
        
        normalized = (band - vmin) / (vmax - vmin)
        return np.clip(normalized, 0, 1)
    
    def scale_to_dn_range(
        self,
        band: np.ndarray,
        target_range: Tuple[int, int] = (0, 10000)
    ) -> np.ndarray:
        """
        Scale band values to DN (Digital Number) range.
        
        Args:
            band: Input band (normalized to [0,1] or any range)
            target_range: Target DN range (min, max)
        
        Returns:
            Band scaled to target DN range
        """
        band = band.astype(np.float32)
        vmin, vmax = band.min(), band.max()
        
        if vmax - vmin < 1e-8:
            band = np.ones_like(band) * 0.5
        else:
            band = (band - vmin) / (vmax - vmin)
        
        dn_min, dn_max = target_range
        return band * (dn_max - dn_min) + dn_min
    
    def stack_rgb_nir(
        self,
        red: np.ndarray,
        green: np.ndarray,
        nir: np.ndarray,
        normalize: bool = True,
        scale_to_dn: bool = False,
        dn_range: Tuple[int, int] = (0, 10000)
    ) -> np.ndarray:
        """
        Stack RGB bands with NIR into (3, H, W) format required by OmniCloudMask.
        
        IMPORTANT: Band order MUST be [RED, GREEN, NIR]
        OmniCloudMask expects this specific order!
        
        Args:
            red: Red band (H, W)
            green: Green band (H, W)
            nir: NIR band (H, W) - can be synthetic from NIRGAN
            normalize: Whether to normalize each band to [0, 1]
            scale_to_dn: Whether to scale to DN range after normalization
            dn_range: Target DN range if scale_to_dn=True
        
        Returns:
            Stacked array of shape (3, H, W) with bands [RED, GREEN, NIR]
        
        Raises:
            ValueError: If input shapes don't match or dimensions are wrong
        """
        # Ensure 2D input
        if red.ndim == 3 and red.shape[2] == 1:
            red = red[:, :, 0]
        if green.ndim == 3 and green.shape[2] == 1:
            green = green[:, :, 0]
        if nir.ndim == 3 and nir.shape[2] == 1:
            nir = nir[:, :, 0]
        
        # Validate shapes
        if red.ndim != 2 or green.ndim != 2 or nir.ndim != 2:
            raise ValueError(f"All bands must be 2D (H, W). Got: red={red.shape}, green={green.shape}, nir={nir.shape}")
        
        if not (red.shape == green.shape == nir.shape):
            raise ValueError(f"All bands must have same shape. Got: red={red.shape}, green={green.shape}, nir={nir.shape}")
        
        # Convert to float32
        red = red.astype(np.float32)
        green = green.astype(np.float32)
        nir = nir.astype(np.float32)
        
        # Normalize if requested
        if normalize:
            red = self.normalize_band(red, method="percentile")
            green = self.normalize_band(green, method="percentile")
            nir = self.normalize_band(nir, method="percentile")
        
        # Scale to DN range if requested
        if scale_to_dn:
            red = self.scale_to_dn_range(red, dn_range)
            green = self.scale_to_dn_range(green, dn_range)
            nir = self.scale_to_dn_range(nir, dn_range)
        
        # Stack: [RED, GREEN, NIR] - ORDER MATTERS FOR OMNICLOUDMASK
        rgn_stack = np.stack([red, green, nir], axis=0)  # Shape: (3, H, W)
        
        return rgn_stack.astype(np.float32)
    
    def compute_ndvi(self, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """
        Compute NDVI for quality check.
        
        NDVI = (NIR - RED) / (NIR + RED + epsilon)
        
        Args:
            red: Red band (H, W)
            nir: NIR band (H, W)
        
        Returns:
            NDVI map [-1, 1]
        """
        red = red.astype(np.float32)
        nir = nir.astype(np.float32)
        
        ndvi = (nir - red) / (nir + red + 1e-8)
        return np.clip(ndvi, -1, 1)
    
    def validate_nir_quality(self, nir: np.ndarray, threshold: float = 0.3) -> dict:
        """
        Check quality of synthetic NIR band.
        
        Args:
            nir: NIR band (H, W)
            threshold: Minimum variance threshold for validity
        
        Returns:
            Dictionary with quality metrics
        """
        nir = nir.astype(np.float32)
        
        return {
            "mean": float(nir.mean()),
            "std": float(nir.std()),
            "min": float(nir.min()),
            "max": float(nir.max()),
            "variance": float(nir.var()),
            "is_valid": nir.var() > threshold,
            "info": "Synthetic NIR has low variance - may produce poor results" if nir.var() < threshold else "Good NIR quality"
        }


def prepare_rgb_with_synthetic_nir(
    rgb_array: np.ndarray,
    nir_array: np.ndarray,
    normalize: bool = True,
    scale_to_dn: bool = True,
    dn_range: Tuple[int, int] = (0, 10000),
    device: Optional[torch.device] = None,
    validate: bool = True
) -> np.ndarray:
    """
    Convenience function to prepare RGB + synthetic NIR for OmniCloudMask.
    
    This is the recommended approach for handling RGB images with synthetic
    NIR generated from NIRGAN.
    
    Args:
        rgb_array: RGB image as (H, W, 3) or (3, H, W)
        nir_array: Synthetic NIR band as (H, W) or (1, H, W)
        normalize: Whether to normalize bands to [0, 1]
        scale_to_dn: Whether to scale to DN range
        dn_range: Target DN range
        device: Torch device
        validate: Whether to validate the result
    
    Returns:
        Stacked RGB+NIR array of shape (3, H, W) ready for OmniCloudMask
    
    Example:
        >>> rgb = cv2.imread("image.jpg")  # shape (H, W, 3)
        >>> nir = get_NIR(rgb, device=device)  # shape (H, W)
        >>> rgn_input = prepare_rgb_with_synthetic_nir(rgb, nir)
        >>> mask = predict_from_array(rgn_input)
    """
    handler = RGBNIRHandler(device=device)
    
    # Convert RGB to HWC if needed
    if rgb_array.ndim == 3:
        if rgb_array.shape[0] == 3:
            rgb_array = np.transpose(rgb_array, (1, 2, 0))
    
    if rgb_array.shape[2] != 3:
        raise ValueError(f"RGB array must have 3 channels, got {rgb_array.shape[2]}")
    
    # Extract individual channels
    red = rgb_array[:, :, 0]
    green = rgb_array[:, :, 1]
    blue = rgb_array[:, :, 2]
    
    # Ensure NIR is 2D
    if nir_array.ndim == 3:
        nir_array = nir_array.squeeze()
    
    # Validate shapes match
    if red.shape != nir_array.shape:
        print(f"Warning: RGB shape {red.shape} != NIR shape {nir_array.shape}")
        print("Attempting to resize NIR to match RGB...")
        nir_array = cv2.resize(nir_array, (red.shape[1], red.shape[0]))
    
    # Validate NIR quality
    if validate:
        nir_quality = handler.validate_nir_quality(nir_array)
        print(f"NIR Quality: {nir_quality['info']}")
        print(f"  Mean: {nir_quality['mean']:.4f}, Std: {nir_quality['std']:.4f}")
    
    # Stack bands - IMPORTANT: RED, GREEN, NIR order
    rgn_stack = handler.stack_rgb_nir(
        red=red,
        green=green,
        nir=nir_array,
        normalize=normalize,
        scale_to_dn=scale_to_dn,
        dn_range=dn_range
    )
    
    return rgn_stack


# Optional: Import cv2 if available for convenience
try:
    import cv2
except ImportError:
    cv2 = None
