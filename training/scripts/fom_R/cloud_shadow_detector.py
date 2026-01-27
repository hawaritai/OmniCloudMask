import os
import sys
from pathlib import Path
import rasterio as rio
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from omnicloudmask import predict_from_array
import omnicloudmask
import cv2
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OmniCloudShadowDetector:
    """
    A class to detect clouds and shadows in RGB images using OmniCloudMask.
    """

    def __init__(self, output_folder: str = "cloud_shadow_output"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        logger.info(f"Using device: {self.device}")
        logger.info(f"OmniCloudMask version: {omnicloudmask.__version__}")
        logger.info(f"Output will be saved to: {self.output_folder.resolve()}")



    def energy_preserving_downsample(
        self,
        rgb_chw,
        target_size=(480, 520),
        sigma_scale=0.5,
        device="cuda"
    ):
        """
        rgb_chw: (3, H, W) float tensor
        target_size: (H_new, W_new)
        sigma_scale: controls Gaussian blur strength
        """

        rgb = torch.tensor(rgb_chw, dtype=torch.float32, device=device)
        rgb = rgb.unsqueeze(0)  # (1,3,H,W)

        H, W = rgb.shape[-2:]
        Ht, Wt = target_size

        # --------------------------------------------------
        # 1. Compute Gaussian sigma based on scale factor
        # --------------------------------------------------
        scale_h = H / Ht
        scale_w = W / Wt
        sigma = sigma_scale * max(scale_h, scale_w)

        if sigma > 0:
            kernel_size = int(2 * math.ceil(3 * sigma) + 1)
            kernel_size = min(kernel_size, min(H, W))

            def gaussian_kernel(k, s, device):
                x = torch.arange(k, device=device) - k // 2
                g = torch.exp(-(x ** 2) / (2 * s ** 2))
                g /= g.sum()
                return g

            g1d = gaussian_kernel(kernel_size, sigma, device)
            g2d = g1d[:, None] * g1d[None, :]
            g2d = g2d.expand(3, 1, kernel_size, kernel_size)

            rgb = F.conv2d(
                rgb,
                g2d,
                padding=kernel_size // 2,
                groups=3
            )

        # --------------------------------------------------
        # 2. Area-based downsampling (best for energy)
        # --------------------------------------------------
        rgb_down = F.interpolate(
            rgb,
            size=target_size,
            mode="area"
        )

        # --------------------------------------------------
        # 3. Optional energy normalization (VERY IMPORTANT)
        # --------------------------------------------------
        energy_before = rgb.pow(2).mean(dim=(-1, -2), keepdim=True)
        energy_after = rgb_down.pow(2).mean(dim=(-1, -2), keepdim=True)

        rgb_down = rgb_down * torch.sqrt(
            (energy_before + 1e-8) / (energy_after + 1e-8)
        )

        return rgb_down.squeeze(0)  # (3, H_new, W_new)

    def enhance_for_shadow_detection(self, rgb_array: np.ndarray) -> np.ndarray:
        """
        Enhance RGB data for better shadow detection.
        Shadows appear darker in all bands and with a cooler tone.
        """
        rgb_float = rgb_array.astype(np.float32)
        rgb_norm = rgb_float / (rgb_float.max() + 1e-8)
        
        # Calculate luminance
        lum = 0.299 * rgb_norm[:, :, 0] + 0.587 * rgb_norm[:, :, 1] + 0.114 * rgb_norm[:, :, 2]
        shadow_prior = (lum < np.percentile(lum, 40)).astype(np.float32)
        
        # Enhance shadows: boost blue, reduce red in dark areas
        enhanced = rgb_float.copy()
        enhanced[:, :, 2] = enhanced[:, :, 2] * (1.0 + 0.15 * shadow_prior)  # Boost blue
        enhanced[:, :, 0] = enhanced[:, :, 0] * (1.0 - 0.05 * shadow_prior)  # Reduce red
        
        # Clip to valid range
        if rgb_float.max() > 1:
            enhanced = np.clip(enhanced, 0, rgb_float.max())
        else:
            enhanced = np.clip(enhanced, 0, 1)
        
        return enhanced.astype(rgb_array.dtype)

    def prepare_input_array(
        self, 
        rgb_array: np.ndarray,
        resize_to: tuple = (768, 1024),
        normalize: bool = True,
        scale_to_dn: bool = True,
        dn_range: tuple = (0, 10000),
        validate_nir: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepares the R-G-NIR stack for OmniCloudMask from an RGB image.
        
        Process:
        1. Validates RGB input and handles RGBA images
        2. Resizes image for efficient processing
        3. Generates synthetic NIR band using NIRGAN
        4. Normalizes and stacks RGB+NIR bands for OmniCloudMask
        
        OmniCloudMask Requirements:
        - Input shape: (3, H, W)
        - Band order: [RED, GREEN, NIR] (CRITICAL)
        - Data type: float32
        
        Args:
            rgb_array: Input RGB image as (H, W, 3)
            resize_to: Target size for processing as (H, W)
            normalize: Whether to normalize bands to [0, 1]
            scale_to_dn: Whether to scale bands to DN range
            dn_range: Digital Number range (min, max)
            validate_nir: Whether to validate synthetic NIR quality
        
        Returns:
            Tuple of:
            - rgn_stack: (3, H, W) RGB+NIR stack for OmniCloudMask
            - rgb_resized: (3, H, W) resized RGB for visualization
        
        Raises:
            ValueError: If input is invalid
        """
        # ===== STEP 1: Validate and prepare RGB =====
        if rgb_array.shape[2] == 4:  # RGBA
            logger.warning("⚠️  4-channel image detected, dropping alpha channel.")
            rgb_array = rgb_array[:, :, :3]
        
        if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
            raise ValueError(f"Input must be RGB image (H, W, 3), got {rgb_array.shape}")
        
        logger.debug(f"Original RGB shape: {rgb_array.shape}")
        
        # ===== STEP 1b: Enhance for shadow detection =====
        logger.debug("🔄 Enhancing image for shadow detection...")
        rgb_array = self.enhance_for_shadow_detection(rgb_array)
        
        # ===== STEP 2: Resize RGB for processing =====
        # Convert to CHW for torch operations
        rgb_chw = np.transpose(rgb_array, (2, 0, 1))
        rgb_tensor = torch.tensor(rgb_chw, dtype=torch.float32, device=self.device)
        
        # Use energy-preserving downsampling to maintain spectral information
        # Optimal size: 512+ for better patch processing (avoids OmniCloudMask downsampling)
        # if resize_to[0] < 512 or resize_to[1] < 512:
        #     optimal_size = (max(512, resize_to[0]), max(512, resize_to[1]))
        #     print(f"Resizing to {resize_to} → {optimal_size} (optimal for patch_size=512)...")
        #     resize_to = optimal_size
        # else:
        #     print(f"Resizing to {resize_to}...")
        
        # rgb_tensor_resized = self.energy_preserving_downsample(
        #     rgb_tensor,
        #     target_size=resize_to,
        #     sigma_scale=0.5,
        #     device=self.device
        # )

        # Convert back to HWC for NIRGAN
        rgb_tensor_resized = F.interpolate(
            rgb_tensor.unsqueeze(0),
            size=resize_to,
            mode="area"
        ).squeeze(0)
        
        
        # Convert back to HWC for NIRGAN
        rgb_for_nir_gan = rgb_tensor_resized.permute(1, 2, 0).cpu().numpy()
        
        # ===== STEP 3: Generate synthetic NIR =====
        logger.debug("🔄 Generating synthetic NIR band from NIRGAN...")
        nir_synthetic = get_NIR(rgb_for_nir_gan, device=self.device)
        
        # Ensure NIR is 2D (H, W)
        if nir_synthetic.ndim == 3 and nir_synthetic.shape[0] == 1:
            nir_synthetic = nir_synthetic.squeeze(0)
        elif nir_synthetic.ndim == 3:
            nir_synthetic = nir_synthetic.squeeze()
        
        logger.debug(f"Generated NIR shape: {nir_synthetic.shape}")
        
        # ===== STEP 4: Stack RGB + Synthetic NIR =====
        # Extract individual RGB channels (already in CHW format)
        red = rgb_tensor_resized[0].cpu().numpy()      # (H, W)
        green = rgb_tensor_resized[1].cpu().numpy()    # (H, W)
        # blue is not used by OmniCloudMask
        
        # Use the new RGB+NIR handler for proper stacking
        handler = RGBNIRHandler(device=self.device)
        
        # Stack with proper band order: [RED, GREEN, NIR]
        rgn_stack = handler.stack_rgb_nir(
            red=red,
            green=green,
            nir=nir_synthetic,
            normalize=normalize,
            scale_to_dn=scale_to_dn,
            dn_range=dn_range
        )
        
        # ===== STEP 5: Validate result =====
        if validate_nir:
            nir_quality = handler.validate_nir_quality(nir_synthetic)
            logger.info(f"✓ NIR Quality: {nir_quality['info']}")
            logger.debug(f"  Mean: {nir_quality['mean']:.4f}, Std: {nir_quality['std']:.4f}")
        
        logger.debug(f"✓ Final RG+NIR stack shape: {rgn_stack.shape}")
        logger.debug(f"  Band order: [RED(0), GREEN(1), NIR(2)]")
        logger.debug(f"  Value range: [{rgn_stack.min():.4f}, {rgn_stack.max():.4f}]")
        
        return rgn_stack, rgb_tensor_resized.cpu().numpy()
    

    def mask_shadow_confidence(self, result, conf_thresh=-0.1):
        """
        Extract shadow detections with morphological refinement.
        Shadows have lower confidence than clouds, so use normalized thresholds.
        
        Args:
            result: Model confidence output (num_classes, H, W)
            conf_thresh: Threshold as percentage of max (-0.1 = use shadows above 10% of max)
        Returns:
            Tuple of shadow mask and average confidence
        """
        mask = np.argmax(result, axis=0)
        conf_shadow = result[3]
        
        # Base detection from argmax
        shadow_binary = (mask == 3).astype(np.uint8)
        
        # Apply morphological operations to connect shadow regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Closing to fill small gaps in shadows
        shadow_binary = cv2.morphologyEx(shadow_binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Dilation to expand shadow edges (shadow boundaries are typically soft)
        shadow_binary = cv2.dilate(shadow_binary, kernel, iterations=1)
        
        # Also include high-confidence non-argmax shadow pixels
        shadow_confidence_max = np.max(conf_shadow)
        if shadow_confidence_max > 0 and conf_thresh < 0:
            # Include pixels with confidence above threshold percentage
            high_confidence_shadows = (conf_shadow >= (-conf_thresh * shadow_confidence_max)).astype(np.uint8)
            shadow_binary = np.maximum(shadow_binary, high_confidence_shadows)
        
        shadow_mask = shadow_binary * 3
        shadow_pixels = shadow_mask == 3
        avg_conf_shadow = np.mean(conf_shadow[shadow_pixels]) if np.any(shadow_pixels) else 0.0
        
        return shadow_mask, avg_conf_shadow
    
    def mask_cloud_confidence(self, result, conf_thresh=0.0):
        """
        Extract cloud detections using argmax prediction (most reliable).
        Merges thick cloud (class 1) and thin cloud (class 2).
        
        Args:
            result (np.ndarray): Model output, shape (num_classes, H, W)
            conf_thresh (float): Confidence threshold (default 0.0 = no filtering)
        Returns:
            Tuple[np.ndarray, float]: Cloud mask and average confidence
        """
        mask = np.argmax(result, axis=0)
        conf_thick = result[1]
        conf_thin = result[2]
        conf_cloud = np.maximum(conf_thick, conf_thin)
        
        # Use argmax prediction for classes 1 or 2
        cloud_mask = np.where((mask == 1) | (mask == 2), 1, 0).astype(np.uint8)
        
        # Apply threshold only if specified
        if conf_thresh > 0:
            cloud_mask = np.where(
                ((mask == 1) | (mask == 2)) & (conf_cloud >= conf_thresh), 
                1, 
                0
            ).astype(np.uint8)
        
        cloud_pixels = cloud_mask == 1
        avg_conf_cloud = np.mean(conf_cloud[cloud_pixels]) if np.any(cloud_pixels) else 0.0
        
        return cloud_mask, avg_conf_cloud

    def predict(self, image_path: str):
        """
        Loads an image, detects clouds and shadows, and saves the result.
        """
        image_path = Path(image_path)
        logger.info(f"Processing: {image_path.name}")
        
        with rio.open(image_path) as src:
            # Read image as HWC
            rgb_array = np.transpose(src.read(), (1, 2, 0))

        rgn_input, original_rgb_resized = self.prepare_input_array(
            rgb_array,
            resize_to=(480, 520),
            normalize=True,
            scale_to_dn=True,
            validate_nir=True
        )

        logger.info("Running cloud and shadow detection...")
        # predict_from_array expects (bands, height, width)
        mask = predict_from_array(rgn_input)
        mask_2d = mask[0]  # The actual mask is the first item

        mask_conf = predict_from_array(rgn_input, export_confidence=True)
        
        # Shadows have lower confidence - use adaptive threshold (-0.1 = 10% of max)
        shadow_mask, shadow_conf = self.mask_shadow_confidence(mask_conf, conf_thresh=-0.1)
        
        cloud_mask, cloud_conf = self.mask_cloud_confidence(mask_conf, conf_thresh=0.0)
        
        mask_final = shadow_mask | cloud_mask
        
        # Log detection statistics
        total_pixels = mask_final.size
        cloud_pixels = np.sum(cloud_mask)
        shadow_pixels = np.sum(shadow_mask)
        cloud_pct = (cloud_pixels / total_pixels) * 100
        shadow_pct = (shadow_pixels / total_pixels) * 100
        logger.info(f"  Cloud pixels: {cloud_pixels:,} ({cloud_pct:.2f}%)")
        logger.info(f"  Shadow pixels: {shadow_pixels:,} ({shadow_pct:.2f}%)")
        logger.debug(f"  Cloud confidence: {cloud_conf:.3f}")
        logger.debug(f"  Shadow confidence: {shadow_conf:.3f}")

        logger.info("Detection complete. Saving visualization...")
        self.save_visualization(original_rgb_resized, mask_2d, image_path)

        return mask_final

    def save_visualization(self, rgb_array: np.ndarray, mask: np.ndarray, original_path: Path):
        """
        Saves a plot showing the original image, contours, and the mask.
        """
        output_path = self.output_folder / f"{original_path.stem}_cloud_shadow_mask.png"
        
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        
        # Convert CHW to HWC for displaying
        rgb_display = np.transpose(rgb_array, (1, 2, 0))
        # Normalize for display if it's not 8-bit
        if rgb_display.max() > 1:
             rgb_display = (rgb_display / rgb_display.max())
        
        # Panel 1: True color
        ax[0].imshow(rgb_display)
        ax[0].set_title("Original Image (Resized)")
        ax[0].axis("off")

        # Panel 2: True color + contours
        ax[1].imshow(rgb_display)
        # Clouds (class 1 and 2)
        ax[1].contour((mask == 1) | (mask == 2), colors="cyan", linewidths=1)
        # Shadow (class 3)
        ax[1].contour(mask == 3, colors="red", linewidths=1)
        ax[1].set_title("Prediction Contours")
        ax[1].axis("off")

        # Panel 3: Mask visualization
        cmap = ListedColormap(["black", "white", "lightblue", "yellow"])
        legend_labels = ["Clear", "Thick Cloud", "Thin Cloud", "Shadow"]
        patches = [mpatches.Patch(color=cmap(i/3.), label=legend_labels[i]) for i in range(4)]
        
        im = ax[2].imshow(mask, vmin=0, vmax=3, cmap=cmap, interpolation="nearest")
        ax[2].set_title("Cloud/Shadow Mask")
        ax[2].axis("off")
        
        fig.legend(handles=patches, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Visualization saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_dir = sys.argv[1]
    else:
        # The user-provided image path as a default
        img_dir = r"D:\Projects\QI47\2025_Projects\Image_QC\1_Data\Kavel10Data\testjanuary\dataset3\certiflAI_detected_shadows"
    
    if not os.path.exists(img_dir):
        logger.error(f"Error: The path does not exist: {img_dir}")
        logger.error("Please ensure the network path is accessible or provide a local file path as an argument.")
        sys.exit(1)

    output_dir = r"D:\Projects\QI47\2025_Projects\Image_QC\1_Data\Kavel10Data\testjanuary\dataset3\hitaish10"
    detector = OmniCloudShadowDetector(output_folder=output_dir)
    for img_path in Path(img_dir).glob("*.tif"):
        detector.predict(img_path)