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
# Add parent directory to path to import from thirdparty
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from thirdparty.NIRGAN.create_NIR import get_NIR
from rgb_nir_handler import RGBNIRHandler, prepare_rgb_with_synthetic_nir
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

    def prepare_input_array(self, rgb_array: np.ndarray, scale_factor: int = 2, to_size = (160, 220)) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepares the R-G-NIR stack for OmniCloudMask from an RGB image.
        - Resizes image for efficient processing.
        - Generates a synthetic NIR band using NIRGAN.
        - Scales Red, Green and NIR bands.
        """
        if rgb_array.shape[2] == 4:  # RGBA
            logger.warning("4-channel image detected, dropping alpha channel.")
            rgb_array = rgb_array[:, :, :3]
        
        if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
            raise ValueError("Input must be an RGB image (H, W, 3)")
            
        
        # Half size
        h, w, _ = rgb_array.shape
        rgb_half = rgb_array[h//2:, w//2:]
        # Convert to CHW format (Channels, Height, Width)
        rgb_chw = np.transpose(rgb_array, (2, 0, 1))
        # rgb_chw = np.transpose(rgb_array, (2, 0, 1))
            
        rgb_tensor = torch.tensor(rgb_chw, dtype=torch.float32, device=self.device)

        # Downsample for faster processing
        original_size = rgb_tensor.shape[1:]

        rgb_tensor_resized = F.interpolate(
            rgb_tensor.unsqueeze(0),
            size=to_size,
            mode="area").squeeze(0)
   
        # We need HWC numpy array for NIRGAN
        rgb_for_nir_gan = rgb_tensor_resized.permute(1, 2, 0).cpu().numpy()
        
        # Generate synthetic NIR
        logger.debug("Generating synthetic NIR band...")
        nir = get_NIR(rgb_for_nir_gan, device=self.device)
        nir = torch.tensor(nir, dtype=torch.float32, device=self.device)
        if nir.ndim == 2:
            nir = nir.unsqueeze(0)  # Add channel dimension -> (1, H, W)
            
        # Scale red, green and nir bands
        red = torch.clamp(rgb_tensor_resized[0] * 3, 0, 65535)
        green = torch.clamp(rgb_tensor_resized[1] * 2, 0, 65535)
        nir = torch.clamp(nir * 1, 0, 65535)
        # Stack to create Red-Green-NIR input
        rgn_stack = torch.stack([red, green, nir.squeeze(0)], dim=0)
        rgn_stack = rgn_stack.cpu().numpy()


        # red = torch.clamp(rgb_tensor_resized[0], 0, 65535)
        # green = torch.clamp(rgb_tensor_resized[1], 0, 65535)

        # # by using normalize  
        # normalize = True
        # scale_to_dn = True
        # dn_range = ((0, 10000))
        # handler = RGBNIRHandler(device=self.device)
        # red = red.cpu().numpy()
        # green = green.cpu().numpy()
        # nir = nir[0].cpu().numpy()
        # rgn_stack = handler.stack_rgb_nir(
        #     red=red,
        #     green=green,
        #     nir=nir,
        #     normalize=normalize,
        #     scale_to_dn=scale_to_dn,
        #     dn_range=dn_range
        # )
        return rgn_stack, rgb_tensor_resized.cpu().numpy()
    

    def mask_shadow_confidence(self, result, conf_thresh=0.10):
        """
        Use only clear and shadow class confidences, normalize their sum to 1, and mask shadow where confidence > conf_thresh.
        Args:
            result (np.ndarray): Model output, shape (num_classes, H, W)
            conf_thresh (float): Confidence threshold for shadow (default 0.10)
        Returns:
            np.ndarray: Mask with 1 for shadow (conf > thresh), 0 for clear.
        """
        # Compute argmax class mask (H, W)
        mask = np.argmax(result, axis=0)
        # Extract shadow confidence map (H, W)
        conf_shadow = result[3]
        
        mask_out = np.where(conf_shadow >= conf_thresh, 3, 0)
        # Mask of high-confidence cloud pixels
        high_conf_mask = conf_shadow > conf_thresh
        avg_conf_shadow = (
            np.mean(conf_shadow[high_conf_mask])
            if np.any(high_conf_mask)
            else 0.0
        )
        mask_out[(mask == 1) | (mask == 2)] = 0 
        final_mask = mask_out
        shadow_pxels = np.sum(final_mask == 3)
        return mask_out, avg_conf_shadow, shadow_pxels
    
    def mask_cloud_confidence(self,result, conf_thresh=0.10):
        """
        Mask based on merged cloud confidence (class 1+2), ignoring class 3.
        If merged cloud confidence >= threshold, mask=1 (cloud), else mask=0 (clear).
        """
        mask = np.argmax(result, axis=0)
        # result shape: (num_classes, H, W)
        conf_clear = result[0]
        conf_cloud = result[1] + result[2]   # Merge cloud classes
        # Remove class 3 (shadow) from consideration
        conf_sum = conf_clear + conf_cloud
        conf_sum = np.clip(conf_sum, 1e-8, None)  # prevent div by zero
        conf_clear_norm = conf_clear / conf_sum
        conf_cloud_norm = conf_cloud / conf_sum
        mask_out = np.where(conf_cloud_norm >= conf_thresh, 1, 0)

        # Mask of high-confidence cloud pixels
        high_conf_mask = conf_cloud_norm > conf_thresh
        avg_conf_cloud = (
            np.mean(conf_cloud_norm[high_conf_mask])
            if np.any(high_conf_mask)
            else 0.0
        )
        mask_out[mask==3] = 0 
        
        final_mask = mask_out

        cloud_px = np.sum(final_mask == 1)
        
        return final_mask , avg_conf_cloud, cloud_px

    def predict(self, image_path: str):
        """
        Loads an image, detects clouds and shadows using dual-resolution approach, and saves the result.
        
        Dual-resolution strategy:
        - Resolution 1 (Default): 480x520
        - Resolution 2 (Small): 160x220
        - Predictions are run on both resolutions.
        - Small resolution masks are upscaled to Default resolution.
        - Final masks are created using a logical OR operation: (Default | Upscaled Small).
        - Clouds take precedence over shadows in the final combined mask.
        - Result is saved at Default resolution (480x520).
        """
        image_path = Path(image_path)
        logger.info(f"Processing: {image_path.name}")

        with rio.open(image_path) as src:
            # Read image as HWC
            rgb_array = np.transpose(src.read(), (1, 2, 0))

        RES_DEFAULT = (480, 520)
        RES_SMALL = (160, 220)
        # RES_DEFAULT = (480, 520)
        # RES_SMALL = (160, 220)

        # Prepare two downsampled versions for dual-resolution detection
      
        # First: Default resolution (480x520)
        rgb_input_def, rgb_resized_def = self.prepare_input_array(rgb_array, scale_factor=2, to_size=RES_DEFAULT)

        # Second: Small resolution (160x220)
        rgn_input_small, _ = self.prepare_input_array(rgb_array, scale_factor=3, to_size=RES_SMALL)
        

        logger.info("Running dual-resolution cloud and shadow detection...")
        logger.debug(f"  Resolution Default: {RES_DEFAULT}")
        logger.debug(f"  Resolution Small: {RES_SMALL}")
        
        # --- Small Resolution Processing ---
        logger.debug(f"  Processing small resolution {RES_SMALL}...")
        mask_conf_small = predict_from_array(rgn_input_small, export_confidence=True)
        shadow_mask_small, _, _ = self.mask_shadow_confidence(mask_conf_small, conf_thresh=0.1)
        cloud_mask_small, _, _ = self.mask_cloud_confidence(mask_conf_small, conf_thresh=0.1)
        
        # --- Default Resolution Processing ---
        logger.debug(f"  Processing default resolution {RES_DEFAULT}...")
        mask_conf_def = predict_from_array(rgb_input_def, export_confidence=True)
        shadow_mask_def, _, _ = self.mask_shadow_confidence(mask_conf_def, conf_thresh=0.1)
        cloud_mask_def, _, _ = self.mask_cloud_confidence(mask_conf_def, conf_thresh=0.1)

        # --- Upscale Small Masks to Default Resolution ---
        logger.debug("  Upscaling small masks and merging...")
        
        # Resize using Nearest Neighbor to preserve class values (0, 1, 3)
        # cv2.resize expects (width, height)
        target_size = (RES_DEFAULT[1], RES_DEFAULT[0]) # (W, H)
        
        shadow_mask_small_upscaled = cv2.resize(shadow_mask_small.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)
        cloud_mask_small_upscaled = cv2.resize(cloud_mask_small.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)

        # --- OR Operation ---
        # Combine masks: if present in either resolution, keep it.
        final_shadow = shadow_mask_def | shadow_mask_small_upscaled
        final_cloud = cloud_mask_def | cloud_mask_small_upscaled
        
        # --- Create Final Categorical Mask ---
        # Initialize with zeros (Clear)
        final_mask = np.zeros(RES_DEFAULT, dtype=np.uint8)
        
        # Apply Shadows (value 3)
        final_mask[final_shadow == 3] = 3
        
        # Apply Clouds (value 1) - Overwrites shadows if they overlap
        final_mask[final_cloud == 1] = 1

        logger.debug(f"  Using default resolution {RES_DEFAULT} for visualization")
        logger.info("Detection complete. Saving visualization...")
        
        # Use the default resolution RGB for visualization
        self.save_visualization(rgb_resized_def, final_mask, image_path)

        return final_mask

    def save_visualization(self, rgb_array: np.ndarray, mask: np.ndarray, original_path: Path):
        """
        Saves a plot showing the original image, contours, and the mask.
        
        Dynamically handles RGB images from different resolutions:
        - Accepts RGB in CHW format and resizes mask if needed to match
        - Ensures proper alignment between RGB display and mask overlay
        """
        output_path = self.output_folder / f"{original_path.stem}_cloud_shadow_mask.png"
        
        # Get RGB and mask dimensions
        rgb_h, rgb_w = rgb_array.shape[1], rgb_array.shape[2]
        mask_h, mask_w = mask.shape[0], mask.shape[1]
        
        # Resize mask if dimensions don't match RGB
        if (mask_h != rgb_h) or (mask_w != rgb_w):
            logger.debug(f"  Resizing mask from {mask_h}x{mask_w} to match RGB {rgb_h}x{rgb_w}")
            mask = cv2.resize(mask.astype(np.uint8), (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)
        
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
        img_dir = r"D:\Projects\QI47\2025_Projects\Image_QC\1_Data\Kavel10Data\testjanuary\dataset3\certiflAI_detected_clouds"
    
    if not os.path.exists(img_dir):
        logger.error(f"Error: The path does not exist: {img_dir}")
        logger.error("Please ensure the network path is accessible or provide a local file path as an argument.")
        sys.exit(1)

    output_dir = r"D:\Projects\QI47\2025_Projects\Image_QC\1_Data\Kavel10Data\testjanuary\dataset3\hitaicl21_v1.7"
    detector = OmniCloudShadowDetector(output_folder=output_dir)
    for img_path in Path(img_dir).glob("*.tif"):
        # if img_name := img_path.name != '25FD0905Dx_40007_008712.tif':
        #     continue
        detector.predict(img_path)