
import sys
import os
from pathlib import Path
import numpy as np
import torch
import cv2
import rasterio as rio
from functools import partial
import timm
from fastai.vision.all import create_unet_model
from safetensors.torch import load_file

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- SETUP PATHS ---
current_script_dir = Path(__file__).parent.resolve()
training_dir = current_script_dir.parent
project_root = training_dir.parent

# Add paths for imports
if str(training_dir) not in sys.path:
    sys.path.append(str(training_dir))
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
if str(current_script_dir) not in sys.path:
    sys.path.append(str(current_script_dir))

# Import third-party modules
try:
    from omnicloudmask.cloud_mask import predict_from_array
    # Assuming NIRGAN is in training/scripts/thirdparty/NIRGAN
    from thirdparty.NIRGAN.create_NIR import get_NIR
except ImportError as e:
    logger.warning(f"Import Error: {e}")
    logger.debug(f"sys.path: {sys.path}")
    # Try alternate path for NIRGAN if running from different context
    try:
        sys.path.append(str(current_script_dir / "thirdparty"))
        from NIRGAN.create_NIR import get_NIR
        from omnicloudmask.cloud_mask import predict_from_array
    except ImportError as e2:
        logger.critical(f"Critical Import Error: {e2}")
        sys.exit(1)

# --- LOCAL CONFIG IMPORT ---
try:
    if str(current_script_dir) not in sys.path:
        sys.path.append(str(current_script_dir))
    from local_config import (
        TEST_MODEL_PATH, 
        TEST_IMAGES_DIR as CFG_TEST_IMAGES_DIR, 
        TEST_OUTPUT_DIR,
        USE_DUAL_RES_METHOD as CFG_USE_DUAL_RES_METHOD
    )
except ImportError:
    logger.critical("CRITICAL: local_config.py not found. Please create 'training/scripts/local_config.py' to define local paths.")
    sys.exit(1)

# --- CONFIGURATION ---
# Defaults (can be overridden by local_config)
MODEL_TYPE = getattr(sys.modules.get('local_config'), 'MODEL_TYPE', "regnety_004.pycls_in1k")
NUM_CHANNELS = 3  # R, G, NIR
MODEL_PATCH_SIZE = (509, 509)

# Scaling: 0.1 converts ~1m pixels to ~10m (Sentinel-2 scale)
# Set to 1.0 if your data is already at target resolution (e.g. Sentinel-2 L2A)
SCALE_FACTOR = getattr(sys.modules.get('local_config'), 'SCALE_FACTOR', 0.1) 

# Bands to read from the input .tif
# Common 4-band Aerial (R, G, B, NIR) -> Use [1, 2, 4] to get R, G, NIR
# Common 3-band Aerial (R, G, NIR)    -> Use [1, 2, 3]
# Common 3-band RGB    (R, G, B)      -> Use [1, 2, 3] AND set GENERATE_SYNTHETIC_NIR = True
BAND_ORDER = getattr(sys.modules.get('local_config'), 'BAND_ORDER', [1, 2, 3]) 

# Set True ONLY if your input is RGB and you need to fake the NIR channel
GENERATE_SYNTHETIC_NIR = getattr(sys.modules.get('local_config'), 'GENERATE_SYNTHETIC_NIR', False)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- INFERENCE METHOD TOGGLE ---
# True: Use Dual-Res (480x520 + 160x220) + Clamp Scaling (Red*3 etc.)
# False: Use Z-Score Norm + Tiled Inference (Current Training Default)
USE_DUAL_RES_METHOD = CFG_USE_DUAL_RES_METHOD 

# Path to the fine-tuned model
MODEL_PATH = TEST_MODEL_PATH

# Test Data Directory
TEST_IMAGES_DIR = CFG_TEST_IMAGES_DIR
OUTPUT_DIR = TEST_OUTPUT_DIR

class OCMTester:
    def __init__(self, model_path: Path):
        self.device = DEVICE
        self.model = self._load_model(model_path)
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Inference Method: {'Dual-Res (New)' if USE_DUAL_RES_METHOD else 'Z-Score Tiled (Legacy)'}")
        logger.info(f"Model Type: {MODEL_TYPE}")
        logger.info(f"Bands: {BAND_ORDER} | Scale: {SCALE_FACTOR} | Synthetic NIR: {GENERATE_SYNTHETIC_NIR}")

    def _load_model(self, model_path: Path):
        logger.info(f"Loading model from {model_path}...")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Create model architecture
        timm_model = partial(
            timm.create_model,
            MODEL_TYPE,
            pretrained=False,
            in_chans=NUM_CHANNELS,
        )
        model = create_unet_model(
            img_size=MODEL_PATCH_SIZE,
            arch=timm_model,
            n_out=4, # Clear, Thick, Thin, Shadow
            pretrained=False,
            act_cls=torch.nn.Mish,
        )

        # Load weights
        try:
            if model_path.suffix == '.safetensors':
                state_dict = load_file(model_path)
            else:
                state_dict = torch.load(model_path, map_location='cpu')
            
            model.load_state_dict(state_dict, strict=False)
            model.to(self.device)
            model.eval()
            logger.info("Model loaded successfully.")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {e}")

    def z_score_normalize(self, image: np.ndarray, no_data_value: float = 0.0) -> np.ndarray:
        """
        Apply Z-Score normalization to the image (C, H, W).
        Matches training normalization logic.
        """
        x = torch.from_numpy(image).float().to(self.device)
        mask = x != no_data_value
        epsilon = 1e-8
        
        valid_pixels = mask.sum(dim=(1, 2), keepdim=True)
        valid_pixels = torch.clamp(valid_pixels, min=1.0)
        
        mean = (x * mask).sum(dim=(1, 2), keepdim=True) / valid_pixels
        
        diff_sq = (x - mean)**2 * mask
        std = torch.sqrt(diff_sq.sum(dim=(1, 2), keepdim=True) / valid_pixels + epsilon)
        
        normalized = torch.where(mask, (x - mean) / std, torch.zeros_like(x))
        return normalized.cpu().numpy()

    def prepare_input_array_dual_res(self, rgb_array: np.ndarray, to_size) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepares the R-G-NIR stack using Clamp & Scale logic (Method B).
        """
        if rgb_array.shape[2] == 4:  # RGBA
            rgb_array = rgb_array[:, :, :3]
        
        # Convert to CHW format (Channels, Height, Width)
        rgb_chw = np.transpose(rgb_array, (2, 0, 1))
        rgb_tensor = torch.tensor(rgb_chw, dtype=torch.float32, device=self.device)

        # Downsample
        rgb_tensor_resized = torch.nn.functional.interpolate(
            rgb_tensor.unsqueeze(0),
            size=to_size,
            mode="area").squeeze(0)
   
        # HWC for NIRGAN
        rgb_for_nir_gan = rgb_tensor_resized.permute(1, 2, 0).cpu().numpy()
        
        # Generate synthetic NIR
        nir = get_NIR(rgb_for_nir_gan, device=self.device)
        nir = torch.tensor(nir, dtype=torch.float32, device=self.device)
        if nir.ndim == 2:
            nir = nir.unsqueeze(0)  # (1, H, W)
            
        # Scale red, green and nir bands (Specific to Method B)
        red = torch.clamp(rgb_tensor_resized[0] * 3, 0, 65535)
        green = torch.clamp(rgb_tensor_resized[1] * 2, 0, 65535)
        nir = torch.clamp(nir * 1, 0, 65535)
        
        # Stack
        rgn_stack = torch.stack([red, green, nir.squeeze(0)], dim=0)
        return rgn_stack.cpu().numpy(), rgb_tensor_resized.cpu().numpy()

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

    def process_image_dual_res(self, image_path: Path):
        """
        Method B: Dual Resolution + Clamp Scaling
        """
        # print(f"Processing (Dual Res) {image_path.name}...")
        try:
            with rio.open(image_path) as src:
                rgb_array = np.transpose(src.read([1, 2, 3]), (1, 2, 0))

            RES_DEFAULT = (480, 520)
            RES_SMALL = (160, 220)

            # 1. Prepare Default Res
            rgb_input_def, rgb_resized_def = self.prepare_input_array_dual_res(rgb_array, to_size=RES_DEFAULT)
            # 2. Prepare Small Res
            rgn_input_small, _ = self.prepare_input_array_dual_res(rgb_array, to_size=RES_SMALL)

            # --- Small Res Inference ---
            mask_conf_small = predict_from_array(
                rgn_input_small, 
                custom_models=[self.model],
                inference_device=self.device,
                export_confidence=True,
                batch_size=1, # Dual res is single image based
                patch_size=2000 # Large patch size to avoid tiling on small images
            )
            shadow_mask_small, _ = self.mask_shadow_confidence(mask_conf_small, conf_thresh=0.1)
            cloud_mask_small, _ = self.mask_cloud_confidence(mask_conf_small, conf_thresh=0.1)

            # --- Default Res Inference ---
            mask_conf_def = predict_from_array(
                rgb_input_def, 
                custom_models=[self.model],
                inference_device=self.device,
                export_confidence=True,
                batch_size=1,
                patch_size=2000
            )
            shadow_mask_def, _ = self.mask_shadow_confidence(mask_conf_def, conf_thresh=0.1)
            cloud_mask_def, _ = self.mask_cloud_confidence(mask_conf_def, conf_thresh=0.1)

            # --- Upscale Small & Merge ---
            target_size = (RES_DEFAULT[1], RES_DEFAULT[0]) # (W, H)
            
            shadow_mask_small_upscaled = cv2.resize(shadow_mask_small.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)
            cloud_mask_small_upscaled = cv2.resize(cloud_mask_small.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)

            final_shadow = shadow_mask_def | shadow_mask_small_upscaled
            final_cloud = cloud_mask_def #| cloud_mask_small_upscaled
            
            final_mask = np.zeros(RES_DEFAULT, dtype=np.uint8)
            final_mask[final_shadow == 3] = 3
            final_mask[final_cloud == 1] = 1

            # Save visualization (Using the Default Res RGB)
            # Need to transpose rgb_resized_def back to HWC for viz
            rgb_viz = np.transpose(rgb_resized_def, (1, 2, 0))
            self.save_visualization(rgb_viz, final_mask, image_path.stem)

        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {e}")
            import traceback
            traceback.print_exc()

    def process_image(self, image_path: Path):
        """
        Method A: Z-Score Tiled (Legacy)
        
        ADAPTED FOR CUSTOM DATA:
        - Reads bands specified in BAND_ORDER.
        - Supports direct NIR reading (no GAN) if available.
        - Scales based on SCALE_FACTOR.
        """
        logger.info(f"Processing (Z-Score) {image_path.name}...")
        
        try:
            with rio.open(image_path) as src:
                # Read specified bands
                # Note: src.read expects bands to be 1-indexed
                try:
                    raw_bands = src.read(BAND_ORDER)
                except Exception as e:
                    logger.error(f"Error reading bands {BAND_ORDER} from {image_path.name}: {e}")
                    return

                if raw_bands.shape[0] != 3:
                    logger.warning(f"Skipping {image_path.name}: Expected 3 bands (R, G, NIR), got {raw_bands.shape[0]}")
                    return

            # Transpose to HWC for resizing/processing
            # raw_bands is (3, H, W)
            img_hwc = np.transpose(raw_bands, (1, 2, 0))
            h_native, w_native = img_hwc.shape[:2]

            # --- DOWNSAMPLE ---
            if SCALE_FACTOR != 1.0:
                h_infer = int(h_native * SCALE_FACTOR)
                w_infer = int(w_native * SCALE_FACTOR)
                
                # Ensure minimum size
                h_infer = max(h_infer, 64)
                w_infer = max(w_infer, 64)

                logger.debug(f"Resizing: {w_native}x{h_native} -> {w_infer}x{h_infer} (Scale: {SCALE_FACTOR})")
                img_infer = cv2.resize(img_hwc, (w_infer, h_infer), interpolation=cv2.INTER_AREA)
            else:
                h_infer, w_infer = h_native, w_native
                img_infer = img_hwc

            # --- PREPARE STACK ---
            if GENERATE_SYNTHETIC_NIR:
                logger.debug("Generating synthetic NIR (GENERATE_SYNTHETIC_NIR=True)...")
                # Assuming img_infer is RGB
                # GAN requires 0-255 uint8 input
                if img_infer.dtype != np.uint8:
                     # Normalize to 0-255 for GAN
                     rgb_gan = ((img_infer - img_infer.min()) / (img_infer.max() - img_infer.min() + 1e-8) * 255).astype(np.uint8)
                else:
                    rgb_gan = img_infer
                
                nir_syn = get_NIR(rgb_gan, device=self.device)
                
                # Ensure shape match
                if nir_syn.shape != (h_infer, w_infer):
                     nir_syn = cv2.resize(nir_syn, (w_infer, h_infer), interpolation=cv2.INTER_LINEAR)
                
                red = img_infer[:, :, 0].astype(np.float32)
                green = img_infer[:, :, 1].astype(np.float32)
                nir = nir_syn.astype(np.float32)
                
                rgn_stack = np.stack([red, green, nir], axis=0) # (3, H, W)
            else:
                # Assume input bands are already R, G, NIR (or whatever the model expects)
                # Just transpose back to CHW for normalization
                rgn_stack = np.transpose(img_infer, (2, 0, 1)).astype(np.float32)

            # --- NORMALIZE ---
            rgn_norm = self.z_score_normalize(rgn_stack)
            
            # --- INFERENCE ---
            logger.info(f"Running inference on image ({w_infer}x{h_infer})...")
            mask_conf = predict_from_array(
                rgn_norm,
                custom_models=[self.model],
                inference_device=self.device,
                batch_size=4,
                patch_size=MODEL_PATCH_SIZE[0],
                export_confidence=True
            )
            
            # --- POST-PROCESSING ---
            shadow_mask, shadow_conf = self.mask_shadow_confidence(mask_conf, conf_thresh=-0.1)
            cloud_mask, cloud_conf = self.mask_cloud_confidence(mask_conf, conf_thresh=0.0)
            
            mask_final = np.zeros_like(shadow_mask, dtype=np.uint8)
            mask_final[shadow_mask == 3] = 3
            mask_final[cloud_mask == 1] = 1 

            # --- SAVE RESULT ---
            # We pass the inference image for viz
            self.save_visualization(img_infer, mask_final, image_path.stem)
            
        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {e}")
            import traceback
            traceback.print_exc()

    def save_visualization(self, rgb_img, mask, base_name):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # Match RGB to mask size for display
        h_mask, w_mask = mask.shape
        h_rgb, w_rgb = rgb_img.shape[:2]
        
        if (h_rgb != h_mask) or (w_rgb != w_mask):
            rgb_disp_raw = cv2.resize(rgb_img, (w_mask, h_mask), interpolation=cv2.INTER_AREA)
        else:
            rgb_disp_raw = rgb_img

        # Normalize for display
        if rgb_disp_raw.dtype != np.uint8:
             rgb_disp = ((rgb_disp_raw - rgb_disp_raw.min()) / (rgb_disp_raw.max() - rgb_disp_raw.min()) * 255).astype(np.uint8)
        else:
            rgb_disp = rgb_disp_raw
            
        ax[0].imshow(rgb_disp)
        ax[0].set_title(f"RGB ({w_mask}x{h_mask})")
        ax[0].axis("off")
        
        # Mask
        # 0: Clear, 1: Thick Cloud, 2: Thin Cloud, 3: Shadow
        cmap = ListedColormap(["black", "white", "lightgray", "red"])
        
        ax[1].imshow(mask, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        ax[1].set_title("Prediction (White=Thick, Gray=Thin, Red=Shadow)")
        ax[1].axis("off")
        
        out_file = self.output_dir / f"{base_name}_prediction.png"
        plt.tight_layout()
        plt.savefig(out_file)
        plt.close()
        # logger.info(f"Saved visualization to {out_file}")

if __name__ == "__main__":
    if not MODEL_PATH.exists():
        logger.error(f"Error: Model not found at {MODEL_PATH}")
        logger.error("Please ensure you have run the training script or updated the path.")
        sys.exit(1)
        
    tester = OCMTester(MODEL_PATH)
    
    # Process images
    # You can change the glob pattern or directory
    images = list(TEST_IMAGES_DIR.glob("*.tif")) + list(TEST_IMAGES_DIR.glob("*.iiq")) + list(TEST_IMAGES_DIR.glob("*.jpg"))
    if not images:
        logger.warning(f"No .tif images found in {TEST_IMAGES_DIR}")
    else:
        logger.info(f"Found {len(images)} images. Processing...")
        # Process a few samples to save time, or all
        for img_path in images: # Process first 5 for testing
            if USE_DUAL_RES_METHOD:
                tester.process_image_dual_res(img_path)
            else:
                tester.process_image(img_path)
