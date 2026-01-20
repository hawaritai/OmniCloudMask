
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
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

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
    print(f"Import Error: {e}")
    print(f"sys.path: {sys.path}")
    # Try alternate path for NIRGAN if running from different context
    try:
        sys.path.append(str(current_script_dir / "thirdparty"))
        from NIRGAN.create_NIR import get_NIR
        from omnicloudmask.cloud_mask import predict_from_array
    except ImportError as e2:
        print(f"Critical Import Error: {e2}")
        sys.exit(1)

# --- LOCAL CONFIG IMPORT ---
try:
    if str(current_script_dir) not in sys.path:
        sys.path.append(str(current_script_dir))
    from local_config import TEST_MODEL_PATH, TEST_IMAGES_DIR as CFG_TEST_IMAGES_DIR, TEST_OUTPUT_DIR
except ImportError:
    print("CRITICAL: local_config.py not found. Please create 'training/scripts/local_config.py' to define local paths.")
    sys.exit(1)

# --- CONFIGURATION ---
MODEL_TYPE = "regnety_004.pycls_in1k"
NUM_CHANNELS = 3  # R, G, NIR
MODEL_PATCH_SIZE = (509, 509)
# TARGET_RESOLUTION = (480, 520) # Deprecated
SCALE_FACTOR = 0.1 # Match the training preprocessing (1m -> 10m GSD = 0.1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        print(f"Output directory: {self.output_dir}")

    def _load_model(self, model_path: Path):
        print(f"Loading model from {model_path}...")
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
            print("Model loaded successfully.")
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

    def process_image(self, image_path: Path):
        print(f"Processing {image_path.name}...")
        
        try:
            with rio.open(image_path) as src:
                # Read RGB (first 3 bands)
                rgb = src.read([1, 2, 3])
                if rgb.shape[0] != 3:
                    print(f"Skipping {image_path.name}: Expected 3 bands, got {rgb.shape[0]}")
                    return

            # Transpose to HWC for processing
            rgb_hwc = np.transpose(rgb, (1, 2, 0))
            h_native, w_native = rgb_hwc.shape[:2]

            # --- DOWNSAMPLE to Match Training Scale ---
            # Training used ~10m GSD (Scale 0.1 from ~1m source)
            # We must replicate this or the model will see "zoomed in" noise
            h_infer = int(h_native * SCALE_FACTOR)
            w_infer = int(w_native * SCALE_FACTOR)
            
            # Ensure minimum size (at least 32x32 for model stability, though OCM handles padding)
            h_infer = max(h_infer, 64)
            w_infer = max(w_infer, 64)

            print(f"Downsampling Input: {w_native}x{h_native} -> {w_infer}x{h_infer} (Scale: {SCALE_FACTOR})")
            rgb_infer = cv2.resize(rgb_hwc, (w_infer, h_infer), interpolation=cv2.INTER_AREA)

            # --- GENERATE NIR ---
            # Prepare for GAN (requires 0-255 uint8)
            if rgb_infer.dtype != np.uint8:
                rgb_gan_input = ((rgb_infer - rgb_infer.min()) / (rgb_infer.max() - rgb_infer.min() + 1e-8) * 255).astype(np.uint8)
            else:
                rgb_gan_input = rgb_infer

            print("Generating synthetic NIR...")
            nir_infer = get_NIR(rgb_gan_input, device=self.device)
            
            # Ensure shape match exactly
            if nir_infer.shape != (h_infer, w_infer):
                 nir_infer = cv2.resize(nir_infer, (w_infer, h_infer), interpolation=cv2.INTER_LINEAR)

            # --- PREPARE STACK ---
            red = rgb_infer[:, :, 0].astype(np.float32)
            green = rgb_infer[:, :, 1].astype(np.float32)
            nir = nir_infer.astype(np.float32)
            
            rgn_stack = np.stack([red, green, nir], axis=0) # (3, H, W)
            
            # --- NORMALIZE ---
            rgn_norm = self.z_score_normalize(rgn_stack)
            
            # --- INFERENCE ---
            print(f"Running inference on scaled image ({w_infer}x{h_infer})...")
            # predict_from_array will tile this scaled image if it's larger than 509x509
            # Use export_confidence=True to get raw probabilities
            mask_conf = predict_from_array(
                rgn_norm,
                custom_models=[self.model],
                inference_device=self.device,
                batch_size=4,
                patch_size=MODEL_PATCH_SIZE[0],
                export_confidence=True
            )
            
            # --- POST-PROCESSING (Refine Clouds & Shadows) ---
            # Shadows: Aggressive refinement with morphology & thresholds
            shadow_mask, shadow_conf = self.mask_shadow_confidence(mask_conf, conf_thresh=-0.1)
            
            # Clouds: Merge classes 1 & 2
            cloud_mask, cloud_conf = self.mask_cloud_confidence(mask_conf, conf_thresh=0.0)
            
            # Combine: 1=Cloud, 3=Shadow. If both, Cloud wins (usually) or use logic.
            # Here we follow detector logic: mask_final = shadow_mask | cloud_mask
            # Since shadow_mask is 0 or 3, and cloud_mask is 0 or 1.
            # Overlap handling: Cloud usually obscures shadow, so if both exist, let's say Cloud takes precedence?
            # Or just bitwise OR: 3 | 1 = 3 (Shadow wins visual? No 1=01, 3=11). 
            # Actually, let's just layer them.
            
            mask_final_low_res = np.zeros_like(shadow_mask, dtype=np.uint8)
            mask_final_low_res[shadow_mask == 3] = 3
            mask_final_low_res[cloud_mask == 1] = 1 # Clouds overwrite shadows if overlap

            # --- SKIP UPSAMPLING ---
            # Save the mask at the inference resolution (0.1x)
            print(f"Keeping Mask at Inference Resolution: {w_infer}x{h_infer}")
            final_mask = mask_final_low_res
            
            # --- SAVE RESULT ---
            # We pass the original RGB and the small mask. 
            # The visualization function will handle the size difference.
            self.save_visualization(rgb_infer, final_mask, image_path.stem)
            
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
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
        print(f"Saved visualization to {out_file}")

if __name__ == "__main__":
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please ensure you have run the training script or updated the path.")
        sys.exit(1)
        
    tester = OCMTester(MODEL_PATH)
    
    # Process images
    # You can change the glob pattern or directory
    images = list(TEST_IMAGES_DIR.glob("*.tif"))
    if not images:
        print(f"No .tif images found in {TEST_IMAGES_DIR}")
    else:
        print(f"Found {len(images)} images. Processing...")
        # Process a few samples to save time, or all
        for img_path in images: # Process first 5 for testing
            tester.process_image(img_path)
