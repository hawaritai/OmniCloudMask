
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

# --- CONFIGURATION ---
MODEL_TYPE = "regnety_004.pycls_in1k"
NUM_CHANNELS = 3  # R, G, NIR
MODEL_PATCH_SIZE = (509, 509)
TARGET_RESOLUTION = (480, 520) # (Height, Width)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the fine-tuned model
MODEL_PATH = project_root / "models" / "PM_model_OCM_7.43_R_G_NIR_test_regnety_004.pycls_in1k_PT_state.safetensors"

# Test Data Directory
TEST_IMAGES_DIR = project_root / "training" / "data" / "2051_102025077_D09_Arriege_D" / "images"
OUTPUT_DIR = project_root / "test_results"

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

    def process_image(self, image_path: Path):
        print(f"Processing {image_path.name}...")
        
        try:
            with rio.open(image_path) as src:
                # Read RGB (first 3 bands)
                rgb = src.read([1, 2, 3])
                if rgb.shape[0] != 3:
                    print(f"Skipping {image_path.name}: Expected 3 bands, got {rgb.shape[0]}")
                    return

            # --- DOWNSAMPLE LOGIC ---
            # Transpose to HWC for resizing
            rgb_hwc = np.transpose(rgb, (1, 2, 0))
            h_target, w_target = TARGET_RESOLUTION
            
            print(f"Downsampling to {w_target}x{h_target}...")
            rgb_resized_hwc = cv2.resize(rgb_hwc, (w_target, h_target), interpolation=cv2.INTER_AREA)

            # --- GENERATE NIR ---
            # Normalize for NIRGAN input if needed
            if rgb_resized_hwc.dtype != np.uint8:
                rgb_gan = ((rgb_resized_hwc - rgb_resized_hwc.min()) / (rgb_resized_hwc.max() - rgb_resized_hwc.min() + 1e-8) * 255).astype(np.uint8)
            else:
                rgb_gan = rgb_resized_hwc

            print("Generating synthetic NIR...")
            nir = get_NIR(rgb_gan, device=self.device)
            
            # NIR should already be at target resolution, but ensure exact match
            if nir.shape != (h_target, w_target):
                nir = cv2.resize(nir, (w_target, h_target), interpolation=cv2.INTER_LINEAR)

            # --- PREPARE STACK ---
            red = rgb_resized_hwc[:, :, 0].astype(np.float32)
            green = rgb_resized_hwc[:, :, 1].astype(np.float32)
            nir = nir.astype(np.float32)
            
            rgn_stack = np.stack([red, green, nir], axis=0) # (3, H, W)
            
            # --- NORMALIZE ---
            rgn_norm = self.z_score_normalize(rgn_stack)
            
            # --- INFERENCE ---
            print("Running inference...")
            mask = predict_from_array(
                rgn_norm,
                custom_models=[self.model],
                inference_device=self.device,
                batch_size=4,
                patch_size=MODEL_PATCH_SIZE[0]
            )
            
            final_mask = mask[0]
            
            # --- SAVE RESULT ---
            self.save_visualization(rgb_resized_hwc, final_mask, image_path.stem)
            
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            import traceback
            traceback.print_exc()

    def save_visualization(self, rgb_img, mask, base_name):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # RGB
        # Normalize for display
        if rgb_img.dtype != np.uint8:
             rgb_disp = ((rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min()) * 255).astype(np.uint8)
        else:
            rgb_disp = rgb_img
            
        ax[0].imshow(rgb_disp)
        ax[0].set_title("Original RGB")
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
