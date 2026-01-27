import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from pathlib import Path
import numpy as np
from tqdm import tqdm
import math
import sys
import os
import torch
import warnings
import cv2
import logging

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- PATH SETUP ---
# Add training/scripts to path to allow imports of helper modules
# We assume this script is running from the repo root
current_dir = Path.cwd()
scripts_dir = current_dir / "training" / "scripts"

if str(scripts_dir) not in sys.path:
    sys.path.append(str(scripts_dir))

# Also add thirdparty/NIRGAN to path if needed by its internal imports, 
# though create_NIR handles its own path. 
# But to import it via 'thirdparty.NIRGAN...', 'training/scripts' in path should suffice 
# if 'thirdparty' is a folder with __init__.py inside scripts.
# If not, we might need to adjust.

# Try imports
try:
    from thirdparty.NIRGAN.create_NIR import get_NIR
    from fom_R.rgb_nir_handler import RGBNIRHandler
    logger.info("Successfully imported NIRGAN and RGBNIRHandler.")
except ImportError as e:
    logger.warning(f"Error importing required modules: {e}")
    logger.warning(f"sys.path: {sys.path}")
    logger.info("Attempting to add subdirectories to path...")
    # Fallback: add thirdparty explicitly
    sys.path.append(str(scripts_dir / "thirdparty"))
    try:
        from NIRGAN.create_NIR import get_NIR
        from fom_R.rgb_nir_handler import RGBNIRHandler
        logger.info("Successfully imported NIRGAN (fallback).")
    except ImportError as e2:
        logger.critical(f"Critical Import Error: {e2}")
        sys.exit(1)

# --- CONFIGURATION ---
try:
    from local_config import (
        PREPROCESS_BASE_DIR,
        PREPROCESS_INPUT_IMAGES_DIR,
        PREPROCESS_INPUT_LABELS_DIR,
        PREPROCESS_OUTPUT_DIR
    )
    BASE_DATA_DIR = PREPROCESS_BASE_DIR
    INPUT_IMAGES_DIR = PREPROCESS_INPUT_IMAGES_DIR
    INPUT_LABELS_DIR = PREPROCESS_INPUT_LABELS_DIR
    OUTPUT_DIR = PREPROCESS_OUTPUT_DIR
except ImportError:
    logger.critical("CRITICAL: local_config.py not found. Please create 'training/scripts/local_config.py' to define local paths.")
    sys.exit(1)

TARGET_GSD_M = 10.0  # Target resolution in meters (Match Sentinel-2 approx)
SOURCE_GSD_M = 1  # Your source resolution (5cm) - ADJUST IF NEEDED
TILE_SIZE = 509      # Required by OCM training script

# --- PREPROCESSING METHOD TOGGLE ---
# False: Use RGBNIRHandler with normalization (Method A)
# True: Use Clamp & Scale (Red*3, Green*2, NIR*1) (Method B)
USE_METHOD_B_PREPROCESSING = False 
# ---------------------

def preprocess_images():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Preprocessing Method: {'Clamp & Scale (Method B)' if USE_METHOD_B_PREPROCESSING else 'RGBNIRHandler (Method A)'}")

    # Initialize Handler (only needed for Method A)
    handler = RGBNIRHandler(device=device) if not USE_METHOD_B_PREPROCESSING else None

    # Create output directories
    (OUTPUT_DIR / "train").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "validation").mkdir(parents=True, exist_ok=True)

    # Calculate scale factor (e.g. 0.05 / 10 = 0.005)
    scale_factor = SOURCE_GSD_M / TARGET_GSD_M
    logger.info(f"Downsampling scale factor: {scale_factor} (Source: {SOURCE_GSD_M}m -> Target: {TARGET_GSD_M}m)")

    # Find all images
    image_files = list(INPUT_IMAGES_DIR.glob("*.tif"))
    logger.info(f"Found {len(image_files)} images to process.")

    for img_path in tqdm(image_files):
        # Look for label in label dir with same name
        label_path = INPUT_LABELS_DIR / img_path.name
        
        if not label_path.exists():
            logger.warning(f"Skipping {img_path.name}, label not found in {INPUT_LABELS_DIR}")
            continue

        try:
            with rasterio.open(img_path) as src_img, rasterio.open(label_path) as src_lbl:
                # 1. Calculate new dimensions after downsampling
                new_height = int(src_img.height * scale_factor)
                new_width = int(src_img.width * scale_factor)
                
                # Check for minimum dimensions
                if new_height < 1 or new_width < 1:
                    logger.warning(f"Skipping {img_path.name}: Downsampled size too small ({new_width}x{new_height})")
                    continue

                # 2. Read and Resample (Downsample) entire image to memory
                # Read RGB (3 bands)
                count = min(3, src_img.count)
                if count < 3:
                    logger.warning(f"Skipping {img_path.name}: Not enough bands ({count})")
                    continue
                    
                data_img_rgb = src_img.read(
                    [1, 2, 3], # Read bands 1, 2, 3 (RGB)
                    out_shape=(3, new_height, new_width),
                    resampling=Resampling.bilinear
                )
                
                # Use nearest neighbor for labels to preserve class integers (0,1,2,3)
                data_lbl = src_lbl.read(
                    out_shape=(src_lbl.count, new_height, new_width),
                    resampling=Resampling.nearest
                )

                # --- 2b. Generate NIR and Stack ---
                # Prepare RGB for NIRGAN (H, W, 3)
                rgb_for_gan = np.transpose(data_img_rgb, (1, 2, 0)) # CHW -> HWC
                
                # Check dtype & Scale to 0-255 for GAN
                if data_img_rgb.dtype == np.uint16:
                    rgb_for_gan_u8 = (rgb_for_gan / 65535.0 * 255.0).astype(np.uint8)
                elif data_img_rgb.dtype != np.uint8:
                    if data_img_rgb.max() <= 1.0:
                        rgb_for_gan_u8 = (rgb_for_gan * 255.0).astype(np.uint8)
                    else:
                         rgb_for_gan_u8 = rgb_for_gan.astype(np.uint8)
                else:
                    rgb_for_gan_u8 = rgb_for_gan

                # Now run GAN
                nir_synthetic = get_NIR(rgb_for_gan_u8, device=device)
                
                # --- RESIZE NIR TO MATCH RGB IF NEEDED ---
                if nir_synthetic.shape != (new_height, new_width):
                    nir_synthetic = nir_synthetic.astype(np.float32)
                    nir_synthetic = cv2.resize(nir_synthetic, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

                # --- STACKING LOGIC ---
                if USE_METHOD_B_PREPROCESSING:
                    # --- METHOD B: Clamp & Scale ---
                    # Logic from cloud_shadow_detect_main.py:
                    # red = clamp(input * 3, 0, 65535)
                    # green = clamp(input * 2, 0, 65535)
                    # nir = clamp(nir * 1, 0, 65535)
                    
                    # Ensure inputs are float for math
                    r = data_img_rgb[0].astype(np.float32)
                    g = data_img_rgb[1].astype(np.float32)
                    n = nir_synthetic.astype(np.float32)
                    
                    # Normalize inputs to 0-1 range first if they aren't already
                    # Assuming input might be uint8 or uint16.
                    # Method B usually expects roughly "visual" values to be multiplied.
                    # Let's check source: `rgb_tensor_resized[0] * 3`. `rgb_tensor_resized` came from `rgb_array` (0-255 if uint8, or 0-1 float).
                    # If source is uint8 (0-255): 255 * 3 = 765. Not 65535.
                    # If source is uint16 (0-65535): 65535 * 3 >> 65535.
                    # Actually `cloud_shadow_detect_main.py` reads with rasterio, gets whatever dtype. 
                    # If we assume standard RGB (0-255), then the output is approx 0-765 range.
                    # If we assume 16-bit, it saturates immediately.
                    # Let's assume we want to map the *relative* intensity.
                    
                    # Implementation detail: The user's provided script uses `rgb_tensor_resized` which came from `rgb_array`.
                    # It likely expects standard normalized input or raw DNs.
                    # We will implement exact logic: Raw Value * Factor.
                    
                    r_scaled = np.clip(r * 3, 0, 65535)
                    g_scaled = np.clip(g * 2, 0, 65535)
                    n_scaled = np.clip(n * 1, 0, 65535)
                    
                    rgn_stack = np.stack([r_scaled, g_scaled, n_scaled], axis=0).astype(np.float32)
                    
                else:
                    # --- METHOD A: RGBNIRHandler (Normalized) ---
                    red = data_img_rgb[0].astype(np.float32)
                    green = data_img_rgb[1].astype(np.float32)
                    
                    rgn_stack = handler.stack_rgb_nir(
                        red=red,
                        green=green,
                        nir=nir_synthetic,
                        normalize=True,
                        scale_to_dn=True,
                        dn_range=(0, 10000)
                    )
                
                # rgn_stack is (3, H, W) float32

                # 3. Tile into 509x509 patches
                n_cols = math.ceil(new_width / TILE_SIZE)
                n_rows = math.ceil(new_height / TILE_SIZE)

                for row in range(n_rows):
                    for col in range(n_cols):
                        # Define window
                        x_off = col * TILE_SIZE
                        y_off = row * TILE_SIZE
                        
                        # Adjust offset if tile exceeds boundaries (force overlap for last tiles)
                        if x_off + TILE_SIZE > new_width:
                            x_off = max(0, new_width - TILE_SIZE)
                        
                        if y_off + TILE_SIZE > new_height:
                            y_off = max(0, new_height - TILE_SIZE)

                        # Extract tile from RGN stack
                        tile_img = rgn_stack[:, y_off:y_off+TILE_SIZE, x_off:x_off+TILE_SIZE]
                        
                        # Extract tile from label
                        tile_lbl = data_lbl[:, y_off:y_off+TILE_SIZE, x_off:x_off+TILE_SIZE]

                        # Verify it's not empty (optional)
                        if tile_img.max() == 0: continue

                        # Define output filenames
                        split_dir = "validation" if np.random.rand() < 0.2 else "train"
                        base_name = f"{img_path.stem}_tile_{row}_{col}"
                        
                        out_img_path = OUTPUT_DIR / split_dir / f"{base_name}_image.tif"
                        out_lbl_path = OUTPUT_DIR / split_dir / f"{base_name}_label.tif"

                        # Create a basic transform for the tile (we lose absolute georef but keep pixel scale relative)
                        # We use an identity transform or similar since these are training chips
                        # Or we can construct one if needed, but typically OCM training works on pixel values.
                        dst_transform = rasterio.Affine(scale_factor, 0, 0, 0, -scale_factor, 0) # Simplified

                        # Save Tile (Image)
                        profile = src_img.profile.copy()
                        profile.update({
                            'height': TILE_SIZE,
                            'width': TILE_SIZE,
                            'count': 3,
                            'dtype': 'float32', 
                            'driver': 'GTiff',
                            'compress': 'lzw',
                            'transform': dst_transform,
                            'crs': None # remove CRS to avoid warnings if not valid
                        })

                        with rasterio.open(out_img_path, 'w', **profile) as dst:
                            dst.write(tile_img)
                        
                        # Save Tile (Label)
                        profile_lbl = src_lbl.profile.copy()
                        profile_lbl.update({
                            'height': TILE_SIZE,
                            'width': TILE_SIZE,
                            'count': 1,
                            'dtype': 'uint8',
                            'compress': 'lzw',
                            'transform': dst_transform,
                            'crs': None
                        })
                        with rasterio.open(out_lbl_path, 'w', **profile_lbl) as dst:
                            dst.write(tile_lbl)
                            
        except Exception as e:
            logger.error(f"Failed to process {img_path.name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    preprocess_images()
