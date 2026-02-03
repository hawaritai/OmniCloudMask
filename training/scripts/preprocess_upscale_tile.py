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
        PREPROCESS_INPUT_IMAGES_DIR,
        PREPROCESS_INPUT_LABELS_DIR,
        PREPROCESS_OUTPUT_DIR
    )

    # INPUT_IMAGES_DIR = Path(r"E:\ImageQC\dataset\Test_all_gt\images_pos")
    # INPUT_LABELS_DIR = Path(r"D:\projects\Image_QC_GUI\2_Repos\OmniCloudMask\training\data\all_masks")
    # OUTPUT_DIR = Path(r"D:\projects\Image_QC_GUI\2_Repos\OmniCloudMask\training\data\processed_data_upscale")

    INPUT_IMAGES_DIR = PREPROCESS_INPUT_IMAGES_DIR
    INPUT_LABELS_DIR = PREPROCESS_INPUT_LABELS_DIR
    OUTPUT_DIR = PREPROCESS_OUTPUT_DIR
    
except ImportError:
    logger.critical("CRITICAL: local_config.py not found.")
    logger.critical("Required variables: PREPROCESS_INPUT_IMAGES_DIR, PREPROCESS_INPUT_LABELS_DIR, PREPROCESS_OUTPUT_DIR")
    sys.exit(1)

# Target size for smallest dimension
TARGET_SIZE = 509

def calculate_target_dimensions(curr_h, curr_w, target_size=509):
    """
    Calculate new dimensions where smallest side = target_size, preserving aspect ratio.
    
    Examples:
        640×480 → 509×382 (width is smaller, scale to 509)
        640×411 → 509×327 (height is smaller, scale to 509)
        480×640 → 382×509 (width is smaller, scale to 509)
    """
    aspect_ratio = curr_w / curr_h
    
    if curr_w < curr_h:
        # Width is smaller
        new_w = target_size
        new_h = int(target_size / aspect_ratio)
    else:
        # Height is smaller (or equal)
        new_h = target_size
        new_w = int(target_size * aspect_ratio)
    
    return new_h, new_w

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
    (OUTPUT_DIR / "test").mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_files = []
    for ext in ['*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png', '*.iiq']:
        image_files.extend(INPUT_IMAGES_DIR.glob(ext))
    
    logger.info(f"Found {len(image_files)} images to process.")
    
    stats = {
        "total": 0,
        "train": 0,
        "validation": 0,
        "test": 0,
        "skipped": 0,
        "total_tiles": 0
    }

    for img_path in tqdm(image_files, desc="Processing images"):
        # Look for label in label dir with same name
        label_path = INPUT_LABELS_DIR / f"{img_path.stem}.tif"
        
        if not label_path.exists():
            logger.warning(f"Skipping {img_path.name}, label not found")
            stats["skipped"] += 1
            continue
        
        try:
            with rasterio.open(img_path) as src_img, rasterio.open(label_path) as src_lbl:
                # Get current dimensions
                curr_h, curr_w = src_img.shape
                
                # Calculate target dimensions (smallest side = 509)
                new_h, new_w = calculate_target_dimensions(curr_h, curr_w, TARGET_SIZE)
                
                logger.info(f"{img_path.name}: {curr_w}×{curr_h} → {new_w}×{new_h}")
                
                # Read all bands from image
                count = src_img.count
                if count < 3:
                    logger.warning(f"Skipping {img_path.name}: Only {count} bands")
                    stats["skipped"] += 1
                    continue
                
                # Upsample image (use first 3 bands as RGB)
                data_img_rgb = src_img.read(
                    [1, 2, 3],
                    out_shape=(3, new_h, new_w),
                    resampling=Resampling.bilinear
                )
                
                # Upsample mask (use nearest neighbor to preserve class values)
                data_lbl = src_lbl.read(
                    out_shape=(src_lbl.count, new_h, new_w),
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
                if nir_synthetic.shape != (new_h, new_w):
                    nir_synthetic = nir_synthetic.astype(np.float32)
                    nir_synthetic = cv2.resize(nir_synthetic, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

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

                # Tile into 509×509 patches
                # Calculate number of tiles
                n_cols = math.ceil(new_w / TARGET_SIZE)
                n_rows = math.ceil(new_h / TARGET_SIZE)
                
                tile_count = 0
                for row in range(n_rows):
                    for col in range(n_cols):
                        # Define window
                        x_off = col * TARGET_SIZE
                        y_off = row * TARGET_SIZE
                        
                        # Adjust if tile exceeds boundaries (force overlap for last tiles)
                        if x_off + TARGET_SIZE > new_w:
                            x_off = max(0, new_w - TARGET_SIZE)
                        
                        if y_off + TARGET_SIZE > new_h:
                            y_off = max(0, new_h - TARGET_SIZE)
                        
                        # Extract tile from image
                        tile_img = rgn_stack[:, y_off:y_off+TARGET_SIZE, x_off:x_off+TARGET_SIZE]
                        
                        # Extract tile from label
                        tile_lbl = data_lbl[:, y_off:y_off+TARGET_SIZE, x_off:x_off+TARGET_SIZE]
                        
                        # Skip if tile is wrong size (shouldn't happen with above logic)
                        if tile_img.shape[1] != TARGET_SIZE or tile_img.shape[2] != TARGET_SIZE:
                            logger.warning(f"  Skipping tile {row}_{col}: wrong size {tile_img.shape}")
                            continue
                        
                        # Skip if tile is completely empty
                        if tile_img.max() == 0:
                            continue
                        
                        # Randomly assign to train (60%), validation (20%), or test (20%)
                        rand_val = np.random.rand()
                        if rand_val < 0.7:
                            split_dir = "train"
                            stats["train"] += 1
                        elif rand_val < 0.85:
                            split_dir = "validation"
                            stats["validation"] += 1
                        else:
                            split_dir = "test"
                            stats["test"] += 1
                        
                        base_name = f"{img_path.stem}_tile_{row}_{col}"
                        
                        out_img_path = OUTPUT_DIR / split_dir / f"{base_name}_image.tif"
                        out_lbl_path = OUTPUT_DIR / split_dir / f"{base_name}_label.tif"
                        
                        # Identity transform for pixel coordinates
                        dst_transform = rasterio.Affine(1, 0, 0, 0, 1, 0)
                        
                        # Save tile (Image)
                        profile_img = {
                            'height': TARGET_SIZE,
                            'width': TARGET_SIZE,
                            'count': 3,
                            'dtype': 'float32',
                            'driver': 'GTiff',
                            'compress': 'lzw',
                            'transform': dst_transform,
                            'crs': None
                        }
                        
                        with rasterio.open(out_img_path, 'w', **profile_img) as dst:
                            dst.write(tile_img)
                        
                        # Save tile (Label)
                        profile_lbl = {
                            'height': TARGET_SIZE,
                            'width': TARGET_SIZE,
                            'count': 1,
                            'dtype': 'uint8',
                            'compress': 'lzw',
                            'transform': dst_transform,
                            'crs': None
                        }
                        
                        with rasterio.open(out_lbl_path, 'w', **profile_lbl) as dst:
                            dst.write(tile_lbl)
                        
                        tile_count += 1
                
                stats["total"] += 1
                stats["total_tiles"] += tile_count
                logger.info(f"  Generated {tile_count} tiles")
                
        except Exception as e:
            logger.error(f"Failed to process {img_path.name}: {e}")
            import traceback
            traceback.print_exc()
            stats["skipped"] += 1
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("PROCESSING COMPLETE - SUMMARY")
    logger.info("="*80)
    logger.info(f"Images processed: {stats['total']}")
    logger.info(f"Images skipped: {stats['skipped']}")
    logger.info(f"Total tiles generated: {stats['total_tiles']}")
    logger.info(f"  Train: {stats['train']}")
    logger.info(f"  Validation: {stats['validation']}")
    logger.info(f"  Test: {stats['test']}")
    logger.info(f"\nOutput directory: {OUTPUT_DIR}")
    logger.info("="*80)

if __name__ == "__main__":
    preprocess_images()