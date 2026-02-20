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
import json
from typing import Dict

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
        PREPROCESS_OUTPUT_DIR,
        TARGET_SIZE,
        GT_LIST_PATH,
        FN_LIST_PATH,
        HARD_NEGATIVES_LIST_PATH,
        WEIGHT_GT,
        WEIGHT_FN,
        WEIGHT_HARD_NEGATIVE,
        USE_METHOD_B_PREPROCESSING,
        GENERATE_SYNTHETIC_NIR,
        # Dynamic weighting configuration
        USE_DYNAMIC_TILE_WEIGHTS,
        DYNAMIC_WEIGHT_MODE,
    )

    INPUT_IMAGES_DIR = PREPROCESS_INPUT_IMAGES_DIR
    INPUT_LABELS_DIR = PREPROCESS_INPUT_LABELS_DIR
    OUTPUT_DIR = PREPROCESS_OUTPUT_DIR
    
except ImportError as e:
    logger.critical(f"CRITICAL: local_config.py not found or missing required config: {e}")
    logger.critical("Required variables: PREPROCESS_INPUT_IMAGES_DIR, PREPROCESS_INPUT_LABELS_DIR, PREPROCESS_OUTPUT_DIR")
    sys.exit(1)

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


def calculate_sliding_window_positions(image_size, tile_size):
    """
    Calculate tile positions for sliding window with minimal overlap.
    
    Args:
        image_size: Size of the image dimension (width or height)
        tile_size: Size of the tile (509)
        
    Returns:
        List of starting positions for tiles
    
    Example:
        image_size=1323, tile_size=509
        Returns: [0, 509, 814]
        - Tile 1: 0-508 (509 pixels)
        - Tile 2: 509-1017 (509 pixels)
        - Tile 3: 814-1322 (509 pixels, starting from right edge)
    """
    if image_size <= tile_size:
        return [0]
    
    positions = []
    num_tiles = math.ceil(image_size / tile_size)
    
    # Generate positions with no overlap initially
    for i in range(num_tiles - 1):
        positions.append(i * tile_size)
    
    # Last tile starts from the right edge
    last_position = image_size - tile_size
    positions.append(last_position)
    
    return positions


def calculate_dynamic_tile_weight(tile_label: np.ndarray, base_weight: float = 1.0) -> float:
    """
    Calculate dynamic weight for a tile based on its cloud/shadow content.
    
    This function computes a weight that increases with the proportion of
    cloud and shadow pixels in the tile. This helps counteract the severe
    class imbalance (95% clear pixels) by giving more importance to tiles
    that contain cloud/shadow pixels.
    
    Weight formula:
        - 0% cloud/shadow: weight = base_weight * 0.5 (clear tiles get lower weight)
        - 10% cloud/shadow: weight = base_weight * 2.0
        - 50% cloud/shadow: weight = base_weight * 4.0
        - 100% cloud/shadow: weight = base_weight * 5.0
    
    Args:
        tile_label: 2D numpy array with class labels (0=clear, 1=thick, 2=thin, 3=shadow)
        base_weight: Base weight multiplier (default 1.0)
        
    Returns:
        Calculated weight for the tile
    """
    # Classes: 0=clear, 1=thick cloud, 2=thin cloud, 3=shadow
    cloud_shadow_pixels = np.sum((tile_label == 1) | (tile_label == 2) | (tile_label == 3))
    total_pixels = tile_label.size
    
    if total_pixels == 0:
        return base_weight
    
    cloud_shadow_ratio = cloud_shadow_pixels / total_pixels
    
    # Weight formula: 0.5 + ratio * 4.5
    # This gives: 0% -> 0.5, 10% -> 0.95, 50% -> 2.75, 100% -> 5.0
    # Multiplied by base_weight for scaling
    dynamic_weight = base_weight * (0.5 + cloud_shadow_ratio * 4.5)
    
    return dynamic_weight


def calculate_class_aware_weight(tile_label: np.ndarray, base_weight: float = 1.0) -> float:
    """
    Calculate weight based on presence of specific classes (more aggressive).
    
    This function gives higher weights to tiles containing shadow and thin cloud
    pixels, which are the hardest classes to detect.
    
    Weight multipliers:
        - Shadow present: 4.0x (hardest class)
        - Thin cloud present: 3.0x
        - Thick cloud present: 2.0x
        - Clear only: 0.5x (hard negative, lower weight)
    
    Args:
        tile_label: 2D numpy array with class labels
        base_weight: Base weight multiplier
        
    Returns:
        Calculated weight for the tile
    """
    has_shadow = np.any(tile_label == 3)
    has_thin_cloud = np.any(tile_label == 2)
    has_thick_cloud = np.any(tile_label == 1)
    
    # Assign weight based on hardest class present
    if has_shadow:
        multiplier = 4.0
    elif has_thin_cloud:
        multiplier = 3.0
    elif has_thick_cloud:
        multiplier = 2.0
    else:
        multiplier = 0.5  # Clear image (hard negative)
    
    return base_weight * multiplier


def load_image_weights_from_lists() -> Dict[str, float]:
    """
    Load image weights from GT, FN, and Hard Negatives list files.
    
    Supports all image formats (.tif, .png, .jpg, .iiq, etc.) as long as
    filenames in the list files match the source image stems (without extension).
    
    Weight Assignment Logic (applied during tiling):
    - GT/FN images: Only tiles containing cloud/shadow pixels get the GT/FN weight.
                    Clear tiles from these images get weight 1.0.
    - Hard Negative images: All tiles get the Hard Negative weight (these are
                           challenging clear images where all tiles are relevant).
    - Other images: Default weight 1.0.
    
    This ensures that weights are applied based on actual tile content, not just
    the source image marking. For example, a tile with no clouds from an FN image
    won't get the FN weight since it doesn't contain the false negative content.
    
    Returns:
        Dictionary mapping image stem (without extension) to weight value.
    """
    image_weights = {}
    
    # Load GT images (weight = WEIGHT_GT)
    if GT_LIST_PATH.exists():
        with open(GT_LIST_PATH, 'r') as f:
            for line in f:
                img_stem = line.strip()
                if img_stem:
                    image_weights[img_stem] = WEIGHT_GT
        gt_count = len([k for k, v in image_weights.items() if v == WEIGHT_GT])
        logger.info(f"Loaded {gt_count} GT images (weight={WEIGHT_GT}) from {GT_LIST_PATH}")
    else:
        logger.warning(f"GT list not found: {GT_LIST_PATH}")
    
    # Load FN images (weight = WEIGHT_FN)
    if FN_LIST_PATH.exists():
        with open(FN_LIST_PATH, 'r') as f:
            for line in f:
                img_stem = line.strip()
                if img_stem:
                    image_weights[img_stem] = WEIGHT_FN
        fn_count = len([k for k, v in image_weights.items() if v == WEIGHT_FN])
        logger.info(f"Loaded {fn_count} FN images (weight={WEIGHT_FN}) from {FN_LIST_PATH}")
    else:
        logger.warning(f"FN list not found: {FN_LIST_PATH}")
    
    # Load Hard Negative images (weight = WEIGHT_HARD_NEGATIVE)
    if HARD_NEGATIVES_LIST_PATH.exists():
        with open(HARD_NEGATIVES_LIST_PATH, 'r') as f:
            for line in f:
                img_stem = line.strip()
                if img_stem:
                    image_weights[img_stem] = WEIGHT_HARD_NEGATIVE
        hn_count = len([k for k, v in image_weights.items() if v == WEIGHT_HARD_NEGATIVE])
        logger.info(f"Loaded {hn_count} Hard Negative images (weight={WEIGHT_HARD_NEGATIVE}) from {HARD_NEGATIVES_LIST_PATH}")
    else:
        logger.warning(f"Hard Negatives list not found: {HARD_NEGATIVES_LIST_PATH}")
    
    logger.info(f"Total images with weights: {len(image_weights)}")
    return image_weights

def assign_image_splits(image_files, image_weights):
    """
    Assign each image to train/val/test split with defect-aware prioritization.
    
    Strategy:
    1. GT/FN images → 80% to train (need defect examples for training)
    2. Hard Negatives → 70/15/15 split (balanced distribution)
    3. Regular images → 70/15/15 split
    
    This ensures training set gets majority of valuable defect-containing images.
    
    Returns:
        Dictionary mapping image stem to split name ('train', 'validation', 'test')
    """
    image_splits = {}
    
    # Categorize images
    gt_fn_images = []
    hard_neg_images = []
    regular_images = []
    
    for img_path in image_files:
        # Check label exists
        label_path = INPUT_LABELS_DIR / f"{img_path.stem}.png"
        if not label_path.exists():
            continue
            
        weight = image_weights.get(img_path.stem, 1.0)
        
        if weight == WEIGHT_GT or weight == WEIGHT_FN:
            gt_fn_images.append(img_path)
        elif weight == WEIGHT_HARD_NEGATIVE:
            hard_neg_images.append(img_path)
        else:
            regular_images.append(img_path)
    
    logger.info(f"Image categorization:")
    logger.info(f"  GT/FN images: {len(gt_fn_images)}")
    logger.info(f"  Hard Negative images: {len(hard_neg_images)}")
    logger.info(f"  Regular images: {len(regular_images)}")
    
    # Shuffle each category for randomness
    np.random.shuffle(gt_fn_images)
    np.random.shuffle(hard_neg_images)
    np.random.shuffle(regular_images)
    
    # Assign GT/FN images: 80% train, 10% val, 10% test
    # These contain defects - we need most in training
    for idx, img_path in enumerate(gt_fn_images):
        if idx < len(gt_fn_images) * 0.7:
            image_splits[img_path.stem] = "train"
        elif idx < len(gt_fn_images) * 0.85:
            image_splits[img_path.stem] = "validation"
        else:
            image_splits[img_path.stem] = "test"
    
    # Assign Hard Negatives: 70% train, 15% val, 15% test
    for idx, img_path in enumerate(hard_neg_images):
        if idx < len(hard_neg_images) * 0.7:
            image_splits[img_path.stem] = "train"
        elif idx < len(hard_neg_images) * 0.85:
            image_splits[img_path.stem] = "validation"
        else:
            image_splits[img_path.stem] = "test"
    
    # Assign Regular images: 70% train, 15% val, 15% test
    for idx, img_path in enumerate(regular_images):
        if idx < len(regular_images) * 0.7:
            image_splits[img_path.stem] = "train"
        elif idx < len(regular_images) * 0.85:
            image_splits[img_path.stem] = "validation"
        else:
            image_splits[img_path.stem] = "test"
    
    # Log split statistics
    split_counts = {"train": 0, "validation": 0, "test": 0}
    for split in image_splits.values():
        split_counts[split] += 1
    
    logger.info(f"Image split assignment:")
    logger.info(f"  Train: {split_counts['train']} images")
    logger.info(f"  Validation: {split_counts['validation']} images")
    logger.info(f"  Test: {split_counts['test']} images")
    
    return image_splits


def preprocess_images():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Preprocessing Method: {'Clamp & Scale (Method B)' if USE_METHOD_B_PREPROCESSING else 'RGBNIRHandler (Method A)'}")
    logger.info(f"Dynamic Tile Weighting: {'Enabled' if USE_DYNAMIC_TILE_WEIGHTS else 'Disabled'}")
    if USE_DYNAMIC_TILE_WEIGHTS:
        logger.info(f"  Weight Mode: {DYNAMIC_WEIGHT_MODE}")

    # --- LOAD IMAGE WEIGHTS ---
    image_weights = load_image_weights_from_lists()
    
    # Find all images
    image_files = []
    for ext in ['*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png', '*.iiq']:
        image_files.extend(INPUT_IMAGES_DIR.glob(ext))
    
    logger.info(f"Found {len(image_files)} images to process.")
    
    # --- ASSIGN IMAGE-LEVEL SPLITS ---
    logger.info("\n" + "="*80)
    logger.info("ASSIGNING IMAGE-LEVEL SPLITS (defect-aware)")
    logger.info("="*80)
    image_splits = assign_image_splits(image_files, image_weights)
    
    # Track weights per split for saving JSON files
    split_weights = {
        "train": {},
        "validation": {},
        "test": {}
    }
    
    # Track weight statistics for summary
    weight_stats = {
        "train": {"total": 0.0, "count": 0, "min": float('inf'), "max": 0.0},
        "validation": {"total": 0.0, "count": 0, "min": float('inf'), "max": 0.0},
        "test": {"total": 0.0, "count": 0, "min": float('inf'), "max": 0.0}
    }
    
    # Track defect tile statistics
    defect_tile_stats = {
        "train": {"with_defects": 0, "clear": 0},
        "validation": {"with_defects": 0, "clear": 0},
        "test": {"with_defects": 0, "clear": 0}
    }
    
    # Initialize Handler (only needed for Method A)
    handler = RGBNIRHandler(device=device) if not USE_METHOD_B_PREPROCESSING else None

    # Create output directories
    (OUTPUT_DIR / "train").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "validation").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "test").mkdir(parents=True, exist_ok=True)
    
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
        label_path = INPUT_LABELS_DIR / f"{img_path.stem}.png"
        
        if not label_path.exists():
            logger.warning(f"Skipping {img_path.name}, label not found")
            stats["skipped"] += 1
            continue
        
        # Get pre-assigned split for this image
        split_dir = image_splits.get(img_path.stem)
        if split_dir is None:
            logger.warning(f"Skipping {img_path.name}, no split assigned")
            stats["skipped"] += 1
            continue
        
        try:
            with rasterio.open(img_path) as src_img, rasterio.open(label_path) as src_lbl:
                # Get current dimensions
                curr_h, curr_w = src_img.shape
                min_dim = min(curr_h, curr_w)
                
                # Read all bands from image
                count = src_img.count
                if count < 3:
                    logger.warning(f"Skipping {img_path.name}: Only {count} bands")
                    stats["skipped"] += 1
                    continue
                
                # Decide whether to upsample or use original resolution
                if min_dim < TARGET_SIZE:
                    # CASE 1: Image is smaller than target - upsample to make smallest side = 509
                    new_h, new_w = calculate_target_dimensions(curr_h, curr_w, TARGET_SIZE)
                    logger.info(f"{img_path.name}: {curr_w}×{curr_h} → {new_w}×{new_h} (upsampled)")
                    
                    # Upsample image (use first 3 bands as RGB)
                    data_img_rgb = src_img.read(
                        [1, 2, 3],
                        out_shape=(3, new_h, new_w),
                        resampling=Resampling.cubic
                    )
                    
                    # Upsample mask (use nearest neighbor to preserve class values)
                    data_lbl = src_lbl.read(
                        out_shape=(src_lbl.count, new_h, new_w),
                        resampling=Resampling.nearest
                    )
                else:
                    # CASE 2: Image is >= 509 - use original resolution
                    new_h, new_w = curr_h, curr_w
                    logger.info(f"{img_path.name}: {curr_w}×{curr_h} (original, sliding window)")
                    
                    # Read at original resolution
                    data_img_rgb = src_img.read([1, 2, 3])
                    data_lbl = src_lbl.read()
                
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

                if GENERATE_SYNTHETIC_NIR:
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
                    blue = data_img_rgb[2].astype(np.float32)
                    
                    if GENERATE_SYNTHETIC_NIR:
                        rgn_stack = handler.stack_rgb_nir(
                            red=red,
                            green=green,
                            nir=nir_synthetic,
                            # normalize=True,
                            # scale_to_dn=True,
                            # dn_range=(0, 10000)
                        )
                    else:
                        # rgn_stack = np.stack([((red / 255.0) * 10000.0), ((green / 255.0) * 10000.0),  ((blue / 255.0) * 10000.0)], axis=0)
                        rgn_stack = np.stack([red, green, blue], axis=0)

                # Calculate tile positions using sliding window with minimal overlap
                x_positions = calculate_sliding_window_positions(new_w, TARGET_SIZE)
                y_positions = calculate_sliding_window_positions(new_h, TARGET_SIZE)
                
                logger.info(f"  Generating tiles: {len(y_positions)} rows × {len(x_positions)} cols")
                
                tile_count = 0
                for row_idx, y_off in enumerate(y_positions):
                    for col_idx, x_off in enumerate(x_positions):
                        # Extract tile from image
                        tile_img = rgn_stack[:, y_off:y_off+TARGET_SIZE, x_off:x_off+TARGET_SIZE]
                        
                        # Extract tile from label
                        tile_lbl = data_lbl[:, y_off:y_off+TARGET_SIZE, x_off:x_off+TARGET_SIZE]
                        
                        # Verify tile size (should always be correct with our calculation)
                        if tile_img.shape[1] != TARGET_SIZE or tile_img.shape[2] != TARGET_SIZE:
                            logger.warning(f"  Skipping tile {row_idx}_{col_idx}: wrong size {tile_img.shape}")
                            continue
                        
                        # Skip if tile is completely empty
                        if tile_img.max() == 0:
                            continue
                        
                        # Use pre-assigned split for this image (all tiles from same image go to same split)
                        # This prevents data leakage and ensures GT/FN images are properly distributed
                        stats[split_dir] += 1
                        
                        base_name = f"{img_path.stem}_tile_{row_idx}_{col_idx}"
                        
                        out_img_path = OUTPUT_DIR / split_dir / f"{base_name}_image.tif"
                        out_lbl_path = OUTPUT_DIR / split_dir / f"{base_name}_label.tif"
                        
                        # Extract 2D label for content checking
                        tile_label_2d = tile_lbl[0]  # Extract 2D label from (1, H, W)
                        
                        # Check if tile contains cloud/shadow pixels
                        has_cloud_shadow = np.any((tile_label_2d == 1) | (tile_label_2d == 2) | (tile_label_2d == 3))
                        
                        # Track defect statistics
                        if has_cloud_shadow:
                            defect_tile_stats[split_dir]["with_defects"] += 1
                        else:
                            defect_tile_stats[split_dir]["clear"] += 1
                        
                        # Get base image weight from source image
                        source_img_weight = image_weights.get(img_path.stem, 1.0)
                        
                        # Determine base weight based on tile content and source image marking
                        if source_img_weight == WEIGHT_HARD_NEGATIVE:
                            # Hard Negative images: all tiles get HN weight (challenging clear images)
                            base_img_weight = WEIGHT_HARD_NEGATIVE
                        elif has_cloud_shadow:
                            # Tile has cloud/shadow: use source image weight (GT or FN)
                            base_img_weight = source_img_weight
                        else:
                            # Tile is clear but from GT/FN image: treat as regular tile
                            # Don't apply GT/FN weight since this tile doesn't have the defects
                            base_img_weight = 1.0
                        
                        # Apply dynamic weighting if enabled
                        if USE_DYNAMIC_TILE_WEIGHTS:
                            if DYNAMIC_WEIGHT_MODE == 'content_ratio':
                                # Weight based on cloud/shadow pixel ratio
                                img_weight = calculate_dynamic_tile_weight(tile_label_2d, base_img_weight)
                            elif DYNAMIC_WEIGHT_MODE == 'class_aware':
                                # Weight based on presence of specific classes
                                img_weight = calculate_class_aware_weight(tile_label_2d, base_img_weight)
                            else:
                                # Fallback to base weight
                                img_weight = base_img_weight
                        else:
                            img_weight = base_img_weight
                        
                        # Store weight for this tile
                        tile_weight_key = f"{base_name}_image.tif"
                        split_weights[split_dir][tile_weight_key] = img_weight
                        
                        # Update weight statistics
                        weight_stats[split_dir]["total"] += img_weight
                        weight_stats[split_dir]["count"] += 1
                        weight_stats[split_dir]["min"] = min(weight_stats[split_dir]["min"], img_weight)
                        weight_stats[split_dir]["max"] = max(weight_stats[split_dir]["max"], img_weight)
                        
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
    
    # --- SAVE IMAGE WEIGHTS TO JSON FILES ---
    for split_dir, weights_dict in split_weights.items():
        weights_path = OUTPUT_DIR / split_dir / "image_weights.json"
        with open(weights_path, 'w') as f:
            json.dump(weights_dict, f, indent=2)
        logger.info(f"Saved {len(weights_dict)} tile weights to {weights_path}")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("PROCESSING COMPLETE - SUMMARY")
    logger.info("="*80)
    logger.info(f"Images processed: {stats['total']}")
    logger.info(f"Images skipped: {stats['skipped']}")
    logger.info(f"Total tiles generated: {stats['total_tiles']}")
    logger.info(f"\n--- Tile Distribution ---")
    logger.info(f"  Train: {stats['train']} tiles")
    logger.info(f"  Validation: {stats['validation']} tiles")
    logger.info(f"  Test: {stats['test']} tiles")
    
    # Print defect tile statistics
    logger.info(f"\n--- Defect Tile Distribution ---")
    for split_name in ["train", "validation", "test"]:
        dts = defect_tile_stats[split_name]
        total = dts["with_defects"] + dts["clear"]
        if total > 0:
            defect_pct = (dts["with_defects"] / total) * 100
            logger.info(f"  {split_name}:")
            logger.info(f"    With defects: {dts['with_defects']} ({defect_pct:.1f}%)")
            logger.info(f"    Clear: {dts['clear']} ({100-defect_pct:.1f}%)")
    
    # Print weight statistics
    if USE_DYNAMIC_TILE_WEIGHTS:
        logger.info("\n--- Dynamic Weight Statistics ---")
        for split_name in ["train", "validation", "test"]:
            ws = weight_stats[split_name]
            if ws["count"] > 0:
                avg_weight = ws["total"] / ws["count"]
                logger.info(f"  {split_name}: avg={avg_weight:.2f}, min={ws['min']:.2f}, max={ws['max']:.2f}")
    
    logger.info(f"\nOutput directory: {OUTPUT_DIR}")
    logger.info("="*80)

if __name__ == "__main__":
    preprocess_images()