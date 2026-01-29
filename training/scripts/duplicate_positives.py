import shutil
from pathlib import Path
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
try:
    from local_config import (
        PREPARE_MASKS_IMAGE_DIR,
        PREPARE_MASKS_GPKG_DIR,
        PREPARE_MASKS_OUTPUT_DIR,
        DUPLICATED_IMAGE_DIR,
        DUPLICATED_MASK_DIR
    )
    IMAGE_DIR = PREPARE_MASKS_IMAGE_DIR
    GPKG_DIR = PREPARE_MASKS_GPKG_DIR
    MASK_DIR = PREPARE_MASKS_OUTPUT_DIR
    OUTPUT_IMAGE_DIR = DUPLICATED_IMAGE_DIR
    OUTPUT_MASK_DIR = DUPLICATED_MASK_DIR
except ImportError:
    logger.error("CRITICAL: local_config.py not found or missing required variables.")
    logger.error("Please add to local_config.py:")
    logger.error("  DUPLICATED_IMAGE_DIR = Path('path/to/duplicated/images')")
    logger.error("  DUPLICATED_MASK_DIR = Path('path/to/duplicated/masks')")
    import sys
    sys.exit(1)

# Duplication factor (2 = create 2 additional copies, total 3 copies)
DUPLICATION_FACTOR = 2

def duplicate_positives():
    """
    Duplicate positive samples (those with .gpkg annotations) to balance the dataset.
    
    Workflow:
    1. Scan GPKG_DIR to identify which images are positives
    2. For positives: Create DUPLICATION_FACTOR copies (e.g., 2 copies = 3 total)
    3. For negatives: Copy once (as-is)
    4. Output to DUPLICATED_IMAGE_DIR and DUPLICATED_MASK_DIR
    """
    
    # Create output directories
    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_files = list(IMAGE_DIR.glob("*.tif"))
    logger.info(f"Found {len(image_files)} total images in {IMAGE_DIR}")
    
    # Identify positives by checking for corresponding .gpkg files
    positives = []
    negatives = []
    
    for img_path in image_files:
        gpkg_path = GPKG_DIR / f"{img_path.stem}.gpkg"
        if gpkg_path.exists():
            positives.append(img_path)
        else:
            negatives.append(img_path)
    
    logger.info(f"Identified {len(positives)} positives (have .gpkg annotations)")
    logger.info(f"Identified {len(negatives)} negatives (no annotations)")
    logger.info(f"Duplication factor: {DUPLICATION_FACTOR} (total {DUPLICATION_FACTOR + 1} copies per positive)")
    
    # Calculate final counts
    final_positive_count = len(positives) * (DUPLICATION_FACTOR + 1)
    final_negative_count = len(negatives)
    total_final = final_positive_count + final_negative_count
    ratio = final_negative_count / final_positive_count if final_positive_count > 0 else 0
    
    logger.info(f"\nFinal dataset composition:")
    logger.info(f"  Positives: {len(positives)} × {DUPLICATION_FACTOR + 1} = {final_positive_count}")
    logger.info(f"  Negatives: {len(negatives)} × 1 = {final_negative_count}")
    logger.info(f"  Total: {total_final}")
    logger.info(f"  Ratio (pos:neg): 1:{ratio:.2f}")
    
    # Process positives - duplicate them
    logger.info(f"\nDuplicating {len(positives)} positive samples...")
    for img_path in tqdm(positives, desc="Duplicating positives"):
        mask_path = MASK_DIR / f"{img_path.stem}.tif"
        
        if not mask_path.exists():
            logger.warning(f"Warning: Mask not found for {img_path.name}, skipping.")
            continue
        
        # Copy original + create duplicates
        for dup_idx in range(DUPLICATION_FACTOR + 1):
            if dup_idx == 0:
                # Original (no suffix)
                out_img_name = img_path.name
                out_mask_name = mask_path.name
            else:
                # Duplicate (with suffix _dup1, _dup2, etc.)
                out_img_name = f"{img_path.stem}_dup{dup_idx}{img_path.suffix}"
                out_mask_name = f"{mask_path.stem}_dup{dup_idx}{mask_path.suffix}"
            
            out_img_path = OUTPUT_IMAGE_DIR / out_img_name
            out_mask_path = OUTPUT_MASK_DIR / out_mask_name
            
            # Copy files
            shutil.copy2(img_path, out_img_path)
            shutil.copy2(mask_path, out_mask_path)
    
    # Process negatives - copy as-is
    logger.info(f"\nCopying {len(negatives)} negative samples...")
    for img_path in tqdm(negatives, desc="Copying negatives"):
        mask_path = MASK_DIR / f"{img_path.stem}.tif"
        
        if not mask_path.exists():
            logger.warning(f"Warning: Mask not found for {img_path.name}, skipping.")
            continue
        
        out_img_path = OUTPUT_IMAGE_DIR / img_path.name
        out_mask_path = OUTPUT_MASK_DIR / mask_path.name
        
        # Copy files
        shutil.copy2(img_path, out_img_path)
        shutil.copy2(mask_path, out_mask_path)
    
    # Verify output
    final_images = list(OUTPUT_IMAGE_DIR.glob("*.tif"))
    final_masks = list(OUTPUT_MASK_DIR.glob("*.tif"))
    
    logger.info(f"\n✅ Duplication complete!")
    logger.info(f"Output directories:")
    logger.info(f"  Images: {OUTPUT_IMAGE_DIR} ({len(final_images)} files)")
    logger.info(f"  Masks:  {OUTPUT_MASK_DIR} ({len(final_masks)} files)")
    
    if len(final_images) != len(final_masks):
        logger.warning(f"⚠️ Warning: Mismatch between images ({len(final_images)}) and masks ({len(final_masks)})")
    
    if len(final_images) != total_final:
        logger.warning(f"⚠️ Warning: Expected {total_final} images, but got {len(final_images)}")

if __name__ == "__main__":
    duplicate_positives()