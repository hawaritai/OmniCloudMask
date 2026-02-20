import re
from pathlib import Path
from collections import defaultdict
import geopandas as gpd
import warnings
import logging
import sys

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# add training/scripts to path to allow imports of helper modules
# the script is running from training.scripts.test it needs to go up one level to find local_config.py
current_script_dir = Path(__file__).parent.resolve()
training_dir = current_script_dir.parent
if str(training_dir) not in sys.path:
    sys.path.append(str(training_dir))

# --- CONFIGURATION ---
try:
    from local_config import (
        PREPARE_MASKS_IMAGE_DIR,
        PREPARE_MASKS_GPKG_DIR,
    )
    IMAGE_DIR = PREPARE_MASKS_IMAGE_DIR
    GPKG_DIR = PREPARE_MASKS_GPKG_DIR
    
except ImportError as e:
    logger.error(f"CRITICAL: local_config.py not found: {e}")
    import sys
    sys.exit(1)

def debug_gpkg_image_mismatch():
    """
    Find all GPKG-to-Image mismatches and issues.
    """
    
    if not IMAGE_DIR.exists():
        logger.error(f"Error: Image directory not found: {IMAGE_DIR}")
        return
    
    if not GPKG_DIR.exists():
        logger.error(f"Error: GPKG directory not found: {GPKG_DIR}")
        return

    # Find all images
    image_files = []
    for ext in ['*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png', '*.iiq']:
        image_files.extend(IMAGE_DIR.glob(ext))
    image_stems = {img.stem: img for img in image_files}
    
    # Find all GPKG files
    gpkg_files = sorted(GPKG_DIR.glob('*.gpkg'))
    
    logger.info(f"Found {len(image_files)} image files")
    logger.info(f"Found {len(gpkg_files)} GPKG files\n")
    
    # Statistics
    stats = {
        "total_gpkg": len(gpkg_files),
        "gpkg_with_matching_image": 0,
        "gpkg_without_matching_image": [],
        "gpkg_empty": [],
        "gpkg_no_remark": [],
        "gpkg_with_invalid_geoms": [],
        "gpkg_errors": [],
        "images_without_gpkg": 0,
        "images_without_gpkg_list": [],
    }
    
    logger.info("="*100)
    logger.info("SCANNING GPKG FILES")
    logger.info("="*100)
    
    # Check each GPKG
    for idx, gpkg_path in enumerate(gpkg_files, 1):
        stem = gpkg_path.stem
        has_matching_image = stem in image_stems
        
        try:
            gdf = gpd.read_file(gpkg_path)
            num_features = len(gdf)
            
            # Check for remark column
            has_remark = 'remark' in gdf.columns
            
            # Check for valid geometries
            valid_geoms = gdf['geometry'].is_valid.sum()
            invalid_geoms = len(gdf) - valid_geoms
            
            # Get remark value distribution
            remark_counts = {}
            if has_remark:
                remark_counts = gdf['remark'].value_counts().to_dict()
            
            # Status
            if not has_matching_image:
                status = "❌ NO IMAGE"
                stats["gpkg_without_matching_image"].append(stem)
            elif num_features == 0:
                status = "⚠️  EMPTY"
                stats["gpkg_empty"].append(stem)
            elif not has_remark:
                status = "⚠️  NO REMARK"
                stats["gpkg_no_remark"].append(stem)
            elif invalid_geoms > 0:
                status = "⚠️  INVALID GEOMS"
                stats["gpkg_with_invalid_geoms"].append(stem)
            else:
                status = "✅ OK"
                stats["gpkg_with_matching_image"] += 1
            
            logger.info(f"[{idx:3d}/{len(gpkg_files)}] {stem}")
            logger.info(f"         {status} | Features: {num_features} | Remark: {has_remark} | Valid: {valid_geoms}/{num_features}")
            
            if remark_counts:
                remark_str = ", ".join([f"{k}:{v}" for k, v in remark_counts.items()])
                logger.info(f"         Values: {remark_str}")
            
        except Exception as e:
            logger.error(f"[{idx:3d}/{len(gpkg_files)}] {stem}")
            logger.error(f"         ❌ ERROR: {e}")
            stats["gpkg_errors"].append((stem, str(e)))
    
    # Check for images without GPKG
    logger.info("\n" + "="*100)
    logger.info("SCANNING IMAGE FILES")
    logger.info("="*100)
    
    images_without_gpkg_list = []
    for stem in image_stems.keys():
        if stem not in [g.stem for g in gpkg_files]:
            images_without_gpkg_list.append(stem)
    
    stats["images_without_gpkg"] = len(images_without_gpkg_list)
    
    if images_without_gpkg_list:
        logger.info(f"\n⚠️  Found {len(images_without_gpkg_list)} images WITHOUT GPKG files:")
        for idx, stem in enumerate(sorted(images_without_gpkg_list)[:20], 1):  # Show first 20
            logger.info(f"   {idx:3d}. {stem}")
        if len(images_without_gpkg_list) > 20:
            logger.info(f"   ... and {len(images_without_gpkg_list) - 20} more")
    
    # Print detailed summary
    logger.info("\n" + "="*100)
    logger.info("DEBUG SUMMARY")
    logger.info("="*100)
    
    logger.info(f"\nGPKG Statistics:")
    logger.info(f"  Total GPKG files: {stats['total_gpkg']}")
    logger.info(f"  ✅ Valid (image exists, has features, has remark): {stats['gpkg_with_matching_image']}")
    logger.info(f"  ❌ No matching image: {len(stats['gpkg_without_matching_image'])}")
    logger.info(f"  ⚠️  Empty (0 features): {len(stats['gpkg_empty'])}")
    logger.info(f"  ⚠️  Missing 'remark' column: {len(stats['gpkg_no_remark'])}")
    logger.info(f"  ⚠️  Invalid geometries: {len(stats['gpkg_with_invalid_geoms'])}")
    logger.info(f"  ❌ Read errors: {len(stats['gpkg_errors'])}")
    
    total_issues = (len(stats['gpkg_without_matching_image']) + 
                   len(stats['gpkg_empty']) + 
                   len(stats['gpkg_no_remark']) + 
                   len(stats['gpkg_with_invalid_geoms']) +
                   len(stats['gpkg_errors']))
    
    logger.info(f"\n  📊 Expected masks: {stats['gpkg_with_matching_image']}")
    logger.info(f"  📊 Missing masks: {total_issues}")
    
    logger.info(f"\nImage Statistics:")
    logger.info(f"  Total image files: {len(image_files)}")
    logger.info(f"  Images with GPKG: {len(image_files) - stats['images_without_gpkg']}")
    logger.info(f"  Images without GPKG: {stats['images_without_gpkg']}")
    
    # Show GPKG files without images
    if stats['gpkg_without_matching_image']:
        logger.info(f"\n{'='*100}")
        logger.info(f"GPKG FILES WITHOUT MATCHING IMAGES ({len(stats['gpkg_without_matching_image'])})")
        logger.info(f"{'='*100}")
        for idx, stem in enumerate(sorted(stats['gpkg_without_matching_image'])[:30], 1):
            logger.info(f"  {idx:3d}. {stem}")
        if len(stats['gpkg_without_matching_image']) > 30:
            logger.info(f"  ... and {len(stats['gpkg_without_matching_image']) - 30} more")
    
    # Show empty GPKG files
    if stats['gpkg_empty']:
        logger.info(f"\n{'='*100}")
        logger.info(f"EMPTY GPKG FILES ({len(stats['gpkg_empty'])})")
        logger.info(f"{'='*100}")
        for idx, stem in enumerate(sorted(stats['gpkg_empty'])[:20], 1):
            logger.info(f"  {idx:3d}. {stem}")
        if len(stats['gpkg_empty']) > 20:
            logger.info(f"  ... and {len(stats['gpkg_empty']) - 20} more")
    
    # Show GPKG without remark column
    if stats['gpkg_no_remark']:
        logger.info(f"\n{'='*100}")
        logger.info(f"GPKG FILES WITHOUT 'remark' COLUMN ({len(stats['gpkg_no_remark'])})")
        logger.info(f"{'='*100}")
        logger.info(f"⚠️  These need to be re-normalized (run normalize_gpkg_files.py again)")
        for idx, stem in enumerate(sorted(stats['gpkg_no_remark'])[:20], 1):
            logger.info(f"  {idx:3d}. {stem}")
        if len(stats['gpkg_no_remark']) > 20:
            logger.info(f"  ... and {len(stats['gpkg_no_remark']) - 20} more")
    
    # Show GPKG with invalid geometries
    if stats['gpkg_with_invalid_geoms']:
        logger.info(f"\n{'='*100}")
        logger.info(f"GPKG FILES WITH INVALID GEOMETRIES ({len(stats['gpkg_with_invalid_geoms'])})")
        logger.info(f"{'='*100}")
        for idx, stem in enumerate(sorted(stats['gpkg_with_invalid_geoms'])[:20], 1):
            logger.info(f"  {idx:3d}. {stem}")
        if len(stats['gpkg_with_invalid_geoms']) > 20:
            logger.info(f"  ... and {len(stats['gpkg_with_invalid_geoms']) - 20} more")
    
    # Show read errors
    if stats['gpkg_errors']:
        logger.info(f"\n{'='*100}")
        logger.info(f"GPKG READ ERRORS ({len(stats['gpkg_errors'])})")
        logger.info(f"{'='*100}")
        for idx, (stem, error) in enumerate(sorted(stats['gpkg_errors'])[:20], 1):
            logger.info(f"  {idx:3d}. {stem}")
            logger.info(f"       Error: {error[:80]}")
        if len(stats['gpkg_errors']) > 20:
            logger.info(f"  ... and {len(stats['gpkg_errors']) - 20} more")
    
    # Recommendations
    logger.info(f"\n{'='*100}")
    logger.info("RECOMMENDATIONS")
    logger.info(f"{'='*100}")
    
    if stats['gpkg_without_matching_image']:
        logger.info(f"\n1. {len(stats['gpkg_without_matching_image'])} GPKG files have no matching image file.")
        logger.info(f"   → Check if image files were deleted or have different naming")
        logger.info(f"   → Or these GPKG files are extras and can be archived")
    
    if stats['gpkg_no_remark']:
        logger.info(f"\n2. {len(stats['gpkg_no_remark'])} GPKG files don't have 'remark' column.")
        logger.info(f"   → Run normalize_gpkg_files.py again to fix these")
    
    if stats['gpkg_errors']:
        logger.info(f"\n3. {len(stats['gpkg_errors'])} GPKG files have read errors.")
        logger.info(f"   → These files may be corrupted")
        logger.info(f"   → Try to re-create them or restore from backup")
    
    logger.info(f"\n4. Expected successful masks: {stats['gpkg_with_matching_image']}")
    logger.info(f"   Actual successful masks: 356")
    if stats['gpkg_with_matching_image'] > 356:
        logger.info(f"   → {stats['gpkg_with_matching_image'] - 356} additional masks should be generated")
    
    logger.info(f"\n{'='*100}\n")

if __name__ == "__main__":
    debug_gpkg_image_mismatch()