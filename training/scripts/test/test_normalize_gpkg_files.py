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
        PREPARE_MASKS_GPKG_DIR,
    )
    GPKG_DIR = PREPARE_MASKS_GPKG_DIR
    # GPKG_DIR = Path(r"D:\projects\Image_QC_GUI\2_Repos\OmniCloudMask\training\data\new_vexcel_projects\1466_102024325_Gemeente_Dronten_2025\all_gpkg")
    
except ImportError as e:
    logger.error(f"CRITICAL: local_config.py not found: {e}")
    import sys
    sys.exit(1)

# Possible remark column names (will be normalized to 'remark')
POSSIBLE_REMARK_COLS = ['Remark', 'Remake', 'Remarks', 'Renark', 'remark', 'Rewmark', 'Rermark', 'Remark ']

# Mapping from all messy values to standardized values
VALUE_NORMALIZATION = {
    # Thin/Light Cloud variants
    'Cloud Light': 'thin_cloud',
    'Cloud Lite': 'thin_cloud',
    'Cloud_Lite': 'thin_cloud',
    'Cloud_LIte': 'thin_cloud',
    'Cloude Light': 'thin_cloud',
    'Light Cloud': 'thin_cloud',
    'Light cloud': 'thin_cloud',
    'Lioght cloud': 'thin_cloud',
    'Lihgt Cloud': 'thin_cloud',
    
    # Thick/Deep Cloud variants
    'Cloud Deep': 'thick_cloud',
    'Cloud_Deep': 'thick_cloud',
    'Deep Cloud': 'thick_cloud',
    'Deep cloud': 'thick_cloud',
    'Deep Clear': 'thick_cloud',
    
    # Cloud Shadow variants
    'Shadow': 'cloud_shadow',
    'Shaddow': 'cloud_shadow',
    'Cloud Shadow': 'cloud_shadow',
}

def normalize_gpkg_files():
    """
    Normalize all GPKG files:
    1. Rename remark column to lowercase 'remark'
    2. Standardize all remark values to: thin_cloud, thick_cloud, cloud_shadow
    """
    
    if not GPKG_DIR.exists():
        logger.error(f"Error: GPKG directory not found: {GPKG_DIR}")
        return

    # Find all GPKG files
    gpkg_files = sorted(GPKG_DIR.glob('*.gpkg'))
    logger.info(f"Found {len(gpkg_files)} GPKG files\n")
    
    if len(gpkg_files) == 0:
        logger.warning("No GPKG files found!")
        return
    
    # Statistics
    stats = {
        "total_processed": 0,
        "total_features": 0,
        "features_normalized": 0,
        "by_project": defaultdict(lambda: {"count": 0, "features": 0}),
        "value_changes": defaultdict(int),
        "column_renames": defaultdict(int),
        "no_remark_col": 0,
        "skipped": 0,
        "failed": 0,
    }
    
    for idx, gpkg_path in enumerate(gpkg_files, 1):
        try:
            # if not gpkg_pa/
            # Read GPKG
            gdf = gpd.read_file(gpkg_path)
            
            # Find which remark column exists
            found_col = None
            for col in POSSIBLE_REMARK_COLS:
                if col in gdf.columns:
                    found_col = col
                    break
            
            if found_col is None:
                logger.warning(f"[{idx}/{len(gpkg_files)}] {gpkg_path.name}: ❌ NO REMARK COLUMN - SKIPPING")
                stats["no_remark_col"] += 1
                stats["skipped"] += 1
                continue
            
            stats["total_processed"] += 1
            stats["by_project"][gpkg_path.stem].update({"count": 1, "features": len(gdf)})
            stats["total_features"] += len(gdf)
            
            # Rename column if needed
            if found_col != 'remark':
                gdf = gdf.rename(columns={found_col: 'remark'})
                stats["column_renames"][found_col] += 1
                logger.info(f"[{idx}/{len(gpkg_files)}] {gpkg_path.name}: Column '{found_col}' → 'remark'")
            else:
                logger.info(f"[{idx}/{len(gpkg_files)}] {gpkg_path.name}: Column already named 'remark'")
            
            # Track original values
            features_with_changes = 0
            original_values = gdf['remark'].dropna().unique()
            
            # Normalize values
            def normalize_value(val):
                if not isinstance(val, str):
                    return val
                
                val_stripped = val.strip()
                
                # Check exact match first
                if val_stripped in VALUE_NORMALIZATION:
                    return VALUE_NORMALIZATION[val_stripped]
                
                # If no exact match, try fuzzy matching (shouldn't happen with our data)
                logger.warning(f"    ⚠️ Unknown remark value: '{val_stripped}' - keeping as-is")
                return val_stripped
            
            # Apply normalization
            old_values = gdf['remark'].copy()
            gdf['remark'] = gdf['remark'].apply(normalize_value)
            new_values = gdf['remark']
            
            # Count changes
            features_changed = (old_values != new_values).sum()
            features_with_changes += features_changed
            
            # Track value changes
            for old_val in original_values:
                if isinstance(old_val, str) and old_val.strip() in VALUE_NORMALIZATION:
                    new_val = VALUE_NORMALIZATION[old_val.strip()]
                    count = (old_values == old_val).sum()
                    stats["value_changes"][f"{old_val} → {new_val}"] += count
            
            stats["features_normalized"] += features_changed
            
            # Verify all values are valid
            unique_values = gdf['remark'].unique()
            valid_values = {'thin_cloud', 'thick_cloud', 'cloud_shadow'}
            unknown_values = set(unique_values) - valid_values
            
            if unknown_values:
                logger.warning(f"    ⚠️ WARNING: Unknown values after normalization: {unknown_values}")
            
            # Save normalized GPKG
            gdf.to_file(gpkg_path, driver='GPKG', overwrite=True)
            logger.info(f"    ✅ Saved: {features_changed} features normalized")
        
        except Exception as e:
            logger.error(f"[{idx}/{len(gpkg_files)}] {gpkg_path.name}: ERROR - {e}")
            import traceback
            logger.error(traceback.format_exc())
            stats["failed"] += 1
    
    # Print summary
    logger.info("\n" + "="*100)
    logger.info("GPKG NORMALIZATION COMPLETE")
    logger.info("="*100)
    logger.info(f"\nProcessing Summary:")
    logger.info(f"  Total GPKG files: {len(gpkg_files)}")
    logger.info(f"  Successfully processed: {stats['total_processed']}")
    logger.info(f"  Skipped (no remark column): {stats['no_remark_col']}")
    logger.info(f"  Failed: {stats['failed']}")
    
    logger.info(f"\nFeature Statistics:")
    logger.info(f"  Total features processed: {stats['total_features']}")
    logger.info(f"  Features with normalized values: {stats['features_normalized']}")
    
    logger.info(f"\nColumn Renames:")
    for col, count in sorted(stats['column_renames'].items()):
        logger.info(f"  '{col}' → 'remark': {count} file(s)")
    
    logger.info(f"\nValue Normalizations (top changes):")
    sorted_changes = sorted(stats['value_changes'].items(), key=lambda x: x[1], reverse=True)
    for change, count in sorted_changes[:20]:  # Show top 20
        logger.info(f"  {change}: {count} features")
    
    logger.info("\n" + "="*100)
    logger.info("✅ All GPKG files have been normalized!")
    logger.info("   Column: 'remark'")
    logger.info("   Values: 'thin_cloud', 'thick_cloud', 'cloud_shadow'")
    logger.info("="*100 + "\n")

if __name__ == "__main__":
    normalize_gpkg_files()