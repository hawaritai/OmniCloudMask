import re
import os
import sys
from pathlib import Path
from collections import defaultdict
import geopandas as gpd
import warnings
import logging

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

# Possible remark column names
POSSIBLE_REMARK_COLS = ['Remark', 'Remake', 'Remarks', 'Renark', 'remark', 'Rewmark']

def extract_gpkg_metadata():
    """
    Extract all unique remark values from all GPKG files.
    Groups results by actual column name found and shows frequency.
    """
    
    if not GPKG_DIR.exists():
        logger.error(f"Error: GPKG directory not found: {GPKG_DIR}")
        return

    # Find all GPKG files
    gpkg_files = list(GPKG_DIR.glob('*.gpkg'))
    logger.info(f"Found {len(gpkg_files)} GPKG files\n")
    
    if len(gpkg_files) == 0:
        logger.warning("No GPKG files found!")
        return
    
    # Dictionary to store results: {column_name: {value: count}}
    all_remarks = defaultdict(lambda: defaultdict(int))
    
    # Also track which GPKG files have which columns
    gpkg_column_map = defaultdict(set)
    
    # Track GPKGs with no remark columns
    gpkgs_no_remark = []
    
    for idx, gpkg_path in enumerate(gpkg_files, 1):
        try:
            gdf = gpd.read_file(gpkg_path)
            
            # Find which remark columns exist in this GPKG
            found_cols = [c for c in POSSIBLE_REMARK_COLS if c in gdf.columns]
            
            if not found_cols:
                gpkgs_no_remark.append(gpkg_path.name)
                logger.warning(f"[{idx}/{len(gpkg_files)}] {gpkg_path.name}: ❌ NO REMARK COLUMN FOUND")
                continue
            
            logger.info(f"[{idx}/{len(gpkg_files)}] {gpkg_path.name}: Found columns {found_cols}")
            
            # Extract values from each found column
            for col in found_cols:
                gpkg_column_map[col].add(gpkg_path.name)
                
                # Get unique values and their counts
                values = gdf[col].dropna()
                for val in values:
                    if isinstance(val, str):
                        val_stripped = val.strip()
                        all_remarks[col][val_stripped] += 1
        
        except Exception as e:
            logger.error(f"[{idx}/{len(gpkg_files)}] {gpkg_path.name}: ERROR - {e}")
    
    # Print summary
    logger.info("\n" + "="*100)
    logger.info("GPKG METADATA EXTRACTION COMPLETE")
    logger.info("="*100)
    
    if gpkgs_no_remark:
        logger.warning(f"\n⚠️  {len(gpkgs_no_remark)} GPKG(s) have NO remark column:")
        for name in gpkgs_no_remark:
            logger.warning(f"   - {name}")
    
    # Print results by column
    logger.info(f"\n📊 UNIQUE REMARK VALUES BY COLUMN:\n")
    
    for col in POSSIBLE_REMARK_COLS:
        if col not in all_remarks or len(all_remarks[col]) == 0:
            continue
        
        values_dict = all_remarks[col]
        num_unique = len(values_dict)
        total_count = sum(values_dict.values())
        num_gpkgs = len(gpkg_column_map[col])
        
        logger.info(f"Column: '{col}'")
        logger.info(f"  Found in: {num_gpkgs} GPKG file(s)")
        logger.info(f"  Unique values: {num_unique}")
        logger.info(f"  Total occurrences: {total_count}")
        logger.info(f"  Values (sorted by frequency):")
        
        # Sort by frequency (descending)
        sorted_values = sorted(values_dict.items(), key=lambda x: x[1], reverse=True)
        
        for val, count in sorted_values:
            pct = (count / total_count) * 100
            logger.info(f"    • '{val}' ({count} occurrences, {pct:.1f}%)")
        
        logger.info("")
    
    # Print recommendations for CLASS_MAPPING
    logger.info("="*100)
    logger.info("📝 RECOMMENDED CLASS MAPPING & SYNONYMS")
    logger.info("="*100)
    logger.info("\nAnalyzing patterns to suggest class groupings...\n")
    
    # Collect all unique values across all columns
    all_values = set()
    for col_dict in all_remarks.values():
        all_values.update(col_dict.keys())
    
    # Group by semantic meaning
    cloud_lite = []
    cloud_deep = []
    cloud_shadow = []
    other = []
    
    for val in sorted(all_values):
        val_lower = val.lower()
        val_normalized = re.sub(r'[_\s-]', '', val_lower)
        
        if 'shadow' in val_lower or 'ombre' in val_lower:
            cloud_shadow.append(val)
        elif 'lite' in val_normalized or 'light' in val_normalized or 'thin' in val_normalized:
            cloud_lite.append(val)
        elif 'deep' in val_lower or 'thick' in val_lower or 'dense' in val_lower:
            cloud_deep.append(val)
        elif 'cloud' in val_lower:
            cloud_lite.append(val)  # Default cloud to lite
        else:
            other.append(val)
    
    if cloud_lite:
        logger.info("🔹 Cloud Lite / Thin Cloud:")
        for val in cloud_lite:
            logger.info(f"   '{val}'")
    
    if cloud_deep:
        logger.info("\n🔹 Cloud Deep / Thick Cloud:")
        for val in cloud_deep:
            logger.info(f"   '{val}'")
    
    if cloud_shadow:
        logger.info("\n🔹 Cloud Shadow:")
        for val in cloud_shadow:
            logger.info(f"   '{val}'")
    
    if other:
        logger.info("\n🔹 Other (unknown):")
        for val in other:
            logger.info(f"   '{val}'")
    
    logger.info("\n" + "="*100)
    logger.info("Next step: Review the groupings above and provide them to update CLASS_MAPPING and SYNONYMS")
    logger.info("="*100 + "\n")
    
    # Also save to a text file for easy reference
    output_file = Path("gpkg_remarks_summary.txt")
    with open(output_file, 'w') as f:
        f.write("GPKG REMARK METADATA SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        
        for col in POSSIBLE_REMARK_COLS:
            if col not in all_remarks or len(all_remarks[col]) == 0:
                continue
            
            values_dict = all_remarks[col]
            num_unique = len(values_dict)
            total_count = sum(values_dict.values())
            num_gpkgs = len(gpkg_column_map[col])
            
            f.write(f"Column: '{col}'\n")
            f.write(f"  Found in: {num_gpkgs} GPKG file(s)\n")
            f.write(f"  Unique values: {num_unique}\n")
            f.write(f"  Total occurrences: {total_count}\n")
            f.write(f"  Values (sorted by frequency):\n")
            
            sorted_values = sorted(values_dict.items(), key=lambda x: x[1], reverse=True)
            
            for val, count in sorted_values:
                pct = (count / total_count) * 100
                f.write(f"    • '{val}' ({count} occurrences, {pct:.1f}%)\n")
            
            f.write("\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("SUGGESTED GROUPINGS\n")
        f.write("=" * 100 + "\n\n")
        
        if cloud_lite:
            f.write("Cloud Lite / Thin Cloud:\n")
            for val in cloud_lite:
                f.write(f"  '{val}'\n")
            f.write("\n")
        
        if cloud_deep:
            f.write("Cloud Deep / Thick Cloud:\n")
            for val in cloud_deep:
                f.write(f"  '{val}'\n")
            f.write("\n")
        
        if cloud_shadow:
            f.write("Cloud Shadow:\n")
            for val in cloud_shadow:
                f.write(f"  '{val}'\n")
            f.write("\n")
        
        if other:
            f.write("Other (unknown):\n")
            for val in other:
                f.write(f"  '{val}'\n")
    
    logger.info(f"✅ Summary saved to: {output_file}")

if __name__ == "__main__":
    extract_gpkg_metadata()