import re
import os
import glob
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import geopandas as gpd
from shapely.affinity import scale, translate
from shapely.geometry import box
from pathlib import Path
from tqdm import tqdm
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
try:
    from local_config import (
        PREPARE_MASKS_IMAGE_DIR,
        PREPARE_MASKS_GPKG_DIR,
        PREPARE_MASKS_OUTPUT_DIR,
        PREPARE_MASKS_TFW_DIR,
        PROJECT_CONFIGS,
        CLASS_MAPPING,
    )
    IMAGE_DIR = PREPARE_MASKS_IMAGE_DIR
    GPKG_DIR = PREPARE_MASKS_GPKG_DIR
    OUTPUT_DIR = PREPARE_MASKS_OUTPUT_DIR
    TFW_DIR = PREPARE_MASKS_TFW_DIR
    
except ImportError as e:
    logger.error(f"CRITICAL: local_config.py not found or missing required config: {e}. Please create 'training/scripts/local_config.py' to define local paths.")
    import sys
    sys.exit(1)

# Note: PROJECT_CONFIGS, CLASS_MAPPING, and SYNONYMS are now imported from local_config.py

def detect_project_type(filename):
    """
    Detect project type from filename.
    
    Examples:
        25FD8205Ax00016_10808.tif -> D82A
        25FD0905Ex07650_07650.tif -> D09E
        25FD0905Dx_40007_008715.tif -> D09D
        08048-qvRGB.tif -> FHSTG (or DRONTEN)
        199002098_0053_01_0191_P00_01.tif -> PHKIO
    """
    filename = str(filename)
    
    # Pattern for D82A, D09E, D09D format: 25FD[0-9]{2}[0-9]{2}[A-Z]x
    match = re.search(r'25FD(\d{2})(\d{2})([A-Z])x', filename)
    if match:
        dept = match.group(1)  # e.g., "82" or "09"
        subdept = match.group(2)  # e.g., "05"
        letter = match.group(3)  # e.g., "A", "E", "D"
        code = f"D{dept}{letter}"
        if code in PROJECT_CONFIGS:
            return code
    
    # Pattern for PHKIO: 9digits_4digits_2digits_4digits_Pxx_2digits
    # Example: 199002098_0053_01_0191_P00_01.tif
    if re.search(r'\d{9}_\d{4}_\d{2}_\d{4}_P\d{2}_\d{2}', filename):
        return "PHKIO"
    
    # Pattern for FHSTG/DRONTEN: contains "qvRGB"
    if "qvRGB" in filename:
        # Need additional logic to distinguish FHSTG from DRONTEN
        # Option 1: Check for specific patterns in filename
        # e.g., if filename starts with digits followed by dash
        if re.match(r'^\d+-qvRGB', filename):
            return "FHSTG"
        # Option 2: Check for other distinguishing characteristics
        # Add your specific logic here based on filename patterns
        return "FHSTG"  # or "DRONTEN" as default
    
    # Default: Could not detect
    return None


def process_masks():
    if not IMAGE_DIR.exists():
        logger.error(f"Error: Image directory not found: {IMAGE_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all images (support multiple formats)
    image_files = []
    for ext in ['*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png', '*.iiq']:
        image_files.extend(IMAGE_DIR.glob(ext))
    
    logger.info(f"Found {len(image_files)} images.")
    
    # Statistics
    stats = {
        "total": 0,
        "by_project": {},
        "no_gpkg": 0,
        "success": 0,
        "failed": 0
    }

    for img_path in tqdm(image_files):
        try:
            # if not img_path.stem == '00295-qvRGB':
            #     continue
            # Detect project type
            project_type = detect_project_type(img_path.name)
            
            if project_type is None:
                logger.warning(f"Could not detect project type for {img_path.name}, skipping.")
                stats["failed"] += 1
                continue
            
            # Get project config
            config = PROJECT_CONFIGS[project_type]
            orig_w, orig_h = config["original_size"]
            has_tfw = config["has_tfw"]
            
            # Track stats
            stats["total"] += 1
            if project_type not in stats["by_project"]:
                stats["by_project"][project_type] = 0
            stats["by_project"][project_type] += 1
            
            # Read current image to get actual dimensions
            with rasterio.open(img_path) as src:
                curr_h, curr_w = src.shape
                src_profile = src.profile.copy()
            
            # Calculate scale factors
            scale_x = curr_w / orig_w
            scale_y = curr_h / orig_h
            
            logger.info(f"\n{img_path.name}:")
            logger.info(f"  Project: {project_type} ({config['description']})")
            logger.info(f"  Original: {orig_w}×{orig_h} → Current: {curr_w}×{curr_h}")
            logger.info(f"  Scale: {scale_x:.4f}×{scale_y:.4f}")
            
            # Create empty mask
            mask = np.zeros((curr_h, curr_w), dtype=np.uint8)
            
            # Identity transform for pixel coordinates
            dst_transform = rasterio.Affine(1, 0, 0, 0, 1, 0)
            dst_crs = None
            
            # Construct corresponding GPKG path
            gpkg_path = GPKG_DIR / f"{img_path.stem}.gpkg"
            tfw_path = TFW_DIR / f"{img_path.stem}.tfw"
            
            from rasterio.transform import Affine

            def transform_from_tfw(tfw_path):
                with open(tfw_path, "r") as f:
                    A = float(f.readline())  # pixel width
                    D = float(f.readline())  # row rotation
                    B = float(f.readline())  # column rotation
                    E = float(f.readline())  # pixel height (negative)
                    C = float(f.readline())  # x origin (center of UL pixel)
                    F = float(f.readline())  # y origin (center of UL pixel)

                return Affine(A, B, C,
                            D, E, F)


            if gpkg_path.exists():
                # Read GPKG
                gdf = gpd.read_file(gpkg_path)
                
                # Check for normalized 'remark' column
                if 'remark' not in gdf.columns:
                    logger.warning(f"  Warning: 'remark' column missing in {gpkg_path.name}. Generating black mask.")
                    logger.warning(f"           GPKG has not been normalized. Run normalize_gpkg_files.py first.")
                    stats["no_gpkg"] += 1
                else:
                    logger.info(f"  GPKG found with {len(gdf)} features")
                    logger.info(f"  GPKG CRS: {gdf.crs}")
                    logger.info(f"  GPKG Bounds (original): {gdf.total_bounds}")
                    
                    # Handle legacy .tfw project specially
                    if has_tfw:
                        logger.info(f"  LEGACY PROJECT: Stripping CRS (coordinates match .tfw values)")
                        # Strip CRS to use raw coordinate values (they're in geo-space but match .tfw)
                        # gdf.crs = None
                        # Strip CRS to use raw coordinate values (they match .tfw numerically)
                        gdf_geo = gdf.copy()
                        gdf_geo.crs = None
                        geo_transform = transform_from_tfw(tfw_path)
                        
                        # Convert from geo coordinates to pixel coordinates
                        # geo_transform maps: pixel -> geo
                        # We need the inverse: geo -> pixel
                        from rasterio.transform import rowcol
                        from shapely.geometry import mapping, shape
                        from shapely.ops import transform as shp_transform
                        
                        def geo_to_pixel(x, y):
                            """Convert geo coordinates to pixel coordinates"""
                            row, col = rowcol(geo_transform, x, y)
                            return (col, row)  # Return as (x, y) in pixel space
                        
                        # Transform all geometries from geo space to pixel space
                        gdf_raw = gdf_geo.copy()
                        gdf_raw['geometry'] = gdf_raw['geometry'].apply(
                            lambda geom: shp_transform(geo_to_pixel, geom)
                        )
                        gdf = gdf_raw.copy()
                        scale_y = -1 * scale_y
                    
                    # Scale geometries from original size to current size
                    logger.info(f"  Scaling geometries to current image size...")
                    # remove empty geometries
                    gdf = gdf[gdf['geometry'].is_valid]
                    gdf['geometry'] = gdf['geometry'].apply(
                        lambda geom: scale(geom, xfact=scale_x, yfact=-scale_y, origin=(0, 0))
                    )
                    
                    logger.info(f"  GPKG Bounds (scaled): {gdf.total_bounds}")
                    
                    shapes_to_burn = []

                    for idx, row in gdf.iterrows():
                        remark_value = row['remark']

                        if not isinstance(remark_value, str):
                            continue

                        remark_value = remark_value.strip()

                        # Map normalized remark values to class codes
                        if remark_value in CLASS_MAPPING:
                            class_code = CLASS_MAPPING[remark_value]
                            shapes_to_burn.append((row['geometry'], class_code))
                        else:
                            logger.warning(f"    Unknown remark value: '{remark_value}' - skipping")
                    
                    # Sort by priority (cloud_shadow > thick_cloud > thin_cloud)
                    # This ensures shadows/clouds are rendered on top
                    shapes_to_burn.sort(key=lambda x: x[1], reverse=True)
                    
                    if shapes_to_burn:
                        logger.info(f"  Rasterizing {len(shapes_to_burn)} valid shapes...")
                        
                        # Rasterize
                        rasterize(
                            shapes=shapes_to_burn,
                            out=mask,
                            transform=dst_transform,
                            fill=0,
                            default_value=0,
                            dtype=np.uint8,
                            all_touched=True  # Use all_touched for better coverage
                        )
                        
                        if mask.max() > 0:
                            logger.info(f"  ✅ Mask generated! Max value: {mask.max()}, Non-zero pixels: {np.count_nonzero(mask)}")
                            stats["success"] += 1
                        else:
                            logger.warning(f"  ⚠️ Mask is empty despite having shapes!")
                            logger.warning(f"     Image bounds: (0, 0, {curr_w}, {curr_h})")
                            logger.warning(f"     Shape bounds: {gdf.total_bounds}")
                            stats["failed"] += 1
                    else:
                        logger.warning(f"  No valid shapes found (check 'remark' column values)")
                        stats["no_gpkg"] += 1
            else:
                # No GPKG, create black mask
                logger.info(f"  No GPKG found, creating black mask")
                stats["no_gpkg"] += 1
            
            # Save mask
            out_path = OUTPUT_DIR / f"{img_path.stem}.png"
            
            profile = {
                'driver': 'GTiff',
                'dtype': 'uint8',
                'count': 1,
                'height': curr_h,
                'width': curr_w,
                'compress': 'lzw',
                'nodata': 99,
                'transform': dst_transform,
                'crs': None
            }
            
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(mask, 1)
                
        except Exception as e:
            logger.error(f"Failed to process {img_path.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            stats["failed"] += 1
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("PROCESSING COMPLETE - SUMMARY")
    logger.info("="*80)
    logger.info(f"Total images processed: {stats['total']}")
    logger.info(f"Successful masks: {stats['success']}")
    logger.info(f"Black masks (no GPKG): {stats['no_gpkg']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"\nBy project:")
    for proj, count in stats["by_project"].items():
        logger.info(f"  {proj} ({PROJECT_CONFIGS[proj]['description']}): {count} images")
    logger.info("\nClass Mapping:")
    for class_name, class_code in CLASS_MAPPING.items():
        logger.info(f"  {class_name}: {class_code}")
    logger.info("="*80)

if __name__ == "__main__":
    process_masks()