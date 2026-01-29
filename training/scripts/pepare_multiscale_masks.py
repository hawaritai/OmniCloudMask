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
        PREPARE_MASKS_OUTPUT_DIR
    )
    # IMAGE_DIR = Path(r"E:\ImageQC\dataset\Test_all_gt\images_pos")
    # GPKG_DIR = Path(r"D:\projects\Image_QC_GUI\2_Repos\OmniCloudMask\training\data\all_gpkg")
    # OUTPUT_DIR = Path(r"D:\projects\Image_QC_GUI\2_Repos\OmniCloudMask\training\data\all_masks")
    IMAGE_DIR = PREPARE_MASKS_IMAGE_DIR
    GPKG_DIR = PREPARE_MASKS_GPKG_DIR
    OUTPUT_DIR = PREPARE_MASKS_OUTPUT_DIR
    TFW_DIR = Path(r"Q:\02_PROJECTS\2051_102025082_D82_Tarn-et-Garonne_A\60_UM\LVL03_CertiflAI_0610")
    
except ImportError:
    logger.error("CRITICAL: local_config.py not found. Please create 'training/scripts/local_config.py' to define local paths.")
    import sys
    sys.exit(1)

# --- PROJECT CONFIGURATIONS ---
# Map project codes to their original resolutions
# Format: "project_code": (original_width, original_height)
PROJECT_CONFIGS = {
    "D82A": {  # Garonne_A - LEGACY with .tfw
        "original_size": (9370, 6020),
        "current_size": (640, 411),
        "has_tfw": True,  # Special handling for geo-coordinates
        "description": "2051_102025082_D82_Tarn-et-Garonne_A (LEGACY)"
    },
    "D09E": {  # Arriege_E
        "original_size": (26460, 17004),
        "current_size": (640, 411),
        "has_tfw": False,
        "description": "2051_102025077_D09_Arriege_E"
    },
    "D09D": {  # Arriege_D
        "original_size": (8820, 5668),
        "current_size": (640, 411),
        "has_tfw": False,
        "description": "2051_102025077_D09_Arriege_D"
    },
    "FHSTG": {  # AIR_FHSTG
        "original_size": (1681, 1080),
        "current_size": (640, 411),
        "has_tfw": False,
        "description": "2025_08_03_AIR_FHSTG"
    },
    "PHKIO": {  # AIR_PHKIO
        "original_size": (640, 480),
        "current_size": (640, 480),
        "has_tfw": False,
        "description": "2025_08_10_AIR_PHKIO"
    }
}

# CLASS MAPPING
# 'Remark' attribute values -> Integer Class ID
CLASS_MAPPING = {
    'Cloud Deep': 1,
    'Cloud Lite': 2,
    'Shadow': 3
}

def detect_project_type(filename):
    """
    Detect project type from filename.
    
    Examples:
        25FD8205Ax00016_10808.tif -> D82A
        25FD0905Ex07650_07650.tif -> D09E
        25FD0905Dx_40007_008715.tif -> D09D
        08048-qvRGB.jpg -> FHSTG
        __199002098_0053_01_0191_P00_01.iiq -> PHKIO
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
    
    # Pattern for FHSTG: contains "qvRGB"
    if "qvRGB" in filename:
        return "FHSTG"
    
    # Pattern for PHKIO: starts with "__" and contains .iiq
    if filename.startswith("__") and ".iiq" in filename:
        return "PHKIO"
    
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
                
                if 'Remark' not in gdf.columns:
                    logger.warning(f"  Warning: 'Remark' column missing in {gpkg_path.name}. Generating black mask.")
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
                    gdf['geometry'] = gdf['geometry'].apply(
                        lambda geom: scale(geom, xfact=scale_x, yfact=-scale_y, origin=(0, 0))
                    )
                    
                    logger.info(f"  GPKG Bounds (scaled): {gdf.total_bounds}")
                    
                    # Prepare shapes to burn
                    shapes_to_burn = []
                    
                    for idx, row in gdf.iterrows():
                        raw_cls = row['Remark']
                        
                        if isinstance(raw_cls, str):
                            cls_name = re.sub(r'\s+', ' ', raw_cls.strip())
                        else:
                            continue
                        
                        if cls_name in CLASS_MAPPING:
                            val = CLASS_MAPPING[cls_name]
                            shapes_to_burn.append((row['geometry'], val))
                    
                    # Sort by priority (Shadow > Cloud Lite > Cloud Deep)
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
                        logger.warning(f"  No valid shapes found (check 'Remark' column values)")
                        stats["no_gpkg"] += 1
            else:
                # No GPKG, create black mask
                logger.info(f"  No GPKG found, creating black mask")
                stats["no_gpkg"] += 1
            
            # Save mask
            out_path = OUTPUT_DIR / f"{img_path.stem}.tif"
            
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
    logger.info("="*80)

if __name__ == "__main__":
    process_masks()