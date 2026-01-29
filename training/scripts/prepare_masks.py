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
# Default CRS for the project (France)
PROJECT_CRS = "EPSG:2154" 

try:
    from local_config import (
        PREPARE_MASKS_IMAGE_DIR,
        PREPARE_MASKS_GPKG_DIR,
        PREPARE_MASKS_OUTPUT_DIR
    )
    IMAGE_DIR = PREPARE_MASKS_IMAGE_DIR
    GPKG_DIR = PREPARE_MASKS_GPKG_DIR
    OUTPUT_DIR = PREPARE_MASKS_OUTPUT_DIR
except ImportError:
    logger.error("CRITICAL: local_config.py not found. Please create 'training/scripts/local_config.py' to define local paths.")
    # Fallback to prevent immediate crash if user is just reading code, but will likely fail later
    IMAGE_DIR = Path("DATA_DIR_NOT_SET/images")
    GPKG_DIR = Path("DATA_DIR_NOT_SET/gpkg")
    OUTPUT_DIR = Path("DATA_DIR_NOT_SET/masks_2")

# CLASS MAPPING
# 'Remark' attribute values -> Integer Class ID
CLASS_MAPPING = {
    'Cloud Deep': 1,
    'Cloud Lite': 2,
    'Shadow': 3
}

def process_masks():
    if not IMAGE_DIR.exists():
        logger.error(f"Error: Image directory not found: {IMAGE_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    image_files = list(IMAGE_DIR.glob("*.tif"))
    logger.info(f"Found {len(image_files)} images.")

    for img_path in tqdm(image_files):
        try:
            # 1. Read Image Metadata
            with rasterio.open(img_path) as src:
                height, width = src.shape
                src_profile = src.profile.copy()
                
                # Always use pixel coordinate transform for output masks
                # This creates a simple pixel-space coordinate system: (0,0) to (width, height)
                dst_transform = rasterio.Affine(1, 0, 0, 0, 1, 0)  # Standard image coords: origin top-left, y increases downward
                dst_crs = None
                
                # Check for World File (.tfw) to know if we need coordinate transformation
                tfw_path = img_path.with_suffix(".tfw")
                is_georeferenced = tfw_path.exists()
                
                if is_georeferenced:
                    # Store the geo transform for coordinate conversion
                    geo_transform = src.transform
                    logger.info(f"  Image has georeferencing: {geo_transform}")
            
            # Create empty mask (black mask by default)
            mask = np.zeros((height, width), dtype=np.uint8)

            # Construct corresponding GPKG path
            gpkg_path = GPKG_DIR / f"{img_path.stem}.gpkg"
            
            if gpkg_path.exists():
                # 2. Read GPKG
                gdf = gpd.read_file(gpkg_path)
                
                if 'Remark' not in gdf.columns:
                    logger.warning(f"Warning: 'Remark' column missing in {gpkg_path.name}. Generating black mask.")
                else:
                    # 3. Coordinate Transformation Logic
                    if is_georeferenced:
                        # Georeferenced Case: Convert from geo coordinates to pixel coordinates
                        # The GPKG contains coordinates in geo space (stored as EPSG:28992 but actually matching .tfw values)
                        # We need to convert these to pixel coordinates (0,0 to width,height)
                        
                        logger.info(f"  GPKG CRS: {gdf.crs}")
                        logger.info(f"  GPKG Bounds (raw): {gdf.total_bounds}")
                        
                        # Strip CRS to use raw coordinate values (they match .tfw numerically)
                        gdf_geo = gdf.copy()
                        gdf_geo.crs = None
                        
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
                        
                        logger.info(f"  Converted to pixel coordinates. Bounds: {gdf_raw.total_bounds}")
                        logger.info(f"  Image size: {width}x{height}")

                            
                    else:
                        # Non-georeferenced logic (Pixel coordinates)
                        gdf_raw = gdf.copy()
                        bounds = gdf_raw.total_bounds # [minx, miny, maxx, maxy]
                        min_y, max_y = bounds[1], bounds[3]
                        
                        # Helper to flip Y
                        def flip_y(geom):
                            return scale(geom, xfact=1.0, yfact=-1.0, origin=(0,0))
                        
                        # If coordinates look like they are flipped (negative Y), flip them back
                        # if max_y <= 0:
                        gdf_raw['geometry'] = gdf_raw['geometry'].apply(flip_y)
                    
                    # Filter and Rasterize (use gdf_raw instead of gdf)
                    shapes_to_burn = []

                    for idx, row in gdf_raw.iterrows():
                        raw_cls = gdf.loc[idx, 'Remark']  # Get from original gdf

                        if isinstance(raw_cls, str):
                            # remove leading/trailing spaces + collapse multiple spaces
                            cls_name = re.sub(r'\s+', ' ', raw_cls.strip())
                        else:
                            continue

                        if cls_name in CLASS_MAPPING:
                            val = CLASS_MAPPING[cls_name]
                            shapes_to_burn.append((row['geometry'], val))
                    
                    # Priority: Shadow(3), Thin(2), Thick(1). Last one drawn wins.
                    shapes_to_burn.sort(key=lambda x: x[1], reverse=True) 
                    
                    if shapes_to_burn:
                        logger.info(f"  Rasterizing {len(shapes_to_burn)} shapes...")
                        
                        # 1. Initial Attempt (Default)
                        rasterize(
                            shapes=shapes_to_burn,
                            out=mask,
                            transform=dst_transform,
                            fill=0,
                            default_value=0,
                            dtype=np.uint8
                        )

                        # CHECK: If mask is empty but shapes exist, investigate and try fallbacks
                        if mask.max() == 0:
                            logger.warning(f"⚠️ Mask is empty for {img_path.name} despite finding {len(shapes_to_burn)} shapes! Attempting fallbacks...")
                            
                            # Get image bounds for debugging
                            with rasterio.open(img_path) as src:
                                img_bounds = src.bounds
                            
                            shp_bounds = gdf_raw.total_bounds # [minx, miny, maxx, maxy]
                            
                            img_box = box(img_bounds.left, img_bounds.bottom, img_bounds.right, img_bounds.top)
                            shp_box = box(shp_bounds[0], shp_bounds[1], shp_bounds[2], shp_bounds[3])
                            
                            logger.info(f"  Image Bounds: {img_bounds}")
                            logger.info(f"  Shape Bounds: {shp_bounds}")
                            logger.info(f"  Intersects: {img_box.intersects(shp_box)}")
                            logger.info(f"  Distance between centers: {img_box.centroid.distance(shp_box.centroid):.2f}")

                            # Strategy 1: all_touched=True
                            # Helps with very small polygons or lines that don't cover pixel centers
                            logger.info("  🔄 Strategy 1: Retry with all_touched=True...")
                            rasterize(
                                shapes=shapes_to_burn,
                                out=mask,
                                transform=dst_transform,
                                fill=0,
                                default_value=0,
                                dtype=np.uint8,
                                all_touched=True
                            )
                            
                            if mask.max() == 0:
                                # Strategy 2: Y-Flip (scale y=-1)
                                # Handles cases where Y-axis direction is inverted (e.g. Cartesian vs Image)
                                logger.info("  🔄 Strategy 2: Y-Flip (scale y=-1)...")
                                flipped_shapes = []
                                for geom, val in shapes_to_burn:
                                    f_geom = scale(geom, xfact=1.0, yfact=-1.0, origin=(0,0))
                                    flipped_shapes.append((f_geom, val))
                                
                                rasterize(
                                    shapes=flipped_shapes,
                                    out=mask,
                                    transform=dst_transform,
                                    fill=0,
                                    default_value=0,
                                    dtype=np.uint8,
                                    all_touched=True
                                )

                            if mask.max() == 0:
                                # Strategy 3: Y-Flip + Translate (y = height - y)
                                # Handles "Bottom-Left Origin" vs "Top-Left Origin" coordinate systems
                                # Logic: y_new = height - y_old
                                # Implementation: Scale y=-1 (around 0), then Translate y=+height
                                if not is_georeferenced:
                                    # This typically applies to pixel coordinates
                                    logger.info(f"  🔄 Strategy 3: Y-Flip + Translate (y = {height} - y)...")
                                    # Re-use flipped shapes from Strategy 2 or generate them
                                    # We need to make sure we are flipping original shapes
                                    
                                    flip_translate_shapes = []
                                    for geom, val in shapes_to_burn:
                                        # 1. Flip Y around 0 (y -> -y)
                                        f_geom = scale(geom, xfact=1.0, yfact=-1.0, origin=(0,0))
                                        # 2. Translate by image height (y -> -y + height)
                                        ft_geom = translate(f_geom, xoff=0.0, yoff=float(height))
                                        flip_translate_shapes.append((ft_geom, val))
                                        
                                    rasterize(
                                        shapes=flip_translate_shapes,
                                        out=mask,
                                        transform=dst_transform,
                                        fill=0,
                                        default_value=0,
                                        dtype=np.uint8,
                                        all_touched=True
                                    )

                            if mask.max() > 0:
                                logger.info(f"  ✅ Fallback successful! Mask generated.")
                            else:
                                logger.error(f"  ❌ All fallbacks failed. Mask remains empty.")
                        else:
                            logger.info(f"  ✅ Mask generated successfully! Max value: {mask.max()}")
                
            else:
                # GPKG not found, mask remains all zeros (full black)
                logger.info(f"  No GPKG found for {img_path.name}, creating empty mask.")
            
            # 4. Save Mask
            out_path = OUTPUT_DIR / f"{img_path.stem}.tif"
            
            # Profile: Single band, uint8, LZW compressed, pixel coordinates
            profile = {
                'driver': 'GTiff',
                'dtype': 'uint8',
                'count': 1,
                'height': height,
                'width': width,
                'compress': 'lzw',
                'nodata': 99, 
                'transform': dst_transform,
                'crs': None  # No CRS - pure pixel coordinates
            }
            
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(mask, 1)

        except Exception as e:
            logger.error(f"Failed to process {img_path.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    process_masks()