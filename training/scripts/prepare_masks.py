
import os
import glob
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import geopandas as gpd
from shapely.affinity import scale, translate
from pathlib import Path
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
BASE_DIR = Path(r"D:\\Projects\\QI47\\2025_Projects\\Image_QC_GUI\\1_Data\\training\\2051_102025077_D09_Arriege_D")
# Note: User provided path differs slightly in previous prompt vs 'working dir' context.
# I will use the path relative to the repo if possible, or the absolute path provided.
# The user provided: D:\\Projects\\QI47\\2025_Projects\\Image_QC_GUI\\1_Data\\training\\2051_102025077_D09_Arriege_D
# But earlier ls showed: D:\\Projects\\QI47\\2025_Projects\\Image_QC_GUI\\2_Repo\\OmniCloudMask\\training\\data\\2051_102025077_D09_Arriege_D
# I'll try to find where the data actually is. The 'ls' command in the previous turn worked on the '2_Repo' path.
# So I will use that one.

IMAGE_DIR = Path(r"D:\\Projects\\QI47\\2025_Projects\\Image_QC_GUI\\2_Repo\\OmniCloudMask\\training\\data\\2051_102025077_D09_Arriege_D\\images")
GPKG_DIR = Path(r"D:\\Projects\\QI47\\2025_Projects\\Image_QC_GUI\\2_Repo\\OmniCloudMask\\training\\data\\2051_102025077_D09_Arriege_D\\gpkg")
OUTPUT_DIR = Path(r"D:\\Projects\\QI47\\2025_Projects\\Image_QC_GUI\\2_Repo\\OmniCloudMask\\training\\data\\2051_102025077_D09_Arriege_D\\masks")

# CLASS MAPPING
# 'Remark' attribute values -> Integer Class ID
CLASS_MAPPING = {
    'Cloud Deep': 1,
    'Cloud Lite': 2,
    'Shadow': 3
}

def process_masks():
    if not IMAGE_DIR.exists():
        print(f"Error: Image directory not found: {IMAGE_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    image_files = list(IMAGE_DIR.glob("*.tif"))
    print(f"Found {len(image_files)} images.")

    for img_path in tqdm(image_files):
        try:
            # 1. Read Image Metadata
            with rasterio.open(img_path) as src:
                height, width = src.shape
                src_profile = src.profile.copy()
                # Create a default transform for the mask (0,0 top-left)
                dst_transform = rasterio.Affine(1, 0, 0, 0, 1, 0)
            
            # Create empty mask (black mask by default)
            mask = np.zeros((height, width), dtype=np.uint8)

            # Construct corresponding GPKG path
            gpkg_path = GPKG_DIR / f"{img_path.stem}.gpkg"
            
            if gpkg_path.exists():
                # 2. Read GPKG
                gdf = gpd.read_file(gpkg_path)
                
                if 'Remark' not in gdf.columns:
                    print(f"Warning: 'Remark' column missing in {gpkg_path.name}. Generating black mask.")
                else:
                    # 3. Coordinate Transformation Logic
                    bounds = gdf.total_bounds # [minx, miny, maxx, maxy]
                    min_y, max_y = bounds[1], bounds[3]
                    
                    # Helper to flip Y
                    def flip_y(geom):
                        return scale(geom, xfact=1.0, yfact=-1.0, origin=(0,0))
                    
                    if max_y <= 0:
                        gdf['geometry'] = gdf['geometry'].apply(flip_y)
                    
                    # Filter and Rasterize
                    shapes_to_burn = []
                    for idx, row in gdf.iterrows():
                        cls_name = row['Remark']
                        if cls_name in CLASS_MAPPING:
                            val = CLASS_MAPPING[cls_name]
                            shapes_to_burn.append((row['geometry'], val))
                    
                    # Priority: Shadow(3), Thin(2), Thick(1). Last one drawn wins.
                    shapes_to_burn.sort(key=lambda x: x[1], reverse=True) 
                    
                    if shapes_to_burn:
                        rasterize(
                            shapes=shapes_to_burn,
                            out=mask,
                            transform=dst_transform,
                            fill=0,
                            default_value=0,
                            dtype=np.uint8
                        )
            else:
                # GPKG not found, mask remains all zeros (full black)
                pass
            
            # 4. Save Mask
            out_path = OUTPUT_DIR / f"{img_path.stem}.tif"
            
            # Profile: Single band, uint8, LZW compressed
            profile = src_profile.copy()
            profile.update({
                'driver': 'GTiff',
                'dtype': 'uint8',
                'count': 1,
                'compress': 'lzw',
                'nodata': 99, 
                'transform': dst_transform 
            })
            
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(mask, 1)

        except Exception as e:
            print(f"Failed to process {img_path.name}: {e}")

if __name__ == "__main__":
    process_masks()
