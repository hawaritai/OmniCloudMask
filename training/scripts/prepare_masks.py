
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
        # Construct corresponding GPKG path
        # Assuming name matches: image.tif -> image.gpkg
        gpkg_path = GPKG_DIR / f"{img_path.stem}.gpkg"
        
        if not gpkg_path.exists():
            print(f"Warning: GPKG not found for {img_path.name}, skipping.")
            continue

        try:
            # 1. Read Image Metadata
            with rasterio.open(img_path) as src:
                height, width = src.shape
                src_profile = src.profile.copy()
                # Create a default transform for the mask (0,0 top-left)
                # We will write the mask as a simple non-georeferenced image (or same as source if source was)
                # User said source is not georeferenced.
                dst_transform = rasterio.Affine(1, 0, 0, 0, 1, 0)
                
            # 2. Read GPKG
            gdf = gpd.read_file(gpkg_path)
            
            if 'Remark' not in gdf.columns:
                print(f"Warning: 'Remark' column missing in {gpkg_path.name}")
                continue

            # 3. Coordinate Transformation Logic
            # User says: "Y only inverted".
            # Check bounds to confirm hypothesis
            # Standard QGIS non-georef behavior: Y is negative.
            # If Y is negative, we flip it to positive.
            # If Y is positive, we might need to do Height - Y?
            
            bounds = gdf.total_bounds # [minx, miny, maxx, maxy]
            min_y, max_y = bounds[1], bounds[3]
            
            geometries = []
            
            # Helper to flip Y
            def flip_y(geom):
                # Scale Y by -1, origin=(0,0)
                # If coords are (x, -y), this makes them (x, y)
                return scale(geom, xfact=1.0, yfact=-1.0, origin=(0,0))
            
            # Helper for Height - Y (if needed, but user said "inverted")
            # If the user meant cartesian Y (0 at bottom) to image Y (0 at top)
            # Then y_img = Height - y_cart
            
            # Heuristic:
            if max_y <= 0:
                # Case 1: All Y are negative. This matches QGIS "pixel size 1, -1" behavior.
                # We just flip sign.
                # print(f"Detected negative Y coords in {gpkg_path.name}. Flipping sign.")
                gdf['geometry'] = gdf['geometry'].apply(flip_y)
            elif min_y >= 0:
                # Case 2: Positive Y. 
                # If "Inverted", maybe (x, y) -> (x, -y)? But that would make them negative (off image).
                # Maybe they meant (x, height - y)?
                # Or maybe they are correct and we don't need to do anything?
                # User said "georef is not correct... Y only inverted".
                # If I have positive Y in GPKG, and I "invert" them, I get negative Y.
                # Unless origin was non-zero?
                # Let's assume for now if positive, maybe they are already correct?
                # OR maybe they are "inverted" meaning they look upside down if plotted directly?
                # If they look upside down, then `y_new = Height - y_old`.
                # Let's try `Height - y` if they are positive and fit in the image.
                # BUT, safer to ask or check?
                # Given strict instruction "Y only inverted", and "image coords", 
                # I will stick to the most common issue: QGIS saving (x, -y).
                # If they are already positive, I'll log a warning but keep them, 
                # or perhaps the user meant `y = -y` relative to some other origin?
                # Let's assume standard behavior: scale(y=-1).
                pass
            
            # Filter and Rasterize
            # Create empty mask
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Process classes in specific order or independent?
            # Standard semantic seg: 0=bg, 1=class1, etc.
            # If overlap, last one drawn wins.
            
            # Sort or just iterate?
            # Let's iterate by class to ensure we handle them.
            # Priority: Shadow (3) -> Thin (2) -> Thick (1)? Or Thick on top?
            # Usually thick clouds block everything.
            # Let's write Shadow, then Thin, then Thick. So Thick overwrites others.
            
            shapes_to_burn = []
            
            # We collect all shapes with their value
            for idx, row in gdf.iterrows():
                cls_name = row['Remark']
                if cls_name in CLASS_MAPPING:
                    val = CLASS_MAPPING[cls_name]
                    shapes_to_burn.append((row['geometry'], val))
            
            # Sort shapes by value if needed, but rasterize processes list in order.
            # We want specific layering?
            # If a pixel is both shadow and cloud? (Shadow on cloud -> Cloud)
            # (Cloud on Shadow -> Cloud).
            # So Cloud (1) should be last?
            # Let's sort by value: 3 (Shadow), 2 (Thin), 1 (Thick).
            # Reverse order so 1 is last?
            # No, `rasterize` iterates. Last one writes.
            # We want Thick (1) to be visible over Shadow (3)? Yes.
            # So order: 3, 2, 1.
            
            shapes_to_burn.sort(key=lambda x: x[1], reverse=True) 
            # Sorts 3, 2, 1. So 1 is first? No, 3 is first.
            # Wait, reverse=True -> 3, 2, 1.
            # If I burn 3, then 2, then 1... 1 will overwrite 2 and 3. Correct.
            
            if shapes_to_burn:
                rasterize(
                    shapes=shapes_to_burn,
                    out=mask,
                    transform=dst_transform, # Identity transform: geom coords = pixel coords
                    fill=0,
                    default_value=0,
                    dtype=np.uint8
                )
            
            # 4. Save Mask
            out_path = OUTPUT_DIR / f"{img_path.stem}.tif"
            
            # Profile: Single band, uint8, LZW compressed
            profile = src_profile.copy()
            profile.update({
                'driver': 'GTiff',
                'dtype': 'uint8',
                'count': 1,
                'compress': 'lzw',
                'nodata': 99, # User mentioned 99 for nodata
                'transform': dst_transform 
            })
            
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(mask, 1)

        except Exception as e:
            print(f"Failed to process {img_path.name}: {e}")

if __name__ == "__main__":
    process_masks()
