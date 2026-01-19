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

# Suppress warnings
warnings.filterwarnings("ignore")

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
    print("Successfully imported NIRGAN and RGBNIRHandler.")
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print(f"sys.path: {sys.path}")
    print("Attempting to add subdirectories to path...")
    # Fallback: add thirdparty explicitly
    sys.path.append(str(scripts_dir / "thirdparty"))
    try:
        from NIRGAN.create_NIR import get_NIR
        from fom_R.rgb_nir_handler import RGBNIRHandler
        print("Successfully imported NIRGAN (fallback).")
    except ImportError as e2:
        print(f"Critical Import Error: {e2}")
        sys.exit(1)

# --- CONFIGURATION ---
# UPDATE THESE PATHS TO MATCH YOUR LOCAL SETUP
BASE_DATA_DIR = Path(r"D:\\Projects\\QI47\\2025_Projects\\Image_QC_GUI\\2_Repo\\OmniCloudMask\\training\\data\\2051_102025077_D09_Arriege_D")
INPUT_IMAGES_DIR = BASE_DATA_DIR / "images"
INPUT_LABELS_DIR = BASE_DATA_DIR / "masks"
OUTPUT_DIR = BASE_DATA_DIR / "processed_dataset_2" # Where to save training chips

TARGET_GSD_M = 10.0  # Target resolution in meters (Match Sentinel-2 approx)
SOURCE_GSD_M = 1  # Your source resolution (5cm) - ADJUST IF NEEDED
TILE_SIZE = 509      # Required by OCM training script
# ---------------------

def preprocess_images():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Handler
    handler = RGBNIRHandler(device=device)

    # Create output directories
    (OUTPUT_DIR / "train").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "validation").mkdir(parents=True, exist_ok=True)

    # Calculate scale factor (e.g. 0.05 / 10 = 0.005)
    scale_factor = SOURCE_GSD_M / TARGET_GSD_M
    print(f"Downsampling scale factor: {scale_factor} (Source: {SOURCE_GSD_M}m -> Target: {TARGET_GSD_M}m)")

    # Find all images
    image_files = list(INPUT_IMAGES_DIR.glob("*.tif"))
    print(f"Found {len(image_files)} images to process.")

    for img_path in tqdm(image_files):
        # Look for label in label dir with same name
        label_path = INPUT_LABELS_DIR / img_path.name
        
        if not label_path.exists():
            print(f"Skipping {img_path.name}, label not found in {INPUT_LABELS_DIR}")
            continue

        try:
            with rasterio.open(img_path) as src_img, rasterio.open(label_path) as src_lbl:
                # 1. Calculate new dimensions after downsampling
                new_height = int(src_img.height * scale_factor)
                new_width = int(src_img.width * scale_factor)
                
                # Check for minimum dimensions
                if new_height < 1 or new_width < 1:
                    print(f"Skipping {img_path.name}: Downsampled size too small ({new_width}x{new_height})")
                    continue

                # 2. Read and Resample (Downsample) entire image to memory
                # Read RGB (3 bands)
                # Ensure we read only first 3 bands if there are more
                count = min(3, src_img.count)
                # We expect at least 3 bands for RGB
                if count < 3:
                    print(f"Skipping {img_path.name}: Not enough bands ({count})")
                    continue
                    
                data_img_rgb = src_img.read(
                    [1, 2, 3], # Read bands 1, 2, 3 (RGB)
                    out_shape=(3, new_height, new_width),
                    resampling=Resampling.bilinear
                )
                
                # Use nearest neighbor for labels to preserve class integers (0,1,2,3)
                data_lbl = src_lbl.read(
                    out_shape=(src_lbl.count, new_height, new_width),
                    resampling=Resampling.nearest
                )

                # --- 2b. Generate NIR and Stack ---
                # Prepare RGB for NIRGAN (H, W, 3)
                # data_img_rgb is (3, H, W)
                rgb_for_gan = np.transpose(data_img_rgb, (1, 2, 0)) # CHW -> HWC
                
                # Generate NIR (returns (H, W) scaled to 0-10000 usually)
                # Note: get_NIR might expect 0-255 uint8 or float 0-1.
                # data_img_rgb is likely uint8 or uint16 depending on source.
                # create_NIR.to_tensor divides by 255 if it detects uint8? No, let's check create_NIR code.
                # It does: `tensor = torch.from_numpy(arr.transpose((2, 0, 1))).float() / 255.0`
                # So it assumes 0-255 input!
                
                # Check dtype
                if data_img_rgb.dtype == np.uint16:
                    # Scale to 0-255 for GAN? Or normalize?
                    # The GAN expects [0,1] or [0,255].
                    # Let's normalize to 0-255 roughly for GAN input
                    rgb_for_gan = (rgb_for_gan / 65535.0 * 255.0).astype(np.uint8)
                elif data_img_rgb.dtype != np.uint8:
                    # If float, assume 0-1?
                    if data_img_rgb.max() <= 1.0:
                        rgb_for_gan = (rgb_for_gan * 255.0).astype(np.uint8)
                
                # Now run GAN
                nir_synthetic = get_NIR(rgb_for_gan, device=device)
                
                # --- RESIZE NIR TO MATCH RGB IF NEEDED ---
                if nir_synthetic.shape != (new_height, new_width):
                    # Ensure nir_synthetic is float32 for resizing
                    nir_synthetic = nir_synthetic.astype(np.float32)
                    
                    # cv2.resize expects (width, height)
                    nir_synthetic = cv2.resize(nir_synthetic, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

                # Stack using handler [Scaled R, Scaled G, NIR]
                # RGBNIRHandler.stack_rgb_nir expects 2D arrays (H, W).
                # It can handle normalization.
                # We pass the original downsampled bands (red, green) to preserve radiometric quality if possible,
                # but we need to pass them as float32 for the handler.
                
                red = data_img_rgb[0].astype(np.float32)
                green = data_img_rgb[1].astype(np.float32)
                
                # Use handler to stack and scale
                # scale_to_dn=True will scale normalized [0,1] R/G to [0,10000]
                # The handler normalizes input red/green first.
                rgn_stack = handler.stack_rgb_nir(
                    red=red,
                    green=green,
                    nir=nir_synthetic,
                    normalize=True,
                    scale_to_dn=True,
                    dn_range=(0, 10000)
                )
                # rgn_stack is (3, H, W) float32

                # 3. Tile into 509x509 patches
                n_cols = math.ceil(new_width / TILE_SIZE)
                n_rows = math.ceil(new_height / TILE_SIZE)

                for row in range(n_rows):
                    for col in range(n_cols):
                        # Define window
                        x_off = col * TILE_SIZE
                        y_off = row * TILE_SIZE
                        
                        # Adjust offset if tile exceeds boundaries (force overlap for last tiles)
                        if x_off + TILE_SIZE > new_width:
                            x_off = max(0, new_width - TILE_SIZE)
                        
                        if y_off + TILE_SIZE > new_height:
                            y_off = max(0, new_height - TILE_SIZE)

                        # Extract tile from RGN stack
                        tile_img = rgn_stack[:, y_off:y_off+TILE_SIZE, x_off:x_off+TILE_SIZE]
                        
                        # Extract tile from label
                        tile_lbl = data_lbl[:, y_off:y_off+TILE_SIZE, x_off:x_off+TILE_SIZE]

                        # Verify it's not empty (optional)
                        if tile_img.max() == 0: continue

                        # Define output filenames
                        split_dir = "validation" if np.random.rand() < 0.2 else "train"
                        base_name = f"{img_path.stem}_tile_{row}_{col}"
                        
                        out_img_path = OUTPUT_DIR / split_dir / f"{base_name}_image.tif"
                        out_lbl_path = OUTPUT_DIR / split_dir / f"{base_name}_label.tif"

                        # Create a basic transform for the tile (we lose absolute georef but keep pixel scale relative)
                        # We use an identity transform or similar since these are training chips
                        # Or we can construct one if needed, but typically OCM training works on pixel values.
                        dst_transform = rasterio.Affine(scale_factor, 0, 0, 0, -scale_factor, 0) # Simplified

                        # Save Tile (Image)
                        profile = src_img.profile.copy()
                        profile.update({
                            'height': TILE_SIZE,
                            'width': TILE_SIZE,
                            'count': 3,
                            'dtype': 'float32', 
                            'driver': 'GTiff',
                            'compress': 'lzw',
                            'transform': dst_transform,
                            'crs': None # remove CRS to avoid warnings if not valid
                        })

                        with rasterio.open(out_img_path, 'w', **profile) as dst:
                            dst.write(tile_img)
                        
                        # Save Tile (Label)
                        profile_lbl = src_lbl.profile.copy()
                        profile_lbl.update({
                            'height': TILE_SIZE,
                            'width': TILE_SIZE,
                            'count': 1,
                            'dtype': 'uint8',
                            'compress': 'lzw',
                            'transform': dst_transform,
                            'crs': None
                        })
                        with rasterio.open(out_lbl_path, 'w', **profile_lbl) as dst:
                            dst.write(tile_lbl)
                            
        except Exception as e:
            print(f"Failed to process {img_path.name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    preprocess_images()
