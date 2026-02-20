import cv2
import os
from pathlib import Path

# --- CONFIGURATION ---
INPUT_DIR = Path(r"Q:\02_PROJECTS\1731_102024377_Gemeente_Westerwolde_2025\60_UM\LVL03")
OUTPUT_DIR = Path(r"D:\projects\Image_QC_GUI\2_Repos\OmniCloudMask\training\data\1731_102024377_Gemeente_Westerwolde_2025\images")
MAX_DIMENSION = 640  # The largest side will be resized to this
TARGET_FORMAT = ".tif"

# Valid extensions to look for in input
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".iiq"}

def resize_and_convert():
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get list of all files
    files = [f for f in INPUT_DIR.iterdir() if f.is_file()]
    
    count = 0
    errors = 0

    print(f"Starting process. Max dimension: {MAX_DIMENSION}px. Output format: {TARGET_FORMAT}")

    for file_path in files:
        # Check if valid image extension
        if file_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        try:
            # Read image
            img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"[SKIP] Could not read: {file_path.name}")
                errors += 1
                continue

            # Get current dimensions
            h, w = img.shape[:2]

            # Calculate scale to fit MAX_DIMENSION
            if h > w:
                # Height is the longest side
                scale = MAX_DIMENSION / h
                new_h = MAX_DIMENSION
                new_w = int(w * scale)
            else:
                # Width is the longest side (or square)
                scale = MAX_DIMENSION / w
                new_w = MAX_DIMENSION
                new_h = int(h * scale)

            # Resize
            # Use INTER_AREA for downsampling (shrinking), INTER_LINEAR for upscaling
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=interp)

            # Construct new filename with .tif extension
            new_filename = file_path.stem + TARGET_FORMAT
            dest_path = OUTPUT_DIR / new_filename

            # Save
            cv2.imwrite(str(dest_path), resized_img)
            
            print(f"[OK] {file_path.name} -> {new_w}x{new_h}")
            count += 1

        except Exception as e:
            print(f"[ERROR] processing {file_path.name}: {e}")
            errors += 1

    print("-" * 30)
    print(f"Processing complete.")
    print(f"Converted: {count}")
    print(f"Errors:    {errors}")

if __name__ == "__main__":
    resize_and_convert()
