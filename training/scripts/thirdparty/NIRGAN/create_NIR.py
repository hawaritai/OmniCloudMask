# Cleaned version of the NIRGAN create_NIR module
# NOTE: The public function signature get_NIR(image_array, device, img_name="generated_nir", output_dir=None)
#       is preserved exactly as requested.

import os
import sys
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from model.pix2pix import Px2Px_PL
# -----------------------------------------------------------------------------
# Resource Path Helper (PyInstaller + Dev Mode)
# -----------------------------------------------------------------------------
def get_resource_path(relative_path: str) -> Path:
    """
    Get absolute path to a resource inside the NIRGAN package.

    Works in:
    - Development environment
    - PyInstaller (sys._MEIPASS)
    """
    if getattr(sys, "frozen", False):
        base_path = Path(sys._MEIPASS) / "NIRGAN"  # PyInstaller extract dir
    else:
        base_path = Path(__file__).parent

    return base_path / relative_path

# Ensure local imports word
# -----------------------------------------------------------------------------
# Utility: Convert image to tensor
# -----------------------------------------------------------------------------
def to_tensor(img):
    """Convert a PIL Image or NumPy array to torch.FloatTensor in [0,1]."""
    if isinstance(img, np.ndarray):
        arr = img
    else:
        arr = np.array(img, dtype=np.float32)

    if arr.ndim == 2:  # grayscale
        arr = arr[:, :, None]

    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float() 
    return tensor

# -----------------------------------------------------------------------------
# Utility: Convert .iiq to NumPy Array
# -----------------------------------------------------------------------------
def iiq_to_image_array(iiq_path: str) -> np.ndarray:
    """Convert PhaseOne .iiq image to uint8 RGB array."""
    import rasterio

    with rasterio.open(iiq_path) as src:
        if src.count >= 3:
            data = src.read([1, 2, 3])
        else:
            data = src.read()

        data = np.transpose(data, (1, 2, 0))
        data = (data - data.min()) / (data.max() - data.min()) * 255.0

    return data.astype(np.uint8)

# -----------------------------------------------------------------------------
# Load JIT Model (Lazy Loading)
# -----------------------------------------------------------------------------
# torchscript_path = get_resource_path("ckpts/nir_S2_jit.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = torch.jit.load(torchscript_path, map_location=device)
# model.eval().to(device)

_model_instance = None

def _load_model():
    """Lazy load the NIRGAN model."""
    global _model_instance
    if _model_instance is not None:
        return _model_instance
        
    try:
        config_path = get_resource_path("configs/config_px2px.yaml")
        ckpt_path = get_resource_path("ckpts/S2.ckpt")
        
        if not ckpt_path.exists():
            raise FileNotFoundError(f"NIRGAN checkpoint not found at: {ckpt_path}")
            
        config = OmegaConf.load(config_path)
        model = Px2Px_PL(config)
        state_dict = torch.load(ckpt_path, map_location=device)['state_dict']
        model.load_state_dict(state_dict, strict=False)
        model.eval().to(device)
        _model_instance = model
        return model
    except Exception as e:
        print(f"Error loading NIRGAN model: {e}")
        raise

# -----------------------------------------------------------------------------
# Main Function (Signature preserved)
# -----------------------------------------------------------------------------
def get_NIR(image_array, device="cpu", img_name="generated_nir", output_dir=None):
    """
    Generate NIR band from RGB image.
    Returns a numpy array (H, W) scaled to [0, 10000].
    """
    # Ensure model is loaded
    model = _load_model()

    hr = to_tensor(image_array).unsqueeze(0)

    with torch.no_grad():
        hr = hr.to(device)
        pred_nir = model(hr)  # shape [1, 1, H, W]

    pred_nir = pred_nir.squeeze().cpu().to(torch.float16).detach().numpy()
    pred_nir = pred_nir * 10000

    # ----------------------------------
    # Optional Saving
    # ----------------------------------
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        base = os.path.splitext(img_name)[0]
        npz_path = os.path.join(output_dir, f"{base}_nir.npz")
        np.savez_compressed(npz_path, nir=pred_nir)

        # PNG preview
        normalized = (pred_nir - pred_nir.min()) / (pred_nir.max() - pred_nir.min())
        png_nir = (normalized * 255).astype(np.uint8)
        Image.fromarray(png_nir).save(os.path.join(output_dir, f"{base}_nir.png"))

    return pred_nir


if __name__ == "__main__":
    
    # Generate NIR for the selected image

    image_path= r"\\192.168.2.80\d\Projects\QI47\2025_Projects\Image_QC\1_Data\all_real_samples\test_fp\199000129_0385_01_0457_P00_01.iiq"
    output_folder = r"\\192.168.2.80\d\Projects\QI47\2025_Projects\Image_QC\3_Code\Repo\NIR-GAN\output" 

    img_name= os.path.splitext(os.path.basename(image_path))[0]
    #if the image extension is .iiq convert to image array
    if image_path.lower().endswith('.iiq'):
        img_arr= iiq_to_image_array(image_path)
    else:    
        img_arr= Image.open(image_path).convert("RGB")
        
    nir_value = get_NIR(img_arr, device).astype(np.uint16)
    print(nir_value)

    print("NIR generation complete.")