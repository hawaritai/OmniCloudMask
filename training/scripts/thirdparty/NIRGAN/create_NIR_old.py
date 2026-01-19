import sys, os
import time
os.environ["MPLBACKEND"] = "Agg"
# modules = [
#     "numpy",
#     "PIL.Image",
#     "omegaconf",
#     # "torchvision.transforms",
#     "pathlib",
# ]

# for mod in modules:
#     start = time.perf_counter()
#     __import__(mod)
#     end = time.perf_counter()
#     print(f"{mod:<25} loaded in {end - start:.4f} seconds")
    
    
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms
# import torch.nn.functional as F
from pathlib import Path 

# Add this helper function at the top of the file
def get_resource_path(relative_path):
    """
    Get absolute path to resource, works for both development and PyInstaller.
    
    In PyInstaller: Files are in _internal/ (e.g., _internal/NIRGAN/configs/...)
    In development: Files are in third_party/ (e.g., third_party/NIRGAN/configs/...)
    
    Args:
        relative_path: Path relative to NIRGAN root (e.g., 'configs/config.yaml')
                      NOT 'third_party/NIRGAN/configs/config.yaml'
    
    Returns:
        Absolute path to the resource
    """
    if getattr(sys, 'frozen', False):
        # Running in PyInstaller bundle
        # Files are at: _internal/NIRGAN/...
        base_path = Path(sys._MEIPASS) / 'NIRGAN'
    else:
        # Running in development
        # Files are at: third_party/NIRGAN/...
        base_path = Path(__file__).parent  # This is already third_party/NIRGAN/
    
    return base_path / relative_path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

# t0 = time.perf_counter()
from model.pix2pix import Px2Px_PL
# t1 = time.perf_counter()
# print("Px2Px_PL import:", t1 - t0, "seconds")
# ----------------------------------
# Configuration
# ----------------------------------
# input_folder = r"\\192.168.2.80\d\Projects\QI47\2025_Projects\Image_QC\3_Code\Repo\NIR-GAN\input\rgb\main_199001068_0386_01_0482_P00_01.png"   # <<-- Folder containing RGB images
# output_folder = r"\\192.168.2.80\d\Projects\QI47\2025_Projects\Image_QC\1_Data\Online\TEST"     # <<-- Folder to save output NIRs
# image_index = 0                   # <<-- Index of the image to process (0 = first image)



# ----------------------------------
# Utility: Load single RGB image
# ----------------------------------
def load_rgb_image(path):
    image = Image.open(path).convert("RGB")
    # transform = transforms.ToTensor()  # [0, 1], shape [C, H, W]
    # hr = transform(image).unsqueeze(0)
    hr = to_tensor(image).unsqueeze(0)
    return hr  # [1, 3, H, W]

# ----------------------------------
# Select image from folder
# ----------------------------------
# image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg','.tif' ,'.iiq'))])

# if not image_files:
#     raise FileNotFoundError(f"No image files found in '{input_folder}'")

# if image_index >= len(image_files):
#     raise IndexError(f"Image index {image_index} out of range. Found {len(image_files)} images.")

# image_name = image_files[image_index]
# image_path = os.path.join(input_folder, image_name)

# ----------------------------------
# Load and Predict
# ----------------------------------
# hr = load_rgb_image(image_path).to(device)

def iiq_to_image_array(iiq_path):
    ''' Convert .iiq image to numpy array '''
    #use rasterio to convert .iiq to .tif in memory
    import rasterio
    with rasterio.open(iiq_path) as src:
        #read 3 bands rgb if it has more than 3 bands
        if src.count >= 3:
            data = src.read([1,2,3])
        
        data = src.read()
        #convert to numpy array
        data = np.transpose(data, (1, 2, 0))  # Change from (C, H, W) to (H, W, C)
        #normalize to 0-255
        data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255.0
        data = data.astype(np.uint8)
        
    return data


 #config and checkpoint paths
# config_path = "configs/config_px2px.yaml"
# ckpt_path = "ckpts/S2.ckpt"
# config_path = "./shadow_cloud/Libraries/NIRGAN/configs/config_px2px.yaml"
# ckpt_path = "shadow_cloud/Libraries/NIRGAN/ckpts/S2.ckpt"
# config_path = './third_party/NIRGAN/configs/config_px2px.yaml'
# ckpt_path = './third_party/NIRGAN/ckpts/S2.ckpt'
# config_path = r"\\192.168.2.80\d\Projects\QI47\2025_Projects\Image_QC_GUI\towhid-rana-shadow\Image_QC_GUI\shadow_cloud\Libraries\NIRGAN\configs\config_px2px.yaml"
# ckpt_path = r"\\192.168.2.80\d\Projects\QI47\2025_Projects\Image_QC_GUI\towhid-rana-shadow\Image_QC_GUI\shadow_cloud\Libraries\NIRGAN\ckpts\S2.ckpt"
config_path = get_resource_path('configs/config_px2px.yaml')
if not config_path.exists():
    raise FileNotFoundError(
        f"Config file not found at: {config_path}\n"
        f"Frozen: {getattr(sys, 'frozen', False)}\n"
        f"sys._MEIPASS: {getattr(sys, '_MEIPASS', 'N/A')}"
    )
    
ckpt_path = get_resource_path("ckpts/S2.ckpt")

# Add a check to ensure the file is found, which helps in debugging
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Could not find config file at: {config_path}")
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Could not find checkpoint file at: {ckpt_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------
# Load Model
# ----------------------------------
# t0 = time.perf_counter()
config = OmegaConf.load(config_path)
model = Px2Px_PL(config)
state_dict = torch.load(ckpt_path, map_location=device)['state_dict']
model.load_state_dict(state_dict, strict=False)
model.eval().to(device)


# example_input = torch.randn(1, 3, 512, 512).to(device)
# scripted_model = torch.jit.trace(model.netG, example_input)
# torchscript_path =  get_resource_path("ckpts/nir_S2_jit.pt")
# torch.jit.save(scripted_model, torchscript_path )

# t1 = time.perf_counter()
# print("torchvision import:", t1 - t0, "seconds")

# t0 = time.perf_counter()

# torchscript_path =  get_resource_path("ckpts/nir_S2_jit.pt")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = torch.jit.load(torchscript_path, map_location=device)
# model.eval().to(device)

# t1 = time.perf_counter()
# print("model load with jit :", t1 - t0, "seconds")

def to_tensor(img):
    """Convert a PIL Image or numpy array to a Torch tensor in [0,1]."""
    if isinstance(img, np.ndarray):
        arr = img
    else:
        arr = np.array(img, dtype=np.float32)
    if arr.ndim == 2:  # grayscale
        arr = arr[:, :, None]
    tensor = torch.from_numpy(arr.transpose((2, 0, 1))).float() / 255.0
    return tensor

def get_NIR(image_array, device= "cpu",  img_name="generated_nir",output_dir=None):
    
    ''' Generate NIR band from a single RGB image array using a pre-trained model.
        Args:
            image_array (np.ndarray): Input RGB image as a NumPy array of shape (H, W, 3) with values in [0, 255].
            output_folder (str): Directory to save the output NIR image.
        Returns:
            np.ndarray: Generated NIR band as a NumPy array of shape (H, W) with values scaled to [0, 10000].
        '''
    
    transform = transforms.ToTensor()  # [0, 1], shape [C, H, W]
    hr= transform(image_array).unsqueeze(0)
    hr = to_tensor(image_array).unsqueeze(0)
   
    with torch.no_grad():
        hr = hr.to(device)  
        pred_nir = model(hr)  # Output: [1, 1, H, W]

    # ----------------------------------
    # Save Output
    # ----------------------------------
    # pred_nir = pred_nir.squeeze().cpu().to(torch.float16).numpy()
    pred_nir = pred_nir.squeeze().cpu().to(torch.float16).detach().numpy()

    pred_nir= pred_nir*10000  # Scale to typical NIR range
    
    # ----------------------------------
    # Save Output
    # ----------------------------------

    # Make output directory
    

    # Save as .npz
    #create dynamic name using counter
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        npz_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_nir.npz")
        np.savez_compressed(npz_path, nir=pred_nir)
        print(f"Saved NIR to: {npz_path}")

        # Optionally save NIR image as PNG
        normalized_nir = (pred_nir - pred_nir.min()) / (pred_nir.max() - pred_nir.min())
        png_nir = (normalized_nir * 255).astype(np.uint8)
        Image.fromarray(png_nir).save(os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_nir.png"))
        print(f"Saved preview PNG to: {output_dir}")
        print("Done.")
    
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