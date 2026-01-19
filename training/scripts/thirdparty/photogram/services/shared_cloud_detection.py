# services/shared_cloud_detection.py

"""
Shared components for cloud and shadow detection
Cloud and Shadow Detection Systems for Factory Pattern
====================================================

Factory-compatible cloud and shadow detection systems using OmniCloudMask
technology with synthetic NIR generation.

Author: Shah Jawad Islam, Rana Sarkar
Version: 3.0.0 - Factory Pattern Compatible
"""
import sys
import os
from pathlib import Path

def get_resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and for PyInstaller.
    Handles the directory structure difference where 'third_party' exists in dev
    but its contents (NIRGAN, omnicloudmask) are flattened to root in _internal during freeze.
    """
    relative_path = Path(relative_path)
    
    if getattr(sys, 'frozen', False):
        # Running in PyInstaller bundle (_internal folder)
        base_path = Path(sys._MEIPASS)
        
        parts = list(relative_path.parts)
        
        # Check if path starts with 'third_party'
        if len(parts) > 0 and parts[0] == 'third_party':
            # Remove 'third_party' from the path components
            parts = parts[1:]
            
            # Handle OmniCloudMask rename (Folder is usually 'OmniCloudMask' in dev, 
            # but package/folder often becomes 'omnicloudmask' in PyInstaller root)
            if len(parts) > 0 and parts[0] == 'OmniCloudMask':
                parts[0] = 'omnicloudmask'
                
            # Reconstruct path relative to MEIPASS root
            relative_path = Path(*parts)
            
        return base_path / relative_path
    else:
        # Running in normal Python environment
        # Get the directory containing this file (services/)
        current_file_dir = Path(__file__).resolve().parent
        # Go up to reach src/ directory (services -> imageqc_cli -> src)
        base_path = current_file_dir.parent.parent
        return base_path / relative_path

import re
import csv
import cv2
import time
import logging
import warnings
import numpy as np
from PIL import Image, ImageDraw
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import torch
import urllib.request
import shutil

warnings.filterwarnings('ignore')

from photogram.imageio import ImageIOFactory
from photogram.entities.image_record import ImageRecord
from photogram.services.shared_utils import FileManager, AnalysisReporter, DetectionMode, ProcessingStatus, ProcessingResult

# -----------------------------------------------------------------------------
# CONDITIONAL IMPORTS FOR OMNICLOUDMASK
# -----------------------------------------------------------------------------
# In Dev: Located at src/third_party/OmniCloudMask/omnicloudmask
# In Exe: Located at _internal/omnicloudmask
if getattr(sys, 'frozen', False):
    try:
        from omnicloudmask.cloud_mask import predict_from_array
        from omnicloudmask.model_utils import load_model_from_weights
    except ImportError as e:
        # Fallback in case folder structure is preserved differently
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "omnicloudmask")))
        from omnicloudmask.cloud_mask import predict_from_array
        from omnicloudmask.model_utils import load_model_from_weights
else:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
    from third_party.OmniCloudMask.omnicloudmask.cloud_mask import predict_from_array
    from third_party.OmniCloudMask.omnicloudmask.model_utils import load_model_from_weights

# -----------------------------------------------------------------------------
# MODEL CONFIGURATION
# -----------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'regnety_004'

# get_resource_path will strip 'third_party' and lowercase 'OmniCloudMask' if frozen
ckpts_dir = get_resource_path(Path('third_party', 'OmniCloudMask', 'ckpts', 
                                   'PM_model_OCM_7.43_R_G_NIR_regnety_004.pycls_in1k_PT_state.safetensors'))

# -----------------------------------------------------------------------------
# MODEL DOWNLOAD HELPERS
# -----------------------------------------------------------------------------
# [INSTRUCTIONS]
# 1. Upload 'S2.ckpt' to a public Hugging Face model repo.
# 2. Copy the download link (it should contain 'resolve/main').
# 3. Paste it below.
# Example: "https://huggingface.co/username/repo_name/resolve/main/S2.ckpt"
NIRGAN_MODEL_URL = "https://huggingface.co/HawarIT/HIT_imageqc/resolve/main/S2.ckpt"
OCM_MODEL_URL = "https://huggingface.co/HawarIT/HIT_imageqc/resolve/main/PM_model_OCM_7.43_R_G_NIR_regnety_004.pycls_in1k_PT_state.safetensors"

def download_file(url: str, dest_path: Path):
    """
    Download a file from a URL to a destination path.
    Includes User-Agent header to satisfy Hugging Face security checks.
    """
    print(f"Downloading model from {url}...")
    print(f"Destination: {dest_path}")
    
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Hugging Face often requires a User-Agent
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) ImageQC/1.0'}
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req) as response:
            total_size = int(response.info().get('Content-Length', -1))
            chunk_size = 8192
            downloaded = 0
            
            with open(dest_path, 'wb') as out_file:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    
                    # Single-line progress update
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        mb_downloaded = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        sys.stdout.write(f"\r[Download] Progress: {percent:.1f}% ({mb_downloaded:.1f} MB / {mb_total:.1f} MB)")
                        sys.stdout.flush()
            
            # Clear the progress line and print completion
            sys.stdout.write("\n")
                        
        print(f"Download complete: {dest_path.name}")
        
    except Exception as e:
        sys.stdout.write("\n") # Ensure error prints on new line
        print(f"Failed to download {url}: {e}")
        if dest_path.exists():
            dest_path.unlink() # Remove partial file
        # Re-raise so the app knows something went wrong
        raise

# -----------------------------------------------------------------------------
# CHECK & DOWNLOAD MODELS (NIRGAN & OmniCloudMask)
# -----------------------------------------------------------------------------
# 1. NIRGAN Model
nirgan_ckpt_path = get_resource_path(Path("third_party", "NIRGAN", "ckpts", "S2.ckpt"))
if not nirgan_ckpt_path.exists():
    try:
        # Check if URL is still the default placeholder
        if "YOUR_USERNAME" in NIRGAN_MODEL_URL:
             print(f"[WARNING] NIRGAN model missing and NIRGAN_MODEL_URL is not configured.")
             print(f"Please upload S2.ckpt to Hugging Face and update NIRGAN_MODEL_URL in {__file__}")
        else:
             print(f"NIRGAN model missing at {nirgan_ckpt_path}. Attempting download...")
             download_file(NIRGAN_MODEL_URL, nirgan_ckpt_path)
    except Exception as e:
        print(f"Error downloading NIRGAN model: {e}")
        print("NIRGAN functionality will be disabled or fallback to synthetic generation.")

# 2. OmniCloudMask Model
# ocm_ckpt_path = get_resource_path(Path('third_party', 'OmniCloudMask', 'ckpts', 
#                                    'PM_model_OCM_7.43_R_G_NIR_regnety_004.pycls_in1k_PT_state.safetensors'))
# OCM path calculated via get_resource_path earlier
ocm_ckpt_path = ckpts_dir 

if not ocm_ckpt_path.exists():
    try:
        print(f"OmniCloudMask model missing at {ocm_ckpt_path}. Attempting download...")
        download_file(OCM_MODEL_URL, ocm_ckpt_path)
    except Exception as e:
        print(f"Error downloading OmniCloudMask model: {e}")
        # Note: If this fails, the subsequent load_model_from_weights call will likely fail too.

# Update ckpts_dir to point to the file we just ensured exists
# ckpts_dir = ocm_ckpt_path
# # # Print results
# models = []
# if model_files:
#     print("Found model files:")
#     for model_file in model_files:
#         match = re.search(r'NIR_([^.]+)\.', model_file.name)
#         model_name = match.group(1)
#         model = load_model_from_weights(
#                 model_name=model_name,
#                 weights_path=model_file,
#                 device=device,
#                 dtype=torch.float32,
#             )
#         models.append(model)
#     models = models[0]
# else:
#     from omnicloudmask.download_models import get_models
#     try:
#         get_models(model_version=3.0, source="hugging_face")
#         # print("OmnicloudMask Models downloaded successfully")
#     except Exception as e:
#         print(f"Error downloading models: {e}")
    

# load the single model from weights 
# weight_dir = Path('third_party', 'OmniCloudMask', 'ckpts', 'PM_model_OCM_7.43_R_G_NIR_edgenext_small.pt')
# model = torch.load(weight_dir, map_location='cpu', weights_only= False)
if ckpts_dir.exists():
    models = load_model_from_weights(
                    model_name=model_name,
                    weights_path=ckpts_dir,
                    device=device,
                    dtype=torch.float32,
                )
else:
    from omnicloudmask.download_models import get_models
    try:
        model_out_dir = get_resource_path(Path('third_party', 'OmniCloudMask', 'ckpts'))
        get_models(model_version=3.0, model_dir=model_out_dir, source="hugging_face")
        models = load_model_from_weights(
                    model_name=model_name,
                    weights_path=ckpts_dir,
                    device=device,
                    dtype=torch.float32,
                )
    except Exception as e:
        print(f"Error downloading models: {e}")


## load the jit model
# if getattr(sys, 'frozen', False):
#     weight_dir = get_resource_path('omnicloudmask/ckpts/PM_model_OCM_7.43_R_G_NIR_regnety_004.pycls_in1k_PT_state.pt')
# else:
#     weight_dir = get_resource_path('third_party/OmniCloudMask/ckpts/PM_model_OCM_7.43_R_G_NIR_regnety_004.pycls_in1k_PT_state.pt')
# omni_model = torch.jit.load(weight_dir, map_location=device)  
# omni_model.eval()
# models = omni_model.to(torch.float32).to(device)



# from pyproj import datadir
# # Automatically use the conda environment's PROJ data directory
# os.environ["PROJ_LIB"] = datadir.get_data_dir()
# os.environ["PROJ_DATA"] = datadir.get_data_dir()

# import fiona
# from shapely.geometry import Polygon, MultiPolygon, mapping, GeometryCollection
# from pyproj import CRS

# -----------------------------------------------------------------------------
# NIRGAN IMPORTS
# -----------------------------------------------------------------------------
try:
    from src.third_party.NIRGAN.create_NIR import get_NIR
    NIR_GAN_AVAILABLE = True
except ImportError:
    try:
        # Fallback for frozen app where NIRGAN is at root
        from NIRGAN.create_NIR import get_NIR
        NIR_GAN_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: NIR GAN library not available: {e}")
        print("Using synthetic NIR generation fallback.")
        NIR_GAN_AVAILABLE = False

from .detector_base import DetectorBase  # Import base for inheritance
from .detector_factory import DetectorFactory # Import DetectorFactory for registration


def downsample_image(image_array: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        Downsamples the image by a given scale factor using area interpolation (best for shrinking).
        
        Parameters:
            image_array (np.ndarray): Input image as H x W x C (or H x W for grayscale).
            scale_factor (float): Factor by which to downsample (e.g., 2.0 means 2× smaller).
            
        Returns:
            np.ndarray: Downsampled image.
        """
        try:
            if scale_factor <= 0:
                raise ValueError("Scale factor must be greater than 0.")
                
            original_height, original_width = image_array.shape[:2]
            new_width = int(original_width / scale_factor)
            new_height = int(original_height / scale_factor)

            resized = cv2.resize(
                image_array,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA  # Good for downsampling
            )
            return resized
        except ValueError as e:
            raise ValueError(f"Invalid scale factor or image shape in downsample_image: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error in downsample_image: {str(e)}") from e

class CloudShadowDetectorCore:
    """
    Core detection engine for cloud and shadow identification.
    Shared functionality for both cloud and shadow detectors.
    """
    
    def __init__(self, scale_factor: int = 2, use_gan_nir: bool = True, 
                 conf: float = 0.10, cloud_conf: Optional[float] = None, shadow_conf: Optional[float] = None, 
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Initialize detector core.
        
        Args:
            scale_factor: Scaling factor for image processing
            use_gan_nir: Whether to use GAN-based NIR generation
            conf: Default confidence threshold.
            cloud_conf: Confidence threshold for clouds. If None, `conf` is used.
            shadow_conf: Confidence threshold for shadows. If None, `conf` is used.
        """
        try:
            self.scale_factor = scale_factor
            self.use_gan_nir = use_gan_nir and NIR_GAN_AVAILABLE
            self.device = device
            self.conf = conf
            self.cloud_conf = cloud_conf if cloud_conf is not None else conf
            self.shadow_conf = shadow_conf if shadow_conf is not None else conf
            
            # Setup logger
            self.logger = logging.getLogger(self.__class__.__name__)
            self.io_factory = ImageIOFactory()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CloudShadowDetectorCore: {str(e)}") from e

    def mask_cloud_confidence(self,result, conf_thresh=0.25):
        """
        Mask based on merged cloud confidence (class 1+2), ignoring class 3.
        If merged cloud confidence >= threshold, mask=1 (cloud), else mask=0 (clear).
        """
        try:
            mask = np.argmax(result, axis=0)
            # result shape: (num_classes, H, W)
            conf_clear = result[0]
            conf_cloud = result[1] + result[2]   # Merge cloud classes
            # Remove class 3 (shadow) from consideration
            conf_sum = conf_clear + conf_cloud
            conf_sum = np.clip(conf_sum, 1e-8, None)  # prevent div by zero
            conf_clear_norm = conf_clear / conf_sum
            conf_cloud_norm = conf_cloud / conf_sum
            mask_out = np.where(conf_cloud_norm >= conf_thresh, 1, 0)

            # Mask of high-confidence cloud pixels
            high_conf_mask = conf_cloud_norm > conf_thresh
            avg_conf_cloud = (
                np.mean(conf_cloud_norm[high_conf_mask])
                if np.any(high_conf_mask)
                else 0.0
            )
            mask_out[mask==3] = 0 
            
            final_mask = mask_out.astype(np.uint8)
            
            if avg_conf_cloud == 0.0:
                avg_conf_cloud = np.max(conf_cloud_norm)    
            
            # # 1. Argmax class per pixel (H, W)
            # argmax_mask = np.argmax(result, axis=0)
            # # 2. Confidence of argmax class per pixel (H, W)
            # argmax_confidence = np.max(result, axis=0)
            # # 3. Final mask:
            # #    if class in {1, 2} and confidence > threshold → 1 else 0
            # final_mask = np.where(
            #     ((argmax_mask == 1) | (argmax_mask == 2)) &
            #     (argmax_confidence > conf_thresh),
            #     1,
            #     0
            # ).astype(np.uint8)
            
            # avg_conf_cloud = (
            #         argmax_confidence[final_mask == 1].mean()
            #         if np.any(final_mask == 1)
            #         else 0.0
            #     )
            return final_mask, avg_conf_cloud
        except ValueError as e:
            raise ValueError(f"Invalid mask dimensions or values in mask_cloud_confidence: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error in mask_cloud_confidence: {str(e)}") from e
    
    def mask_shadow_confidence_(self,result, conf_thresh=0.10):
        """
        Post-process model output to reassign low-confidence shadow predictions to clear class.

        Args:
            result (np.ndarray): Model output of shape (num_classes, H, W)
            conf_thresh (float): Confidence threshold below which shadow predictions are reassigned to clear

        Returns:
            np.ndarray: Mask with class indices after threshold-based refinement
        """
        try:
            # Compute argmax class mask (H, W)
            mask = np.argmax(result, axis=0)

            # Extract shadow confidence map (H, W)
            conf_shadow = result[3]

            # Find pixels predicted as shadow (class 3) but confidence is low
            # low_conf_shadow = (mask == 3) & (conf_shadow < conf_thresh)
            
            mask_out = np.where(conf_shadow >= conf_thresh, 3, 0)
            # Mask of high-confidence cloud pixels
            high_conf_mask = conf_shadow > conf_thresh
            avg_conf_shadow = (
                np.mean(conf_shadow[high_conf_mask])
                if np.any(high_conf_mask)
                else 0.0
            )
            mask_out[(mask == 1) | (mask == 2)] = 0
            
            final_mask = mask_out.astype(np.uint8) 
            
            if avg_conf_shadow == 0.0:
                avg_conf_shadow = np.max(conf_shadow)
            
            # Reassign these pixels to clear (class 0)
            # mask[low_conf_shadow] = 0
            # 1. Argmax class per pixel (H, W)
            # argmax_mask = np.argmax(result, axis=0)

            # # 2. Confidence of argmax class per pixel (H, W)
            # argmax_confidence = np.max(result, axis=0)

            # # 3. Final mask: keep class 3 only if confidence > threshold
            # final_mask = np.where(
            #     (argmax_mask == 3) & (argmax_confidence > conf_thresh),
            #     3,
            #     0
            # ).astype(np.uint8)
            # avg_conf_shadow = (
            #         argmax_confidence[final_mask == 3].mean()
            #         if np.any(final_mask == 3)
            #         else 0.0
            #     )
            # if avg_conf_shadow > conf_thresh:
            #     print(avg_conf_shadow)
            return final_mask, avg_conf_shadow
        
        except IndexError as e:
            raise IndexError(f"Index error in mask_shadow_confidence_, check array shapes: {str(e)}") from e
        except ValueError as e:
            raise ValueError(f"Invalid values in mask_shadow_confidence_: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error in mask_shadow_confidence_: {str(e)}") from e
    
    def _detect_single_resolution(self, image_array: np.ndarray, mode: DetectionMode, new_gsd: float) -> Tuple:
        """
        Internal method to perform single-resolution detection.
        
        Returns:
            Tuple: (cloud_mask, shadow_mask, cloud_px, shadow_px, avg_conf_cloud, avg_conf_shadow, has_cloud, cloud_area_sqm, cloud_prob, has_shadow, shadow_area_sqm, shadow_prob)
        """
        try:
            # Prepare channels
            red = image_array[:, :, 0].astype(np.float32) * self.scale_factor
            green = image_array[:, :, 1].astype(np.float32) * self.scale_factor

            # Generate NIR band
            if self.use_gan_nir:
                nir = get_NIR(image_array, self.device).astype(np.uint16)
            else:
                nir = self._generate_synthetic_nir(image_array)
            
            # Clip values
            red = np.clip(red, 0, 65535).astype(np.uint16)
            green = np.clip(green, 0, 65535).astype(np.uint16)
            
            # Stack bands for prediction
            if nir.shape != red.shape:
                nir = nir[:red.shape[0], :red.shape[1]]

            rgn_stack = np.stack([red, green, nir], axis=0)
            
            # Perform prediction
            mask_con = predict_from_array(rgn_stack, inference_device=self.device,
                                        export_confidence=True, custom_models=models)
            
            # Extract cloud and shadow masks
            cloud_mask, avg_conf_cloud = self.mask_cloud_confidence(mask_con, conf_thresh=self.cloud_conf)
            shadow_mask, avg_conf_shadow = self.mask_shadow_confidence_(mask_con, conf_thresh=self.shadow_conf)
            
            # Count pixels
            cloud_px = np.count_nonzero((cloud_mask == 1) | (cloud_mask == 2))
            shadow_px = np.count_nonzero(shadow_mask == 3)
            
            # Analyze masks for metrics
            pixel_area = self._calculate_pixel_area(new_gsd)
            has_cloud, cloud_area_sqm, cloud_prob, _, has_shadow, shadow_area_sqm = self._analyze_mask(
                cloud_mask | shadow_mask, pixel_area
            )
            
            return (cloud_mask, shadow_mask, cloud_px, shadow_px, avg_conf_cloud, avg_conf_shadow,
                   has_cloud, cloud_area_sqm, cloud_prob, has_shadow, shadow_area_sqm)
        except Exception as e:
            raise RuntimeError(f"Error in _detect_single_resolution: {str(e)}") from e
    
    def detect_from_array_dual_resolution(self, image_array: np.ndarray, mode: DetectionMode,
                                         resolution: Tuple[float, float], new_gsd: float) -> Tuple[bool, float, float, float, bool, float, np.ndarray]:
        """
        Perform dual-resolution cloud/shadow detection and merge results.
        
        Runs detection at two different downsampling factors and intelligently merges
        the results using resolution voting based on pixel counts.
        
        Args:
            image_array: RGB image array (H, W, 3)
            mode: Detection mode (CLOUD_ONLY, SHADOW_ONLY, or BOTH)
            resolution: Pixel resolution (x, y)
            new_gsd: Ground sampling distance after downsampling
            
        Returns:
            Tuple: (has_cloud, cloud_area_sqm, avg_conf_cloud, avg_conf_shadow, has_shadow, shadow_area_sqm, mask_final)
        """
        try:
            if image_array.ndim != 3 or image_array.shape[2] != 3:
                raise ValueError("Input must be (H, W, 3) RGB image")
            
            # First resolution: moderate downsampling (160x220 scale)
            scale_factor_1 = 3.0
            img_1 = downsample_image(image_array, scale_factor_1)
            gsd_1 = new_gsd * scale_factor_1
            
            (cloud_mask_1, shadow_mask_1, cloud_px_1, shadow_px_1,
             avg_conf_cloud_1, avg_conf_shadow_1,
             has_cloud_1, cloud_area_1, cloud_prob_1, has_shadow_1, shadow_area_1) = \
                self._detect_single_resolution(img_1, mode, gsd_1)
            
            # Second resolution: higher resolution (480x520 scale)
            scale_factor_2 = 1.5
            img_2 = downsample_image(image_array, scale_factor_2)
            gsd_2 = new_gsd * scale_factor_2
            
            (cloud_mask_2, shadow_mask_2, cloud_px_2, shadow_px_2,
             avg_conf_cloud_2, avg_conf_shadow_2,
             has_cloud_2, cloud_area_2, cloud_prob_2, has_shadow_2, shadow_area_2) = \
                self._detect_single_resolution(img_2, mode, gsd_2)
            
            # Upsample masks back to original resolution for merging
            h_orig, w_orig = image_array.shape[:2]
            h_1, w_1 = img_1.shape[:2]
            h_2, w_2 = img_2.shape[:2]
            
            # Upsample to original size
            cloud_mask_1_upsampled = cv2.resize(cloud_mask_1, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            shadow_mask_1_upsampled = cv2.resize(shadow_mask_1, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            cloud_mask_2_upsampled = cv2.resize(cloud_mask_2, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            shadow_mask_2_upsampled = cv2.resize(shadow_mask_2, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            
            # Merge predictions using intelligent resolution voting
            mask_final, primary_resolution = self.merge_dual_predictions(
                shadow_mask_1_upsampled, cloud_mask_1_upsampled, shadow_px_1, cloud_px_1,
                shadow_mask_2_upsampled, cloud_mask_2_upsampled, shadow_px_2, cloud_px_2
            )
            
            # Use results from primary resolution for confidence metrics
            if primary_resolution == 1:
                avg_conf_cloud = avg_conf_cloud_1
                avg_conf_shadow = avg_conf_shadow_1
            else:
                avg_conf_cloud = avg_conf_cloud_2
                avg_conf_shadow = avg_conf_shadow_2
            
            # Filter out very small detections
            non_zero_count = np.count_nonzero(mask_final)
            if non_zero_count <= 180:
                mask_final[:] = 0
            
            # Analyze final merged mask
            pixel_area = self._calculate_pixel_area(new_gsd)
            has_cloud, cloud_area_sqm, cloud_probability, shadow_probability, has_shadow, shadow_area_sqm = \
                self._analyze_mask(mask_final, pixel_area)
            
            return has_cloud, cloud_area_sqm, avg_conf_cloud, avg_conf_shadow, has_shadow, shadow_area_sqm, mask_final
        except Exception as e:
            raise RuntimeError(f"Error in detect_from_array_dual_resolution: {str(e)}") from e

    def detect_from_array(self, image_array: np.ndarray, mode: DetectionMode,
                         resolution: Tuple[float, float], new_gsd:float, use_dual_resolution: bool = False) -> Tuple[bool, float, float, float, bool, float, np.ndarray]:
        """
        Perform cloud/shadow detection on image array.
        
        Args:
            image_array: RGB image array (H, W, 3)
            mode: Detection mode (CLOUD_ONLY, SHADOW_ONLY, or BOTH)
            resolution: Pixel resolution (x, y)
            new_gsd: Ground sampling distance
            use_dual_resolution: If True, use dual-resolution detection with intelligent merging
            
        Returns:
            Tuple: (has_cloud, cloud_area_sqm, cloud_probability, has_shadow, shadow_area_sqm, mask)
        """
        try:
            # Use dual-resolution detection if requested
            if use_dual_resolution:
                return self.detect_from_array_dual_resolution(image_array, mode, resolution, new_gsd)
            
            if image_array.ndim != 3 or image_array.shape[2] != 3:
                raise ValueError("Input must be (H, W, 3) RGB image")
            
            # Prepare channels
            red = image_array[:, :, 0].astype(np.float32) * self.scale_factor
            green = image_array[:, :, 1].astype(np.float32) * self.scale_factor

            # Generate NIR band
            if self.use_gan_nir:
                nir = get_NIR(image_array, self.device).astype(np.uint16)
            else:
                nir = self._generate_synthetic_nir(image_array)
            
            # Clip values
            red = np.clip(red, 0, 65535).astype(np.uint16)
            green = np.clip(green, 0, 65535).astype(np.uint16)
            
            # Stack bands for prediction
            if nir.shape != red.shape:
                nir = nir[:red.shape[0], :red.shape[1]]

            rgn_stack = np.stack([red, green, nir], axis=0)
            
            # Perform prediction
            # mask_con = predict_from_array(rgn_stack, inference_device=self.device, export_confidence=True)
            mask_con = predict_from_array(rgn_stack,inference_device= self.device,
                                        export_confidence=True, custom_models= models)
            avg_conf_cloud, avg_conf_shadow = 0.0, 0.0
            if mode == DetectionMode.SHADOW_ONLY:
                mask_, avg_conf_shadow = self.mask_shadow_confidence_(mask_con, conf_thresh=self.shadow_conf)
            elif mode == DetectionMode.CLOUD_ONLY:
                mask_, avg_conf_cloud = self.mask_cloud_confidence(mask_con, conf_thresh=self.cloud_conf)
            elif mode == DetectionMode.BOTH:
                # Perform both cloud and shadow masking from single inference
                cloud_mask, avg_conf_cloud = self.mask_cloud_confidence(mask_con, conf_thresh=self.cloud_conf)
                shadow_mask, avg_conf_shadow = self.mask_shadow_confidence_(mask_con, conf_thresh=self.shadow_conf)
                
                # Combine masks: prioritize shadow over cloud, cloud over clear
                # The combined mask will have values: 0 (clear), 1/2 (cloud), 3 (shadow)
                # Where shadow takes precedence over cloud if both are detected for a pixel.
                mask_ = np.where(shadow_mask == 3, 3, cloud_mask)
            else: # Default for blur or other unrecognized modes
                mask_ = np.zeros_like(mask_con[0], dtype=np.uint8) # Return an empty mask
            non_zero_count = np.count_nonzero(mask_)
                
            # filterout less and small pixel cloud / shadow removed 
            if non_zero_count <= 180:
                mask_[:] = 0
            # debugpy.breakpoint()
            # Analyze mask
            has_cloud, cloud_area_sqm, cloud_probability, shadow_probability, has_shadow, shadow_area_sqm = self._analyze_mask(
                mask_, self._calculate_pixel_area(new_gsd)
            )
            
            # return has_cloud, cloud_area_sqm, cloud_probability, shadow_probability, has_shadow, shadow_area_sqm, mask_
            return has_cloud, cloud_area_sqm, avg_conf_cloud, avg_conf_shadow, has_shadow, shadow_area_sqm, mask_
        except ValueError as e:
            raise ValueError(f"Invalid input or processing error in detect_from_array: {str(e)}") from e
        except RuntimeError as e:
            raise RuntimeError(f"Prediction failed in detect_from_array: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error in detect_from_array: {str(e)}") from e
    
    def _generate_synthetic_nir(self, rgb_array: np.ndarray, method: str = "weighted") -> np.ndarray:
        """Generate synthetic NIR from RGB."""
        try:
            red = rgb_array[:, :, 0].astype(np.float32)
            green = rgb_array[:, :, 1].astype(np.float32)
            blue = rgb_array[:, :, 2].astype(np.float32)
            
            if method == "weighted":
                nir = 0.3 * red + 0.5 * green + 0.2 * blue
            elif method == "green_emphasis":
                nir = green * 1.2 - red * 0.2
            elif method == "ndvi_like":
                nir = (green + blue) / 2.0
            else:
                raise ValueError(f"Unsupported NIR generation method: {method}")
            
            nir = nir * self.scale_factor
            return np.clip(nir, 0, 65535).astype(np.uint16)
        except IndexError as e:
            raise IndexError(f"Invalid RGB array shape in _generate_synthetic_nir: {str(e)}") from e
        except ValueError as e:
            raise ValueError(f"Value error in _generate_synthetic_nir: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error in _generate_synthetic_nir: {str(e)}") from e
    
    def _calculate_pixel_area(self, new_gsd:float) -> float:
        """Calculate area per pixel in square meters."""
        try:
            # return abs(resolution[0] * resolution[1])
            return abs(new_gsd * new_gsd)
        except TypeError as e:
            raise TypeError(f"Invalid resolution types in _calculate_pixel_area: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error in _calculate_pixel_area: {str(e)}") from e
    
    def _analyze_mask(self, mask: np.ndarray, pixel_area: float) -> Tuple[bool, float, float, bool, float]:
        """
        Analyze prediction mask for cloud/shadow metrics.
        
        Mask values: 0=Clear, 1=Cloud, 2=Cloud, 3=Shadow
        """
        try:
            cloud_pixels = np.sum((mask == 1) | (mask == 2))
            shadow_pixels = np.sum(mask == 3)
            total_pixels = mask.size
            
            has_cloud = cloud_pixels > 0
            has_shadow = shadow_pixels > 0
            
            cloud_area_sqm = cloud_pixels * pixel_area
            shadow_area_sqm = shadow_pixels * pixel_area
            cloud_probability = cloud_pixels / total_pixels if total_pixels > 0 else 0.0
            shadow_probability = shadow_pixels / total_pixels if total_pixels > 0 else 0.0
            
            return has_cloud, cloud_area_sqm, cloud_probability, shadow_probability, has_shadow, shadow_area_sqm
        except ZeroDivisionError:
            return False, 0.0, 0.0, 0.0, False, 0.0
        except ValueError as e:
            raise ValueError(f"Invalid mask in _analyze_mask: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error in _analyze_mask: {str(e)}") from e
    
    def _create_visualization(self, rgb_array: np.ndarray, mask: np.ndarray,
                         output_path: Path, detection_mode: DetectionMode) -> None:
        """Create and save visualization image with minimal memory usage using OpenCV + Pillow."""

        try:
            # Ensure RGB format (uint8)
            if rgb_array.dtype != np.uint8:
                rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)

            h, w, _ = rgb_array.shape

            # Panel 1: True color
            panel1 = rgb_array.copy()

            # Panel 2: True color + contours
            panel2 = rgb_array.copy()
            
            new_mask = mask.copy()

            if detection_mode in [DetectionMode.CLOUD_ONLY, DetectionMode.BOTH]:
                cloud_mask = ((mask == 1) | (mask == 2)).astype(np.uint8)
                new_mask = cloud_mask.copy()
                new_mask[new_mask == 1] = 2  # Retain cloud areas if both
                contours, _ = cv2.findContours(cloud_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(panel2, contours, -1, (0, 0, 255), 1)  # Blue contours (BGR)

            if detection_mode in [DetectionMode.SHADOW_ONLY, DetectionMode.BOTH]:
                shadow_mask = (mask == 3).astype(np.uint8)
                new_mask = shadow_mask.copy()
                new_mask[new_mask == 1] = 3  # Retain cloud areas if both
                contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(panel2, contours, -1, (255, 0, 0), 1)  # Red contours (BGR)

            # Panel 3: Mask visualization
            cmap = {
                0: (0, 0, 0),      # Clear = black
                1: (255, 255, 255),# Cloud = white
                2: (255, 255, 255),# Cloud = white
                3: (128, 128, 128) # Shadow = grey
            }
            panel3 = np.zeros((h, w, 3), dtype=np.uint8)
            for k, v in cmap.items():
                panel3[new_mask == k] = v

            # Combine panels horizontally
            combined = np.concatenate([panel1, panel2, panel3], axis=1)

            # Convert to PIL for titles + legend
            img_pil = Image.fromarray(combined)
            draw = ImageDraw.Draw(img_pil)

            # Titles
            titles = ["True Colour", "True Colour + Prediction", "Prediction Mask"]
            step = w
            for i, title in enumerate(titles):
                draw.text((step * i + 10, 10), title, fill="yellow")

            # Legend
            legend_y = h - 30
            legends = [("Clear", "black")]
            if detection_mode in [DetectionMode.CLOUD_ONLY, DetectionMode.BOTH]:
                legends.append(("Cloud", "white"))
            if detection_mode in [DetectionMode.SHADOW_ONLY, DetectionMode.BOTH]:
                legends.append(("Shadow", "grey"))

            base_x = 2 * w + 40
            for i, (label, color) in enumerate(legends):
                x0 = base_x + i * 120
                draw.rectangle([x0, legend_y, x0 + 20, legend_y + 20], fill=color, outline="#2525B9", width=2)
                draw.text((x0 + 30, legend_y), label, fill="yellow")

            # Save final image
            img_pil.save(output_path, format="PNG", optimize=True)
        except ValueError as e:
            raise ValueError(f"Invalid array or mask in _create_visualization: {str(e)}") from e
        except OSError as e:
            raise OSError(f"Failed to save visualization: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error in _create_visualization: {str(e)}") from e             
    def create_visualization(self, rgb_array: np.ndarray, mask: np.ndarray,
                            output_path: Path, detection_mode: DetectionMode) -> None:
        """Create and save visualization image with minimal memory usage using OpenCV + Pillow."""

        try:
            # Ensure RGB format (uint8)
            if rgb_array.dtype != np.uint8:
                rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)

            h, w, _ = rgb_array.shape

            # Panel 2: True color + contours
            panel2 = rgb_array.copy()
            
            new_mask = mask.copy()

            if detection_mode in [DetectionMode.CLOUD_ONLY, DetectionMode.BOTH]:
                cloud_mask = ((mask == 1) | (mask == 2)).astype(np.uint8)
                new_mask = cloud_mask.copy()
                new_mask[new_mask == 1] = 2  # Retain cloud areas if both
                contours, _ = cv2.findContours(cloud_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(panel2, contours, -1, (0, 0, 255), 1)  # Blue contours (BGR)

            if detection_mode in [DetectionMode.SHADOW_ONLY, DetectionMode.BOTH]:
                shadow_mask = (mask == 3).astype(np.uint8)
                new_mask = shadow_mask.copy()
                new_mask[new_mask == 1] = 3  # Retain cloud areas if both
                contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(panel2, contours, -1, (255, 0, 0), 1)  # Red contours (BGR)

            # Panel 3: Mask visualization
            cmap = {
                0: (0, 0, 0),      # Clear = black
                1: (255, 255, 255),# Cloud = white
                2: (255, 255, 255),# Cloud = white
                3: (128, 128, 128) # Shadow = grey
            }
            panel3 = np.zeros((h, w, 3), dtype=np.uint8)
            for k, v in cmap.items():
                panel3[new_mask == k] = v

            # Combine panels horizontally (only panel2 + panel3)
            combined = np.concatenate([panel2, panel3], axis=1)

            # Convert to PIL for titles + legend
            img_pil = Image.fromarray(combined)
            draw = ImageDraw.Draw(img_pil)

            # Titles
            titles = ["True Colour + Prediction", "Prediction Mask"]
            step = w
            for i, title in enumerate(titles):
                draw.text((step * i + 10, 10), title, fill="yellow")

            # Legend
            legend_y = h - 30
            legends = [("Clear", "black")]
            if detection_mode in [DetectionMode.CLOUD_ONLY, DetectionMode.BOTH]:
                legends.append(("Cloud", "white"))
            if detection_mode in [DetectionMode.SHADOW_ONLY, DetectionMode.BOTH]:
                legends.append(("Shadow", "grey"))

            base_x = w + 40  # Adjusted since we only have 2 panels now
            for i, (label, color) in enumerate(legends):
                x0 = base_x + i * 120
                draw.rectangle([x0, legend_y, x0 + 20, legend_y + 20], fill=color, outline="#2525B9", width=2)
                draw.text((x0 + 30, legend_y), label, fill="yellow")

            # Save final image
            img_pil.save(output_path, format="PNG", optimize=True)
        except ValueError as e:
            raise ValueError(f"Invalid array or mask in create_visualization: {str(e)}") from e
        except OSError as e:
            raise OSError(f"Failed to save visualization: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error in create_visualization: {str(e)}") from e

    def merge_dual_predictions(self, shadow_mask_1, cloud_mask_1, shadow_px_1, cloud_px_1,
                               shadow_mask_2, cloud_mask_2, shadow_px_2, cloud_px_2):
        """
        Merge predictions from two different downsampling resolutions using intelligent decision logic.
        
        Args:
            shadow_mask_1, cloud_mask_1: Masks from first downsampling (160x220) - smaller
            shadow_px_1, cloud_px_1: Pixel counts from first downsampling
            shadow_mask_2, cloud_mask_2: Masks from second downsampling (480x520) - larger  
            shadow_px_2, cloud_px_2: Pixel counts from second downsampling
        
        Returns:
            tuple: (mask_final, primary_resolution)
                - mask_final: Final merged mask based on decision logic
                - primary_resolution: Which resolution was used (1 or 2) for final mask
        
        Decision Logic:
        1. For each class (shadow and cloud):
           - If both detect: use result with MORE pixels (higher detection confidence)
           - If only one detects: use the one that detected
           - If neither detect: no mask for that class
        2. Default to first downsampling (160x220) if counts are equal
        3. Track which resolution was primary for visualization alignment
        """
        resolution_votes = {1: 0, 2: 0}  # Track which resolution is used more
        
        # Process shadows
        shadow_detected_1 = shadow_px_1 > 0
        shadow_detected_2 = shadow_px_2 > 0
        shadow_source_res = 1  # Track which resolution was used
        
        if shadow_detected_1 and shadow_detected_2:
            # Both detected shadows: use the one with more pixels
            if shadow_px_2 > shadow_px_1:
                shadow_mask_final = shadow_mask_2
                shadow_source_res = 2
                resolution_votes[2] += 1
                detected_from = "both (using max pixels from 480x520)"
                used_px = shadow_px_2
            else:
                shadow_mask_final = shadow_mask_1
                shadow_source_res = 1
                resolution_votes[1] += 1
                detected_from = "both (using max pixels from 160x220)"
                used_px = shadow_px_1
        elif shadow_detected_1:
            # Only first resolution detected
            shadow_mask_final = shadow_mask_1
            shadow_source_res = 1
            resolution_votes[1] += 1
            detected_from = "first resolution (160x220)"
            used_px = shadow_px_1
        elif shadow_detected_2:
            # Only second resolution detected
            shadow_mask_final = shadow_mask_2
            shadow_source_res = 2
            resolution_votes[2] += 1
            detected_from = "second resolution (480x520)"
            used_px = shadow_px_2
        else:
            # Neither detected
            shadow_mask_final = np.zeros_like(shadow_mask_1, dtype=np.uint8)
            shadow_source_res = 1
            detected_from = "none"
            used_px = 0
        
        # Process clouds
        cloud_detected_1 = cloud_px_1 > 0
        cloud_detected_2 = cloud_px_2 > 0
        cloud_source_res = 1  # Track which resolution was used
        
        if cloud_detected_1 and cloud_detected_2:
            # Both detected clouds: use the one with more pixels
            if cloud_px_2 > cloud_px_1:
                cloud_mask_final = cloud_mask_2
                cloud_source_res = 2
                resolution_votes[2] += 1
            else:
                cloud_mask_final = cloud_mask_1
                cloud_source_res = 1
                resolution_votes[1] += 1
        elif cloud_detected_1:
            # Only first resolution detected
            cloud_mask_final = cloud_mask_1
            cloud_source_res = 1
            resolution_votes[1] += 1
        elif cloud_detected_2:
            # Only second resolution detected
            cloud_mask_final = cloud_mask_2
            cloud_source_res = 2
            resolution_votes[2] += 1
        else:
            # Neither detected
            cloud_mask_final = np.zeros_like(cloud_mask_1, dtype=np.uint8)
            cloud_source_res = 1
        
        # Combine shadows and clouds
        mask_final = shadow_mask_final | cloud_mask_final
        
        # Determine primary resolution for visualization (whichever was used more)
        primary_resolution = 2 if resolution_votes[2] > resolution_votes[1] else 1
        
        return mask_final, primary_resolution


class CloudShadowDetectorBase(DetectorBase):
    """
    Base class for cloud and shadow detection systems, inheriting from DetectorBase.
    This class eliminates repetition by handling common logic for initialization,
    image loading, detection running, and visualization saving. Subclasses specify
    the detection mode (CLOUD_ONLY or SHADOW_ONLY) and inherit all shared behavior.
    """
    
    def __init__(self, mode: DetectionMode, threshold: float = 0.01, output_dir: Optional[str] = None, 
                 scale_factor: int = 3, use_gan_nir: bool = True, 
                 save_visualizations: bool = True, **kwargs):
        """
        Initialize the base detection system.
        
        Args:
            mode: Detection mode (CLOUD_ONLY or SHADOW_ONLY)
            threshold: Detection threshold (probability for cloud, area for shadow)
            output_dir: Directory for saving outputs
            scale_factor: Image scaling factor for processing
            use_gan_nir: Use GAN for NIR generation
            save_visualizations: Whether to save visualization plots
            **kwargs: Additional configuration parameters
        """
        try:
            super().__init__(threshold, **kwargs)
            self.mode = mode
            self.output_dir = output_dir
            self.scale_factor = scale_factor
            self.save_visualizations = save_visualizations
            
            self.cloud_threshold = kwargs.get('cloud_threshold', threshold)
            self.cloud_shadow_threshold = kwargs.get('shadow_threshold', threshold)

            # Initialize core components
            self.detector_core = CloudShadowDetectorCore(scale_factor, use_gan_nir, conf=self.threshold,
                                                         cloud_conf=self.cloud_threshold,
                                                         shadow_conf=self.cloud_shadow_threshold)
            
            # Dual-resolution detection flag (can be enabled per run)
            self.use_dual_resolution = kwargs.get('use_dual_resolution', True)
            
            # Conditional Manager/Reporter Initialization
            if self.mode == DetectionMode.BOTH:
                self.cloud_fm = FileManager(output_dir, DetectionMode.CLOUD_ONLY, job_id=kwargs.get("job_id", "JOB_0001"))
                self.cloud_reporter = AnalysisReporter(self.cloud_fm)
                
                self.shadow_fm = FileManager(output_dir, DetectionMode.SHADOW_ONLY, job_id=kwargs.get("job_id", "JOB_0001"))
                self.shadow_reporter = AnalysisReporter(self.shadow_fm)
                
                # Assign one FM to self.file_manager for helper compatibility (e.g. get_output_path)
                self.file_manager = self.cloud_fm
                
                # Set self.reporter to one of them for general logging compatibility (e.g. load_image uses self.reporter.logger)
                self.reporter = self.cloud_reporter
                logger = self.cloud_reporter.logger
            else:
                self.file_manager = FileManager(output_dir, mode, job_id=kwargs.get("job_id", "JOB_0001"))
                self.reporter = AnalysisReporter(self.file_manager)
                logger = self.reporter.logger
            
            # Image processing variables
            self.image_path = None
            self.image_array = None
            self.resolution = None
            self.mask = None
            # self.original_gsd = 0.03  # Default GSD in meters (can be updated per image)
            self.original_gsd = kwargs.get("gsd", 0.03)  # Default GSD in meters (can be updated per image)
            
            # Log initialization details
            logger.debug(f"{self.__class__.__name__} initialized")
            if self.mode == DetectionMode.CLOUD_ONLY:
                logger.debug(f"Cloud probability threshold: {threshold}")
            elif self.mode == DetectionMode.SHADOW_ONLY:
                logger.debug(f"Shadow area threshold: {threshold}")
            logger.debug(f"Output folder: {self.file_manager.output_folder}")
            # logger.debug(f"CloudMask version: {omnicloudmask.__version__}")
            logger.debug(f"Using GAN NIR: {self.detector_core.use_gan_nir}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CloudShadowDetectorBase: {str(e)}") from e
    
    def load_image(self, image_path: str) -> None:
        """Load image for detection - required by DetectorBase."""
        try:
            self.image_path = Path(image_path)
            
            if not self.image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            rgb_array = self.detector_core.io_factory.read(image_path)
            metadata = self.detector_core.io_factory.metadata(image_path)
            
            self.image_array = rgb_array
            self.resolution = metadata.get("resolution", (1.0, 1.0))
            
            if self.image_array is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            self.reporter.logger.debug(f"Image loaded: {self.image_path.name} | Shape: {self.image_array.shape}")
        except FileNotFoundError as e:
            raise FileNotFoundError(str(e)) from e
        except ValueError as e:
            raise ValueError(str(e)) from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error in load_image: {str(e)}") from e
        

    
    def save_polygons_in_pixel_space(self,
                                    mask: np.ndarray,
                                    output_shapefile: Path,
                                    detection_mode: DetectionMode,
                                    image_name: str):
        """
        Extracts cloud/shadow polygons from a mask and saves them as MultiPolygon to a shapefile,
        with 'image_name' as an attribute, using local (pixel) coordinates.
        Appends new image entries to the shapefile if it already exists.
        """
        try:
            masks_to_extract = []

            if detection_mode in [DetectionMode.CLOUD_ONLY, DetectionMode.BOTH]:
                cloud_mask = ((mask == 1) | (mask == 2)).astype(np.uint8)
                masks_to_extract.append(cloud_mask)

            if detection_mode in [DetectionMode.SHADOW_ONLY, DetectionMode.BOTH]:
                shadow_mask = (mask == 3).astype(np.uint8)
                masks_to_extract.append(shadow_mask)

            all_polygons = []

            for binary_mask in masks_to_extract:
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    if cnt.shape[0] < 3:
                        continue

                    coords = cnt.squeeze().tolist()
                    if isinstance(coords[0], int):
                        coords = [coords]

                    polygon = Polygon(coords)
                    
                    # #adding buffer of 0 to fix potential invalid geometries like self-intersections
                    # # polygon = polygon.buffer(0)

                    # if not polygon.is_valid or polygon.area == 0:
                    #     continue

                    # all_polygons.append(polygon)
                    
                    cleaned = polygon.buffer(0)

                    # Ensure we only collect valid Polygon or MultiPolygon parts
                    if cleaned.is_empty:
                        continue
                    if isinstance(cleaned, GeometryCollection):
                        cleaned = [geom for geom in cleaned.geoms if isinstance(geom, (Polygon, MultiPolygon)) and not geom.is_empty]
                        for geom in cleaned:
                            if isinstance(geom, Polygon):
                                all_polygons.append(geom)
                            elif isinstance(geom, MultiPolygon):
                                all_polygons.extend([p for p in geom.geoms if not p.is_empty])
                                
                    elif isinstance(cleaned, Polygon):
                        all_polygons.append(cleaned)
                    elif isinstance(cleaned, MultiPolygon):
                        all_polygons.extend([p for p in cleaned.geoms if not p.is_empty])
                    else:
                        self.reporter.logger.warning(f"[WARNING] Skipping unsupported geometry type: {type(cleaned)}")
                    

            if not all_polygons:
                self.reporter.logger.info(f"[INFO] No valid polygons found for image: {image_name}")
                return

            multi = MultiPolygon(all_polygons)

            # Schema definition
            schema = {
                'geometry': 'MultiPolygon',
                'properties': {'image_name': 'str'}
            }

            # Dummy CRS (pixel space, placeholder)
            dummy_crs = CRS.from_epsg(28992)

            # Determine if file exists for appending
            mode = 'a' if output_shapefile.exists() else 'w'

            # When appending, schema and crs must match the existing file
            with fiona.open(output_shapefile, mode,
                            driver='ESRI Shapefile',
                            schema=schema,
                            crs=dummy_crs) as shp:
                shp.write({
                    'geometry': mapping(multi),
                    'properties': {'image_name': image_name}
                })

            self.reporter.logger.info(f" Saved polygons for '{image_name}' to {output_shapefile.resolve()}")
        except ImportError as e:
            raise ImportError(f"Missing dependencies for save_polygons_in_pixel_space (e.g., fiona, shapely): {str(e)}") from e
        except ValueError as e:
            raise ValueError(f"Invalid mask or geometry in save_polygons_in_pixel_space: {str(e)}") from e
        except OSError as e:
            raise OSError(f"Failed to write shapefile: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error in save_polygons_in_pixel_space: {str(e)}") from e
    
    def calculate_new_gsd(self,original_shape, new_shape, original_gsd):
        """
        Calculate the new Ground Sampling Distance (GSD) after resizing an aerial image.
        
        Parameters
        ----------
        original_shape : tuple
            (height, width) of the original image
        new_shape : tuple
            (height, width) of the resized image
        original_gsd : float
            Original ground sampling distance (e.g., meters per pixel)
        
        Returns
        -------
        float
            New ground sampling distance (same units as original_gsd)
        """
        # orig_height, orig_width = original_shape
        # new_height, new_width = new_shape
        orig_width, orig_height= original_shape
        new_width,new_height = new_shape

        # Compute scale ratio (assuming aspect ratio preserved)
        height_ratio = orig_height / new_height
        width_ratio = orig_width / new_width

        # They should be almost equal; use the average for safety
        scale_ratio = (height_ratio + width_ratio) / 2

        # New GSD is scaled by that ratio
        new_gsd = original_gsd * scale_ratio

        return new_gsd

    def run(self, image_record: ImageRecord) -> Dict:
        """
        Run detection system - required by DetectorBase.
        
        Performs detection, processes results based on mode, logs, and optionally saves visualizations.
        
        Returns:
            Dict: Mode-specific detection results
        """
        try:
            self.image_path = Path(image_record.path)
            self.image_array = image_record._data
            # self.resolution = image_record._metadata.get("resolution", (1.0, 1.0))
            self.resolution = [image_record.width, image_record.height] if image_record.width and image_record.height else (1.0, 1.0)
            
            # # Downsample image for faster processing
            if self.resolution[0] > self.resolution[1]:
                desired_resolution = (640, 480) # width, heitght  # Target resolution for the longer side
                down_factor =  self.resolution[1] / desired_resolution[1]
                self.image_array = downsample_image(self.image_array, down_factor)
            elif self.resolution[1] > self.resolution[0]:
                desired_resolution = (480, 640) # width, heitght  # Target resolution for the longer side
                down_factor =  self.resolution[0] / desired_resolution[0]
                self.image_array = downsample_image(self.image_array, down_factor)
            else:
                desired_resolution = self.resolution
                
            new_shape = self.image_array.shape[1], self.image_array.shape[0]
            new_gsd = self.calculate_new_gsd(self.resolution, new_shape, self.original_gsd)

            # self.resolution = [self.image_array.shape[1], self.image_array.shape[0]]
            
            if self.image_array is None:
                raise RuntimeError("No image loaded. Call load_image() first.")
            
            start_time = time.time()
            
            # Perform detection (computes both cloud and shadow metrics)
            # Use dual-resolution if enabled, otherwise single resolution
            (has_cloud, cloud_area_sqm, cloud_probability, shadow_probability,
                has_shadow, shadow_area_sqm, mask) = self.detector_core.detect_from_array(
                self.image_array, self.mode, self.resolution, new_gsd,
                use_dual_resolution=self.use_dual_resolution
            )
        
            
            self.mask = mask
            processing_time = time.time() - start_time
            
            # Determine threshold exceedance for cloud and shadow independently
            threshold_exceeded_cloud = cloud_probability > self.cloud_threshold
            threshold_exceeded_shadow = shadow_probability > self.cloud_shadow_threshold

            # Determine whether cloud/shadow are "detected" based on their respective flags and thresholds
            cloud_detected = has_cloud and threshold_exceeded_cloud
            shadow_detected = has_shadow and threshold_exceeded_shadow

            # Create ProcessingResult, populating all relevant metrics based on mode
            result = ProcessingResult(
                image_name=self.image_path.name,
                has_cloud=cloud_detected if self.mode in [DetectionMode.CLOUD_ONLY, DetectionMode.BOTH] else False,
                cloud_area_sqm=cloud_area_sqm if self.mode in [DetectionMode.CLOUD_ONLY, DetectionMode.BOTH] else 0.0,
                cloud_probability=cloud_probability if self.mode in [DetectionMode.CLOUD_ONLY, DetectionMode.BOTH] else 0.0,
                has_shadow=shadow_detected if self.mode in [DetectionMode.SHADOW_ONLY, DetectionMode.BOTH] else False,
                shadow_area_sqm=shadow_area_sqm if self.mode in [DetectionMode.SHADOW_ONLY, DetectionMode.BOTH] else 0.0,
                shadow_probability=shadow_probability if self.mode in [DetectionMode.SHADOW_ONLY, DetectionMode.BOTH] else 0.0,
                processing_time=processing_time,
                status=ProcessingStatus.SUCCESS
            )

            # Log results - Split reporting for BOTH mode
            if self.mode == DetectionMode.BOTH:
                # Log to Cloud Reporter
                cloud_res = ProcessingResult(
                    image_name=self.image_path.name,
                    has_cloud=cloud_detected,
                    cloud_area_sqm=cloud_area_sqm,
                    cloud_probability=cloud_probability,
                    processing_time=processing_time,
                    status=ProcessingStatus.SUCCESS
                )
                self.cloud_reporter.log_result(cloud_res)
                
                # Log to Shadow Reporter
                shadow_res = ProcessingResult(
                    image_name=self.image_path.name,
                    has_shadow=shadow_detected,
                    shadow_area_sqm=shadow_area_sqm,
                    shadow_probability=shadow_probability,
                    processing_time=processing_time,
                    status=ProcessingStatus.SUCCESS
                )
                self.shadow_reporter.log_result(shadow_res)
            else:
                # Standard single reporting
                self.reporter.log_result(result)

            # Determine if visualization should be saved
            save_vis_for_cloud = self.save_visualizations and cloud_detected and (self.mode in [DetectionMode.CLOUD_ONLY, DetectionMode.BOTH])
            save_vis_for_shadow = self.save_visualizations and shadow_detected and (self.mode in [DetectionMode.SHADOW_ONLY, DetectionMode.BOTH])
            
            cloud_mask_path = None
            shadow_mask_path = None
            mask_save_path = None

            if self.mode == DetectionMode.BOTH:
                # Save separately for unified detector
                if save_vis_for_cloud:
                    cloud_mask_path = self._save_visualization(result, override_mode=DetectionMode.CLOUD_ONLY)
                if save_vis_for_shadow:
                    shadow_mask_path = self._save_visualization(result, override_mode=DetectionMode.SHADOW_ONLY)
                # Use one as default/fallback for base dict
                mask_save_path = cloud_mask_path if cloud_mask_path else shadow_mask_path
            else:
                # Traditional single mode behavior
                if save_vis_for_cloud or save_vis_for_shadow:
                    mask_save_path = self._save_visualization(result)
                    # Assign to specific variables for consistency in dict update
                    if self.mode == DetectionMode.CLOUD_ONLY:
                        cloud_mask_path = mask_save_path
                    elif self.mode == DetectionMode.SHADOW_ONLY:
                        shadow_mask_path = mask_save_path

            # Build and return results dict
            base_dict = {
                "image_name": self.image_path.name,
                "processing_time": processing_time,
                "status": "success",
                "mask_path": str(mask_save_path) if mask_save_path else None
            }

            if self.mode in [DetectionMode.CLOUD_ONLY, DetectionMode.BOTH]:
                base_dict.update({
                    "cloud": cloud_detected,
                    "cloud_area_sqm": cloud_area_sqm,
                    "cloud_probability": cloud_probability,
                    "cloud_detected": cloud_detected,
                    "cloud_threshold_exceeded": threshold_exceeded_cloud,
                    "mask_path": str(cloud_mask_path) if cloud_mask_path else None,
                    "cloud_mask_path": str(cloud_mask_path) if cloud_mask_path else None
                })
            if self.mode in [DetectionMode.SHADOW_ONLY, DetectionMode.BOTH]: # Use if, not elif, to allow both
                base_dict.update({
                    "cloud_shadow": shadow_detected,
                    "shadow_area_sqm": shadow_area_sqm,
                    "shadow_probability": shadow_probability,
                    "shadow_detected": shadow_detected,
                    "shadow_threshold_exceeded": threshold_exceeded_shadow,
                    "mask_path": str(shadow_mask_path) if shadow_mask_path else None,
                    "shadow_mask_path": str(shadow_mask_path) if shadow_mask_path else None
                })
            return base_dict
            
        except Exception as e:
            processing_time = time.time() - start_time
            # Create error result (all metrics zeroed)
            error_result = ProcessingResult(
                image_name=self.image_path.name if self.image_path else "unknown",
                has_cloud=False,
                cloud_area_sqm=0.0,
                cloud_probability=0.0,
                has_shadow=False,
                shadow_area_sqm=0.0,
                processing_time=processing_time,
                status=ProcessingStatus.ERROR,
                error_message=str(e)
            )
            
            if self.mode == DetectionMode.BOTH:
                self.cloud_reporter.log_result(error_result)
                self.shadow_reporter.log_result(error_result)
            else:
                self.reporter.log_result(error_result)
            
            # Build and return mode-specific error dict
            base_error_dict = {
                "image_name": self.image_path.name if self.image_path else "unknown",
                "processing_time": processing_time,
                "status": "error",
                "error_message": str(e)
            }
            if self.mode in [DetectionMode.CLOUD_ONLY, DetectionMode.BOTH]:
                base_error_dict.update({
                    "cloud": False,
                    "cloud_area_sqm": 0.0,
                    "cloud_probability": 0.0,
                    "cloud_detected": False,
                    "cloud_threshold_exceeded": False
                })
            if self.mode in [DetectionMode.SHADOW_ONLY, DetectionMode.BOTH]: # Use if, not elif, to allow both
                base_error_dict.update({
                    "cloud_shadow": False,
                    "shadow_area_sqm": 0.0,
                    "shadow_probability": 0.0,
                    "shadow_detected": False,
                    "shadow_threshold_exceeded": False
                })
            return base_error_dict
    
    def _save_visualization(self, result: ProcessingResult, override_mode: Optional[DetectionMode] = None) -> Optional[Path]:
        """Save detection results and visualizations if mask is available."""
        try:
            if self.mask is None:
                return None
            
            # Use override mode if provided, else default to instance mode
            current_mode = override_mode if override_mode else self.mode
                
            output_path = self.file_manager.get_output_path(result.has_cloud, result.has_shadow, mode=current_mode)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp and stem for filename
            timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
            image_stem = self.image_path.stem
            vis_path_suffix = ""
            if current_mode == DetectionMode.BOTH:
                vis_path_suffix = "_cloud_shadow"
            else:
                # This handles CLOUD_ONLY -> _cloud and SHADOW_ONLY -> _shadow
                vis_path_suffix = f"_{current_mode.value}"
                
            vis_path = output_path / f"{image_stem}{vis_path_suffix}.jpg"
            
            self.detector_core.create_visualization(
                self.image_array, self.mask, vis_path, current_mode
            )
            
            return vis_path
        except OSError as e:
            raise OSError(f"Failed to create or save visualization path: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error in _save_visualization: {str(e)}") from e


@DetectorFactory.register_detector("cloud_and_shadow")
class CloudAndShadowDetector(CloudShadowDetectorBase):
    """
    A detector that performs both cloud and shadow detection in a single pass.
    Inherits from CloudShadowDetectorBase and is configured for DetectionMode.BOTH.
    """
    def __init__(self, **kwargs):
        super().__init__(mode=DetectionMode.BOTH, **kwargs)