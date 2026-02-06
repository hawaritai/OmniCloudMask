import sys
import os
from pathlib import Path
import numpy as np
import torch
import cv2
import rasterio as rio
from functools import partial
import timm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import logging
import json
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, precision_recall_curve, average_precision_score
from safetensors.torch import load_file
from typing import List, Dict, Optional
import gc
import time

from custom_model_utils import build_custom_model, load_custom_weights

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- SETUP PATHS ---
current_script_dir = Path(__file__).parent.resolve()
training_dir = current_script_dir.parent
project_root = training_dir.parent

# Add paths for imports
if str(training_dir) not in sys.path:
    sys.path.append(str(training_dir))
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
if str(current_script_dir) not in sys.path:
    sys.path.append(str(current_script_dir))

# Import third-party modules
try:
    from thirdparty.NIRGAN.create_NIR import get_NIR
except ImportError as e:
    logger.warning(f"Import Error: {e}")
    # Try alternate path for NIRGAN
    try:
        sys.path.append(str(current_script_dir / "thirdparty"))
        from NIRGAN.create_NIR import get_NIR
    except ImportError as e2:
        logger.critical(f"Critical Import Error: {e2}")
        sys.exit(1)

# --- LOCAL CONFIG IMPORT ---
try:
    if str(current_script_dir) not in sys.path:
        sys.path.append(str(current_script_dir))
    from local_config import (
        TRAIN_DATA_DIR,
        BAND_ORDER,
        USE_DUAL_RES_METHOD,
        CUSTOM_MODEL_VERSION
    )
    
    # Try to import comparison specific configs if they exist, otherwise use defaults
    try:
        from local_config import (
            COMPARISON_QUICK_TEST,
            COMPARISON_QUICK_TEST_SAMPLES,
            COMPARISON_BATCH_SIZE,
            COMPARISON_USE_BF16,
            MODEL_CONFIG,
            INFERENCE_CONFIG
        )
    except ImportError:
        COMPARISON_QUICK_TEST = False
        COMPARISON_QUICK_TEST_SAMPLES = 50
        COMPARISON_BATCH_SIZE = 8
        COMPARISON_USE_BF16 = False
        MODEL_CONFIG = {
            "v4": {"model_library": "smp"},
            "v3": {"model_library": "fastai"}
        }
        INFERENCE_CONFIG = {
            "compile_models": False,
            "compile_mode": "default",
            "inference_dtype": "float32"
        }
        
except ImportError:
    logger.critical("CRITICAL: local_config.py not found. Please create 'training/scripts/local_config.py'.")
    sys.exit(1)

# --- CONSTANTS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATCH_SIZE = (509, 509)
NUM_CHANNELS = 3
CLASS_NAMES = ['Clear', 'Thick Cloud', 'Thin Cloud', 'Cloud Shadow']
GENERATE_SYNTHETIC_NIR = False # Set to False for comparison script as we assume processed data
MAX_SAVE_RETRIES = 3  # Maximum retry attempts for file operations
SAVE_RETRY_DELAY = 0.5  # Delay between retries in seconds


# --- MODEL TYPE MAPPINGS ---
# Mapping for v4 smp models (timm-unet style)
SMP_MODEL_TYPES = {
    'regnety': 'tu-regnety_004',
    'edgenext': 'tu-edgenext_small',
    'convnext': 'tu-convnextv2_nano',
}

# Mapping for v3 fastai models
FASTAI_MODEL_TYPES = {
    'regnety': 'regnety_004.pycls_in1k',
    'edgenext': 'edgenext_small.usi_in1k',
    'convnext': 'convnextv2_nano.fcmae_ft_in1k',
}


# --- OPTIONAL COMPILATION SUPPORT ---
# Optional: Import compilation support for v4 models
try:
    from omnicloudmask.model_utils import compile_torch_model
    HAS_COMPILATION = True
except ImportError:
    HAS_COMPILATION = False
    logger.warning("Model compilation not available (omnicloudmask.model_utils not found)")

# --- HELPER FUNCTIONS ---


def safe_file_save(
    save_func,
    *args,
    operation_name: str = "file save",
    max_retries: int = MAX_SAVE_RETRIES,
    retry_delay: float = SAVE_RETRY_DELAY,
    **kwargs
) -> bool:
    """
    Safely save a file with retry logic for Windows file locking issues.
    
    Args:
        save_func: Function to call for saving (e.g., save_file, torch.save)
        *args: Positional arguments for save_func
        operation_name: Description of operation for logging
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        **kwargs: Keyword arguments for save_func
        
    Returns:
        True if save was successful, False otherwise
    """
    for attempt in range(max_retries):
        try:
            # Force garbage collection before save
            gc.collect()
            
            # Small delay before save to allow file system to settle
            if attempt > 0:
                time.sleep(retry_delay * (attempt + 1))
            
            # Attempt the save
            save_func(*args, **kwargs)
            
            if attempt > 0:
                logger.info(f"     {operation_name} succeeded on attempt {attempt + 1}/{max_retries}")
            return True
            
        except OSError as e:
            error_code = getattr(e, 'winerror', None) or getattr(e, 'errno', None)
            
            # Error 1224: The requested operation cannot be performed on a file with a user-mapped section open
            # This is a Windows-specific file locking issue
            if error_code == 1224 or "user-mapped section" in str(e):
                logger.warning(f"    File lock detected (attempt {attempt + 1}/{max_retries}): {error_code}")
                
                if attempt < max_retries - 1:
                    # More retries available, try again
                    logger.debug(f"    Retrying {operation_name} in {retry_delay * (attempt + 1):.2f}s...")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    # No more retries
                    logger.error(f"     {operation_name} failed after {max_retries} attempts: {e}")
                    return False
            else:
                # Different OS error, don't retry
                logger.error(f"     {operation_name} failed: {e}")
                return False
        
        except Exception as e:
            logger.error(f"     {operation_name} failed with unexpected error: {e}")
            return False
    
    return False


def discover_model_checkpoints(ckpts_dir: Path) -> List[Dict]:
    """
    Discover model checkpoints in the ckpts directory with proper v4 smp support.
    
    Args:
        ckpts_dir: Path to the ckpts directory containing model checkpoints
        
    Returns:
        List of dictionaries with model information
    """
    model_configs = []
    
    if not ckpts_dir.exists():
        logger.error(f"Checkpoints directory not found: {ckpts_dir}")
        return model_configs
    
    # Look for .safetensors files (preferred)
    try:
        safetensors_files = list(ckpts_dir.glob("*best.safetensors"))
        if len(safetensors_files) == 0:
            safetensors_files = list(ckpts_dir.glob("*.safetensors"))
    except Exception as e:
        safetensors_files = list(ckpts_dir.glob("*.safetensors"))
    
    # # Look for .pth files
    # pth_files = list(ckpts_dir.glob("*.pth"))
    
    # Combine all checkpoint files
    all_checkpoints = safetensors_files #+ pth_files
    
    logger.info(f"Found {len(all_checkpoints)} checkpoint files in {ckpts_dir}")
    
    for checkpoint_path in all_checkpoints:
        # Extract model information from filename
        filename = checkpoint_path.stem
        
        # Determine model library (smp for v4, fastai for v3)
        model_library = "smp" if "smp" in filename.lower() or "v4" in filename.lower() else "fastai"
        
        # Determine model type based on library and filename
        model_type = None
        
        # Try to identify model architecture from filename
        if "regnety" in filename.lower():
            model_type = SMP_MODEL_TYPES['regnety'] if model_library == "smp" else FASTAI_MODEL_TYPES['regnety']
        elif "edgenext" in filename.lower():
            model_type = SMP_MODEL_TYPES['edgenext'] if model_library == "smp" else FASTAI_MODEL_TYPES['edgenext']
        elif "convnext" in filename.lower():
            model_type = SMP_MODEL_TYPES['convnext'] if model_library == "smp" else FASTAI_MODEL_TYPES['convnext']
        else:
            # Fallback default
            model_type = SMP_MODEL_TYPES['regnety'] if model_library == "smp" else FASTAI_MODEL_TYPES['regnety']
            logger.warning(f"Could not determine model type from {filename}, using default: {model_type}")
        
        model_configs.append({
            'name': filename,
            'path': checkpoint_path,
            'model_type': model_type,
            'model_library': model_library
        })
    
    return model_configs


def verify_safetensors_integrity(safetensor_path: Path, model: torch.nn.Module) -> bool:
    """
    Verify that the saved safetensors file can be loaded and matches the model structure.
    
    Args:
        safetensor_path: Path to the safetensors file
        model: The model to compare against
        
    Returns:
        True if integrity check passes, False otherwise
    """
    try:
        logger.info("  Verifying safetensors file integrity...")
        loaded_state = load_file(safetensor_path)
        model_state = model.state_dict()
        
        # Check key count
        if len(loaded_state) != len(model_state):
            logger.error(f"  Key count mismatch: loaded={len(loaded_state)}, model={len(model_state)}")
            return False
        
        # Check all keys exist
        for key in model_state.keys():
            if key not in loaded_state:
                logger.error(f"  Missing key in safetensors: {key}")
                return False
        
        # Check tensor shapes
        for key in model_state.keys():
            if loaded_state[key].shape != model_state[key].shape:
                logger.error(f"  Shape mismatch for {key}: loaded={loaded_state[key].shape}, model={model_state[key].shape}")
                return False
        
        logger.info("   Safetensors integrity check passed")
        return True
        
    except Exception as e:
        logger.error(f"  Safetensors integrity check failed: {e}")
        return False


# --- MODEL CONFIGURATIONS ---
# Define models to compare here
model_configs = []

# Auto-discover models from models directory using the improved discover_model_checkpoints function
# models_dir = project_root / "ckpts"
fine_tuned_models_dir = Path(r"D:\projects\Image_QC_GUI\2_Repos\OmniCloudMask\fine_tuning_results_OCM_test1x_kavel_n_cloudsen_v9.1\models")
bsae_model_dir = Path(r"D:\projects\Image_QC_GUI\2_Repos\OmniCloudMask\ckpts")

if fine_tuned_models_dir.exists() or bsae_model_dir.exists():
    discovered_fined_tuned_models = discover_model_checkpoints(fine_tuned_models_dir)
    discovered_models = discover_model_checkpoints(bsae_model_dir)
    model_configs.extend(discovered_fined_tuned_models)
    model_configs.extend(discovered_models)

# Optional: Add additional manual model configurations
# Example:
# model_configs.append(
#     {
#         'name': 'Tuned OCM 7.43_v4',
#         'path': Path(r"D:\Projects\QI47\2025_Projects\Image_QC_GUI\2_Repo\OmniCloudMask\models\PM_model_OCM_7.43_R_G_NIR_test_4_regnety_004.pycls_in1k_PT_state.safetensors"),
#         'model_type': 'regnety_004.pycls_in1k',
#         'model_library': 'fastai' # Assuming this specific file is legacy based on name. Change to 'smp' if it's a V4 training.
#     }
# )

class ModelComparator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = self.output_dir / "metrics"
        self.plots_dir = self.output_dir / "plots"
        self.metrics_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        
        # Setup logging to file
        file_handler = logging.FileHandler(self.output_dir / "model_comparison.log")
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

    def load_model(self, model_config):
        """
        Load a model checkpoint using the modern loading mechanism.
        
        Args:
            model_config: Dictionary containing model information
            
        Returns:
            Loaded model or None if failed
        """
        model_name = model_config['name']
        checkpoint_path = model_config['path']
        model_type = model_config['model_type']
        model_library = model_config.get('model_library', 'smp')
        
        # Handle relative paths
        if not checkpoint_path.exists():
            # Try finding it relative to project root
            checkpoint_path = project_root / checkpoint_path
            
        if not checkpoint_path.exists():
            logger.error(f"Model file not found: {checkpoint_path}, skipping...")
            return None

        logger.info(f"Loading model: {model_name} from {checkpoint_path}")
        logger.info(f"  Model type: {model_type} ({model_library})")
        
        try:
            # Create model architecture using build_custom_model
            model = build_custom_model(
                model_name=model_type,
                model_library=model_library,
                in_chans=NUM_CHANNELS,
                n_out=len(CLASS_NAMES)
            )
            
            # Load weights with device parameter
            load_custom_weights(model, checkpoint_path, device=DEVICE, strict=False)
            logger.info("  Successfully loaded checkpoint weights.")
            
            # Set model to eval mode for inference
            model.eval()
            logger.info("  Model set to eval mode for inference.")
            
            # Apply bfloat16 if requested
            if COMPARISON_USE_BF16 and DEVICE.type == 'cuda':
                model = model.bfloat16()
                logger.info("  Model converted to bfloat16.")
            
            # Optional compilation for v4 models
            if INFERENCE_CONFIG.get("compile_models", False) and model_library == 'smp' and HAS_COMPILATION:
                logger.info("  Compiling model for v4 smp architecture...")
                model = compile_torch_model(
                    model,
                    patch_size=MODEL_PATCH_SIZE[0],
                    batch_size=COMPARISON_BATCH_SIZE,
                    dtype=torch.bfloat16 if COMPARISON_USE_BF16 else torch.float32,
                    device=DEVICE,
                    compile_mode=INFERENCE_CONFIG.get("compile_mode", "default")
                )
                logger.info("  Model compiled successfully.")
            
            # Verify safetensors integrity if applicable
            if checkpoint_path.suffix == '.safetensors':
                model_cpu = model.to("cpu").float()
                integrity_valid = verify_safetensors_integrity(checkpoint_path, model_cpu)
                if not integrity_valid:
                    logger.error(f"  CRITICAL: Safetensors file failed integrity check: {checkpoint_path.name}")
                    return None
                # Restore to original device and dtype
                if COMPARISON_USE_BF16 and DEVICE.type == 'cuda':
                    model = model_cpu.to(DEVICE).bfloat16()
                else:
                    model = model_cpu.to(DEVICE)
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_validation_data(self):
        """
        Scan TRAIN_DATA_DIR/validation for images and labels with improved error handling.
        Supports both structure types:
        1. Separate folders (images/, labels/ or masks/)
        2. Colocated files (_image.tif, _label.tif)
        
        Returns:
            Tuple of (image_files, label_files) or ([], []) on failure
        """
        val_dir = TRAIN_DATA_DIR / "test"
        if not val_dir.exists():
            logger.error(f"Validation directory not found: {val_dir}")
            return [], []

        image_files = []
        label_files = []

        # Strategy 1: Colocated _image.tif and _label.tif
        candidates = list(val_dir.glob("*_image.tif"))
        if candidates:
            logger.info(f"Detected colocated dataset structure in {val_dir}")
            for img_path in candidates:
                lbl_path = val_dir / img_path.name.replace("_image.tif", "_label.tif")
                if lbl_path.exists():
                    image_files.append(img_path)
                    label_files.append(lbl_path)
                else:
                    logger.warning(f"  No matching label found for {img_path.name}")
        
        # Strategy 2: Standard folder structure or just .tif files
        if not image_files:
            logger.info("Checking for standard dataset structure or fallback...")
            # Try to find images folder inside validation
            img_dir = val_dir
            if (val_dir / "images").exists():
                img_dir = val_dir / "images"
            
            # Find labels folder
            lbl_dir = val_dir
            if (val_dir / "masks").exists():
                lbl_dir = val_dir / "masks"
            elif (val_dir / "labels").exists():
                lbl_dir = val_dir / "labels"
            
            # Get all TIFs that are not labels
            all_tifs = list(img_dir.glob("*.tif"))
            possible_images = [p for p in all_tifs if "_label.tif" not in p.name and "mask" not in p.name.lower()]
            
            for img_path in possible_images:
                # Try to find matching label
                lbl_candidates = [
                    lbl_dir / img_path.name,
                    lbl_dir / f"{img_path.stem}_label{img_path.suffix}",
                    lbl_dir / f"label_{img_path.name}",
                    lbl_dir / img_path.name.replace("image", "label")
                ]
                
                found_label = False
                for lbl_path in lbl_candidates:
                    if lbl_path.exists() and lbl_path != img_path:
                        image_files.append(img_path)
                        label_files.append(lbl_path)
                        found_label = True
                        break
                
                if not found_label:
                    logger.warning(f"  No matching label found for {img_path.name}")

        logger.info(f"Found {len(image_files)} validation image-label pairs")
        
        if len(image_files) == 0:
            logger.error("No valid image-label pairs found in validation directory")
            return [], []
        
        if COMPARISON_QUICK_TEST and len(image_files) > COMPARISON_QUICK_TEST_SAMPLES:
            logger.info(f"Quick test mode: limiting to {COMPARISON_QUICK_TEST_SAMPLES} samples")
            indices = np.random.choice(len(image_files), COMPARISON_QUICK_TEST_SAMPLES, replace=False)
            image_files = [image_files[i] for i in indices]
            label_files = [label_files[i] for i in indices]
            
        return image_files, label_files

    def preprocess_batch(self, image_paths, label_paths):
        """
        Preprocess a batch of images and labels with improved error handling.
        
        Args:
            image_paths: List of image file paths
            label_paths: List of label file paths
            
        Returns:
            Tuple of (images_tensor, labels_tensor) or (None, None) on failure
        """
        images = []
        labels = []
        failed_count = 0
        
        for img_path, lbl_path in zip(image_paths, label_paths):
            try:
                # Load Image
                with rio.open(img_path) as src:
                    raw_bands = src.read(BAND_ORDER) # (C, H, W)
                 
                img_hwc = np.transpose(raw_bands, (1, 2, 0))
                
                # Resize
                img_final = cv2.resize(img_hwc, MODEL_PATCH_SIZE, interpolation=cv2.INTER_AREA)
                
                # Stack bands
                if GENERATE_SYNTHETIC_NIR:
                     # NIRGAN logic skipped for brevity/speed as per user request to keep simple
                     # Assuming inputs are already R,G,NIR or don't need GAN for this eval
                     pass
                
                rgn_stack = np.transpose(img_final, (2, 0, 1)).astype(np.float32)
                
                # Normalize (Z-Score)
                tensor_img = torch.from_numpy(rgn_stack).float()
                mean = tensor_img.mean(dim=(1, 2), keepdim=True)
                std = tensor_img.std(dim=(1, 2), keepdim=True) + 1e-6
                tensor_img = (tensor_img - mean) / std
                
                # Check for NaN or Inf in normalized image
                if torch.isnan(tensor_img).any() or torch.isinf(tensor_img).any():
                    logger.warning(f"  Invalid values in normalized image {img_path.name}, skipping")
                    failed_count += 1
                    continue
                
                # Load Label
                with rio.open(lbl_path) as src_lbl:
                    lbl = src_lbl.read(1)
                
                # Resize label
                lbl_resized = cv2.resize(lbl, MODEL_PATCH_SIZE, interpolation=cv2.INTER_NEAREST)
                tensor_lbl = torch.from_numpy(lbl_resized).long()
                
                # Check label validity
                if tensor_lbl.min() < 0 or tensor_lbl.max() >= len(CLASS_NAMES):
                    logger.warning(f"  Invalid label values in {lbl_path.name}, skipping")
                    failed_count += 1
                    continue
                
                images.append(tensor_img)
                labels.append(tensor_lbl)
                
            except Exception as e:
                logger.warning(f"  Error reading pair {img_path.name}: {e}")
                failed_count += 1
                continue
                
        if not images:
            logger.warning(f"  No valid images in batch (failed: {failed_count}/{len(image_paths)})")
            return None, None
        
        if failed_count > 0:
            logger.warning(f"  Failed to process {failed_count}/{len(image_paths)} images")
            
        return torch.stack(images), torch.stack(labels)

    def evaluate_model(self, model, image_files, label_files):
        """
        Evaluate model on validation data with improved error handling.
        
        Args:
            model: The loaded model to evaluate
            image_files: List of image file paths
            label_files: List of label file paths
            
        Returns:
            Tuple of (predictions, targets, probabilities) or (None, None, None) on failure
        """
        all_preds = []
        all_targets = []
        all_probs = []
        processed_count = 0
        failed_count = 0
        
        # Batch processing
        num_batches = (len(image_files) + COMPARISON_BATCH_SIZE - 1) // COMPARISON_BATCH_SIZE
        logger.info(f"Processing {len(image_files)} images in {num_batches} batches (batch_size={COMPARISON_BATCH_SIZE})")
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * COMPARISON_BATCH_SIZE
                end_idx = min((i + 1) * COMPARISON_BATCH_SIZE, len(image_files))
                
                batch_imgs = image_files[start_idx:end_idx]
                batch_lbls = label_files[start_idx:end_idx]
                
                try:
                    xb, yb = self.preprocess_batch(batch_imgs, batch_lbls)
                    
                    if xb is None:
                        logger.warning(f"  Batch {i+1}/{num_batches} failed preprocessing (no valid images)")
                        failed_count += len(batch_imgs)
                        continue
                    
                    xb = xb.to(DEVICE)
                    if COMPARISON_USE_BF16 and DEVICE.type == 'cuda':
                        xb = xb.bfloat16()
                    
                    out = model(xb)
                    probs = torch.softmax(out, dim=1).cpu().numpy()
                    preds = torch.argmax(out, dim=1).cpu().numpy()
                    
                    all_probs.append(probs)
                    all_preds.append(preds)
                    all_targets.append(yb.numpy())
                    
                    processed_count += len(batch_imgs)
                    
                    if (i + 1) % 5 == 0:
                        logger.info(f"  Processed {processed_count}/{len(image_files)} images")
                
                except Exception as e:
                    logger.error(f"  Error processing batch {i+1}/{num_batches}: {e}")
                    failed_count += len(batch_imgs)
                    continue
        
        if not all_preds:
            logger.error(f"  CRITICAL: All batches failed. Processed: {processed_count}, Failed: {failed_count}")
            return None, None, None
        
        logger.info(f"  Evaluation complete. Processed: {processed_count}, Failed: {failed_count}")
        
        return np.concatenate(all_preds), np.concatenate(all_targets), np.concatenate(all_probs)

    def compute_metrics(self, y_true, y_pred, y_probs):
        """
        Compute evaluation metrics with improved error handling.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_probs: Prediction probabilities
            
        Returns:
            Dictionary of metrics
        """
        try:
            # Flatten
            y_true_flat = y_true.flatten()
            y_pred_flat = y_pred.flatten()
            
            # Filter valid classes
            valid_mask = (y_true_flat >= 0) & (y_true_flat < len(CLASS_NAMES))
            y_true_valid = y_true_flat[valid_mask]
            y_pred_valid = y_pred_flat[valid_mask]
            
            metrics = {}
            
            # Overall Accuracy
            metrics['Accuracy'] = float(accuracy_score(y_true_valid, y_pred_valid))
            
            # Per-class metrics
            precision = precision_score(y_true_valid, y_pred_valid, average=None, labels=range(len(CLASS_NAMES)), zero_division=0)
            recall = recall_score(y_true_valid, y_pred_valid, average=None, labels=range(len(CLASS_NAMES)), zero_division=0)
            f1 = f1_score(y_true_valid, y_pred_valid, average=None, labels=range(len(CLASS_NAMES)), zero_division=0)
            iou = jaccard_score(y_true_valid, y_pred_valid, average=None, labels=range(len(CLASS_NAMES)), zero_division=0)
            
            metrics['Mean_IoU'] = float(np.mean(iou))
            metrics['Mean_F1'] = float(np.mean(f1))
            metrics['Weighted_Precision'] = float(precision_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0))
            metrics['Weighted_Recall'] = float(recall_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0))
            
            # Per class dicts
            metrics['per_class'] = {}
            
            # PR Curve data (requires probabilities)
            # Reshape probs: (B, C, H, W) -> (B*H*W, C)
            y_probs_flat = y_probs.transpose(0, 2, 3, 1).reshape(-1, len(CLASS_NAMES))
            
            metrics['pr_data'] = {}
            avg_aps = []
            
            for i, name in enumerate(CLASS_NAMES):
                cls_metrics = {
                    'IoU': float(iou[i]),
                    'F1': float(f1[i]),
                    'Precision': float(precision[i]),
                    'Recall': float(recall[i])
                }
                
                # PR Curve for this class
                # CRITICAL FIX: Filter probabilities per-class based on valid labels
                # Only include pixels where true label is valid (not invalid/out-of-range)
                class_valid_mask = (y_true_flat >= 0) & (y_true_flat < len(CLASS_NAMES))
                
                # Apply class-specific mask to probabilities and labels
                y_probs_class_valid = y_probs_flat[class_valid_mask]
                y_true_class_valid = y_true_flat[class_valid_mask]
                
                # Binarize true labels for this class
                y_true_cls = (y_true_class_valid == i).astype(int)
                y_score_cls = y_probs_class_valid[:, i]
                
                try:
                    prec, rec, _ = precision_recall_curve(y_true_cls, y_score_cls)
                    ap = average_precision_score(y_true_cls, y_score_cls)
                except Exception as e:
                    logger.warning(f"  Error computing PR curve for class {name}: {e}")
                    # Set default values if PR curve computation fails
                    prec = np.array([1.0, 0.0])
                    rec = np.array([0.0, 1.0])
                    ap = 0.0
                
                cls_metrics['AP'] = float(ap)
                avg_aps.append(ap)
                
                # Downsample PR curve for saving to JSON (too large otherwise)
                # Use 100 points with linear interpolation for smoother curves
                if len(prec) > 100:
                    recall_interp = np.linspace(0, 1, 100)
                    # Reverse arrays for interpolation (recall is decreasing in sklearn output)
                    prec_interp = np.interp(recall_interp, rec[::-1], prec[::-1])
                    metrics['pr_data'][name] = {
                        'precision': prec_interp.tolist(),
                        'recall': recall_interp.tolist(),
                        'ap': float(ap)
                    }
                else:
                    metrics['pr_data'][name] = {
                        'precision': prec.tolist(),
                        'recall': rec.tolist(),
                        'ap': float(ap)
                    }
                    
                metrics['per_class'][name] = cls_metrics
                
            metrics['Mean_AP'] = float(np.mean(avg_aps))
            
            # Confusion Matrix (normalized for visualization)
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true_valid, y_pred_valid, labels=range(len(CLASS_NAMES)), normalize='true')
            metrics['confusion_matrix'] = cm.tolist()
            
            # Raw Confusion Matrix (for detailed analysis)
            cm_raw = confusion_matrix(y_true_valid, y_pred_valid, labels=range(len(CLASS_NAMES)), normalize=None)
            metrics['confusion_matrix_raw'] = cm_raw.tolist()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            import traceback
            traceback.print_exc()
            # Return empty metrics on failure
            return {
                'Accuracy': 0.0,
                'Mean_IoU': 0.0,
                'Mean_F1': 0.0,
                'Mean_AP': 0.0,
                'Weighted_Precision': 0.0,
                'Weighted_Recall': 0.0,
                'per_class': {},
                'pr_data': {},
                'confusion_matrix': [],
                'confusion_matrix_raw': []
            }

    def plot_comparisons(self, all_metrics):
        """
        Generate comparison plots with improved error handling.
        
        Args:
            all_metrics: Dictionary of metrics for each model
        """
        try:
            # 1. Metrics Bar Chart
            df_rows = []
            for model_name, m in all_metrics.items():
                df_rows.append({
                    'Model': model_name,
                    'Accuracy': m.get('Accuracy', 0.0),
                    'Mean IoU': m.get('Mean_IoU', 0.0),
                    'Mean F1': m.get('Mean_F1', 0.0),
                    'Mean AP': m.get('Mean_AP', 0.0)
                })
            
            df = pd.DataFrame(df_rows)
            df_melted = df.melt('Model', var_name='Metric', value_name='Score')
            
            plt.figure(figsize=(12, 6))
            sns.barplot(data=df_melted, x='Metric', y='Score', hue='Model')
            plt.title('Overall Model Comparison')
            plt.ylim(0, 1.0)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'metrics_comparison.png')
            plt.close()
            logger.info("   Saved metrics comparison plot")
            
            # 2. Per-Class F1 Score Comparison
            class_rows = []
            for model_name, m in all_metrics.items():
                for cls_name in CLASS_NAMES:
                    per_class = m.get('per_class', {})
                    if cls_name in per_class:
                        class_rows.append({
                            'Model': model_name,
                            'Class': cls_name,
                            'F1 Score': per_class[cls_name].get('F1', 0.0),
                            'IoU': per_class[cls_name].get('IoU', 0.0)
                        })
            
            if class_rows:
                df_cls = pd.DataFrame(class_rows)
                
                plt.figure(figsize=(14, 6))
                sns.barplot(data=df_cls, x='Class', y='F1 Score', hue='Model')
                plt.title('Per-Class F1 Score Comparison')
                plt.ylim(0, 1.0)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig(self.plots_dir / 'per_class_f1.png')
                plt.close()
                logger.info("   Saved per-class F1 plot")
                
                plt.figure(figsize=(14, 6))
                sns.barplot(data=df_cls, x='Class', y='IoU', hue='Model')
                plt.title('Per-Class IoU Comparison')
                plt.ylim(0, 1.0)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig(self.plots_dir / 'per_class_iou.png')
                plt.close()
                logger.info("   Saved per-class IoU plot")

            # 3. Confusion Matrices
            num_models = len(all_metrics)
            if num_models > 0:
                # Use grid layout (columns of 4) for better visibility with 4+ models
                n_cols = min(4, num_models)
                n_rows = (num_models + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
                axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
                
                for idx, (model_name, m) in enumerate(all_metrics.items()):
                    ax = axes[idx]
                    cm = np.array(m.get('confusion_matrix', []))
                    if cm.size > 0:
                        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, vmin=0, vmax=1)
                        ax.set_title(f'{model_name}')
                        ax.set_ylabel('True Label')
                        ax.set_xlabel('Predicted Label')
                
                # Hide unused subplots
                for idx in range(num_models, len(axes)):
                    fig.delaxes(axes[idx])
                
                plt.tight_layout()
                plt.savefig(self.plots_dir / 'confusion_matrices.png')
                plt.close()
                logger.info("   Saved confusion matrices plot")
                
                # 3b. Raw Confusion Matrices (for detailed analysis)
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
                axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
                
                for idx, (model_name, m) in enumerate(all_metrics.items()):
                    ax = axes[idx]
                    cm_raw = np.array(m.get('confusion_matrix_raw', []))
                    if cm_raw.size > 0:
                        sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues',
                                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
                        ax.set_title(f'{model_name}')
                        ax.set_ylabel('True Label')
                        ax.set_xlabel('Predicted Label')
                
                # Hide unused subplots
                for idx in range(num_models, len(axes)):
                    fig.delaxes(axes[idx])
                
                plt.tight_layout()
                plt.savefig(self.plots_dir / 'confusion_matrices_raw.png')
                plt.close()
                logger.info("   Saved raw confusion matrices plot")
                
                # 4. PR Curves
                fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(6 * len(CLASS_NAMES), 5))
                if len(CLASS_NAMES) == 1: axes = [axes]
                
                for i, cls_name in enumerate(CLASS_NAMES):
                    ax = axes[i]
                    for model_name, m in all_metrics.items():
                        pr_data = m.get('pr_data', {})
                        if cls_name in pr_data:
                            ax.plot(pr_data[cls_name]['recall'], pr_data[cls_name]['precision'], 
                                     label=f"{model_name} (AP={pr_data[cls_name]['ap']:.2f})")
                    
                    ax.set_title(f'PR Curve: {cls_name}')
                    ax.set_xlabel('Recall')
                    ax.set_ylabel('Precision')
                    ax.set_ylim(0, 1.05)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                plt.tight_layout()
                plt.savefig(self.plots_dir / 'pr_curves.png')
                plt.close()
                logger.info("   Saved PR curves plot")
            
        except Exception as e:
            logger.error(f"Error generating comparison plots: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        """Run the model comparison with improved error handling and logging."""
        logger.info(f"Starting comparison for {len(model_configs)} models")
        
        # 1. Get Data
        image_files, label_files = self.get_validation_data()
        if not image_files:
            logger.error("No validation data found. Exiting.")
            return
            
        all_metrics = {}
        success_count = 0
        failure_count = 0
        
        # 2. Evaluate Each Model
        for i, config in enumerate(model_configs, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating model {i}/{len(model_configs)}: {config['name']}")
            logger.info(f"{'='*60}")
            
            start_time = time.time()
            
            model = self.load_model(config)
            if model is None:
                failure_count += 1
                logger.error(f" Failed to load model {config['name']}")
                continue
            
            try:
                y_pred, y_true, y_probs = self.evaluate_model(model, image_files, label_files)
                
                if y_pred is None or y_true is None or y_probs is None:
                    failure_count += 1
                    logger.error(f" Failed to evaluate model {config['name']}")
                    del model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()
                    continue
                
                metrics = self.compute_metrics(y_true, y_pred, y_probs)
                metrics['eval_time'] = time.time() - start_time
                
                all_metrics[config['name']] = metrics
                
                # Save individual metrics with safe file save
                metrics_path = self.metrics_dir / f"{config['name'].replace(' ', '_')}_metrics.json"
                success = safe_file_save(
                    lambda f, m: json.dump(m, f, indent=4),
                    open(metrics_path, 'w'),
                    metrics,
                    operation_name=f"Save metrics for {config['name']}"
                )
                
                if success:
                    logger.info(f" Successfully evaluated {config['name']}")
                    logger.info(f"  Accuracy: {metrics['Accuracy']:.4f}")
                    logger.info(f"  Mean IoU: {metrics['Mean_IoU']:.4f}")
                    logger.info(f"  Mean F1: {metrics['Mean_F1']:.4f}")
                    logger.info(f"  Mean AP: {metrics['Mean_AP']:.4f}")
                    logger.info(f"  Evaluation time: {metrics['eval_time']:.2f}s")
                    success_count += 1
                else:
                    failure_count += 1
                    logger.error(f" Failed to save metrics for {config['name']}")
                
                # Clear memory
                del model, y_pred, y_true, y_probs, metrics
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
                
            except Exception as e:
                failure_count += 1
                logger.error(f" Error during evaluation of {config['name']}: {e}")
                import traceback
                traceback.print_exc()
                # Clear memory on error
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
        
        # 3. Create Summary Table
        if all_metrics:
            self.create_summary_table(all_metrics)
            
            # 4. Generate Comparison Plots
            self.plot_comparisons(all_metrics)
        else:
            logger.warning("No models were successfully evaluated. Skipping summary and plots.")
        
        # 5. Print Summary
        logger.info(f"\n{'='*60}")
        logger.info("COMPARISON SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total models processed: {len(model_configs)}")
        logger.info(f"Successful: {success_count}")
        logger.info(f"Failed: {failure_count}")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"{'='*60}\n")

    def create_summary_table(self, all_metrics):
        """
        Create and save a summary table of all model metrics.
        
        Args:
            all_metrics: Dictionary of metrics for each model
        """
        try:
            rows = []
            for name, m in all_metrics.items():
                row = {
                    'Model': name,
                    'Accuracy': m.get('Accuracy', 0.0),
                    'Mean_IoU': m.get('Mean_IoU', 0.0),
                    'Mean_F1': m.get('Mean_F1', 0.0),
                    'Mean_AP': m.get('Mean_AP', 0.0),
                    'Time(s)': m.get('eval_time', 0.0)
                }
                # Add per-class IoU
                for cls_name in CLASS_NAMES:
                    if cls_name in m.get('per_class', {}):
                        row[f'{cls_name}_IoU'] = m['per_class'][cls_name]['IoU']
                    else:
                        row[f'{cls_name}_IoU'] = 0.0
                rows.append(row)
                
            df = pd.DataFrame(rows)
            print("\n=== Summary Metrics ===")
            print(df.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
            
            # Save summary metrics with safe file save
            summary_path = self.metrics_dir / 'summary_metrics.csv'
            success = safe_file_save(
                df.to_csv,
                summary_path,
                index=False,
                operation_name=f"Save summary metrics to {summary_path.name}"
            )
            
            if success:
                logger.info(f"   Saved summary metrics: {summary_path.name}")
            else:
                logger.error(f"   Failed to save summary metrics")
                
        except Exception as e:
            logger.error(f"Error creating summary table: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    """Main entry point for model comparison with improved error handling."""
    try:
        logger.info("="*60)
        logger.info("OMNICLOUDMASK MODEL COMPARISON")
        logger.info("="*60)
        
        # Log system information
        logger.info(f"Python version: {sys.version}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Device: {DEVICE}")
        logger.info(f"Number of GPUs available: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
        logger.info(f"Batch size: {COMPARISON_BATCH_SIZE}")
        logger.info(f"Use BF16: {COMPARISON_USE_BF16}")
        logger.info(f"Quick test mode: {COMPARISON_QUICK_TEST}")
        if COMPARISON_QUICK_TEST:
            logger.info(f"Quick test samples: {COMPARISON_QUICK_TEST_SAMPLES}")
        logger.info("="*60)
        
        # Create output directory
        output_dir = project_root / f"model_comparison_results_{CUSTOM_MODEL_VERSION}"
        logger.info(f"Output directory: {output_dir}")
        
        # Initialize comparator
        comparator = ModelComparator(output_dir)
        
        # Run comparison
        comparator.run()
        
        logger.info("="*60)
        logger.info("COMPARISON COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.warning("\nComparison interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nFATAL ERROR during comparison: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)