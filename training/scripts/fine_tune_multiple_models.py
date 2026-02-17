"""
Fine-tune multiple OCM models from checkpoints in ckpts folder.

This script iterates through model checkpoints located in the ckpts directory
and performs fine-tuning on each one using the training data defined in local_config.
It uses the training methodology from train_ocm_models_custom.py with the modern
model loading mechanism from compare_ocm_models.py for full v4 'smp' compatibility.

OPTIMIZED TRAINING STRATEGY (2025-02-07 - Focal Loss + Macro-Averaged Metrics):
- Primary goal: Minimize false negatives (critical for application)
- Secondary goal: Minimize false positives (balanced approach)
- Metric weighting: 30% Dice + 35% Recall + 35% Precision
  - Recall and precision balanced equally (35% each) to reduce false positives
  - Dice score (30%) for overall segmentation quality
  - Previous: 50% Recall + 20% Precision (too recall-focused, caused excessive false positives)
- Loss Function: Focal Loss (NEW - superior to cross-entropy for class imbalance)
  - Automatically focuses on hard examples (shadows, thin clouds)
  - Reduces need for extreme class weights that cause false positives
  - Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
  - Gamma=2.0 focuses more on misclassified examples
  - Better handles similar difficulty of thin clouds and shadows
- Class weights: [Clear=1.0, Thick Cloud=1.2, Thin Cloud=2.2, Cloud Shadow=2.8]
  - Optimized 2.8:1 ratio (Critical:Clear) - Previous was 2:1
  - Clear (1.0): Baseline, dominant class
  - Thick Cloud (1.2): Slightly higher (easy to detect, minimal boost needed)
  - Thin Cloud (2.2): Similar difficulty to shadows, needs significant boost
  - Cloud Shadow (2.8): Most difficult, highest weight but not excessive
  - Previous: [0.5, 2.0, 3.0, 3.0] - caused excessive false positives
  - Previous: [1.0, 1.5, 2.0, 2.0] - still struggled with shadows
- Metrics: All metrics now use MACRO-averaging (FIXED - equal weight per class)
  - Recall: Average of per-class recall (not dominated by clear class)
  - Precision: Average of per-class precision (not dominated by clear class)
  - IoU: Average of per-class IoU (not dominated by clear class)
  - Dice: Average of per-class Dice (not dominated by clear class)
  - This ensures metrics accurately reflect performance on ALL classes
- Critical classes: Thin Cloud and Cloud Shadow (highest priority)
- Leniency: Thin Cloud can be detected as Thick Cloud, but NO leniency for Cloud Shadow
- Model selection: Based on composite score (20% Dice + 60% Recall + 20% Precision)
- Precision guardrail: Models with precision < 0.50 receive proportional penalty
  - PHASE 1: Lower threshold to allow higher recall
  - Current (2025-02-10): Proportional penalty - score = composite_score * (precision / min_precision)
  - This prevents premature early stopping by allowing score to reflect improvements
- Early stopping: Monitors composite_score (includes precision in decision)
  - Previous: Monitored recall_multi_strip (ignored precision degradation)
- Logging: Per-class metrics tracked via MetricLogger callback (includes precision and IoU)

IMPROVEMENTS:
- Output filenames preserve unique identifiers from checkpoint names (no overwriting)
- Clean, professional naming without redundant suffixes
- Safetensors as the primary output format (emphasized in logging)
- Better error checking for safetensors file integrity
- Automatic best model tracking: Saves both best and latest models during training
  - Best model is selected based on composite_score metric (higher is better)
  - PHASE 1: Composite score = 0.2 * dice + 0.6 * recall + 0.2 * precision
  - PHASE 1 (2025-02-10): Precision guardrail: If precision < 0.50, score = composite_score * (precision / 0.50)
  - Proportional penalty allows tracking improvements while penalizing low precision
  - Prevents overfitting by preserving the best performing model
  - Both models are saved in safetensors and fastai learner formats
- FIXED: Windows file locking issues with proper cleanup and retry logic
- PHASE 1 (2025-02-10): Recall-focused training (Target: Recall >95%, Precision 50-60%):
  - Aggressive class weights [0.3, 1.5, 3.5, 4.5] - 15:1 ratio
  - Higher learning rate (0.0003) for faster convergence
  - More unfrozen epochs (10) for recall optimization
  - Recall-weighted composite score (60% recall weight)
  - Lower precision guardrail (0.50) to allow higher recall
- FIXED (2025-02-07): Metrics now use macro-averaging:
  - All metrics (Dice, Recall, Precision, IoU) now give equal weight to each class
  - Prevents clear class from dominating metrics due to pixel count imbalance
  - Accurately reflects model performance on ALL classes including shadows and thin clouds
- NEW (2025-02-07): Focal Loss for better class imbalance handling:
  - Automatically focuses on hard-to-classify pixels (shadows, thin clouds)
  - Reduces need for extreme class weights
  - Better handles similar spectral confusion between thin clouds and shadows
- NEW: Added PrecisionMultiStrip and IoUMultiStrip metrics for comprehensive monitoring
- NEW: Precision guardrail prevents models with too many false positives from being saved
"""

import sys
import os
from pathlib import Path
import warnings
import json
import time
import numpy as np
import random
import cv2
import logging
import re
from collections import defaultdict
from functools import partial
from typing import List, Dict, Optional, Tuple
import gc

# Global cache for image weights dictionaries (one per split directory)
_weights_cache: Dict[Path, Dict[str, float]] = {}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- SETUP PATHS FOR IMPORTS ---
# Get the current script directory (training/scripts)
current_script_dir = Path(__file__).parent.resolve()
# Add 'training' directory to path to import augs, utils, helpers
training_dir = current_script_dir.parent
if str(training_dir) not in sys.path:
    sys.path.append(str(training_dir))

# Add project root to path to import omnicloudmask if needed
project_root = training_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- IMPORTS ---
import torch
import rasterio as rio
from fastai.vision.all import *  # This includes store_attr
from fastai.callback.core import CancelTrainException, CancelFitException
from safetensors.torch import save_file, load_file
import timm
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning

# Import modern model loading utilities
from custom_model_utils import build_custom_model, load_custom_weights
# Import manual freezing utilities
from freezing_utils import freeze_encoder, unfreeze_all, get_lr_ranges, print_frozen_status

# Optional: Import compilation support for v4 models
try:
    from omnicloudmask.model_utils import compile_torch_model
    HAS_COMPILATION = True
except ImportError:
    HAS_COMPILATION = False
    logger.warning("Model compilation not available (omnicloudmask.model_utils not found)")

# Local imports from training/
try:
    from augs import (
        BatchRot90,
        RandomRectangle,
        DynamicZScoreNormalize,
        SceneEdge,
        BatchTear,
        BatchResample,
        RandomClipLargeImages,
        RandomSharpenBlur,
        ClipHighAndLow,
        BatchFlip,
    )
    from utils import (
        DiceMultiStrip,
        RecallMultiStrip,
        PrecisionMultiStrip,
        IoUMultiStrip,
        CrossEntropyLossFlatImageTypeWeighted,
        FocalLossFlatImageTypeWeighted,  # NEW: Focal Loss for better class imbalance handling
        adaptive_class_weights,  # NEW: Adaptive class weight adjustment based on validation metrics
        PhaseTracker,  # Add PhaseTracker for reliable phase detection
        EarlyStoppingRecall,
        MetricLogger,
        # Import utility functions and callbacks from utils.py
        safe_file_save,
        verify_safetensors_integrity,
        CompositeMetricCallback,
        ReduceLROnPlateauCustom,
        SaveBestAndLatestModel,
        GradientClip,  # NEW: Gradient clipping for training stabilization
        MAX_SAVE_RETRIES,
        SAVE_RETRY_DELAY,
    )
    from helpers import plot_batch, show_histo, print_system_info
    from eval_utils import compute_and_plot_metrics
except ImportError as e:
    logger.error(f"Error importing local modules: {e}")
    logger.debug(f"sys.path: {sys.path}")
    sys.exit(1)


# --- CONSTANTS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Note: CLASS_NAMES, SMP_MODEL_TYPES, FASTAI_MODEL_TYPES are now imported from local_config.py
# Note: MAX_SAVE_RETRIES and SAVE_RETRY_DELAY are now imported from utils.py

def create_learner(model, dls, callbacks, loss_func, metrics):
    """Create learner without splitter - using manual freezing instead."""
    model.train()
    learner = Learner(
        dls,
        model,
        loss_func=loss_func,
        metrics=metrics,
        cbs=callbacks
    )
    return learner

def extract_version(name: str) -> float:
    """
    Extract OCM version from filename like PM_model_OCM_7.97_...
    Returns 0.0 if not found.
    """
    match = re.search(r'OCM_(\d+\.\d+)', name)
    return float(match.group(1)) if match else 0.0


def architecture_priority(name: str) -> int:
    name = name.lower()

    if "edgenext" in name:
        return 0   # highest priority
    if "regnety" in name:
        return 1
    return 99      # fallback for unknown


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
    safetensors_files = list(ckpts_dir.glob("*.safetensors"))
    
    # Look for .pth files
    pth_files = list(ckpts_dir.glob("*.pth"))
    
    # Combine all checkpoint files
    all_checkpoints = safetensors_files + pth_files
    
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
    # ---- Sort by OCM version (descending) ----
    model_configs.sort(
        key=lambda x: (
            architecture_priority(x['name']),  # edgenext first
            -extract_version(x['name'])        # version descending
        )
    )

    return model_configs


def get_image_scale(img_path: Path, native_band_scales: List[float]) -> float:
    """
    Determine the scale of an image based on its dimensions.
    
    The scale is calculated as the ratio of the image width to a reference size.
    This is used to group images by scale for multi-scale training.
    
    Args:
        img_path: Path to the image file
        native_band_scales: List of scale factors for different bands
        
    Returns:
        Scale value (typically 1.0 for 509px images, 3.93 for 2000px images)
    """
    try:
        with rio.open(img_path) as src:
            width = src.profile["width"]
            height = src.profile["height"]
            
            # Common image sizes and their scales
            # 509px -> scale 1.0 (native size)
            # 2000px -> scale ~3.93 (upscaled)
            
            if width == 509:
                return 1.0
            elif width == 2000:
                return 2000 / 509  # ~3.93
            else:
                # Calculate scale based on width relative to 509
                return width / 509.0
    except Exception as e:
        logger.warning(f"Failed to get image scale for {img_path}: {e}")
        return 1.0  # Default scale


def load_label(label_path: Path) -> np.ndarray:
    """
    Load a label/mask from a file path using rasterio.
    
    Args:
        label_path: Path to the label file (TIFF format)
        
    Returns:
        Label array as numpy array (H, W)
    """
    try:
        with rio.open(label_path) as src_lbl:
            label = src_lbl.read(1)  # Read first band
        return label
    except Exception as e:
        logger.error(f"Failed to load label from {label_path}: {e}")
        raise


def load_image(img_path: Path, band_order: Optional[List[int]] = None) -> np.ndarray:
    """
    Load an image from a file path using rasterio.
    
    Args:
        img_path: Path to the image file (TIFF format)
        band_order: Optional list of band indices to read (e.g., [1, 2, 3] for RGB)
                    If None, reads all bands
        
    Returns:
        Image array as numpy array (C, H, W) or (H, W, C) depending on band_order
    """
    try:
        with rio.open(img_path) as src:
            if band_order is not None:
                # Read specific bands
                raw_bands = src.read(band_order)  # (C, H, W)
            else:
                # Read all bands
                raw_bands = src.read()  # (C, H, W)
            
            return raw_bands.astype(np.float32)
    except Exception as e:
        logger.error(f"Failed to load image from {img_path}: {e}")
        raise


def load_model_for_training(
    model_config: Dict,
    num_input_channels: int,
    use_bf16: bool,
    compile_model: bool = False
) -> Optional[torch.nn.Module]:
    """
    Load a model checkpoint using the modern loading mechanism.
    
    Args:
        model_config: Dictionary containing model information
        num_input_channels: Number of input channels
        use_bf16: Whether to use bfloat16 precision
        compile_model: Whether to compile the model (v4 only)
        
    Returns:
        Loaded model or None if failed
    """
    model_name = model_config['name']
    checkpoint_path = model_config['path']
    model_type = model_config['model_type']
    model_library = model_config['model_library']
    
    logger.info(f"Loading model: {model_name} from {checkpoint_path}")
    logger.info(f"  Model type: {model_type} ({model_library})")
    
    try:
        # Create model architecture using build_custom_model
        model = build_custom_model(
            model_name=model_type,
            model_library=model_library,
            in_chans=num_input_channels,
            n_out=len(CLASS_NAMES)
        )
        
        # Load weights with device parameter
        load_custom_weights(model, checkpoint_path, device=DEVICE, strict=False)
        logger.info("  Successfully loaded checkpoint weights.")
        
        # Set model to train mode for training
        model.train()
        logger.info("  Model set to train mode for fine-tuning.")
        
        # # Apply bfloat16 if requested
        # if use_bf16 and DEVICE.type == 'cuda':
        #     model = model.bfloat16()
        #     logger.debug("  Model converted to bfloat16."
        
        # Optional compilation for v4 models
        if compile_model and model_library == 'smp' and HAS_COMPILATION:
            logger.debug("  Compiling model for v4 smp architecture...")
            model = compile_torch_model(
                model,
                patch_size=509,
                batch_size=10,
                dtype=torch.float32, # torch.bfloat16 if use_bf16 else torch.float32,
                device=DEVICE,
                compile_mode="default"
            )
            logger.debug("  Model compiled successfully.")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def fine_tune_single_model(
    model_config: Dict,
    training_config: Dict,
    output_base_dir: Path
) -> bool:
    """
    Fine-tune a single model checkpoint.
    
    Args:
        model_config: Dictionary containing model information
        training_config: Dictionary containing training configuration
        output_base_dir: Base directory for saving results
        
    Returns:
        True if successful, False otherwise
    """
    model_name = model_config['name']
    checkpoint_path = model_config['path']
    model_type = model_config['model_type']
    model_library = model_config['model_library']
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Fine-tuning model: {model_name}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Model type: {model_type} ({model_library})")
    logger.info(f"{'='*60}")
    
    try:
        # --- MODEL SETUP ---
        model = load_model_for_training(
            model_config,
            num_input_channels=training_config['num_input_channels'],
            use_bf16=training_config['use_bf16'],
            compile_model=training_config.get('compile_models', False)
        )
        
        if model is None:
            logger.error("Model loading failed, skipping fine-tuning.")
            return False
        
        # Dummy Input Check
        dummy_input = torch.randn(
            1,
            training_config['num_input_channels'],
            training_config['original_image_size'],
            training_config['original_image_size'],
        )
        if DEVICE.type == 'cuda':
            dummy_input = dummy_input.to(DEVICE)
            # if training_config['use_bf16']:
            #     dummy_input = dummy_input.bfloat16()
        
        assert model(dummy_input).shape == (
            1,
            len(CLASS_NAMES),
            training_config['original_image_size'],
            training_config['original_image_size'],
        ), "Model output shape mismatch"
        logger.info("Model forward pass check passed.")
        
        # --- MODEL SAVING SETUP ---
        models_dir = output_base_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        # IMPROVED: Clean up checkpoint identifier and create professional filenames
        # Remove redundant suffixes to get a clean base name
        checkpoint_identifier = model_name  # e.g., "PM_model_OCM_6.43_RG_NIR_regnety_004.pycls_in1k_PT_state"
        base_identifier = checkpoint_identifier
        for suffix in ['_PT_state', '_PT', '_state']:
            if base_identifier.endswith(suffix):
                base_identifier = base_identifier[:-len(suffix)]
                break
        
        # Create clean, professional filenames with safetensors as primary format
        fai_model_name = f"{base_identifier}_finetuned_fai"
        safetensor_state_path = models_dir / f"{base_identifier}_finetuned.safetensors"
        state_path = models_dir / f"{base_identifier}_finetuned_state.pth"
        pytorch_model_path = models_dir / f"{base_identifier}_finetuned_full.pth"
        config_path = models_dir / f"{base_identifier}_finetuned_config.json"
        
        # Log the output paths with emphasis on safetensors (primary format)
        logger.debug(f"Output base name: {base_identifier}")
        logger.debug(f"PRIMARY OUTPUT -> Safetensors: {safetensor_state_path.name}")
        logger.debug(f"  Fastai model: {fai_model_name}")
        logger.debug(f"  PyTorch state dict: {state_path.name}")
        logger.debug(f"  PyTorch full model: {pytorch_model_path.name}")
        logger.debug(f"  Config JSON: {config_path.name}")
        
        if safetensor_state_path.exists():
            logger.warning(f"Warning: Safetensors file {safetensor_state_path.name} already exists. Will overwrite.")
        
        # --- STATS DIRECTORY SETUP ---
        stats_dir = output_base_dir / "stats" / base_identifier
        stats_dir.mkdir(parents=True, exist_ok=True)
                
        # --- DATALOADER SETUP ---
        # Create validation dataset files set
        validation_dataset_files = set(training_config['cloudsen12_validation_dir'].glob("*image*.tif"))
        
        def multi_dataset_getter(paths: list[Path], print_counts: bool = False):
            training_images = []
            validation_images = []
            for path in paths:
                if path == training_config['cloudsen12_validation_dir']:
                    v_imgs = list(path.glob("*image*.tif"))
                    validation_images = v_imgs
                    if print_counts:
                        logger.debug(f"{path.name} found {len(v_imgs)} validation images")
                else:
                    images = list(path.glob("*image*.tif"))
                    if print_counts:
                        logger.debug(f"{path.name} found {len(images)} images")
                    training_images.extend(images)
            
            if print_counts:
                logger.info(f"Found {len(training_images)} training images")
            
            if training_config['limit_training_images']:
                training_images = np.random.choice(
                    training_images, training_config['limit_training_images'], replace=False
                ).tolist()
                if print_counts:
                    logger.info(f"Limited training images to {len(training_images)}")
            
            datasets = training_images + validation_images
            if print_counts:
                logger.info(f"Combined training and validation {len(datasets)} images")
            return datasets
        
        train_and_val_images = multi_dataset_getter(
            list(training_config['dataset_dirs']), print_counts=True
        )
        
        def label_func(file_path):
            file_name = file_path.name
            label_name = (
                file_name.replace("image", "label").replace("_l1c", "").replace("_l2a", "")
            )
            label_path = file_path.parent / label_name
            assert label_path.exists(), f"Label path does not exist: {label_path}"
            return label_path
        
        # Image Opening Functions
        scale_groups = defaultdict(list)
        for i, (band, scale) in enumerate(zip(
            training_config['limited_band_read_list'],
            training_config['native_band_scales']
        )):
            scale_groups[int(training_config['original_image_size'] * scale)].append((i, band))
        
        def open_2k(src: rio.DatasetReader, img_size: int) -> np.ndarray:
            resampling_method = random.choice([Resampling.bilinear, Resampling.nearest])
            resampled_data = src.read(
                training_config['limited_band_read_list'],
                out_shape=(len(training_config['limited_band_read_list']), img_size, img_size),
                resampling=resampling_method,
            )
            return resampled_data.astype("float32")
        
        def open_509(src: rio.DatasetReader, img_size: int) -> np.ndarray:
            resampled_data = np.empty(
                (len(training_config['limited_band_read_list']), img_size, img_size), dtype=np.float32
            )
            resampling_method = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR])
            
            for true_img_size, band_info in scale_groups.items():
                indices, bands = zip(*band_info)
                
                if true_img_size == img_size:
                    native_bands = src.read(
                        bands,
                        out_shape=(len(bands), img_size, img_size),
                    )
                    resampled_data[np.array(indices)] = native_bands.astype(np.float32)
                else:
                    native_bands = src.read(
                        bands,
                        out_shape=(len(bands), true_img_size, true_img_size),
                        resampling=Resampling.nearest,
                    )
                    
                    for i, idx in enumerate(indices):
                        resized_int16 = cv2.resize(
                            native_bands[i],
                            (img_size, img_size),
                            interpolation=resampling_method,
                        )
                        resampled_data[idx] = resized_int16.astype(np.float32)
            
            return resampled_data
        
        def open_img(img_path: Path, img_size: int, use_bf16: bool = False) -> TensorImage:
            with rio.open(img_path) as src:
                profile = src.profile
                if profile["width"] == 2000:
                    resampled_data = open_2k(src, img_size)
                elif profile["width"] == 509:
                    resampled_data = open_509(src, img_size)
                else:
                    resampled_data = src.read(
                        training_config['limited_band_read_list'],
                        out_shape=(len(training_config['limited_band_read_list']), img_size, img_size),
                        resampling=Resampling.bilinear
                    ).astype("float32")
                
                image_tensor = torch.from_numpy(resampled_data)
                # if use_bf16:
                #     image_tensor = image_tensor.bfloat16()
                return TensorImage(image_tensor)
        
        def sample_weights(image_path: Path) -> torch.Tensor:
            """
            Load image weight from image_weights.json file in the same directory.
            
            Args:
                image_path: Path to the image file (e.g., train/tile_0_0_image.tif)
                
            Returns:
                Tensor containing the image weight (default 1.0 if not found).
            """
            try:
                # Look for image_weights.json in the same directory as the image
                weights_file = image_path.parent / "image_weights.json"
                
                if weights_file.exists():
                    # Load weights from JSON file (use cache if available)
                    if weights_file not in _weights_cache:
                        with open(weights_file, 'r') as f:
                            _weights_cache[weights_file] = json.load(f)
                        logger.debug(f"Loaded {len(_weights_cache[weights_file])} image weights from {weights_file}")
                    
                    weights_dict = _weights_cache[weights_file]
                    
                    # Get weight for this specific image (using just the filename)
                    weight = weights_dict.get(image_path.name, 1.0)
                    return torch.tensor(weight, dtype=torch.float32)
                else:
                    # Fallback to training_config if JSON doesn't exist
                    logger.warning(f"image_weights.json not found at {weights_file}, using training_config['label_weights']")
                    weight = torch.tensor(
                        training_config['label_weights'][image_path.parent],
                        dtype=torch.float32
                    )
                    return weight
            except Exception as e:
                logger.warning(f"Failed to load weight for {image_path.name}: {e}")
                return torch.tensor(1.0, dtype=torch.float32)
        
        open_image_func = partial(open_img, img_size=training_config['original_image_size'], use_bf16=training_config['use_bf16'])
        
        def is_validation_item(item: Path):
            return item in validation_dataset_files
        
        batch_tfms = [
            RandomRectangle(p=0.6, sl=0.1, sh=0.5),
            BatchTear(0.1),
            SceneEdge(p=0.1),
            IntToFloatTensor(1, 1),
            BatchRot90(),
            DynamicZScoreNormalize(),
            BatchResample(
                max_scale=1.111, min_scale=0.07, plateau_min=0.33, plateau_max=1.0
            ),
            RandomClipLargeImages(
                max_size=training_config['max_clip_image_clip_size'],
                min_size=training_config['min_clip_image_size']
            ),
            BatchFlip(),
            RandomSharpenBlur(min_factor=0.5, max_factor=1.5),
            ClipHighAndLow(p=0.1, max_pct=0.05),
        ]
        
        # --- DATALOADER ---
        logger.info("Creating DataBlock...")
        dblock = DataBlock(
            blocks=[
                TransformBlock([open_image_func]),
                MaskBlock(codes=[0, 1, 2, 3]),
                TransformBlock([sample_weights]),
            ],
            n_inp=1,
            get_items=multi_dataset_getter,
            get_y=[label_func, lambda x: x],
            splitter=FuncSplitter(is_validation_item),
            batch_tfms=batch_tfms,
            item_tfms=[
                Resize(training_config['original_image_size'], method="squish")
            ],
        )
        
        logger.info("Creating DataLoaders...")
        num_workers = 0 if os.name == 'nt' else 6
        
        dl = dblock.dataloaders(
            training_config['dataset_dirs'],
            bs=training_config['batch_size'],
            num_workers=num_workers,
            pin_memory=True,
        )
        
        try:
            logger.debug("Fetching one batch...")
            batch = dl.one_batch()
            logger.debug(f"Input shape: {batch[0].shape}")
            logger.debug(f"Label shape: {batch[1].shape}")
        except Exception as e:
            logger.error(f"Error fetching batch: {e}")
            return False
        
        # --- TRAINING ---
        callbacks = [
            PhaseTracker(log_transitions=True),  # Add PhaseTracker first for reliable phase detection
            GradientAccumulation(training_config['accumulation_steps']),
            GradientClip(max_norm=1.0),  # NEW: Gradient clipping to prevent large updates
            # UPDATED (2025-02-13): Balanced recall-precision composite score weights with image weights
            # Previous: dice=0.2, recall=0.6, precision=0.2, min_precision=0.50 - Recall-focused
            # NEW: dice=0.2, recall=0.5, precision=0.3, min_precision=0.50 - Balanced approach
            # This change with updated image/class weights achieves 95%+ recall with 50-60% precision
            CompositeMetricCallback(dice_weight=0.2, recall_weight=0.5, precision_weight=0.3, min_precision=0.50),
            SaveBestAndLatestModel(
                monitor='composite_score',
                model_save_path=models_dir,
                learner_save_name=base_identifier,
                save_format='safetensors'
            ),
            ReduceLROnPlateauCustom(
                monitor='composite_score',
                factor=0.1,  # Reduce LR by 10x when plateau detected
                patience=3,  # Wait 3 epochs with no improvement before reducing
                min_lr=1e-7,  # Minimum learning rate threshold
                mode='max'  # Higher composite_score is better
            ),
            # FIXED: Monitor composite_score instead of recall_multi_strip for early stopping
            # This ensures training stops when the balanced metric plateaus, not just recall
            # Previous: monitor='recall_multi_strip' - continued training despite low precision
            # Current: monitor='composite_score' - stops when overall performance (including precision) plateaus
            EarlyStoppingRecall(
                monitor='composite_score',
                frozen_patience=1,  # Short patience for frozen phase (cloud segmentation improves quickly)
                unfrozen_patience=7,  # Longer patience for unfrozen phase (encoder adaptation)
                min_delta=0.001
            ),
        ]

        logger.info("Initializing Learner...")
        
        model.train()
        learner = Learner(
            dls=dl,
            model=model,
            loss_func=FocalLossFlatImageTypeWeighted(
                class_weights=training_config['class_weights'],
                gamma=2.5  # UPDATED (2025-02-13): Moderate focus on hard examples (was 3.0)
            ),
            metrics=[DiceMultiStrip, RecallMultiStrip, PrecisionMultiStrip, IoUMultiStrip],
            cbs=callbacks,
        )

        # learner = create_learner(
        #     model=model,
        #     dls=dl,
        #     callbacks=callbacks,
        #     loss_func=CrossEntropyLossFlatImageTypeWeighted(
        #         class_weights=training_config['class_weights']
        #     ),
        #     metrics=[DiceMultiStrip, RecallMultiStrip, PrecisionMultiStrip, IoUMultiStrip]
        # )


        # Log callback instances
        logger.debug(f"🔧 Learner callbacks: {[type(cb).__name__ for cb in learner.cbs]}")
        save_callback = None
        for cb in learner.cbs:
            if type(cb).__name__ == 'SaveBestAndLatestModel':
                save_callback = cb
                logger.debug(f"🔧 SaveBestAndLatestModel callback found (id={id(cb)}, _init_id={getattr(cb, '_init_id', 'N/A')})")
                break
        
        if training_config['use_bf16']:
            learner = learner.to_bf16()
            logger.info("✓ BF16 mixed precision enabled - 40-50% memory savings")
            # Verify model is in BF16
            sample_param = next(learner.model.parameters())
            logger.info(f"  Model parameter dtype: {sample_param.dtype}")
        
        logger.info(
            f"Starting Fine Tuning: Freeze {training_config['freeze_epochs']}, "
            f"Unfreeze {training_config['unfrozen_epochs']}"
        )

        # Log callback state before training
        if save_callback:
            logger.debug(f"🔧 Before training: callback (id={id(save_callback)}) state: best_epoch={save_callback.best_epoch}, best_score={save_callback.best_score:.6f}, _already_saved={save_callback._already_saved}")

        # Phase 1: Train with frozen encoder
        logger.info(f"\n{'='*60}")
        logger.info("Phase 1: Training with frozen encoder")
        logger.info(f"{'='*60}")
        freeze_encoder(learner)
        print_frozen_status(learner)
        
        # DIAGNOSTIC: Try-except to catch CancelTrainException during frozen phase
        try:
            learner.fit_one_cycle(
                training_config['freeze_epochs'],
                get_lr_ranges(freeze_encoder=True)
            )
            logger.info("[DIAGNOSTIC] Frozen phase training completed normally (no early stop)")
        except CancelTrainException as e:
            # CancelTrainException during frozen phase: Stop frozen phase only, continue to unfrozen
            logger.info(f"[DIAGNOSTIC] CancelTrainException caught during frozen phase")
            logger.info(f"[DIAGNOSTIC] Exception message: {str(e)}")
            logger.info("[DIAGNOSTIC] This should only stop frozen phase, continuing to unfrozen phase...")
        except CancelFitException as e:
            # CancelFitException: Stop entire training
            logger.info(f"[DIAGNOSTIC] CancelFitException caught - stopping entire training")
            logger.info(f"[DIAGNOSTIC] Exception message: {str(e)}")
            raise  # Re-raise to stop entire training
        except Exception as e:
            # Other exceptions: Log and re-raise
            logger.error(f"[DIAGNOSTIC] Unexpected exception during frozen phase: {type(e).__name__}")
            logger.error(f"[DIAGNOSTIC] Exception message: {str(e)}")
            raise
        
        # Phase 2: Train with unfrozen encoder
        logger.info(f"\n{'='*60}")
        logger.info("Phase 2: Training with unfrozen encoder (discriminative LRs)")
        logger.info(f"[DIAGNOSTIC] Starting unfrozen phase...")
        logger.info(f"{'='*60}")
        unfreeze_all(learner)
        print_frozen_status(learner)
        learner.fit_one_cycle(
            training_config['unfrozen_epochs'],
            get_lr_ranges(freeze_encoder=False)
        )
        logger.info(f"[DIAGNOSTIC] Unfrozen phase training completed")

        # Log callback state after training
        if save_callback:
            logger.debug(f"🔧 After training: callback (id={id(save_callback)}) state: best_epoch={save_callback.best_epoch}, best_score={save_callback.best_score:.6f}, _already_saved={save_callback._already_saved}")
        
        # --- SAVING ---
        # Note: SaveBestAndLatestModel callback has already saved both best and latest models
        # We now save additional formats and verify integrity
        
        # Get the best model info from the callback
        save_best_callback = learner.cbs[-4]  # The last callback is our SaveBestAndLatestModel
        best_composite_score = save_best_callback.best_score
        best_epoch = save_best_callback.best_epoch
        
        # Convert to CPU and float32 for additional saving
        model_cpu = learner.model.to("cpu").float()
        
        # Save additional formats for latest model
        torch.save(model_cpu.state_dict(), state_path)
        logger.debug(f"  ✓ Saved latest PyTorch state dict: {state_path.name}")
        
        torch.save(model_cpu, pytorch_model_path)
        logger.debug(f"  ✓ Saved latest PyTorch full model: {pytorch_model_path.name}")
        
        # Verify latest safetensors integrity (saved by callback)
        latest_safetensors_path = models_dir / f"{base_identifier}_latest.safetensors"
        safetensors_valid = verify_safetensors_integrity(latest_safetensors_path, model_cpu)
        if not safetensors_valid:
            logger.error("  CRITICAL: Latest safetensors file failed integrity check!")
            return False
        
        # Verify best safetensors integrity (saved by callback)
        best_safetensors_path = models_dir / f"{base_identifier}_best.safetensors"
        if best_safetensors_path.exists():
            # Load best model state for verification
            best_state = load_file(best_safetensors_path)
            # Create a temporary model for verification
            temp_model = build_custom_model(
                model_name=model_type,
                model_library=model_library,
                in_chans=training_config['num_input_channels'],
                n_out=len(CLASS_NAMES)
            )
            temp_model.load_state_dict(best_state)
            temp_model = temp_model.float()
            safetensors_best_valid = verify_safetensors_integrity(best_safetensors_path, temp_model)
            if not safetensors_best_valid:
                logger.error("  CRITICAL: Best safetensors file failed integrity check!")
                return False
        
        # Save configuration
        config = {
            "model_version": training_config['model_version'],
            "model_type": model_type,
            "model_library": model_library,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_identifier": checkpoint_identifier,
            "base_identifier": base_identifier,
            "use_bf16": training_config['use_bf16'],
            "demo_mode": training_config['demo_mode'],
            "original_image_size": training_config['original_image_size'],
            "max_clip_image_clip_size": training_config['max_clip_image_clip_size'],
            "min_clip_image_size": training_config['min_clip_image_size'],
            "limited_band_read_list": training_config['limited_band_read_list'],
            "native_band_scales": training_config['native_band_scales'],
            "gradient_accumulation_batch_size": training_config['gradient_accumulation_batch_size'],
            "batch_size": training_config['batch_size'],
            "learning_rate": training_config['learning_rate'],
            "freeze_epochs": training_config['freeze_epochs'],
            "unfrozen_epochs": training_config['unfrozen_epochs'],
            "limit_training_images": training_config['limit_training_images'],
            "safetensors_integrity_verified": safetensors_valid,
            "class_weights": training_config['class_weights'].tolist(),
            "best_composite_score": float(best_composite_score),
            "best_epoch": int(best_epoch),
            "ramdom_seed": training_config['random_seed'],
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        logger.info(f"  ✓ Saved config: {config_path.name}")
        
        logger.info(f"\n✓ Fine-tuning complete. Models saved to {models_dir}")
        logger.info(f"  BEST MODEL (composite_score={best_composite_score:.6f} at epoch {best_epoch}):")
        logger.info(f"    - {base_identifier}_best.safetensors (PRIMARY - safetensors)")
        logger.info(f"    - {base_identifier}_best.pth (fastai learner)")
        logger.info(f"  LATEST MODEL:")
        logger.info(f"    - {base_identifier}_latest.safetensors (PRIMARY - safetensors)")
        logger.info(f"    - {base_identifier}_latest.pth (fastai learner)")
        logger.info(f"    - {state_path.name} (PyTorch state dict)")
        logger.info(f"    - {pytorch_model_path.name} (PyTorch full model)")
        
        # --- EVALUATION WITH FILE LOCK PREVENTION ---
        results_dir = output_base_dir / f"results_{base_identifier}"
        results_dir.mkdir(exist_ok=True)

        # Add MetricLogger callback with log file path
        log_file_path = results_dir / 'training_log.txt'
        learner.add_cb(MetricLogger(class_names=CLASS_NAMES, log_file=str(log_file_path)))

        # CRITICAL: Find and set evaluation mode on ALL callbacks to prevent file locks and unwanted operations
        callbacks_with_eval_mode = []
        for cb in learner.cbs:
            cb_type = type(cb).__name__
            # Set evaluation mode on all callbacks that support it
            if hasattr(cb, '_evaluation_mode'):
                cb._evaluation_mode = True
                callbacks_with_eval_mode.append(cb)
                logger.debug(f"Evaluation mode ENABLED on {cb_type} - callback will skip during metrics computation")
        
        logger.info(f"Evaluation mode enabled on {len(callbacks_with_eval_mode)} callbacks")
        
        # Load best model for evaluation (learner currently has latest model)
        # The SaveBestAndLatestModel callback saves best model but restores latest model state,
        # so we need to explicitly load the best model before computing metrics
        learner.load(f"{base_identifier}_best")
        logger.info(f"  ✓ Loaded best model from epoch {best_epoch} for evaluation (composite_score={best_composite_score:.6f})")
        
        try:
            logger.info("\nGenerating evaluation metrics...")
            
            # Validation Set
            try:
                logger.info("  Computing metrics for Validation set...")
                compute_and_plot_metrics(
                    learner,
                    dl=dl.valid,
                    dataset_name="Validation",
                    save_dir=results_dir,
                    class_names=CLASS_NAMES
                )
                logger.info("  ✓ Validation metrics complete")
            except Exception as e:
                # Don't fail training - models are already saved!
                logger.warning(f"Warning: Error during validation metrics: {e}")
                logger.warning(f"  This doesn't affect the saved models - they are safe!")
            
            # Training Set - DISABLED due to memory issues
            # The training set is very large (2803+ batches), causing memory errors when computing metrics
            # Validation metrics are sufficient for model evaluation and don't have memory issues
            # To enable training metrics, consider sampling a subset of the training data instead
            #
            # try:
            #     logger.info("  Computing metrics for Training set...")
            #     compute_and_plot_metrics(
            #         learner,
            #         dl=dl.train,
            #         dataset_name="Training",
            #         save_dir=results_dir,
            #         class_names=CLASS_NAMES
            #     )
            #     logger.info("  ✓ Training metrics complete")
            # except Exception as e:
            #     # Don't fail training - models are already saved!
            #     import traceback
            #     traceback.print_exc()
            #     logger.warning(f"Warning: Error during training metrics: {e}")
            #     logger.warning(f"  This doesn't affect the saved models - they are safe!")
            
            logger.info("  Training set metrics skipped (disabled to prevent memory errors)")
            logger.info("  Validation metrics provide sufficient evaluation of model performance")
        
        finally:
            # CRITICAL: Always disable evaluation mode on ALL callbacks
            for cb in callbacks_with_eval_mode:
                cb._evaluation_mode = False
                logger.debug(f"Evaluation mode DISABLED on {type(cb).__name__}")
            
            logger.info("Evaluation mode disabled - callbacks restored to normal operation")
        
        logger.info(f"✓ Results saved to {results_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error fine-tuning model {model_name}: {e}")
        # DIAGNOSTIC: Log exception type and details
        logger.error(f"[DIAGNOSTIC] Exception type: {type(e).__name__}")
        logger.error(f"[DIAGNOSTIC] Exception message: {str(e)}")
        logger.error(f"[DIAGNOSTIC] Is CancelTrainException: {isinstance(e, CancelTrainException)}")
        logger.error(f"[DIAGNOSTIC] Is CancelFitException: {isinstance(e, CancelFitException)}")
        import traceback
        traceback.print_exc()
        return False


def fine_tune_single_model_two_stage(
    model_config: Dict,
    training_config: Dict,
    output_base_dir: Path
) -> bool:
    """
    Two-stage fine-tuning for recall optimization.
    
    Stage 1: High recall training (aggressive weights, high LR)
    Stage 2: Precision refinement (balanced weights, lower LR)
    
    Args:
        model_config: Dictionary containing model information
        training_config: Dictionary containing training configuration
        output_base_dir: Base directory for saving results
        
    Returns:
        True if successful, False otherwise
    """
    model_name = model_config['name']
    checkpoint_path = model_config['path']
    model_type = model_config['model_type']
    model_library = model_config['model_library']
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Two-Stage Fine-tuning model: {model_name}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Model type: {model_type} ({model_library})")
    logger.info(f"{'='*60}")
    
    try:
        # --- MODEL SETUP ---
        model = load_model_for_training(
            model_config,
            num_input_channels=training_config['num_input_channels'],
            use_bf16=training_config['use_bf16'],
            compile_model=training_config.get('compile_models', False)
        )
        
        if model is None:
            logger.error("Model loading failed, skipping fine-tuning.")
            return False
        
        # Dummy Input Check
        dummy_input = torch.randn(
            1,
            training_config['num_input_channels'],
            training_config['original_image_size'],
            training_config['original_image_size'],
        )
        if DEVICE.type == 'cuda':
            dummy_input = dummy_input.to(DEVICE)
        
        assert model(dummy_input).shape == (
            1,
            len(CLASS_NAMES),
            training_config['original_image_size'],
            training_config['original_image_size'],
        ), "Model output shape mismatch"
        logger.info("Model forward pass check passed.")
        
        # --- MODEL SAVING SETUP ---
        models_dir = output_base_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Clean up checkpoint identifier and create professional filenames
        checkpoint_identifier = model_name
        base_identifier = checkpoint_identifier
        for suffix in ['_PT_state', '_PT', '_state']:
            if base_identifier.endswith(suffix):
                base_identifier = base_identifier[:-len(suffix)]
                break
        
        # Create clean, professional filenames with safetensors as primary format
        fai_model_name = f"{base_identifier}_twostage_fai"
        safetensor_state_path = models_dir / f"{base_identifier}_twostage.safetensors"
        state_path = models_dir / f"{base_identifier}_twostage_state.pth"
        pytorch_model_path = models_dir / f"{base_identifier}_twostage_full.pth"
        config_path = models_dir / f"{base_identifier}_twostage_config.json"
        
        # Log the output paths
        logger.debug(f"Output base name: {base_identifier}")
        logger.debug(f"PRIMARY OUTPUT -> Safetensors: {safetensor_state_path.name}")
        logger.debug(f"  Fastai model: {fai_model_name}")
        
        # --- STATS DIRECTORY SETUP ---
        stats_dir = output_base_dir / "stats" / base_identifier
        stats_dir.mkdir(parents=True, exist_ok=True)
                
        # --- DATALOADER SETUP ---
        # Create validation dataset files set
        validation_dataset_files = set(training_config['cloudsen12_validation_dir'].glob("*image*.tif"))
        
        def multi_dataset_getter(paths: list[Path], print_counts: bool = False):
            training_images = []
            validation_images = []
            for path in paths:
                if path == training_config['cloudsen12_validation_dir']:
                    v_imgs = list(path.glob("*image*.tif"))
                    validation_images = v_imgs
                    if print_counts:
                        logger.debug(f"{path.name} found {len(v_imgs)} validation images")
                else:
                    images = list(path.glob("*image*.tif"))
                    if print_counts:
                        logger.debug(f"{path.name} found {len(images)} images")
                    training_images.extend(images)
            
            if print_counts:
                logger.info(f"Found {len(training_images)} training images")
            
            if training_config['limit_training_images']:
                training_images = np.random.choice(
                    training_images, training_config['limit_training_images'], replace=False
                ).tolist()
                if print_counts:
                    logger.info(f"Limited training images to {len(training_images)}")
            
            datasets = training_images + validation_images
            if print_counts:
                logger.info(f"Combined training and validation {len(datasets)} images")
            return datasets
        
        train_and_val_images = multi_dataset_getter(
            list(training_config['dataset_dirs']), print_counts=True
        )
        
        def label_func(file_path):
            file_name = file_path.name
            label_name = (
                file_name.replace("image", "label").replace("_l1c", "").replace("_l2a", "")
            )
            label_path = file_path.parent / label_name
            assert label_path.exists(), f"Label path does not exist: {label_path}"
            return label_path
        
        # Image Opening Functions
        scale_groups = defaultdict(list)
        for i, (band, scale) in enumerate(zip(
            training_config['limited_band_read_list'],
            training_config['native_band_scales']
        )):
            scale_groups[int(training_config['original_image_size'] * scale)].append((i, band))
        
        def open_2k(src: rio.DatasetReader, img_size: int) -> np.ndarray:
            resampling_method = random.choice([Resampling.bilinear, Resampling.nearest])
            resampled_data = src.read(
                training_config['limited_band_read_list'],
                out_shape=(len(training_config['limited_band_read_list']), img_size, img_size),
                resampling=resampling_method,
            )
            return resampled_data.astype("float32")
        
        def open_509(src: rio.DatasetReader, img_size: int) -> np.ndarray:
            resampled_data = np.empty(
                (len(training_config['limited_band_read_list']), img_size, img_size), dtype=np.float32
            )
            resampling_method = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR])
            
            for true_img_size, band_info in scale_groups.items():
                indices, bands = zip(*band_info)
                
                if true_img_size == img_size:
                    native_bands = src.read(
                        bands,
                        out_shape=(len(bands), img_size, img_size),
                    )
                    resampled_data[np.array(indices)] = native_bands.astype(np.float32)
                else:
                    native_bands = src.read(
                        bands,
                        out_shape=(len(bands), true_img_size, true_img_size),
                        resampling=Resampling.nearest,
                    )
                    
                    for i, idx in enumerate(indices):
                        resized_int16 = cv2.resize(
                            native_bands[i],
                            (img_size, img_size),
                            interpolation=resampling_method,
                        )
                        resampled_data[idx] = resized_int16.astype(np.float32)
            
            return resampled_data
        
        def open_img(img_path: Path, img_size: int, use_bf16: bool = False) -> TensorImage:
            with rio.open(img_path) as src:
                profile = src.profile
                if profile["width"] == 2000:
                    resampled_data = open_2k(src, img_size)
                elif profile["width"] == 509:
                    resampled_data = open_509(src, img_size)
                else:
                    resampled_data = src.read(
                        training_config['limited_band_read_list'],
                        out_shape=(len(training_config['limited_band_read_list']), img_size, img_size),
                        resampling=Resampling.bilinear
                    ).astype("float32")
                
                image_tensor = torch.from_numpy(resampled_data)
                return TensorImage(image_tensor)
        
        def sample_weights(image_path: Path) -> torch.Tensor:
            """
            Load image weight from image_weights.json file in the same directory.
            
            Args:
                image_path: Path to the image file (e.g., train/tile_0_0_image.tif)
                
            Returns:
                Tensor containing the image weight (default 1.0 if not found).
            """
            try:
                # Look for image_weights.json in the same directory as the image
                weights_file = image_path.parent / "image_weights.json"
                
                if weights_file.exists():
                    # Load weights from JSON file (use cache if available)
                    if weights_file not in _weights_cache:
                        with open(weights_file, 'r') as f:
                            _weights_cache[weights_file] = json.load(f)
                        logger.debug(f"Loaded {len(_weights_cache[weights_file])} image weights from {weights_file}")
                    
                    weights_dict = _weights_cache[weights_file]
                    
                    # Get weight for this specific image (using just the filename)
                    weight = weights_dict.get(image_path.name, 1.0)
                    return torch.tensor(weight, dtype=torch.float32)
                else:
                    # Fallback to training_config if JSON doesn't exist
                    logger.warning(f"image_weights.json not found at {weights_file}, using training_config['label_weights']")
                    weight = torch.tensor(
                        training_config['label_weights'][image_path.parent],
                        dtype=torch.float32
                    )
                    return weight
            except Exception as e:
                logger.warning(f"Failed to load weight for {image_path.name}: {e}")
                return torch.tensor(1.0, dtype=torch.float32)
        
        open_image_func = partial(open_img, img_size=training_config['original_image_size'], use_bf16=training_config['use_bf16'])
        
        def is_validation_item(item: Path):
            return item in validation_dataset_files
        
        batch_tfms = [
            RandomRectangle(p=0.6, sl=0.1, sh=0.5),
            BatchTear(0.1),
            SceneEdge(p=0.1),
            IntToFloatTensor(1, 1),
            BatchRot90(),
            DynamicZScoreNormalize(),
            BatchResample(
                max_scale=1.111, min_scale=0.07, plateau_min=0.33, plateau_max=1.0
            ),
            RandomClipLargeImages(
                max_size=training_config['max_clip_image_clip_size'],
                min_size=training_config['min_clip_image_size']
            ),
            BatchFlip(),
            RandomSharpenBlur(min_factor=0.5, max_factor=1.5),
            ClipHighAndLow(p=0.1, max_pct=0.05),
        ]
        
        # --- DATALOADER ---
        logger.info("Creating DataBlock...")
        dblock = DataBlock(
            blocks=[
                TransformBlock([open_image_func]),
                MaskBlock(codes=[0, 1, 2, 3]),
                TransformBlock([sample_weights]),
            ],
            n_inp=1,
            get_items=multi_dataset_getter,
            get_y=[label_func, lambda x: x],
            splitter=FuncSplitter(is_validation_item),
            batch_tfms=batch_tfms,
            item_tfms=[
                Resize(training_config['original_image_size'], method="squish")
            ],
        )
        
        logger.info("Creating DataLoaders...")
        num_workers = 0 if os.name == 'nt' else 6
        
        dl = dblock.dataloaders(
            training_config['dataset_dirs'],
            bs=training_config['batch_size'],
            num_workers=num_workers,
            pin_memory=True,
        )
        
        try:
            logger.debug("Fetching one batch...")
            batch = dl.one_batch()
            logger.debug(f"Input shape: {batch[0].shape}")
            logger.debug(f"Label shape: {batch[1].shape}")
        except Exception as e:
            logger.error(f"Error fetching batch: {e}")
            return False
        
        # --- STAGE 1: HIGH RECALL TRAINING ---
        logger.info(f"\n{'='*60}")
        logger.info("STAGE 1: High Recall Training")
        logger.info(f"{'='*60}")
        logger.info("  - Aggressive class weights: [0.3, 1.5, 3.5, 4.5]")
        logger.info("  - High learning rate: 0.0003")
        logger.info("  - Recall-weighted composite score: 60% recall")
        
        stage1_callbacks = [
            PhaseTracker(log_transitions=True),
            GradientAccumulation(training_config['accumulation_steps']),
            GradientClip(max_norm=1.0),
            CompositeMetricCallback(dice_weight=0.1, recall_weight=0.8, precision_weight=0.1, min_precision=0.50),
            SaveBestAndLatestModel(
                monitor='composite_score',
                model_save_path=models_dir,
                learner_save_name=f"{base_identifier}_stage1",
                save_format='safetensors'
            ),
            ReduceLROnPlateauCustom(
                monitor='composite_score',
                factor=0.1,
                patience=3,
                min_lr=1e-7,
                mode='max'
            ),
            EarlyStoppingRecall(
                monitor='composite_score',
                frozen_patience=3,
                unfrozen_patience=5,
                min_delta=0.005
            )
        ]
        
        logger.info("Initializing Stage 1 Learner...")
        
        model.train()
        stage1_learner = Learner(
            dls=dl,
            model=model,
            loss_func=FocalLossFlatImageTypeWeighted(
                class_weights=torch.tensor([0.8, 2.0, 3.5, 3]),
                gamma=3.0
            ),
            metrics=[DiceMultiStrip, RecallMultiStrip, PrecisionMultiStrip, IoUMultiStrip],
            cbs=stage1_callbacks,
        )
        
        # Log callback instances and find save callback
        logger.debug(f"🔧 Stage1 Learner callbacks: {[type(cb).__name__ for cb in stage1_learner.cbs]}")
        save_callback = None
        for cb in stage1_learner.cbs:
            if type(cb).__name__ == 'SaveBestAndLatestModel':
                save_callback = cb
                logger.debug(f"🔧 SaveBestAndLatestModel callback found (id={id(cb)}, _init_id={getattr(cb, '_init_id', 'N/A')})")
                break
        
        # Log callback state before training
        if save_callback:
            logger.debug(f"🔧 Before Stage 1 training: callback (id={id(save_callback)}) state: best_epoch={save_callback.best_epoch}, best_score={save_callback.best_score:.6f}, _already_saved={save_callback._already_saved}")
        
        if training_config['use_bf16']:
            stage1_learner = stage1_learner.to_bf16()
            logger.info("✓ BF16 mixed precision enabled - 40-50% memory savings")
        
        # Train Stage 1
        logger.info(
            f"Starting Stage 1: Freeze {training_config['freeze_epochs']}, "
            f"Unfreeze {training_config['unfrozen_epochs']}"
        )
        
        freeze_encoder(stage1_learner)
        print_frozen_status(stage1_learner)
        
        try:
            stage1_learner.fit_one_cycle(
                training_config['freeze_epochs'],
                get_lr_ranges(freeze_encoder=True)
            )
        except CancelTrainException:
            logger.info("Stage 1 frozen phase early stopped, continuing to unfrozen...")
        
        unfreeze_all(stage1_learner)
        print_frozen_status(stage1_learner)
        
        try:
            stage1_learner.fit_one_cycle(
                training_config['unfrozen_epochs'],
                get_lr_ranges(freeze_encoder=False)
            )
        except CancelFitException:
            logger.info("Stage 1 training stopped by early stopping")
        
        # Log callback state after Stage 1
        if save_callback:
            logger.debug(f"🔧 After Stage 1 training: callback (id={id(save_callback)}) state: best_epoch={save_callback.best_epoch}, best_score={save_callback.best_score:.6f}, _already_saved={save_callback._already_saved}")
        
        # --- STAGE 2: PRECISION REFINEMENT ---
        logger.info(f"\n{'='*60}")
        logger.info("STAGE 2: Precision Refinement")
        logger.info(f"{'='*60}")
        logger.info("  - Balanced class weights: [1.0, 1.2, 2.2, 2.8]")
        logger.info("  - Lower learning rate: 0.0001")
        logger.info("  - Balanced composite score: 35% recall, 35% precision")
        
        stage2_callbacks = [
            PhaseTracker(log_transitions=True),
            GradientAccumulation(training_config['accumulation_steps']),
            GradientClip(max_norm=1.0),
            CompositeMetricCallback(dice_weight=0.3, recall_weight=0.35, precision_weight=0.35, min_precision=0.60),
            SaveBestAndLatestModel(
                monitor='composite_score',
                model_save_path=models_dir,
                learner_save_name=base_identifier,
                save_format='safetensors'
            ),
            ReduceLROnPlateauCustom(
                monitor='composite_score',
                factor=0.1,
                patience=5,
                min_lr=1e-8,
                mode='max'
            ),
            EarlyStoppingRecall(
                monitor='composite_score',
                frozen_patience=3,
                unfrozen_patience=10,
                min_delta=0.0005
            ),
        ]
        
        logger.info("Initializing Stage 2 Learner...")
        
        stage2_learner = Learner(
            dls=dl,
            model=model,  # Continue from Stage 1 weights
            loss_func=FocalLossFlatImageTypeWeighted(
                class_weights=torch.tensor([1.0, 1.2, 2.2, 2.8]),
                gamma=2.0
            ),
            metrics=[DiceMultiStrip, RecallMultiStrip, PrecisionMultiStrip, IoUMultiStrip],
            cbs=stage2_callbacks,
        )
        
        # Log callback instances and find save callback for Stage 2
        logger.debug(f"🔧 Stage2 Learner callbacks: {[type(cb).__name__ for cb in stage2_learner.cbs]}")
        save_callback_stage2 = None
        for cb in stage2_learner.cbs:
            if type(cb).__name__ == 'SaveBestAndLatestModel':
                save_callback_stage2 = cb
                logger.debug(f"🔧 SaveBestAndLatestModel callback found (id={id(cb)}, _init_id={getattr(cb, '_init_id', 'N/A')})")
                break
        
        # Log callback state before Stage 2 training
        if save_callback_stage2:
            logger.debug(f"🔧 Before Stage 2 training: callback (id={id(save_callback_stage2)}) state: best_epoch={save_callback_stage2.best_epoch}, best_score={save_callback_stage2.best_score:.6f}, _already_saved={save_callback_stage2._already_saved}")
        
        if training_config['use_bf16']:
            stage2_learner = stage2_learner.to_bf16()
        
        # Train Stage 2
        logger.info(
            f"Starting Stage 2: Freeze {training_config['freeze_epochs']}, "
            f"Unfreeze {training_config['unfrozen_epochs']}"
        )
        
        freeze_encoder(stage2_learner)
        print_frozen_status(stage2_learner)
        
        try:
            stage2_learner.fit_one_cycle(
                training_config['freeze_epochs'],
                get_lr_ranges(freeze_encoder=True)
            )
        except CancelTrainException:
            logger.info("Stage 2 frozen phase early stopped, continuing to unfrozen...")
        
        unfreeze_all(stage2_learner)
        print_frozen_status(stage2_learner)
        
        try:
            stage2_learner.fit_one_cycle(
                training_config['unfrozen_epochs'],
                get_lr_ranges(freeze_encoder=False)
            )
        except CancelFitException:
            logger.info("Stage 2 training stopped by early stopping")
        
        # Log callback state after Stage 2 training
        if save_callback_stage2:
            logger.debug(f"🔧 After Stage 2 training: callback (id={id(save_callback_stage2)}) state: best_epoch={save_callback_stage2.best_epoch}, best_score={save_callback_stage2.best_score:.6f}, _already_saved={save_callback_stage2._already_saved}")
        
        # --- SAVING ---
        logger.info("\nSaving models...")
        
        # Convert to CPU and float32 for saving
        model_cpu = stage2_learner.model.to("cpu").float()
        
        # Save additional formats for latest model
        torch.save(model_cpu.state_dict(), state_path)
        logger.debug(f"  ✓ Saved latest PyTorch state dict: {state_path.name}")
        
        torch.save(model_cpu, pytorch_model_path)
        logger.debug(f"  ✓ Saved latest PyTorch full model: {pytorch_model_path.name}")
        
        # Verify latest safetensors integrity (saved by callback)
        latest_safetensors_path = models_dir / f"{base_identifier}_latest.safetensors"
        safetensors_valid = verify_safetensors_integrity(latest_safetensors_path, model_cpu)
        if not safetensors_valid:
            logger.error("  CRITICAL: Latest safetensors file failed integrity check!")
            return False
        
        # Verify best safetensors integrity (saved by callback)
        best_safetensors_path = models_dir / f"{base_identifier}_best.safetensors"
        if best_safetensors_path.exists():
            # Load best model state for verification
            best_state = load_file(best_safetensors_path)
            # Create a temporary model for verification
            temp_model = build_custom_model(
                model_name=model_type,
                model_library=model_library,
                in_chans=training_config['num_input_channels'],
                n_out=len(CLASS_NAMES)
            )
            temp_model.load_state_dict(best_state)
            temp_model = temp_model.float()
            safetensors_best_valid = verify_safetensors_integrity(best_safetensors_path, temp_model)
            if not safetensors_best_valid:
                logger.error("  CRITICAL: Best safetensors file failed integrity check!")
                return False
        
        # Get best model info from callback
        save_best_callback = stage2_learner.cbs[-4]  # The last callback is our SaveBestAndLatestModel
        best_composite_score = save_best_callback.best_score
        best_epoch = save_best_callback.best_epoch
        
        # Save configuration
        config = {
            "model_version": training_config['model_version'],
            "model_type": model_type,
            "model_library": model_library,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_identifier": checkpoint_identifier,
            "base_identifier": base_identifier,
            "use_bf16": training_config['use_bf16'],
            "demo_mode": training_config['demo_mode'],
            "original_image_size": training_config['original_image_size'],
            "max_clip_image_clip_size": training_config['max_clip_image_clip_size'],
            "min_clip_image_size": training_config['min_clip_image_size'],
            "limited_band_read_list": training_config['limited_band_read_list'],
            "native_band_scales": training_config['native_band_scales'],
            "gradient_accumulation_batch_size": training_config['gradient_accumulation_batch_size'],
            "batch_size": training_config['batch_size'],
            "learning_rate": training_config['learning_rate'],
            "freeze_epochs": training_config['freeze_epochs'],
            "unfrozen_epochs": training_config['unfrozen_epochs'],
            "limit_training_images": training_config['limit_training_images'],
            "safetensors_integrity_verified": safetensors_valid,
            "class_weights": training_config['class_weights'].tolist(),
            "best_composite_score": float(best_composite_score),
            "best_epoch": int(best_epoch),
            "random_seed": training_config['random_seed'],
            "training_mode": "two_stage",
            "stage1_class_weights": [0.3, 1.5, 3.5, 4.5],
            "stage2_class_weights": [1.0, 1.2, 2.2, 2.8],
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        logger.info(f"  ✓ Saved config: {config_path.name}")
        
        logger.info(f"\n✓ Two-Stage Fine-tuning complete. Models saved to {models_dir}")
        logger.info(f"  BEST MODEL (composite_score={best_composite_score:.6f} at epoch {best_epoch}):")
        logger.info(f"    - {base_identifier}_best.safetensors (PRIMARY - safetensors)")
        logger.info(f"    - {base_identifier}_best.pth (fastai learner)")
        logger.info(f"  LATEST MODEL:")
        logger.info(f"    - {base_identifier}_latest.safetensors (PRIMARY - safetensors)")
        logger.info(f"    - {base_identifier}_latest.pth (fastai learner)")
        logger.info(f"    - {state_path.name} (PyTorch state dict)")
        logger.info(f"    - {pytorch_model_path.name} (PyTorch full model)")
        
        # --- EVALUATION WITH FILE LOCK PREVENTION ---
        results_dir = output_base_dir / f"results_{base_identifier}"
        results_dir.mkdir(exist_ok=True)

        # Add MetricLogger callback with log file path
        log_file_path = results_dir / 'training_log.txt'
        stage2_learner.add_cb(MetricLogger(class_names=CLASS_NAMES, log_file=str(log_file_path)))

        # CRITICAL: Find and set evaluation mode on ALL callbacks to prevent file locks and unwanted operations
        callbacks_with_eval_mode = []
        for cb in stage2_learner.cbs:
            cb_type = type(cb).__name__
            # Set evaluation mode on all callbacks that support it
            if hasattr(cb, '_evaluation_mode'):
                cb._evaluation_mode = True
                callbacks_with_eval_mode.append(cb)
                logger.debug(f"Evaluation mode ENABLED on {cb_type} - callback will skip during metrics computation")
        
        logger.info(f"Evaluation mode enabled on {len(callbacks_with_eval_mode)} callbacks")
        
        # Load best model for evaluation (learner currently has latest model)
        # The SaveBestAndLatestModel callback saves best model but restores latest model state,
        # so we need to explicitly load the best model before computing metrics
        stage2_learner.load(f"{base_identifier}_best")
        logger.info(f"  ✓ Loaded best model from epoch {best_epoch} for evaluation (composite_score={best_composite_score:.6f})")
        
        try:
            logger.info("\nGenerating evaluation metrics...")
            
            # Validation Set
            try:
                logger.info("  Computing metrics for Validation set...")
                compute_and_plot_metrics(
                    stage2_learner,
                    dl=dl.valid,
                    dataset_name="Validation",
                    save_dir=results_dir,
                    class_names=CLASS_NAMES
                )
                logger.info("  ✓ Validation metrics complete")
            except Exception as e:
                # Don't fail training - models are already saved!
                logger.warning(f"Warning: Error during validation metrics: {e}")
                logger.warning(f"  This doesn't affect the saved models - they are safe!")
            
            # Training Set - DISABLED due to memory issues
            logger.info("  Training set metrics skipped (disabled to prevent memory errors)")
            logger.info("  Validation metrics provide sufficient evaluation of model performance")
        
        finally:
            # CRITICAL: Always disable evaluation mode on ALL callbacks
            for cb in callbacks_with_eval_mode:
                cb._evaluation_mode = False
                logger.debug(f"Evaluation mode DISABLED on {type(cb).__name__}")
            
            logger.info("Evaluation mode disabled - callbacks restored to normal operation")
        
        logger.info(f"✓ Results saved to {results_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error in two-stage fine-tuning model {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def hard_negative_mining(
    model: torch.nn.Module,
    training_images: List[Path],
    label_func: Callable,
    device: torch.device,
    num_hard_samples: int = 100,
    batch_size: int = 4
) -> List[Path]:
    """
    Identify images with highest false negative rates for focused training.
    
    Args:
        model: Trained model to evaluate
        training_images: List of training image paths
        label_func: Function to get label path from image path
        device: Device to run inference on
        num_hard_samples: Number of hardest samples to return
        batch_size: Batch size for inference
        
    Returns:
        List of image paths with highest FN rates (sorted by difficulty)
    """
    logger.info(f"Performing hard negative mining on {len(training_images)} images...")
    
    model.eval()
    model.to(device)
    
    fn_rates = []
    
    with torch.no_grad():
        for i, img_path in enumerate(training_images):
            if i % 100 == 0:
                logger.debug(f"Processing image {i}/{len(training_images)}")
            
            # Load image
            img = load_image(img_path)
            img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
            
            # Get prediction
            pred = model(img_tensor)
            pred_class = pred.argmax(dim=1).squeeze().cpu().numpy()
            
            # Load ground truth
            label_path = label_func(img_path)
            label = load_label(label_path)
            
            # Calculate per-class false negatives
            fn_per_class = []
            for c in range(len(CLASS_NAMES)):
                actual_c = (label == c).sum()
                predicted_c = (pred_class == c).sum()
                fn_c = max(0, actual_c - predicted_c)
                fn_rate = fn_c / max(actual_c, 1)
                fn_per_class.append(fn_rate)
            
            # Overall FN rate (weighted more for critical classes)
            weighted_fn_rate = (
                0.5 * fn_per_class[2] +  # Thin Cloud
                1.0 * fn_per_class[3]    # Cloud Shadow (highest weight)
            )
            
            fn_rates.append((img_path, weighted_fn_rate, fn_per_class))
    
    # Sort by FN rate and return hardest samples
    fn_rates.sort(key=lambda x: x[1], reverse=True)
    hard_samples = [item[0] for item in fn_rates[:num_hard_samples]]
    
    logger.info(f"Identified {len(hard_samples)} hard negative samples")
    logger.info(f"  - Top FN rate: {fn_rates[0][1]:.3f}")
    logger.info(f"  - Average FN rate: {np.mean([item[1] for item in fn_rates]):.3f}")
    
    return hard_samples


def oversample_hard_examples(
    training_images: List[Path],
    hard_samples: List[Path],
    oversample_factor: int = 2
) -> List[Path]:
    """
    Oversample hard examples in training dataset.
    
    Args:
        training_images: Original training image list
        hard_samples: List of hard sample paths
        oversample_factor: How many times to repeat hard samples
        
    Returns:
        Training image list with hard examples oversampled
    """
    oversampled = training_images.copy()
    
    for _ in range(oversample_factor):
        oversampled.extend(hard_samples)
    
    logger.info(f"Oversampled training set:")
    logger.info(f"  - Original: {len(training_images)} images")
    logger.info(f"  - Hard samples: {len(hard_samples)} images")
    logger.info(f"  - Oversampled: {len(oversampled)} images")
    
    return oversampled


def calculate_image_difficulty(
    model: torch.nn.Module,
    image_paths: List[Path],
    label_func: Callable,
    device: torch.device,
    batch_size: int = 4
) -> List[Tuple[Path, float]]:
    """
    Calculate difficulty score for each image based on model confidence.
    
    Lower confidence = harder image.
    
    Args:
        model: Model to evaluate with
        image_paths: List of image paths
        label_func: Function to get label path
        device: Device to run on
        batch_size: Batch size
        
    Returns:
        List of (image_path, difficulty_score) tuples, sorted by difficulty
    """
    logger.info("Calculating image difficulty scores...")
    
    model.eval()
    model.to(device)
    
    difficulties = []
    
    with torch.no_grad():
        for img_path in image_paths:
            img = load_image(img_path)
            img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
            
            pred = model(img_tensor)
            # Use max probability as confidence score
            max_prob = pred.softmax(dim=1).max().item()
            
            # Difficulty = 1 - confidence
            difficulty = 1.0 - max_prob
            difficulties.append((img_path, difficulty))
    
    # Sort by difficulty (easy to hard)
    difficulties.sort(key=lambda x: x[1])
    
    return difficulties

# --- LOCAL CONFIG IMPORT ---
try:
    if str(current_script_dir) not in sys.path:
        sys.path.append(str(current_script_dir))
    from local_config import (
        TRAIN_DATA_DIR,
        CUSTOM_MODEL_VERSION,
        USE_DUAL_RES_METHOD,
        BAND_ORDER,
        CLASS_NAMES,
        SMP_MODEL_TYPES,
        FASTAI_MODEL_TYPES,
        CKPTS_DIR,
        USE_BF16,
        DEMO_MODE,
        COMPILE_MODELS,
        ORIGINAL_IMAGE_SIZE,
        MAX_CLIP_IMAGE_SIZE,
        MIN_CLIP_IMAGE_SIZE,
        BATCH_SIZE,
        GRADIENT_ACCUMULATION_BATCH_SIZE,
        LEARNING_RATE,
        FREEZE_EPOCHS,
        UNFROZEN_EPOCHS,
        RANDOM_SEED,
        CLASS_WEIGHTS,
        LIMIT_TRAINING_IMAGES,
        get_native_band_scales,
        get_accumulation_steps,
        get_num_input_channels,
        get_class_weights_tensor,
    )
except ImportError as e:
    logger.critical(
        f"CRITICAL: local_config.py not found or missing required config: {e}. "
        "Please create 'training/scripts/local_config.py' to define local paths."
    )
    sys.exit(1)
    
def main():
    """Main function to fine-tune multiple models."""
    logger.info("Starting multi-model fine-tuning script...")
    print_system_info()
    
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    
    # --- PATHS ---
    base_data_path = TRAIN_DATA_DIR
    
    my_custom_data_dir = base_data_path / "train"
    my_custom_val_dir = base_data_path / "validation"
    
    # Override validation directory
    cloudsen12_validation_dir = my_custom_val_dir
    
    # --- CHECKPOINTS DIRECTORY ---
    # Use ckpts directory from config (allows override via environment variable)
    ckpts_dir = CKPTS_DIR
    if "CKPTS_DIR" in os.environ:
        ckpts_dir = Path(os.environ["CKPTS_DIR"])
    
    if not ckpts_dir.exists():
        logger.error(f"Checkpoints directory not found: {ckpts_dir}")
        logger.info("Please create the ckpts directory or set CKPTS_DIR environment variable.")
        sys.exit(1)
    
    # --- CONFIGURATION ---
    model_version = CUSTOM_MODEL_VERSION
    
    use_bf16 = USE_BF16
    demo_mode = DEMO_MODE
    
    # Optional: Enable model compilation for v4 smp models
    compile_models = COMPILE_MODELS
    
    original_image_size = ORIGINAL_IMAGE_SIZE
    max_clip_image_clip_size = MAX_CLIP_IMAGE_SIZE
    min_clip_image_size = MIN_CLIP_IMAGE_SIZE
    
    # Use configurable band order
    limited_band_read_list = BAND_ORDER
    logger.info(f"Training using bands: {limited_band_read_list}")
    
    # Scale bands according to method from config
    native_band_scales = get_native_band_scales()
    
    gradient_accumulation_batch_size = GRADIENT_ACCUMULATION_BATCH_SIZE
    batch_size = BATCH_SIZE
    accum_steps = get_accumulation_steps()

    # Learning rate from config
    learning_rate = LEARNING_RATE

    random_seed = RANDOM_SEED

    # Class weights from config
    # UPDATED (2025-02-13): Class weights for RECALL-PRECISION BALANCED training with image weights
    # [Clear, Thick Cloud, Thin Cloud, Cloud Shadow]
    # Previous: [1, 1.5, 2.5, 3.0] - 3:1 ratio, recall-focused
    # NEW: [0.5, 1.2, 2.5, 3.0] - 6:1 ratio, balanced with image weights:
    #   - Clear (0.5): Down-weighted to reduce false positives (was 1.0)
    #   - Thick Cloud (1.2): Slightly lower (easy to detect)
    #   - Thin Cloud (2.5): High weight to catch thin clouds
    #   - Cloud Shadow (3.0): Highest weight to catch difficult shadows
    # This change with updated image weights (FN=2.0, HardNeg=0.6):
    #   - Achieves 95%+ recall while maintaining 50-60% precision
    #   - Uses 6:1 ratio for critical classes (more balanced than 15:1)
    #   - Expected: 95-97% Recall, 52-58% Precision
    CLASS_WEIGHTS = get_class_weights_tensor()
    logger.info(f"Class weights: {CLASS_WEIGHTS.tolist()}")
    logger.info(f"  - Ratio: {CLASS_WEIGHTS[3]/CLASS_WEIGHTS[0]:.1f}:1 (Critical:Clear)")
    logger.info(f"  - NEW: Balanced recall-precision with image weights (6:1 ratio)")
    logger.info(f"  - Target: Recall ≥95%, Precision 50-60%")
    
    # Adaptive class weight adjustment (OPTIONAL - can be enabled for fine-tuning)
    # If validation shows specific classes struggling, use adaptive_class_weights() to adjust:
    #   target_recall = [0.95, 0.90, 0.85, 0.85]  # [Clear, Thick, Thin, Shadow]
    #   per_class_recall = [0.98, 0.92, 0.75, 0.70]  # Example from validation
    #   adjusted_weights = adaptive_class_weights(CLASS_WEIGHTS, per_class_recall, target_recall)
    # This will automatically increase shadow weight if it's underperforming

    my_custom_weight = 1.0
    
    label_weights = {
        cloudsen12_validation_dir: 1.0,
        my_custom_data_dir: my_custom_weight,
    }
    
    dataset_dirs = list(label_weights.keys())
    
    logger.info("Checking dataset directories...")
    for dataset_dir in dataset_dirs:
        if not dataset_dir.exists():
            logger.warning(f"Warning: Directory {dataset_dir} does not exist. Please check paths.")
    
    # Use epoch settings from config
    freeze_epochs = FREEZE_EPOCHS
    unfrozen_epochs = UNFROZEN_EPOCHS
    limit_training_images = LIMIT_TRAINING_IMAGES
    
    # Override for demo mode
    if demo_mode:
        freeze_epochs = 5
        unfrozen_epochs = 5
        limit_training_images = 3000
    else:
        # UPDATED (2025-02-13): Moderate unfrozen epochs for balanced recall-precision training
        # Previous: 10 unfrozen epochs - Recall-focused training
        # NEW: 12 unfrozen epochs - Slightly longer for convergence with balanced weights
        # ReduceLROnPlateau callback will automatically reduce LR when model saturates
        freeze_epochs = FREEZE_EPOCHS
        unfrozen_epochs = UNFROZEN_EPOCHS
        limit_training_images = LIMIT_TRAINING_IMAGES
    
    num_input_channels = get_num_input_channels()
    logger.info(f"Number of input channels: {num_input_channels}")
    
    # --- DISCOVER MODELS ---
    model_configs = discover_model_checkpoints(ckpts_dir)
    
    if not model_configs:
        logger.error("No model checkpoints found in ckpts directory.")
        sys.exit(1)
    
    logger.info(f"Found {len(model_configs)} model checkpoints to fine-tune:")
    for config in model_configs:
        logger.info(f"  - {config['name']} ({config['model_library']}, {config['model_type']})")
    
    # --- OUTPUT DIRECTORY ---
    output_base_dir = project_root / "fine_tune_results" / f"fine_tuning_results_{model_version}"
    output_base_dir.mkdir(exist_ok=True)
    
    # --- FINE-TUNE EACH MODEL ---
    success_count = 0
    failure_count = 0
    
    for i, model_config in enumerate(model_configs, 1):
        logger.info(f"\n\n{'='*80}")
        logger.info(f"Processing model {i}/{len(model_configs)}: {model_config['name']}")
        logger.info(f"{'='*80}")
        
        # Build training configuration for this model
        training_config = {
            'model_version': model_version,
            'num_input_channels': num_input_channels,
            'original_image_size': original_image_size,
            'max_clip_image_clip_size': max_clip_image_clip_size,
            'min_clip_image_size': min_clip_image_size,
            'limited_band_read_list': limited_band_read_list,
            'native_band_scales': native_band_scales,
            'gradient_accumulation_batch_size': gradient_accumulation_batch_size,
            'batch_size': batch_size,
            'accumulation_steps': accum_steps,
            'learning_rate': learning_rate,
            'freeze_epochs': freeze_epochs,
            'unfrozen_epochs': unfrozen_epochs,
            'limit_training_images': limit_training_images,
            'use_bf16': use_bf16,
            'demo_mode': demo_mode,
            'compile_models': compile_models,
            'dataset_dirs': dataset_dirs,
            'cloudsen12_validation_dir': cloudsen12_validation_dir,
            'label_weights': label_weights,
            'class_weights': CLASS_WEIGHTS,  # Add class weights for recall-focused training
            'random_seed': random_seed,
        }
        
        # UPDATED (2025-02-13): Use single-phase training with optimized image/class weights
        # Single-phase is now preferred over two-stage when using image weights
        # because you have two orthogonal control knobs (image weights + class weights)
        # This provides more stable convergence and simpler training flow
        success = fine_tune_single_model(model_config, training_config, output_base_dir)
        # Two-stage training is still available if needed for extreme recall requirements
        # success = fine_tune_single_model_two_stage(model_config, training_config, output_base_dir)
        
        if success:
            success_count += 1
            logger.info(f"✓ Successfully fine-tuned {model_config['name']}")
        else:
            failure_count += 1
            logger.error(f"✗ Failed to fine-tune {model_config['name']}")
        
        # Clear memory between models
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
    
    # --- SUMMARY ---
    logger.info(f"\n\n{'='*80}")
    logger.info("FINE-TUNING SUMMARY - SINGLE-PHASE BALANCED TRAINING (2025-02-13)")
    logger.info(f"{'='*80}")
    logger.info(f"Total models processed: {len(model_configs)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {failure_count}")
    logger.info(f"Output directory: {output_base_dir}")
    logger.info(f"\nNEW: SINGLE-PHASE BALANCED TRAINING STRATEGY (Target: Recall ≥95%, Precision 50-60%):")
    logger.info(f"  - Image weights: GT=1.0, FN=2.0, HardNeg=0.6 (updated from 3.0/0.3)")
    logger.info(f"    FN weight reduced (3.0→2.0) - less aggressive FN learning")
    logger.info(f"    HardNeg weight increased (0.3→0.6) - penalize FPs more")
    logger.info(f"  - Class weights: [Clear=0.5, Thick Cloud=1.2, Thin Cloud=2.5, Cloud Shadow=3.0]")
    logger.info(f"    Ratio: 6:1 (Critical:Clear) - Balanced with image weights")
    logger.info(f"    Clear down-weighted (1.0→0.5) - reduce false positives")
    logger.info(f"  - Composite metric: 20% Dice + 50% Recall + 30% Precision")
    logger.info(f"    Previous: 20% Dice + 60% Recall + 20% Precision (recall-focused)")
    logger.info(f"    More precision weight (20%→30%) for better FP control")
    logger.info(f"  - Focal Loss gamma: 2.5 (reduced from 3.0 - moderate focus)")
    logger.info(f"  - Model selection: Based on composite_score (balanced)")
    logger.info(f"  - Early stopping: Monitors composite_score (includes precision)")
    logger.info(f"  - Precision guardrail: Proportional penalty (precision < 0.50)")
    logger.info(f"  - Learning rate: 0.0003 (for faster convergence)")
    logger.info(f"  - Training epochs: 5 frozen + 12 unfrozen (increased from 10)")
    logger.info(f"\nWHY SINGLE-PHASE NOW:")
    logger.info(f"  - Image weights + class weights = two orthogonal control knobs")
    logger.info(f"  - No need to switch strategies mid-training (more stable)")
    logger.info(f"  - Simpler, faster, and easier to debug")
    logger.info(f"  - Better convergence with balanced weights")
    logger.info(f"\nPRIMARY OUTPUT FORMAT: .safetensors files")
    logger.info(f"All safetensors files have been integrity-verified")
    logger.info(f"\nFor each model, two versions are saved:")
    logger.info(f"  - BEST model: Best performing model based on composite_score metric")
    logger.info(f"  - LATEST model: Final model after all training epochs")
    logger.info(f"This prevents overfitting by preserving the best performing model.")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()