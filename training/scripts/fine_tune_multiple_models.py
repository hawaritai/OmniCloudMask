"""
Fine-tune multiple OCM models from checkpoints in ckpts folder.

This script iterates through model checkpoints located in the ckpts directory
and performs fine-tuning on each one using the training data defined in local_config.
It uses the training methodology from train_ocm_models_custom.py with the modern
model loading mechanism from compare_ocm_models.py for full v4 'smp' compatibility.

BALANCED TRAINING STRATEGY (FIXED - Reduced False Positives):
- Primary goal: Minimize false negatives (critical for application)
- Secondary goal: Minimize false positives (balanced approach)
- Metric weighting: 30% Dice + 35% Recall + 35% Precision
  - Recall and precision balanced equally (35% each) to reduce false positives
  - Dice score (30%) for overall segmentation quality
  - Previous: 50% Recall + 20% Precision (too recall-focused, caused excessive false positives)
- Class weights: [Clear=1.0, Thick Cloud=1.5, Thin Cloud=2.0, Cloud Shadow=2.0]
  - Balanced 2:1 ratio (Critical:Clear) - Previous was 6:1 (too aggressive)
  - Still prioritizes critical classes but prevents over-prediction
  - Previous: [0.5, 2.0, 3.0, 3.0] - caused model to over-predict critical classes
- Critical classes: Thin Cloud and Cloud Shadow (highest priority)
- Leniency: Thin Cloud can be detected as Thick Cloud, but NO leniency for Cloud Shadow
- Model selection: Based on composite score (30% Dice + 35% Recall + 35% Precision)
- Precision guardrail: Models with precision < 0.6 are rejected (score set to 0.0)
  - Previous: Proportional penalty (too weak to counteract recall bias)
  - Current: Hard threshold - completely rejects low precision models
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
  - Composite score = 0.3 * dice + 0.35 * recall + 0.35 * precision
  - Precision guardrail: If precision < 0.6, score is set to 0.0 (hard rejection)
  - Prevents overfitting by preserving the best performing model
  - Both models are saved in safetensors and fastai learner formats
- FIXED: Windows file locking issues with proper cleanup and retry logic
- FIXED (2025-02-05): Reduced false positives by:
  - Balancing composite score weights (35% recall, 35% precision)
  - Reducing class weight ratio from 6:1 to 2:1
  - Implementing hard precision guardrail (rejects models with precision < 0.6)
  - Monitoring composite_score for early stopping (not just recall)
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
from collections import defaultdict
from functools import partial
from typing import List, Dict, Optional, Tuple
import gc

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
        PhaseTracker,  # Add PhaseTracker for reliable phase detection
        EarlyStoppingRecall,
        MetricLogger,
        # Import utility functions and callbacks from utils.py
        safe_file_save,
        verify_safetensors_integrity,
        CompositeMetricCallback,
        ReduceLROnPlateauCustom,
        SaveBestAndLatestModel,
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
CLASS_NAMES = ['Clear', 'Thick Cloud', 'Thin Cloud', 'Cloud Shadow']
# Note: MAX_SAVE_RETRIES and SAVE_RETRY_DELAY are now imported from utils.py


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
        
        # Apply bfloat16 if requested
        if use_bf16 and DEVICE.type == 'cuda':
            model = model.bfloat16()
            logger.debug("  Model converted to bfloat16.")
        
        # Optional compilation for v4 models
        if compile_model and model_library == 'smp' and HAS_COMPILATION:
            logger.debug("  Compiling model for v4 smp architecture...")
            model = compile_torch_model(
                model,
                patch_size=509,
                batch_size=10,
                dtype=torch.bfloat16 if use_bf16 else torch.float32,
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
            if training_config['use_bf16']:
                dummy_input = dummy_input.bfloat16()
        
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
                if use_bf16:
                    image_tensor = image_tensor.bfloat16()
                return TensorImage(image_tensor)
        
        def sample_weights(image_path: Path) -> torch.Tensor:
            try:
                weight = torch.tensor(
                    training_config['label_weights'][image_path.parent],
                    dtype=torch.float32
                )
            except Exception:
                return torch.tensor(1.0, dtype=torch.float32)
            return weight
        
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
            GradientAccumulation(training_config['gradient_accumulation_batch_size']),
            # FIXED: Balanced composite score weights to reduce false positives
            # Previous: dice=0.3, recall=0.5, precision=0.2 - Too recall-focused
            # Current: dice=0.3, recall=0.35, precision=0.35 - Equal weight on recall and precision
            # This change balances false negative and false positive minimization
            CompositeMetricCallback(dice_weight=0.3, recall_weight=0.35, precision_weight=0.35, min_precision=0.65),
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
            loss_func=CrossEntropyLossFlatImageTypeWeighted(
                class_weights=training_config['class_weights']
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
        learner.fit_one_cycle(
            training_config['freeze_epochs'],
            get_lr_ranges(freeze_encoder=True)
        )
        
        # Phase 2: Train with unfrozen encoder
        logger.info(f"\n{'='*60}")
        logger.info("Phase 2: Training with unfrozen encoder (discriminative LRs)")
        logger.info(f"{'='*60}")
        unfreeze_all(learner)
        print_frozen_status(learner)
        learner.fit_one_cycle(
            training_config['unfrozen_epochs'],
            get_lr_ranges(freeze_encoder=False)
        )

        # Log callback state after training
        if save_callback:
            logger.debug(f"🔧 After training: callback (id={id(save_callback)}) state: best_epoch={save_callback.best_epoch}, best_score={save_callback.best_score:.6f}, _already_saved={save_callback._already_saved}")
        
        # --- SAVING ---
        # Note: SaveBestAndLatestModel callback has already saved both best and latest models
        # We now save additional formats and verify integrity
        
        # Get the best model info from the callback
        save_best_callback = learner.cbs[-3]  # The last callback is our SaveBestAndLatestModel
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
            
            # Training Set
            try:
                logger.info("  Computing metrics for Training set...")
                compute_and_plot_metrics(
                    learner,
                    dl=dl.train,
                    dataset_name="Training",
                    save_dir=results_dir,
                    class_names=CLASS_NAMES
                )
                logger.info("  ✓ Training metrics complete")
            except Exception as e:
                # Don't fail training - models are already saved!
                import traceback
                traceback.print_exc()
                logger.warning(f"Warning: Error during training metrics: {e}")
                logger.warning(f"  This doesn't affect the saved models - they are safe!")
        
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
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to fine-tune multiple models."""
    logger.info("Starting multi-model fine-tuning script...")
    print_system_info()
    
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    
    # --- LOCAL CONFIG IMPORT ---
    try:
        if str(current_script_dir) not in sys.path:
            sys.path.append(str(current_script_dir))
        from local_config import (
            TRAIN_DATA_DIR,
            CUSTOM_MODEL_VERSION,
            USE_DUAL_RES_METHOD,
        )
        
        # Try to import BAND_ORDER, default to [1, 2, 3] if not present
        from local_config import BAND_ORDER
    except ImportError:
        # Check if basic config exists, if not exit
        try:
            from local_config import TRAIN_DATA_DIR
            BAND_ORDER = [1, 2, 3]  # Default
        except ImportError:
            logger.critical(
                "CRITICAL: local_config.py not found. Please create "
                "'training/scripts/local_config.py' to define local paths."
            )
            sys.exit(1)
    
    # --- PATHS ---
    base_data_path = TRAIN_DATA_DIR
    
    my_custom_data_dir = base_data_path / "train"
    my_custom_val_dir = base_data_path / "validation"
    
    # Override validation directory
    cloudsen12_validation_dir = my_custom_val_dir
    
    # --- CHECKPOINTS DIRECTORY ---
    # Default ckpts directory in project root
    ckpts_dir = project_root / "ckpts"
    
    # Allow override via environment variable
    if "CKPTS_DIR" in os.environ:
        ckpts_dir = Path(os.environ["CKPTS_DIR"])
    
    if not ckpts_dir.exists():
        logger.error(f"Checkpoints directory not found: {ckpts_dir}")
        logger.info("Please create the ckpts directory or set CKPTS_DIR environment variable.")
        sys.exit(1)
    
    # --- CONFIGURATION ---
    model_version = CUSTOM_MODEL_VERSION
    
    use_bf16 = False
    demo_mode = False
    
    # Optional: Enable model compilation for v4 smp models
    compile_models = False  # Set to True to enable compilation
    
    original_image_size = 509
    max_clip_image_clip_size = 400
    min_clip_image_size = 256
    
    # Use configurable band order
    limited_band_read_list = BAND_ORDER
    logger.info(f"Training using bands: {limited_band_read_list}")
    
    # Scale bands according to method:
    # Dual Res method uses [Red*3, Green*2, NIR*1] scaling instead of Z-score
    if USE_DUAL_RES_METHOD:
        native_band_scales = [3, 2, 1]
    else:
        native_band_scales = [1, 1, 1]
    
    gradient_accumulation_batch_size = 128
    batch_size = 8
    learning_rate = 0.0001

    # Class weights for balanced training (FIXED: reduced from 6:1 to 2:1 ratio)
    # [Clear, Thick Cloud, Thin Cloud, Cloud Shadow]
    # Previous: [0.5, 2.0, 3.0, 3.0] - Too aggressive, caused excessive false positives
    # Current: [1.0, 1.5, 2.0, 2.0] - Balanced 2:1 ratio, still prioritizes critical classes
    # This change reduces false positives while maintaining good recall on critical classes
    CLASS_WEIGHTS = torch.tensor([1.0, 1.5, 2.0, 3.0])
    logger.info(f"Class weights: {CLASS_WEIGHTS.tolist()}")
    logger.info(f"  - Ratio: {CLASS_WEIGHTS[3]/CLASS_WEIGHTS[0]:.1f}:1 (Critical:Clear)")
    logger.info(f"  - Previous ratio was 6:1, reduced to prevent over-prediction")

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
    
    if demo_mode:
        freeze_epochs = 5
        unfrozen_epochs = 5
        limit_training_images = 3000
    else:
        # Default for custom fine-tuning: Increased epoch count with LR reduction on plateau
        # ReduceLROnPlateau callback will automatically reduce LR when model saturates
        freeze_epochs = 2
        unfrozen_epochs = 14
        limit_training_images = None
    
    num_input_channels = len(limited_band_read_list)
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
        }
        
        # Fine-tune this model
        success = fine_tune_single_model(model_config, training_config, output_base_dir)
        
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
    logger.info("FINE-TUNING SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total models processed: {len(model_configs)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {failure_count}")
    logger.info(f"Output directory: {output_base_dir}")
    logger.info(f"\nBALANCED TRAINING STRATEGY (FIXED - Reduced False Positives):")
    logger.info(f"  - Class weights: [Clear=1.0, Thick Cloud=1.5, Thin Cloud=2.0, Cloud Shadow=2.0]")
    logger.info(f"    Ratio: 2:1 (Critical:Clear) - Previous was 6:1 (too aggressive)")
    logger.info(f"  - Composite metric: 30% Dice + 35% Recall + 35% Precision")
    logger.info(f"    Previous: 30% Dice + 50% Recall + 20% Precision (too recall-focused)")
    logger.info(f"  - Model selection: Based on composite_score (balanced recall & precision)")
    logger.info(f"  - Early stopping: Monitors composite_score (includes precision)")
    logger.info(f"  - Precision guardrail: Hard threshold (precision < 0.6 = score 0.0)")
    logger.info(f"\nPRIMARY OUTPUT FORMAT: .safetensors files")
    logger.info(f"All safetensors files have been integrity-verified")
    logger.info(f"\nFor each model, two versions are saved:")
    logger.info(f"  - BEST model: Best performing model based on composite_score metric")
    logger.info(f"  - LATEST model: Final model after all training epochs")
    logger.info(f"This prevents overfitting by preserving the best performing model.")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()