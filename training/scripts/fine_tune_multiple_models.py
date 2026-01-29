"""
Fine-tune multiple OCM models from checkpoints in ckpts folder.

This script iterates through model checkpoints located in the ckpts directory
and performs fine-tuning on each one using the training data defined in local_config.
It uses the training methodology from train_ocm_models_custom.py with the modern
model loading mechanism from compare_ocm_models.py for full v4 'smp' compatibility.

IMPROVEMENTS:
- Output filenames preserve unique identifiers from checkpoint names (no overwriting)
- Clean, professional naming without redundant suffixes
- Safetensors as the primary output format (emphasized in logging)
- Better error checking for safetensors file integrity
"""

import sys
import os
from pathlib import Path
import warnings
import json
import numpy as np
import random
import cv2
import logging
from collections import defaultdict
from functools import partial
from typing import List, Dict, Optional

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
from fastai.vision.all import *
from safetensors.torch import save_file, load_file
import timm
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning

# Import modern model loading utilities
from custom_model_utils import build_custom_model, load_custom_weights

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
        CrossEntropyLossFlatImageTypeWeighted,
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
        
        # Apply bfloat16 if requested
        if use_bf16 and DEVICE.type == 'cuda':
            model = model.bfloat16()
            logger.info("  Model converted to bfloat16.")
        
        # Optional compilation for v4 models
        if compile_model and model_library == 'smp' and HAS_COMPILATION:
            logger.info("  Compiling model for v4 smp architecture...")
            model = compile_torch_model(
                model,
                patch_size=509,
                batch_size=10,
                dtype=torch.bfloat16 if use_bf16 else torch.float32,
                device=DEVICE,
                compile_mode="default"
            )
            logger.info("  Model compiled successfully.")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


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
        
        logger.info("  ✓ Safetensors integrity check passed")
        return True
        
    except Exception as e:
        logger.error(f"  Safetensors integrity check failed: {e}")
        return False


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
        logger.info(f"Output base name: {base_identifier}")
        logger.info(f"PRIMARY OUTPUT -> Safetensors: {safetensor_state_path.name}")
        logger.info(f"  Fastai model: {fai_model_name}")
        logger.info(f"  PyTorch state dict: {state_path.name}")
        logger.info(f"  PyTorch full model: {pytorch_model_path.name}")
        logger.info(f"  Config JSON: {config_path.name}")
        
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
            logger.info("Fetching one batch...")
            batch = dl.one_batch()
            logger.debug(f"Input shape: {batch[0].shape}")
            logger.debug(f"Label shape: {batch[1].shape}")
        except Exception as e:
            logger.error(f"Error fetching batch: {e}")
            return False
        
        # --- TRAINING ---
        callbacks = [
            GradientAccumulation(training_config['gradient_accumulation_batch_size']),
        ]
        
        logger.info("Initializing Learner...")
        learner = Learner(
            dls=dl,
            model=model,
            loss_func=CrossEntropyLossFlatImageTypeWeighted(),
            metrics=[DiceMultiStrip],
            cbs=callbacks,
        )
        
        if training_config['use_bf16']:
            learner = learner.to_bf16()
        
        logger.info(
            f"Starting Fine Tuning: Freeze {training_config['freeze_epochs']}, "
            f"Unfreeze {training_config['unfrozen_epochs']}"
        )
        learner.fine_tune(
            epochs=training_config['unfrozen_epochs'],
            freeze_epochs=training_config['freeze_epochs'],
            base_lr=training_config['learning_rate'],
        )
        
        # --- SAVING ---
        logger.info("\nSaving models...")
        
        # Save Fastai learner
        learner.save(fai_model_name)
        logger.info(f"  ✓ Saved Fastai learner: {fai_model_name}")
        
        # Convert to CPU and float32 for saving
        model_cpu = learner.model.to("cpu").float()
        
        # PRIORITY 1: Save safetensors (PRIMARY FORMAT)
        logger.info(f"  Saving PRIMARY format: {safetensor_state_path.name}")
        save_file(model_cpu.state_dict(), safetensor_state_path)
        
        # Verify safetensors integrity
        safetensors_valid = verify_safetensors_integrity(safetensor_state_path, model_cpu)
        if not safetensors_valid:
            logger.error("  CRITICAL: Safetensors file failed integrity check!")
            return False
        
        # Save additional formats
        torch.save(model_cpu.state_dict(), state_path)
        logger.info(f"  ✓ Saved PyTorch state dict: {state_path.name}")
        
        torch.save(model_cpu, pytorch_model_path)
        logger.info(f"  ✓ Saved PyTorch full model: {pytorch_model_path.name}")
        
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
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        logger.info(f"  ✓ Saved config: {config_path.name}")
        
        logger.info(f"\n✓ Fine-tuning complete. Models saved to {models_dir}")
        logger.info(f"  PRIMARY OUTPUT: {safetensor_state_path}")
        
        # --- EVALUATION ---
        results_dir = output_base_dir / f"results_{base_identifier}"
        results_dir.mkdir(exist_ok=True)
        
        # Validation Set
        try:
            compute_and_plot_metrics(
                learner,
                dl=dl.valid,
                dataset_name="Validation",
                save_dir=results_dir,
                class_names=CLASS_NAMES
            )
        except Exception as e:
            logger.error(f"Error evaluating validation set: {e}")
        
        # Training Set
        try:
            compute_and_plot_metrics(
                learner,
                dl=dl.train,
                dataset_name="Training",
                save_dir=results_dir,
                class_names=CLASS_NAMES
            )
        except Exception as e:
            logger.error(f"Error evaluating training set: {e}")
        
        logger.info(f"Results saved to {results_dir}")
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
    batch_size = 10
    learning_rate = 0.0001
    
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
        # Default for custom fine-tuning: Conservative epoch count to prevent catastrophic forgetting
        freeze_epochs = 6
        unfrozen_epochs = 12
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
    output_base_dir = project_root / f"fine_tuning_results_{model_version}"
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
    
    # --- SUMMARY ---
    logger.info(f"\n\n{'='*80}")
    logger.info("FINE-TUNING SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total models processed: {len(model_configs)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {failure_count}")
    logger.info(f"Output directory: {output_base_dir}")
    logger.info(f"\nPRIMARY OUTPUT FORMAT: .safetensors files")
    logger.info(f"All safetensors files have been integrity-verified")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()