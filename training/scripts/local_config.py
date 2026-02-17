from pathlib import Path
import torch

# ==============================================================================
# LOCAL CONFIGURATION
# This file is ignored by git. Use it to set local paths for data and models.
# ==============================================================================

# ==============================================================================
# 1. CORE SETTINGS
# ==============================================================================

# Centralized Model Versioning
# Change this in one place to affect training outputs and testing inputs
CUSTOM_MODEL_VERSION = "OCM_test1x_kavel_n_cloudsen_v14.1" #PM_model_OCM_6.43_RG_NIR_test_merged_100_regnety_004.pycls_in1k_PT_state
# CUSTOM_MODEL_VERSION = "OCM_test1x_kavel_n_cloudsen_v10.2" #PM_model_OCM_6.43_RG_NIR_test_merged_100_regnety_004.pycls_in1k_PT_state

# Optimization Settings
# Set to True if you want to use dual-resolution processing (if supported by loader)
USE_DUAL_RES_METHOD = False

# ==============================================================================
# 2. PATH CONFIGURATION
# ==============================================================================

# Common Root for the Dataset (Optional helper)
# _DATASET_ROOT = Path(r"D:\projects\Image_QC_GUI\2_Repos\OmniCloudMask\training\data\2051_102025077_D09_Arriege_D")
_DATASET_ROOT = Path(r"D:\projects\Image_QC_GUI\2_Repos\OmniCloudMask\training\data")

# Repo Root (Optional helper, assuming this file is in training/scripts/)
_REPO_ROOT = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------------------------
# 2.1. prepare_masks.py / pepare_multiscale_masks.py
# ------------------------------------------------------------------------------
PREPARE_MASKS_IMAGE_DIR = _DATASET_ROOT / "all_images"
PREPARE_MASKS_GPKG_DIR = _DATASET_ROOT / "all_gpkg"
PREPARE_MASKS_OUTPUT_DIR = _DATASET_ROOT / "all_masks_v6"

# TFW directory for legacy projects with .tfw files
PREPARE_MASKS_TFW_DIR = Path(r"Q:\02_PROJECTS\2051_102025082_D82_Tarn-et-Garonne_A\60_UM\LVL03_CertiflAI_0610")

# ------------------------------------------------------------------------------
# 2.2. duplicate_positives.py (NEW - for dataset balancing)
# ------------------------------------------------------------------------------
# Output directories for duplicated data
DUPLICATED_IMAGE_DIR = _DATASET_ROOT / "duplicated" / "images"
DUPLICATED_MASK_DIR = _DATASET_ROOT / "duplicated" / "masks"

# ------------------------------------------------------------------------------
# 2.3. preprocess_custom_data.py / preprocess_upscale_tile.py
# ------------------------------------------------------------------------------
PREPROCESS_BASE_DIR = _DATASET_ROOT

# OPTION A: Use original images (no duplication) - 60:200 ratio
PREPROCESS_INPUT_IMAGES_DIR = PREPARE_MASKS_IMAGE_DIR
PREPROCESS_INPUT_LABELS_DIR = PREPARE_MASKS_OUTPUT_DIR

# # OPTION B: Use duplicated dataset (balanced 180:200 ratio) - RECOMMENDED
# PREPROCESS_INPUT_IMAGES_DIR = DUPLICATED_IMAGE_DIR
# PREPROCESS_INPUT_LABELS_DIR = DUPLICATED_MASK_DIR

PREPROCESS_OUTPUT_DIR = _DATASET_ROOT / "processed_data_upscale_kavel_rgb_v1"

# ------------------------------------------------------------------------------
# 2.4. train_ocm_models_custom.py / fine_tune_multiple_models.py
# ------------------------------------------------------------------------------
# Directory containing 'train' and 'validation' folders created by preprocess
TRAIN_DATA_DIR = PREPROCESS_OUTPUT_DIR
# TRAIN_DATA_DIR = Path(r"D:\projects\Image_QC_GUI\2_Repos\OmniCloudMask\training\data\merged_v3")

# Path to the pretrained weights (checkpoint)
TRAIN_PRETRAINED_WEIGHTS_PATH = _REPO_ROOT / "ckpts" / "PM_model_OCM_7.43_R_G_NIR_regnety_004.pycls_in1k_PT_state.safetensors"

# Checkpoints directory for fine-tuning
CKPTS_DIR = _REPO_ROOT / "ckpts"

# ------------------------------------------------------------------------------
# 2.5. test_ocm_models_custom.py
# ------------------------------------------------------------------------------
# Path to the trained model to test (Uses the version defined above)
# TEST_MODEL_PATH = _REPO_ROOT / "models" / f"PM_model_{CUSTOM_MODEL_VERSION}_regnety_004.pycls_in1k_PT_state.safetensors"
TEST_MODEL_PATH = Path(r"D:\Projects\QI47\2025_Projects\Image_QC_GUI\2_Repo\OmniCloudMask\models\tuned_models\PM_model_OCM_test1x_kavel_n_cloudsen_v1_regnety_004.pycls_in1k_PT_state.safetensors")

# Directory of images to test on
TEST_IMAGES_DIR = _DATASET_ROOT / "images"
# TEST_IMAGES_DIR = PREPROCESS_OUTPUT_DIR / "validation"

# Directory for test results
TEST_OUTPUT_DIR = _REPO_ROOT / f"test_results_{CUSTOM_MODEL_VERSION}"

# ------------------------------------------------------------------------------
# 2.6. compare_ocm_models.py
# ------------------------------------------------------------------------------
# Fine-tuned models directory for comparison
COMPARISON_FINE_TUNED_MODELS_DIR = Path(r"D:\projects\Image_QC_GUI\2_Repos\OmniCloudMask\fine_tuning_results_OCM_test1x_kavel_n_cloudsen_v10.1.2\models")

# Base model directory for comparison
COMPARISON_BASE_MODEL_DIR = _REPO_ROOT / "ckpts"

# ==============================================================================
# 3. TRAINING CONFIGURATION
# ==============================================================================

# ------------------------------------------------------------------------------
# 3.1. Image Settings
# ------------------------------------------------------------------------------
# Target size for images (square)
ORIGINAL_IMAGE_SIZE = 509

# Clipping sizes for data augmentation
MAX_CLIP_IMAGE_SIZE = 400
MIN_CLIP_IMAGE_SIZE = 256

# ------------------------------------------------------------------------------
# 3.2. Training Hyperparameters
# ------------------------------------------------------------------------------
# Batch size for training
BATCH_SIZE = 10

# Gradient accumulation batch size
GRADIENT_ACCUMULATION_BATCH_SIZE = 128

# Learning rate
LEARNING_RATE = 0.0003

# Number of epochs for frozen encoder phase
FREEZE_EPOCHS = 5

# Number of epochs for unfrozen encoder phase
UNFROZEN_EPOCHS = 12

# Random seed for reproducibility
RANDOM_SEED = 42

# ------------------------------------------------------------------------------
# 3.3. Precision Settings
# ------------------------------------------------------------------------------
# Use bfloat16 precision (saves memory on Ampere GPUs)
USE_BF16 = True

# Demo mode (reduced epochs and limited images for quick testing)
DEMO_MODE = False

# Compile models for inference (v4 smp models only)
COMPILE_MODELS = False

# ------------------------------------------------------------------------------
# 3.4. Class Weights for Loss Function
# ------------------------------------------------------------------------------
# Class weights for recall-precision balanced training with image weights
# [Clear, Thick Cloud, Thin Cloud, Cloud Shadow]
# Ratio: 6:1 (Critical:Clear) - Balanced with image weights
CLASS_WEIGHTS = [1.0, 1.5, 3.0, 3.0]

# ------------------------------------------------------------------------------
# 3.5. Training Limits
# ------------------------------------------------------------------------------
# Limit training images (None for no limit, or set a number for demo/testing)
LIMIT_TRAINING_IMAGES = None

# ==============================================================================
# 4. COMPARISON CONFIGURATION
# ==============================================================================

# Quick test mode (True = fast testing, False = full evaluation)
# If True, it will randomly sample a subset of the validation data for speed.
COMPARISON_QUICK_TEST = False
COMPARISON_QUICK_TEST_SAMPLES = 50  # Number of samples for quick test

# Batch size for evaluation
COMPARISON_BATCH_SIZE = 16

# Use bfloat16 precision for comparison (saves memory on Ampere GPUs)
COMPARISON_USE_BF16 = False

# ==============================================================================
# 5. GLOBAL DATA SETTINGS (Used by Train & Test)
# ==============================================================================

# ------------------------------------------------------------------------------
# 5.1. Band Selection
# ------------------------------------------------------------------------------
# If your images are [Red, Green, Blue, NIR], use [1, 2, 4] to get R-G-NIR
# If your images are [Red, Green, NIR], use [1, 2, 3]
BAND_ORDER = [1, 2, 3] 

# ------------------------------------------------------------------------------
# 5.2. Scaling Factor
# ------------------------------------------------------------------------------
# Match the resolution used in training. 
# If you trained on 10m data and your input is 1m, use 0.1.
# If your input is already at the target resolution, use 1.0.
SCALE_FACTOR = 0.1 

# ------------------------------------------------------------------------------
# 5.3. Synthetic NIR
# ------------------------------------------------------------------------------
# Set to True ONLY if you are using RGB data and need to fake the NIR band.
GENERATE_SYNTHETIC_NIR = False 

# ==============================================================================
# 6. MODEL CONFIGURATION
# ==============================================================================

# ------------------------------------------------------------------------------
# 6.1. Model Library & Compilation Config (v1.7.0+)
# ------------------------------------------------------------------------------
MODEL_CONFIG = {
    "v4": {
        "model_library": "smp",
        "models": ["tu-regnety_004", "tu-edgenext_small"],
        "weight_format": ".safetensors"
    },
    "v3": {
        "model_library": "fastai",
        "models": ["regnety_004", "edgenext_small"],
        "weight_format": ".safetensors"
    },
    "v2": {
        "model_library": "fastai",
        "models": ["regnety_004", "edgenext_small"],
        "weight_format": ".safetensors"
    },
    "v1": {
        "model_library": "fastai",
        "models": ["regnety_004", "convnextv2_nano"],
        "weight_format": ".pth"
    }
}

INFERENCE_CONFIG = {
    "compile_models": False,  # Set to True for production
    "compile_mode": "default",
    "inference_dtype": "float32"
}

# ------------------------------------------------------------------------------
# 6.2. Model Type Mappings
# ------------------------------------------------------------------------------
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

# ==============================================================================
# 7. PREPROCESSING CONFIGURATION
# ==============================================================================

# ------------------------------------------------------------------------------
# 7.1. Target Size for Preprocessing
# ------------------------------------------------------------------------------
TARGET_SIZE = 509

# ------------------------------------------------------------------------------
# 7.2. Image Weights Configuration
# ------------------------------------------------------------------------------
# Paths to image weight list files
GT_LIST_PATH = Path("training/data/gt_list.txt")
FN_LIST_PATH = Path("training/data/fn_list.txt")
HARD_NEGATIVES_LIST_PATH = Path("training/data/hard_negatives_list.txt")

# Image weight values
# Optimized for 95%+ recall with 50-60% precision
WEIGHT_GT = 1.0             # Ground Truth (baseline)
WEIGHT_FN = 2.0             # False Negatives (reduced from 3.0 to 2.0 - less aggressive FN learning)
WEIGHT_HARD_NEGATIVE = 2.0  # Hard Negatives (increased from 0.3 to 0.6 - penalize FPs more)

# ------------------------------------------------------------------------------
# 7.3. Preprocessing Method Toggle
# ------------------------------------------------------------------------------
# False: Use RGBNIRHandler with normalization (Method A)
# True: Use Clamp & Scale (Red*3, Green*2, NIR*1) (Method B)
USE_METHOD_B_PREPROCESSING = False 

# ==============================================================================
# 8. MASK PREPARATION CONFIGURATION
# ==============================================================================

# ------------------------------------------------------------------------------
# 8.1. Project Configurations
# ------------------------------------------------------------------------------
# Map project codes to their original resolutions
# Format: "project_code": {config dict}
PROJECT_CONFIGS = {
    "D82A": {  # Garonne_A - LEGACY with .tfw
        "original_size": (9370, 6020),
        "current_size": (640, 411),
        "has_tfw": True,  # Special handling for geo-coordinates
        "description": "2051_102025082_D82_Tarn-et-Garonne_A (LEGACY)"
    },
    "D09E": {  # Arriege_E
        "original_size": (4000, 2571),
        "current_size": (640, 411),
        "has_tfw": False,
        "description": "2051_102025077_D09_Arriege_E"
    },
    "D09D": {  # Arriege_D
        "original_size": (8820, 5668),
        "current_size": (640, 411),
        "has_tfw": False,
        "description": "2051_102025077_D09_Arriege_D"
    },
    "FHSTG": {  # AIR_FHSTG
        "original_size": (1681, 1080),
        "current_size": (640, 411),
        "has_tfw": False,
        "description": "2025_08_03_AIR_FHSTG"
    },
    "PHKIO": {  # AIR_PHKIO
        "original_size": (640, 480),
        "current_size": (640, 480),
        "has_tfw": False,
        "description": "2025_08_10_AIR_PHKIO"
    },
    "GLSDF": {  # GLSDF_UCO42_1101_310525_5cm
        "original_size": (1422, 1080),
        "current_size": (1422, 1080),
        "has_tfw": False,
        "description": "GLSDF_UCO42_1101_310525_5cm"
    }
}

# ------------------------------------------------------------------------------
# 8.2. Class Mapping
# ------------------------------------------------------------------------------
# 'Remark' attribute values -> Integer Class ID
CLASS_MAPPING = {
    'Cloud Deep': 1,
    'Cloud Lite': 2,
    'Shadow': 3
}

# Synonym normalization (extendable)
SYNONYMS = {
    'light': 'lite',
    'lite': 'lite',
    'deep': 'deep',
    'cloud': 'cloud',
    'shadow': 'shadow'
}

# ==============================================================================
# 9. CLASS NAMES (Used across all scripts)
# ==============================================================================
CLASS_NAMES = ['Clear', 'Thick Cloud', 'Thin Cloud', 'Cloud Shadow']

# ==============================================================================
# 10. FILE SAVE CONFIGURATION
# ==============================================================================
# Maximum retry attempts for file operations (Windows file locking issues)
MAX_SAVE_RETRIES = 3

# Delay between retries in seconds
SAVE_RETRY_DELAY = 0.5

# ==============================================================================
# 11. CONVENIENCE FUNCTIONS
# ==============================================================================

def get_class_weights_tensor() -> torch.Tensor:
    """Return class weights as a torch tensor."""
    return torch.tensor(CLASS_WEIGHTS)

def get_accumulation_steps() -> int:
    """Calculate gradient accumulation steps."""
    return GRADIENT_ACCUMULATION_BATCH_SIZE // BATCH_SIZE

def get_num_input_channels() -> int:
    """Get number of input channels based on band order."""
    return len(BAND_ORDER)

def get_native_band_scales() -> list:
    """
    Get native band scales based on USE_DUAL_RES_METHOD.
    Dual Res method uses [Red*3, Green*2, NIR*1] scaling instead of Z-score.
    """
    if USE_DUAL_RES_METHOD:
        return [3, 2, 1]
    else:
        return [1, 1, 1]
