from pathlib import Path

# ==============================================================================
# LOCAL CONFIGURATION
# This file is ignored by git. Use it to set local paths for data and models.
# ==============================================================================

# Centralized Model Versioning
# Change this in one place to affect training outputs and testing inputs
CUSTOM_MODEL_VERSION = "OCM_7.43_R_G_NIR_test_4"

# Optimization Settings
# Set to True if you want to use dual-resolution processing (if supported by loader)
USE_DUAL_RES_METHOD = True

# Common Root for the Dataset (Optional helper)
_DATASET_ROOT = Path(r"D:\Projects\QI47\2025_Projects\Image_QC_GUI\2_Repo\OmniCloudMask\training\data\2051_102025077_D09_Arriege_D")
# _DATASET_ROOT = Path(r"D:\Projects\QI47\2025_Projects\Image_QC_GUI\1_Data\downsampled_images_recent_3_datasets\2051_102025077_D09_Arriege_E")

# Repo Root (Optional helper, assuming this file is in training/scripts/)
_REPO_ROOT = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------------------------
# 1. prepare_masks.py
# ------------------------------------------------------------------------------
PREPARE_MASKS_IMAGE_DIR = _DATASET_ROOT / "images"
PREPARE_MASKS_GPKG_DIR = _DATASET_ROOT / "gpkg"
PREPARE_MASKS_OUTPUT_DIR = _DATASET_ROOT / "masks_3"

# ------------------------------------------------------------------------------
# 2. preprocess_custom_data.py
# ------------------------------------------------------------------------------
PREPROCESS_BASE_DIR = _DATASET_ROOT
PREPROCESS_INPUT_IMAGES_DIR = _DATASET_ROOT / "images"
PREPROCESS_INPUT_LABELS_DIR = _DATASET_ROOT / "masks_3"
PREPROCESS_OUTPUT_DIR = _DATASET_ROOT / "processed_dataset_3"

# ------------------------------------------------------------------------------
# 3. train_ocm_models_custom.py
# ------------------------------------------------------------------------------
# Directory containing 'train' and 'validation' folders created by preprocess
TRAIN_DATA_DIR = PREPROCESS_OUTPUT_DIR

# Path to the pretrained weights (checkpoint)
TRAIN_PRETRAINED_WEIGHTS_PATH = _REPO_ROOT / "ckpts" / "PM_model_OCM_7.43_R_G_NIR_regnety_004.pycls_in1k_PT_state.safetensors"

# ------------------------------------------------------------------------------
# 4. test_ocm_models_custom.py
# ------------------------------------------------------------------------------
# Path to the trained model to test (Uses the version defined above)
TEST_MODEL_PATH = _REPO_ROOT / "models" / f"PM_model_{CUSTOM_MODEL_VERSION}_regnety_004.pycls_in1k_PT_state.safetensors"

# Directory of images to test on
TEST_IMAGES_DIR = _DATASET_ROOT / "images"
# TEST_IMAGES_DIR = PREPROCESS_OUTPUT_DIR / "validation"

# Directory for test results
TEST_OUTPUT_DIR = _REPO_ROOT / f"test_results_{CUSTOM_MODEL_VERSION}_v1"

# ------------------------------------------------------------------------------
# 5. compare_ocm_models.py
# ------------------------------------------------------------------------------
# Quick test mode (True = fast testing, False = full evaluation)
# If True, it will randomly sample a subset of the validation data for speed.
COMPARISON_QUICK_TEST = True
COMPARISON_QUICK_TEST_SAMPLES = 50  # Number of samples for quick test

# Batch size for evaluation
COMPARISON_BATCH_SIZE = 16

# Use bfloat16 precision (saves memory on Ampere GPUs)
COMPARISON_USE_BF16 = False

# ------------------------------------------------------------------------------
# 6. Global Data Settings (Used by Train & Test)
# ------------------------------------------------------------------------------
# 1. BAND SELECTION
# If your images are [Red, Green, Blue, NIR], use [1, 2, 4] to get R-G-NIR
# If your images are [Red, Green, NIR], use [1, 2, 3]
BAND_ORDER = [1, 2, 3] 

# 2. SCALING FACTOR
# Match the resolution used in training. 
# If you trained on 10m data and your input is 1m, use 0.1.
# If your input is already at the target resolution, use 1.0.
SCALE_FACTOR = 1 

# 3. SYNTHETIC NIR
# Set to True ONLY if you are using RGB data and need to fake the NIR band.
GENERATE_SYNTHETIC_NIR = False 
