from pathlib import Path

# ==============================================================================
# LOCAL CONFIGURATION
# This file is ignored by git. Use it to set local paths for data and models.
# ==============================================================================

# Centralized Model Versioning
# Change this in one place to affect training outputs and testing inputs
CUSTOM_MODEL_VERSION = "OCM_test1x_kavel_n_cloudsen_v14.1" #PM_model_OCM_6.43_RG_NIR_test_merged_100_regnety_004.pycls_in1k_PT_state
# CUSTOM_MODEL_VERSION = "OCM_test1x_kavel_n_cloudsen_v10.2" #PM_model_OCM_6.43_RG_NIR_test_merged_100_regnety_004.pycls_in1k_PT_state

# Optimization Settings
# Set to True if you want to use dual-resolution processing (if supported by loader)
USE_DUAL_RES_METHOD = False

# Common Root for the Dataset (Optional helper)
# _DATASET_ROOT = Path(r"D:\projects\Image_QC_GUI\2_Repos\OmniCloudMask\training\data\2051_102025077_D09_Arriege_D")
_DATASET_ROOT = Path(r"D:\projects\Image_QC_GUI\2_Repos\OmniCloudMask\training\data")

# Repo Root (Optional helper, assuming this file is in training/scripts/)
_REPO_ROOT = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------------------------
# 1. prepare_masks.py
# ------------------------------------------------------------------------------
PREPARE_MASKS_IMAGE_DIR = _DATASET_ROOT / "all_images"
PREPARE_MASKS_GPKG_DIR = _DATASET_ROOT / "all_gpkg"
PREPARE_MASKS_OUTPUT_DIR = _DATASET_ROOT / "all_masks_v6"

# ------------------------------------------------------------------------------
# 1.5. duplicate_positives.py (NEW - for dataset balancing)
# ------------------------------------------------------------------------------
# Output directories for duplicated data
DUPLICATED_IMAGE_DIR = _DATASET_ROOT / "duplicated" / "images"
DUPLICATED_MASK_DIR = _DATASET_ROOT / "duplicated" / "masks"

# ------------------------------------------------------------------------------
# 2. preprocess_custom_data.py
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
# 3. train_ocm_models_custom.py
# ------------------------------------------------------------------------------
# Directory containing 'train' and 'validation' folders created by preprocess
TRAIN_DATA_DIR = PREPROCESS_OUTPUT_DIR
# TRAIN_DATA_DIR = Path(r"D:\projects\Image_QC_GUI\2_Repos\OmniCloudMask\training\data\merged_v3")

# Path to the pretrained weights (checkpoint)
TRAIN_PRETRAINED_WEIGHTS_PATH = _REPO_ROOT / "ckpts" / "PM_model_OCM_6.43_RG_NIR_regnety_004.pycls_in1k_PT_state.safetensors"

# ------------------------------------------------------------------------------
# 4. test_ocm_models_custom.py
# ------------------------------------------------------------------------------
# Path to the trained model to test (Uses the version defined above)
TEST_MODEL_PATH = _REPO_ROOT / "models" / f"PM_model_{CUSTOM_MODEL_VERSION}_regnety_004.pycls_in1k_PT_state.safetensors"

# Directory of images to test on
TEST_IMAGES_DIR = _DATASET_ROOT / "images"

# Directory for test results
TEST_OUTPUT_DIR = _REPO_ROOT / f"test_results_{CUSTOM_MODEL_VERSION}"

# ------------------------------------------------------------------------------
# 5. Global Data Settings (Used by Train & Test)
# ------------------------------------------------------------------------------
# 1. BAND SELECTION
# If your images are [Red, Green, Blue, NIR], use [1, 2, 4] to get R-G-NIR
# If your images are [Red, Green, NIR], use [1, 2, 3]
BAND_ORDER = [1, 2, 3] 

# 2. SCALING FACTOR
# Match the resolution used in training. 
# If you trained on 10m data and your input is 1m, use 0.1.
# If your input is already at the target resolution, use 1.0.
SCALE_FACTOR = 0.1 

# 3. SYNTHETIC NIR
# Set to True ONLY if you are using RGB data and need to fake the NIR band.
GENERATE_SYNTHETIC_NIR = False 
