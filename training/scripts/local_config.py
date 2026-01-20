from pathlib import Path

# ==============================================================================
# LOCAL CONFIGURATION
# This file is ignored by git. Use it to set local paths for data and models.
# ==============================================================================

# Common Root for the Dataset (Optional helper)
# Adjust this to where your "2051_102025077_D09_Arriege_D" folder is located
_DATASET_ROOT = Path(r"D:\Projects\QI47\2025_Projects\Image_QC_GUI\2_Repo\OmniCloudMask\training\data\2051_102025077_D09_Arriege_D")

# Repo Root (Optional helper, assuming this file is in training/scripts/)
_REPO_ROOT = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------------------------
# 1. prepare_masks.py
# ------------------------------------------------------------------------------
PREPARE_MASKS_IMAGE_DIR = _DATASET_ROOT / "images"
PREPARE_MASKS_GPKG_DIR = _DATASET_ROOT / "gpkg"
PREPARE_MASKS_OUTPUT_DIR = _DATASET_ROOT / "masks_2"

# ------------------------------------------------------------------------------
# 2. preprocess_custom_data.py
# ------------------------------------------------------------------------------
PREPROCESS_BASE_DIR = _DATASET_ROOT
PREPROCESS_INPUT_IMAGES_DIR = _DATASET_ROOT / "images"
# Note: In your original script this was 'masks', but prepare_masks outputs to 'masks_2'.
# Adjust this if you want to use the output of prepare_masks.
PREPROCESS_INPUT_LABELS_DIR = _DATASET_ROOT / "masks" 
PREPROCESS_OUTPUT_DIR = _DATASET_ROOT / "processed_dataset_2"

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
# Path to the trained model to test
TEST_MODEL_PATH = _REPO_ROOT / "models" / "PM_model_OCM_7.43_R_G_NIR_test_3_regnety_004.pycls_in1k_PT_state.safetensors"

# Directory of images to test on
TEST_IMAGES_DIR = _DATASET_ROOT / "images"

# Directory for test results
TEST_OUTPUT_DIR = _REPO_ROOT / "test_results_3"
