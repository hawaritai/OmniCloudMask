# Training OmniCloudMask on Custom Datasets

This guide explains how to prepare your data and modify the `Train OCM models.ipynb` notebook to fine-tune the OmniCloudMask model on your own imagery without training from scratch.

## 1. Data Preparation

To use the existing training pipeline, your data must strictly follow specific formats regarding file types, band order, and file naming.

### A. Image Format
*   **Format:** GeoTIFF (`.tif`)
*   **Channels:** 3 Bands.
*   **Band Order:** **Red, Green, NIR** (Narrow NIR/B8A preferred).
*   **Resolution/Size:**
    *   The pipeline is optimized for **509x509** pixels (approx 5km tiles at 10m GSD).
    *   It also supports **2000x2000** pixels (which it will resample during loading).
    *   *Recommendation:* Chip your large scenes into **509x509** pixel tiles.
*   **Data Type:** `uint16` (Raw DN) or `float32` (Reflectance). The pipeline uses dynamic normalization, so exact scaling (0-10000 vs 0-1) is less critical, but consistency is key.

### B. Label Format
*   **Format:** GeoTIFF (`.tif`)
*   **Channels:** 1 Band.
*   **Dimensions:** Must match the image dimensions exactly.
*   **Class Values (Integer):**
    *   `0`: Clear / Land
    *   `1`: Thick Cloud
    *   `2`: Cloud Shadow
    *   `3`: Thin Cloud / Cirrus
    *   `99`: No Data / Ignore / Border
*   **Compression:** `LZW` is recommended to save space.

### C. File Naming Convention
The training loader looks for paired files based on specific string replacements. Your files **must** follow this naming pattern:

1.  **Image:** `UniqueName_image.tif`
2.  **Label:** `UniqueName_label.tif`

*Note: The code specifically replaces `"image"` with `"label"`. Do not include `_l1c` or `_l2a` in your filenames unless you want to edit the `label_func` in the code.*

## 2. Directory Organization

Organize your new dataset on your hard drive. You can keep your dataset separate from the original OCM datasets.

**Example Structure:**
```text
/path/to/my/custom_dataset/
├── train/
│   ├── tile_001_image.tif
│   ├── tile_001_label.tif
│   ├── tile_002_image.tif
│   └── tile_002_label.tif
└── validation/
    ├── tile_999_image.tif
    └── tile_999_label.tif
```

## 3. Modifying `Train OCM models.ipynb`

Open `training/Train OCM models.ipynb` and perform the following steps:

### Step 1: Define Your Data Path
In the cell defining dataset directories, add your custom path.

```python
# Existing code
base_dataset_dir = Path("/media/nick/4TB Working 7/Datasets/OCM datasets")
# ... existing paths ...

# ADD THIS:
my_custom_data_dir = Path("/path/to/my/custom_dataset/train")
my_custom_val_dir = Path("/path/to/my/custom_dataset/validation")
```

### Step 2: Update Dataset Weights
Update the `label_weights` dictionary. This dictates how often your data is sampled during training. If fine-tuning, you might want to weight your new data higher.

```python
# Define a weight (0.0 to 1.0)
my_custom_weight = 1.0 

label_weights = {
    # ... existing datasets ...
    
    # ADD THIS:
    my_custom_data_dir: my_custom_weight,
    
    # Ensure validation is included if you want to use it for metrics
}
```

### Step 3: Handle Validation Split
The existing code uses a specific directory (`cloudsen12_validation_dir`) for validation metrics. Override this variable to point to your validation folder.

```python
# To validate ONLY on your data:
cloudsen12_validation_dir = my_custom_val_dir
```

### Step 4: Load Pretrained OCM Weights (Fine-Tuning)
By default, the notebook loads `timm` ImageNet weights. To fine-tune an *existing* OCM model, you need to load its weights before creating the `Learner`.

Find the cell creating the model:
```python
model = create_unet_model(
    img_size=(509, 509),
    arch=timm_model,
    n_out=4,
    pretrained=True, # This loads ImageNet weights
    act_cls=torch.nn.Mish,
)
```

**Immediately after that cell, add a new cell to load your OCM model:**

```python
# Path to your pretrained OCM model (e.g., downloaded from HuggingFace)
pretrained_ocm_path = Path("path/to/existing_ocm_model.pth") 

if pretrained_ocm_path.exists():
    print(f"Loading pretrained OCM weights from {pretrained_ocm_path}")
    state_dict = torch.load(pretrained_ocm_path, map_location='cpu')
    
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print("Standard load failed, trying strict=False or key adjustment...")
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
    print("Weights loaded successfully!")
else:
    print("Warning: Pretrained path not found, training from ImageNet weights.")
```

### Step 5: Adjust Learning Rate and Epochs
Since you are fine-tuning, you generally want fewer epochs and a lower learning rate.

Modify the configuration cell:
```python
freeze_epochs = 1        # Reduced from 15
unfrozen_epochs = 5      # Reduced from 15
learning_rate = 0.0001   # Reduced from 0.001 (1e-4)
```

## 4. Running the Training

1.  **Start the Notebook:** Run all cells sequentially.
2.  **Verify Data Loading:** Look at the output of `train_and_val_images` and the `plot_batch` cells. Ensure your images look correct (RGB visualization) and masks align with the features (Clouds/Shadows).
3.  **Monitor Training:** Watch the `DiceMultiStrip` metric. If it drops significantly at the start of the `unfrozen` phase, your learning rate might be too high.

## 5. Saving the Result

The notebook automatically saves the model in FastAI format, PyTorch format (`.pth`), and SafeTensors (`.safetensors`).

*   **Config:** A JSON config is saved alongside the model.
*   **Location:** Look in the `models/` directory relative to where you run the notebook.

## Common Issues & Fixes

*   **`ValueError: Unsupported image width`**: Your training images are not 509 or 2000 pixels wide. You must resize/chip them before training.
*   **`AssertionError: File path and label path are the same`**: Your naming convention is incorrect. Ensure `_image.tif` and `_label.tif` suffixes are used correctly.
*   **Validation Loss is High**: Ensure your Class 99 (Ignore) pixels are being used correctly. If your new dataset has undefined areas, mark them as 99 in the label mask.
