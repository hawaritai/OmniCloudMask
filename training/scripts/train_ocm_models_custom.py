import sys
import os
from pathlib import Path
import warnings
import json
import numpy as np
import random
import cv2
from collections import defaultdict
from functools import partial

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
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)

def main():
    print("Starting training script...")
    print_system_info()

    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

    # --- LOCAL CONFIG IMPORT ---
    try:
        if str(current_script_dir) not in sys.path:
            sys.path.append(str(current_script_dir))
        from local_config import TRAIN_DATA_DIR, TRAIN_PRETRAINED_WEIGHTS_PATH
    except ImportError:
        print("CRITICAL: local_config.py not found. Please create 'training/scripts/local_config.py' to define local paths.")
        sys.exit(1)

    # --- PATHS ---
    base_data_path = TRAIN_DATA_DIR
    
    my_custom_data_dir = base_data_path / "train"
    my_custom_val_dir = base_data_path / "validation"
    
    # Override validation directory
    cloudsen12_validation_dir = my_custom_val_dir

    # --- CONFIGURATION ---
    # Set model type to match the checkpoint we want to fine-tune
    model_type = "regnety_004.pycls_in1k" 
    model_version = "OCM_7.43_R_G_NIR_test_3"
    
    use_bf16 = True
    demo_mode = False
    
    original_image_size = 509
    max_clip_image_clip_size = 400
    min_clip_image_size = 256
    limited_band_read_list = [1, 2, 3]  # Red Green NIR
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
    
    print("Checking dataset directories...")
    for dataset_dir in dataset_dirs:
        if not dataset_dir.exists():
            print(f"Warning: Directory {dataset_dir} does not exist. Please check paths.")

    if demo_mode:
        freeze_epochs = 5
        unfrozen_epochs = 5
        limit_training_images = 3000
    else:
        freeze_epochs = 50
        unfrozen_epochs = 50
        limit_training_images = None

    num_input_channels = len(limited_band_read_list)
    print(f"Number of input channels: {num_input_channels}")

    # --- MODEL SETUP ---
    print(f"Creating model: {model_type}")
    timm_model = partial(
        timm.create_model,
        model_type,
        pretrained=True,
        in_chans=num_input_channels,
    )
    model = create_unet_model(
        img_size=(509, 509),
        arch=timm_model,
        n_out=4,
        pretrained=True,
        act_cls=torch.nn.Mish,
    )

    # --- LOAD PRETRAINED WEIGHTS ---
    # Path to the specific checkpoint
    pretrained_weights_path = TRAIN_PRETRAINED_WEIGHTS_PATH
    
    if pretrained_weights_path.exists():
        print(f"Loading pretrained weights from {pretrained_weights_path}")
        try:
            # Load safetensors file
            state_dict = load_file(pretrained_weights_path)
            
            # Load into model
            # strict=False is often useful if there are minor mismatches (e.g. head size), 
            # though here we expect a match if it's the same model architecture.
            model.load_state_dict(state_dict, strict=False)
            print("Successfully loaded pretrained weights.")
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Continuing with ImageNet weights (from timm)...")
    else:
        print(f"Warning: Pretrained weights not found at {pretrained_weights_path}")
        print("Continuing with ImageNet weights (from timm)...")

    # Dummy Input Check
    dummy_input = torch.randn(
        1, num_input_channels, original_image_size, original_image_size
    )
    assert model(dummy_input).shape == (
        1,
        4,
        original_image_size,
        original_image_size,
    ), "Model output shape mismatch"

    # --- MODEL SAVING SETUP ---
    models_dir = Path.cwd() / "models"
    models_dir.mkdir(exist_ok=True)
    
    fai_model_name = f"PM_model_{model_version}_{model_type}_fai"
    pytorch_model_name = f"PM_model_{model_version}_{model_type}_PT.pth"
    pytorch_model_path = models_dir / pytorch_model_name
    state_path = pytorch_model_path.parent / f"{pytorch_model_path.stem}_state.pth"
    safetensor_state_path = pytorch_model_path.parent / f"{pytorch_model_path.stem}_state.safetensors"
    config_path = pytorch_model_path.parent / f"{pytorch_model_path.stem}_config.json"

    if pytorch_model_path.exists():
        print(f"Warning: Model {pytorch_model_name} already exists.")

    print(f"Fastai model name: {fai_model_name}")
    print(f"PyTorch model path: {pytorch_model_path}")

    # --- DATA LOADING ---
    validation_dataset_files = set(cloudsen12_validation_dir.glob("*image*.tif"))
    print(f"Validation images found: {len(validation_dataset_files)}")

    def multi_dataset_getter(paths: list[Path], print_counts: bool = False):
        training_images = []
        validation_images = []
        for path in paths:
            if path == cloudsen12_validation_dir:
                v_imgs = list(path.glob("*image*.tif"))
                validation_images = v_imgs
                if print_counts:
                    print(f"{path.name} found {len(v_imgs)} validation images")
            else:
                images = list(path.glob("*image*.tif"))
                if print_counts:
                    print(f"{path.name} found {len(images)} images")
                training_images.extend(images)
        
        if print_counts:
            print(f"Found {len(training_images)} training images")

        if limit_training_images:
            training_images = np.random.choice(
                training_images, limit_training_images, replace=False
            ).tolist()
            if print_counts:
                print(f"Limited training images to {len(training_images)}")

        datasets = training_images + validation_images
        if print_counts:
            print(f"Combined training and validation {len(datasets)} images")
        return datasets

    train_and_val_images = multi_dataset_getter(list(dataset_dirs), print_counts=True)

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
    for i, (band, scale) in enumerate(zip(limited_band_read_list, native_band_scales)):
        scale_groups[int(original_image_size * scale)].append((i, band))

    def open_2k(src: rio.DatasetReader, img_size: int) -> np.ndarray:
        resampling_method = random.choice([Resampling.bilinear, Resampling.nearest])
        resampled_data = src.read(
            limited_band_read_list,
            out_shape=(len(limited_band_read_list), img_size, img_size),
            resampling=resampling_method,
        )
        return resampled_data.astype("float32")

    def open_509(src: rio.DatasetReader, img_size: int) -> np.ndarray:
        resampled_data = np.empty(
            (len(limited_band_read_list), img_size, img_size), dtype=np.float32
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
                    limited_band_read_list,
                    out_shape=(len(limited_band_read_list), img_size, img_size),
                    resampling=Resampling.bilinear
                ).astype("float32")

            image_tensor = torch.from_numpy(resampled_data)
            if use_bf16:
                image_tensor = image_tensor.bfloat16()
            return TensorImage(image_tensor)

    def sample_weights(image_path: Path) -> torch.Tensor:
        try:
            weight = torch.tensor(label_weights[image_path.parent], dtype=torch.float32)
        except Exception:
            return torch.tensor(1.0, dtype=torch.float32)
        return weight

    open_image_func = partial(open_img, img_size=original_image_size, use_bf16=use_bf16)

    def is_validation_item(item: Path):
        return item in validation_dataset_files

    batch_tfms = [
        RandomRectangle(p=0.6, sl=0.1, sh=0.5),
        BatchTear(0.1),
        SceneEdge(p=0.1),
        IntToFloatTensor(1, 1), 
        BatchRot90(),
        DynamicZScoreNormalize(),
        BatchResample(max_scale=1.111, min_scale=0.07, plateau_min=0.33, plateau_max=1.0),
        RandomClipLargeImages(max_size=max_clip_image_clip_size, min_size=min_clip_image_size),
        BatchFlip(),
        RandomSharpenBlur(min_factor=0.5, max_factor=1.5),
        ClipHighAndLow(p=0.1, max_pct=0.05),
    ]

    # --- DATALOADER ---
    print("Creating DataBlock...")
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
            Resize(original_image_size, method="squish")
        ],
    )

    print("Creating DataLoaders...")
    num_workers = 0 if os.name == 'nt' else 6
    
    dl = dblock.dataloaders(
        dataset_dirs, 
        bs=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    try:
        print("Fetching one batch...")
        batch = dl.one_batch()
        print(f"Input shape: {batch[0].shape}")
        print(f"Label shape: {batch[1].shape}")
    except Exception as e:
        print(f"Error fetching batch: {e}")
        return

    # --- TRAINING ---
    callbacks = [
        GradientAccumulation(gradient_accumulation_batch_size),
    ]

    print("Initializing Learner...")
    learner = Learner(
        dls=dl,
        model=model,
        loss_func=CrossEntropyLossFlatImageTypeWeighted(),
        metrics=[DiceMultiStrip],
        cbs=callbacks,
    )

    if use_bf16:
        learner = learner.to_bf16()

    print(f"Starting Fine Tuning: Freeze {freeze_epochs}, Unfreeze {unfrozen_epochs}")
    learner.fine_tune(
        epochs=unfrozen_epochs,
        freeze_epochs=freeze_epochs,
        base_lr=learning_rate,
    )

    # --- SAVING ---
    print("Saving models...")
    learner.save(fai_model_name)
    
    model_cpu = learner.model.to("cpu").float()
    torch.save(model_cpu, pytorch_model_path)
    torch.save(model_cpu.state_dict(), state_path)
    save_file(model_cpu.state_dict(), safetensor_state_path)
    
    config = {
        "model_version": model_version,
        "model_type": model_type,
        "use_bf16": use_bf16,
        "demo_mode": demo_mode,
        "original_image_size": original_image_size,
        "max_clip_image_clip_size": max_clip_image_clip_size,
        "min_clip_image_size": min_clip_image_size,
        "limited_band_read_list": limited_band_read_list,
        "native_band_scales": native_band_scales,
        "gradient_accumulation_batch_size": gradient_accumulation_batch_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "freeze_epochs": freeze_epochs,
        "unfrozen_epochs": unfrozen_epochs,
        "limit_training_images": limit_training_images,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
        
    print(f"Training complete. Models saved to {models_dir}")

if __name__ == "__main__":
    main()