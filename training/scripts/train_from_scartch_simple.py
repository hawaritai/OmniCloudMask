"""
Simplified training script for Experiment 2: Training from Scratch with ImageNet Encoder.

This script provides a cleaner training approach with:
- Discriminative learning rates (lower for encoder, higher for decoder)
- Fewer callbacks for less confusion
- Better learning rate schedule
- Longer frozen phase for decoder to learn first

Usage:
    python train_from_scratch_simple.py --encoder mit_b5
    python train_from_scratch_simple.py --encoder mit_b4 --freeze-epochs 15
"""

import sys
from pathlib import Path
import logging
import argparse
import warnings
import numpy as np
import random
import cv2

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import segmentation_models_pytorch as smp
import rasterio as rio
from rasterio.enums import Resampling
from safetensors.torch import save_file

# Import configuration
try:
    from local_config import (
        EXPERIMENT2_CONFIG,
        TRAIN_DATA_DIR,
        ORIGINAL_IMAGE_SIZE,
        MAX_CLIP_IMAGE_SIZE,
        MIN_CLIP_IMAGE_SIZE,
        BATCH_SIZE,
        GRADIENT_ACCUMULATION_BATCH_SIZE,
        USE_BF16,
        CLASS_WEIGHTS,
        LIMIT_TRAINING_IMAGES,
        BAND_ORDER,
        SMP_MODEL_TYPES,
        CLASS_NAMES,
        get_native_band_scales,
    )
except ImportError as e:
    logger.error(f"Failed to import from local_config: {e}")
    sys.exit(1)

# Import helpers
try:
    from training.helpers import print_system_info
except ImportError as e:
    logger.error(f"Failed to import helpers: {e}")
    sys.exit(1)

# # Import augmentations from augs.py
# try:
#     from augs import (
#         BatchRot90,
#         RandomRectangle,
#         DynamicZScoreNormalize,
#         BatchFlip,
#         RandomClipLargeImages,
#         RandomSharpenBlur,
#         ClipHighAndLow,
#     )
# except ImportError as e:
#     logger.warning(f"Could not import augmentations from augs.py: {e}")
#     logger.info("Will use basic augmentations only")


# Simple metric classes
class MetricBase:
    """Base class for metrics."""
    def __init__(self, name):
        self.name = name
        self.reset()
    
    def reset(self):
        self.total = 0
        self.count = 0
    
    def accumulate(self, preds, labels):
        raise NotImplementedError
    
    @property
    def value(self):
        return self.total / self.count if self.count > 0 else 0


class DiceMetric(MetricBase):
    """Dice coefficient metric for multi-class segmentation."""
    def __init__(self, num_classes=4, smooth=1e-6):
        super().__init__("dice")
        self.num_classes = num_classes
        self.smooth = smooth
    
    def reset(self):
        self.dice_scores = []
    
    def accumulate(self, preds, labels):
        # preds: (B, C, H, W) logits
        # labels: (B, H, W) class indices
        preds = preds.argmax(dim=1)  # (B, H, W)
        
        for cls in range(1, self.num_classes):  # Skip background (class 0)
            pred_cls = (preds == cls).float()
            label_cls = (labels == cls).float()
            
            intersection = (pred_cls * label_cls).sum()
            union = pred_cls.sum() + label_cls.sum()
            
            if union > 0:
                dice = (2 * intersection + self.smooth) / (union + self.smooth)
                self.dice_scores.append(dice.item())
    
    @property
    def value(self):
        return np.mean(self.dice_scores) if self.dice_scores else 0


class RecallMetric(MetricBase):
    """Recall metric for multi-class segmentation."""
    def __init__(self, num_classes=4):
        super().__init__("recall")
        self.num_classes = num_classes
    
    def reset(self):
        self.true_positives = 0
        self.actual_positives = 0
    
    def accumulate(self, preds, labels):
        preds = preds.argmax(dim=1)
        
        for cls in range(1, self.num_classes):  # Skip background
            pred_cls = (preds == cls).float()
            label_cls = (labels == cls).float()
            
            self.true_positives += (pred_cls * label_cls).sum().item()
            self.actual_positives += label_cls.sum().item()
    
    @property
    def value(self):
        return self.true_positives / self.actual_positives if self.actual_positives > 0 else 0


class PrecisionMetric(MetricBase):
    """Precision metric for multi-class segmentation."""
    def __init__(self, num_classes=4):
        super().__init__("precision")
        self.num_classes = num_classes
    
    def reset(self):
        self.true_positives = 0
        self.predicted_positives = 0
    
    def accumulate(self, preds, labels):
        preds = preds.argmax(dim=1)
        
        for cls in range(1, self.num_classes):  # Skip background
            pred_cls = (preds == cls).float()
            label_cls = (labels == cls).float()
            
            self.true_positives += (pred_cls * label_cls).sum().item()
            self.predicted_positives += pred_cls.sum().item()
    
    @property
    def value(self):
        return self.true_positives / self.predicted_positives if self.predicted_positives > 0 else 0

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CloudDataset(Dataset):
    """Simple dataset for cloud segmentation."""
    
    def __init__(self, image_paths, band_order, img_size=509, is_train=True):
        self.image_paths = image_paths
        self.band_order = band_order
        self.img_size = img_size
        self.is_train = is_train
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Get label path - only replace in filename, not the full path
        # Handle both 'image' and 'Image' in filename
        filename = img_path.name
        if 'image' in filename.lower():
            label_filename = filename.replace('image', 'label').replace('Image', 'Label')
        else:
            label_filename = filename  # fallback
        label_path = img_path.parent / label_filename
        
        # Load image
        with rio.open(img_path) as src:
            image = src.read(self.band_order).astype(np.float32)
        
        # Load label
        with rio.open(label_path) as src:
            label = src.read(1).astype(np.int64)
        
        # Simple augmentation for training
        if self.is_train:
            # Random flip
            if random.random() > 0.5:
                image = np.flip(image, axis=2).copy()
                label = np.flip(label, axis=1).copy()
            if random.random() > 0.5:
                image = np.flip(image, axis=1).copy()
                label = np.flip(label, axis=0).copy()
            
            # Random rotation (90, 180, 270)
            k = random.randint(0, 3)
            if k > 0:
                image = np.rot90(image, k, axes=(1, 2)).copy()
                label = np.rot90(label, k, axes=(0, 1)).copy()
        
        # Convert to tensor
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        
        return image, label


def create_model(encoder_name, num_classes=4, num_input_channels=3):
    """Create SMP model with ImageNet pretrained encoder."""
    model_type = SMP_MODEL_TYPES.get(encoder_name, encoder_name)
    
    model = smp.Unet(
        encoder_name=model_type,
        encoder_weights='imagenet',
        in_channels=num_input_channels,
        classes=num_classes,
    )
    
    return model


def get_dataloaders(train_dir, val_dir, band_order, batch_size, img_size, limit_images=None):
    """Create train and validation dataloaders."""
    
    # Get image paths
    train_images = list(Path(train_dir).glob("*image*.tif"))
    val_images = list(Path(val_dir).glob("*image*.tif"))
    
    # Limit training images if specified
    if limit_images:
        train_images = random.sample(train_images, min(limit_images, len(train_images)))
    
    logger.info(f"Training images: {len(train_images)}")
    logger.info(f"Validation images: {len(val_images)}")
    
    # Create datasets
    train_dataset = CloudDataset(train_images, band_order, img_size, is_train=True)
    val_dataset = CloudDataset(val_images, band_order, img_size, is_train=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    return train_loader, val_loader


def train_epoch(model, dataloader, criterion, optimizer, device, use_bf16=False, epoch=0, total_epochs=10):
    """Train for one epoch with progress bar."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]", 
                leave=False, ncols=120, unit="batch")
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Use autocast for BF16 instead of manual conversion
        if use_bf16:
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{total_loss/num_batches:.4f}"})
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device, metrics, use_bf16=False, epoch=0, total_epochs=10):
    """Validate the model with progress bar."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    # Reset metrics
    for metric in metrics:
        metric.reset()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]", 
                leave=False, ncols=120, unit="batch")
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Use autocast for BF16 instead of manual conversion
            if use_bf16:
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update metrics
            preds = outputs.argmax(dim=1)
            for metric in metrics:
                metric.accumulate(outputs, labels)
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{total_loss/num_batches:.4f}"})
    
    # Get metric values
    metric_values = {m.name: m.value for m in metrics}
    
    return total_loss / num_batches, metric_values


def freeze_encoder(model):
    """Freeze encoder parameters."""
    for name, param in model.named_parameters():
        if 'encoder' in name:
            param.requires_grad = False
    logger.info("Encoder frozen")


def unfreeze_encoder(model):
    """Unfreeze encoder parameters."""
    for name, param in model.named_parameters():
        if 'encoder' in name:
            param.requires_grad = True
    logger.info("Encoder unfrozen")


def count_parameters(model):
    """Count trainable vs total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Simplified training from scratch')
    parser.add_argument('--encoder', type=str, default='mit_b5',
                       choices=['mit_b5', 'mit_b4', 'efficientnet_b5', 'efficientnet_b4', 'resnet34'])
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (default: 1e-3 frozen, 1e-4 unfrozen)')
    parser.add_argument('--freeze-epochs', type=int, default=15, help='Epochs with frozen encoder')
    parser.add_argument('--unfreeze-epochs', type=int, default=30, help='Epochs with unfrozen encoder')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--limit-images', type=int, default=None, help='Limit training images')
    args = parser.parse_args()
    
    # Print system info
    print_system_info()
    
    # Configuration
    train_dir = TRAIN_DATA_DIR / "train"
    val_dir = TRAIN_DATA_DIR / "validation"
    img_size = ORIGINAL_IMAGE_SIZE
    batch_size = args.batch_size or BATCH_SIZE
    use_bf16 = USE_BF16
    
    # Learning rates
    lr_frozen = args.lr or 1e-3  # Higher LR for decoder-only training
    lr_unfrozen = 1e-4  # Lower LR when encoder is unfrozen
    
    # Log configuration
    logger.info("\n" + "="*60)
    logger.info("SIMPLIFIED TRAINING FROM SCRATCH")
    logger.info("="*60)
    logger.info(f"Encoder: {args.encoder}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Freeze epochs: {args.freeze_epochs} (LR: {lr_frozen})")
    logger.info(f"Unfreeze epochs: {args.unfreeze_epochs} (LR: {lr_unfrozen})")
    logger.info(f"BF16: {use_bf16}")
    logger.info("="*60 + "\n")
    
    # Create model
    model = create_model(args.encoder, num_classes=len(CLASS_NAMES), num_input_channels=len(BAND_ORDER))
    model = model.to(DEVICE)
    
    total, trainable = count_parameters(model)
    logger.info(f"Model parameters: {total:,} total, {trainable:,} trainable")
    
    # Create dataloaders
    train_loader, val_loader = get_dataloaders(
        train_dir, val_dir, BAND_ORDER, batch_size, img_size, args.limit_images
    )
    
    # Loss function with class weights
    class_weights = torch.tensor(CLASS_WEIGHTS, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Metrics
    metrics = [
        DiceMetric(num_classes=len(CLASS_NAMES)),
        RecallMetric(num_classes=len(CLASS_NAMES)),
        PrecisionMetric(num_classes=len(CLASS_NAMES)),
    ]
    
    # Best model tracking
    best_score = 0
    best_epoch = 0
    best_model_state = None
    
    # ========================================
    # PHASE 1: FROZEN ENCODER
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("PHASE 1: Training with frozen encoder")
    logger.info("="*60)
    
    freeze_encoder(model)
    total, trainable = count_parameters(model)
    logger.info(f"Parameters: {total:,} total, {trainable:,} trainable")
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_frozen, weight_decay=1e-4
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.freeze_epochs, eta_min=lr_frozen/100)
    
    for epoch in range(args.freeze_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE, use_bf16, epoch, args.freeze_epochs)
        val_loss, metric_values = validate(model, val_loader, criterion, DEVICE, metrics, use_bf16, epoch, args.freeze_epochs)
        scheduler.step()
        
        # Calculate composite score (30% Dice + 35% Recall + 35% Precision)
        dice = metric_values.get('dice', 0)
        recall = metric_values.get('recall', 0)
        precision = metric_values.get('precision', 0)
        composite_score = 0.3 * dice + 0.35 * recall + 0.35 * precision
        
        # Track best model
        if composite_score > best_score:
            best_score = composite_score
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"  📈 New best model at epoch {epoch} with score {composite_score:.4f}")
        
        logger.info(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                   f"Dice: {dice:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f} | Score: {composite_score:.4f}")
    
    # ========================================
    # PHASE 2: UNFROZEN ENCODER
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("PHASE 2: Training with unfrozen encoder")
    logger.info("="*60)
    
    unfreeze_encoder(model)
    total, trainable = count_parameters(model)
    logger.info(f"Parameters: {total:,} total, {trainable:,} trainable")
    
    # Use discriminative learning rates
    encoder_params = [p for n, p in model.named_parameters() if 'encoder' in n]
    decoder_params = [p for n, p in model.named_parameters() if 'encoder' not in n]
    
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': lr_unfrozen},  # Lower LR for encoder
        {'params': decoder_params, 'lr': lr_unfrozen * 10},  # Higher LR for decoder
    ], weight_decay=1e-4)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.unfreeze_epochs, eta_min=lr_unfrozen/100)
    
    patience = 10
    no_improve = 0
    
    for epoch in range(args.unfreeze_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE, use_bf16, epoch, args.unfreeze_epochs)
        val_loss, metric_values = validate(model, val_loader, criterion, DEVICE, metrics, use_bf16, epoch, args.unfreeze_epochs)
        scheduler.step()
        
        # Calculate composite score
        dice = metric_values.get('dice', 0)
        recall = metric_values.get('recall', 0)
        precision = metric_values.get('precision', 0)
        composite_score = 0.3 * dice + 0.35 * recall + 0.35 * precision
        
        # Track best model
        if composite_score > best_score:
            best_score = composite_score
            best_epoch = args.freeze_epochs + epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            logger.info(f"  📈 New best model at epoch {args.freeze_epochs + epoch} with score {composite_score:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"  ⏹️ Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break
        
        logger.info(f"Epoch {args.freeze_epochs + epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                   f"Dice: {dice:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f} | Score: {composite_score:.4f}")
    
    # ========================================
    # SAVE MODELS (both safetensor and .pth formats)
    # ========================================
    output_dir = Path(f"fine_tune_results/exp2_simple_{args.encoder}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get latest model state
    latest_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # Save best model in both formats
    if best_model_state:
        # Save as .pth
        best_pth_path = output_dir / f"best_model_{args.encoder}.pth"
        torch.save(best_model_state, best_pth_path)
        
        # Save as safetensor
        best_safetensor_path = output_dir / f"best_model_{args.encoder}.safetensors"
        save_file(best_model_state, best_safetensor_path)
        
        logger.info(f"\n✅ Best model saved from epoch {best_epoch} with score {best_score:.4f}")
        logger.info(f"   .pth: {best_pth_path}")
        logger.info(f"   .safetensors: {best_safetensor_path}")
    
    # Save latest model in both formats
    # Save as .pth
    latest_pth_path = output_dir / f"latest_model_{args.encoder}.pth"
    torch.save(latest_model_state, latest_pth_path)
    
    # Save as safetensor
    latest_safetensor_path = output_dir / f"latest_model_{args.encoder}.safetensors"
    save_file(latest_model_state, latest_safetensor_path)
    
    logger.info(f"\n✅ Latest model saved from epoch {args.freeze_epochs + epoch}")
    logger.info(f"   .pth: {latest_pth_path}")
    logger.info(f"   .safetensors: {latest_safetensor_path}")
    
    # Save training config
    config = {
        'encoder': args.encoder,
        'freeze_epochs': args.freeze_epochs,
        'unfreeze_epochs': args.unfreeze_epochs,
        'lr_frozen': lr_frozen,
        'lr_unfrozen': lr_unfrozen,
        'batch_size': batch_size,
        'best_score': best_score,
        'best_epoch': best_epoch,
    }
    
    import json
    with open(output_dir / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETED")
    logger.info(f"Best score: {best_score:.4f} at epoch {best_epoch}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
