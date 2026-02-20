"""
Train OCM model from scratch with ImageNet pretrained encoder.

This script demonstrates Experiment 2: Training from scratch using
ImageNet pretrained encoders with a fresh decoder.

RECOMMENDED: Vision Transformers (MIT-b5/b4) for limited data
Why ViT?
- Better transfer learning from ImageNet (global receptive field from day 1)
- Natural regularization through multi-head attention (harder to overfit)
- Proven strong generalization with small datasets in satellite imagery
- Better capture of global cloud patterns from limited examples

Usage:
    # RECOMMENDED: MIT-b5 (best accuracy for limited data)
    python train_from_scratch.py --encoder mit_b5
    
    # ALTERNATIVE: MIT-b4 (faster training)
    python train_from_scratch.py --encoder mit_b4
    
    # CNN alternatives (good but more prone to overfitting)
    python train_from_scratch.py --encoder efficientnet_b5
    python train_from_scratch.py --encoder efficientnet_b4
    python train_from_scratch.py --encoder resnet34
    python train_from_scratch.py --encoder mit_b3
"""

import sys
from pathlib import Path
import logging
import argparse

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import training utilities
from fine_tune_multiple_models import (
    fine_tune_single_model,
)

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
        UNFROZEN_EPOCHS,
        USE_BF16,
        CLASS_WEIGHTS,
        LIMIT_TRAINING_IMAGES,
        BAND_ORDER,
        SMP_MODEL_TYPES,  # REQUIRED: Used in create_model_config()
        get_native_band_scales,
        get_class_weights_tensor
    )
except ImportError as e:
    logger.error(f"Failed to import from local_config: {e}")
    sys.exit(1)

# Import additional utilities
try:
    from helpers import print_system_info
except ImportError as e:
    logger.error(f"Failed to import training utilities: {e}")
    sys.exit(1)


def create_model_config(encoder_name: str) -> dict:
    """
    Create model configuration for training from scratch.
    
    Args:
        encoder_name: Name of encoder (e.g., 'efficientnet_b4', 'resnet34')
        
    Returns:
        Dictionary with model configuration
    """
    # Map encoder name to SMP model type
    model_type = SMP_MODEL_TYPES.get(encoder_name, encoder_name)
    
    return {
        'name': f'OCM_imagenet_{encoder_name}',
        'path': None,  # No checkpoint - training from scratch
        'model_type': model_type,
        'model_library': 'smp'
    }


def create_training_config(encoder_name: str, override_lr: float = None) -> dict:
    """
    Create training configuration for ImageNet encoder training.
    
    Args:
        encoder_name: Name of encoder
        override_lr: Optional learning rate override
        
    Returns:
        Dictionary with training configuration
    """
    config = EXPERIMENT2_CONFIG.copy()
    
    # Update encoder name
    config['encoder_name'] = encoder_name
    
    # Override learning rate if specified
    if override_lr is not None:
        config['learning_rate'] = override_lr
        logger.info(f"Using custom learning rate: {override_lr}")
    
    # Set batch size based on encoder size
    if 'efficientnet_b5' in encoder_name or 'mit_b3' in encoder_name:
        config['batch_size'] = 6  # Reduce for larger models
        logger.info(f"Reduced batch size to {config['batch_size']} for larger encoder")
    else:
        config['batch_size'] = BATCH_SIZE
    
    # Add required keys for fine_tune_single_model
    # dataset_dirs should contain both train and validation directories
    train_dir = TRAIN_DATA_DIR / "train"
    val_dir = TRAIN_DATA_DIR / "validation"
    config['dataset_dirs'] = [train_dir, val_dir]
    config['cloudsen12_validation_dir'] = val_dir  # Used to identify validation split
    
    config['limited_band_read_list'] = BAND_ORDER
    config['native_band_scales'] = get_native_band_scales()
    config['limit_training_images'] = LIMIT_TRAINING_IMAGES
    config['original_image_size'] = ORIGINAL_IMAGE_SIZE
    config['max_clip_image_size'] = MAX_CLIP_IMAGE_SIZE
    config['min_clip_image_size'] = MIN_CLIP_IMAGE_SIZE
    config['gradient_accumulation_batch_size'] = GRADIENT_ACCUMULATION_BATCH_SIZE
    config['unfrozen_epochs'] = config.get('unfrozen_epochs', UNFROZEN_EPOCHS)
    config['use_bf16'] = USE_BF16
    config['compile_models'] = config.get('compile_models', False)
    
    # Use class_weights_scratch for training from scratch (more aggressive)
    # Falls back to CLASS_WEIGHTS if not defined
    config['class_weights'] = config.get('class_weights_scratch', get_class_weights_tensor())
    
    # Additional required keys
    config['num_input_channels'] = len(BAND_ORDER)
    config['batch_size'] = config.get('batch_size', BATCH_SIZE)
    config['accumulation_steps'] = GRADIENT_ACCUMULATION_BATCH_SIZE // config['batch_size']
    config['freeze_epochs'] = config.get('freeze_encoder_epochs', 15)
    config['model_version'] = f"OCM_imagenet_{encoder_name}"
    config['demo_mode'] = False
    config['learning_rate'] = config.get('learning_rate', 0.001)
    config['random_seed'] = 42
    
    # Create label_weights dictionary (weights for each dataset directory)
    config['label_weights'] = {
        train_dir: 1.0,
        val_dir: 1.0,
    }
    
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train OCM model from scratch with ImageNet encoder')
    parser.add_argument(
        '--encoder',
        type=str,
        default='mit_b5',  # RECOMMENDED: Vision Transformer for limited data
        choices=['mit_b5', 'mit_b4', 'efficientnet_b5', 'efficientnet_b4', 'efficientnet_b3', 'resnet34', 'mit_b3'],
        help='Encoder architecture to use (RECOMMENDED: mit_b5 or mit_b4 for limited data)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Override learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--freeze-epochs',
        type=int,
        default=None,
        help='Override freeze encoder epochs (default: 10)'
    )
    parser.add_argument(
        '--unfrozen-epochs',
        type=int,
        default=None,
        help='Override unfrozen epochs (default: 30)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Print system info
    print_system_info()
    
    # Create configurations
    model_config = create_model_config(args.encoder)
    training_config = create_training_config(args.encoder, args.lr)
    
    # Override epochs if specified
    if args.freeze_epochs:
        training_config['freeze_encoder_epochs'] = args.freeze_epochs
    if args.unfrozen_epochs:
        training_config['unfrozen_epochs'] = args.unfrozen_epochs
    
    # Set output directory
    if args.output_dir:
        output_base_dir = Path(args.output_dir)
    else:
        output_base_dir = Path(f"fine_tune_results/exp2_imagenet_{args.encoder}_v4")
    
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Log configuration
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 2: Training from Scratch with ImageNet Encoder")
    logger.info("="*60)
    logger.info(f"Encoder: {args.encoder}")
    logger.info(f"Model type: {model_config['model_type']}")
    logger.info(f"Learning rate: {training_config['learning_rate']}")
    logger.info(f"Batch size: {training_config['batch_size']}")
    logger.info(f"Freeze epochs: {training_config['freeze_encoder_epochs']}")
    logger.info(f"Unfrozen epochs: {training_config['unfrozen_epochs']}")
    logger.info(f"Output directory: {output_base_dir}")
    logger.info("="*60 + "\n")
    
    # Train model
    success = fine_tune_single_model(
        model_config=model_config,
        training_config=training_config,
        output_base_dir=output_base_dir
    )
    
    if success:
        logger.info("\n" + "="*60)
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to: {output_base_dir}")
        logger.info("="*60)
    else:
        logger.error("\n" + "="*60)
        logger.error("Training failed!")
        logger.error("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()
