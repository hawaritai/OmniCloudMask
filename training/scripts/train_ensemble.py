"""
Train ensemble of models for better performance with limited data.

This script trains multiple models (e.g., MIT-b4 + EfficientNet-b5)
and combines their predictions at test time. This compensates for limited
training data by leveraging diverse architectures.

Usage:
    # Train ensemble (recommended for limited data)
    python train_ensemble.py --models mit_b4 efficientnet_b5
    
    # Train larger ensemble
    python train_ensemble.py --models mit_b5 efficientnet_b5 resnet34
"""

import sys
from pathlib import Path
import logging
import argparse
from typing import List

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import training utilities
from train_from_scratch import (
    create_model_config,
    create_training_config,
    print_system_info
)

# Import fine-tuning function
from fine_tune_multiple_models import fine_tune_single_model


def train_ensemble(
    encoder_names: List[str],
    output_base_dir: Path,
    lr: float = None,
    freeze_epochs: int = None,
    unfreeze_epochs: int = None
) -> List[Path]:
    """
    Train multiple models as an ensemble.
    
    Args:
        encoder_names: List of encoder names to train
        output_base_dir: Base directory for results
        lr: Optional learning rate override
        freeze_epochs: Optional freeze epochs override
        unfreeze_epochs: Optional unfreeze epochs override
        
    Returns:
        List of paths to trained models
    """
    trained_models = []
    
    for i, encoder_name in enumerate(encoder_names, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"Training ensemble model {i}/{len(encoder_names)}: {encoder_name}")
        logger.info(f"{'='*70}\n")
        
        # Create configurations for this encoder
        model_config = create_model_config(encoder_name)
        training_config = create_training_config(encoder_name, lr)
        
        # Override epochs if specified
        if freeze_epochs:
            training_config['freeze_encoder_epochs'] = freeze_epochs
        if unfreeze_epochs:
            training_config['unfreeze_epochs'] = unfreeze_epochs
        
        # Set output directory for this model
        model_output_dir = output_base_dir / f"ensemble_model_{i}_{encoder_name}"
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Train model
        success = fine_tune_single_model(
            model_config=model_config,
            training_config=training_config,
            output_base_dir=model_output_dir
        )
        
        if success:
            # Find best model path
            models_dir = model_output_dir / "models"
            best_model = list(models_dir.glob("*_best.safetensors"))
            if best_model:
                trained_models.append(best_model[0])
                logger.info(f"✓ Successfully trained {encoder_name}")
                logger.info(f"  Best model: {best_model[0]}")
            else:
                logger.warning(f"⚠ No best model found for {encoder_name}")
        else:
            logger.error(f"✗ Failed to train {encoder_name}")
    
    return trained_models


def create_ensemble_config(trained_models: List[Path]) -> dict:
    """
    Create configuration file for ensemble inference.
    
    Args:
        trained_models: List of paths to trained models
        
    Returns:
        Dictionary with ensemble configuration
    """
    return {
        'ensemble_type': 'averaging',  # Can be 'averaging' or 'voting'
        'models': [str(model) for model in trained_models],
        'num_models': len(trained_models),
        'weights': 'equal',  # Equal weights for averaging
    }


def save_ensemble_config(
    config: dict,
    output_dir: Path
):
    """
    Save ensemble configuration to JSON file.
    
    Args:
        config: Ensemble configuration dictionary
        output_dir: Directory to save config
    """
    import json
    
    config_path = output_dir / "ensemble_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Saved ensemble configuration to: {config_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train ensemble of OCM models for better performance with limited data'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['mit_b4', 'efficientnet_b5'],  # RECOMMENDED: ViT + CNN
        choices=['mit_b5', 'mit_b4', 'efficientnet_b5', 'efficientnet_b4', 'efficientnet_b3', 'resnet34', 'mit_b3'],
        help='Encoder architectures to train in ensemble (space-separated)'
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
        '--unfreeze-epochs',
        type=int,
        default=None,
        help='Override unfreeze epochs (default: 30)'
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
    
    # Set output directory
    if args.output_dir:
        output_base_dir = Path(args.output_dir)
    else:
        output_base_dir = Path(f"fine_tune_results/ensemble_{'_'.join(args.models)}")
    
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Log configuration
    logger.info("\n" + "="*70)
    logger.info("ENSEMBLE TRAINING: Multiple Models for Limited Data")
    logger.info("="*70)
    logger.info(f"Models to train: {', '.join(args.models)}")
    logger.info(f"Number of models: {len(args.models)}")
    logger.info(f"Learning rate: {args.lr if args.lr else '0.001 (default)'}")
    logger.info(f"Freeze epochs: {args.freeze_epochs if args.freeze_epochs else '10 (default)'}")
    logger.info(f"Unfreeze epochs: {args.unfreeze_epochs if args.unfreeze_epochs else '30 (default)'}")
    logger.info(f"Output directory: {output_base_dir}")
    logger.info("="*70 + "\n")
    
    # Train ensemble
    trained_models = train_ensemble(
        encoder_names=args.models,
        output_base_dir=output_base_dir,
        lr=args.lr,
        freeze_epochs=args.freeze_epochs,
        unfreeze_epochs=args.unfreeze_epochs
    )
    
    # Save ensemble configuration
    if trained_models:
        logger.info("\n" + "="*70)
        logger.info(f"ENSEMBLE TRAINING COMPLETED: {len(trained_models)}/{len(args.models)} models")
        logger.info("="*70)
        
        ensemble_config = create_ensemble_config(trained_models)
        save_ensemble_config(ensemble_config, output_base_dir)
        
        logger.info("\nTrained models:")
        for i, model_path in enumerate(trained_models, 1):
            logger.info(f"  {i}. {model_path}")
        
        logger.info("\nNext steps:")
        logger.info("  1. Evaluate each model on validation set")
        logger.info("  2. Compare individual and ensemble performance")
        logger.info("  3. Use ensemble for inference (average predictions)")
        logger.info("="*70)
    else:
        logger.error("\n" + "="*70)
        logger.error("ENSEMBLE TRAINING FAILED: No models trained successfully")
        logger.error("="*70)
        sys.exit(1)


if __name__ == "__main__":
    main()
