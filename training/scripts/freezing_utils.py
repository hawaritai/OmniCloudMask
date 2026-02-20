"""
Universal freezing utilities for BOTH SMP Unet and fastai DynamicUnet models.

Works with:
- SMP models: model.encoder, model.decoder, model.segmentation_head
- fastai DynamicUnet: SequentialEx where model[0] is encoder

USAGE:
    from freezing_utils import freeze_encoder, unfreeze_all, print_frozen_status
    
    learner = create_learner(...)
    
    # Phase 1: Frozen
    freeze_encoder(learner)
    learner.fit_one_cycle(freeze_epochs, 1e-3)
    
    # Phase 2: Unfrozen
    unfreeze_all(learner)
    learner.fit_one_cycle(unfrozen_epochs, slice(1e-5, 1e-2))
"""

import torch.nn as nn
from fastai.vision.models.unet import DynamicUnet 

def is_smp_model(model):
    """Check if model is SMP Unet (has encoder attribute)."""
    return hasattr(model, 'encoder')


def is_fastai_dynamic_unet(model):
    """Check if model is fastai DynamicUnet (Sequential with indexed access)."""
    return isinstance(model, DynamicUnet) and len(list(model.children())) > 0


def freeze_encoder(learner):
    """
    Freeze encoder for both SMP and fastai DynamicUnet models.
    
    Automatically detects model type and freezes appropriately:
    - SMP: Freezes model.encoder
    - fastai DynamicUnet: Freezes model[0] (first sequential layer = encoder)
    
    Args:
        learner: fastai Learner object
        
    Returns:
        int: Number of frozen parameters
    """
    model = learner.model
    frozen = 0
    
    # ========== SMP Model ==========
    if is_smp_model(model):
        for param in model.encoder.parameters():
            param.requires_grad = False
        frozen = sum(1 for p in learner.model.parameters() if not p.requires_grad)
        total = sum(1 for p in learner.model.parameters())
        print(f"✓ SMP Model: Encoder frozen")
        print(f"✓ Frozen: {frozen}/{total} parameters ({frozen/total*100:.1f}%)")
        return frozen
    
    # ========== fastai DynamicUnet ==========
    elif is_fastai_dynamic_unet(model):
        # In DynamicUnet, layers[0] is the encoder (timm backbone)
        encoder = model[0]
        for param in encoder.parameters():
            param.requires_grad = False
        frozen = sum(1 for p in learner.model.parameters() if not p.requires_grad)
        total = sum(1 for p in learner.model.parameters())
        print(f"✓ fastai DynamicUnet: Encoder (layer 0) frozen")
        print(f"✓ Frozen: {frozen}/{total} parameters ({frozen/total*100:.1f}%)")
        return frozen
    
    # ========== Unknown ==========
    else:
        print("⚠ ERROR: Unknown model structure!")
        print(f"  - Has 'encoder': {hasattr(model, 'encoder')}")
        print(f"  - Is Sequential: {isinstance(model, nn.Sequential)}")
        return 0


def unfreeze_all(learner):
    """
    Unfreeze all parameters for both SMP and fastai models.
    
    Args:
        learner: fastai Learner object
        
    Returns:
        int: Number of frozen parameters (should be 0)
    """
    for param in learner.model.parameters():
        param.requires_grad = True
    
    frozen = sum(1 for p in learner.model.parameters() if not p.requires_grad)
    total = sum(1 for p in learner.model.parameters())
    
    print(f"✓ All unfrozen: {frozen}/{total} parameters frozen ({frozen/total*100:.1f}%)")
    return frozen


def print_frozen_status(learner):
    """
    Print detailed freeze status.
    
    Args:
        learner: fastai Learner object
    """
    frozen = sum(1 for p in learner.model.parameters() if not p.requires_grad)
    total = sum(1 for p in learner.model.parameters())
    trainable = total - frozen
    
    print(f"\n{'='*60}")
    print(f"Model Freeze Status:")
    print(f"{'='*60}")
    print(f"  Total parameters:     {total:,}")
    print(f"  Frozen parameters:    {frozen:,} ({frozen/total*100:.1f}%)")
    print(f"  Trainable parameters: {trainable:,} ({trainable/total*100:.1f}%)")
    print(f"{'='*60}\n")


def get_lr_ranges(freeze_encoder: bool = True, training_config: dict = None):
    """
    Get learning rates for frozen vs unfrozen phases.
    
    Args:
        freeze_encoder: Whether encoder is frozen
        training_config: Optional training config with custom LR values
        
    Returns:
        float or slice:
            - Single LR if frozen (decoder only)
            - slice(encoder_lr, decoder_lr) if unfrozen (discriminative LRs)
    """
    # Default values for fine-tuning
    lr_frozen_default = 1e-3
    lr_encoder_default = 1e-5
    lr_decoder_default = 1e-2
    
    # Use config values if provided
    if training_config is not None:
        lr_frozen = training_config.get('lr_frozen', lr_frozen_default)
        lr_encoder = training_config.get('lr_encoder', lr_encoder_default)
        lr_decoder = training_config.get('lr_decoder', lr_decoder_default)
    else:
        lr_frozen = lr_frozen_default
        lr_encoder = lr_encoder_default
        lr_decoder = lr_decoder_default
    
    if freeze_encoder:
        return lr_frozen
    else:
        return slice(lr_encoder, lr_decoder)


def inspect_model_structure(learner):
    """
    Inspect and print model structure (for debugging).
    
    Args:
        learner: fastai Learner object
    """
    model = learner.model
    
    print(f"\n{'='*70}")
    print(f"Model Structure Inspection:")
    print(f"{'='*70}")
    print(f"Model type: {type(model).__name__}")
    
    # SMP model
    if is_smp_model(model):
        print(f"\n✓ Detected: SMP Unet Model")
        print(f"\nTop-level components:")
        for name, module in model.named_children():
            param_count = sum(1 for _ in module.parameters())
            print(f"  - {name}: {type(module).__name__} ({param_count:,} parameters)")
    
    # fastai DynamicUnet
    elif is_fastai_dynamic_unet(model):
        print(f"\n✓ Detected: fastai DynamicUnet (SequentialEx)")
        print(f"\nLayers in sequence:")
        for idx, layer in enumerate(model):
            param_count = sum(1 for _ in layer.parameters())
            layer_name = type(layer).__name__
            print(f"  [{idx}] {layer_name} ({param_count:,} parameters)")
            if idx == 0:
                print(f"       ^ ENCODER (timm backbone) - frozen during phase 1")
            else:
                print(f"       ^ Part of decoder")
    
    else:
        print(f"\n⚠ Unknown model type")
    
    print(f"{'='*70}\n")