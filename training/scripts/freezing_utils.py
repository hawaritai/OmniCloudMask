"""
Manual freezing utilities for both SMP and fastai models.
Works directly with requires_grad instead of relying on fastai's layer_groups.

Place this file in the same directory as your training script.
"""

def freeze_encoder(learner):
    """
    Manually freeze the encoder by setting requires_grad=False on all encoder parameters.
    Works for both SMP and fastai models.
    
    Args:
        learner: fastai Learner object
        
    Returns:
        int: Number of frozen parameters
    """
    for param in learner.model.encoder.parameters():
        param.requires_grad = False
    
    frozen = sum(1 for p in learner.model.parameters() if not p.requires_grad)
    total = sum(1 for p in learner.model.parameters())
    
    print(f"✓ Encoder frozen: {frozen}/{total} parameters frozen ({frozen/total*100:.1f}%)")
    return frozen


def unfreeze_all(learner):
    """
    Unfreeze all parameters by setting requires_grad=True.
    
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


def freeze_stages(learner, num_frozen_stages: int = 3):
    """
    For fastai models with explicit stages, freeze first N stages of encoder.
    Only works with fastai models that have encoder.model.stages_X structure.
    
    For SMP models or models without stages, use freeze_encoder() instead.
    
    Args:
        learner: fastai Learner object
        num_frozen_stages: Number of stages to freeze (0-4 for typical encoders)
        
    Returns:
        int: Number of frozen parameters
    """
    model = learner.model
    
    # Check if this is a fastai model with stages
    if not hasattr(model.encoder, 'model'):
        print("⚠ Model doesn't have encoder.model structure. Using freeze_encoder() instead.")
        return freeze_encoder(learner)
    
    encoder_model = model.encoder.model
    
    # Freeze stems
    for param in encoder_model.stem_0.parameters():
        param.requires_grad = False
    for param in encoder_model.stem_1.parameters():
        param.requires_grad = False
    
    # Freeze stages 0 to num_frozen_stages-1
    for stage_idx in range(num_frozen_stages):
        stage = getattr(encoder_model, f'stages_{stage_idx}', None)
        if stage:
            for param in stage.parameters():
                param.requires_grad = False
    
    frozen = sum(1 for p in learner.model.parameters() if not p.requires_grad)
    total = sum(1 for p in learner.model.parameters())
    
    print(f"✓ Stages 0-{num_frozen_stages-1} frozen: {frozen}/{total} parameters frozen ({frozen/total*100:.1f}%)")
    return frozen


def get_lr_ranges(freeze_encoder: bool = True):
    """
    Get learning rate ranges for discriminative fine-tuning.
    
    If encoder is frozen, use single LR. If unfrozen, use discriminative LRs
    where earlier layers get lower LRs.
    
    Args:
        freeze_encoder: Whether encoder is frozen
        
    Returns:
        float or slice: 
            - float (1e-3) if freeze_encoder=True (single LR for decoder)
            - slice(1e-5, 1e-2) if freeze_encoder=False (discriminative LRs)
    """
    if freeze_encoder:
        # Single LR for decoder only
        return 1e-3
    else:
        # Discriminative LRs: earlier layers train slower
        # slice(min_lr, max_lr) tells fastai to scale LRs across layers
        return slice(1e-5, 1e-2)


def print_frozen_status(learner):
    """
    Print detailed information about which parameters are frozen.
    Useful for debugging.
    
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