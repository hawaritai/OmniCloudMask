import sys
from pathlib import Path

# Add project root to sys.path to allow importing omnicloudmask
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Union, Optional
import torch
from omnicloudmask.model_utils import load_model_from_weights, build_model, load_weights

def build_custom_model(
    model_name: str,
    model_library: str = "smp",
    in_chans: int = 3,
    n_out: int = 4,
    encoder_weights: Optional[str] = None,
) -> torch.nn.Module:
    """Build model with optional ImageNet pre-trained encoder."""
    
    # For timm-efficientnet encoders, use 'imagenet' weights
    if encoder_weights is None and model_name.startswith('timm-'):
        encoder_weights = None  # Set to 'imagenet' if you want pretrained
    
    return build_model(
        model_name=model_name,
        model_library=model_library,
        in_chans=in_chans,
        n_out=n_out,
        encoder_weights=encoder_weights
    )

def load_custom_weights(
    model: torch.nn.Module,
    weights_path: Union[Path, str],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    strict: bool = True,
) -> torch.nn.Module:
    """Wrapper around omnicloudmask.model_utils.load_weights."""
    return load_weights(
        model=model,
        weights_path=weights_path,
        device=device,
        dtype=dtype,
        strict=strict
    )

def load_custom_model_from_weights(
    model_name: str,
    weights_path: Union[Path, str],
    model_library: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    in_chans: int = 3,
    n_out: int = 4,
    compile_models: bool = False,
    patch_size: int = 1000,
    batch_size: int = 1,
    compile_mode: str = "default",
    encoder_weights: Optional[str] = None,
) -> torch.nn.Module:
    """Wrapper around omnicloudmask.model_utils.load_model_from_weights.
    
    Now that the core function supports encoder_weights, this wrapper simply
    delegates to it, avoiding logic duplication while maintaining the 
    interface used by training scripts.
    """
    return load_model_from_weights(
        model_name=model_name,
        weights_path=weights_path,
        model_library=model_library,
        device=device,
        dtype=dtype,
        in_chans=in_chans,
        n_out=n_out,
        compile_models=compile_models,
        patch_size=patch_size,
        batch_size=batch_size,
        compile_mode=compile_mode,
        encoder_weights=encoder_weights,
    )
