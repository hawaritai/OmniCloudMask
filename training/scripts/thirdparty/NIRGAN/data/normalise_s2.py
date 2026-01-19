import torch

def minmax_percentile(img: torch.Tensor, perc: float = 2.0) -> torch.Tensor:
    """
    Perform percentile-based normalization on a tensor image.
    
    Args:
        img (torch.Tensor): Input tensor of shape (C, H, W) or (B, C, H, W).
        perc (float): Percentile value for stretching (e.g., 2.0 means 2nd and 98th percentiles).
        
    Returns:
        torch.Tensor: Normalized tensor with values in [0, 1], stretched based on percentiles.
    """
    assert 0 <= perc < 50, "Percentile must be in the range [0, 50)."

    orig_shape = img.shape
    is_batched = img.dim() == 4  # (B, C, H, W)
    
    if not is_batched:
        img = img.unsqueeze(0)  # Add batch dimension

    B, C, H, W = img.shape
    img_reshaped = img.view(B, C, -1)  # Flatten H and W

    low = torch.quantile(img_reshaped, perc / 100.0, dim=-1, keepdim=True)
    high = torch.quantile(img_reshaped, 1.0 - perc / 100.0, dim=-1, keepdim=True)

    # Stretching
    img_stretched = (img_reshaped - low) / (high - low + 1e-6)
    img_stretched = img_stretched.clamp(0, 1)

    # Reshape back
    img_out = img_stretched.view(B, C, H, W)

    return img_out[0] if not is_batched else img_out
