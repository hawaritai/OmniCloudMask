import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from pathlib import Path
import seaborn as sns
from fastai.vision.all import *

def compute_and_plot_metrics(learner, dl, dataset_name, save_dir, class_names=None, ignore_index=99):
    """
    Computes confusion matrix and PR curves for a given learner and dataloader.
    
    Args:
        learner: The FastAI learner.
        dl: The dataloader to evaluate (train or valid).
        dataset_name: 'Train' or 'Validation' (for labeling plots).
        save_dir: Path to save the plots.
        class_names: List of class names. Defaults to ['Clear', 'Thick Cloud', 'Thin Cloud', 'Cloud Shadow'].
        ignore_index: Index to ignore in metrics (e.g., NoData).
    """
    if class_names is None:
        class_names = ['Clear', 'Thick Cloud', 'Thin Cloud', 'Cloud Shadow']
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating metrics for {dataset_name} set...")
    
    # Get predictions
    # preds: [N, C, H, W], targs: [N, H, W]
    preds_tensor, targs_tensor = learner.get_preds(dl=dl)
    
    # Convert to probabilities (softmax)
    probs_tensor = torch.softmax(preds_tensor, dim=1)
    
    # Flatten for metric calculation
    # Move to CPU and numpy
    # We process in chunks or all at once? 
    # For hundreds of images (509x509), flat arrays will be large.
    # 500 images * 500 * 500 = 125,000,000 pixels.
    # This fits in RAM (125M * 4 bytes ~ 500MB).
    
    # 1. Prepare for Confusion Matrix (Hard predictions)
    hard_preds = preds_tensor.argmax(dim=1).flatten()
    flat_targs = targs_tensor.flatten()
    
    # Filter out ignore_index
    mask = flat_targs != ignore_index
    valid_preds = hard_preds[mask].numpy()
    valid_targs = flat_targs[mask].numpy()
    
    # --- Confusion Matrix ---
    print(f"Computing Confusion Matrix for {dataset_name}...")
    cm = confusion_matrix(valid_targs, valid_preds, labels=range(len(class_names)))
    
    # Normalize CM
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm) # Handle divide by zero if a class is missing
    
    # Plot CM
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{dataset_name} Confusion Matrix (Normalized)')
    plt.tight_layout()
    plt.savefig(save_dir / f'{dataset_name.lower()}_confusion_matrix.png')
    plt.close()
    
    # --- Precision-Recall Curves ---
    print(f"Computing PR Curves for {dataset_name}...")
    # For PR curves, we need probabilities per class.
    # We already have probs_tensor: [N, C, H, W]
    # We need to index the probabilities by class.
    
    plt.figure(figsize=(10, 8))
    
    # Use the same mask to filter probabilities
    # We need to permute probs to [N*H*W, C] to apply the 1D mask
    num_classes = len(class_names)
    probs_flat = probs_tensor.permute(0, 2, 3, 1).reshape(-1, num_classes)
    
    # Filter using the mask derived from targets
    valid_probs = probs_flat[mask].numpy()
    # valid_targs is already numpy and filtered
    
    for i, class_name in enumerate(class_names):
        # Create binary target for this class
        binary_targets = (valid_targs == i).astype(int)
        
        # Get probabilities for this class
        class_probs = valid_probs[:, i]
        
        # Compute PR curve
        precision, recall, _ = precision_recall_curve(binary_targets, class_probs)
        ap = average_precision_score(binary_targets, class_probs)
        
        plt.plot(recall, precision, lw=2, label=f'{class_name} (AP = {ap:.2f})')
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{dataset_name} Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / f'{dataset_name.lower()}_pr_curve.png')
    plt.close()
    
    print(f"Metrics saved to {save_dir}")

