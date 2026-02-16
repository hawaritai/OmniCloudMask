import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Ensure headless compatibility

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, ConfusionMatrixDisplay, PrecisionRecallDisplay
from pathlib import Path
from fastai.vision.all import *

def compute_and_plot_metrics(learner, dl, dataset_name, save_dir, class_names=None, ignore_index=99):
    """
    Computes confusion matrix and PR curves for a given learner and dataloader.
    Uses robust batch processing to handle variable-sized inputs and FastAI L types.
    """
    if class_names is None:
        class_names = ['Clear', 'Thick Cloud', 'Thin Cloud', 'Cloud Shadow']
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating metrics for {dataset_name} set...")
    
    # Get predictions
    # preds_result is usually (preds, targs)
    preds_result = learner.get_preds(dl=dl)
    
    # Handle different return types from get_preds
    if isinstance(preds_result, tuple):
        preds_output, targs_output = preds_result[0], preds_result[1]
    else:
        preds_output, targs_output = preds_result, preds_result

    # Fix for multiple targets (e.g., mask + weights) at the dataset level
    # If targs_output is a tuple of lists (masks, weights), we want just the masks list
    if isinstance(targs_output, (tuple, list)) and len(targs_output) >= 2:
        # Check if the first element corresponds to preds in length
        if hasattr(targs_output[0], '__len__') and len(targs_output[0]) == len(preds_output):
             print(f"Handling multiple targets at dataset level (e.g. masks, weights). Using first element.")
             targs_output = targs_output[0]

    # Helper to extract mask from (mask, weights) tuple if present (per item/batch)
    def extract_mask(t):
        if isinstance(t, (tuple, list)) and len(t) >= 2:
            if isinstance(t[0], torch.Tensor) and isinstance(t[1], torch.Tensor):
                if t[0].shape != t[1].shape:
                    return t[0]
        return t

    valid_probs_list = []
    valid_targs_list = []

    # Check if we have a list of batches (variable sizes) or a single tensor
    is_list_of_batches = (hasattr(preds_output, '__iter__') and not isinstance(preds_output, torch.Tensor))
    
    if not is_list_of_batches:
        preds_batches = [preds_output]
        targs_batches = [targs_output]
    else:
        preds_batches = list(preds_output)
        targs_batches = list(targs_output)

    print(f"Processing {len(preds_batches)} batches/items for metrics...")

    for batch_idx, (p_batch, t_batch) in enumerate(zip(preds_batches, targs_batches)):
        # Convert FastAI L to standard list if encountered
        if hasattr(p_batch, '__iter__') and not isinstance(p_batch, torch.Tensor):
            p_batch = list(p_batch)
        if hasattr(t_batch, '__iter__') and not isinstance(t_batch, torch.Tensor):
            t_batch = list(t_batch)

        t_mask = extract_mask(t_batch)
        
        if hasattr(t_mask, '__iter__') and not isinstance(t_mask, torch.Tensor):
            t_mask = list(t_mask)

        # Attempt to convert to tensor if possible (fixed size batch)
        is_var_size_p = isinstance(p_batch, list) and len(p_batch) > 0 and isinstance(p_batch[0], torch.Tensor)
        if not is_var_size_p and not isinstance(p_batch, torch.Tensor):
             try: p_batch = torch.as_tensor(p_batch)
             except Exception: pass

        is_var_size_t = isinstance(t_mask, list) and len(t_mask) > 0 and isinstance(t_mask[0], torch.Tensor)
        if not is_var_size_t and not isinstance(t_mask, torch.Tensor):
             try: t_mask = torch.as_tensor(t_mask)
             except TypeError: pass

        # Handle variable sized batch (list of tensors) - iterate within batch
        if isinstance(t_mask, (list, tuple)) and isinstance(p_batch, (list, tuple, torch.Tensor)):
            p_iterable = p_batch if not isinstance(p_batch, torch.Tensor) else p_batch
            
            sub_probs = []
            sub_targs = []
            for sub_p, sub_t in zip(p_iterable, t_mask):
                sub_p = sub_p.cpu() if isinstance(sub_p, torch.Tensor) else torch.as_tensor(sub_p).cpu()
                sub_t = sub_t.cpu() if isinstance(sub_t, torch.Tensor) else torch.as_tensor(sub_t).cpu()
                
                # sub_p is [C, H, W] -> softmax over dim 0
                sp_probs = torch.softmax(sub_p, dim=0)
                num_c = sp_probs.shape[0]
                sp_flat = sp_probs.permute(1, 2, 0).reshape(-1, num_c)
                st_flat = sub_t.flatten()
                
                m = st_flat != ignore_index
                if m.sum() > 0:
                    sub_probs.append(sp_flat[m].numpy())
                    sub_targs.append(st_flat[m].numpy())
            
            if sub_probs:
                valid_probs_list.extend(sub_probs)
                valid_targs_list.extend(sub_targs)
            continue

        # Standard processing (Single item or Collated Batch)
        # Ensure they are tensors
        if not isinstance(p_batch, torch.Tensor): p_batch = torch.as_tensor(p_batch)
        if not isinstance(t_mask, torch.Tensor): t_mask = torch.as_tensor(t_mask)

        p_batch = p_batch.cpu()
        t_mask = t_mask.cpu()
        
        # Check dimensions. If [C, H, W] (single item), unsqueeze to [1, C, H, W]
        if p_batch.ndim == 3:
            p_batch = p_batch.unsqueeze(0)
        # If mask is [H, W], unsqueeze to [1, H, W]
        if t_mask.ndim == 2:
            t_mask = t_mask.unsqueeze(0)

        # Now p_batch is [B, C, H, W], t_mask is [B, H, W]
        probs = torch.softmax(p_batch, dim=1)
        num_classes = probs.shape[1]
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, num_classes)
        targs_flat = t_mask.flatten()
        
        mask = targs_flat != ignore_index
        if mask.sum() > 0:
            valid_probs_list.append(probs_flat[mask].numpy())
            valid_targs_list.append(targs_flat[mask].numpy())
        
        if batch_idx % 20 == 0 and batch_idx > 0:
            print(f"Processed {batch_idx}/{len(preds_batches)} batches...")

    if not valid_probs_list:
        print(f"Warning: No valid pixels found. Metrics cannot be computed.")
        return

    print("Concatenating results...")
    valid_probs = np.concatenate(valid_probs_list)
    valid_targs = np.concatenate(valid_targs_list)
    valid_preds = valid_probs.argmax(axis=1)
     
    # --- Confusion Matrix ---
    print(f"Computing Confusion Matrix for {dataset_name}...")
    cm = confusion_matrix(valid_targs, valid_preds, labels=range(len(class_names)))
    
    # Normalize CM
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    # Plot CM using sklearn
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
    disp.plot(cmap='Blues', ax=ax, values_format='.4f')
    ax.set_title(f'{dataset_name} Confusion Matrix (Normalized)')
    plt.tight_layout()
    plt.savefig(save_dir / f'{dataset_name.lower()}_confusion_matrix.png')
    plt.close()
    
    # --- Precision-Recall Curves ---
    print(f"Computing PR Curves for {dataset_name}...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i, class_name in enumerate(class_names):
        binary_targets = (valid_targs == i).astype(int)
        class_probs = valid_probs[:, i]
        
        precision, recall, _ = precision_recall_curve(binary_targets, class_probs)
        ap = average_precision_score(binary_targets, class_probs)
        
        # Use estimator_name or name depending on sklearn version, but estimator_name is safer for older ones
        display = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=ap, estimator_name=class_name)
        display.plot(ax=ax, linewidth=2)
        
    ax.set_title(f'{dataset_name} Precision-Recall Curve')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / f'{dataset_name.lower()}_pr_curve.png')
    plt.close()
    
    print(f"Metrics saved to {save_dir}")
