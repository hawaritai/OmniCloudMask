import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Add this FIRST

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
    preds_result = learner.get_preds(dl=dl)
    
    # Handle different return types from get_preds
    if isinstance(preds_result, tuple):
        preds_output, targs_output = preds_result[0], preds_result[1]
    else:
        preds_output, targs_output = preds_result, preds_result

    # Helper to extract mask from (mask, weights) tuple if present
    def extract_mask(t):
        # If it's a tuple/list of tensors, check if it looks like (mask, weights)
        if isinstance(t, (tuple, list)) and len(t) >= 2:
            if isinstance(t[0], torch.Tensor) and isinstance(t[1], torch.Tensor):
                # Heuristic: weights usually have one less dimension or different shape
                if t[0].shape != t[1].shape:
                    return t[0]
        return t

    valid_preds_list = []
    valid_targs_list = []
    valid_probs_list = []

    # Check if we have a list of batches (variable sizes) or a single tensor
    # FastAI L is iterable but not isinstance(list)
    is_list_of_batches = (hasattr(preds_output, '__iter__') and not isinstance(preds_output, torch.Tensor))
    
    # If it's a single tensor, wrap it in a list to treat it uniformly as a "single batch"
    if not is_list_of_batches:
        preds_batches = [preds_output]
        targs_batches = [targs_output]
    else:
        preds_batches = list(preds_output) # Convert L to list
        targs_batches = list(targs_output)

    print(f"Processing {len(preds_batches)} batches/items for metrics...")

    for batch_idx, (p_batch, t_batch) in enumerate(zip(preds_batches, targs_batches)):
        # p_batch: [B, C, H, W] (or [C, H, W] if not batched, but get_preds usually batches)
        # t_batch: [B, H, W] or tuple ([B, H, W], [B])
        
        # Convert FastAI L to standard list if encountered
        if hasattr(p_batch, '__iter__') and not isinstance(p_batch, torch.Tensor):
            p_batch = list(p_batch)
        if hasattr(t_batch, '__iter__') and not isinstance(t_batch, torch.Tensor):
            t_batch = list(t_batch)

        # 1. Handle (mask, weights) tuple in target
        t_mask = extract_mask(t_batch)
        
        # If t_mask became a list (from L conversion or extract_mask), ensure it's a list
        if hasattr(t_mask, '__iter__') and not isinstance(t_mask, torch.Tensor):
            t_mask = list(t_mask)

        # 2. Ensure tensors and move to CPU
        # If p_batch is a list of tensors (variable size batch), we cannot convert to tensor directly
        is_var_size = isinstance(p_batch, list) and len(p_batch) > 0 and isinstance(p_batch[0], torch.Tensor)
        
        if not is_var_size and not isinstance(p_batch, torch.Tensor):
             try:
                 p_batch = torch.as_tensor(p_batch)
             except Exception as e:
                 # It might be a list of tensors that failed the is_var_size check?
                 pass

        # Same for t_mask
        is_mask_list = isinstance(t_mask, list) and len(t_mask) > 0 and isinstance(t_mask[0], torch.Tensor)
        
        if not is_mask_list and not isinstance(t_mask, torch.Tensor):
             try:
                t_mask = torch.as_tensor(t_mask)
             except TypeError:
                 pass # Will be handled below if it's a list

        # Handle case where t_mask is still a list (variable sized batch)
        if isinstance(t_mask, (list, tuple)) and isinstance(p_batch, (list, tuple, torch.Tensor)):
            # If t_mask is a list, p_batch should also be iterable matching it
            # We will iterate inside this batch
            sub_probs = []
            sub_targs = []
            
            # Ensure p_batch is iterable
            if isinstance(p_batch, torch.Tensor):
                p_iterable = p_batch # Iterating tensor [B, C, H, W] gives [C, H, W] slices
            else:
                p_iterable = p_batch
            
            # If lengths don't match, something is wrong, but zip will truncate
            for sub_p, sub_t in zip(p_iterable, t_mask):
                sub_p = sub_p.cpu() if isinstance(sub_p, torch.Tensor) else torch.as_tensor(sub_p).cpu()
                sub_t = sub_t.cpu() if isinstance(sub_t, torch.Tensor) else torch.as_tensor(sub_t).cpu()
                
                # sub_p might be [C, H, W]
                sp_probs = torch.softmax(sub_p, dim=0) # [C, H, W] -> softmax over C
                
                # Flatten
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
            continue # Skip standard processing for this batch

        p_batch = p_batch.cpu()
        t_mask = t_mask.cpu()

        # 3. Compute Softmax (Probabilities)
        # p_batch is [B, C, H, W]
        probs = torch.softmax(p_batch, dim=1)
        
        # 4. Flatten Spatial Dimensions
        # We need to pair every pixel prediction with its target.
        # Permute to [B, H, W, C] then flatten to [N_pixels, C]
        num_classes = probs.shape[1]
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, num_classes)
        targs_flat = t_mask.flatten()
        
        # 5. Filter Ignore Index
        # This drastically reduces memory usage if there's a lot of padding/nodata
        mask = targs_flat != ignore_index
        
        if mask.sum() > 0:
            valid_probs_list.append(probs_flat[mask].numpy())
            valid_targs_list.append(targs_flat[mask].numpy())
        
        # Optional: Print progress for large sets
        if batch_idx % 20 == 0 and batch_idx > 0:
            print(f"Processed {batch_idx}/{len(preds_batches)} batches...")

    if not valid_probs_list:
        print(f"Warning: No valid pixels found (all matched ignore_index={ignore_index}). Metrics cannot be computed.")
        return

    # Concatenate all valid pixels from all batches
    print("Concatenating results...")
    valid_probs = np.concatenate(valid_probs_list)
    valid_targs = np.concatenate(valid_targs_list)
    
    # Argmax for confusion matrix
    valid_preds = valid_probs.argmax(axis=1)
     
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
    
    plt.figure(figsize=(10, 8))
    
    # valid_probs is already [N_valid_pixels, num_classes]
    # valid_targs is [N_valid_pixels]
    
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