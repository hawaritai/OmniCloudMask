from typing import Optional, List, Dict
import logging
import time
import gc

import torch
import torch.nn.functional as F
from fastai.metrics import DiceMulti, RecallMulti, Metric
from fastai.callback.core import Callback, CancelFitException, CancelTrainException
from fastai.basics import store_attr
from torch import Tensor
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# --- CONSTANTS ---
MAX_SAVE_RETRIES = 3  # Maximum retry attempts for file operations
SAVE_RETRY_DELAY = 0.5  # Delay between retries in seconds


class DiceMultiStrip(DiceMulti):
    """DiceMulti that only looks at the first element if y is a tuple"""

    def accumulate(self, learn):
        # Temporarily modify learn.yb (the underlying batch) instead of learn.y
        yb_backup = learn.yb

        # Extract targets if it's a tuple
        if isinstance(learn.yb, (tuple, list)) and len(learn.yb) > 0:
            if isinstance(learn.yb[0], (tuple, list)):
                # If yb contains tuples, extract the first element of each tuple
                learn.yb = (learn.yb[0][0],)  # Just the targets, keep it as a tuple
            else:
                learn.yb = (learn.yb[0],)  # Wrap in tuple to maintain structure

        try:
            # Call the parent accumulate method
            super().accumulate(learn)
        finally:
            # Always restore the original yb
            learn.yb = yb_backup



class RecallMultiStrip(Metric):
    """Recall metric for multi-class segmentation that only looks at the first element if y is a tuple"""

    def __init__(self, axis=1):
        self.axis = axis
        self.reset()

    def reset(self):
        """Reset metric state"""
        self.tp = 0  # True positives
        self.fn = 0  # False negatives

    def accumulate(self, learn):
        """Accumulate metric for one batch"""
        # Temporarily modify learn.yb (the underlying batch) instead of learn.y
        yb_backup = learn.yb

        # Extract targets if it's a tuple
        if isinstance(learn.yb, (tuple, list)) and len(learn.yb) > 0:
            if isinstance(learn.yb[0], (tuple, list)):
                # If yb contains tuples, extract the first element of each tuple
                learn.yb = (learn.yb[0][0],)  # Just the targets, keep it as a tuple
            else:
                learn.yb = (learn.yb[0],)  # Wrap in tuple to maintain structure

        try:
            # Get predictions and targets
            pred = learn.pred.argmax(dim=self.axis)
            targ = learn.y

            # Flatten for per-pixel comparison
            pred_flat = pred.flatten()
            targ_flat = targ.flatten()

            # Calculate True Positives and False Negatives for each class
            for c in range(learn.pred.shape[self.axis]):
                # True positives: predicted as c and actually c
                tp = ((pred_flat == c) & (targ_flat == c)).sum().item()
                # False negatives: actually c but not predicted as c
                fn = ((pred_flat != c) & (targ_flat == c)).sum().item()

                self.tp += tp
                self.fn += fn

        finally:
            # Always restore the original yb
            learn.yb = yb_backup

    @property
    def value(self):
        """Calculate recall: TP / (TP + FN)"""
        # Avoid division by zero
        if self.tp + self.fn == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)


class PrecisionMultiStrip(Metric):
    """
    Precision metric for multi-class segmentation that only looks at the first element if y is a tuple.

    Precision measures how many of the predicted positive pixels are actually positive.
    Formula: Precision = TP / (TP + FP)

    This metric is important as a guardrail against false positives - a model with
    high recall but low precision will predict many false positives, which is undesirable.
    """

    def __init__(self, axis=1):
        self.axis = axis
        self.reset()

    def reset(self):
        """Reset metric state"""
        self.tp = 0  # True positives
        self.fp = 0  # False positives

    def accumulate(self, learn):
        """Accumulate metric for one batch"""
        # Temporarily modify learn.yb (the underlying batch) instead of learn.y
        yb_backup = learn.yb

        # Extract targets if it's a tuple
        if isinstance(learn.yb, (tuple, list)) and len(learn.yb) > 0:
            if isinstance(learn.yb[0], (tuple, list)):
                # If yb contains tuples, extract the first element of each tuple
                learn.yb = (learn.yb[0][0],)  # Just the targets, keep it as a tuple
            else:
                learn.yb = (learn.yb[0],)  # Wrap in tuple to maintain structure

        try:
            # Get predictions and targets
            pred = learn.pred.argmax(dim=self.axis)
            targ = learn.y

            # Flatten for per-pixel comparison
            pred_flat = pred.flatten()
            targ_flat = targ.flatten()

            # Calculate True Positives and False Positives for each class
            for c in range(learn.pred.shape[self.axis]):
                # True positives: predicted as c and actually c
                tp = ((pred_flat == c) & (targ_flat == c)).sum().item()
                # False positives: predicted as c but actually not c
                fp = ((pred_flat == c) & (targ_flat != c)).sum().item()

                self.tp += tp
                self.fp += fp

        finally:
            # Always restore the original yb
            learn.yb = yb_backup

    @property
    def value(self):
        """Calculate precision: TP / (TP + FP)"""
        # Avoid division by zero
        if self.tp + self.fp == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)


class IoUMultiStrip(Metric):
    """
    IoU (Intersection over Union) metric for multi-class segmentation that only looks
    at the first element if y is a tuple.

    IoU measures the overlap between predicted and actual masks.
    Formula: IoU = TP / (TP + FP + FN)

    This metric provides a balanced view of model performance by considering both
    false positives and false negatives. It's useful for monitoring but not
    directly used in the composite score.
    """

    def __init__(self, axis=1):
        self.axis = axis
        self.reset()

    def reset(self):
        """Reset metric state"""
        self.tp = 0  # True positives
        self.fp = 0  # False positives
        self.fn = 0  # False negatives

    def accumulate(self, learn):
        """Accumulate metric for one batch"""
        # Temporarily modify learn.yb (the underlying batch) instead of learn.y
        yb_backup = learn.yb

        # Extract targets if it's a tuple
        if isinstance(learn.yb, (tuple, list)) and len(learn.yb) > 0:
            if isinstance(learn.yb[0], (tuple, list)):
                # If yb contains tuples, extract the first element of each tuple
                learn.yb = (learn.yb[0][0],)  # Just the targets, keep it as a tuple
            else:
                learn.yb = (learn.yb[0],)  # Wrap in tuple to maintain structure

        try:
            # Get predictions and targets
            pred = learn.pred.argmax(dim=self.axis)
            targ = learn.y

            # Flatten for per-pixel comparison
            pred_flat = pred.flatten()
            targ_flat = targ.flatten()

            # Calculate TP, FP, and FN for each class
            for c in range(learn.pred.shape[self.axis]):
                # True positives: predicted as c and actually c
                tp = ((pred_flat == c) & (targ_flat == c)).sum().item()
                # False positives: predicted as c but actually not c
                fp = ((pred_flat == c) & (targ_flat != c)).sum().item()
                # False negatives: actually c but not predicted as c
                fn = ((pred_flat != c) & (targ_flat == c)).sum().item()

                self.tp += tp
                self.fp += fp
                self.fn += fn

        finally:
            # Always restore the original yb
            learn.yb = yb_backup

    @property
    def value(self):
        """Calculate IoU: TP / (TP + FP + FN)"""
        # Avoid division by zero
        if self.tp + self.fp + self.fn == 0:
            return 0.0
        return self.tp / (self.tp + self.fp + self.fn)


class CrossEntropyLossFlatImageTypeWeighted:
    """
    Weighted cross-entropy loss for segmentation that supports:
    - Class weighting (penalize misclassification of specific classes more heavily)
    - Image weighting (penalize losses on hard/important images more heavily)
    - Ignore index for background/padding pixels
    """
    
    def __init__(
        self,
        axis: int = 1,
        ignore_index: int = 99,
        class_weights: Optional[Tensor] = None,
    ):
        self.axis = axis
        self.ignore_index = ignore_index
        # Ensure class_weights is on the correct device when needed
        self.class_weights = class_weights
        if self.class_weights is not None:
            self.class_weights = self.class_weights.float()

    def __call__(self, preds: Tensor, targets: Tensor, image_weights: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            preds: [bs, classes, h, w] - model predictions
            targets: [bs, h, w] - ground truth labels
            image_weights: [bs] - optional weights per image (may not be provided by fastai)
        """

        # Handle case where image_weights is not provided by fastai
        if image_weights is None:
            image_weights = torch.ones(targets.shape[0], device=targets.device)

        weights = image_weights.to(targets.device, dtype=torch.float)

        # Move class_weights to the correct device if they exist
        class_weights = self.class_weights
        if class_weights is not None:
            class_weights = class_weights.to(preds.device)

        _, classes, _, _ = preds.shape
        preds_flat = preds.permute(0, 2, 3, 1).contiguous().view(-1, classes)
        targets_flat = targets.reshape(-1)

        # Get cross-entropy loss with class weights, no reduction so we can apply image weights later
        pixel_losses = F.cross_entropy(
            preds_flat,
            targets_flat,
            ignore_index=self.ignore_index,
            reduction="none",
            weight=class_weights,  # This penalizes misclassification of important classes
        )
        
        # Some images have large areas of the ignore index, so we take the mean of each image loss
        image_losses = pixel_losses.reshape(targets.shape[0], -1).mean(dim=1)

        # Apply the image weights (for hard negative mining, etc)
        weighted_losses = image_losses * weights

        # Return mean loss
        return weighted_losses.mean()


class EarlyStoppingRecall(Callback):
    """
    Phase-aware early stopping based on RecallMulti with separate patience for frozen and unfrozen phases.
    
    This callback:
    - Maintains separate patience counters for frozen and unfrozen training phases
    - Tracks a single global best metric across both phases
    - Frozen phase early stop terminates only the frozen phase (continues to unfrozen)
    - Unfrozen phase early stop terminates the entire fine_tune process
    
    Args:
        monitor: Metric name to monitor for early stopping (default: 'recall_multi_strip')
        frozen_patience: Patience for frozen phase (default: 3)
        unfrozen_patience: Patience for unfrozen phase (default: 3)
        min_delta: Minimum improvement to consider as improvement (default: 0.001)
    """
    order = 60 # Run after Recorder

    def __init__(self, monitor: str = 'recall_multi_strip', frozen_patience: int = 3, 
                 unfrozen_patience: int = 3, min_delta: float = 0.001):
        self.monitor = monitor
        self.frozen_patience = frozen_patience
        self.unfrozen_patience = unfrozen_patience
        self.min_delta = min_delta
        
        # Global best tracking (shared across both phases)
        self.global_best_value = -np.inf
        self.global_best_epoch = 0
        
        # Phase-specific patience counters
        self.phase_states = {
            "frozen": {
                "patience": frozen_patience,
                "counter": 0
            },
            "unfrozen": {
                "patience": unfrozen_patience,
                "counter": 0
            }
        }
        
        # Track last phase for logging transitions
        self.last_phase = None
        self._evaluation_mode = False  # Guard: prevent saves during evaluation
        logger.debug(f"[EarlyStoppingRecall] Initialized with monitor='{monitor}', "
                    f"frozen_patience={frozen_patience}, unfrozen_patience={unfrozen_patience}")
    
    def after_epoch(self):
        """Called after epoch at the end of each epoch."""
        # CRITICAL: Skip during evaluation to prevent early stopping during get_preds()
        if getattr(self, '_evaluation_mode', False):
            logger.debug(f"[EarlyStoppingRecall] Evaluation mode ACTIVE - skipping early stop check")
            return
        
        # DIAGNOSTIC: Log recorder state
        logger.debug(f"[EarlyStoppingRecall] after_epoch called at epoch {self.epoch}")
        logger.debug(f"  self.monitor = '{self.monitor}'")
        
        # Determine current phase based on optimizer frozen_idx
        # frozen_idx > 0 means some layers are frozen (frozen phase)
        # frozen_idx == 0 means all layers are trainable (unfrozen phase)
        phase = "frozen" if self.learn.opt.frozen_idx > 0 else "unfrozen"
        
        # Log phase transitions
        if phase != self.last_phase:
            logger.info(f"[EarlyStoppingRecall] Switched to {phase} phase")
            self.last_phase = phase
        
        logger.debug(f"  Current phase: {phase} (frozen_idx={self.learn.opt.frozen_idx})")
        
        # Get the current metric value from recorder
        # Check if values list is not empty (may be empty before validation completes)
        if not self.learn.recorder.values or len(self.learn.recorder.values) == 0:
            logger.debug(f"  No values in recorder, returning early")
            return
        
        current_epoch_metrics = self.learn.recorder.values[-1]
        metric_names = self.learn.recorder.metric_names
        
        logger.debug(f"  metric_names = {metric_names}")
        logger.debug(f"  current_epoch_metrics = {current_epoch_metrics}")
        logger.debug(f"  len(metric_names) = {len(metric_names)}, len(current_epoch_metrics) = {len(current_epoch_metrics)}")
        
        # Find the index of the monitor metric
        if self.monitor not in metric_names:
            logger.debug(f"  WARNING: '{self.monitor}' not found in metric_names!")
            logger.debug(f"  Available metrics: {metric_names}")
            logger.debug(f"  Returning early - metric not found")
            return
        
        monitor_idx = metric_names.index(self.monitor)
        logger.debug(f"  monitor_idx = {monitor_idx} (for '{self.monitor}')")
        
        # Adjust index: metric_names includes 'epoch' at index 0, but values doesn't
        values_idx = monitor_idx - 1
        logger.debug(f"  values_idx = {values_idx} (monitor_idx - 1 to skip 'epoch')")
        
        if values_idx < 0 or values_idx >= len(current_epoch_metrics):
            logger.debug(f"  ERROR: values_idx {values_idx} is out of bounds for current_epoch_metrics (len={len(current_epoch_metrics)})")
            return
        
        current_value = current_epoch_metrics[values_idx]
        logger.debug(f"  current_value = {current_value}")
        
        # Validate metric value (NaN / Inf checks)
        if not isinstance(current_value, (int, float)):
            logger.debug(f"  WARNING: Metric is not a number: {type(current_value)}")
            return
        
        if np.isnan(current_value) or np.isinf(current_value):
            logger.debug(f"  WARNING: Invalid metric value at epoch {self.epoch}: {current_value}")
            return
        
        # Check if this is an improvement (compared to global best)
        if current_value > self.global_best_value + self.min_delta:
            # Update global best
            old_best_value = self.global_best_value
            old_best_epoch = self.global_best_epoch
            self.global_best_value = current_value
            self.global_best_epoch = self.epoch
            
            # Reset patience counter ONLY for current phase
            self.phase_states[phase]["counter"] = 0
            
            logger.info(f"  📈 New global best {self.monitor}: {current_value:.6f} at epoch {self.epoch} (phase={phase})")
            logger.debug(f"  📊 Previous best: {old_best_value:.6f} at epoch {old_best_epoch}")
            logger.debug(f"  🔍 Reset patience counter for {phase} phase")
        else:
            # Increment patience counter for current phase
            self.phase_states[phase]["counter"] += 1
            logger.debug(f"  No improvement. {phase} phase patience counter: "
                        f"{self.phase_states[phase]['counter']}/{self.phase_states[phase]['patience']}")
        
        # Check if patience exceeded for current phase
        phase_counter = self.phase_states[phase]["counter"]
        phase_patience = self.phase_states[phase]["patience"]
        
        if phase_counter >= phase_patience:
            if phase == "frozen":
                # Frozen phase stop: terminate frozen training only
                logger.info(f"\n[EarlyStoppingRecall] Frozen phase early stopping at epoch {self.epoch}. "
                           f"Best {self.monitor}: {self.global_best_value:.4f} at epoch {self.global_best_epoch}")
                logger.info(f"[EarlyStoppingRecall] Terminating frozen phase, continuing to unfrozen phase...")
                raise CancelTrainException()
            else:
                # Unfrozen phase stop: terminate entire training
                logger.info(f"\n[EarlyStoppingRecall] Unfrozen phase early stopping at epoch {self.epoch}. "
                           f"Best {self.monitor}: {self.global_best_value:.4f} at epoch {self.global_best_epoch}")
                logger.info(f"[EarlyStoppingRecall] Terminating entire fine_tune process...")
                raise CancelFitException()


class MetricLogger(Callback):
    """
    Callback to log and optionally save per-class metrics during training.
    Useful for monitoring recall on specific classes like 'Thin Cloud' and 'Cloud Shadow'.
    """
    order = 65  # Run after CompositeMetricCallback (60)

    def __init__(self, class_names: list, log_file: Optional[str] = None):
        self.class_names = class_names
        self.log_file = log_file
        self.epoch_logs = []
        self._evaluation_mode = False  # Guard: prevent saves during evaluation

    def after_epoch(self):
        """Log metrics after each epoch"""
        # CRITICAL: Skip during evaluation
        if getattr(self, '_evaluation_mode', False):
            logger.debug(f"[MetricLogger] Evaluation mode ACTIVE - skipping metric logging")
            return
        
        # DIAGNOSTIC: Log recorder state
        logger.debug(f"[MetricLogger] after_epoch called at epoch {self.epoch}")
        
        # Check if values list is not empty (may be empty before validation completes)
        if not self.learn.recorder.values or len(self.learn.recorder.values) == 0:
            logger.debug(f"  No values in recorder, returning early")
            return
        
        values = self.learn.recorder.values[-1]
        metric_names = self.learn.recorder.metric_names
        
        logger.debug(f"  metric_names = {metric_names}")
        logger.debug(f"  values = {values}")
        logger.debug(f"  len(metric_names) = {len(metric_names)}, len(values) = {len(values)}")
        
        epoch_log = {
            'epoch': self.epoch,
            'loss': values[0] if len(values) > 0 else None,
        }

        # Try to get per-class metrics from recorder
        try:
            # Log dice and recall if available
            if 'dice_multi_strip' in metric_names:
                dice_idx = metric_names.index('dice_multi_strip')
                # Adjust index: metric_names includes 'epoch' at index 0, but values doesn't
                values_idx = dice_idx - 1
                logger.debug(f"  dice_multi_strip: metric_idx={dice_idx}, values_idx={values_idx}")
                if values_idx >= 0 and values_idx < len(values):
                    epoch_log['dice'] = values[values_idx]
                else:
                    logger.debug(f"  WARNING: values_idx {values_idx} out of bounds for dice_multi_strip")

            if 'recall_multi_strip' in metric_names:
                recall_idx = metric_names.index('recall_multi_strip')
                # Adjust index: metric_names includes 'epoch' at index 0, but values doesn't
                values_idx = recall_idx - 1
                logger.debug(f"  recall_multi_strip: metric_idx={recall_idx}, values_idx={values_idx}")
                if values_idx >= 0 and values_idx < len(values):
                    epoch_log['recall'] = values[values_idx]
                else:
                    logger.debug(f"  WARNING: values_idx {values_idx} out of bounds for recall_multi_strip")

            # Log precision if available
            if 'precision_multi_strip' in metric_names:
                precision_idx = metric_names.index('precision_multi_strip')
                # Adjust index: metric_names includes 'epoch' at index 0, but values doesn't
                values_idx = precision_idx - 1
                logger.debug(f"  precision_multi_strip: metric_idx={precision_idx}, values_idx={values_idx}")
                if values_idx >= 0 and values_idx < len(values):
                    epoch_log['precision'] = values[values_idx]
                else:
                    logger.debug(f"  WARNING: values_idx {values_idx} out of bounds for precision_multi_strip")

            # Log IoU if available
            if 'iou_multi_strip' in metric_names:
                iou_idx = metric_names.index('iou_multi_strip')
                # Adjust index: metric_names includes 'epoch' at index 0, but values doesn't
                values_idx = iou_idx - 1
                logger.debug(f"  iou_multi_strip: metric_idx={iou_idx}, values_idx={values_idx}")
                if values_idx >= 0 and values_idx < len(values):
                    epoch_log['iou'] = values[values_idx]
                else:
                    logger.debug(f"  WARNING: values_idx {values_idx} out of bounds for iou_multi_strip")

            # Log composite score if available
            if 'composite_score' in metric_names:
                composite_idx = metric_names.index('composite_score')
                # Adjust index: metric_names includes 'epoch' at index 0, but values doesn't
                values_idx = composite_idx - 1
                logger.debug(f"  composite_score: metric_idx={composite_idx}, values_idx={values_idx}")
                if values_idx >= 0 and values_idx < len(values):
                    epoch_log['composite_score'] = values[values_idx]
                else:
                    logger.debug(f"  WARNING: values_idx {values_idx} out of bounds for composite_score")
            else:
                logger.debug(f"  NOTE: 'composite_score' not found in metric_names")

        except Exception as e:
            # If metrics aren't available yet, just log what we have
            logger.debug(f"  Exception in MetricLogger: {e}")
            import traceback
            traceback.print_exc()

        self.epoch_logs.append(epoch_log)

        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"Epoch {self.epoch}: {epoch_log}\n")


# --- UTILITY FUNCTIONS FOR TRAINING SCRIPTS ---

def safe_file_save(
    save_func,
    *args,
    operation_name: str = "file save",
    max_retries: int = MAX_SAVE_RETRIES,
    retry_delay: float = SAVE_RETRY_DELAY,
    **kwargs
) -> bool:
    """
    Safely save a file with retry logic for Windows file locking issues.

    Args:
        save_func: Function to call for saving (e.g., save_file, torch.save)
        *args: Positional arguments for save_func
        operation_name: Description of operation for logging
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        **kwargs: Keyword arguments for save_func

    Returns:
        True if save was successful, False otherwise
    """
    for attempt in range(max_retries):
        try:
            # Force garbage collection before save
            gc.collect()

            # Small delay before save to allow file system to settle
            if attempt > 0:
                time.sleep(retry_delay * (attempt + 1))

            # Attempt the save
            save_func(*args, **kwargs)

            if attempt > 0:
                logger.debug(f"    ✓ {operation_name} succeeded on attempt {attempt + 1}/{max_retries}")
            return True

        except OSError as e:
            error_code = getattr(e, 'winerror', None) or getattr(e, 'errno', None)

            # Error 1224: The requested operation cannot be performed on a file with a user-mapped section open
            # This is a Windows-specific file locking issue
            if error_code == 1224 or "user-mapped section" in str(e):
                logger.warning(f"    File lock detected (attempt {attempt + 1}/{max_retries}): {error_code}")

                if attempt < max_retries - 1:
                    # More retries available, try again
                    logger.debug(f"    Retrying {operation_name} in {retry_delay * (attempt + 1):.2f}s...")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    # No more retries
                    logger.error(f"    ✗ {operation_name} failed after {max_retries} attempts: {e}")
                    return False
            else:
                # Different OS error, don't retry
                logger.error(f"    ✗ {operation_name} failed: {e}")
                return False

        except Exception as e:
            logger.error(f"    ✗ {operation_name} failed with unexpected error: {e}")
            return False

    return False


def verify_safetensors_integrity(safetensor_path: Path, model: torch.nn.Module) -> bool:
    """
    Verify that the saved safetensors file can be loaded and matches the model structure.

    Args:
        safetensor_path: Path to the safetensors file
        model: The model to compare against

    Returns:
        True if integrity check passes, False otherwise
    """
    try:
        from safetensors.torch import load_file

        logger.debug("  Verifying safetensors file integrity...")
        loaded_state = load_file(safetensor_path)
        model_state = model.state_dict()

        # Check key count
        if len(loaded_state) != len(model_state):
            logger.error(f"  Key count mismatch: loaded={len(loaded_state)}, model={len(model_state)}")
            return False

        # Check all keys exist
        for key in model_state.keys():
            if key not in loaded_state:
                logger.error(f"  Missing key in safetensors: {key}")
                return False

        # Check tensor shapes
        for key in model_state.keys():
            if loaded_state[key].shape != model_state[key].shape:
                logger.error(f"  Shape mismatch for {key}: loaded={loaded_state[key].shape}, model={model_state[key].shape}")
                return False

        logger.debug("  ✓ Safetensors integrity check passed")
        return True

    except Exception as e:
        logger.error(f"  Safetensors integrity check failed: {e}")
        return False


# --- CALLBACK CLASSES FOR TRAINING SCRIPTS ---

class CompositeMetricCallback(Callback):
    """
    Callback that computes a composite metric combining Dice, Recall, and Precision.

    Uses 30% Dice + 50% Recall + 20% Precision to balance:
    - Recall prioritization (minimize false negatives)
    - Precision guardrail (minimize false positives)
    - Dice score (overall segmentation quality)

    This callback:
    - Computes composite_score = 0.3 * dice + 0.5 * recall + 0.2 * precision
    - Applies precision guardrail: models with precision < min_precision are penalized
    - Stores the composite score in recorder for use by other callbacks
    - Logs the composite score for monitoring

    Args:
        dice_weight: Weight for dice metric (default: 0.3)
        recall_weight: Weight for recall metric (default: 0.5)
        precision_weight: Weight for precision metric (default: 0.2)
        min_precision: Minimum precision threshold for guardrail (default: 0.5)
            Models with precision below this threshold receive a penalty to their score
    """

    order = 55  # Run AFTER Recorder (50) but BEFORE EarlyStoppingRecall (60) and SaveBest (70)

    def __init__(
        self,
        dice_weight: float = 0.3,
        recall_weight: float = 0.5,
        precision_weight: float = 0.2,
        min_precision: float = 0.5
    ):
        self.dice_weight = dice_weight
        self.recall_weight = recall_weight
        self.precision_weight = precision_weight
        self.min_precision = min_precision
        self._evaluation_mode = False  # Guard: prevent saves during evaluation
        store_attr()

    def after_epoch(self):
        """Called after epoch at the end of each epoch."""
        # CRITICAL: Skip during evaluation to prevent unnecessary metric computation
        if getattr(self, '_evaluation_mode', False):
            logger.debug(f"[CompositeMetricCallback] Evaluation mode ACTIVE - skipping")
            return
        
        # DIAGNOSTIC: Log recorder state
        logger.debug(f"[CompositeMetricCallback] after_epoch called at epoch {self.learn.epoch}")

        try:
            recorder = getattr(self.learn, 'recorder', None)
            if recorder is None:
                logger.debug(f"  No recorder found, returning early")
                return

            values = getattr(recorder, 'values', None)
            if values is None or len(values) == 0:
                logger.debug(f"  No values in recorder, returning early")
                return

            latest_values = values[-1]
            metric_names = getattr(recorder, 'metric_names', None)

            logger.debug(f"  metric_names = {metric_names}")
            logger.debug(f"  latest_values = {latest_values}")
            logger.debug(f"  len(latest_values) = {len(latest_values)}")

            if latest_values is None or len(latest_values) < 5:
                logger.debug(f"  ERROR: Not enough values in latest_values (need at least 5, got {len(latest_values)})")
                return

            # Get dice, recall, and precision values using metric_names for robustness
            dice_value = None
            recall_value = None
            precision_value = None

            if metric_names is not None:
                if 'dice_multi_strip' in metric_names:
                    dice_idx = metric_names.index('dice_multi_strip')
                    # Adjust index: metric_names includes 'epoch' at index 0, but values doesn't
                    values_idx = dice_idx - 1
                    logger.debug(f"  dice_multi_strip: metric_idx={dice_idx}, values_idx={values_idx}")
                    if values_idx >= 0 and values_idx < len(latest_values):
                        dice_value = latest_values[values_idx]
                    else:
                        logger.debug(f"  WARNING: values_idx {values_idx} out of bounds for dice_multi_strip")

                if 'recall_multi_strip' in metric_names:
                    recall_idx = metric_names.index('recall_multi_strip')
                    # Adjust index: metric_names includes 'epoch' at index 0, but values doesn't
                    values_idx = recall_idx - 1
                    logger.debug(f"  recall_multi_strip: metric_idx={recall_idx}, values_idx={values_idx}")
                    if values_idx >= 0 and values_idx < len(latest_values):
                        recall_value = latest_values[values_idx]
                    else:
                        logger.debug(f"  WARNING: values_idx {values_idx} out of bounds for recall_multi_strip")

                if 'precision_multi_strip' in metric_names:
                    precision_idx = metric_names.index('precision_multi_strip')
                    # Adjust index: metric_names includes 'epoch' at index 0, but values doesn't
                    values_idx = precision_idx - 1
                    logger.debug(f"  precision_multi_strip: metric_idx={precision_idx}, values_idx={values_idx}")
                    if values_idx >= 0 and values_idx < len(latest_values):
                        precision_value = latest_values[values_idx]
                    else:
                        logger.debug(f"  WARNING: values_idx {values_idx} out of bounds for precision_multi_strip")
            else:
                # Fallback: assume format [train_loss, valid_loss, dice_multi_strip, recall_multi_strip, precision_multi_strip, ...]
                logger.debug(f"  WARNING: metric_names is None, using hardcoded indices")
                dice_value = latest_values[2]
                recall_value = latest_values[3]
                precision_value = latest_values[4]
                logger.debug(f"  Using hardcoded indices: dice=[2], recall=[3], precision=[4]")

            if dice_value is None or recall_value is None or precision_value is None:
                logger.debug(f"  ERROR: Could not extract dice_value, recall_value, or precision_value")
                return

            logger.debug(f"  dice_value = {dice_value}, recall_value = {recall_value}, precision_value = {precision_value}")

            # Check if values are valid
            if not (isinstance(dice_value, (int, float)) and
                    isinstance(recall_value, (int, float)) and
                    isinstance(precision_value, (int, float))):
                logger.debug(f"  ERROR: dice_value, recall_value, or precision_value is not a number")
                return

            if np.isnan(dice_value) or np.isinf(dice_value) or \
               np.isnan(recall_value) or np.isinf(recall_value) or \
               np.isnan(precision_value) or np.isinf(precision_value):
                logger.debug(f"  ERROR: dice_value, recall_value, or precision_value is NaN or Inf")
                return

            # Compute composite score
            composite_score = (self.dice_weight * dice_value +
                            self.recall_weight * recall_value +
                            self.precision_weight * precision_value)

            # Apply precision guardrail: if precision is below threshold, apply penalty
            adjusted_score = composite_score
            precision_warning = None

            if precision_value < self.min_precision:
                # Soft penalty: reduce score proportionally to how far below threshold
                penalty_factor = precision_value / self.min_precision
                adjusted_score = composite_score * penalty_factor
                precision_warning = (
                    f"⚠️ PRECISION GUARDRAIL: Precision ({precision_value:.4f}) below threshold "
                    f"({self.min_precision:.2f}). Penalty applied: {penalty_factor:.4f}x"
                )
                logger.warning(f"[CompositeMetricCallback] {precision_warning}")

            logger.debug(f"  composite_score = {composite_score:.6f} (dice_weight={self.dice_weight}, recall_weight={self.recall_weight}, precision_weight={self.precision_weight})")
            if adjusted_score != composite_score:
                logger.debug(f"  adjusted_score = {adjusted_score:.6f} (penalty applied)")

            # CRITICAL FIX: Add composite_score to recorder values and metric_names
            # This is needed for other callbacks to see it and for it to be displayed
            logger.debug(f"  Adding composite_score to recorder...")

            # Add to metric_names if not already present
            if metric_names is not None and 'composite_score' not in metric_names:
                # Find where to insert (after precision_multi_strip, before 'time')
                if 'precision_multi_strip' in metric_names:
                    insert_idx = metric_names.index('precision_multi_strip') + 1
                elif 'recall_multi_strip' in metric_names:
                    insert_idx = metric_names.index('recall_multi_strip') + 1
                else:
                    insert_idx = len(metric_names) - 1  # Insert before 'time'
                metric_names.insert(insert_idx, 'composite_score')
                logger.debug(f"  Added 'composite_score' to metric_names at index {insert_idx}")
                logger.debug(f"  Updated metric_names = {metric_names}")

            # Add to values array
            if metric_names is not None and 'composite_score' in metric_names:
                composite_idx = metric_names.index('composite_score')
                # Adjust index for values array (skip 'epoch' at index 0)
                values_idx = composite_idx - 1
                logger.debug(f"  composite_score in metric_names at index {composite_idx}, values_idx={values_idx}")

                # Check if we need to extend the values array
                if values_idx >= len(latest_values):
                    # Pad with None values if needed
                    while len(latest_values) < values_idx + 1:
                        latest_values.append(None)
                    logger.debug(f"  Extended latest_values to length {len(latest_values)}")

                # Set the composite score (use adjusted score if penalty was applied)
                latest_values[values_idx] = adjusted_score
                logger.debug(f"  Set latest_values[{values_idx}] = {adjusted_score}")
                logger.debug(f"  Updated latest_values = {latest_values}")

            # Log the composite score with precision warning if applicable
            score_msg = f"  📊 Composite Score: {adjusted_score:.6f} (Dice: {dice_value:.6f}, Recall: {recall_value:.6f}, Precision: {precision_value:.6f})"
            if precision_warning:
                score_msg = f"{score_msg}\n  {precision_warning}"
            logger.info(score_msg)

        except Exception as e:
            logger.error(f"Error in CompositeMetricCallback at epoch {self.learn.epoch}: {e}")
            import traceback
            traceback.print_exc()


class ReduceLROnPlateauCustom(Callback):
    """
    Custom ReduceLROnPlateau callback that properly monitors custom metrics.

    This callback monitors the composite_score metric (30% Dice + 70% Recall)
    and reduces the learning rate when the metric stops improving. Unlike fastai's
    built-in ReduceLROnPlateau, this version correctly handles custom metrics
    and prioritizes recall to minimize false negatives.

    Args:
        monitor: Metric name to monitor (default: 'composite_score')
        factor: Factor to reduce learning rate by (default: 0.1)
        patience: Number of epochs to wait before reducing LR (default: 3)
        min_lr: Minimum learning rate (default: 1e-7)
        mode: 'max' for metrics where higher is better, 'min' for lower is better (default: 'max')
    """
    order = 70 # Run AFTER CompositeMetricCallback

    def __init__(
        self,
        monitor: str = 'composite_score',
        factor: float = 0.1,
        patience: int = 3,
        min_lr: float = 1e-7,
        mode: str = 'max'
    ):
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.mode = mode
        self.wait = 0
        self.best_score = None
        self.lr_reduced_count = 0
        self._evaluation_mode = False  # Guard: prevent saves during evaluation
        store_attr()

        if mode == 'max':
            self.is_better = lambda current, best: current > best
        else:
            self.is_better = lambda current, best: current < best

        logger.debug(f"[ReduceLROnPlateauCustom] Initialized with monitor='{monitor}', mode='{mode}'")

    def after_epoch(self):
        """Called after epoch at the end of each epoch."""
        # CRITICAL: Skip during evaluation to prevent LR adjustments during get_preds()
        if getattr(self, '_evaluation_mode', False):
            logger.debug(f"[ReduceLROnPlateauCustom] Evaluation mode ACTIVE - skipping LR reduction check")
            return
        
        # DIAGNOSTIC: Log recorder state
        logger.debug(f"[ReduceLROnPlateauCustom] after_epoch called at epoch {self.learn.epoch}")
        logger.debug(f"  self.monitor = '{self.monitor}'")

        try:
            # Get the current metric value
            recorder = getattr(self.learn, 'recorder', None)
            if recorder is None:
                logger.debug(f"  No recorder found, returning early")
                return

            values = getattr(recorder, 'values', None)
            if values is None or len(values) == 0:
                logger.debug(f"  No values in recorder, returning early")
                return

            latest_values = values[-1]
            if latest_values is None or len(latest_values) == 0:
                logger.debug(f"  No latest_values, returning early")
                return

            # Find the metric index
            metric_index = -1
            values_index = -1

            # Try to get metric_names safely (FIXED: was metrics_names typo)
            metric_names = getattr(recorder, 'metric_names', None)

            logger.debug(f"  metric_names = {metric_names}")
            logger.debug(f"  latest_values = {latest_values}")

            if metric_names is not None and self.monitor in metric_names:
                metric_index = metric_names.index(self.monitor)
                # Adjust index: metric_names includes 'epoch' at index 0, but values doesn't
                values_index = metric_index - 1
                logger.debug(f"  Found '{self.monitor}' at metric_index={metric_index}, values_index={values_index}")
            else:
                # Fallback: assume dice_multi_strip is at index 2
                if len(latest_values) >= 3:
                    metric_index = 2
                    values_index = 2  # Already adjusted for values array
                    logger.debug(f"  WARNING: '{self.monitor}' not found in metric_names, using fallback index 2")
                else:
                    logger.debug(f"  ERROR: '{self.monitor}' not found and not enough values for fallback")
                    return

            if values_index < 0 or values_index >= len(latest_values):
                logger.debug(f"  ERROR: values_index {values_index} is out of bounds for latest_values (len={len(latest_values)})")
                return

            current_score = latest_values[values_index]
            logger.debug(f"  current_score = {current_score}")

            # Check if the score is valid
            if not isinstance(current_score, (int, float)):
                return
            if np.isnan(current_score) or np.isinf(current_score):
                return

            # Check if this is the best score so far
            if self.best_score is None or self.is_better(current_score, self.best_score):
                self.best_score = current_score
                self.wait = 0
            else:
                self.wait += 1
                logger.info(f"  📉 No improvement for {self.wait} epoch(s) (best {self.monitor}={self.best_score:.6f}, current={current_score:.6f})")

                # Check if we should reduce the learning rate
                if self.wait >= self.patience:
                    old_lr = self.opt.hypers[-1]['lr']
                    new_lr = max(old_lr * self.factor, self.min_lr)

                    if new_lr < old_lr:
                        self.learn.opt.set_hyper('lr', new_lr)
                        self.lr_reduced_count += 1
                        self.wait = 0
                        logger.info(f"  ⚠️ Learning rate reduced from {old_lr:.2e} to {new_lr:.2e} (reduction #{self.lr_reduced_count})")
                    else:
                        logger.info(f"  ℹ️ Learning rate already at minimum ({old_lr:.2e}), not reducing further")

        except Exception as e:
            logger.error(f"Error in ReduceLROnPlateauCustom at epoch {self.learn.epoch}: {e}")
            import traceback
            traceback.print_exc()


class SaveBestAndLatestModel(Callback):
    """
    Custom callback to save the best model (based on composite_score metric)
    and the latest model after each epoch.

    This callback monitors the composite_score metric (30% Dice + 70% Recall)
    and saves the model when a better score is achieved. It also saves the
    latest model at the end of training. The composite score prioritizes
    recall to minimize false negatives on critical classes.

    FIXED: Includes Windows file locking robustness with proper cleanup.
    """
    order = 70 # Run AFTER CompositeMetricCallback

    def __init__(
        self,
        monitor: str = 'composite_score',
        model_save_path: Path = None,
        learner_save_name: str = None,
        save_format: str = 'safetensors'
    ):
        """
        Args:
            monitor: Metric name to monitor for best model (default: composite_score)
            model_save_path: Directory path to save models
            learner_save_name: Base name for saving fastai learner
            save_format: Format to save models ('safetensors' or 'pth')
        """
        self.monitor = monitor
        self.model_save_path = model_save_path
        self.learner_save_name = learner_save_name
        self.save_format = save_format
        self.best_score = -float('inf')
        self.best_epoch = 0
        self.best_model_state = None
        self._already_saved = False  # Guard: prevent double-save after training
        self._evaluation_mode = False  # Guard: prevent saves during evaluation
        self._init_id = id(self)  # Track callback instance
        logger.debug(f"🔧 SaveBestAndLatestModel callback initialized (instance id={self._init_id})")
        logger.debug(f"🔧 Monitoring metric: '{self.monitor}'")
        store_attr()

    def after_epoch(self):
        """Called after epoch at the end of each epoch."""
        # CRITICAL: Skip during evaluation to prevent file locks during get_preds()
        if getattr(self, '_evaluation_mode', False):
            logger.debug(f"[SaveBestAndLatestModel] Evaluation mode ACTIVE - skipping save")
            return
        
        # DIAGNOSTIC: Log recorder state
        logger.debug(f"[SaveBestAndLatestModel] after_epoch called at epoch {self.learn.epoch}")
        logger.debug(f"  self.monitor = '{self.monitor}'")

        # Completely defensive approach to get the current metric value
        try:
            # Method 1: Try to get recorder safely using getattr
            recorder = getattr(self.learn, 'recorder', None)

            if recorder is None:
                logger.debug(f"No recorder found at epoch {self.learn.epoch}")
                return

            # Method 2: Try to get values from recorder
            values = getattr(recorder, 'values', None)

            if values is None or len(values) == 0:
                logger.debug(f"No values in recorder at epoch {self.learn.epoch}")
                return

            # Get the latest recorded values
            latest_values = values[-1]

            if latest_values is None or len(latest_values) == 0:
                logger.debug(f"No latest_values at epoch {self.learn.epoch}")
                return

            # Method 3: Try to find the metric index
            metric_index = -1
            values_index = -1

            # Try to get metric_names safely (FIXED: was metrics_names typo)
            metric_names = getattr(recorder, 'metric_names', None)

            logger.debug(f"  metric_names = {metric_names}")
            logger.debug(f"  latest_values = {latest_values}")

            if metric_names is not None and self.monitor in metric_names:
                metric_index = metric_names.index(self.monitor)
                # Adjust index: metric_names includes 'epoch' at index 0, but values doesn't
                values_index = metric_index - 1
                logger.debug(f"  Found '{self.monitor}' at metric_index={metric_index}, values_index={values_index}")
            else:
                # Fallback: assume dice_multi_strip is at index 2
                # Format is usually: [train_loss, valid_loss, metric1, metric2, ...]
                if len(latest_values) >= 3:
                    metric_index = 2
                    values_index = 2  # Already adjusted for values array
                    logger.debug(f"Using index 2 for metric (assumed {self.monitor})")
                    logger.debug(f"  WARNING: '{self.monitor}' not found in metric_names, using fallback index 2")
                else:
                    logger.debug(f"Not enough values in latest_values: {len(latest_values)}")
                    logger.debug(f"  ERROR: '{self.monitor}' not found and not enough values for fallback")
                    return

            # Check if values_index is valid
            if values_index < 0 or values_index >= len(latest_values):
                logger.debug(f"Invalid values_index {values_index} for latest_values length {len(latest_values)}")
                logger.debug(f"  ERROR: values_index {values_index} is out of bounds for latest_values (len={len(latest_values)})")
                return

            # Get the current metric value
            current_score = latest_values[values_index]
            logger.debug(f"  current_score = {current_score}")

            # Check if the score is valid (not NaN or inf)
            if not isinstance(current_score, (int, float)):
                logger.debug(f"Metric is not a number: {type(current_score)}")
                return

            if np.isnan(current_score) or np.isinf(current_score):
                logger.debug(f"Invalid metric value at epoch {self.learn.epoch}: {current_score}")
                return

            # Update best model if current score is better
            if current_score > self.best_score:
                old_best_score = self.best_score
                old_best_epoch = self.best_epoch
                
                # Determine current phase for logging
                phase = "frozen" if self.learn.opt.frozen_idx > 0 else "unfrozen"
                
                # CRITICAL: Save model state BEFORE updating best_score/epoch
                # This ensures we capture the correct model state for the new best score
                self.best_model_state = {k: v.cpu().clone() for k, v in self.learn.model.state_dict().items()}
                # Now update the tracking variables
                self.best_score = current_score
                self.best_epoch = self.learn.epoch
                logger.info(f"  📈 New best model saved at epoch {self.learn.epoch} (phase={phase}) with {self.monitor}={current_score:.6f}")
                logger.debug(f"  📊 Previous best: epoch {old_best_epoch}, {self.monitor}={old_best_score:.6f}")
                logger.debug(f"  🔍 Callback instance id={id(self)}, _already_saved={getattr(self, '_already_saved', False)}")
                logger.debug(f"  🔧 Model state captured: {len(self.best_model_state)} parameters")
                
                # Optional: Save temporary checkpoint for crash-safety
                # This prevents loss of best weights if training crashes mid-run
                if self.learner_save_name:
                    try:
                        self.learn.save(f"{self.learner_save_name}_best_tmp")
                        logger.debug(f"  💾 Saved temporary checkpoint: {self.learner_save_name}_best_tmp")
                    except Exception as e:
                        logger.debug(f"  ⚠️ Failed to save temporary checkpoint: {e}")

        except Exception as e:
            logger.error(f"Error in after_epoch at epoch {self.learn.epoch}: {e}")
            import traceback
            traceback.print_exc()
            # Don't raise the exception to allow training to continue

    def after_fit(self):
        """Called at the end of training."""
        # Guard: Skip if already saved OR in evaluation mode
        already_saved = getattr(self, '_already_saved', False)
        eval_mode = getattr(self, '_evaluation_mode', False)

        logger.debug(f"🔧 after_fit() called (callback id={id(self)}, _already_saved={already_saved}, _evaluation_mode={eval_mode})")
        logger.debug(f"🔧 Current best model state: epoch={self.best_epoch}, {self.monitor}={self.best_score:.6f}")

        if already_saved or eval_mode:
            if eval_mode:
                logger.debug("Evaluation mode - skipping save to prevent file locks")
            else:
                logger.debug("Models already saved - skipping duplicate save")
            return

        self._already_saved = True

        # Save the latest model
        logger.info("\nSaving models...")

        try:
            # Save latest fastai learner
            self.learn.save(f"{self.learner_save_name}_latest")
            logger.debug(f"  ✓ Saved latest Fastai learner: {self.learner_save_name}_latest")

            # Convert to CPU and float32 for saving
            model_cpu = self.learn.model.to("cpu").float()

            # Release GPU memory but keep the model reference in learner
            # (we need it for loading best model state later)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            gc.collect()

            # Small delay to allow Windows to release file locks
            time.sleep(0.2)

            # Save latest model in safetensors/pytorch format
            if self.save_format == 'safetensors':
                from safetensors.torch import save_file

                latest_path = self.model_save_path / f"{self.learner_save_name}_latest.safetensors"

                # Use safe_file_save with retry logic
                success = safe_file_save(
                    save_file,
                    model_cpu.state_dict(),
                    latest_path,
                    operation_name=f"Save latest model (safetensors): {latest_path.name}",
                    max_retries=MAX_SAVE_RETRIES,
                    retry_delay=SAVE_RETRY_DELAY
                )

                if success:
                    logger.debug(f"  ✓ Saved latest model (safetensors): {latest_path.name}")
                else:
                    logger.error(f"  ✗ Failed to save latest safetensors after retries")

                # Increased delay between saves
                time.sleep(0.3)

                # Save best model in safetensors format
                if self.best_model_state is not None:
                    logger.debug(f"🔧 Saving best model state: epoch={self.best_epoch}, {self.monitor}={self.best_score:.6f}")
                    best_path = self.model_save_path / f"{self.learner_save_name}_best.safetensors"

                    # Clear any references to prevent file locking
                    gc.collect()
                    time.sleep(0.2)

                    success = safe_file_save(
                        save_file,
                        self.best_model_state,
                        best_path,
                        operation_name=f"Save best model (safetensors): {best_path.name}",
                        max_retries=MAX_SAVE_RETRIES,
                        retry_delay=SAVE_RETRY_DELAY
                    )

                    if success:
                        logger.debug(f"  ✓ Saved best model (safetensors): {best_path.name}")
                        logger.info(f"  Best model from epoch {self.best_epoch} with {self.monitor}={self.best_score:.6f}")

                        # Also save best fastai learner
                        self.learn.model.load_state_dict(self.best_model_state)
                        self.learn.save(f"{self.learner_save_name}_best")
                        logger.debug(f"  ✓ Saved best Fastai learner: {self.learner_save_name}_best")

                        # Restore the latest model state
                        self.learn.model.load_state_dict(model_cpu.state_dict())
                    else:
                        logger.error(f"  ✗ Failed to save best safetensors after retries")
                        logger.warning(f"  ⚠ Best model not saved in safetensors format, using latest model as fallback")
                        # Fall back to latest model
                        self.learn.save(f"{self.learner_save_name}_best")
                        logger.debug(f"  ✓ Saved latest as best Fastai learner (fallback): {self.learner_save_name}_best")
                else:
                    logger.warning(f"  ⚠ No best model state saved (metric tracking may have failed)")
                    logger.debug(f"  🔧 Callback state: best_epoch={self.best_epoch}, best_score={self.best_score:.6f}, best_model_state is None={self.best_model_state is None}")
                    # Save latest as best if no best was found
                    best_path = self.model_save_path / f"{self.learner_save_name}_best.safetensors"

                    gc.collect()
                    time.sleep(0.2)

                    success = safe_file_save(
                        save_file,
                        model_cpu.state_dict(),
                        best_path,
                        operation_name=f"Save latest as best (safetensors): {best_path.name}",
                        max_retries=MAX_SAVE_RETRIES,
                        retry_delay=SAVE_RETRY_DELAY
                    )

                    if success:
                        logger.debug(f"  ✓ Saved latest model as best (safetensors): {best_path.name}")

                    self.learn.save(f"{self.learner_save_name}_best")
                    logger.debug(f"  ✓ Saved latest as best Fastai learner: {self.learner_save_name}_best")
            else:
                # Save in pytorch format
                latest_path = self.model_save_path / f"{self.learner_save_name}_latest_state.pth"

                success = safe_file_save(
                    torch.save,
                    model_cpu.state_dict(),
                    latest_path,
                    operation_name=f"Save latest model (pytorch): {latest_path.name}",
                    max_retries=MAX_SAVE_RETRIES,
                    retry_delay=SAVE_RETRY_DELAY
                )

                if success:
                    logger.debug(f"  ✓ Saved latest model (pytorch): {latest_path.name}")

                time.sleep(0.2)

                if self.best_model_state is not None:
                    best_path = self.model_save_path / f"{self.learner_save_name}_best_state.pth"

                    gc.collect()
                    time.sleep(0.2)

                    success = safe_file_save(
                        torch.save,
                        self.best_model_state,
                        best_path,
                        operation_name=f"Save best model (pytorch): {best_path.name}",
                        max_retries=MAX_SAVE_RETRIES,
                        retry_delay=SAVE_RETRY_DELAY
                    )

                    if success:
                        logger.debug(f"  ✓ Saved best model (pytorch): {best_path.name}")
                        logger.info(f"  Best model from epoch {self.best_epoch} with {self.monitor}={self.best_score:.6f}")

                        # Also save best fastai learner
                        self.learn.model.load_state_dict(self.best_model_state)
                        self.learn.save(f"{self.learner_save_name}_best")
                        logger.debug(f"  ✓ Saved best Fastai learner: {self.learner_save_name}_best")

                        # Restore the latest model state
                        self.learn.model.load_state_dict(model_cpu.state_dict())
                    else:
                        logger.error(f"  ✗ Failed to save best pytorch after retries")
                        # Save latest as best if no best was found
                        best_path = self.model_save_path / f"{self.learner_save_name}_best_state.pth"

                        gc.collect()
                        time.sleep(0.2)

                        safe_file_save(
                            torch.save,
                            model_cpu.state_dict(),
                            best_path,
                            operation_name=f"Save latest as best (pytorch): {best_path.name}",
                            max_retries=MAX_SAVE_RETRIES,
                            retry_delay=SAVE_RETRY_DELAY
                        )

                        self.learn.save(f"{self.learner_save_name}_best")
                        logger.debug(f"  ✓ Saved latest as best Fastai learner: {self.learner_save_name}_best")
                else:
                    logger.warning(f"  ⚠ No best model state saved (metric tracking may have failed)")
                    # Save latest as best if no best was found
                    best_path = self.model_save_path / f"{self.learner_save_name}_best_state.pth"

                    gc.collect()
                    time.sleep(0.2)

                    safe_file_save(
                        torch.save,
                        model_cpu.state_dict(),
                        best_path,
                        operation_name=f"Save latest as best (pytorch): {best_path.name}",
                        max_retries=MAX_SAVE_RETRIES,
                        retry_delay=SAVE_RETRY_DELAY
                    )

                    self.learn.save(f"{self.learner_save_name}_best")
                    logger.debug(f"  ✓ Saved latest as best Fastai learner: {self.learner_save_name}_best")

        except Exception as e:
            logger.error(f"Error in after_fit: {e}")
            import traceback
            traceback.print_exc()
            # Don't raise exception to allow training to complete