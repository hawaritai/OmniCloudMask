import sys
import os
from pathlib import Path
import numpy as np
import torch
import cv2
import rasterio as rio
from functools import partial
import timm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import logging
import json
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, precision_recall_curve, average_precision_score

from custom_model_utils import build_custom_model, load_custom_weights
from omnicloudmask.model_utils import compile_torch_model, load_model_from_weights

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- SETUP PATHS ---
current_script_dir = Path(__file__).parent.resolve()
training_dir = current_script_dir.parent
project_root = training_dir.parent

# Add paths for imports
if str(training_dir) not in sys.path:
    sys.path.append(str(training_dir))
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
if str(current_script_dir) not in sys.path:
    sys.path.append(str(current_script_dir))

# Import third-party modules
try:
    from thirdparty.NIRGAN.create_NIR import get_NIR
except ImportError as e:
    logger.warning(f"Import Error: {e}")
    # Try alternate path for NIRGAN
    try:
        sys.path.append(str(current_script_dir / "thirdparty"))
        from NIRGAN.create_NIR import get_NIR
    except ImportError as e2:
        logger.critical(f"Critical Import Error: {e2}")
        sys.exit(1)

# --- LOCAL CONFIG IMPORT ---
try:
    if str(current_script_dir) not in sys.path:
        sys.path.append(str(current_script_dir))
    from local_config import (
        TRAIN_DATA_DIR,
        BAND_ORDER,
        USE_DUAL_RES_METHOD,
        CUSTOM_MODEL_VERSION
    )
    
    # Try to import comparison specific configs if they exist, otherwise use defaults
    try:
        from local_config import (
            COMPARISON_QUICK_TEST,
            COMPARISON_QUICK_TEST_SAMPLES,
            COMPARISON_BATCH_SIZE,
            COMPARISON_USE_BF16,
            MODEL_CONFIG,
            INFERENCE_CONFIG
        )
    except ImportError:
        COMPARISON_QUICK_TEST = True
        COMPARISON_QUICK_TEST_SAMPLES = 50
        COMPARISON_BATCH_SIZE = 8
        COMPARISON_USE_BF16 = True
        MODEL_CONFIG = {
            "v4": {"model_library": "smp"},
            "v3": {"model_library": "fastai"}
        }
        INFERENCE_CONFIG = {
            "compile_models": False,
            "compile_mode": "default",
            "inference_dtype": "float32"
        }
        
except ImportError:
    logger.critical("CRITICAL: local_config.py not found. Please create 'training/scripts/local_config.py'.")
    sys.exit(1)

# --- CONSTANTS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATCH_SIZE = (509, 509)
NUM_CHANNELS = 3
CLASS_NAMES = ['Clear', 'Thick Cloud', 'Thin Cloud', 'Cloud Shadow']
GENERATE_SYNTHETIC_NIR = False # Set to False for comparison script as we assume processed data

# --- MODEL CONFIGURATIONS ---
# Define models to compare here
model_configs = []

# Auto-discover models from models directory (Optional)
models_dir = project_root / "ckpts"
if models_dir.exists():
    for model_file in models_dir.glob("*_state.safetensors"):
        # Skip if it's already in the list
        if any(m['path'].name == model_file.name for m in model_configs):
            continue
            
        model_name = model_file.stem.replace("_state", "").replace("PM_model_", "")
        
        # Infer library
        if "smp" in model_name.lower():
            lib = "smp"
        else:
            lib = "fastai"

        # Try to infer model type from name or default
        if "regnety_004" in model_name.lower():
            model_type = 'tu-regnety_004' if lib == 'smp' else 'regnety_004.pycls_in1k'
        elif "edgenext_small" in model_name.lower():
            model_type = 'tu-edgenext_small' if lib == 'smp' else 'edgenext_small.usi_in1k'
        elif "convnextv2_nano" in model_name.lower():
            model_type = 'tu-convnextv2_nano' if lib == 'smp' else 'convnextv2_nano.fcmae_ft_in1k'
        else:
            # Fallback default
            model_type = 'tu-regnety_004' if lib == 'smp' else 'regnety_004.pycls_in1k'
        
        model_configs.append({
            'name': model_name,
            'path': model_file,
            'model_type': model_type,
            'model_library': lib
        })

    model_configs.append(
        {
            'name': 'Tuned OCM 7.43_v4',
            'path': Path(r"D:\Projects\QI47\2025_Projects\Image_QC_GUI\2_Repo\OmniCloudMask\models\PM_model_OCM_7.43_R_G_NIR_test_4_regnety_004.pycls_in1k_PT_state.safetensors"),
            'model_type': 'regnety_004.pycls_in1k',
            'model_library': 'fastai' # Assuming this specific file is legacy based on name. Change to 'smp' if it's a V4 training.
        }
    )

class ModelComparator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = self.output_dir / "metrics"
        self.plots_dir = self.output_dir / "plots"
        self.metrics_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        
        # Setup logging to file
        file_handler = logging.FileHandler(self.output_dir / "model_comparison.log")
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

    def load_model(self, model_config):
        path = model_config['path']
        model_type = model_config['model_type']
        model_library = model_config.get('model_library', 'smp') # Default to smp for new
        
        # Handle relative paths
        if not path.exists():
            # Try finding it relative to project root
            path = project_root / path
            
        if not path.exists():
            logger.error(f"Model file not found: {path}, skipping...")
            return None

        logger.info(f"Loading model: {model_config['name']} from {path} ({model_library})")
        
        try:
            # Create model architecture using build_model
            model = build_custom_model(
                model_name=model_type,
                model_library=model_library,
                in_chans=NUM_CHANNELS,
                n_out=len(CLASS_NAMES)
            )
            
            # Load weights
            load_custom_weights(model, path, device=DEVICE, strict=False)
            
            if COMPARISON_USE_BF16 and DEVICE.type == 'cuda':
                model = model.bfloat16()
            
            # Optional Compilation
            if INFERENCE_CONFIG.get("compile_models", False):
                 logger.info("Compiling model...")
                 model = compile_torch_model(
                     model, 
                     patch_size=MODEL_PATCH_SIZE[0],
                     batch_size=COMPARISON_BATCH_SIZE, 
                     dtype=torch.bfloat16 if COMPARISON_USE_BF16 else torch.float32,
                     device=DEVICE,
                     compile_mode=INFERENCE_CONFIG.get("compile_mode", "default")
                 )

            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_config['name']}: {e}")
            return None

    def get_validation_data(self):
        """
        Scan TRAIN_DATA_DIR/validation for images and labels.
        Supports both structure types:
        1. Separate folders (images/, labels/ or masks/)
        2. Colocated files (_image.tif, _label.tif)
        """
        val_dir = TRAIN_DATA_DIR / "validation"
        if not val_dir.exists():
            logger.error(f"Validation directory not found: {val_dir}")
            return [], []

        image_files = []
        label_files = []

        # Strategy 1: Colocated _image.tif and _label.tif
        candidates = list(val_dir.glob("*_image.tif"))
        if candidates:
            logger.info(f"Detected colocated dataset structure in {val_dir}")
            for img_path in candidates:
                lbl_path = val_dir / img_path.name.replace("_image.tif", "_label.tif")
                if lbl_path.exists():
                    image_files.append(img_path)
                    label_files.append(lbl_path)
        
        # Strategy 2: Standard folder structure or just .tif files
        if not image_files:
            logger.info("Checking for standard dataset structure or fallback...")
            # Try to find images folder inside validation
            img_dir = val_dir
            if (val_dir / "images").exists():
                img_dir = val_dir / "images"
            
            # Find labels folder
            lbl_dir = val_dir
            if (val_dir / "masks").exists():
                lbl_dir = val_dir / "masks"
            elif (val_dir / "labels").exists():
                lbl_dir = val_dir / "labels"
            
            # Get all TIFs that are not labels
            all_tifs = list(img_dir.glob("*.tif"))
            possible_images = [p for p in all_tifs if "_label.tif" not in p.name and "mask" not in p.name.lower()]
            
            for img_path in possible_images:
                # Try to find matching label
                lbl_candidates = [
                    lbl_dir / img_path.name,
                    lbl_dir / f"{img_path.stem}_label{img_path.suffix}",
                    lbl_dir / f"label_{img_path.name}",
                    lbl_dir / img_path.name.replace("image", "label")
                ]
                
                for lbl_path in lbl_candidates:
                    if lbl_path.exists() and lbl_path != img_path:
                        image_files.append(img_path)
                        label_files.append(lbl_path)
                        break

        logger.info(f"Found {len(image_files)} validation image-label pairs")
        
        if COMPARISON_QUICK_TEST and len(image_files) > COMPARISON_QUICK_TEST_SAMPLES:
            logger.info(f"Quick test mode: limiting to {COMPARISON_QUICK_TEST_SAMPLES} samples")
            indices = np.random.choice(len(image_files), COMPARISON_QUICK_TEST_SAMPLES, replace=False)
            image_files = [image_files[i] for i in indices]
            label_files = [label_files[i] for i in indices]
            
        return image_files, label_files

    def preprocess_batch(self, image_paths, label_paths):
        images = []
        labels = []
        
        for img_path, lbl_path in zip(image_paths, label_paths):
            try:
                # Load Image
                with rio.open(img_path) as src:
                    raw_bands = src.read(BAND_ORDER) # (C, H, W)
                
                img_hwc = np.transpose(raw_bands, (1, 2, 0))
                
                # Resize
                img_final = cv2.resize(img_hwc, MODEL_PATCH_SIZE, interpolation=cv2.INTER_AREA)
                
                # Stack bands
                if GENERATE_SYNTHETIC_NIR:
                     # NIRGAN logic skipped for brevity/speed as per user request to keep simple
                     # Assuming inputs are already R,G,NIR or don't need GAN for this eval
                     pass
                
                rgn_stack = np.transpose(img_final, (2, 0, 1)).astype(np.float32)
                
                # Normalize (Z-Score)
                tensor_img = torch.from_numpy(rgn_stack).float()
                mean = tensor_img.mean(dim=(1, 2), keepdim=True)
                std = tensor_img.std(dim=(1, 2), keepdim=True) + 1e-6
                tensor_img = (tensor_img - mean) / std
                
                # Load Label
                with rio.open(lbl_path) as src_lbl:
                    lbl = src_lbl.read(1)
                
                # Resize label
                lbl_resized = cv2.resize(lbl, MODEL_PATCH_SIZE, interpolation=cv2.INTER_NEAREST)
                tensor_lbl = torch.from_numpy(lbl_resized).long()
                
                images.append(tensor_img)
                labels.append(tensor_lbl)
                
            except Exception as e:
                logger.warning(f"Error reading pair {img_path.name}: {e}")
                continue
                
        if not images:
            return None, None
            
        return torch.stack(images), torch.stack(labels)

    def evaluate_model(self, model, image_files, label_files):
        all_preds = []
        all_targets = []
        all_probs = []
        
        # Batch processing
        num_batches = (len(image_files) + COMPARISON_BATCH_SIZE - 1) // COMPARISON_BATCH_SIZE
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * COMPARISON_BATCH_SIZE
                end_idx = min((i + 1) * COMPARISON_BATCH_SIZE, len(image_files))
                
                batch_imgs = image_files[start_idx:end_idx]
                batch_lbls = label_files[start_idx:end_idx]
                
                xb, yb = self.preprocess_batch(batch_imgs, batch_lbls)
                
                if xb is None:
                    continue
                
                xb = xb.to(DEVICE)
                if COMPARISON_USE_BF16 and DEVICE.type == 'cuda':
                    xb = xb.bfloat16()
                
                out = model(xb)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                preds = torch.argmax(out, dim=1).cpu().numpy()
                
                all_probs.append(probs)
                all_preds.append(preds)
                all_targets.append(yb.numpy())
                
                if (i + 1) % 5 == 0:
                    logger.info(f"Processed {end_idx}/{len(image_files)} images")

        return np.concatenate(all_preds), np.concatenate(all_targets), np.concatenate(all_probs)

    def compute_metrics(self, y_true, y_pred, y_probs):
        # Flatten
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Filter valid classes
        valid_mask = (y_true_flat >= 0) & (y_true_flat < len(CLASS_NAMES))
        y_true_valid = y_true_flat[valid_mask]
        y_pred_valid = y_pred_flat[valid_mask]
        
        metrics = {}
        
        # Overall Accuracy
        metrics['Accuracy'] = accuracy_score(y_true_valid, y_pred_valid)
        
        # Per-class metrics
        precision = precision_score(y_true_valid, y_pred_valid, average=None, labels=range(len(CLASS_NAMES)), zero_division=0)
        recall = recall_score(y_true_valid, y_pred_valid, average=None, labels=range(len(CLASS_NAMES)), zero_division=0)
        f1 = f1_score(y_true_valid, y_pred_valid, average=None, labels=range(len(CLASS_NAMES)), zero_division=0)
        iou = jaccard_score(y_true_valid, y_pred_valid, average=None, labels=range(len(CLASS_NAMES)), zero_division=0)
        
        metrics['Mean_IoU'] = np.mean(iou)
        metrics['Mean_F1'] = np.mean(f1)
        metrics['Weighted_Precision'] = precision_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0)
        metrics['Weighted_Recall'] = recall_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0)
        
        # Per class dicts
        metrics['per_class'] = {}
        
        # PR Curve data (requires probabilities)
        # Flatten probabilities: (N, C)
        # Reshape probs: (B, C, H, W) -> (B*H*W, C)
        y_probs_flat = y_probs.transpose(0, 2, 3, 1).reshape(-1, len(CLASS_NAMES))
        y_probs_valid = y_probs_flat[valid_mask]
        
        metrics['pr_data'] = {}
        avg_aps = []
        
        for i, name in enumerate(CLASS_NAMES):
            cls_metrics = {
                'IoU': float(iou[i]),
                'F1': float(f1[i]),
                'Precision': float(precision[i]),
                'Recall': float(recall[i])
            }
            
            # PR Curve for this class
            # Binarize true labels for this class
            y_true_cls = (y_true_valid == i).astype(int)
            y_score_cls = y_probs_valid[:, i]
            
            prec, rec, _ = precision_recall_curve(y_true_cls, y_score_cls)
            ap = average_precision_score(y_true_cls, y_score_cls)
            
            cls_metrics['AP'] = float(ap)
            avg_aps.append(ap)
            
            # Downsample PR curve for saving to JSON (too large otherwise)
            # Take 100 points
            if len(prec) > 100:
                indices = np.linspace(0, len(prec)-1, 100, dtype=int)
                metrics['pr_data'][name] = {
                    'precision': prec[indices].tolist(),
                    'recall': rec[indices].tolist(),
                    'ap': float(ap)
                }
            else:
                metrics['pr_data'][name] = {
                    'precision': prec.tolist(),
                    'recall': rec.tolist(),
                    'ap': float(ap)
                }
                
            metrics['per_class'][name] = cls_metrics
            
        metrics['Mean_AP'] = np.mean(avg_aps)
        
        # Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true_valid, y_pred_valid, labels=range(len(CLASS_NAMES)), normalize='true')
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics

    def plot_comparisons(self, all_metrics):
        # 1. Metrics Bar Chart
        df_rows = []
        for model_name, m in all_metrics.items():
            df_rows.append({
                'Model': model_name,
                'Accuracy': m['Accuracy'],
                'Mean IoU': m['Mean_IoU'],
                'Mean F1': m['Mean_F1'],
                'Mean AP': m['Mean_AP']
            })
        
        df = pd.DataFrame(df_rows)
        df_melted = df.melt('Model', var_name='Metric', value_name='Score')
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_melted, x='Metric', y='Score', hue='Model')
        plt.title('Overall Model Comparison')
        plt.ylim(0, 1.0)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'metrics_comparison.png')
        plt.close()
        
        # 2. Per-Class F1 Score Comparison
        class_rows = []
        for model_name, m in all_metrics.items():
            for cls_name in CLASS_NAMES:
                class_rows.append({
                    'Model': model_name,
                    'Class': cls_name,
                    'F1 Score': m['per_class'][cls_name]['F1'],
                    'IoU': m['per_class'][cls_name]['IoU']
                })
        
        df_cls = pd.DataFrame(class_rows)
        
        plt.figure(figsize=(14, 6))
        sns.barplot(data=df_cls, x='Class', y='F1 Score', hue='Model')
        plt.title('Per-Class F1 Score Comparison')
        plt.ylim(0, 1.0)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'per_class_f1.png')
        plt.close()
        
        plt.figure(figsize=(14, 6))
        sns.barplot(data=df_cls, x='Class', y='IoU', hue='Model')
        plt.title('Per-Class IoU Comparison')
        plt.ylim(0, 1.0)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'per_class_iou.png')
        plt.close()

        # 3. Confusion Matrices
        num_models = len(all_metrics)
        fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))
        if num_models == 1: axes = [axes]
        
        for ax, (model_name, m) in zip(axes, all_metrics.items()):
            cm = np.array(m['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, vmin=0, vmax=1)
            ax.set_title(f'Confusion Matrix: {model_name}')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'confusion_matrices.png')
        plt.close()
        
        # 4. PR Curves
        fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(6 * len(CLASS_NAMES), 5))
        if len(CLASS_NAMES) == 1: axes = [axes]
        
        for i, cls_name in enumerate(CLASS_NAMES):
            ax = axes[i]
            for model_name, m in all_metrics.items():
                pr_data = m['pr_data'][cls_name]
                ax.plot(pr_data['recall'], pr_data['precision'], label=f"{model_name} (AP={pr_data['ap']:.2f})")
            
            ax.set_title(f'PR Curve: {cls_name}')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_ylim(0, 1.05)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'pr_curves.png')
        plt.close()

    def run(self):
        logger.info(f"Starting comparison for {len(model_configs)} models")
        
        # 1. Get Data
        image_files, label_files = self.get_validation_data()
        if not image_files:
            logger.error("No validation data found. Exiting.")
            return
            
        all_metrics = {}
        
        # 2. Evaluate Each Model
        for config in model_configs:
            logger.info(f"Evaluating {config['name']}...")
            import time
            start_time = time.time()
            
            model = self.load_model(config)
            if model is None:
                continue
                
            y_pred, y_true, y_probs = self.evaluate_model(model, image_files, label_files)
            
            metrics = self.compute_metrics(y_true, y_pred, y_probs)
            metrics['eval_time'] = time.time() - start_time
            
            all_metrics[config['name']] = metrics
            
            # Save individual metrics
            with open(self.metrics_dir / f"{config['name'].replace(' ', '_')}_metrics.json", 'w') as f:
                json.dump(metrics, f, indent=4)
                
            # Clear memory
            del model, y_pred, y_true, y_probs
            torch.cuda.empty_cache()
            
        # 3. Create Summary Table
        self.create_summary_table(all_metrics)
        
        # 4. Generate Comparison Plots
        self.plot_comparisons(all_metrics)
        
        logger.info(f"Comparison complete. Results saved to {self.output_dir}")

    def create_summary_table(self, all_metrics):
        rows = []
        for name, m in all_metrics.items():
            row = {
                'Model': name,
                'Accuracy': m['Accuracy'],
                'Mean_IoU': m['Mean_IoU'],
                'Mean_F1': m['Mean_F1'],
                'Mean_AP': m['Mean_AP'],
                'Time(s)': m['eval_time']
            }
            # Add per-class IoU
            for cls_name in CLASS_NAMES:
                row[f'{cls_name}_IoU'] = m['per_class'][cls_name]['IoU']
            rows.append(row)
            
        df = pd.DataFrame(rows)
        print("\n=== Summary Metrics ===")
        print(df.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
        df.to_csv(self.metrics_dir / 'summary_metrics.csv', index=False)

if __name__ == "__main__":
    output_dir = project_root / f"model_comparison_results_{CUSTOM_MODEL_VERSION}"
    comparator = ModelComparator(output_dir)
    comparator.run()