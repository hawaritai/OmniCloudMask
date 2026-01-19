"""
Professional Blur Detection System
==================================

A comprehensive blur detection library for quality inspection reports.
Implements multiple blur detection algorithms with configurable thresholds
and batch processing capabilities.

Author: 
Version: 2.0.0 - Refactored with Factory Pattern
"""
import rawpy
import cv2
import numpy as np
from typing import Union, List, Tuple, Dict, Optional, Any
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum
import json
import time
import pandas as pd
import os
from tqdm import tqdm
import csv
import torch
from photogram.services.detector_factory import DetectorFactory
from photogram.services.detector_factory import DetectorBase

from photogram.imageio import ImageIOFactory
from photogram.entities.image_record import ImageRecord

# from imageqc_gui.adapters.image_holder import ImageHolder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BlurMethod(Enum):
    """Available blur detection methods."""
    LAPLACIAN_VARIANCE = "laplacian_variance"
    FFT_ANALYSIS = "fft_analysis"
    GRADIENT_MAGNITUDE = "gradient_magnitude"
    SOBEL_VARIANCE = "sobel_variance"
    TENENGRAD = "tenengrad"
    # NN_BLUR = "nn_blur"


@dataclass
class BlurResult:
    """Result container for blur detection."""
    is_blurred: bool
    blur_score: float
    method: BlurMethod
    threshold: float
    confidence: float
    image_shape: Tuple[int, int, int]
    processing_time: float


@dataclass
class BlurConfig:
    """Configuration for blur detection parameters."""
    laplacian_threshold: float = 100.0
    fft_threshold: float = 10.0
    gradient_threshold: float = 50.0
    sobel_threshold: float = 100.0
    tenengrad_threshold: float = 500.0
    # nn_threshold: float = 0.5
    resize_width: int = None
    patch_size: Tuple[int, int] = (1024, 1024)
    gaussian_blur_ksize: Tuple[int, int] = (5, 5)
    confidence_multiplier: float = 1.0


# ===== DETECTOR IMPLEMENTATIONS WITH FACTORY REGISTRATION =====

@DetectorFactory.register_detector("blur")
class BlurDetectionSystem(DetectorBase):
    """
    Main blur detection system for inspection reports - now factory-compatible.
    
    Features:
    - Multiple blur detection algorithms
    - Configurable thresholds
    - Batch processing
    - Performance monitoring
    - JSON export capabilities
    """
    
    methods = [
        BlurMethod.LAPLACIAN_VARIANCE,
        BlurMethod.FFT_ANALYSIS,
        BlurMethod.SOBEL_VARIANCE,
        # BlurMethod.NN_BLUR
    ]

    # Start with fixed fields
    blur_schema = {
        "image_name": str,
        "blur": bool
    }

    # Dynamically add method fields
    for method in methods:
        method_name = method.name if hasattr(method, "name") else str(method)
        blur_schema[f"{method_name}_area"] = int
        blur_schema[f"{method_name}_mean"] = float
        blur_schema[f"{method_name}_min"] = float
    
    
    def __init__(self, threshold: float = 60.0, output_dir: Optional[Path] = None, **kwargs):
        """
        Initialize blur detection system.
        
        Args:
            threshold: Main threshold for blur detection
            output_dir: Directory for saving outputs
            **kwargs: Additional configuration parameters
        """
        super().__init__(threshold, **kwargs)
        self.output_dir = output_dir / "blur"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize config with custom threshold
        config = BlurConfig(
            laplacian_threshold=threshold,  # Use passed threshold
            fft_threshold=5.8,
            sobel_threshold=100.0,
            # nn_threshold=0.5,
            patch_size=(1024, 1024)
        )
        self.config = config
        
        # Initialize internal blur method detectors
        self.detectors = {
            BlurMethod.LAPLACIAN_VARIANCE: LaplacianVarianceDetector(),
            BlurMethod.FFT_ANALYSIS: FFTAnalysisDetector(),
            BlurMethod.GRADIENT_MAGNITUDE: GradientMagnitudeDetector(),
            BlurMethod.SOBEL_VARIANCE: SobelVarianceDetector(),
            BlurMethod.TENENGRAD: TenengradDetector(),
            # BlurMethod.NN_BLUR: nnBlurDetector()
        }
        
        logger.info("BlurDetectionSystem initialized with methods: %s", 
                   list(self.detectors.keys()))
        
        # Image processing variables
        self.img_path = None
        self.img = None
        self.img_gray = None
        self.img_patches = None
        self.img_patches_gray = None
    
        # Image reader
        self.io_factory = ImageIOFactory()
    
    def load_image(self, image_data: str|Path|np.ndarray) -> None:
        """Load image for processing - required by DetectorBase"""
        if isinstance(image_data, (str, Path)):
            self.img_path = Path(image_data)
            if not self.img_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_data}")
            
            # self.img = self.io_factory.read(str(self.img_path), preferred_reader="rawpy")
            img_suffix = self.img_path.suffix.lower()
            # Check extension
            if img_suffix == '.iiq':
                self.img = self.process_iiq_to_rgb(str(self.img_path))
            else:
                self.img = cv2.imread(str(image_data))
            
            self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) if len(self.img.shape) == 3 else self.img
            
            if self.img is None:
                raise ValueError(f"Could not load image: {image_data}")
        else:
            self.img = image_data  # Assume it's already a numpy array
            self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) if len(self.img.shape) == 3 else self.img
        
        # Resize image for consistent processing
        if self.config.resize_width is not None and self.img.shape[1] > self.config.resize_width:
            height = int(self.img.shape[0] * self.config.resize_width / self.img.shape[1])
            self.img = cv2.resize(self.img, (self.config.resize_width, height))
            self.img_gray = cv2.resize(self.img_gray, (self.config.resize_width, height))
        
        logger.info("Image loaded: %s", self.img.shape)
    
    def image_record_reader(self, image_record: ImageRecord) -> None:
        """Load image for processing - required by DetectorBase"""
        self.img_path = Path(image_record.path)
        self.img = image_record._raw_data if image_record._raw_data is not None else image_record._data
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) if len(self.img.shape) == 3 else self.img
        
        # Resize image for consistent processing
        if self.config.resize_width is not None and self.img.shape[1] > self.config.resize_width:
            height = int(self.img.shape[0] * self.config.resize_width / self.img.shape[1])
            self.img = cv2.resize(self.img, (self.config.resize_width, height))
            self.img_gray = cv2.resize(self.img_gray, (self.config.resize_width, height))
        
        logger.info("Image loaded: %s", self.img.shape)
    
    def run(self, image_record: ImageRecord) -> Dict:
        """
        Run the blur detection system - required by DetectorBase.
        
        Returns:
            Dict: Detection results summary
        """
        # self.image_record_reader(image_record)
        
        if self.img_gray is None:
            raise RuntimeError("No image loaded. Call preprocess_image() first.")
            
        self.generate_patches(gray=True)
        # Single image detection by patch
        result, _ = self.detect_blur_by_patch_LP_FFT_Sobel_nn(mask=True)
        result_df = pd.DataFrame(result)
        methods = result_df['method'].unique()
        img_name = self.img_path.name
        
        output = {
            "image_name": img_name,
            "blur": False
        }
        
        # Per-method stats + weighted blur calculation
        for method in methods:
            df = result_df[result_df['method'] == method]
            blurred = df[df['is_blurred'] == True]
            
            if blurred.empty:
                output[f"{method}_area"] = 0
                output[f"{method}_mean"] = 0
                output[f"{method}_min"] = 0
                continue
            blurred_score_mean = blurred['blur_score'].mean()
            blurred_count = len(blurred)
            blurred_score_min = blurred['blur_score'].min()
            
            # If more than 10 patches are blurred for any one of the methods, consider the image blurred
            if blurred_count > 10:
                output['blur'] = True
            
            output[f"{method}_area"] = blurred_count
            output[f"{method}_mean"] = blurred_score_mean
            output[f"{method}_min"] = blurred_score_min
        
        # save the results 
        csv_save_dir = self.output_dir / "blur_results.csv"
        self.export_results(output, output_path=csv_save_dir)
        
        return output
    
    # ===== ALL EXISTING METHODS REMAIN THE SAME =====
    
    def process_iiq_to_rgb(self, file_path: str) -> np.ndarray:
        """Processes an IIQ file into a 16-bit RGB NumPy array."""
        with rawpy.imread(file_path) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                output_bps=8
            )
            print("Processed RGB Shape:", rgb.shape)
            print("Processed RGB dtype:", rgb.dtype)
            return rgb
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for blur detection."""
        return image
    
    def generate_patches(self, gray = True):
        """Generate image patches for analysis."""
        patch_size = self.config.patch_size
        if gray:
            patches_gray = []
            for i in range(0, self.img_gray.shape[0], patch_size[0]):
                for j in range(0, self.img_gray.shape[1], patch_size[1]):
                    patch = self.img_gray[i:i+patch_size[0], j:j+patch_size[1]]
                    if patch.shape == patch_size:
                        patches_gray.append(patch)
            self.img_patches_gray = np.array(patches_gray)
        else:
            patches = []
            patches_gray = []
            for i in range(0, self.img_gray.shape[0], patch_size[0]):
                for j in range(0, self.img_gray.shape[1], patch_size[1]):
                    patch = self.img[i:i+patch_size[0], j:j+patch_size[1]]
                    if patch.shape == patch_size:
                        patches.append(patch)
                    patch_gray = self.img_gray[i:i+patch_size[0], j:j+patch_size[1]]
                    if patch_gray.shape == patch_size:
                        patches_gray.append(patch_gray)
            self.img_patches = np.array(patches)
            self.img_patches_gray = np.array(patches_gray)
    
    def calculate_confidence(self, blur_score: float, threshold: float, method: BlurMethod) -> float:
        """Calculate confidence score for blur detection result."""
        if method == BlurMethod.FFT_ANALYSIS:
            ratio = blur_score / threshold
        else:
            ratio = blur_score / threshold
        
        confidence = min(1.0, ratio * self.config.confidence_multiplier)
        return max(0.0, confidence)
    
    def detect_blur_by_patch_LP_FFT_Sobel_nn(self, image: Union[np.ndarray, str, Path]= None, gray = True,  mask: bool=True) -> Union[BlurResult, Tuple[BlurResult, np.ndarray]]:
        """Detect blur using multiple methods on image patches."""
        if gray:  
            if image is None or self.img is None:
                pass
            else:
                self.load_image(image)
                self.generate_patches(True)
            patches = self.img_patches_gray
        
        methods = [BlurMethod.LAPLACIAN_VARIANCE,
                   BlurMethod.FFT_ANALYSIS,
                   BlurMethod.SOBEL_VARIANCE,
                #    BlurMethod.NN_BLUR
                   ]
        
        results_array = []
        blur_scores = {BlurMethod.LAPLACIAN_VARIANCE.name: [],
                   BlurMethod.FFT_ANALYSIS.name: [],
                   BlurMethod.SOBEL_VARIANCE.name: [],
                #    BlurMethod.NN_BLUR.name: []
                   }
        
        for patch in tqdm(patches, desc="Detecting blur on patches", total=len(patches)):        
            start_time = time.time()
            original_shape = patch.shape
            processed_image = self.preprocess_image(patch)
            
            for method in methods:
                detector = self.detectors[method]
                threshold = detector.get_threshold(self.config)
                blur_score = detector.detect_blur(processed_image)
                blur_scores[method.name].append(blur_score)
                is_blurred = blur_score < threshold
                confidence = self.calculate_confidence(blur_score, threshold, method)
                processing_time = time.time() - start_time
                
                data = {
                    'is_blurred': is_blurred,
                    'blur_score': blur_score,
                    'method': method.name,
                    'threshold': threshold,
                    'confidence': confidence,
                    'image_shape': original_shape,
                    'processing_time': processing_time
                }
                results_array.append(data)
        
        if mask and hasattr(self, 'img') and self.img is not None:
            blur_masks = {BlurMethod.LAPLACIAN_VARIANCE.name: [],
                   BlurMethod.FFT_ANALYSIS.name: [],
                   BlurMethod.SOBEL_VARIANCE.name: [],
                #    BlurMethod.NN_BLUR.name: []
                   }
            for method in blur_scores:
                blur_mask = self.create_blur_mask(blur_scores[method])
                blur_path = self.output_dir / f"{self.img_path.stem}_{method.split('_')[0]}_p{str(self.config.patch_size[0])}_mask_.tif"
                self.save_blur_mask(blur_mask, blur_path)
                blur_masks[method].append(blur_mask)

            return results_array, blur_masks
        
        return results_array
    
    def create_blur_mask(self, blur_scores: List[float]) -> np.ndarray:
        """Create a mask where each pixel value corresponds to the blur score of its patch"""
        if len(self.img.shape) == 3:
            h, w, _ = self.img.shape
        else:
            h, w = self.img.shape
        
        blur_mask = np.zeros((h, w), dtype=np.float32)
        
        if hasattr(self, 'img_patches_gray') and len(self.img_patches_gray) > 0:
            patch_h, patch_w = self.img_patches_gray[0].shape[:2]
        else:
            total_patches = len(blur_scores)
            patch_h = patch_w = int(np.sqrt(h * w / total_patches))
        
        patches_per_row = w // patch_w
        patches_per_col = h // patch_h
        
        for i, blur_score in enumerate(blur_scores):
            row = i // patches_per_row
            col = i % patches_per_row
            
            start_y = row * patch_h
            end_y = min(start_y + patch_h, h)
            start_x = col * patch_w
            end_x = min(start_x + patch_w, w)
            
            blur_mask[start_y:end_y, start_x:end_x] = blur_score
        
        return blur_mask
    
    def create_blur_mask_with_coordinates(self, blur_scores: List[float], patch_coordinates: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Create a blur mask using stored patch coordinates
        
        Args:
            blur_scores: List of blur scores for each patch
            patch_coordinates: List of (start_y, end_y, start_x, end_x) for each patch
            
        Returns:
            np.ndarray: Blur score mask
        """
        # Get original image dimensions
        if len(self.img.shape) == 3:  # Color image
            h, w, _ = self.img.shape
        else:  # Grayscale image
            h, w = self.img.shape
        
        # Initialize mask
        blur_mask = np.zeros((h, w), dtype=np.float32)
        
        # Fill mask using coordinates
        for blur_score, (start_y, end_y, start_x, end_x) in zip(blur_scores, patch_coordinates):
            blur_mask[start_y:end_y, start_x:end_x] = blur_score
        
        return blur_mask

    def save_blur_mask(self, blur_mask: np.ndarray, output_path: Union[str, Path], 
                    mask_type: str = 'original', colormap: str = 'viridis',
                    threshold: float = None) -> None:
        """Save blur mask as PNG or JPEG image"""
        import cv2
        import matplotlib.pyplot as plt
        
        output_path = Path(output_path)
        
        if mask_type == 'heatmap':
            normalized_mask = (blur_mask - blur_mask.min()) / (blur_mask.max() - blur_mask.min())
            cmap = plt.get_cmap(colormap)
            colored_mask = cmap(normalized_mask)
            
            if colored_mask.shape[-1] == 4:
                colored_mask = colored_mask[:, :, :3]
            colored_mask = (colored_mask * 255).astype(np.uint8)
            colored_mask = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
            
        elif mask_type == 'binary':
            if threshold is None:
                threshold = np.mean(blur_mask)
            binary_mask = (blur_mask < threshold).astype(np.uint8) * 255
            colored_mask = binary_mask
            
        elif mask_type == 'normalized':
            normalized_mask = ((blur_mask - blur_mask.min()) / 
                            (blur_mask.max() - blur_mask.min()) * 255).astype(np.uint8)
            colored_mask = normalized_mask
            
        elif mask_type == 'raw':
            clipped_mask = np.clip(blur_mask, np.percentile(blur_mask, 1), 
                                np.percentile(blur_mask, 99))
            raw_mask = ((clipped_mask - clipped_mask.min()) / 
                    (clipped_mask.max() - clipped_mask.min()) * 255).astype(np.uint8)
            colored_mask = raw_mask
            
        elif mask_type == 'original':
            colored_mask = blur_mask.astype(np.float32)
        else:
            raise ValueError(f"Unknown mask_type: {mask_type}")
        
        if mask_type == 'original':
            if str(output_path).lower().endswith('.npy'):
                np.save(str(output_path), colored_mask)
                print(f"Original blur mask saved as numpy array to: {output_path}")
            else:
                if str(output_path).lower().endswith(('.tif', '.tiff')):
                    import tifffile
                    # colored_mask = colored_mask.astype(np.uint8)
                    # tifffile.imwrite(str(output_path), colored_mask,
                    #                 compression='lzw')
                    tifffile.imwrite(str(output_path), colored_mask, 
                        compression='lzw',
                        predictor=True)

                    print(f"Original blur mask saved as TIFF to: {output_path}")
                else:
                    mask_16bit = ((colored_mask - colored_mask.min()) / 
                                (colored_mask.max() - colored_mask.min()) * 65535).astype(np.uint16)
                    success = cv2.imwrite(str(output_path), mask_16bit)
                    if success:
                        print(f"Original blur mask saved as 16-bit image to: {output_path}")
                    else:
                        print(f"Failed to save blur mask to: {output_path}")
        else:
            success = cv2.imwrite(str(output_path), colored_mask)
            if success:
                print(f"Blur mask saved to: {output_path}")
            else:
                print(f"Failed to save blur mask to: {output_path}")
    
    def detect_blur_by_patch(self, image: Union[np.ndarray, str, Path]= None, gray = True, method: BlurMethod = BlurMethod.LAPLACIAN_VARIANCE, mask: bool=True) -> Union[BlurResult, Tuple[BlurResult, np.ndarray]]:
        """
        Detect blur in a single image by patching them
        
        Args:
            image: Input image (numpy array, file path, or Path object)
            gray: Whether to use grayscale patches
            method: Blur detection method to use
            mask: Whether to generate a blur score mask
            
        Returns:
            BlurResult or Tuple[BlurResult, np.ndarray]: Detection result(s) and optionally blur mask
        """
        if gray:  
            if image is None:
                pass
            else:
                self.load_image(image)
                self.generate_patches(True)
            patches = self.img_patches_gray
        
        results = []
        blur_scores = []
        
        for patch in patches:        
            start_time = time.time()
            
            original_shape = patch.shape
            
            # Preprocess image
            processed_image = self.preprocess_image(patch)
            
            # Get detector and threshold
            detector = self.detectors[method]
            threshold = detector.get_threshold(self.config)
            
            # Calculate blur score
            blur_score = detector.detect_blur(processed_image)
            blur_scores.append(blur_score)
            
            # Determine if image is blurred
            is_blurred = blur_score < threshold
            
            # Calculate confidence
            confidence = self.calculate_confidence(blur_score, threshold, method)
            
            processing_time = time.time() - start_time
            
            result = BlurResult(
                is_blurred=is_blurred,
                blur_score=blur_score,
                method=method,
                threshold=threshold,
                confidence=confidence,
                image_shape=original_shape,
                processing_time=processing_time
            )
            results.append(result)
        
        if mask and hasattr(self, 'img') and self.img is not None:
            blur_mask = self.create_blur_mask(blur_scores)
            self.save_blur_mask(blur_mask, f"{self.output_dir}/{self.img_path.stem}_{method.name.split('_')[0]}_p{str(self.config.patch_size[0])}_mask_.tif")
            return results, blur_mask
        
        return results

    def detect_blur_by_patch_all_methods(self, image: Union[np.ndarray, str, Path]= None, gray = True):
        """
        Detect blur in a single image by patching them
        
        Args:
            image: Input image (numpy array, file path, or Path object)
            method: Blur detection method to use
            
        Returns:
            BlurResult: Detection result with scores and metadata
        """
        if gray:  
            if image is None:
                pass
            else:
                self.load_image(image)
                self.generate_patches(True)
            patches = self.img_patches_gray
        results = []
        for patch in patches:
            results.append([self.detect_blur(patch, method = BlurMethod.LAPLACIAN_VARIANCE),
                            self.detect_blur(patch, method = BlurMethod.FFT_ANALYSIS),
                            self.detect_blur(patch, method = BlurMethod.GRADIENT_MAGNITUDE),
                            self.detect_blur(patch, method = BlurMethod.SOBEL_VARIANCE),
                            self.detect_blur(patch, method = BlurMethod.TENENGRAD)])
        return results
    
    def detect_blur(self, 
                   image: Union[np.ndarray, str, Path] = None,
                   method: BlurMethod = BlurMethod.LAPLACIAN_VARIANCE) -> BlurResult:
        """
        Detect blur in a single image.
        
        Args:
            image: Input image (numpy array, file path, or Path object)
            method: Blur detection method to use
            
        Returns:
            BlurResult: Detection result with scores and metadata
        """
        if image is None:
            image = self.img_gray
        else:
            pass
        
        start_time = time.time()
        
        original_shape = image.shape
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Get detector and threshold
        detector = self.detectors[method]
        threshold = detector.get_threshold(self.config)
        
        # Calculate blur score
        blur_score = detector.detect_blur(processed_image)
        
        # Determine if image is blurred
        is_blurred = blur_score < threshold
        
        # Calculate confidence
        confidence = self.calculate_confidence(blur_score, threshold, method)
        
        processing_time = time.time() - start_time
        
        result = BlurResult(
            is_blurred=is_blurred,
            blur_score=blur_score,
            method=method,
            threshold=threshold,
            confidence=confidence,
            image_shape=original_shape,
            processing_time=processing_time
        )
        
        logger.debug("Blur detection completed: %s (score: %.2f, threshold: %.2f)", 
                    "BLURRED" if is_blurred else "SHARP", blur_score, threshold)
        
        return result
    
    def detect_blur_multi_method(self, 
                                image: Union[np.ndarray, str, Path],
                                methods: Optional[List[BlurMethod]] = None) -> Dict[BlurMethod, BlurResult]:
        """
        Detect blur using multiple methods for comprehensive analysis.
        
        Args:
            image: Input image
            methods: List of methods to use (default: all methods)
            
        Returns:
            Dict[BlurMethod, BlurResult]: Results for each method
        """
        if methods is None:
            methods = list(BlurMethod)
        
        results = {}
        for method in methods:
            try:
                results[method] = self.detect_blur(image, method)
            except Exception as e:
                logger.error("Error detecting blur with method %s: %s", method, e)
                continue
        
        return results
    
    def batch_detect_blur(self, 
                         image_paths: List[Union[str, Path]],
                         method: BlurMethod = BlurMethod.LAPLACIAN_VARIANCE) -> Dict[str, BlurResult]:
        """
        Process multiple images for blur detection.
        
        Args:
            image_paths: List of image file paths
            method: Blur detection method to use
            
        Returns:
            Dict[str, BlurResult]: Results keyed by image path
        """
        results = {}
        
        logger.info("Starting batch blur detection for %d images", len(image_paths))
        
        for i, image_path in enumerate(image_paths):
            try:
                result = self.detect_blur(image_path, method)
                results[str(image_path)] = result
                
                if (i + 1) % 10 == 0:
                    logger.info("Processed %d/%d images", i + 1, len(image_paths))
                    
            except Exception as e:
                logger.error("Error processing image %s: %s", image_path, e)
                continue
        
        logger.info("Batch processing completed. %d/%d images processed successfully", 
                   len(results), len(image_paths))
        
        return results
    
    def generate_report(self, results: Dict[str, BlurResult]) -> Dict:
        """
        Generate summary report from batch detection results.
        
        Args:
            results: Batch detection results
            
        Returns:
            Dict: Summary report
        """
        if not results:
            return {"error": "No results to summarize"}
        
        total_images = len(results)
        blurred_images = sum(1 for r in results.values() if r.is_blurred)
        sharp_images = total_images - blurred_images
        
        scores = [r.blur_score for r in results.values()]
        avg_score = np.mean(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        avg_processing_time = np.mean([r.processing_time for r in results.values()])
        
        # Get method and threshold from first result
        first_result = next(iter(results.values()))
        
        report = {
            "summary": {
                "total_images": total_images,
                "blurred_images": blurred_images,
                "sharp_images": sharp_images,
                "blur_percentage": (blurred_images / total_images) * 100,
                "method_used": first_result.method.value,
                "threshold": first_result.threshold
            },
            "statistics": {
                "average_blur_score": avg_score,
                "min_blur_score": min_score,
                "max_blur_score": max_score,
                "average_processing_time": avg_processing_time
            },
            "blurred_images": [
                {
                    "path": path,
                    "blur_score": result.blur_score,
                    "confidence": result.confidence
                }
                for path, result in results.items() if result.is_blurred
            ]
        }
        
        return report
    
    def _export_results(self, results: Dict[str, BlurResult], output_path: Union[str, Path]):
        """
        Export results to JSON file.
        
        Args:
            results: Detection results to export
            output_path: Output file path
        """
        def serialize_result(result: BlurResult) -> Dict:
            return {
                "is_blurred": result.is_blurred,
                "blur_score": result.blur_score,
                "method": result.method.value,
                "threshold": result.threshold,
                "confidence": result.confidence,
                "image_shape": result.image_shape,
                "processing_time": result.processing_time
            }
        
        export_data = {
            "results": {path: serialize_result(result) for path, result in results.items()},
            "report": self.generate_report(results)
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info("Results exported to %s", output_path)
    
    def export_results(self, results: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """Export results dictionary into a CSV file with JSON-safe serialization."""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to native Python
        def json_safe(obj):
            if isinstance(obj, (np.generic,)):
                return obj.item()
            return obj

        safe_results = {k: json_safe(v) for k, v in results.items()}

        # Write CSV
        file_exists = output_path.exists()
        with output_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=safe_results.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(safe_results)
        
    def update_config(self, **kwargs):
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info("Updated config: %s = %s", key, value)
            else:
                logger.warning("Unknown config parameter: %s", key)
    

# ===== INTERNAL DETECTOR CLASSES =====

class LaplacianVarianceDetector():
    """Laplacian variance based blur detection on GPU with PyTorch."""
    
    def detect_blur(self, image: np.ndarray = None) -> float:
        """Calculate Laplacian variance."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gray = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
        
        laplacian_kernel = torch.tensor([[[[0, 1, 0],
                                           [1, -4, 1],
                                           [0, 1, 0]]]]).float().to(device)
        
        laplacian = torch.nn.functional.conv2d(gray, laplacian_kernel, padding=1)
        return laplacian.var().item()
    
    def get_blur_strength(self, value: float, config) -> float:
        threshold = self.get_threshold(config)
        if value >= threshold:
            return 0.0
        return min(100.0, 100 * (1 - (value / threshold)))
    
    def get_threshold(self, config) -> float:
        return config.laplacian_threshold
    
    def get_method(self) -> BlurMethod:
        return BlurMethod.LAPLACIAN_VARIANCE

class FFTAnalysisDetector():
    """FFT-based blur detection using frequency domain analysis on GPU with PyTorch."""
    
    def detect_blur(self, image: np.ndarray = None) -> float:
        """Analyze high frequency components using FFT."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gray = torch.from_numpy(image).float().to(device)
        
        f_transform = torch.fft.fft2(gray)
        f_shift = torch.fft.fftshift(f_transform)
        magnitude_spectrum = torch.log(torch.abs(f_shift) + 1)
        
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Create mask on CPU then to GPU, or better on GPU
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        y = y.to(device)
        x = x.to(device)
        mask = ((x - center_w)**2 + (y - center_h)**2 > (min(center_h, center_w) // 3)**2).float()
        
        high_freq_energy = torch.mean(magnitude_spectrum * mask)
        return high_freq_energy.item()
    
    def get_blur_strength(self, value: float, config) -> float:
        threshold = self.get_threshold(config)
        if value >= threshold:
            return 0.0
        return min(100.0, 100 * (1 - (value / threshold)))
    
    def get_threshold(self, config) -> float:
        return config.fft_threshold
    
    def get_method(self) -> BlurMethod:
        return BlurMethod.FFT_ANALYSIS

class SobelVarianceDetector():
    """Sobel operator variance based blur detection on GPU with PyTorch."""
    
    def detect_blur(self, image: np.ndarray = None) -> float:
        """Calculate Sobel variance."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gray = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
        
        sobel_x_kernel = torch.tensor([[[[-1, 0, 1],
                                          [-2, 0, 2],
                                          [-1, 0, 1]]]]).float().to(device)
        sobel_y_kernel = torch.tensor([[[[-1, -2, -1],
                                          [0, 0, 0],
                                          [1, 2, 1]]]]).float().to(device)
        
        sobel_x = torch.nn.functional.conv2d(gray, sobel_x_kernel, padding=1)
        sobel_y = torch.nn.functional.conv2d(gray, sobel_y_kernel, padding=1)
        sobel_combined = sobel_x + sobel_y
        
        return sobel_combined.var().item()
    
    def get_blur_strength(self, value: float, config) -> float:
        threshold = self.get_threshold(config)
        if value >= threshold:
            return 0.0
        return min(100.0, 100 * (1 - (value / threshold)))
    
    def get_threshold(self, config) -> float:
        return config.sobel_threshold
    
    def get_method(self) -> BlurMethod:
        return BlurMethod.SOBEL_VARIANCE

class TenengradDetector():
    """Tenengrad based blur detection on GPU with PyTorch."""
    
    def detect_blur(self, image: np.ndarray = None) -> float:
        """Calculate Tenengrad focus measure."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gray = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
        
        sobel_x_kernel = torch.tensor([[[[-1, 0, 1],
                                          [-2, 0, 2],
                                          [-1, 0, 1]]]]).float().to(device)
        sobel_y_kernel = torch.tensor([[[[-1, -2, -1],
                                          [0, 0, 0],
                                          [1, 2, 1]]]]).float().to(device)
        
        sobel_x = torch.nn.functional.conv2d(gray, sobel_x_kernel, padding=1)
        sobel_y = torch.nn.functional.conv2d(gray, sobel_y_kernel, padding=1)
        tenengrad = torch.sqrt(sobel_x**2 + sobel_y**2)
        return tenengrad.mean().item()
    
    def get_blur_strength(self, value: float, config) -> float:
        threshold = self.get_threshold(config)
        if value >= threshold:
            return 0.0
        return min(100.0, 100 * (1 - (value / threshold)))
    
    def get_threshold(self, config) -> float:
        return config.tenengrad_threshold
    
    def get_method(self) -> BlurMethod:
        return BlurMethod.TENENGRAD

def dfs(x, y, pronounced, visited, rows, cols):
    # Note: DFS is serial, remains on CPU for simplicity
    stack = [(x, y)]
    size = 0
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while stack:
        cx, cy = stack.pop()
        if not visited[cx, cy]:
            visited[cx, cy] = True
            size += 1
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < rows and 0 <= ny < cols and pronounced[nx, ny] and not visited[nx, ny]:
                    stack.append((nx, ny))
    return size
class GradientMagnitudeDetector():
    """Gradient magnitude based blur detection."""
    
    def detect_blur(self, image: np.ndarray = None) -> float:
        """Calculate gradient magnitude variance."""
        gray = image
        
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return magnitude.var()
    
    def get_threshold(self, config: BlurConfig) -> float:
        return config.gradient_threshold
    
    def get_method(self) -> BlurMethod:
        return BlurMethod.GRADIENT_MAGNITUDE

class HaarWaveletDetector():
    """Haar Wavelet Transform based blur detection, partially on GPU with PyTorch."""
    
    def detect_blur(self, image: np.ndarray = None) -> float:
        """Calculate sharpness ratio using HWT."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gray_np = image
        h, w = gray_np.shape
        N = 16  # Tile size
        niter = 4  # Number of iterations
        theta = 10.0  # Threshold for pronounced changes
        csz = 3  # Minimum cluster size
        
        rows = h // N
        cols = w // N
        
        # Crop to fit
        if h % N != 0 or w % N != 0:
            gray_np = gray_np[:rows*N, :cols*N]
        
        gray = torch.from_numpy(gray_np).float().to(device)
        
        pronounced = torch.zeros((rows, cols), dtype=torch.bool, device=device)
        
        for i in range(rows):
            for j in range(cols):
                tile = gray[i*N:(i+1)*N, j*N:(j+1)*N]
                current = tile.clone()
                for _ in range(niter):
                    ch, cw = current.shape
                    ll = (current[0::2, 0::2] + current[0::2, 1::2] + current[1::2, 0::2] + current[1::2, 1::2]) / 2
                    lh = (current[0::2, 0::2] + current[0::2, 1::2] - current[1::2, 0::2] - current[1::2, 1::2]) / 2  # Vertical
                    hl = (current[0::2, 0::2] - current[0::2, 1::2] + current[1::2, 0::2] - current[1::2, 1::2]) / 2  # Horizontal
                    hh = (current[0::2, 0::2] - current[0::2, 1::2] - current[1::2, 0::2] + current[1::2, 1::2]) / 2  # Diagonal
                    current = ll
                ah = torch.mean(torch.abs(hl))
                av = torch.mean(torch.abs(lh))
                ad = torch.mean(torch.abs(hh))
                if ah > theta or av > theta or ad > theta:
                    pronounced[i, j] = True
        
        # For DFS, move to CPU as it's serial
        pronounced_cpu = pronounced.cpu().numpy()
        visited = np.zeros((rows, cols), dtype=bool)
        total_area = 0
        tile_area = N * N
        
        for i in range(rows):
            for j in range(cols):
                if pronounced_cpu[i, j] and not visited[i, j]:
                    cluster_size = dfs(i, j, pronounced_cpu, visited, rows, cols)
                    if cluster_size > csz:
                        total_area += cluster_size * tile_area
        
        image_area = h * w
        ratio = total_area / image_area
        return ratio
    
    def get_blur_strength(self, value: float, config) -> float:
        threshold = self.get_threshold(config)
        if value >= threshold:
            return 0.0
        return min(100.0, 100 * (1 - (value / threshold)))
    
    def get_threshold(self, config) -> float:
        return config.hwt_threshold
    
    def get_method(self) -> BlurMethod:
        return BlurMethod.HAARWAVELET
    
if __name__ == "__main__":
    # Initiate the class
    from photogram.entities.image_record import ImageRecord
    from photogram.entities.enums import ColorSpace
    from pathlib import Path
    width = height = bands = 0
    dtype = "unknown"
    geo = None
    
    threshold = 60.0  # Main threshold for blur detection
    image_dir = r"E:\ImageQC\towhid\test_image"
    image_list = os.listdir(image_dir)
    image_list = sorted(image_list, reverse=True)
    output_dir = r"E:\ImageQC\towhid\10_10_25\1"
    blur_detector = BlurDetectionSystem(threshold, Path(output_dir))
    for image in image_list:
        input_path = os.path.join(image_dir, image)
        blur_detector.load_image(input_path)

        blur_detector.run(blur_detector.img)
        print(" ")
    print(" ")
    