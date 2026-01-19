#!/usr/bin/env python3
"""
Professional Cloud Shadow Detection System
==========================================

A modular, command-line driven system for detecting clouds and shadows in IIQ images
using OmniCloudMask technology with synthetic NIR generation.

Author: Professional Developer
Version: 2.0.0
"""

import os
import csv
import cv2
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import rasterio as rio
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from tqdm.auto import tqdm
from omnicloudmask import predict_from_array
import omnicloudmask
from datetime import datetime
import sys
from PIL import Image, ImageDraw, ImageFont

# Add custom library path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from photogram.imageio import ImageIOFactory

io_factory = ImageIOFactory()

# try:
from third_party.NIRGAN.create_NIR import get_NIR
NIR_GAN_AVAILABLE = True
# except ImportError:
#     print("Warning: NIR GAN library not available. Using synthetic NIR generation.")
#     NIR_GAN_AVAILABLE = False


class DetectionMode(Enum):
    """Detection modes supported by the system."""
    CLOUD_ONLY = "cloud"
    SHADOW_ONLY = "shadow" 
    BOTH = "both"


class ProcessingStatus(Enum):
    """Processing status enumeration."""
    SUCCESS = "Success"
    ERROR = "Error"
    SKIPPED = "Skipped"


@dataclass
class ProcessingResult:
    """Data class to store processing results for each image."""
    image_name: str
    has_cloud: bool
    cloud_area_sqm: float
    cloud_probability: float
    has_shadow: bool
    shadow_area_sqm: float
    processing_time: float
    status: ProcessingStatus
    error_message: Optional[str] = None


class FileManager:
    """Handles all file I/O operations and path management."""
    
    SUPPORTED_EXTENSIONS = ['.iiq', '.tif', '.tiff']
    
    def __init__(self, output_folder: Optional[str] = None, detection_mode: DetectionMode = DetectionMode.BOTH):
        """
        Initialize file manager.
        
        Args:
            output_folder: Path to output directory (optional)
            detection_mode: Detection mode for proper CSV naming
        """
        self.detection_mode = detection_mode
        self.output_folder = Path(output_folder) if output_folder else Path("CloudShadow_Results")
        self._setup_output_structure()
    
    def _setup_output_structure(self) -> None:
        """Create the output directory structure."""
        self.clouds_folder = self.output_folder / "clouds"
        self.shadows_folder = self.output_folder / "shadow"
        self.cloud_shadow_folder = self.output_folder / "cloud_shadow"
        
        # Create directories based on detection mode
        if self.detection_mode == DetectionMode.CLOUD_ONLY:
            self.clouds_folder.mkdir(parents=True, exist_ok=True)
        elif self.detection_mode == DetectionMode.SHADOW_ONLY:
            self.shadows_folder.mkdir(parents=True, exist_ok=True)
        else:  # BOTH
            for folder in [self.clouds_folder, self.shadows_folder, self.cloud_shadow_folder]:
                folder.mkdir(parents=True, exist_ok=True)
    
    def get_output_path(self, has_cloud: bool, has_shadow: bool) -> Path:
        """Determine output path based on detection results."""
        if self.detection_mode == DetectionMode.CLOUD_ONLY:
            return self.clouds_folder
        elif self.detection_mode == DetectionMode.SHADOW_ONLY:
            return self.shadows_folder
        else:  # BOTH mode
            if has_cloud and has_shadow:
                return self.cloud_shadow_folder
            elif has_cloud:
                return self.clouds_folder
            elif has_shadow:
                return self.shadows_folder
            else:
                return self.clouds_folder  # Default for clear images
    
    @property
    def csv_path(self) -> Path:
        """Get CSV report file path based on detection mode."""
        if self.detection_mode == DetectionMode.CLOUD_ONLY:
            return self.output_folder / 'clouds' / "ImgQC_results_cloud.csv"
        elif self.detection_mode == DetectionMode.SHADOW_ONLY:
            return self.output_folder / 'shadow' / "ImgQC_results_Shadow.csv"
        else:  # BOTH
            return self.output_folder / 'cloud_shadow' / "ImgQC_results_Cloud_Shadow.csv"
    
    @property
    def log_path(self) -> Path:
        """Get log file path."""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        return self.output_folder / f"ImgQC_{self.detection_mode.value}_processing_{timestamp}.log"


class CloudShadowDetector:
    """
    Core detection engine for cloud and shadow identification.
    """
    
    def __init__(self, detection_mode: DetectionMode = DetectionMode.BOTH, 
                 scale_factor: int = 20, use_gan_nir: bool = True):
        """
        Initialize detector.
        
        Args:
            detection_mode: Type of detection to perform
            scale_factor: Scaling factor for image processing
            use_gan_nir: Whether to use GAN-based NIR generation
        """
        self.detection_mode = detection_mode
        self.scale_factor = scale_factor
        self.use_gan_nir = use_gan_nir and NIR_GAN_AVAILABLE
        
        # Setup logger
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def detect_from_array(self, image_array: np.ndarray, 
                         resolution: Tuple[float, float]) -> Tuple[bool, float, float, bool, float]:
        """
        Perform cloud/shadow detection on image array.
        
        Args:
            image_array: RGB image array (H, W, 3)
            resolution: Pixel resolution (x, y)
            
        Returns:
            Tuple: (has_cloud, cloud_area_sqm, cloud_probability, has_shadow, shadow_area_sqm)
        """
        
        if image_array.ndim != 3 or image_array.shape[2] != 3:
            raise ValueError("Input must be (H, W, 3) RGB image")
        
        # Prepare channels
        red = image_array[:, :, 0].astype(np.float32) * self.scale_factor
        green = image_array[:, :, 1].astype(np.float32) * self.scale_factor
        
        # Generate NIR band
        if self.use_gan_nir:
            nir = get_NIR(image_array).astype(np.uint16)
        else:
            nir = self._generate_synthetic_nir(image_array)
        
        # Clip values
        red = np.clip(red, 0, 65535).astype(np.uint16)
        green = np.clip(green, 0, 65535).astype(np.uint16)
        
        # Stack bands for prediction
        rgn_stack = np.stack([red, green, nir], axis=0)
        
        # Perform prediction
        mask = predict_from_array(rgn_stack)
        mask_2d = mask[0]  # Extract 2D mask
        
        # Analyze mask
        return self._analyze_mask(mask_2d, self._calculate_pixel_area(resolution)), mask_2d
    
    def _generate_synthetic_nir(self, rgb_array: np.ndarray, method: str = "weighted") -> np.ndarray:
        """Generate synthetic NIR from RGB."""
        red = rgb_array[:, :, 0].astype(np.float32)
        green = rgb_array[:, :, 1].astype(np.float32)
        blue = rgb_array[:, :, 2].astype(np.float32)
        
        if method == "weighted":
            nir = 0.3 * red + 0.5 * green + 0.2 * blue
        elif method == "green_emphasis":
            nir = green * 1.2 - red * 0.2
        elif method == "ndvi_like":
            nir = (green + blue) / 2.0
        else:
            raise ValueError(f"Unsupported NIR generation method: {method}")
        
        nir = nir * self.scale_factor
        return np.clip(nir, 0, 65535).astype(np.uint16)
    
    def _calculate_pixel_area(self, resolution: Tuple[float, float]) -> float:
        """Calculate area per pixel in square meters."""
        return abs(resolution[0] * resolution[1])
    
    def _analyze_mask(self, mask: np.ndarray, pixel_area: float) -> Tuple[bool, float, float, bool, float]:
        """
        Analyze prediction mask for cloud/shadow metrics.
        
        Mask values: 0=Clear, 1=Cloud, 2=Cloud, 3=Shadow
        """
        cloud_pixels = np.sum((mask == 1) | (mask == 2))
        shadow_pixels = np.sum(mask == 3)
        total_pixels = mask.size
        
        # Apply detection mode filtering
        has_cloud = False
        has_shadow = False
        
        if self.detection_mode in [DetectionMode.CLOUD_ONLY, DetectionMode.BOTH]:
            has_cloud = cloud_pixels > 0
        
        if self.detection_mode in [DetectionMode.SHADOW_ONLY, DetectionMode.BOTH]:
            has_shadow = shadow_pixels > 0
        
        cloud_area_sqm = cloud_pixels * pixel_area
        shadow_area_sqm = shadow_pixels * pixel_area
        cloud_probability = cloud_pixels / total_pixels if total_pixels > 0 else 0.0
        
        return has_cloud, cloud_area_sqm, cloud_probability, has_shadow, shadow_area_sqm
    
    # def create_visualization(self, rgb_array: np.ndarray, mask: np.ndarray, 
    #                        output_path: Path, detection_mode: DetectionMode) -> None:
    #     """Create and save visualization plot based on detection mode."""
    #     fig, ax = plt.subplots(1, 3, figsize=(16, 6))
    #     cmap = ListedColormap(["black", "white", "white", "grey"])  # 0=clear, 1/2=cloud, 3=shadow
            
    #     # Panel 1: True color
    #     ax[0].imshow(rgb_array)
    #     ax[0].set_title("True Colour")

    #     # Panel 2: True color + contours
    #     ax[1].imshow(rgb_array)
        
    #     # Add contours based on detection mode
    #     if detection_mode in [DetectionMode.CLOUD_ONLY, DetectionMode.BOTH]:
    #         ax[1].contour((mask == 1) | (mask == 2), colors="blue", linewidths=1.5)
        
    #     if detection_mode in [DetectionMode.SHADOW_ONLY, DetectionMode.BOTH]:
    #         ax[1].contour(mask == 3, colors="red", linewidths=1.5)
        
    #     ax[1].set_title("True Colour with\nPrediction Contours")

    #     # Panel 3: Mask visualization
    #     ax[2].imshow(mask, vmin=0, vmax=3, cmap=cmap, interpolation="nearest")
    #     ax[2].set_title("Prediction Mask")

    #     # Remove ticks
    #     for a in ax:
    #         a.set_xticks([])
    #         a.set_yticks([])

    #     # Add legend based on detection mode
    #     legend_labels = []
    #     legend_colors = []
        
    #     legend_labels.append("Clear")
    #     legend_colors.append("black")
        
    #     if detection_mode in [DetectionMode.CLOUD_ONLY, DetectionMode.BOTH]:
    #         legend_labels.append("Cloud")
    #         legend_colors.append("white")
        
    #     if detection_mode in [DetectionMode.SHADOW_ONLY, DetectionMode.BOTH]:
    #         legend_labels.append("Shadow")
    #         legend_colors.append("grey")
        
    #     patches = [
    #         mpatches.Patch(facecolor=legend_colors[i], edgecolor="black", label=legend_labels[i])
    #         for i in range(len(legend_labels))
    #     ]
    #     fig.legend(handles=patches, loc="lower right", bbox_to_anchor=(0.904, 0.05))

    #     plt.tight_layout()
    #     plt.savefig(output_path, dpi=300, bbox_inches="tight")
    #     plt.close(fig)
    
    
    def create_visualization(self, rgb_array: np.ndarray, mask: np.ndarray,
                         output_path: Path, detection_mode: DetectionMode) -> None:
        """Create and save visualization image with minimal memory usage using OpenCV + Pillow."""

        # Ensure RGB format (uint8)
        if rgb_array.dtype != np.uint8:
            rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)

        h, w, _ = rgb_array.shape

        # Panel 1: True color
        panel1 = rgb_array.copy()

        # Panel 2: True color + contours
        panel2 = rgb_array.copy()
        
        new_mask = mask.copy()

        if detection_mode in [DetectionMode.CLOUD_ONLY, DetectionMode.BOTH]:
            cloud_mask = ((mask == 1) | (mask == 2)).astype(np.uint8)
            new_mask = cloud_mask.copy()
            new_mask[new_mask == 1] = 2  # Retain cloud areas if both
            contours, _ = cv2.findContours(cloud_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(panel2, contours, -1, (0, 0, 255), 1)  # Blue contours (BGR)

        if detection_mode in [DetectionMode.SHADOW_ONLY, DetectionMode.BOTH]:
            shadow_mask = (mask == 3).astype(np.uint8)
            new_mask = shadow_mask.copy()
            new_mask[new_mask == 1] = 3  # Retain cloud areas if both
            contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(panel2, contours, -1, (255, 0, 0), 1)  # Red contours (BGR)

        # Panel 3: Mask visualization
        cmap = {
            0: (0, 0, 0),      # Clear = black
            1: (255, 255, 255),# Cloud = white
            2: (255, 255, 255),# Cloud = white
            3: (128, 128, 128) # Shadow = grey
        }
        panel3 = np.zeros((h, w, 3), dtype=np.uint8)
        for k, v in cmap.items():
            panel3[new_mask == k] = v

        # Combine panels horizontally
        combined = np.concatenate([panel1, panel2, panel3], axis=1)

        # Convert to PIL for titles + legend
        # img_pil = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        img_pil = Image.fromarray(combined)
        draw = ImageDraw.Draw(img_pil)

        # Titles
        titles = ["True Colour", "True Colour + Prediction", "Prediction Mask"]
        step = w
        for i, title in enumerate(titles):
            draw.text((step * i + 10, 10), title, fill="yellow")

        # Legend
        legend_y = h - 30
        legends = [("Clear", "black")]
        if detection_mode in [DetectionMode.CLOUD_ONLY, DetectionMode.BOTH]:
            legends.append(("Cloud", "white"))
        if detection_mode in [DetectionMode.SHADOW_ONLY, DetectionMode.BOTH]:
            legends.append(("Shadow", "grey"))

        base_x = 2 * w + 40
        for i, (label, color) in enumerate(legends):
            x0 = base_x + i * 120
            # x0 = 20 + i * 120
            draw.rectangle([x0, legend_y, x0 + 20, legend_y + 20], fill=color, outline="#2525B9", width=2)
            draw.text((x0 + 30, legend_y), label, fill="yellow")

        # Save final image
        img_pil.save(output_path, format="PNG", optimize=True)


class AnalysisReporter:
    """Handles analysis reporting and logging functionality."""
    
    def __init__(self, file_manager: FileManager):
        """Initialize reporter with file manager."""
        self.file_manager = file_manager
        self.detection_mode = file_manager.detection_mode
        
        # Set CSV fieldnames based on detection mode
        self._set_csv_fieldnames()
        self._initialize_csv()
        self._setup_logging()
    
    def _set_csv_fieldnames(self) -> None:
        """Set CSV field names based on detection mode."""
        if self.detection_mode == DetectionMode.CLOUD_ONLY:
            self.csv_fieldnames = [
                "Image", "cloud", "Cloud_Area_sqm", "Cloud_Probability", 
                "Processing_Time_s", "Status", "Error"
            ]
        elif self.detection_mode == DetectionMode.SHADOW_ONLY:
            self.csv_fieldnames = [
                "Image", "cloud_shadow", "Shadow_Area_sqm", 
                "Processing_Time_s", "Status", "Error"
            ]
        else:  # BOTH
            self.csv_fieldnames = [
                "Image", "cloud", "Cloud_Area_sqm", "Cloud_Probability", 
                "cloud_shadow", "Shadow_Area_sqm", "Processing_Time_s", "Status", "Error"
            ]
    
    def _initialize_csv(self) -> None:
        """Initialize CSV report file."""
        with open(self.file_manager.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writeheader()
            
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        self.logger = logging.getLogger('CloudShadowAnalysis')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(self.file_manager.log_path)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def log_result(self, result: ProcessingResult) -> None:
        """Log processing result to CSV and logger."""
        # Create CSV row based on detection mode
        row = {"Image": Path(result.image_name).stem}
        
        if self.detection_mode == DetectionMode.CLOUD_ONLY:
            row.update({
                "cloud": result.has_cloud,
                "Cloud_Area_sqm": f"{result.cloud_area_sqm:.2f}",
                "Cloud_Probability": f"{result.cloud_probability:.4f}",
                "Processing_Time_s": f"{result.processing_time:.2f}",
                "Status": result.status.value,
                "Error": result.error_message or ""
            })
        elif self.detection_mode == DetectionMode.SHADOW_ONLY:
            row.update({
                "cloud_shadow": result.has_shadow,
                "Shadow_Area_sqm": f"{result.shadow_area_sqm:.2f}",
                "Processing_Time_s": f"{result.processing_time:.2f}",
                "Status": result.status.value,
                "Error": result.error_message or ""
            })
        else:  # BOTH
            row.update({
                "cloud": result.has_cloud,
                "Cloud_Area_sqm": f"{result.cloud_area_sqm:.2f}",
                "Cloud_Probability": f"{result.cloud_probability:.4f}",
                "cloud_shadow": result.has_shadow,
                "Shadow_Area_sqm": f"{result.shadow_area_sqm:.2f}",
                "Processing_Time_s": f"{result.processing_time:.2f}",
                "Status": result.status.value,
                "Error": result.error_message or ""
            })
        
        # Write to CSV
        with open(self.file_manager.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writerow(row)
        
        # Logger entry
        if result.status == ProcessingStatus.SUCCESS:
            log_msg = f"Processed: {result.image_name} | Time: {result.processing_time:.2f}s"
            if self.detection_mode in [DetectionMode.CLOUD_ONLY, DetectionMode.BOTH]:
                log_msg += f" | Cloud: {result.has_cloud}"
            if self.detection_mode in [DetectionMode.SHADOW_ONLY, DetectionMode.BOTH]:
                log_msg += f" | Shadow: {result.has_shadow}"
            self.logger.info(log_msg)
        else:
            self.logger.error(f"Failed: {result.image_name} | Error: {result.error_message}")


class CloudShadowProcessor:
    """Main processor class orchestrating the entire detection pipeline."""
    
    def __init__(self, output_folder: Optional[str] = None,
                 detection_mode: DetectionMode = DetectionMode.BOTH,
                 scale_factor: int = 20, use_gan_nir: bool = True,
                 save_visualizations: bool = True,
                 img = None, img_name = None, mask = None,
                 resolution = (1, 1)):
        """
        Initialize the main processor.
        
        Args:
            output_folder: Path to output directory
            detection_mode: Detection mode (cloud/shadow/both)
            scale_factor: Image scaling factor
            use_gan_nir: Use GAN for NIR generation
            save_visualizations: Whether to save visualization plots
        """
        self.detection_mode = detection_mode
        self.file_manager = FileManager(output_folder, detection_mode)
        self.detector = CloudShadowDetector(detection_mode, scale_factor, use_gan_nir)
        self.reporter = AnalysisReporter(self.file_manager)
        self.save_visualizations = save_visualizations
        self.img = img
        self.img_name = img_name
        self.mask = mask
        self.resolution = resolution
        
        # Log initialization
        self.reporter.logger.info("CloudShadowProcessor initialized")
        self.reporter.logger.info(f"Detection mode: {detection_mode.value}")
        self.reporter.logger.info(f"Output folder: {self.file_manager.output_folder}")
        self.reporter.logger.info(f"CSV file: {self.file_manager.csv_path}")
        self.reporter.logger.info(f"CloudMask version: {omnicloudmask.__version__}")
        self.reporter.logger.info(f"Using GAN NIR: {self.detector.use_gan_nir}")
    
    def process_single_image(self) -> ProcessingResult:
        """Process a single image."""
        start_time = time.time()
        try:
            # Perform detection
            (has_cloud, cloud_area_sqm, cloud_probability, 
             has_shadow, shadow_area_sqm), mask = self.detector.detect_from_array(self.img, self.resolution)
            self.mask = mask
            processing_time = time.time() - start_time
            return ProcessingResult(
                image_name=self.img_name,
                has_cloud=has_cloud,
                cloud_area_sqm=cloud_area_sqm,
                cloud_probability=cloud_probability,
                has_shadow=has_shadow,
                shadow_area_sqm=shadow_area_sqm,
                processing_time=processing_time,
                status=ProcessingStatus.SUCCESS
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessingResult(
                image_name=self.img_name,
                has_cloud=False,
                cloud_area_sqm=0.0,
                cloud_probability=0.0,
                has_shadow=False,
                shadow_area_sqm=0.0,
                processing_time=processing_time,
                status=ProcessingStatus.ERROR,
                error_message=str(e)
            )
            
    def load_img(self, img_path):
        """Load image from path."""
        self.img_name = img_path.name
        # # Read image
        # with rio.open(img_path) as src:
        #     rgb_array = src.read().transpose(1, 2, 0)
        #     self.img = rgb_array
        #     self.resolution = src.res
        
        rgb_array = io_factory.read(img_path)
        metadata = io_factory.metadata(img_path)
        
        self.img = rgb_array
        self.resolution = metadata.get("resolution", (1.0, 1.0))
        
        # print(metadata)
            
    def save_mask_result(self, result: ProcessingResult, mask: np.ndarray) -> None:
        """Save detection results and visualizations."""
        output_path = self.file_manager.get_output_path(result.has_cloud, result.has_shadow)
        output_path.mkdir(parents=True, exist_ok=True)        
        
        # Save masking visualization
        timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
        image_stem = Path(self.img_name).stem
        
        if self.save_visualizations:
            vis_path = output_path / f"{timestamp}_{image_stem}.jpg"
            self.detector.create_visualization(self.img, self.mask, vis_path, self.detection_mode)
    
    def run(self) -> ProcessingResult:
        """Execute the complete processing pipeline."""
        result = self.process_single_image()
        self.reporter.log_result(result)
        return result


if __name__ == "__main__":
    """Main entry point for command line execution."""
    
    input_dir = r"\\192.168.2.80\d\Projects\QI47\2025_Projects\Image_QC\1_Data\all_real_samples\shadow\For Processing\199001142_0386_01_0556_P00_01.iiq"
    output_dir = r"\\192.168.2.80\d\Projects\QI47\2025_Projects\Image_QC\1_Data\OmniCloudMask_Output\SampleTest11"
    
    detecting_mode = 'shadow'  # Options: 'cloud', 'shadow', 'both'
    scaling_factor = 20
    use_gan_nir = True
    save_visualizations = True
    
    # Create processor
    processor = CloudShadowProcessor(
        output_folder=output_dir,
        detection_mode=DetectionMode(detecting_mode),
        scale_factor=scaling_factor,
        use_gan_nir=use_gan_nir,    
        save_visualizations=save_visualizations
    )
    
    # Load image
    processor.load_img(Path(input_dir))
    
    # Run processing
    result = processor.run()
    # Save results
    processor.save_mask_result(result, processor.mask)
    
    print(f"\nProcessing completed successfully!")
    print(f"Detection mode: {detecting_mode}")
    print(f"Results saved to: {processor.file_manager.output_folder}")
    print(f"CSV report: {processor.file_manager.csv_path}")
    print(f"Log file: {processor.file_manager.log_path}")