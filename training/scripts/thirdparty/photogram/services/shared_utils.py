# services/shared_utils.py
import csv
import json
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, asdict, fields
import numpy as np

# -------------------------------------------------
# ENUMS
# -------------------------------------------------
class DetectionMode(Enum):
    CLOUD_ONLY = "cloud"
    SHADOW_ONLY = "cloud_shadow"
    BOTH = "both"
    BLUR = "blur"


class ProcessingStatus(Enum):
    SUCCESS = "Success"
    ERROR = "Error"
    SKIPPED = "Skipped"


# -------------------------------------------------
# RESULT DATACLASS (shared by all detectors)
# -------------------------------------------------
@dataclass
class ProcessingResult:
    """Unified result for cloud, shadow, and blur detection."""
    image_name: str
    has_cloud: bool = False
    cloud_area_sqm: float = 0.0
    cloud_probability: float = 0.0
    has_shadow: bool = False
    shadow_area_sqm: float = 0.0
    shadow_probability: float = 0.0
    has_blur: bool = False
    blur_score: float = 0.0
    processing_time: float = 0.0
    status: ProcessingStatus = ProcessingStatus.SUCCESS
    error_message: Optional[str] = None


# -------------------------------------------------
# FILE MANAGER
# -------------------------------------------------
class FileManager:
    SUPPORTED_EXTENSIONS = ['.iiq', '.tif', '.tiff']

    def __init__(self,
                 output_folder: Optional[str] = None,
                 detection_mode: DetectionMode = DetectionMode.BOTH,
                 job_id: Optional[str] = None):
        self.detection_mode = detection_mode
        self.job_id = job_id.strip() if job_id else None
        self.output_folder = Path(output_folder) if output_folder else Path("Results")
        self._setup_output_structure()

    def _setup_output_structure(self) -> None:
        folders = {
            DetectionMode.CLOUD_ONLY: ["clouds"],
            DetectionMode.SHADOW_ONLY: ["cloud_shadow"],
            DetectionMode.BLUR: ["blur"],
            DetectionMode.BOTH: ["clouds", "cloud_shadow"]
        }
        for sub in folders.get(self.detection_mode, ["results"]):
            (self.output_folder / sub).mkdir(parents=True, exist_ok=True)

    def get_output_path(self, has_cloud: bool = False, has_shadow: bool = False, mode: Optional[DetectionMode] = None) -> Path:
        target_mode = mode if mode else self.detection_mode
        
        if target_mode == DetectionMode.CLOUD_ONLY:
            return self.output_folder / "clouds"
        if target_mode == DetectionMode.SHADOW_ONLY:
            return self.output_folder / "cloud_shadow"
        if target_mode == DetectionMode.BLUR:
            return self.output_folder / "blur"
            
        # For BOTH or fallback - never return cloud_shadow
        if has_shadow:
            return self.output_folder / "cloud_shadow"
        return self.output_folder / "clouds"

    def _folder(self) -> str:
        return {
            DetectionMode.CLOUD_ONLY: "clouds",
            DetectionMode.SHADOW_ONLY: "cloud_shadow",
            DetectionMode.BLUR: "blur",
            DetectionMode.BOTH: "cloud_and_shadow"
        }[self.detection_mode]

    def _base_name(self) -> str:
        return {
            DetectionMode.CLOUD_ONLY: "ImgQC_results_cloud",
            DetectionMode.SHADOW_ONLY: "ImgQC_results_cloud_shadow",
            DetectionMode.BLUR: "ImgQC_results_blur",
            DetectionMode.BOTH: "ImgQC_results_cloud_and_shadow"
        }[self.detection_mode]

    @property
    def csv_path(self) -> Path:
        prefix = f"{self.job_id}_" if self.job_id else ""
        # prefix = ""
        return self.output_folder / self._folder() / f"{prefix}{self._base_name()}.csv"

    @property
    def json_path(self) -> Path:
        prefix = f"{self.job_id}_" if self.job_id else ""
        return self.output_folder / self._folder() / f"{prefix}{self._base_name()}.json"

    @property
    def log_path(self) -> Path:
        ts = time.strftime('%Y%m%d_%H%M%S')
        prefix = f"{self.job_id}_" if self.job_id else ""
        return self.output_folder / f"{prefix}ImgQC_{self.detection_mode.value}_{ts}.log"


# -------------------------------------------------
# ANALYSIS REPORTER (CSV + JSON + LOG)
# -------------------------------------------------
class AnalysisReporter:
    def __init__(self,
                 file_manager: FileManager,
                 export_csv: bool = True,
                 export_json: bool = True):
        self.file_manager = file_manager
        self.export_csv = export_csv
        self.export_json = export_json
        self.detection_mode = file_manager.detection_mode
        self._define_csv_fields()
        self._initialize_outputs()
        self._setup_logging()

    def _define_csv_fields(self) -> None:
        base = ["Image", "Processing_Time_s", "Status", "Error"]
        if self.detection_mode == DetectionMode.CLOUD_ONLY:
            self.csv_fieldnames = base[:1] + [
                "cloud", "Cloud_Area_sqm", "Cloud_Probability"
            ] + base[1:]
        elif self.detection_mode == DetectionMode.SHADOW_ONLY:
            self.csv_fieldnames = base[:1] + [
                "cloud_shadow", "Shadow_Area_sqm", "Shadow_Probability"
            ] + base[1:]
        elif self.detection_mode == DetectionMode.BLUR:
            self.csv_fieldnames = base[:1] + ["blur", "blur_score"] + base[1:]
        else:  # BOTH
            self.csv_fieldnames = base[:1] + [
                "cloud", "Cloud_Area_sqm", "Cloud_Probability",
                "cloud_shadow", "Shadow_Area_sqm", "Shadow_Probability"
            ] + base[1:]

    def _initialize_outputs(self) -> None:
        if self.export_csv:
            with open(self.file_manager.csv_path, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=self.csv_fieldnames).writeheader()
        if self.export_json:
            with open(self.file_manager.json_path, 'w') as f:
                json.dump([], f)

    def _setup_logging(self) -> None:
        """Setup detector-specific logging with its own file + propagation to root."""
        # Get detector-specific logger
        self.logger = logging.getLogger(f"detector.{self.detection_mode.value}")
        self.logger.setLevel(logging.DEBUG)

        # ====================================================================
        # ADD DETECTOR-SPECIFIC FILE HANDLER (one per detector type)
        # ====================================================================
        detector_log_path = self.file_manager.log_path
        
        # Avoid duplicate handlers
        file_handler_exists = any(
            isinstance(h, logging.FileHandler) and h.baseFilename == str(detector_log_path)
            for h in self.logger.handlers
        )
        
        if not file_handler_exists:
            # Create detector-specific file handler
            fh = logging.FileHandler(detector_log_path, encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(fh)
            self.logger.info(f"Detector-specific log file: {detector_log_path}")

        # ====================================================================
        # ENABLE PROPAGATION to root logger (for master log + console)
        # ====================================================================
        # This allows logs to go to:
        #   1. Detector-specific file (via handler above)
        #   2. Master log file (via root logger's file handler)
        #   3. Console (via root logger's console handler, respecting --verbose)
        self.logger.propagate = True
        
        # Silence verbose third-party libraries
        logging.getLogger("rasterio").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)
                    
    def _build_row(self, result: ProcessingResult) -> Dict:
        """Build CSV row from ProcessingResult with proper image name."""
        row = {
            "Image": result.image_name,  # ← FIX: Add image name first
            "Processing_Time_s": f"{result.processing_time:.2f}",
            "Status": result.status.value,
            "Error": result.error_message or ""
        }
        
        if self.detection_mode == DetectionMode.CLOUD_ONLY:
            row.update({
                "cloud": result.has_cloud,
                "Cloud_Area_sqm": f"{result.cloud_area_sqm:.2f}",
                "Cloud_Probability": f"{result.cloud_probability:.4f}"
            })
        elif self.detection_mode == DetectionMode.SHADOW_ONLY:
            row.update({
                "cloud_shadow": result.has_shadow,
                "Shadow_Area_sqm": f"{result.shadow_area_sqm:.2f}",
                "Shadow_Probability": f"{result.shadow_probability:.4f}"
            })
        elif self.detection_mode == DetectionMode.BLUR:
            row.update({
                "blur": result.has_blur,
                "blur_score": f"{result.blur_score:.2f}"
            })
        else:  # BOTH
            row.update({
                "cloud": result.has_cloud,
                "Cloud_Area_sqm": f"{result.cloud_area_sqm:.2f}",
                "Cloud_Probability": f"{result.cloud_probability:.4f}",
                "cloud_shadow": result.has_shadow,
                "Shadow_Area_sqm": f"{result.shadow_area_sqm:.2f}",
                "Shadow_Probability": f"{result.shadow_probability:.4f}"
            })
        
        return row

    def log_result(self, result: ProcessingResult) -> None:
        row = self._build_row(result)

        # --- CSV ---
        if self.export_csv:
            with open(self.file_manager.csv_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=self.csv_fieldnames).writerow(row)

        # --- JSON ---
        # Fix: Handle empty/invalid JSON file safely
        if self.export_json:
            json_row = {k: (str(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in row.items()}
            path = self.file_manager.json_path
            if path.exists():
                try:
                    with open(path, 'r+', encoding='utf-8') as f:
                        data = json.load(f)
                        data.append(json_row)
                        f.seek(0)
                        json.dump(data, f, indent=2)
                        f.truncate()
                except json.JSONDecodeError:
                    # Corrupted/empty file: rewrite as fresh list
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump([json_row], f, indent=2)
            else:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump([json_row], f, indent=2)

        # --- LOG ---
        if result.status == ProcessingStatus.SUCCESS:
            msg = f"Processed: {result.image_name} | Time: {result.processing_time:.2f}s"
            if self.detection_mode in (DetectionMode.CLOUD_ONLY, DetectionMode.BOTH):
                msg += f" | Cloud: {result.has_cloud}"
            if self.detection_mode in (DetectionMode.SHADOW_ONLY, DetectionMode.BOTH):
                msg += f" | Shadow: {result.has_shadow}"
            if self.detection_mode == DetectionMode.BLUR:
                msg += f" | Blur: {result.has_blur} (score {result.blur_score:.2f})"
            self.logger.debug(msg)
        else:
            self.logger.error(f"Failed: {result.image_name} | {result.error_message}")