# services/detector_base.py

from abc import ABC, abstractmethod
from typing import Any

class DetectorBase(ABC):
    """Base class for all detectors"""
    
    def __init__(self, threshold: float, **kwargs):
        self.threshold = threshold
        # Store any additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    @abstractmethod
    def load_image(self, image_path: str) -> None:
        """Load image for processing"""
        pass
        
    @abstractmethod
    def run(self) -> Any:
        """Run the detection and return results"""
        pass