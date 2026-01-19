# services/detectors/cloud_detector.py

"""
Cloud detection system using factory pattern
"""

from typing import Optional
from ..detector_factory import DetectorFactory
from ..shared_cloud_detection import CloudShadowDetectorBase, DetectionMode


@DetectorFactory.register_detector("cloud")
class CloudDetectionSystem(CloudShadowDetectorBase):
    """
    Cloud detection system compatible with factory pattern.
    
    Inherits common logic from CloudShadowDetectorBase and specifies CLOUD_ONLY mode.
    """
    
    def __init__(self, threshold: float = 0.01, output_dir: Optional[str] = None, 
                 scale_factor: int = 3, use_gan_nir: bool = True, 
                 save_visualizations: bool = True, **kwargs):
        """
        Initialize cloud detection system by passing CLOUD_ONLY mode to base.
        
        Args:
            threshold: Cloud probability threshold (0-1)
            output_dir: Directory for saving outputs
            scale_factor: Image scaling factor for processing
            use_gan_nir: Use GAN for NIR generation
            save_visualizations: Whether to save visualization plots
            **kwargs: Additional configuration parameters
        """
        super().__init__(DetectionMode.CLOUD_ONLY, threshold=threshold, output_dir=output_dir, 
                         scale_factor=scale_factor, use_gan_nir=use_gan_nir, 
                         save_visualizations=save_visualizations, **kwargs)