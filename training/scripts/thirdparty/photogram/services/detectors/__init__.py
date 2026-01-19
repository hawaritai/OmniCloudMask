"""
Import all detector modules to ensure they are registered with the factory.
"""

# Import existing detectors
from .blur_detector import BlurDetectionSystem

# Import new cloud/shadow detectors
from .cloud_detector import CloudDetectionSystem  
from .cloud_shadow_detector import CloudShadowDetectionSystem

# Make them available for direct import if needed
__all__ = [
    'BlurDetectionSystem',
    'CloudDetectionSystem', 
    'CloudShadowDetectionSystem'
]
