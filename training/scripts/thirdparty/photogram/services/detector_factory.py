# services/detector_factory.py

from typing import Dict, Type, Any
from .detector_base import DetectorBase

class DetectorFactory:
    """Auto-registering detector factory using decorators"""
    
    _registry: Dict[str, Type[DetectorBase]] = {}
    
    @classmethod
    def register_detector(cls, detector_type: str):
        """Decorator to auto-register detector classes"""
        def decorator(detector_class: Type[DetectorBase]):
            if not issubclass(detector_class, DetectorBase):
                raise TypeError(f"Detector class {detector_class.__name__} must inherit from DetectorBase")
            
            cls._registry[detector_type] = detector_class
            return detector_class
        return decorator
    
    @classmethod
    def create(cls, detector_type: str, **kwargs) -> DetectorBase:
        """Create a detector instance"""
        if detector_type not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(f"Unknown detector type: '{detector_type}'. Available: {available}")
        
        return cls._registry[detector_type](**kwargs)
    
    @classmethod
    def get_available_types(cls) -> list[str]:
        """Get list of available detector types"""
        return list(cls._registry.keys())