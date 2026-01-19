# photogram/entities/photo_dataset.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterable, Tuple
from .camera_io import CameraIO
from .camera_eo import CameraEO
from .image_record import ImageRecord
from .shot import Shot

@dataclass
class PhotoDataset:
    """Manages a collection of cameras, images, and shots."""
    cameras: Dict[str, CameraIO] = field(default_factory=dict)    # camera_id -> CameraIO
    images: Dict[str, ImageRecord] = field(default_factory=dict)  # image_id  -> ImageRecord
    shots: List[Shot] = field(default_factory=list)

    # derived indices (built from shots)
    _by_camera: Dict[str, List[str]] = field(default_factory=dict, init=False, repr=False)  # cam -> [image_ids]
    _img2cam: Dict[str, str] = field(default_factory=dict, init=False, repr=False)          # image_id -> cam_id

    # optional lazy hooks set by dataset_builder
    _meta_loader: Optional[callable] = field(default=None, init=False, repr=False)
    _open_image: Optional[callable] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize indices after object creation."""
        self.reindex()

    # ---------- Indexing ----------
    def reindex(self) -> None:
        """Rebuild internal indices mapping cameras to images."""
        by_cam: Dict[str, List[str]] = {cid: [] for cid in self.cameras.keys()}
        img2cam: Dict[str, str] = {}
        valid_imgs = set(self.images.keys())
        valid_cams = set(self.cameras.keys())

        for s in self.shots:
            if s.image_id in valid_imgs and s.camera_id in valid_cams:
                by_cam.setdefault(s.camera_id, []).append(s.image_id)
                img2cam[s.image_id] = s.camera_id

        self._by_camera = {k: sorted(v) for k, v in by_cam.items()}
        self._img2cam = img2cam

    # ---------- queries ----------

    def get_camera_list(self) -> List[CameraIO]:
        """Return all cameras in the dataset."""
        return list(self.cameras.values())

    def get_shots_list(self, camera_id: str) -> List[Shot]:
        """Return all shots for the given camera_id."""
        return [s for s in self.shots if s.camera_id == camera_id]

    def get_eo_for_image(self, image_id: str) -> Optional[CameraEO]:
        """Return the EO for a given image_id, or None if not found."""
        shot = next((s for s in self.shots if s.image_id == image_id), None)
        return shot.eo if shot else None
    

    # def shots_by_camera(self, camera_id: str) -> List[Shot]:
    #     """Return all shots taken with a given camera."""
    #     return [s for s in self.shots if s.camera_id == camera_id]
    
    # def get_shot_by_image(self, image_id: str) -> Optional[Shot]:
    #     return next((s for s in self.shots if s.image_id == image_id), None)

    def images_by_camera(self, camera_id: str) -> List[ImageRecord]:
        if not self._by_camera:
            self.reindex()
        return [self.images[i] for i in self._by_camera.get(camera_id, [])]

    def iter_by_camera(self) -> Iterable[Tuple[str, CameraIO, List[ImageRecord]]]:
        """Yield (camera_id, CameraIO, [ImageRecord]) for each camera."""
        if not self._by_camera:
            self.reindex()
        for cam_id, cam in self.cameras.items():
            yield cam_id, cam, [self.images[i] for i in self._by_camera.get(cam_id, [])]

    # def eos_by_camera(self, camera_id: str) -> List[CameraEO]:
    #     """Return all CameraEOs for a given camera."""
    #     return [s.eo for s in self.shots if s.camera_id == camera_id]

    # def eo_for_image(self, image_id: str) -> Optional[CameraEO]:
    #     """Return the EO for a specific image_id (if present)."""
    #     s = self.get_shot_by_image(image_id)
    #     return s.eo if s else None
    
    # ---------- maintenance ----------
    def ensure_consistency(
        self, drop_orphan_shots: bool = True, drop_orphan_images: bool = False
    ) -> Dict[str, int]:
        """Ensure dataset consistency by removing orphaned shots or images.

        Args:
            drop_orphan_shots: If True, remove shots with invalid image or camera IDs.
            drop_orphan_images: If True, remove images not referenced by any shot.

        Returns:
            Dict with counts of orphan/removed shots and images.
        """
        rep = dict(orphan_shots=0, removed_shots=0, orphan_images=0, removed_images=0)

        valid_imgs, valid_cams = set(self.images), set(self.cameras)

        new_shots: List[Shot] = []
        for s in self.shots:
            ok = (s.image_id in valid_imgs) and (s.camera_id in valid_cams)
            if not ok:
                rep["orphan_shots"] += 1
                if drop_orphan_shots:
                    rep["removed_shots"] += 1
                    continue
            new_shots.append(s)
        self.shots = new_shots

        referenced = {s.image_id for s in self.shots}
        for img_id in list(self.images.keys()):
            if img_id not in referenced:
                rep["orphan_images"] += 1
                if drop_orphan_images:
                    self.images.pop(img_id, None)
                    rep["removed_images"] += 1

        self.reindex()
        return rep
