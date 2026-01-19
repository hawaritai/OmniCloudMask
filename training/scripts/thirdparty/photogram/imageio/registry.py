from __future__ import annotations
from typing import List, Type
from .interfaces import ImageReader

_READERS: List[Type[ImageReader]] = []

def register_reader(cls: Type[ImageReader]) -> Type[ImageReader]:
    _READERS.append(cls)
    return cls

def get_readers() -> List[Type[ImageReader]]:
    return list(_READERS)

def clear_readers() -> None:
    _READERS.clear()
