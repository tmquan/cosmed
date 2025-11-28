"""
Custom MONAI transforms for Cosmed.
"""

from typing import Dict, Hashable, Mapping

import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform


class TransposeDict(MapTransform):
    """Transpose image tensors (swap H and W dimensions)."""
    
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
    
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            # Transpose last two dimensions (H, W)
            if isinstance(img, torch.Tensor):
                d[key] = img.transpose(-2, -1)
            elif isinstance(img, np.ndarray):
                d[key] = np.swapaxes(img, -2, -1)
            else:
                # Try to convert to tensor first
                d[key] = torch.as_tensor(img).transpose(-2, -1)
        return d


class ClipMinIntensityDict(MapTransform):
    """
    Dictionary-based wrapper of ClipMinIntensity.
    Clips intensity values below a minimum value.
    
    Args:
        keys: Keys to apply the transform to
        min_val: Minimum value to clip to (default: -512 for CT HU values)
    """
    
    def __init__(
        self, 
        keys: KeysCollection, 
        min_val: float = -512,
        allow_missing_keys: bool = False
    ):
        super().__init__(keys, allow_missing_keys)
        self.min_val = min_val
    
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = torch.clamp(d[key], min=self.min_val)
        return d

