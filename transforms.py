import torch
from monai.transforms import MapTransform


class ClipMinIntensityDict(MapTransform):
    """Clip intensity values to a minimum threshold, leave max unbounded."""

    def __init__(self, keys, min_val: float = -512):
        super().__init__(keys)
        self.min_val = min_val

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = torch.clamp(d[key], min=self.min_val)
        return d

