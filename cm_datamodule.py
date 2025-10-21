import os
import glob
from typing import Callable, Optional, Sequence, List, Dict
from argparse import ArgumentParser

from pytorch_lightning import seed_everything
seed = seed_everything(21, workers=True)

from monai.data import CacheDataset, ThreadDataLoader
from monai.data import list_data_collate
from monai.utils import set_determinism
from monai.transforms import (
    apply_transform,
    Randomizable,
    Compose,
    EnsureChannelFirstDict,
    LoadImageDict,
    SpacingDict,
    OrientationDict,
    DivisiblePadDict,
    CropForegroundDict,
    ResizeDict,
    Rotate90Dict,
    TransposeDict,
    RandFlipDict,
    RandZoomDict,
    RandShiftIntensityDict,
    RandStdShiftIntensityDict,
    ZoomDict,
    RandRotateDict,
    RandGaussianNoiseDict,
    HistogramNormalizeDict,
    ScaleIntensityDict,
    ScaleIntensityRangeDict,
    ToTensorDict,
    MapTransform,
)
import torch

from pytorch_lightning import LightningDataModule


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


class UnpairedDataset(CacheDataset, Randomizable):
    def __init__(
        self,
        keys: Sequence,
        data: Sequence,
        transform: Optional[Callable] = None,
        length: Optional[Callable] = None,
        batch_size: int = 32,
        is_training: bool = True,
    ) -> None:
        self.keys = keys
        self.data = data
        self.length = length
        self.batch_size = batch_size
        self.transform = transform
        self.is_training = is_training

    def __len__(self) -> int:
        if self.length is None:
            return min((len(dataset) for dataset in self.data))
        else:
            return self.length

    def _transform(self, index: int):
        data = {}
        self.R.seed(index)
        for key, dataset in zip(self.keys, self.data):
            if self.is_training:
                rand_idx = self.R.randint(0, len(dataset))
                data[key] = dataset[rand_idx]
                data[f"{key}_idx"] = rand_idx
            else:
                data[key] = dataset[index]
                data[f"{key}_idx"] = index
            data[f"{key}_pth"] = data[key]

        if self.transform is not None:
            data = apply_transform(self.transform, data)

        return data


class UnpairedDataModule(LightningDataModule):
    def __init__(
        self,
        train_image3d_folders: str = "path/to/folder",
        train_image2d_folders: str = "path/to/folder",
        val_image3d_folders: str = "path/to/folder",
        val_image2d_folders: str = "path/to/folder",
        test_image3d_folders: str = "path/to/folder",
        test_image2d_folders: str = "path/to/dir",
        train_samples: int = 1000,
        val_samples: int = 400,
        test_samples: int | None = None,
        img_shape: int = 512,
        vol_shape: int = 256,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.img_shape = img_shape
        self.vol_shape = vol_shape
        self.num_workers = num_workers
        # self.setup()
        self.train_image3d_folders = train_image3d_folders
        self.train_image2d_folders = train_image2d_folders
        self.val_image3d_folders = val_image3d_folders
        self.val_image2d_folders = val_image2d_folders
        self.test_image3d_folders = test_image3d_folders
        self.test_image2d_folders = test_image2d_folders
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples

        # self.setup()
        def glob_files(folders: str = None, extension: str = "*.nii.gz"):
            assert folders is not None
            paths = [
                glob.glob(os.path.join(folder, extension), recursive=True)
                for folder in folders
            ]
            files = sorted([item for sublist in paths for item in sublist])
            print(len(files))
            print(files[:1])
            return files

        self.train_image3d_files = glob_files(
            folders=train_image3d_folders, extension="**/*.nii.gz"
        )
        self.train_image2d_files = glob_files(
            folders=train_image2d_folders, extension="**/*.png"
        )

        self.val_image3d_files = glob_files(
            folders=val_image3d_folders, extension="**/*.nii.gz"
        )  # TODO
        self.val_image2d_files = glob_files(
            folders=val_image2d_folders, extension="**/*.png"
        )

        self.test_image3d_files = glob_files(
            folders=test_image3d_folders, extension="**/*.nii.gz"
        )  # TODO
        self.test_image2d_files = glob_files(
            folders=test_image2d_folders, extension="**/*.png"
        )

    def setup(self, seed: int = 2222, stage: Optional[str] = None):
        set_determinism(seed=seed)

    def train_dataloader(self):
        self.train_transforms = Compose([
            LoadImageDict(keys=["image3d", "image2d"]),
            EnsureChannelFirstDict(keys=["image3d", "image2d"],),
            SpacingDict(keys=["image3d"], pixdim=(1.0, 1.0, 1.0), mode=["bilinear"], align_corners=True,),
            Rotate90Dict(keys=["image2d"], k=3),
            RandFlipDict(keys=["image2d"], prob=1.0, spatial_axis=1),
            OrientationDict(keys=("image3d"), axcodes="ASL"),
            ScaleIntensityDict(keys=["image2d"], minv=0.0, maxv=1.0,),
            HistogramNormalizeDict(keys=["image2d"], min=0.0, max=1.0,),
            ClipMinIntensityDict(keys=["image3d"], min_val=-512),
            ScaleIntensityDict(keys=["image3d"], minv=0.0, maxv=1.0),
            RandShiftIntensityDict(keys=["image3d"], prob=1.0, offsets=0.05, safe=True,),
            RandStdShiftIntensityDict(keys=["image3d"], prob=1.0, factors=0.05, nonzero=True,),
            RandZoomDict(keys=["image3d"], prob=1.0, min_zoom=0.65, max_zoom=1.15, padding_mode="constant", mode=["trilinear"], align_corners=True,),
            RandZoomDict(keys=["image2d"], prob=1.0, min_zoom=0.95, max_zoom=1.15, padding_mode="constant", mode=["area"],),
            CropForegroundDict(keys=["image2d"], source_key="image2d", select_fn=(lambda x: x > 0), margin=0,),
            ZoomDict(keys=["image3d"], zoom=0.95, padding_mode="constant", mode=["area"]),
            ZoomDict(keys=["image2d"], zoom=0.95, padding_mode="constant", mode=["area"]),
            ResizeDict(keys=["image3d"], spatial_size=self.vol_shape, size_mode="longest", mode=["trilinear"], align_corners=True,),
            ResizeDict(keys=["image2d"], spatial_size=self.img_shape, size_mode="longest", mode=["area"],),
            DivisiblePadDict(keys=["image3d"], k=self.vol_shape, mode="constant", constant_values=0,),
            DivisiblePadDict(keys=["image2d"], k=self.img_shape, mode="constant", constant_values=0,),
            ToTensorDict(keys=["image3d", "image2d"],),
        ])

        self.train_datasets = UnpairedDataset(
            keys=["image3d", "image2d"],
            data=[self.train_image3d_files, self.train_image2d_files],
            transform=self.train_transforms,
            length=self.train_samples,
            batch_size=self.batch_size,
            is_training=True,
        )

        self.train_loader = ThreadDataLoader(
            self.train_datasets,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=list_data_collate,
            shuffle=True,
        )
        return self.train_loader

    def val_dataloader(self):
        self.val_transforms = Compose([
            LoadImageDict(keys=["image3d", "image2d"]),
            EnsureChannelFirstDict(keys=["image3d", "image2d"],),
            SpacingDict(keys=["image3d"], pixdim=(1.0, 1.0, 1.0), mode=["bilinear"], align_corners=True,),
            Rotate90Dict(keys=["image2d"], k=3),
            RandFlipDict(keys=["image2d"], prob=1.0, spatial_axis=1),
            OrientationDict(keys=("image3d"), axcodes="ASL"),
            ScaleIntensityDict(keys=["image2d"], minv=0.0, maxv=1.0,),
            HistogramNormalizeDict(keys=["image2d"], min=0.0, max=1.0,),
            ClipMinIntensityDict(keys=["image3d"], min_val=-512),
            ScaleIntensityDict(keys=["image3d"], minv=0.0, maxv=1.0),
            CropForegroundDict(keys=["image2d"], source_key="image2d", select_fn=(lambda x: x > 0), margin=0,),
            ZoomDict(keys=["image3d"], zoom=0.95, padding_mode="constant", mode=["area"]),
            ZoomDict(keys=["image2d"], zoom=0.95, padding_mode="constant", mode=["area"]),
            ResizeDict(keys=["image3d"], spatial_size=self.vol_shape, size_mode="longest", mode=["trilinear"], align_corners=True,),
            ResizeDict(keys=["image2d"], spatial_size=self.img_shape, size_mode="longest", mode=["area"],),
            DivisiblePadDict(keys=["image3d"], k=self.vol_shape, mode="constant", constant_values=0,),
            DivisiblePadDict(keys=["image2d"], k=self.img_shape, mode="constant", constant_values=0,),
            ToTensorDict(keys=["image3d", "image2d"],),
        ])

        self.val_datasets = UnpairedDataset(
            keys=["image3d", "image2d"],
            data=[self.val_image3d_files, self.val_image2d_files],
            transform=self.val_transforms,
            length=self.val_samples,
            batch_size=self.batch_size,
            is_training=True,
        )

        self.val_loader = ThreadDataLoader(
            self.val_datasets,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=list_data_collate,
            shuffle=True,
        )
        return self.val_loader

    def test_dataloader(self):
        self.test_transforms = Compose([
            LoadImageDict(keys=["image3d", "image2d"]),
            EnsureChannelFirstDict(keys=["image3d", "image2d"],),
            SpacingDict(keys=["image3d"], pixdim=(1.0, 1.0, 1.0), mode=["bilinear"], align_corners=True,),
            Rotate90Dict(keys=["image2d"], k=3),
            RandFlipDict(keys=["image2d"], prob=1.0, spatial_axis=1),
            OrientationDict(keys=("image3d"), axcodes="ASL"),
            ScaleIntensityDict(keys=["image2d"], minv=0.0, maxv=1.0,),
            HistogramNormalizeDict(keys=["image2d"], min=0.0, max=1.0,),
            ClipMinIntensityDict(keys=["image3d"], min_val=-512),
            ScaleIntensityDict(keys=["image3d"], minv=0.0, maxv=1.0),
            CropForegroundDict(keys=["image2d"], source_key="image2d", select_fn=(lambda x: x > 0), margin=0,),
            ZoomDict(keys=["image3d"], zoom=0.95, padding_mode="constant", mode=["area"]),
            ZoomDict(keys=["image2d"], zoom=0.95, padding_mode="constant", mode=["area"]),
            ResizeDict(keys=["image3d"], spatial_size=self.vol_shape, size_mode="longest", mode=["trilinear"], align_corners=True,),
            ResizeDict(keys=["image2d"], spatial_size=self.img_shape, size_mode="longest", mode=["area"],),
            DivisiblePadDict(keys=["image3d"], k=self.vol_shape, mode="constant", constant_values=0,),
            DivisiblePadDict(keys=["image2d"], k=self.img_shape, mode="constant", constant_values=0,),
            ToTensorDict(keys=["image3d", "image2d"],),
        ])

        self.test_datasets = UnpairedDataset(
            keys=["image3d", "image2d"],
            data=[self.test_image3d_files, self.test_image2d_files],
            transform=self.test_transforms,
            length=self.test_samples,
            batch_size=self.batch_size,
            is_training=False,
        )

        self.test_loader = ThreadDataLoader(
            self.test_datasets,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=list_data_collate,
            shuffle=False,
        )
        return self.test_loader


def cache_paths_for_ct(project_root: str, ct_path: str) -> tuple[str, str, str, str]:
    """
    Generate cache paths for CT volume, video, image, and text prompt.
    
    Returns:
        Tuple of (volume_path, video_path, image_path, prompt_path)
    """
    import hashlib
    stem = hashlib.sha1(os.path.abspath(ct_path).encode("utf-8")).hexdigest()
    vol_dir = os.path.join(project_root, "cache", "vol")
    vid_dir = os.path.join(project_root, "cache", "vid")
    img_dir = os.path.join(project_root, "cache", "img")
    txt_dir = os.path.join(project_root, "cache", "txt")
    os.makedirs(vol_dir, exist_ok=True)
    os.makedirs(vid_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    return (
        os.path.join(vol_dir, f"{stem}.nii.gz"),
        os.path.join(vid_dir, f"{stem}.mp4"),
        os.path.join(img_dir, f"{stem}.png"),
        os.path.join(txt_dir, f"{stem}.txt"),
    )



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=2222)
    parser.add_argument("--datadir", type=str, default="/workspace/data", help="data directory")
    parser.add_argument("--img_shape", type=int, default=256, help="isotropic img shape")
    parser.add_argument("--vol_shape", type=int, default=256, help="isotropic vol shape")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")

    hparams = parser.parse_args()
    
    # Create data module with example data paths
    print("="*80)
    print("Testing CTMultiviewDataModule")
    print("="*80)
    
    train_ct_folders = [
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/NSCLC/processed/train/images"),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-0"),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-1"),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-2"),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-3"),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-4"),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/Imagenglab/processed/train/images"),
    ]

    train_xr_folders = [
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/VinDr/v1/processed/train/images/"),
    ]

    val_ct_folders = train_ct_folders
    val_xr_folders = [
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/VinDr/v1/processed/test/images/"),
    ]

    test_ct_folders = [
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/TCIA/images/"),
    ]
    test_xr_folders = [
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/VinDr/v1/processed/test/images/"),
    ]

    datamodule = UnpairedDataModule(
        train_image3d_folders=train_ct_folders,
        train_image2d_folders=train_xr_folders,
        val_image3d_folders=val_ct_folders,
        val_image2d_folders=val_xr_folders,
        test_image3d_folders=test_ct_folders,
        test_image2d_folders=test_xr_folders,
        train_samples=100,
        val_samples=20,
        test_samples=20,
        img_shape=hparams.img_shape,
        vol_shape=hparams.vol_shape,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
    )
    
    datamodule.setup(seed=hparams.seed)
    
    print("\n" + "="*80)
    print("Testing train dataloader...")
    print("="*80)
    train_loader = datamodule.train_dataloader()
    print(f"Train loader created with {len(train_loader)} batches")
    
    # Test loading one batch
    for i, batch in enumerate(train_loader):
        print(f"\nBatch {i}:")
        print(f"  CT shape: {batch['image3d'].shape}")
        print(f"  XR shape: {batch['image2d'].shape}")
        print(f"  CT min/max: {batch['image3d'].min():.4f} / {batch['image3d'].max():.4f}")
        print(f"  XR min/max: {batch['image2d'].min():.4f} / {batch['image2d'].max():.4f}")
        if i >= 2:  # Only test 3 batches
            break
    
    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)

