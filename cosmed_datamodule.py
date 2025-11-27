"""
Cosmed DataModule for 360° CT/XR rotation video generation.
This module handles two types of data:
1. XR-like frontal projection from CT -> 360° CT rotation video
2. Real XR frontal -> 360° XR rotation video
"""

import glob
import os
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import CacheDataset, ThreadDataLoader, list_data_collate
from monai.transforms import (
    Compose,
    CropForegroundDict,
    DivisiblePadDict,
    EnsureChannelFirstDict,
    HistogramNormalizeDict,
    LoadImageDict,
    MapTransform,
    OrientationDict,
    RandFlipDict,
    RandGaussianNoiseDict,
    RandRotateDict,
    RandShiftIntensityDict,
    RandStdShiftIntensityDict,
    RandZoomDict,
    Randomizable,
    ResizeDict,
    Rotate90Dict,
    ScaleIntensityDict,
    SpacingDict,
    SpatialPadDict,
    ToTensorDict,
    ZoomDict,
    apply_transform,
)
from monai.utils import set_determinism
from lightning import LightningDataModule, seed_everything

from transforms import ClipMinIntensityDict

try:
    from process_nifty_to_video import cache_paths_for_ct
except ImportError:
    def cache_paths_for_ct(project_root: str, ct_path: str) -> tuple[str, str, str, str]:
        """Generate cache paths for CT volume, video, image, and text prompt."""
        import hashlib
        
        # Determine prefix based on path (train or test)
        ct_path_lower = ct_path.lower()
        if '/train/' in ct_path_lower or '\\train\\' in ct_path_lower:
            prefix = "train_"
        elif '/test/' in ct_path_lower or '\\test\\' in ct_path_lower:
            prefix = "test_"
        else:
            prefix = ""
        
        stem = hashlib.sha1(os.path.abspath(ct_path).encode("utf-8")).hexdigest()
        
        # Organize CT data in cache/ct/ subdirectories
        vol_dir = os.path.join(project_root, "ct", "vol")
        vid_dir = os.path.join(project_root, "ct", "vid")
        img_dir = os.path.join(project_root, "ct", "img")
        txt_dir = os.path.join(project_root, "ct", "txt")
        
        os.makedirs(vol_dir, exist_ok=True)
        os.makedirs(vid_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(txt_dir, exist_ok=True)
        
        return (
            os.path.join(vol_dir, f"{prefix}{stem}.nii.gz"),
            os.path.join(vid_dir, f"{prefix}{stem}.mp4"),
            os.path.join(img_dir, f"{prefix}{stem}.png"),
            os.path.join(txt_dir, f"{prefix}{stem}.txt"),
        )

seed_everything(21, workers=True)


class LoadVideoDict(MapTransform):
    """Load video files (from cache) as tensor."""
    
    def __init__(self, keys, num_frames=361, img_shape=512):
        super().__init__(keys)
        self.num_frames = num_frames
        self.img_shape = img_shape
    
    def __call__(self, data):
        from torchvision.io import read_video
        d = dict(data)
        for key in self.keys:
            video_path = d[key]
            try:
                if not os.path.exists(video_path):
                    print(f"Warning: Video file does not exist: {video_path}")
                    # Create placeholder
                    d[key] = torch.zeros((self.num_frames, self.img_shape, self.img_shape))
                    continue
                
                # Load video using torchvision
                # read_video returns (video, audio, info) where video is (T, H, W, C)
                video, audio, info = read_video(video_path, pts_unit='sec')
                
                # Convert to grayscale if RGB: (T, H, W, C) -> (T, H, W)
                if video.shape[-1] == 3:
                    # Convert RGB to grayscale using standard weights
                    video = (0.299 * video[..., 0] + 0.587 * video[..., 1] + 0.114 * video[..., 2])
                elif video.shape[-1] == 1:
                    video = video.squeeze(-1)
                
                # Normalize to [0, 1] if not already
                if video.max() > 1.0:
                    video = video.float() / 255.0
                else:
                    video = video.float()
                
                # Ensure we have exactly num_frames
                if video.shape[0] != self.num_frames:
                    # Interpolate to get exact number of frames
                    # video is (T, H, W), need to add batch and channel dims for interpolate
                    video = video.unsqueeze(0).unsqueeze(0)  # (1, 1, T, H, W)
                    video = F.interpolate(
                        video, 
                        size=(self.num_frames, video.shape[-2], video.shape[-1]),
                        mode='trilinear',
                        align_corners=True
                    )
                    video = video.squeeze(0).squeeze(0)  # (T, H, W)
                
                d[key] = video
                
            except Exception as e:
                print(f"Error loading video {video_path}: {e}")
                import traceback
                traceback.print_exc()
                # Create placeholder on error
                d[key] = torch.zeros((self.num_frames, self.img_shape, self.img_shape))
        return d


class CosmedDataset(CacheDataset, Randomizable):
    """Dataset for Cosmed 360° rotation training."""
    
    def __init__(
        self,
        keys: Sequence,
        data: Sequence,
        transform: Optional[Callable] = None,
        length: Optional[int] = None,
        batch_size: int = 1,
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
                data[key] = dataset[index % len(dataset)]
                data[f"{key}_idx"] = index % len(dataset)
            data[f"{key}_pth"] = data[key]

        if self.transform is not None:
            data = apply_transform(self.transform, data)

        return data


class CosmedDataModule(LightningDataModule):
    """
    DataModule for Cosmed 360° rotation video generation.
    
    Handles two types of data:
    1. CT volumes -> Generate frontal XR + 360° rotation video (XR-like from CT)
    2. Real XR images -> Generate 360° rotation video (consistency between 0° and 360°)
    """
    
    def __init__(
        self,
        data_dir: str = "/workspace/datasets/ChestMedicalData",
        cache_dir: str = "/workspace/datasets/ChestMedicalDataCache",
        train_ct_folders: Optional[List[str]] = None,
        train_xr_folders: Optional[List[str]] = None,
        val_ct_folders: Optional[List[str]] = None,
        val_xr_folders: Optional[List[str]] = None,
        test_ct_folders: Optional[List[str]] = None,
        test_xr_folders: Optional[List[str]] = None,
        train_samples: int = 1000,
        val_samples: int = 100,
        test_samples: Optional[int] = None,
        img_shape: int = 512,
        vol_shape: int = 256,
        num_frames: int = 361,  # 360° + 1
        batch_size: int = 1,
        num_workers: int = 4,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.img_shape = img_shape
        self.vol_shape = vol_shape
        self.num_frames = num_frames
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        
        # Set default folders if not provided
        if train_ct_folders is None:
            train_ct_folders = [
                # os.path.join(data_dir, "NSCLC/processed/train/images"),
                # os.path.join(data_dir, "MOSMED/processed/train/images/CT-0"),
                # os.path.join(data_dir, "MOSMED/processed/train/images/CT-1"),
                # os.path.join(data_dir, "MOSMED/processed/train/images/CT-2"),
                # os.path.join(data_dir, "MOSMED/processed/train/images/CT-3"),
                # os.path.join(data_dir, "MOSMED/processed/train/images/CT-4"),
                # os.path.join(data_dir, "Imagenglab/processed/train/images"),
                # os.path.join(data_dir, "TCIA/processed/test/images"),
                 os.path.join(data_dir, "maisi/processed/train/images"),
                # os.path.join(data_dir, "maisi/processed/test/images"),
            ]
        
        if train_xr_folders is None:
            train_xr_folders = [
                 os.path.join(data_dir, "VinDr/v1/processed/train/images/"),
            ]
        
        if val_ct_folders is None:
           val_ct_folders = [
                 os.path.join(data_dir, "maisi/processed/test/images"),
            ]
        
        if val_xr_folders is None:
            val_xr_folders = [
                 os.path.join(data_dir, "VinDr/v1/processed/test/images/"),
            ]
        
        if test_ct_folders is None:
            test_ct_folders = [
                 os.path.join(data_dir, "TCIA/images/"),
            ]
        
        if test_xr_folders is None:
            test_xr_folders = [
                 os.path.join(data_dir, "VinDr/v1/processed/test/images/"),
            ]
        
        self.train_ct_folders = train_ct_folders
        self.train_xr_folders = train_xr_folders
        self.val_ct_folders = val_ct_folders
        self.val_xr_folders = val_xr_folders
        self.test_ct_folders = test_ct_folders
        self.test_xr_folders = test_xr_folders
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples

        def glob_files(folders: List[str], extension: str = "*.nii.gz") -> List[str]:
            """Glob files from multiple folders with given extension pattern."""
            assert folders is not None, "folders parameter cannot be None"
            paths = [
                glob.glob(os.path.join(folder, extension), recursive=True)
                for folder in folders
            ]
            files = sorted([item for sublist in paths for item in sublist])
            print(f"Found {len(files)} files with extension {extension}")
            if len(files) > 0:
                print(f"Example: {files[0]}")
            return files

        # Get CT files for video generation
        self.train_ct_files = glob_files(
            folders=train_ct_folders, extension="**/*.nii.gz"
        )
        
        # Get real XR images
        self.train_xr_files = glob_files(
            folders=train_xr_folders, extension="**/*.png"
        )

        self.val_ct_files = glob_files(
            folders=val_ct_folders, extension="**/*.nii.gz"
        )
        self.val_xr_files = glob_files(
            folders=val_xr_folders, extension="**/*.png"
        )

        self.test_ct_files = glob_files(
            folders=test_ct_folders, extension="**/*.nii.gz"
        )
        self.test_xr_files = glob_files(
            folders=test_xr_folders, extension="**/*.png"
        )
        
        # Generate cache paths for CT files (for video loading)
        self.train_ct_cache = [cache_paths_for_ct(cache_dir, f) for f in self.train_ct_files]
        self.val_ct_cache = [cache_paths_for_ct(cache_dir, f) for f in self.val_ct_files]
        self.test_ct_cache = [cache_paths_for_ct(cache_dir, f) for f in self.test_ct_files]

    def setup(self, seed: int = 2222, stage: Optional[str] = None):
        set_determinism(seed=seed)
        
        # Check if cache directory exists and is valid
        if self.cache_dir and os.path.exists(self.cache_dir):
            print(f"\n{'='*80}")
            print("Checking cache validity...")
            print(f"{'='*80}")
            
            cache_valid, missing_files = self._check_cache_validity()
            
            if cache_valid:
                print("✓ All cache files are present and up-to-date")
                print(f"  - Using cache from: {self.cache_dir}")
            else:
                print(f"⚠ Warning: Some cache files are missing ({len(missing_files)} files)")
                if len(missing_files) <= 10:
                    print("  Missing files:")
                    for f in missing_files:
                        print(f"    - {f}")
                else:
                    print(f"  First 10 missing files:")
                    for f in missing_files[:10]:
                        print(f"    - {f}")
                    print(f"    ... and {len(missing_files) - 10} more")
                
                print(f"\n  To regenerate cache, run:")
                print(f"    python cosmed_datamodule.py --mode cache --data_dir {self.data_dir} --cache_dir {self.cache_dir}")
            
            print(f"{'='*80}\n")
    
    def _check_cache_validity(self) -> tuple[bool, list[str]]:
        """
        Check if cache is valid by verifying all expected cache files exist.
        
        Returns:
            tuple: (is_valid, list_of_missing_files)
        """
        missing_files = []
        
        # Check CT cache files
        for ct_file, (vol_cache, vid_cache, img_cache, txt_cache) in zip(
            self.train_ct_files + self.val_ct_files,
            self.train_ct_cache + self.val_ct_cache
        ):
            # We need at minimum the video and frontal image
            if not os.path.exists(vid_cache):
                missing_files.append(f"CT video: {os.path.basename(vid_cache)}")
            if not os.path.exists(img_cache):
                missing_files.append(f"CT frontal: {os.path.basename(img_cache)}")
        
        # Check XR cache files
        xr_cache_dir = os.path.join(self.cache_dir, "xr", "img")
        if os.path.exists(xr_cache_dir):
            import hashlib
            for xr_file in self.train_xr_files + self.val_xr_files:
                xr_path_lower = xr_file.lower()
                if '/train/' in xr_path_lower or '\\train\\' in xr_path_lower:
                    prefix = "train_"
                elif '/test/' in xr_path_lower or '\\test\\' in xr_path_lower:
                    prefix = "test_"
                else:
                    prefix = ""
                
                stem = hashlib.sha1(os.path.abspath(xr_file).encode("utf-8")).hexdigest()
                cache_path = os.path.join(xr_cache_dir, f"{prefix}{stem}.png")
                
                if not os.path.exists(cache_path):
                    missing_files.append(f"XR image: {os.path.basename(cache_path)}")
        
        return len(missing_files) == 0, missing_files

    def train_dataloader(self):
        """
        Training data contains:
        - ct_video: 360° rotation video from CT (shape: T, H, W) - loaded from cache
        - ct_frontal: Frontal XR-like image from CT (shape: H, W) - loaded from cache
        - xr_image: Real XR frontal image (shape: H, W) - loaded from cache if available
        """
        # For cached data, we only need to load the files, not preprocess them
        self.train_transforms = Compose([
            # Load cached CT frontal images and videos (already preprocessed)
            LoadImageDict(keys=["ct_frontal"]),
            EnsureChannelFirstDict(keys=["ct_frontal"]),
            ScaleIntensityDict(keys=["ct_frontal"], minv=0.0, maxv=1.0),  # Normalize PNG from [0, 255] to [0, 1]
            LoadVideoDict(keys=["ct_video"], num_frames=self.num_frames, img_shape=self.img_shape),
            
            # Load XR images - check if cache exists first
            LoadImageDict(keys=["xr_image"]),
            EnsureChannelFirstDict(keys=["xr_image"]),
            ScaleIntensityDict(keys=["xr_image"], minv=0.0, maxv=1.0),
            
            # Add augmentations only (no preprocessing since data is cached)
            # RandShiftIntensityDict(keys=["ct_frontal"], prob=0.5, offsets=0.1, safe=True),
            # RandGaussianNoiseDict(keys=["ct_frontal"], prob=0.3, mean=0.0, std=0.05),
            
            ToTensorDict(keys=["xr_image", "ct_frontal", "ct_video"]),
        ])

        # Create dataset - only reference cached files, not raw CT volumes
        train_data = []
        for ct_file, (vol_cache, vid_cache, img_cache, txt_cache) in zip(
            self.train_ct_files, self.train_ct_cache
        ):
            train_data.append({
                "ct_video": vid_cache,      # Cached video
                "ct_frontal": img_cache,    # Cached frontal image
            })
        
        # Check if XR cache exists, otherwise use raw XR files
        xr_cache_dir = os.path.join(self.cache_dir, "xr", "img")
        xr_data = []
        xr_cached_count = 0
        xr_raw_count = 0
        
        if os.path.exists(xr_cache_dir):
            # Use cached XR images
            for xr_file in self.train_xr_files:
                import hashlib
                xr_path_lower = xr_file.lower()
                if '/train/' in xr_path_lower or '\\train\\' in xr_path_lower:
                    prefix = "train_"
                else:
                    prefix = ""
                stem = hashlib.sha1(os.path.abspath(xr_file).encode("utf-8")).hexdigest()
                cache_path = os.path.join(xr_cache_dir, f"{prefix}{stem}.png")
                
                if os.path.exists(cache_path):
                    xr_data.append(cache_path)
                    xr_cached_count += 1
                else:
                    xr_data.append(xr_file)  # Fall back to raw file if cache missing
                    xr_raw_count += 1
        else:
            # No cache, use raw files
            xr_data = self.train_xr_files
            xr_raw_count = len(xr_data)
        
        print(f"Train data: {len(train_data)} CT samples from cache")
        print(f"Train data: {xr_cached_count} XR cached, {xr_raw_count} XR raw")
        
        self.train_datasets = CosmedDataset(
            keys=["ct_video", "ct_frontal", "xr_image"],
            data=[
                [d["ct_video"] for d in train_data],
                [d["ct_frontal"] for d in train_data],
                xr_data,
            ],
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
        """Validation dataloader - loads cached data only."""
        self.val_transforms = Compose([
            # Load cached CT frontal images and videos (already preprocessed)
            LoadImageDict(keys=["ct_frontal"]),
            EnsureChannelFirstDict(keys=["ct_frontal"]),
            ScaleIntensityDict(keys=["ct_frontal"], minv=0.0, maxv=1.0),  # Normalize PNG from [0, 255] to [0, 1]
            LoadVideoDict(keys=["ct_video"], num_frames=self.num_frames, img_shape=self.img_shape),
            
            # Load XR images from cache
            LoadImageDict(keys=["xr_image"]),
            EnsureChannelFirstDict(keys=["xr_image"]),
            ScaleIntensityDict(keys=["xr_image"], minv=0.0, maxv=1.0),
            
            ToTensorDict(keys=["xr_image", "ct_frontal", "ct_video"]),
        ])

        # Create dataset - only reference cached files
        val_data = []
        for ct_file, (vol_cache, vid_cache, img_cache, txt_cache) in zip(
            self.val_ct_files, self.val_ct_cache
        ):
            val_data.append({
                "ct_video": vid_cache,      # Cached video
                "ct_frontal": img_cache,    # Cached frontal image
            })
        
        # Check if XR cache exists for validation
        xr_cache_dir = os.path.join(self.cache_dir, "xr", "img")
        xr_data = []
        xr_cached_count = 0
        xr_raw_count = 0
        
        if os.path.exists(xr_cache_dir):
            # Use cached XR images
            for xr_file in self.val_xr_files:
                import hashlib
                xr_path_lower = xr_file.lower()
                if '/test/' in xr_path_lower or '\\test\\' in xr_path_lower:
                    prefix = "test_"
                elif '/train/' in xr_path_lower or '\\train\\' in xr_path_lower:
                    prefix = "train_"
                else:
                    prefix = ""
                stem = hashlib.sha1(os.path.abspath(xr_file).encode("utf-8")).hexdigest()
                cache_path = os.path.join(xr_cache_dir, f"{prefix}{stem}.png")
                
                if os.path.exists(cache_path):
                    xr_data.append(cache_path)
                    xr_cached_count += 1
                else:
                    xr_data.append(xr_file)  # Fall back to raw file if cache missing
                    xr_raw_count += 1
        else:
            # No cache, use raw files
            xr_data = self.val_xr_files
            xr_raw_count = len(xr_data)
        
        print(f"Val data: {len(val_data)} CT samples from cache")
        print(f"Val data: {xr_cached_count} XR cached, {xr_raw_count} XR raw")

        self.val_datasets = CosmedDataset(
            keys=["ct_video", "ct_frontal", "xr_image"],
            data=[
                [d["ct_video"] for d in val_data],
                [d["ct_frontal"] for d in val_data],
                xr_data,
            ],
            transform=self.val_transforms,
            length=self.val_samples,
            batch_size=self.batch_size,
            is_training=False,
        )

        self.val_loader = ThreadDataLoader(
            self.val_datasets,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=list_data_collate,
            shuffle=False,
        )
        return self.val_loader

    def test_dataloader(self):
        """Test dataloader (similar to validation)."""
        return self.val_dataloader()


def pre_generate_cache(
    data_dir: str = "/workspace/datasets/ChestMedicalData",
    cache_dir: str = "/workspace/datasets/ChestMedicalDataCache",
    ct_folders: Optional[List[str]] = None,
    xr_folders: Optional[List[str]] = None,
    img_shape: int = 512,
    vol_shape: int = 256,
    num_frames: int = 361,
    max_files: Optional[int] = None,
    force: bool = False,
    process_ct: bool = True,
    process_xr: bool = True,
    num_workers: int = 4,
):
    """
    Pre-generate cache files for CT volumes and XR images.
    
    Creates for CT:
    - Preprocessed CT volumes
    - 360° rotation videos
    - Frontal projection images
    
    Creates for XR:
    - Preprocessed XR images
    
    Args:
        data_dir: Root directory for datasets
        cache_dir: Directory to store cache files
        ct_folders: List of folders containing CT volumes
        xr_folders: List of folders containing XR images
        img_shape: Image size for frontal projections and XR images
        vol_shape: Volume size for CT preprocessing
        num_frames: Number of frames in rotation video (361 for 360°+1)
        max_files: Maximum number of files to process per type (None for all)
        force: Force regeneration even if cache exists
        process_ct: Whether to process CT volumes
        process_xr: Whether to process XR images
        num_workers: Number of parallel workers (for XR processing)
    """
    import numpy as np
    import nibabel as nib
    from PIL import Image
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from utilities import save_vid_as_mp4, save_img_as_png, save_vol_as_nifti
    
    console = Console()
    
    try:
        from dvr.renderer import DifferentiableVolumeRenderer
        from camera import make_cameras_dea
        DVR_AVAILABLE = True
    except ImportError:
        DVR_AVAILABLE = False
        console.print("[yellow]⚠ Warning: DVR not available, using simple projection[/yellow]")
    
    # Print header with rich
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Pre-generating Cache for Cosmed Training[/bold cyan]",
        border_style="cyan"
    ))
    
    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column("Key", style="bold blue")
    info_table.add_column("Value", style="white")
    info_table.add_row("Data directory", data_dir)
    info_table.add_row("Cache directory", cache_dir)
    info_table.add_row("Image shape", f"{img_shape}×{img_shape}")
    info_table.add_row("Volume shape", f"{vol_shape}³")
    info_table.add_row("Number of frames", str(num_frames))
    info_table.add_row("Process CT", "✓ Yes" if process_ct else "✗ No")
    info_table.add_row("Process XR", "✓ Yes" if process_xr else "✗ No")
    info_table.add_row("DVR available", "✓ Yes" if DVR_AVAILABLE else "✗ No (using fallback)")
    info_table.add_row("Parallel workers", str(num_workers))
    console.print(info_table)
    console.print()
    
    # Process CT volumes
    if process_ct:
        _process_ct_volumes(
            data_dir, cache_dir, ct_folders, img_shape, vol_shape, 
            num_frames, max_files, force, console, DVR_AVAILABLE
        )
    
    # Process XR images
    if process_xr:
        _process_xr_images(
            data_dir, cache_dir, xr_folders, img_shape, 
            max_files, force, console, num_workers
        )


def _process_ct_volumes(
    data_dir: str,
    cache_dir: str,
    ct_folders: Optional[List[str]],
    img_shape: int,
    vol_shape: int,
    num_frames: int,
    max_files: Optional[int],
    force: bool,
    console,
    DVR_AVAILABLE: bool,
):
    """Process CT volumes to generate cache using val_transforms and DVR rendering."""
    import numpy as np
    import nibabel as nib
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
    from rich.panel import Panel
    from rich.table import Table
    from utilities import save_vid_as_mp4, save_img_as_png, save_vol_as_nifti
    from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform
    
    # Import MONAI transforms
    from monai.transforms import (
        Compose,
        LoadImageDict,
        EnsureChannelFirstDict,
        SpacingDict,
        OrientationDict,
        ScaleIntensityDict,
        ZoomDict,
        ResizeDict,
        DivisiblePadDict,
        apply_transform,
    )
    
    try:
        from dvr.renderer import ObjectCentricXRayVolumeRenderer
        DVR_AVAILABLE = True
    except ImportError:
        DVR_AVAILABLE = False
        console.print("[bold red]✗ Error: DVR not available. Cannot process CT volumes.[/bold red]")
        console.print("[yellow]Please install PyTorch3D and ensure dvr module is available.[/yellow]")
        return
    
    console.print(Panel.fit("[bold green]Processing CT Volumes[/bold green]", border_style="green"))
    
    # Define transforms with CENTERED padding to prevent jiggling
    ct_transforms = Compose([
        LoadImageDict(keys=["ct_volume"]),
        EnsureChannelFirstDict(keys=["ct_volume"]),
        SpacingDict(keys=["ct_volume"], pixdim=(1.0, 1.0, 1.0), mode=["bilinear"], align_corners=True),
        OrientationDict(keys=["ct_volume"], axcodes="ASL"),
        ClipMinIntensityDict(keys=["ct_volume"], min_val=-512),
        ScaleIntensityDict(keys=["ct_volume"], minv=0.0, maxv=1.0),
        ResizeDict(keys=["ct_volume"], spatial_size=vol_shape, size_mode="longest", mode=["trilinear"], align_corners=True),
        # Use SpatialPadDict with "symmetric" mode for CENTERED padding
        SpatialPadDict(keys=["ct_volume"], spatial_size=(vol_shape, vol_shape, vol_shape), mode="constant", constant_values=0, method="symmetric"),
    ])
    
    # Set default CT folders if not provided
    if ct_folders is None:
        ct_folders = [
            # os.path.join(data_dir, "NSCLC/processed/train/images"),
            # os.path.join(data_dir, "MOSMED/processed/train/images/CT-0"),
            # os.path.join(data_dir, "MOSMED/processed/train/images/CT-1"),
            # os.path.join(data_dir, "MOSMED/processed/train/images/CT-2"),
            # os.path.join(data_dir, "MOSMED/processed/train/images/CT-3"),
            # os.path.join(data_dir, "MOSMED/processed/train/images/CT-4"),
            # os.path.join(data_dir, "Imagenglab/processed/train/images"),
            # os.path.join(data_dir, "TCIA/processed/test/images"),
                 os.path.join(data_dir, "maisi/processed/train/images"),
            # os.path.join(data_dir, "maisi/processed/test/images"),
        ]
    
    # Collect all CT files
    console.print("[bold]Scanning directories...[/bold]")
    ct_files = []
    for folder in ct_folders:
        if os.path.exists(folder):
            files = glob.glob(os.path.join(folder, "**/*.nii.gz"), recursive=True)
            ct_files.extend(files)
            folder_name = os.path.basename(os.path.dirname(folder))
            console.print(f"  [green]✓[/green] Found {len(files):5d} files in {folder_name}")
        else:
            folder_name = os.path.basename(os.path.dirname(folder))
            console.print(f"  [yellow]⚠[/yellow] Skipped {folder_name} (not found)")
    
    if not ct_files:
        console.print("[bold red]✗ Error: No CT files found![/bold red]")
        return
    
    console.print(f"\n[bold cyan]Total CT files:[/bold cyan] {len(ct_files):,}")
    
    if max_files is not None:
        ct_files = ct_files[:max_files]
        console.print(f"[yellow]Processing first {max_files:,} files[/yellow]")
    
    # Generate cache for each CT file
    processed = 0
    skipped = 0
    failed = 0
    
    # Create renderer (reuse for all volumes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    renderer = ObjectCentricXRayVolumeRenderer(
        image_width=img_shape,
        image_height=img_shape,
        n_pts_per_ray=512,
        min_depth=4.0,
        max_depth=8.0,
        ndc_extent=1.0,
    ).to(device)
    
    # Create cameras for all frames (reuse for all volumes)
    cameras = []
    azimuths = torch.linspace(0, 360, num_frames)
    for azimuth in azimuths:
        R, T = look_at_view_transform(
            dist=6.0,
            elev=0.0,
            azim=azimuth.item(),
        )
        camera = FoVPerspectiveCameras(
            R=R,
            T=T,
            fov=20.0,
            device=device,
        )
        cameras.append(camera)
    
    console.print(f"[dim]Created renderer and {len(cameras)} cameras on {device}[/dim]")
    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("[cyan]{task.completed}/{task.total}[/cyan]"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing CT files...", total=len(ct_files))
        
        for ct_file in ct_files:
            progress.update(task, advance=0, description=f"[cyan]Processing {os.path.basename(ct_file)[:30]}...")
            try:
                # Get cache paths
                vol_path, vid_path, img_path, txt_path = cache_paths_for_ct(cache_dir, ct_file)
                
                # Skip if cache exists and not forcing
                if not force and os.path.exists(vid_path) and os.path.exists(img_path):
                    skipped += 1
                    progress.update(task, advance=1)
                    continue
                
                # Prepare data dict with file path (LoadImageDict expects path)
                data = {"ct_volume": ct_file}
                
                # Apply transforms (same as val_dataloader)
                transformed_data = apply_transform(ct_transforms, data)
                vol_tensor = transformed_data["ct_volume"]
                
                # Ensure proper shape for rendering
                if vol_tensor.ndim == 3:
                    vol_tensor = vol_tensor.unsqueeze(0)  # Add channel dimension (C, D, H, W)
                if vol_tensor.ndim == 4:
                    vol_tensor = vol_tensor.unsqueeze(0)  # Add batch dimension (1, C, D, H, W)
                
                # Move to device
                vol_tensor = vol_tensor.to(device)
                
                # Save preprocessed volume
                save_vol_as_nifti(vol_tensor, vol_path)
                
                # Generate 360° rotation video using DVR (same as process_nifty_to_video.py)
                frames = []
                for camera in cameras:
                    with torch.no_grad():
                        projection = renderer(
                            image3d=vol_tensor,
                            cameras=camera,
                            opacity=None,
                            norm_type="minimized",
                            scaling_factor=1.0,
                            is_grayscale=True,
                            return_bundle=False,
                        )
                    frames.append(projection)
                
                # Stack frames into video (1, 1, T, H, W)
                video = torch.stack(frames, dim=2)
                
                # Save video as MP4
                save_vid_as_mp4(video, vid_path, fps=30)
                
                # Save first frame (0° view) as PNG
                save_img_as_png(video, img_path)
                
                # Save text prompt
                with open(txt_path, 'w') as f:
                    f.write(f"360 degree rotation of chest CT scan")
                
                processed += 1
                progress.update(task, advance=1)
                
            except Exception as e:
                console.print(f"\n[red]✗ Error processing {os.path.basename(ct_file)}: {e}[/red]")
                failed += 1
                progress.update(task, advance=1)
                continue
    
    # Print summary
    console.print()
    console.print(Panel.fit(
        "[bold green]Cache Generation Complete![/bold green]",
        border_style="green"
    ))
    
    summary_table = Table(show_header=False, box=None, padding=(0, 2))
    summary_table.add_column("Metric", style="bold blue")
    summary_table.add_column("Count", style="white", justify="right")
    summary_table.add_row("✓ Processed", f"[green]{processed:,}[/green]")
    summary_table.add_row("⊘ Skipped (cached)", f"[yellow]{skipped:,}[/yellow]")
    summary_table.add_row("✗ Failed", f"[red]{failed:,}[/red]")
    summary_table.add_row("━" * 20, "━" * 10)
    summary_table.add_row("Total", f"[cyan]{len(ct_files):,}[/cyan]")
    
    console.print(summary_table)
    console.print()


def _process_single_xr_worker(args):
    """
    Worker function for parallel XR processing.
    Must be at module level for pickling.
    
    Args:
        args: Tuple of (xr_file, xr_img_dir, img_shape)
    
    Returns:
        Tuple of (success, error_message)
    """
    xr_file, xr_img_dir, img_shape = args
    
    try:
        import hashlib
        import numpy as np
        from PIL import Image
        from monai.transforms import (
            Compose,
            LoadImageDict,
            EnsureChannelFirstDict,
            Rotate90Dict,
            RandFlipDict,
            ScaleIntensityDict,
            HistogramNormalizeDict,
            CropForegroundDict,
            ZoomDict,
            ResizeDict,
            DivisiblePadDict,
            apply_transform,
        )
        
        # Define transforms
        xr_transforms = Compose([
            LoadImageDict(keys=["xr_image"]),
            EnsureChannelFirstDict(keys=["xr_image"]),
            Rotate90Dict(keys=["xr_image"], k=3),
            RandFlipDict(keys=["xr_image"], prob=1.0, spatial_axis=1),
            ScaleIntensityDict(keys=["xr_image"], minv=0.0, maxv=1.0),
            HistogramNormalizeDict(keys=["xr_image"], min=0.0, max=1.0),
            CropForegroundDict(keys=["xr_image"], source_key="xr_image", 
                             select_fn=(lambda x: x > 0), margin=0),
            ZoomDict(keys=["xr_image"], zoom=0.95, padding_mode="constant", mode=["area"]),
            ResizeDict(keys=["xr_image"], spatial_size=img_shape, size_mode="longest", mode=["area"]),
            DivisiblePadDict(keys=["xr_image"], k=img_shape, mode="constant", constant_values=0),
        ])
        
        # Generate cache path
        xr_path_lower = xr_file.lower()
        if '/train/' in xr_path_lower or '\\train\\' in xr_path_lower:
            prefix = "train_"
        elif '/test/' in xr_path_lower or '\\test\\' in xr_path_lower:
            prefix = "test_"
        else:
            prefix = ""
        
        stem = hashlib.sha1(os.path.abspath(xr_file).encode("utf-8")).hexdigest()
        cache_path = os.path.join(xr_img_dir, f"{prefix}{stem}.png")
        
        # Prepare data dict with file path
        data = {"xr_image": xr_file}
        
        # Apply transforms
        transformed_data = apply_transform(xr_transforms, data)
        xr_tensor = transformed_data["xr_image"]
        
        # Convert to numpy and save
        if isinstance(xr_tensor, torch.Tensor):
            xr_np = xr_tensor.squeeze().cpu().numpy()
        else:
            xr_np = xr_tensor.squeeze()
        
        # Convert to uint8 and save
        xr_np = (xr_np * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(xr_np).save(cache_path)
        
        return True, None
        
    except Exception as e:
        return False, str(e)


def _process_xr_images(
    data_dir: str,
    cache_dir: str,
    xr_folders: Optional[List[str]],
    img_shape: int,
    max_files: Optional[int],
    force: bool,
    console,
    num_workers: int = 4,
):
    """Process XR images to generate cache using val_transforms with multiprocessing."""
    import numpy as np
    from PIL import Image
    from multiprocessing import Pool
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
    from rich.panel import Panel
    from rich.table import Table
    
    console.print(Panel.fit("[bold magenta]Processing XR Images[/bold magenta]", border_style="magenta"))
    
    # Set default XR folders if not provided
    if xr_folders is None:
        xr_folders = [
                 os.path.join(data_dir, "VinDr/v1/processed/train/images/"),
                 os.path.join(data_dir, "VinDr/v1/processed/test/images/"),
        ]
    
    # Create XR cache directory (organized under cache/xr/img/)
    xr_img_dir = os.path.join(cache_dir, "xr", "img")
    os.makedirs(xr_img_dir, exist_ok=True)
    
    # Collect all XR files
    console.print("[bold]Scanning XR directories...[/bold]")
    xr_files = []
    for folder in xr_folders:
        if os.path.exists(folder):
            files = glob.glob(os.path.join(folder, "**/*.png"), recursive=True)
            files.extend(glob.glob(os.path.join(folder, "**/*.jpg"), recursive=True))
            files.extend(glob.glob(os.path.join(folder, "**/*.jpeg"), recursive=True))
            xr_files.extend(files)
            folder_name = os.path.basename(os.path.dirname(folder))
            console.print(f"  [green]✓[/green] Found {len(files):5d} files in {folder_name}")
        else:
            folder_name = os.path.basename(os.path.dirname(folder))
            console.print(f"  [yellow]⚠[/yellow] Skipped {folder_name} (not found)")
    
    if not xr_files:
        console.print("[bold yellow]⚠ No XR files found, skipping...[/bold yellow]")
        return
    
    console.print(f"\n[bold cyan]Total XR files:[/bold cyan] {len(xr_files):,}")
    
    if max_files is not None:
        xr_files = xr_files[:max_files]
        console.print(f"[yellow]Processing first {max_files:,} files[/yellow]")
    
    # Filter files that need processing
    files_to_process = []
    for xr_file in xr_files:
        import hashlib
        xr_path_lower = xr_file.lower()
        if '/train/' in xr_path_lower or '\\train\\' in xr_path_lower:
            prefix = "train_"
        elif '/test/' in xr_path_lower or '\\test\\' in xr_path_lower:
            prefix = "test_"
        else:
            prefix = ""
        
        stem = hashlib.sha1(os.path.abspath(xr_file).encode("utf-8")).hexdigest()
        cache_path = os.path.join(xr_img_dir, f"{prefix}{stem}.png")
        
        if force or not os.path.exists(cache_path):
            files_to_process.append(xr_file)
    
    skipped = len(xr_files) - len(files_to_process)
    
    if skipped > 0:
        console.print(f"[yellow]Skipping {skipped:,} already cached files[/yellow]")
    
    if not files_to_process:
        console.print("[bold yellow]⚠ All files already cached, nothing to process[/bold yellow]")
        return
    
    console.print(f"[bold cyan]Files to process:[/bold cyan] {len(files_to_process):,}")
    
    # Process XR files with multiprocessing
    console.print(f"[dim]Using {num_workers} parallel workers[/dim]")
    console.print()
    
    # Prepare arguments for parallel processing
    process_args = [(xr_file, xr_img_dir, img_shape) for xr_file in files_to_process]
    
    # Process with multiprocessing
    processed = 0
    failed = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("[cyan]{task.completed}/{task.total}[/cyan]"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[magenta]Processing XR images...", total=len(files_to_process))
        
        # Process with pool
        with Pool(processes=num_workers) as pool:
            for success, error in pool.imap_unordered(_process_single_xr_worker, process_args):
                if success:
                    processed += 1
                else:
                    failed += 1
                    if error:
                        console.print(f"\n[red]✗ Error: {error}[/red]")
                progress.update(task, advance=1)
    
    # Print summary
    console.print()
    console.print(Panel.fit(
        "[bold magenta]XR Processing Complete![/bold magenta]",
        border_style="magenta"
    ))
    
    summary_table = Table(show_header=False, box=None, padding=(0, 2))
    summary_table.add_column("Metric", style="bold blue")
    summary_table.add_column("Count", style="white", justify="right")
    summary_table.add_row("✓ Processed", f"[green]{processed:,}[/green]")
    summary_table.add_row("⊘ Skipped (cached)", f"[yellow]{skipped:,}[/yellow]")
    summary_table.add_row("✗ Failed", f"[red]{failed:,}[/red]")
    summary_table.add_row("━" * 20, "━" * 10)
    summary_table.add_row("Total", f"[cyan]{len(xr_files):,}[/cyan]")
    
    console.print(summary_table)
    console.print()



if __name__ == "__main__":
    import sys
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Cosmed DataModule - Pre-generate cache for training")
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="cache",
        choices=["cache", "test"],
        help="Mode: 'cache' to pre-generate cache, 'test' to test data loading"
    )
    
    # Data directories
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/workspace/datasets/ChestMedicalData",
        help="Root directory for datasets"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/workspace/datasets/ChestMedicalDataCache",
        help="Directory to store cache files"
    )
    
    # Cache generation parameters
    parser.add_argument(
        "--img_shape",
        type=int,
        default=512,
        help="Image size for frontal projections"
    )
    parser.add_argument(
        "--vol_shape",
        type=int,
        default=256,
        help="Volume size for CT preprocessing"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=121,
        help="Number of frames in rotation video"
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to process (None for all)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if cache exists"
    )
    parser.add_argument(
        "--process_ct",
        action="store_true",
        default=True,
        help="Process CT volumes (default: True)"
    )
    parser.add_argument(
        "--process_xr",
        action="store_true",
        default=True,
        help="Process XR images (default: True)"
    )
    parser.add_argument(
        "--skip_ct",
        action="store_true",
        help="Skip CT processing"
    )
    parser.add_argument(
        "--skip_xr",
        action="store_true",
        help="Skip XR processing"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of parallel workers for processing (default: 4)"
    )
    
    # Test parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Batch size for testing"
    )

    parser.add_argument(
        "--train_samples",
        type=int,
        default=100,
        help="Number of training samples to test"
    )
    parser.add_argument(
        "--val_samples",
        type=int,
        default=20,
        help="Number of validation samples to test"
    )
    
    args = parser.parse_args()
    
    if args.mode == "cache":
        # Pre-generate cache
        print("\n🔄 Pre-generating cache files...\n")
        
        # Determine what to process
        process_ct = not args.skip_ct
        process_xr = not args.skip_xr
        
        pre_generate_cache(
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            img_shape=args.img_shape,
            vol_shape=args.vol_shape,
            num_frames=args.num_frames,
            max_files=args.max_files,
            force=args.force,
            process_ct=process_ct,
            process_xr=process_xr,
            num_workers=args.num_workers,
        )
    
    elif args.mode == "test":
        # Test data loading
        print("\n🧪 Testing data loading and cache validation...\n")
        
        # First, check cache directories exist
        print("="*80)
        print("Checking cache directories...")
        print("="*80)
        
        ct_cache_exists = False
        xr_cache_exists = False
        
        ct_vid_dir = os.path.join(args.cache_dir, "ct", "vid")
        ct_img_dir = os.path.join(args.cache_dir, "ct", "img")
        xr_img_dir = os.path.join(args.cache_dir, "xr", "img")
        
        if os.path.exists(ct_vid_dir) and os.path.exists(ct_img_dir):
            ct_vid_files = glob.glob(os.path.join(ct_vid_dir, "*.mp4"))
            ct_img_files = glob.glob(os.path.join(ct_img_dir, "*.png"))
            if ct_vid_files and ct_img_files:
                ct_cache_exists = True
                print(f"✓ CT cache found: {len(ct_vid_files)} videos, {len(ct_img_files)} images")
            else:
                print(f"✗ CT cache directories exist but no files found")
        else:
            print(f"✗ CT cache not found at {ct_vid_dir}")
        
        if os.path.exists(xr_img_dir):
            xr_img_files = glob.glob(os.path.join(xr_img_dir, "*.png"))
            if xr_img_files:
                xr_cache_exists = True
                print(f"✓ XR cache found: {len(xr_img_files)} images")
            else:
                print(f"✗ XR cache directory exists but no files found")
        else:
            print(f"✗ XR cache not found at {xr_img_dir}")
        
        if not ct_cache_exists and not xr_cache_exists:
            print("\n" + "="*80)
            print("✗ ERROR: No cache found!")
            print("="*80)
            print("\nPlease run cache generation first:")
            print(f"  python {__file__} --mode cache --data_dir {args.data_dir} --cache_dir {args.cache_dir}")
            sys.exit(1)
        
        print("\n" + "="*80)
        print("Creating datamodule...")
        print("="*80)
        
        try:
            datamodule = CosmedDataModule(
                data_dir=args.data_dir,
                cache_dir=args.cache_dir,
                train_samples=args.train_samples,
                val_samples=args.val_samples,
                img_shape=args.img_shape,
                vol_shape=args.vol_shape,
                num_frames=args.num_frames,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            
            datamodule.setup(seed=42)
            print(f"✓ DataModule created successfully")
            print(f"  - Train CT files: {len(datamodule.train_ct_files)}")
            print(f"  - Train XR files: {len(datamodule.train_xr_files)}")
            print(f"  - Val CT files: {len(datamodule.val_ct_files)}")
            print(f"  - Val XR files: {len(datamodule.val_xr_files)}")
            
        except Exception as e:
            print(f"✗ Error creating datamodule: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        print("\n" + "="*80)
        print("Testing train dataloader...")
        print("="*80)
        
        try:
            train_loader = datamodule.train_dataloader()
            print(f"✓ Train loader created with {len(train_loader)} batches")
            
            # Test loading a few batches
            print(f"\nLoading {min(3, len(train_loader))} test batches...")
            for i, batch in enumerate(train_loader):
                print(f"\nBatch {i}:")
                
                if 'ct_volume' in batch:
                    print(f"  ✓ CT volume: shape={batch['ct_volume'].shape}, "
                          f"range=[{batch['ct_volume'].min():.4f}, {batch['ct_volume'].max():.4f}]")
                
                if 'ct_frontal' in batch:
                    print(f"  ✓ CT frontal: shape={batch['ct_frontal'].shape}, "
                          f"range=[{batch['ct_frontal'].min():.4f}, {batch['ct_frontal'].max():.4f}]")
                
                if 'ct_video' in batch:
                    print(f"  ✓ CT video: shape={batch['ct_video'].shape}, "
                          f"range=[{batch['ct_video'].min():.4f}, {batch['ct_video'].max():.4f}]")
                    # Verify it's the expected number of frames
                    if batch['ct_video'].shape[1] != args.num_frames:
                        print(f"    ⚠ Warning: Expected {args.num_frames} frames, got {batch['ct_video'].shape[1]}")
                
                if 'xr_image' in batch:
                    print(f"  ✓ XR image: shape={batch['xr_image'].shape}, "
                          f"range=[{batch['xr_image'].min():.4f}, {batch['xr_image'].max():.4f}]")
                
                if i >= 2:  # Only test 3 batches
                    break
            
            print("\n" + "="*80)
            print("Testing validation dataloader...")
            print("="*80)
            
            val_loader = datamodule.val_dataloader()
            print(f"✓ Val loader created with {len(val_loader)} batches")
            
            # Test loading one validation batch
            print(f"\nLoading 1 validation batch...")
            for i, batch in enumerate(val_loader):
                print(f"\nValidation Batch {i}:")
                
                if 'ct_volume' in batch:
                    print(f"  ✓ CT volume: shape={batch['ct_volume'].shape}")
                if 'ct_frontal' in batch:
                    print(f"  ✓ CT frontal: shape={batch['ct_frontal'].shape}")
                if 'ct_video' in batch:
                    print(f"  ✓ CT video: shape={batch['ct_video'].shape}")
                if 'xr_image' in batch:
                    print(f"  ✓ XR image: shape={batch['xr_image'].shape}")
                
                break  # Only test 1 validation batch
            
            print("\n" + "="*80)
            print("✓ All tests passed!")
            print("="*80)
            print("\nCache is working correctly and ready for training!")
            
        except Exception as e:
            print(f"\n✗ Error during testing: {e}")
            import traceback
            traceback.print_exc()
            print("\n" + "="*80)
            print("✗ Tests failed!")
            print("="*80)
