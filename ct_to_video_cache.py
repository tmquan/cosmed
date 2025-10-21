import os
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
import imageio
from typing import List, Optional
import glob

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
)

from dvr.renderer import ObjectCentricXRayVolumeRenderer

# Import MONAI transforms
from monai.transforms import (
    Compose,
    LoadImageDict,
    EnsureChannelFirstDict,
    SpacingDict,
    OrientationDict,
    ScaleIntensityDict,
    ResizeDict,
    DivisiblePadDict,
    ToTensorDict,
)

# Import datamodule components
from cm_datamodule import cache_paths_for_ct, ClipMinIntensityDict


def create_ct_transforms(vol_shape: int = 256):
    """Create transforms for CT preprocessing (matches datamodule)"""
    return Compose([
        LoadImageDict(keys=["image3d"]),
        EnsureChannelFirstDict(keys=["image3d"]),
        SpacingDict(keys=["image3d"], pixdim=(1.0, 1.0, 1.0), mode=["bilinear"], align_corners=True),
        OrientationDict(keys=["image3d"], axcodes="ASL"),
        ClipMinIntensityDict(keys=["image3d"], min_val=-512),
        ScaleIntensityDict(keys=["image3d"], minv=0.0, maxv=1.0),
        ResizeDict(keys=["image3d"], spatial_size=vol_shape, size_mode="longest", mode=["trilinear"], align_corners=True),
        DivisiblePadDict(keys=["image3d"], k=vol_shape, mode="constant", constant_values=0),
        ToTensorDict(keys=["image3d"]),
    ])


def generate_multiview_projections(
    volume: torch.Tensor,
    num_frames: int = 121,
    img_shape: int = 256,
    device: str = "cuda",
    dist: float = 6.0,
    elev: float = 0.0,
) -> torch.Tensor:
    """
    Generate 360-degree rotational views from a CT volume using DVR.
    
    Args:
        volume: CT volume tensor of shape (1, 1, D, H, W) or (C, D, H, W)
        num_frames: Number of frames for 360-degree rotation (default: 121)
        img_shape: Output image shape
        device: Device to use for rendering
        dist: Camera distance
        elev: Camera elevation angle
    
    Returns:
        Video tensor of shape (1, 1, T, H, W) where T = num_frames
    """
    # Ensure correct shape
    if volume.ndim == 4:
        volume = volume.unsqueeze(0)  # Add batch dimension
    
    # Move volume to device
    volume = volume.to(device)
    
    # Initialize the renderer
    renderer = ObjectCentricXRayVolumeRenderer(
        image_width=img_shape,
        image_height=img_shape,
        n_pts_per_ray=512,
        min_depth=4.0,
        max_depth=8.0,
        ndc_extent=1.0,
    ).to(device)
    
    # Generate camera views
    frames = []
    azimuths = torch.linspace(0, 360, num_frames)
    
    for azimuth in tqdm(azimuths, desc="Rendering frames", leave=False):
        # Create camera at this azimuth
        R, T = look_at_view_transform(
            dist=dist,
            elev=elev,
            azim=azimuth.item(),
        )
        
        cameras = FoVPerspectiveCameras(
            R=R,
            T=T,
            fov=20.0,
            device=device,
        )
        
        # Render the view
        with torch.no_grad():
            projection = renderer(
                image3d=volume,
                cameras=cameras,
                opacity=None,
                norm_type="minimized",
                scaling_factor=1.0,
                is_grayscale=True,
                return_bundle=False,
            )
        
        frames.append(projection)
    
    # Stack all frames into video
    video = torch.stack(frames, dim=2)  # (1, 1, T, H, W)
    
    return video


def save_video_as_mp4(
    video: torch.Tensor,
    output_path: str,
    fps: int = 30,
):
    """
    Save video tensor as MP4 file.
    
    Args:
        video: Video tensor of shape (1, 1, T, H, W)
        output_path: Path to save the MP4 file
        fps: Frames per second for the video
    """
    # Remove batch and channel dimensions
    video = video.squeeze(0).squeeze(0)  # (T, H, W)
    
    # Convert to numpy and scale to [0, 255]
    video_np = video.cpu().numpy()
    video_np = (video_np * 255).astype(np.uint8)
    
    # Convert to (T, H, W, 1) for grayscale
    video_np = video_np[..., np.newaxis]
    
    # Save as MP4
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimwrite(output_path, video_np, fps=fps, codec='libx264', quality=8)


def save_image_as_png(
    video: torch.Tensor,
    output_path: str,
):
    """
    Save the first frame (0° view) as XR image in PNG format.
    
    Args:
        video: Video tensor of shape (1, 1, T, H, W)
        output_path: Path to save the PNG file
    """
    # Extract first frame (0° azimuth view)
    xr_frame = video[0, 0, 0, :, :]  # (H, W)
    
    # Convert to numpy and scale to [0, 255]
    xr_np = xr_frame.cpu().numpy()
    xr_np = (xr_np * 255).astype(np.uint8)
    
    # Save as PNG
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.imwrite(output_path, xr_np)


def save_volume_as_nifti(
    volume: torch.Tensor,
    output_path: str,
):
    """
    Save the preprocessed CT volume as NIfTI file.
    
    Args:
        volume: Volume tensor of shape (C, D, H, W) or (1, C, D, H, W)
        output_path: Path to save the .nii.gz file
    """
    # Remove batch dimension if present
    if volume.ndim == 5:
        volume = volume.squeeze(0)  # (C, D, H, W)
    
    # Convert to numpy and remove channel dimension if single channel
    volume_np = volume.cpu().numpy()
    if volume_np.shape[0] == 1:
        volume_np = volume_np[0]  # (D, H, W)
    
    # Create NIfTI image
    nifti_img = nib.Nifti1Image(volume_np, affine=np.eye(4))
    
    # Save as .nii.gz
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(nifti_img, output_path)


def generate_text_prompt(ct_path: str) -> str:
    """
    Generate a T5-style text prompt for the CT volume.
    This can be customized based on metadata or other information.
    
    Args:
        ct_path: Path to the CT file
    
    Returns:
        Text prompt string
    """
    # Extract filename without extension
    filename = Path(ct_path).stem.replace('.nii', '')
    
    # Generate a descriptive prompt
    prompt = f"A 360-degree rotational view of a chest CT scan showing anatomical structures from all angles, rotating from 0 to 360 degrees azimuth. Patient: {filename}."
    
    return prompt


def process_ct_dataset(
    ct_paths: List[str],
    project_root: str,
    num_frames: int = 121,
    img_shape: int = 256,
    vol_shape: int = 256,
    device: str = "cuda",
    skip_existing: bool = True,
):
    """
    Process a list of CT volumes and cache them as MP4 videos with prompts.
    Uses the same preprocessing pipeline as the datamodule.
    
    Args:
        ct_paths: List of paths to CT .nii.gz files
        project_root: Project root directory (for cache path generation)
        num_frames: Number of frames for 360-degree rotation
        img_shape: Output image shape
        vol_shape: Volume shape for preprocessing
        device: Device to use for rendering
        skip_existing: Skip files that already exist in cache
    """
    # Create transforms using datamodule's preprocessing
    transforms = create_ct_transforms(vol_shape=vol_shape)
    
    # Process each CT file
    processed = 0
    skipped = 0
    failed = 0
    
    for ct_path in tqdm(ct_paths, desc="Processing CT volumes"):
        # Get cache paths using datamodule's function
        volume_path, video_path, image_path, prompt_path = cache_paths_for_ct(project_root, ct_path)
        
        # Skip if already processed
        if skip_existing and os.path.exists(volume_path) and os.path.exists(video_path) and os.path.exists(image_path) and os.path.exists(prompt_path):
            skipped += 1
            continue
        
        try:
            # Load and preprocess CT volume using datamodule transforms
            data_dict = {"image3d": ct_path}
            data_dict = transforms(data_dict)
            volume = data_dict["image3d"]
            
            # Save preprocessed CT volume as NIfTI
            save_volume_as_nifti(volume, volume_path)
            
            # Generate multiview projections
            video = generate_multiview_projections(
                volume=volume,
                num_frames=num_frames,
                img_shape=img_shape,
                device=device,
            )
            
            # Save video as MP4
            save_video_as_mp4(video, video_path, fps=30)
            
            # Save first frame (0° view) as image PNG
            save_image_as_png(video, image_path)
            
            # Generate and save text prompt
            prompt = generate_text_prompt(ct_path)
            with open(prompt_path, 'w') as f:
                f.write(prompt)
            
            processed += 1
            
        except Exception as e:
            failed += 1
            tqdm.write(f"✗ Error processing {Path(ct_path).name}: {e}")
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Processed: {processed}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")
    print(f"Total:     {len(ct_paths)}")
    print(f"{'='*80}")


def find_ct_files(folders: List[str]) -> List[str]:
    """Find all CT files in the given folders (matches datamodule logic)"""
    all_files = []
    for folder in folders:
        files = glob.glob(os.path.join(folder, "**/*.nii.gz"), recursive=True)
        all_files.extend(files)
    return sorted(all_files)


def main():
    parser = ArgumentParser(description="Cache CT volumes as multiview videos using datamodule preprocessing")
    parser.add_argument("--ct_folders", type=str, nargs="+", required=True, 
                       help="Directories containing CT .nii.gz files (can specify multiple)")
    parser.add_argument("--project_root", type=str, default=".", 
                       help="Project root directory (default: current directory)")
    parser.add_argument("--num_frames", type=int, default=121, 
                       help="Number of frames for 360-degree rotation")
    parser.add_argument("--img_shape", type=int, default=256, 
                       help="Output image shape")
    parser.add_argument("--vol_shape", type=int, default=256, 
                       help="Volume shape for preprocessing")
    parser.add_argument("--device", type=str, default="cuda", 
                       help="Device to use for rendering")
    parser.add_argument("--skip_existing", action="store_true", 
                       help="Skip files that already exist in cache")
    
    args = parser.parse_args()
    
    print("="*80)
    print("CT TO VIDEO CACHE")
    print("="*80)
    print(f"CT folders: {args.ct_folders}")
    print(f"Project root: {args.project_root}")
    print(f"Num frames: {args.num_frames}")
    print(f"Image shape: {args.img_shape}")
    print(f"Volume shape: {args.vol_shape}")
    print(f"Device: {args.device}")
    print(f"Skip existing: {args.skip_existing}")
    print("="*80)
    
    # Find all CT files using datamodule logic
    ct_paths = find_ct_files(args.ct_folders)
    
    print(f"\nFound {len(ct_paths)} CT files")
    if len(ct_paths) > 0:
        print(f"First file: {ct_paths[0]}")
        print(f"Last file: {ct_paths[-1]}")
    
    if len(ct_paths) == 0:
        print("No CT files found. Exiting.")
        return
    
    # Process the dataset
    process_ct_dataset(
        ct_paths=ct_paths,
        project_root=args.project_root,
        num_frames=args.num_frames,
        img_shape=args.img_shape,
        vol_shape=args.vol_shape,
        device=args.device,
        skip_existing=args.skip_existing,
    )
    
    print("\n✓ CT to video caching complete!")


if __name__ == "__main__":
    main()

