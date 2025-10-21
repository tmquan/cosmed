import glob
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional

import imageio
import nibabel as nib
import numpy as np
import torch
from monai.transforms import (
    Compose,
    DivisiblePadDict,
    EnsureChannelFirstDict,
    LoadImageDict,
    OrientationDict,
    ResizeDict,
    ScaleIntensityDict,
    SpacingDict,
    ToTensorDict,
)
from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    track,
)

from cm_datamodule import ClipMinIntensityDict, cache_paths_for_ct
from dvr.renderer import ObjectCentricXRayVolumeRenderer


def create_ct_transforms(vol_shape: int = 256) -> Compose:
    """
    Create transforms for CT preprocessing (matches datamodule).

    Args:
        vol_shape: Target vol shape (isotropic)

    Returns:
        Composed transforms for CT preprocessing
    """
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
    vol: torch.Tensor,
    num_frames: int = 121,
    img_shape: int = 256,
    device: str = "cuda",
    dist: float = 6.0,
    elev: float = 0.0,
) -> torch.Tensor:
    """
    Generate 360-degree rotational views from a CT vol using DVR.

    Args:
        vol: CT vol tensor of shape (1, 1, D, H, W) or (C, D, H, W)
        num_frames: Number of frames for 360-degree rotation (default: 121)
        img_shape: Output image shape
        device: Device to use for rendering
        dist: Camera distance
        elev: Camera elevation angle

    Returns:
        Video tensor of shape (1, 1, T, H, W) where T = num_frames
    """
    # Ensure correct shape
    if vol.ndim == 4:
        vol = vol.unsqueeze(0)  # Add batch dimension
    
    # Move vol to device
    vol = vol.to(device)
    
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
    
    for azimuth in track(azimuths, description="Rendering frames", transient=True):
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
                image3d=vol,
                cameras=cameras,
                opacity=None,
                norm_type="minimized",
                scaling_factor=1.0,
                is_grayscale=True,
                return_bundle=False,
            )
        
        frames.append(projection)
    
    # Stack all frames into vid
    vid = torch.stack(frames, dim=2)  # (1, 1, T, H, W)
    
    return vid


def save_vid_as_mp4(
    vid: torch.Tensor,
    out: str,
    fps: int = 30,
) -> None:
    """
    Save vid tensor as MP4 file.

    Args:
        vid: Video tensor of shape (1, 1, T, H, W)
        out: Path to save the MP4 file
        fps: Frames per second for the vid
    """
    # Remove batch and channel dimensions
    vid = vid.squeeze(0).squeeze(0)  # (T, H, W)
    
    # Convert to numpy and scale to [0, 255]
    video_np = vid.cpu().numpy()
    video_np = (video_np * 255).astype(np.uint8)
    
    # Convert to (T, H, W, 1) for grayscale
    video_np = video_np[..., np.newaxis]
    
    # Save as MP4
    os.makedirs(os.path.dirname(out), exist_ok=True)
    imageio.mimwrite(out, video_np, fps=fps, codec='libx264', quality=8)


def save_img_as_png(
    vid: torch.Tensor,
    out: str,
) -> None:
    """
    Save the first frame (0° view) as XR image in PNG format.

    Args:
        vid: Video tensor of shape (1, 1, T, H, W)
        out: Path to save the PNG file
    """
    # Extract first frame (0° azimuth view)
    xr_frame = vid[0, 0, 0, :, :]  # (H, W)
    
    # Convert to numpy and scale to [0, 255]
    xr_np = xr_frame.cpu().numpy()
    xr_np = (xr_np * 255).astype(np.uint8)
    
    # Save as PNG
    os.makedirs(os.path.dirname(out), exist_ok=True)
    imageio.imwrite(out, xr_np)


def save_vol_as_nifti(
    vol: torch.Tensor,
    out: str,
) -> None:
    """
    Save the preprocessed CT vol as NIfTI file.

    Args:
        vol: vol tensor of shape (C, D, H, W) or (1, C, D, H, W)
        out: Path to save the .nii.gz file
    """
    # Remove batch dimension if present
    if vol.ndim == 5:
        vol = vol.squeeze(0)  # (C, D, H, W)
    
    # Convert to numpy and remove channel dimension if single channel
    vol_np = vol.cpu().numpy()
    if vol_np.shape[0] == 1:
        vol_np = vol_np[0]  # (D, H, W)
    
    # Create NIfTI image
    nifti_img = nib.Nifti1Image(vol_np, affine=np.eye(4))
    
    # Save as .nii.gz
    os.makedirs(os.path.dirname(out), exist_ok=True)
    nib.save(nifti_img, out)


def generate_txt_prompt(ct_path: str) -> str:
    """
    Generate a T5-style txt prompt for the CT vol.

    This can be customized based on metadata or other information.

    Args:
        ct_path: Path to the CT file

    Returns:
        txt prompt string
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
) -> None:
    """
    Process a list of CT vols and cache them as MP4 videos with prompts.

    Uses the same preprocessing pipeline as the datamodule.

    Args:
        ct_paths: List of paths to CT .nii.gz files
        project_root: Project root directory (for cache path generation)
        num_frames: Number of frames for 360-degree rotation
        img_shape: Output image shape
        vol_shape: vol shape for preprocessing
        device: Device to use for rendering
        skip_existing: Skip files that already exist in cache
    """
    # Create transforms using datamodule's preprocessing
    transforms = create_ct_transforms(vol_shape=vol_shape)
    
    # Process each CT file
    processed = 0
    skipped = 0
    failed = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Processing CT vols...", total=len(ct_paths))
        
        for ct_path in ct_paths:
            # Get cache paths using datamodule's function
            vol_path, vid_path, img_path, prompt_path = cache_paths_for_ct(project_root, ct_path)
            
            # Skip if already processed
            if skip_existing and os.path.exists(vol_path) and os.path.exists(vid_path) and os.path.exists(img_path) and os.path.exists(prompt_path):
                skipped += 1
                progress.advance(task)
                continue
            
            try:
                # Load and preprocess CT vol using datamodule transforms
                data_dict = {"image3d": ct_path}
                data_dict = transforms(data_dict)
                vol = data_dict["image3d"]
                
                # Save preprocessed CT vol as NIfTI
                save_vol_as_nifti(vol, vol_path)
                
                # Generate multiview projections
                vid = generate_multiview_projections(
                    vol=vol,
                    num_frames=num_frames,
                    img_shape=img_shape,
                    device=device,
                )
                
                # Save vid as MP4
                save_vid_as_mp4(vid, vid_path, fps=30)
                
                # Save first frame (0° view) as image PNG
                save_img_as_png(vid, img_path)
                
                # Generate and save txt prompt
                prompt = generate_txt_prompt(ct_path)
                with open(prompt_path, 'w') as f:
                    f.write(prompt)
                
                processed += 1
                
            except Exception as e:
                failed += 1
                progress.console.print(f"[red]✗ Error processing {Path(ct_path).name}: {e}[/red]")
            
            progress.advance(task)
    
    console = Console()
    console.print(f"\n{'='*80}")
    console.print("[bold cyan]SUMMARY[/bold cyan]")
    console.print(f"{'='*80}")
    console.print(f"[green]Processed:[/green] {processed}")
    console.print(f"[yellow]Skipped:[/yellow]   {skipped}")
    console.print(f"[red]Failed:[/red]    {failed}")
    console.print(f"[bold]Total:[/bold]     {len(ct_paths)}")
    console.print(f"{'='*80}")


def glob_files(folders: List[str], extension: str = "*.nii.gz") -> List[str]:
    """Glob files from multiple folders with given extension pattern."""
    assert folders is not None, "folders parameter cannot be None"
    paths = [
        glob.glob(os.path.join(folder, extension), recursive=True)
        for folder in folders
    ]
    files = sorted([item for sublist in paths for item in sublist])
    print(len(files))
    print(files[:1])
    return files


def main() -> None:
    """Main function to cache CT vols as multiview videos."""
    parser = ArgumentParser(description="Cache CT vols as multiview videos using datamodule preprocessing")
    parser.add_argument("--datadir", type=str, default="/workspace/data", 
                       help="Data directory (default: /workspace/data)")
    parser.add_argument("--project_root", type=str, default=".", 
                       help="Project root directory (default: current directory)")
    parser.add_argument("--num_frames", type=int, default=121, 
                       help="Number of frames for 360-degree rotation")
    parser.add_argument("--img_shape", type=int, default=256, 
                       help="Output image shape")
    parser.add_argument("--vol_shape", type=int, default=256, 
                       help="vol shape for preprocessing")
    parser.add_argument("--device", type=str, default="cuda", 
                       help="Device to use for rendering")
    parser.add_argument("--skip_existing", action="store_true", 
                       help="Skip files that already exist in cache")
    
    args = parser.parse_args()
    
    console = Console()
    
    # Set default CT folders if not provided (matches cm_datamodule.py)
    ct_folders = [
        os.path.join(args.datadir, "ChestXRLungSegmentation/NSCLC/processed/train/images"),
        os.path.join(args.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-0"),
        os.path.join(args.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-1"),
        os.path.join(args.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-2"),
        os.path.join(args.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-3"),
        os.path.join(args.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-4"),
        os.path.join(args.datadir, "ChestXRLungSegmentation/Imagenglab/processed/train/images"),
        os.path.join(args.datadir, "ChestXRLungSegmentation/TCIA/images/"),
    ]
    
    console.print("="*80)
    console.print("[bold magenta]CT TO VIDEO CACHE[/bold magenta]")
    console.print("="*80)
    console.print(f"[cyan]Data directory:[/cyan] {args.datadir}")
    console.print(f"[cyan]Project root:[/cyan] {args.project_root}")
    console.print(f"[cyan]Num frames:[/cyan] {args.num_frames}")
    console.print(f"[cyan]Image shape:[/cyan] {args.img_shape}")
    console.print(f"[cyan]vol shape:[/cyan] {args.vol_shape}")
    console.print(f"[cyan]Device:[/cyan] {args.device}")
    console.print(f"[cyan]Skip existing:[/cyan] {args.skip_existing}")
    console.print("="*80)
    
    # Find all CT files using datamodule logic
    ct_paths = glob_files(ct_folders)
    
    console.print(f"\n[bold]Found {len(ct_paths)} CT files[/bold]")
    if len(ct_paths) > 0:
        console.print(f"[dim]First file: {ct_paths[0]}[/dim]")
        console.print(f"[dim]Last file: {ct_paths[-1]}[/dim]")
    
    if len(ct_paths) == 0:
        console.print("[red]No CT files found. Exiting.[/red]")
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
    
    console.print("\n[bold green]✓ CT to vid caching complete![/bold green]")


if __name__ == "__main__":
    main()
