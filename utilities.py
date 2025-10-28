import os

import imageio
import nibabel as nib
import numpy as np
import torch
from torchvision.io import write_video


def save_vid_as_mp4(
    vid: torch.Tensor,
    out: str,
    fps: int = 30,
) -> None:
    """
    Save vid tensor as MP4 file using torchvision.

    Args:
        vid: Video tensor of shape (1, 1, T, H, W)
        out: Path to save the MP4 file
        fps: Frames per second for the vid
    """
    # Remove batch and channel dimensions
    vid = vid.squeeze(0).squeeze(0)  # (T, H, W)
    
    # Scale to [0, 255] and convert to uint8
    vid = (vid * 255).clamp(0, 255).to(torch.uint8)
    
    # Convert from (T, H, W) to (T, H, W, 3) for RGB (torchvision requires 3 channels)
    vid = vid.unsqueeze(-1).repeat(1, 1, 1, 3)  # (T, H, W, 3)
    
    # Save as MP4 using torchvision
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_video(out, vid.cpu(), fps=fps, video_codec='libx264', options={'crf': '23'})


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

