import math

import numpy as np
import torch
from pytorch3d.renderer.cameras import (
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    look_at_view_transform,
)

# Try to import Kaolin - optional dependency
try:
    from kaolin.render.camera import (
        Camera,
        CameraExtrinsics,
        ExtrinsicsRep,
        OrthographicIntrinsics,
        PinholeIntrinsics,
    )
    KAOLIN_AVAILABLE = True
except ImportError:
    KAOLIN_AVAILABLE = False
    Camera = None
    CameraExtrinsics = None
    ExtrinsicsRep = None
    OrthographicIntrinsics = None
    PinholeIntrinsics = None


def make_cameras_dea(
    dist: torch.Tensor,  # ndc
    elev: torch.Tensor,  # degree
    azim: torch.Tensor,  # degree
    fov: int = 40,       # degree
    znear: int = 7.0,    # ndc
    zfar: int = 9.0,     # ndc
    width: int = 800,
    height: int = 600,
    is_orthogonal: bool = False,
    return_pytorch3d: bool = True, 
    device: str = "cuda",
):
    """
    Create camera objects with specified parameters.
    
    Args:
        dist: Distance from camera to origin (NDC)
        elev: Elevation angle in degrees
        azim: Azimuth angle in degrees
        fov: Field of view in degrees (default: 40)
        znear: Near clipping plane (NDC, default: 7.0)
        zfar: Far clipping plane (NDC, default: 9.0)
        width: Image width in pixels (default: 800)
        height: Image height in pixels (default: 600)
        is_orthogonal: Use orthographic projection instead of perspective (default: False)
        return_pytorch3d: Return PyTorch3D camera, otherwise return Kaolin camera (default: True)
    
    Returns:
        Camera object (PyTorch3D or Kaolin depending on return_pytorch3d flag)
    
    Raises:
        RuntimeError: If Kaolin is requested but not available
    """
    assert dist.device == elev.device == azim.device == device
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)

    if return_pytorch3d:
        if is_orthogonal:
            return FoVOrthographicCameras(R=R, T=T, znear=znear, zfar=zfar).to(device)
        else:
            return FoVPerspectiveCameras(R=R, T=T, fov=fov, znear=znear, zfar=zfar).to(device)
    else:
        if not KAOLIN_AVAILABLE:
            raise RuntimeError(
                "Kaolin is not available. Install Kaolin to use Kaolin cameras, "
                "or set return_pytorch3d=True to use PyTorch3D cameras."
            )
        
        extrinsics = ExtrinsicsRep._from_world_in_cam_coords(rotation=R, translation=T, device=device)
        if is_orthogonal:
            intrinsics = OrthographicIntrinsics.from_frustum(
                width=width, 
                height=height, 
                near=znear, 
                far=zfar, 
                fov=np.deg2rad(fov), 
                device=device
            )
            return Camera(extrinsics=extrinsics, intrinsics=intrinsics)
        else:
            intrinsics = PinholeIntrinsics.from_fov(
                width=width, 
                height=height, 
                near=znear, 
                far=zfar, 
                fov_distance=1.0, 
                device=device
            )
            return Camera(extrinsics=extrinsics, intrinsics=intrinsics)
