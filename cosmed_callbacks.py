"""
Cosmed Callbacks for visualization and monitoring during training.

This module provides PyTorch Lightning callbacks for:
- Generating sample 360° rotation videos during validation
- Visualizing frames at key rotation angles (0°, 90°, 180°, 360°)
- Logging to TensorBoard
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision.io import write_video
from torchvision.utils import make_grid, save_image


class CosmedVisualizationCallback(Callback):
    """
    Callback to generate and log sample 360° rotation videos during validation.
    
    Generates visualizations at key rotation angles:
    - 0° (frontal view)
    - 90° (left lateral view)
    - 180° (posterior view)
    - 360° (back to frontal, should match 0°)
    """
    
    def __init__(
        self,
        num_samples: int = 4,
        log_every_n_epochs: int = 1,
        rotation_angles: List[int] = [0, 90, 180, 360],
        save_to_disk: bool = False,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize the visualization callback.
        
        Args:
            num_samples: Number of samples to visualize per validation
            log_every_n_epochs: Generate samples every N epochs
            rotation_angles: Rotation angles (in degrees) to visualize
            save_to_disk: Whether to save images/videos to disk
            output_dir: Directory to save samples (if save_to_disk is True)
        """
        super().__init__()
        self.num_samples = num_samples
        self.log_every_n_epochs = log_every_n_epochs
        self.rotation_angles = rotation_angles
        self.save_to_disk = save_to_disk
        self.output_dir = Path(output_dir) if output_dir else None
        
        # Create output directory if needed
        if self.save_to_disk and self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def on_validation_epoch_end(
        self, 
        trainer: Trainer, 
        pl_module: LightningModule
    ) -> None:
        """Generate and log samples at the end of validation epoch."""
        # Only log every N epochs
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
        
        # Get validation dataloader
        val_dataloader = trainer.val_dataloaders
        if val_dataloader is None:
            return
        
        # Handle both single dataloader and list of dataloaders
        if isinstance(val_dataloader, list):
            if len(val_dataloader) == 0:
                return
            dataloader = val_dataloader[0]
        else:
            dataloader = val_dataloader
        
        # Set model to eval mode
        pl_module.eval()
        device = pl_module.device
        
        # Collect samples
        ct_samples = []
        xr_samples = []
        
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    if len(ct_samples) >= self.num_samples and len(xr_samples) >= self.num_samples:
                        break
                
                # Process CT samples
                if 'ct_frontal' in batch and 'ct_video' in batch and len(ct_samples) < self.num_samples:
                    ct_frontal = batch['ct_frontal'].to(device=device)
                    ct_video_target = batch['ct_video'].to(device=device)
                    
                    # Generate prediction
                    ct_video_pred = pl_module(ct_frontal, prompt="360 degree rotation of chest CT scan")
                    
                    # Store samples
                    for i in range(min(ct_frontal.shape[0], self.num_samples - len(ct_samples))):
                        ct_samples.append({
                            'input': ct_frontal[i],
                            'target': ct_video_target[i],
                            'prediction': ct_video_pred[i],
                        })
                
                # Process XR samples
                if 'xr_image' in batch and len(xr_samples) < self.num_samples:
                    xr_image = batch['xr_image'].to(device=device)
                    
                    # Generate prediction
                    xr_video_pred = pl_module(xr_image, prompt="360 degree rotation of chest X-ray")
                    
                    # Store samples
                    for i in range(min(xr_image.shape[0], self.num_samples - len(xr_samples))):
                        xr_samples.append({
                            'input': xr_image[i],
                            'prediction': xr_video_pred[i],
                        })
        except Exception as e:
            print(f"Warning: Error collecting samples in callback: {e}")
            return
        
        # Generate visualizations
        try:
            if ct_samples:
                self._visualize_ct_samples(ct_samples, trainer, pl_module, trainer.current_epoch)
            
            if xr_samples:
                self._visualize_xr_samples(xr_samples, trainer, pl_module, trainer.current_epoch)
        except Exception as e:
            print(f"Warning: Error generating visualizations in callback: {e}")
        finally:
            # Always set model back to training mode
            pl_module.train()
    
    def _visualize_ct_samples(
        self,
        samples: List[Dict[str, torch.Tensor]],
        trainer: Trainer,
        pl_module: LightningModule,
        epoch: int,
    ) -> None:
        """Visualize CT rotation samples."""
        for sample_idx, sample in enumerate(samples):
            source_image = sample['input']  # (1, H, W)
            target_video = sample['target']  # (T, H, W)
            output_video = sample['prediction']  # (T, H, W)
            
            # Extract frames at key angles
            frames_to_log = []
            
            # Add input image
            input_frame = source_image.squeeze(0).cpu()  # (H, W)
            frames_to_log.append(('Input (Frontal)', input_frame))
            
            # Add target and predicted frames at each angle
            for angle in self.rotation_angles:
                frame_idx = min(angle, target_video.shape[0] - 1)
                
                target_frame = target_video[frame_idx].cpu()  # (H, W)
                output_frame = output_video[frame_idx].cpu()  # (H, W)
                
                frames_to_log.append((f'Target {angle}°', target_frame))
                frames_to_log.append((f'Pred {angle}°', output_frame))
            
            # Create grid for TensorBoard
            grid_images = []
            for label, frame in frames_to_log:
                # Normalize frame to [0, 1]
                frame_norm = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
                # Convert to 3-channel for visualization (H, W) -> (3, H, W)
                frame_rgb = frame_norm.unsqueeze(0).repeat(3, 1, 1)
                grid_images.append(frame_rgb)
            
            # Create grid
            grid = make_grid(grid_images, nrow=3, padding=2, normalize=False)
            
            # Log to TensorBoard
            for logger in trainer.loggers:
                if isinstance(logger, TensorBoardLogger):
                    logger.experiment.add_image(
                        f'CT_Sample_{sample_idx}/rotation_comparison',
                        grid,
                        global_step=epoch,
                    )
            
            # Save to disk if requested
            if self.save_to_disk and self.output_dir:
                try:
                    epoch_dir = self.output_dir / f"epoch_{epoch:04d}"
                    epoch_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save grid image using torchvision
                    save_image(
                        grid,
                        epoch_dir / f"ct_sample_{sample_idx}_comparison.png"
                    )
                    
                    # Save prediction video
                    self._save_video(
                        output_video,
                        epoch_dir / f"ct_sample_{sample_idx}_prediction.mp4"
                    )
                    
                    # Save target video
                    self._save_video(
                        target_video,
                        epoch_dir / f"ct_sample_{sample_idx}_target.mp4"
                    )
                except Exception as e:
                    print(f"Warning: Failed to save CT sample {sample_idx} to disk: {e}")
    
    def _visualize_xr_samples(
        self,
        samples: List[Dict[str, torch.Tensor]],
        trainer: Trainer,
        pl_module: LightningModule,
        epoch: int,
    ) -> None:
        """Visualize XR rotation samples."""
        for sample_idx, sample in enumerate(samples):
            source_image = sample['input']  # (1, H, W)
            output_video = sample['prediction']  # (T, H, W)
            
            # Extract frames at key angles
            frames_to_log = []
            
            # Add input image
            input_frame = source_image.squeeze(0).cpu()  # (H, W)
            frames_to_log.append(('Input (Frontal)', input_frame))
            
            # Add predicted frames at each angle
            for angle in self.rotation_angles:
                frame_idx = min(angle, output_video.shape[0] - 1)
                output_frame = output_video[frame_idx].cpu()  # (H, W)
                frames_to_log.append((f'Pred {angle}°', output_frame))
            
            # Add difference between 0° and 360° (should be minimal)
            frame_0 = output_video[0].cpu()
            frame_360 = output_video[-1].cpu()
            diff_frame = torch.abs(frame_0 - frame_360)
            frames_to_log.append(('Diff 0°-360°', diff_frame))
            
            # Create grid for TensorBoard
            grid_images = []
            for label, frame in frames_to_log:
                # Normalize frame to [0, 1]
                frame_norm = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
                # Convert to 3-channel for visualization (H, W) -> (3, H, W)
                frame_rgb = frame_norm.unsqueeze(0).repeat(3, 1, 1)
                grid_images.append(frame_rgb)
            
            # Create grid
            grid = make_grid(grid_images, nrow=3, padding=2, normalize=False)
            
            # Log to TensorBoard
            for logger in trainer.loggers:
                if isinstance(logger, TensorBoardLogger):
                    logger.experiment.add_image(
                        f'XR_Sample_{sample_idx}/rotation_consistency',
                        grid,
                        global_step=epoch,
                    )
            
            # Save to disk if requested
            if self.save_to_disk and self.output_dir:
                try:
                    epoch_dir = self.output_dir / f"epoch_{epoch:04d}"
                    epoch_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save grid image using torchvision
                    save_image(
                        grid,
                        epoch_dir / f"xr_sample_{sample_idx}_consistency.png"
                    )
                    
                    # Save prediction video
                    self._save_video(
                        output_video,
                        epoch_dir / f"xr_sample_{sample_idx}_prediction.mp4"
                    )
                except Exception as e:
                    print(f"Warning: Failed to save XR sample {sample_idx} to disk: {e}")
    
    def _save_video(
        self,
        video: torch.Tensor,
        output_path: Path,
        fps: int = 30,
    ) -> None:
        """
        Save video tensor as MP4 file using torchvision.
        
        Args:
            video: Video tensor (T, H, W) - grayscale frames
            output_path: Path to save MP4
            fps: Frames per second for saved video
        """
        try:
            # Normalize video to [0, 1] if needed
            video_norm = video.cpu()
            if video_norm.max() > 1.0 or video_norm.min() < 0.0:
                video_norm = (video_norm - video_norm.min()) / (video_norm.max() - video_norm.min() + 1e-8)
            
            # Convert grayscale to RGB: (T, H, W) -> (T, 3, H, W)
            video_rgb = video_norm.unsqueeze(1).repeat(1, 3, 1, 1)
            
            # Convert to uint8: (T, 3, H, W) with values in [0, 1] -> [0, 255]
            video_uint8 = (video_rgb * 255).clamp(0, 255).to(torch.uint8)
            
            # Rearrange to (T, H, W, 3) for torchvision write_video
            video_uint8 = video_uint8.permute(0, 2, 3, 1)  # (T, H, W, 3)
            
            # Save using torchvision
            write_video(
                str(output_path),
                video_uint8.numpy(),
                fps=fps,
                video_codec='libx264',
                options={'crf': '23'}  # Good quality setting
            )
        except Exception as e:
            print(f"Warning: Failed to save video to {output_path}: {e}")
