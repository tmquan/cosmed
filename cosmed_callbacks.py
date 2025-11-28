"""
Cosmed Callbacks for visualization and monitoring during training.

This module provides PyTorch Lightning callbacks for:
- Collecting and displaying metrics in Rich progress bar
- Generating sample 360° rotation videos during validation
- Visualizing frames at key rotation angles (0°, 90°, 180°, 360°)
- Logging to TensorBoard
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision.io import write_video
from torchvision.utils import make_grid, save_image

# Suppress all warnings
warnings.filterwarnings('ignore')


class CosmedMetricsCallback(Callback):
    """
    Callback to collect and format metrics for Rich progress bar display.
    
    Ensures key metrics are properly logged and displayed during training.
    """
    
    def __init__(
        self,
        metrics_to_display: Optional[List[str]] = None,
    ):
        """
        Initialize the metrics callback.
        
        Args:
            metrics_to_display: List of metric keys to display in progress bar.
                               If None, uses default set of important metrics.
        """
        super().__init__()
        
        # Default metrics to display in progress bar
        if metrics_to_display is None:
            self.metrics_to_display = [
                'train/total_loss',
                'train/ct_video_loss',
                'train/xr_input_loss',
                'train/xr_cycle_loss',
                'val/total_loss',
                'val/ct_video_loss',
                'val/xr_input_loss',
                'val/xr_cycle_loss',
            ]
        else:
            self.metrics_to_display = metrics_to_display
        
        # Track metrics across epochs
        self.metrics_history: Dict[str, List[float]] = {}
    
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Ensure training metrics are logged with prog_bar=True."""
        # Metrics are already logged in training_step with prog_bar=True
        # This callback can be used to track or modify metrics if needed
        pass
    
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Ensure validation metrics are logged with prog_bar=True."""
        # Metrics are already logged in validation_step with prog_bar=True
        pass
    
    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Collect training metrics at end of epoch."""
        # Store epoch metrics for tracking
        logged_metrics = trainer.callback_metrics
        for metric_key in self.metrics_to_display:
            if metric_key in logged_metrics:
                value = logged_metrics[metric_key]
                if isinstance(value, torch.Tensor):
                    value = value.item()
                
                if metric_key not in self.metrics_history:
                    self.metrics_history[metric_key] = []
                self.metrics_history[metric_key].append(value)
    
    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Collect validation metrics at end of epoch."""
        # Store epoch metrics for tracking
        logged_metrics = trainer.callback_metrics
        for metric_key in self.metrics_to_display:
            if metric_key in logged_metrics:
                value = logged_metrics[metric_key]
                if isinstance(value, torch.Tensor):
                    value = value.item()
                
                if metric_key not in self.metrics_history:
                    self.metrics_history[metric_key] = []
                self.metrics_history[metric_key].append(value)


class CosmedRichProgressBar(RichProgressBar):
    """
    Custom Rich Progress Bar that displays Cosmed-specific metrics.
    
    Extends Lightning's RichProgressBar to customize metric display.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize custom rich progress bar."""
        super().__init__(*args, **kwargs)
        
        # Metrics to prioritize in display (will be shown first)
        # Include both step and epoch metrics
        self.priority_metrics = [
            'train/total_loss',
            'train/total_loss_step',
            'train/ct_video_loss',
            'train/ct_video_loss_step',
            'train/edm_loss',
            'train/edm_loss_step',
            'train/xr_input_loss',
            'train/xr_input_loss_step',
            'train/xr_cycle_loss',
            'train/xr_cycle_loss_step',
            'val/total_loss',
            'val/ct_video_loss',
            'val/xr_input_loss',
            'val/xr_cycle_loss',
        ]
    
    def get_metrics(self, trainer: Trainer, pl_module: LightningModule) -> Dict[str, Any]:
        """
        Get metrics to display in progress bar.
        
        Override to customize which metrics are shown and their order.
        """
        metrics = super().get_metrics(trainer, pl_module)
        
        # Reorder metrics to show priority metrics first
        if metrics:
            ordered_metrics = {}
            
            # Add priority metrics first (only if they exist)
            for metric_key in self.priority_metrics:
                if metric_key in metrics:
                    ordered_metrics[metric_key] = metrics[metric_key]
            
            # Add remaining metrics (including all step-level metrics)
            for metric_key, value in metrics.items():
                if metric_key not in ordered_metrics:
                    ordered_metrics[metric_key] = value
            
            return ordered_metrics
        
        return metrics


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
        rotation_angles: List[int] = [0, 90, 180, 270, 360],
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
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                for batch_idx, batch in enumerate(dataloader):
                    if len(ct_samples) >= self.num_samples and len(xr_samples) >= self.num_samples:
                        break
                    
                    # Process CT samples - one at a time to avoid batch size mismatch
                    if 'ct_front' in batch and 'ct_video' in batch and len(ct_samples) < self.num_samples:
                        ct_front = batch['ct_front'].to(device=device, dtype=torch.float32)
                        ct_video_target = batch['ct_video'].to(device=device, dtype=torch.float32)
                        
                        # Process one sample at a time to avoid batch size issues
                        for i in range(min(ct_front.shape[0], self.num_samples - len(ct_samples))):
                            single_input = ct_front[i:i+1]  # Keep batch dim (1, C, H, W)
                            
                            # Generate prediction
                            if hasattr(pl_module, 'predict_step'):
                                pred_batch = {'ct_front': single_input}
                                pred_results = pl_module.predict_step(pred_batch, batch_idx)
                                ct_video_pred = pred_results.get('ct_video', None)
                            else:
                                ct_video_pred = pl_module(single_input, prompt="360 degree rotation of chest CT scan")
                            
                            if ct_video_pred is not None:
                                if ct_video_pred.dtype != torch.float32:
                                    ct_video_pred = ct_video_pred.to(dtype=torch.float32)
                                
                                # Remove batch dim if present
                                if ct_video_pred.ndim >= 4 and ct_video_pred.shape[0] == 1:
                                    ct_video_pred = ct_video_pred[0]
                                
                                ct_samples.append({
                                    'input': ct_front[i],
                                    'target': ct_video_target[i] if ct_video_target is not None else None,
                                    'prediction': ct_video_pred,
                                })
                
                    # Process XR samples - one at a time to avoid batch size mismatch
                    if 'xr_image' in batch and len(xr_samples) < self.num_samples:
                        xr_image = batch['xr_image'].to(device=device, dtype=torch.float32)
                        
                        # Process one sample at a time to avoid batch size issues
                        for i in range(min(xr_image.shape[0], self.num_samples - len(xr_samples))):
                            single_input = xr_image[i:i+1]  # Keep batch dim (1, C, H, W)
                            
                            # Generate prediction
                            if hasattr(pl_module, 'predict_step'):
                                pred_batch = {'xr_image': single_input}
                                pred_results = pl_module.predict_step(pred_batch, batch_idx)
                                xr_video_pred = pred_results.get('xr_video', None)
                            else:
                                xr_video_pred = pl_module(single_input, prompt="360 degree rotation of chest X-ray")
                            
                            if xr_video_pred is not None:
                                if xr_video_pred.dtype != torch.float32:
                                    xr_video_pred = xr_video_pred.to(dtype=torch.float32)
                                
                                # Remove batch dim if present
                                if xr_video_pred.ndim >= 4 and xr_video_pred.shape[0] == 1:
                                    xr_video_pred = xr_video_pred[0]
                                
                                xr_samples.append({
                                    'input': xr_image[i],
                                    'prediction': xr_video_pred,
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
    
    def _normalize_video(self, video: torch.Tensor) -> torch.Tensor:
        """Normalize video to (T, H, W) grayscale format.
        
        Handles various input shapes:
        - (T, H, W) -> keep as is
        - (C, T, H, W) -> take first channel, transpose to (T, H, W)
        - (B, C, T, H, W) -> take first batch, first channel
        - (B, T, H, W) -> take first batch
        """
        if video.ndim == 5:  # (B, C, T, H, W)
            video = video[0]  # (C, T, H, W)
        if video.ndim == 4:  # (C, T, H, W) or (B, T, H, W)
            if video.shape[0] in [1, 3]:  # Likely (C, T, H, W)
                video = video[0]  # Take first channel -> (T, H, W)
            else:  # Likely (B, T, H, W)
                video = video[0]  # Take first batch -> (T, H, W)
        return video  # (T, H, W)
    
    def _frame_to_rgb(self, frame: torch.Tensor) -> torch.Tensor:
        """Convert a frame to RGB (3, H, W) format.
        
        Handles various input shapes:
        - (H, W) -> expand to (3, H, W)
        - (1, H, W) -> expand to (3, H, W)
        - (3, H, W) -> keep as is
        - (C, H, W) where C > 3 -> take first 3 channels
        """
        frame = frame.cpu().float()
        
        # Normalize to [0, 1]
        frame_norm = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
        
        if frame_norm.ndim == 2:  # (H, W)
            frame_rgb = frame_norm.unsqueeze(0).expand(3, -1, -1)  # (3, H, W)
        elif frame_norm.ndim == 3:
            if frame_norm.shape[0] == 1:  # (1, H, W)
                frame_rgb = frame_norm.expand(3, -1, -1)  # (3, H, W)
            elif frame_norm.shape[0] == 3:  # (3, H, W)
                frame_rgb = frame_norm
            else:  # (C, H, W) with C > 3
                frame_rgb = frame_norm[:3]  # Take first 3 channels
        else:
            raise ValueError(f"Unexpected frame shape: {frame_norm.shape}")
        
        return frame_rgb
    
    def _visualize_ct_samples(
        self,
        samples: List[Dict[str, torch.Tensor]],
        trainer: Trainer,
        pl_module: LightningModule,
        epoch: int,
    ) -> None:
        """Visualize CT rotation samples."""
        for sample_idx, sample in enumerate(samples):
            source_image = sample['input']  # (1, H, W) or (C, H, W)
            target_video = sample.get('target', None)  # May be various shapes
            output_video = sample['prediction']  # May be various shapes
            
            # Normalize video shapes to (T, H, W)
            output_video = self._normalize_video(output_video)
            if target_video is not None:
                target_video = self._normalize_video(target_video)
            
            # Map angles to frame indices (assuming 121 frames for 360°)
            num_frames = output_video.shape[0]
            angle_to_frame = lambda angle: int((angle / 360.0) * (num_frames - 1))
            
            # Extract target frames at each angle (if available)
            target_frames = []
            if target_video is not None:
                target_num_frames = target_video.shape[0]
                for angle in self.rotation_angles:
                    frame_idx = int((angle / 360.0) * (target_num_frames - 1))
                    frame_idx = min(frame_idx, target_num_frames - 1)
                    target_frames.append(target_video[frame_idx])  # (H, W)
            
            # Extract predicted frames at each angle
            pred_frames = []
            for angle in self.rotation_angles:
                frame_idx = angle_to_frame(angle)
                frame_idx = min(frame_idx, num_frames - 1)
                pred_frames.append(output_video[frame_idx])  # (H, W)
            
            # Create grid: [Input, Target frames (if available), Predicted frames]
            grid_images = []
            
            # First: Input image
            input_rgb = self._frame_to_rgb(source_image.squeeze(0))
            grid_images.append(input_rgb)
            
            # Add target frames (if available)
            for frame in target_frames:
                grid_images.append(self._frame_to_rgb(frame))
            
            # Add predicted frames
            for frame in pred_frames:
                grid_images.append(self._frame_to_rgb(frame))
            
            # Create grid: nrow = number of angles + 1 (for input)
            # Layout: [Input, Target 0°, Target 90°, ..., Target 360°, Pred 0°, Pred 90°, ..., Pred 360°]
            # Or if no target: [Input, Pred 0°, Pred 90°, ..., Pred 360°]
            nrow = len(self.rotation_angles) + 1 if target_frames else len(self.rotation_angles) + 1
            grid = make_grid(grid_images, nrow=nrow, padding=2, normalize=False)
            
            # Log to TensorBoard
            for logger in trainer.loggers:
                if isinstance(logger, TensorBoardLogger):
                    # Log image grid with frames at key angles
                    logger.experiment.add_image(
                        f'CT_Sample_{sample_idx}/rotation_frames',
                        grid,
                        global_step=epoch,
                    )
                    
                    # Log individual frames at key rotation angles
                    for angle, frame in zip(self.rotation_angles, pred_frames):
                        frame_rgb = self._frame_to_rgb(frame)
                        logger.experiment.add_image(
                            f'CT_Sample_{sample_idx}/pred_{angle:03d}deg',
                            frame_rgb,
                            global_step=epoch,
                        )
                    
                    # Log target frames if available
                    if target_frames:
                        for angle, frame in zip(self.rotation_angles, target_frames):
                            frame_rgb = self._frame_to_rgb(frame)
                            logger.experiment.add_image(
                                f'CT_Sample_{sample_idx}/target_{angle:03d}deg',
                                frame_rgb,
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
                    
                    # Save target video (if available)
                    if target_video is not None:
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
            source_image = sample['input']  # (1, H, W) or (C, H, W)
            output_video = sample['prediction']  # May be various shapes
            
            # Normalize video shape to (T, H, W)
            output_video = self._normalize_video(output_video)
            
            # Map angles to frame indices (assuming 121 frames for 360°)
            num_frames = output_video.shape[0]
            angle_to_frame = lambda angle: int((angle / 360.0) * (num_frames - 1))
            
            # Extract frames at key angles: 0°, 90°, 180°, 270°, 360°
            pred_frames = []
            for angle in self.rotation_angles:
                frame_idx = angle_to_frame(angle)
                frame_idx = min(frame_idx, num_frames - 1)
                pred_frames.append(output_video[frame_idx])  # (H, W)
            
            # Create grid: [Input, Predicted frames at key angles]
            grid_images = []
            
            # First: Input image
            input_rgb = self._frame_to_rgb(source_image.squeeze(0))
            grid_images.append(input_rgb)
            
            # Add predicted frames at each angle
            for frame in pred_frames:
                grid_images.append(self._frame_to_rgb(frame))
            
            # Create grid: nrow = number of angles + 1 (for input)
            # Layout: [Input, Pred 0°, Pred 90°, Pred 180°, Pred 270°, Pred 360°]
            grid = make_grid(grid_images, nrow=len(self.rotation_angles) + 1, padding=2, normalize=False)
            
            # Log to TensorBoard
            for logger in trainer.loggers:
                if isinstance(logger, TensorBoardLogger):
                    # Log image grid with frames at key angles
                    logger.experiment.add_image(
                        f'XR_Sample_{sample_idx}/rotation_frames',
                        grid,
                        global_step=epoch,
                    )
                    
                    # Log individual frames at key rotation angles
                    for angle, frame in zip(self.rotation_angles, pred_frames):
                        frame_rgb = self._frame_to_rgb(frame)
                        logger.experiment.add_image(
                            f'XR_Sample_{sample_idx}/pred_{angle:03d}deg',
                            frame_rgb,
                            global_step=epoch,
                        )
                    
                    # Log video: prepare video tensor for TensorBoard
                    # TensorBoard expects (N, T, C, H, W) format
                    output_video_tb = output_video.cpu().unsqueeze(0).unsqueeze(2)  # (1, T, 1, H, W)
                    
                    # Normalize to [0, 1] for TensorBoard
                    output_video_tb = (output_video_tb - output_video_tb.min()) / (output_video_tb.max() - output_video_tb.min() + 1e-8)
                    
                    # Log predicted video
                    logger.experiment.add_video(
                        f'XR_Sample_{sample_idx}/predicted_video',
                        output_video_tb,
                        global_step=epoch,
                        fps=10,  # Lower fps for TensorBoard viewing
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
            video: Video tensor - various shapes supported
            output_path: Path to save MP4
            fps: Frames per second for saved video
        """
        try:
            # Normalize video shape to (T, H, W)
            video = self._normalize_video(video)
            
            # Normalize video to [0, 1] if needed
            video_norm = video.cpu().float()
            if video_norm.max() > 1.0 or video_norm.min() < 0.0:
                video_norm = (video_norm - video_norm.min()) / (video_norm.max() - video_norm.min() + 1e-8)
            
            # Convert grayscale to RGB: (T, H, W) -> (T, 3, H, W)
            video_rgb = video_norm.unsqueeze(1).expand(-1, 3, -1, -1)
            
            # Convert to uint8: (T, 3, H, W) with values in [0, 1] -> [0, 255]
            video_uint8 = (video_rgb * 255).clamp(0, 255).to(torch.uint8)
            
            # Rearrange to (T, H, W, 3) for torchvision write_video
            video_uint8 = video_uint8.permute(0, 2, 3, 1)  # (T, H, W, 3)
            
            # Save using torchvision (warnings are suppressed at module level)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                write_video(
                    str(output_path),
                    video_uint8.numpy(),
                    fps=fps,
                    video_codec='libx264',
                    options={'crf': '23'}  # Good quality setting
                )
        except Exception as e:
            print(f"Warning: Failed to save video to {output_path}: {e}")
