"""
Cosmed Model Wrapper using Cosmos-Predict2.5 Video2World Pipeline.

This wrapper uses the Video2World model from cosmos-predict2.5 for:
1. CT frontal projection -> 360° rotation video (121 frames)
2. Real XR frontal image -> 360° rotation video with cycle consistency (121 frames)

This wrapper follows PyTorch Lightning module structure and leverages MAXIMALLY
the implementation in cosmos-predict2.5 including pipeline, model loading, and inference.

Usage:
    Inside Docker container with cosmos-predict2.5 mounted at /workspace/cosmos-predict2.5
    
    # For inference:
    model = CosmedModelWrapper(num_frames=121)
    video = model.generate(input_image, prompt="360 degree rotation of chest CT scan")
    
    # For training (uses Cosmos's native training infrastructure):
    See cosmed_train.py for training script
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from lightning import LightningModule

# Add cosmos-predict2.5 to PYTHONPATH
COSMOS_ROOT = os.getenv("COSMOS_ROOT", "/workspace/cosmos-predict2.5")
if COSMOS_ROOT not in sys.path:
    sys.path.insert(0, COSMOS_ROOT)

# Import cosmos-predict2.5 modules
from cosmos_predict2._src.imaginaire.lazy_config import LazyConfig
from cosmos_predict2._src.imaginaire.utils import distributed, log, misc
from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

# =============================================================================
# Default Configuration Constants
# =============================================================================

# Default checkpoint (pre-trained 2B model)
DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]

# Default negative prompt for classifier-free guidance
DEFAULT_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, "
    "motion blur, over-saturation, shaky footage, low resolution, grainy texture, "
    "pixelated images, poorly lit areas, underexposed and overexposed scenes, "
    "poor color balance, washed out colors, choppy sequences, jerky movements, "
    "low frame rate, artifacting, color banding, unnatural transitions, "
    "outdated special effects, fake elements, unconvincing visuals, "
    "poorly edited content, jump cuts, visual noise, and flickering. "
    "Overall, the video is of poor quality."
)

# Default positive prompts for medical imaging
DEFAULT_CT_PROMPT = "360 degree rotation of chest CT scan, medical imaging, high quality"
DEFAULT_XR_PROMPT = "360 degree rotation of chest X-ray, medical imaging, high quality"

# Default config file path
DEFAULT_CONFIG_FILE = "cosmos_predict2/_src/predict2/configs/video2world/config.py"


class CosmedModelWrapper(LightningModule):
    """
    PyTorch Lightning wrapper for Cosmos-Predict2.5 Video2World model.
    
    This wrapper provides two modes of operation:
    
    1. INFERENCE MODE: Uses Video2WorldInference pipeline for generating 360° rotation videos
       - Image2World: Single frontal image -> 121-frame 360° rotation video
       - Video2World: Input video -> Extended/predicted video (121 frames)
    
    2. TRAINING MODE: Uses Cosmos's native diffusion training with custom data
       - Leverages ImaginaireTrainer and native training loop
       - Supports LoRA fine-tuning for memory efficiency
       - Compatible with PyTorch Lightning training loop
    
    Args:
        checkpoint_path: Path to model checkpoint (None for auto-download from HF)
        config_file: Path to config file for cosmos-predict2.5
        num_frames: Number of output frames (default: 121 for 360° rotation)
        resolution: Output resolution (H, W) (default: (256, 256))
        learning_rate: Learning rate for training
        weight_decay: Weight decay for optimizer
        guidance_scale: Classifier-free guidance scale
        num_inference_steps: Number of denoising steps
        context_parallel_size: Context parallel size for distributed inference
        use_ema: Whether to use EMA model for inference
        offload_diffusion_model: Offload diffusion model to CPU
        offload_text_encoder: Offload text encoder to CPU
        offload_tokenizer: Offload tokenizer to CPU
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config_file: str = DEFAULT_CONFIG_FILE,
        num_frames: int = 121,
        resolution: tuple[int, int] = (256, 256),
        learning_rate: float = 2 ** (-14.5),
        weight_decay: float = 0.001,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 35,
        context_parallel_size: int = 1,
        use_ema: bool = True,
        offload_diffusion_model: bool = False,
        offload_text_encoder: bool = False,
        offload_tokenizer: bool = False,
    ):
        super().__init__()
        
        # Store configuration
        self.checkpoint_path = checkpoint_path
        self.config_file = config_file
        self.num_frames = num_frames
        self.resolution = list(resolution)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.context_parallel_size = context_parallel_size
        self.use_ema = use_ema
        self.offload_diffusion_model = offload_diffusion_model
        self.offload_text_encoder = offload_text_encoder
        self.offload_tokenizer = offload_tokenizer
        
        # Save hyperparameters for checkpoint restoration
        self.save_hyperparameters()
        
        # Model components (initialized in setup)
        self.model = None  # The cosmos diffusion model
        self.cosmos_config = None  # The model config from cosmos
        self.inference_pipe = None  # Video2WorldInference pipeline
        
        # Training state
        self._is_setup = False
        
        # Setup environment variables for cosmos-predict2.5
        self._setup_environment()
        
        log.info(f"CosmedModelWrapper initialized with {self.num_frames} frames at resolution {self.resolution}")
    
    def _setup_environment(self):
        """Set up environment variables for Cosmos-Predict2.5."""
        hf_cache_dir = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        os.environ.setdefault("COSMOS_INTERNAL", "0")
        log.debug(f"COSMOS_ROOT: {COSMOS_ROOT}")
        log.debug(f"HF_HOME: {hf_cache_dir}")
    
    def _get_checkpoint_path(self) -> str:
        """Get checkpoint path, auto-downloading from HuggingFace if needed."""
        if self.checkpoint_path is not None:
            return self.checkpoint_path
        
        from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
        checkpoint = get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri)
        log.info(f"Using default checkpoint from HuggingFace: {checkpoint}")
        return checkpoint
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup model for training/validation/testing.
        This is called on every process when using DDP.
        """
        if self._is_setup:
            return
        
        log.info(f"Setting up CosmedModelWrapper for stage: {stage}")
        ckpt_path = self._get_checkpoint_path()
        
        if stage in ['fit', 'validate', 'test'] or stage is None:
            self._setup_model_for_training(ckpt_path)
        elif stage == 'predict':
            self._setup_inference_pipeline(ckpt_path)
        
        self._is_setup = True
        log.success(f"Model setup complete on device: {self.device}")
    
    def _setup_model_for_training(self, ckpt_path: str):
        """Setup model for training using cosmos's model loader."""
        log.info(f"Loading model for training from: {ckpt_path}")
        
        # Use experiment name from checkpoint (guaranteed to be registered)
        experiment_name = DEFAULT_CHECKPOINT.experiment
        log.info(f"Using experiment from checkpoint: {experiment_name}")
        
        self.model, self.cosmos_config = load_model_from_checkpoint(
            experiment_name=experiment_name,
            s3_checkpoint_dir=ckpt_path,
            config_file=self.config_file,
            enable_fsdp=False,
            load_ema_to_reg=True,
            instantiate_ema=self.use_ema,
            experiment_opts=["~data_train"],
        )
        
        log.info(f"Model loaded with config: {self.cosmos_config.job.name if hasattr(self.cosmos_config, 'job') else 'unknown'}")
    
    def _setup_inference_pipeline(self, ckpt_path: str):
        """Setup inference pipeline using Video2WorldInference."""
        log.info(f"Setting up inference pipeline from: {ckpt_path}")
        
        experiment_name = DEFAULT_CHECKPOINT.experiment
        log.info(f"Using experiment from checkpoint: {experiment_name}")
        
        self.inference_pipe = Video2WorldInference(
            experiment_name=experiment_name,
            ckpt_path=ckpt_path,
            s3_credential_path="",
            context_parallel_size=self.context_parallel_size,
            config_file=self.config_file,
        )
        
        self.model = self.inference_pipe.model
        self.cosmos_config = self.inference_pipe.config
        log.info("Inference pipeline setup complete")
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass for training - computes diffusion loss."""
        if self.model is None:
            raise RuntimeError("Model not initialized. Call setup() first.")
        
        output_batch, loss = self.model.training_step(batch, iteration=0)
        return {"loss": loss, "output_batch": output_batch}
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step for PyTorch Lightning."""
        if self.model is None:
            raise RuntimeError("Model not initialized. Call setup() first.")
        
        cosmos_batch = self._prepare_cosmos_batch(batch)
        
        try:
            output_batch, loss = self.model.training_step(cosmos_batch, iteration=batch_idx)
            self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
            self.log("train/loss", loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            return loss
        except Exception as e:
            log.error(f"Error in training step: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step for PyTorch Lightning.
        
        Computes three losses:
        1. CT video loss: Generated CT rotation video vs target CT video
        2. XR input loss: First frame of generated XR video vs input XR image
        3. XR cycle loss: First frame vs last frame (360° cycle consistency)
        
        Processes all samples in the batch by generating videos one at a time
        and then computing losses on the full batch.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call setup() first.")
        
        losses = {}
        total_loss = torch.tensor(0.0, device=self.device)
        
        try:
            # 1. CT Video Loss: Generate from ct_front, compare to ct_video target
            if 'ct_front' in batch and 'ct_video' in batch:
                ct_front_batch = batch['ct_front']  # (B, C, H, W)
                ct_video_target_batch = batch['ct_video']  # (B, T, H, W)
                
                # Generate CT rotation video for entire batch
                ct_video_output_batch = self.generate_batch(
                    ct_front_batch, 
                    prompt=DEFAULT_CT_PROMPT, 
                    show_progress=False
                )
                
                # Normalize to (B, T, H, W)
                ct_video_output_batch = self._normalize_video_for_loss(ct_video_output_batch)
                ct_video_target_norm = self._normalize_video_for_loss(ct_video_target_batch)
                
                # Ensure same number of frames by interpolating if needed
                if ct_video_output_batch.shape[1] != ct_video_target_norm.shape[1]:
                    # (B, T, H, W) -> (B, 1, T, H, W) for trilinear interpolation
                    ct_video_output_5d = rearrange(ct_video_output_batch, 'b t h w -> b 1 t h w')
                    ct_video_output_5d = F.interpolate(
                        ct_video_output_5d,
                        size=(ct_video_target_norm.shape[1], ct_video_target_norm.shape[2], ct_video_target_norm.shape[3]),
                        mode='trilinear',
                        align_corners=False
                    )
                    ct_video_output_batch = rearrange(ct_video_output_5d, 'b 1 t h w -> b t h w')
                
                # Ensure same spatial size
                if ct_video_output_batch.shape[-2:] != ct_video_target_norm.shape[-2:]:
                    ct_video_output_batch = F.interpolate(
                        ct_video_output_batch,
                        size=ct_video_target_norm.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Compute MSE loss
                ct_loss = F.mse_loss(ct_video_output_batch, ct_video_target_norm)
                losses['val/ct_video_loss'] = ct_loss
                total_loss = total_loss + ct_loss
            
            # 2 & 3. XR Losses: Generate from xr_image, compute input and cycle losses
            if 'xr_image' in batch:
                xr_image_batch = batch['xr_image']  # (B, C, H, W)
                batch_size = xr_image_batch.shape[0]
                
                # Generate XR rotation video for each sample in batch
                xr_video_preds = []
                for i in range(batch_size):
                    xr_image_single = xr_image_batch[i:i+1]  # (1, C, H, W)
                    xr_video_pred = self.generate(
                        xr_image_single, 
                        prompt=DEFAULT_XR_PROMPT, 
                        show_progress=False
                    )
                    # Normalize to (1, T, H, W)
                    xr_video_pred = self._normalize_video_for_loss(xr_video_pred)
                    xr_video_preds.append(xr_video_pred)
                
                # Stack predictions: (B, T, H, W)
                xr_video_output_batch = torch.cat(xr_video_preds, dim=0)
                
                # XR Input Loss: First frame should match input image
                xr_first_frames = xr_video_output_batch[:, 0:1, :, :]  # (B, 1, H, W)
                xr_input_normed = self._normalize_image_for_loss(xr_image_batch)  # (B, 1, H, W)
                
                # Ensure same spatial size
                if xr_first_frames.shape[-2:] != xr_input_normed.shape[-2:]:
                    xr_first_frames = F.interpolate(
                        xr_first_frames, 
                        size=xr_input_normed.shape[-2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                xr_input_loss = F.mse_loss(xr_first_frames, xr_input_normed)
                losses['val/xr_input_loss'] = xr_input_loss
                total_loss = total_loss + xr_input_loss
                
                # XR Cycle Loss: Last frame should match first frame (360° rotation)
                xr_last_frames = xr_video_output_batch[:, -1:, :, :]  # (B, 1, H, W)
                xr_cycle_loss = F.mse_loss(xr_last_frames, xr_input_normed)
                losses['val/xr_cycle_loss'] = xr_cycle_loss
                total_loss = total_loss + xr_cycle_loss
            
            # Log all losses
            for name, loss_val in losses.items():
                self.log(name, loss_val, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            
            self.log("val/loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            return total_loss
            
        except Exception as e:
            log.error(f"Error in validation step: {e}")
            return torch.tensor(0.0, device=self.device)
    
    def _normalize_video_for_loss(self, video: torch.Tensor) -> torch.Tensor:
        """Normalize video tensor to (B, T, H, W) format for loss computation."""
        # Handle various input shapes using einops
        if video.ndim == 5:  # (B, C, T, H, W) -> (B, T, H, W) - take first channel
            video = video[:, 0]
        elif video.ndim == 3:  # (T, H, W) -> (1, T, H, W)
            video = rearrange(video, 't h w -> 1 t h w')
        # ndim == 4 is already (B, T, H, W)
        
        # Normalize to [0, 1] range
        if video.dtype == torch.uint8:
            video = video.float() / 255.0
        elif video.max() > 1.0:
            video = (video - video.min()) / (video.max() - video.min() + 1e-8)
        
        return video.to(device=self.device, dtype=torch.float32)
    
    def _normalize_image_for_loss(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize image tensor to (B, 1, H, W) format for loss computation."""
        # Handle various input shapes using einops
        if image.ndim == 4:  # (B, C, H, W)
            if image.shape[1] > 1:
                image = image[:, 0:1, :, :]  # Take first channel
        elif image.ndim == 3:  # (B, H, W) -> (B, 1, H, W)
            image = rearrange(image, 'b h w -> b 1 h w')
        elif image.ndim == 2:  # (H, W) -> (1, 1, H, W)
            image = rearrange(image, 'h w -> 1 1 h w')
        
        # Normalize to [0, 1] range
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        elif image.max() > 1.0:
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        return image.to(device=self.device, dtype=torch.float32)
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Prediction step - generates 360° rotation video.
        
        Note: Progress bar is hidden to avoid messing up the training progress bar.
        """
        results = {}
        
        if 'ct_front' in batch:
            ct_video = self.generate(batch['ct_front'], prompt=DEFAULT_CT_PROMPT, show_progress=False)
            results['ct_video'] = ct_video
        
        if 'xr_image' in batch:
            xr_video = self.generate(batch['xr_image'], prompt=DEFAULT_XR_PROMPT, show_progress=False)
            results['xr_video'] = xr_video
        
        return results
    
    def generate_sample(
        self,
        input_image: torch.Tensor,
        prompt: str = DEFAULT_CT_PROMPT,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: int = 0,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Generate 360° rotation video from a single input image.
        
        Args:
            input_image: Single image tensor (1, C, H, W) or (C, H, W)
            prompt: Text prompt for conditioning
            negative_prompt: Negative prompt for classifier-free guidance
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            seed: Random seed
            show_progress: If False, suppresses tqdm progress bar
        
        Returns:
            Generated video tensor
        """
        # Ensure batch dimension: (C, H, W) -> (1, C, H, W)
        if input_image.ndim == 3:
            input_image = rearrange(input_image, 'c h w -> 1 c h w')
        
        # Suppress tqdm progress bar and logging if requested
        import tqdm.auto
        import tqdm.std
        import logging
        original_tqdm_auto = tqdm.auto.tqdm
        original_tqdm_std = tqdm.std.tqdm
        
        # Suppress cosmos logging by setting log level
        cosmos_loggers = [
            'cosmos_predict2',
            'cosmos_predict2._src.predict2.inference.video2world',
            'cosmos_predict2._src.predict2.inference.inference',
            'cosmos_predict2._src.imaginaire',
        ]
        original_log_levels = {}
        
        # Also suppress cosmos's custom log module
        original_log_info = None
        original_log_debug = None
        if not show_progress:
            for logger_name in cosmos_loggers:
                logger = logging.getLogger(logger_name)
                original_log_levels[logger_name] = logger.level
                logger.setLevel(logging.ERROR)
            
            # Suppress cosmos's custom log.info and log.debug
            try:
                from cosmos_predict2._src.imaginaire.utils import log as cosmos_log
                original_log_info = cosmos_log.info
                original_log_debug = cosmos_log.debug
                cosmos_log.info = lambda *args, **kwargs: None
                cosmos_log.debug = lambda *args, **kwargs: None
            except Exception:
                pass
        
        if not show_progress:
            class SilentTqdm:
                def __init__(self, *args, **kwargs):
                    self.iterable = args[0] if args else kwargs.get('iterable', None)
                def __iter__(self):
                    return iter(self.iterable) if self.iterable is not None else iter([])
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
                def update(self, n=1):
                    pass
                def close(self):
                    pass
                def set_description(self, *args, **kwargs):
                    pass
                def set_postfix(self, *args, **kwargs):
                    pass
            
            tqdm.auto.tqdm = SilentTqdm
            tqdm.std.tqdm = SilentTqdm
            import tqdm as tqdm_module
            tqdm_module.tqdm = SilentTqdm
        
        try:
            if self.inference_pipe is None:
                ckpt_path = self._get_checkpoint_path()
                self._setup_inference_pipeline(ckpt_path)
            
            num_steps = num_inference_steps or self.num_inference_steps
            guidance = guidance_scale or self.guidance_scale
            
            # Convert grayscale to RGB: (B, 1, H, W) -> (B, 3, H, W)
            if input_image.shape[1] == 1:
                input_image = repeat(input_image, 'b 1 h w -> b 3 h w')
            
            # Convert to uint8
            if input_image.max() <= 1.0:
                input_image_uint8 = (input_image * 255).to(torch.uint8)
            else:
                input_image_uint8 = input_image.to(torch.uint8)
            
            model_required_frames = self.inference_pipe.model.tokenizer.get_pixel_num_frames(
                self.inference_pipe.model.config.state_t
            )
            
            B, C, H, W = input_image_uint8.shape
            # Create video input: first frame is input image, rest are zeros
            vid_input = torch.zeros(B, C, model_required_frames, H, W, dtype=torch.uint8)
            vid_input[:, :, 0, :, :] = input_image_uint8
            
            resolution = f"{self.resolution[0]},{self.resolution[1]}"
            
            video = self.inference_pipe.generate_vid2world(
                prompt=prompt,
                input_path=vid_input,
                guidance=int(guidance),
                num_video_frames=self.num_frames,
                num_latent_conditional_frames=1,
                resolution=resolution,
                seed=seed,
                negative_prompt=negative_prompt,
                num_steps=num_steps,
                offload_diffusion_model=self.offload_diffusion_model,
                offload_text_encoder=self.offload_text_encoder,
                offload_tokenizer=self.offload_tokenizer,
            )
            
            return video
        finally:
            if not show_progress:
                # Restore tqdm
                tqdm.auto.tqdm = original_tqdm_auto
                tqdm.std.tqdm = original_tqdm_std
                import tqdm as tqdm_module
                tqdm_module.tqdm = original_tqdm_std
                
                # Restore logging levels
                for logger_name, level in original_log_levels.items():
                    logging.getLogger(logger_name).setLevel(level)
                
                # Restore cosmos's custom log functions
                if original_log_info is not None:
                    try:
                        from cosmos_predict2._src.imaginaire.utils import log as cosmos_log
                        cosmos_log.info = original_log_info
                        cosmos_log.debug = original_log_debug
                    except Exception:
                        pass
    
    def generate_batch(
        self,
        input_images: torch.Tensor,
        prompt: str = DEFAULT_CT_PROMPT,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: int = 0,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Generate 360° rotation videos for a batch of input images.
        
        Processes each sample in the batch sequentially and concatenates results.
        
        Args:
            input_images: Batch of images (B, C, H, W)
            prompt: Text prompt for conditioning
            negative_prompt: Negative prompt for classifier-free guidance
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            seed: Random seed (incremented for each sample)
            show_progress: If False, suppresses tqdm progress bar
        
        Returns:
            Batch of generated videos (B, C, T, H, W) or (B, T, H, W)
        """
        batch_size = input_images.shape[0]
        videos = []
        
        for i in range(batch_size):
            single_image = input_images[i:i+1]  # Keep batch dim (1, C, H, W)
            video = self.generate_sample(
                single_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed + i,  # Different seed for each sample
                show_progress=show_progress,
            )
            videos.append(video)
        
        # Concatenate along batch dimension
        # Handle various output shapes
        if videos[0].ndim == 5:  # (1, C, T, H, W)
            return torch.cat(videos, dim=0)  # (B, C, T, H, W)
        elif videos[0].ndim == 4:  # (1, T, H, W) or (C, T, H, W)
            if videos[0].shape[0] == 1:
                return torch.cat(videos, dim=0)  # (B, T, H, W)
            else:
                # (C, T, H, W) - stack to (B, C, T, H, W)
                return torch.stack(videos, dim=0)
        elif videos[0].ndim == 3:  # (T, H, W)
            return torch.stack(videos, dim=0)  # (B, T, H, W)
        else:
            return torch.cat(videos, dim=0)
    
    def generate(
        self,
        input_image: torch.Tensor,
        prompt: str = DEFAULT_CT_PROMPT,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: int = 0,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Generate 360° rotation video from input image(s).
        
        Automatically handles both single samples and batches.
        
        Args:
            input_image: Image tensor (B, C, H, W) or (C, H, W)
            prompt: Text prompt for conditioning
            negative_prompt: Negative prompt for classifier-free guidance
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            seed: Random seed
            show_progress: If False, suppresses tqdm progress bar
        
        Returns:
            Generated video tensor
        """
        # Ensure batch dimension: (C, H, W) -> (1, C, H, W)
        if input_image.ndim == 3:
            input_image = rearrange(input_image, 'c h w -> 1 c h w')
        
        batch_size = input_image.shape[0]
        
        if batch_size == 1:
            return self.generate_sample(
                input_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                show_progress=show_progress,
            )
        else:
            return self.generate_batch(
                input_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                show_progress=show_progress,
            )
    
    def _prepare_cosmos_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare batch data in Cosmos model format.
        
        Cosmos expects video in uint8 format (0-255) with shape (B, C, T, H, W).
        """
        cosmos_batch = {"dataset_name": "cosmed_medical"}
        
        if 'ct_video' in batch:
            ct_video = batch['ct_video']  # (B, T, H, W) or (B, 1, T, H, W)
            
            # Add channel dimension if needed: (B, T, H, W) -> (B, 1, T, H, W)
            if ct_video.ndim == 4:
                ct_video = rearrange(ct_video, 'b t h w -> b 1 t h w')
            
            # Convert grayscale to RGB: (B, 1, T, H, W) -> (B, 3, T, H, W)
            ct_video = repeat(ct_video, 'b 1 t h w -> b 3 t h w')
            
            # Convert to uint8 (0-255) - Cosmos expects uint8 format
            if ct_video.max() <= 1.0:
                ct_video = (ct_video * 255).clamp(0, 255)
            ct_video = ct_video.to(torch.uint8)
            
            cosmos_batch["video"] = ct_video
        
        if 'ct_front' in batch:
            ct_front = batch['ct_front']  # (B, 1, H, W)
            
            # Convert grayscale to RGB: (B, 1, H, W) -> (B, 3, H, W)
            if ct_front.shape[1] == 1:
                ct_front = repeat(ct_front, 'b 1 h w -> b 3 h w')
            
            # Convert to uint8 for conditioning image
            if ct_front.max() <= 1.0:
                ct_front = (ct_front * 255).clamp(0, 255)
            ct_front = ct_front.to(torch.uint8)
            
            cosmos_batch["conditioning_image"] = ct_front
        
        # Add text embeddings
        if self.model is not None and hasattr(self.model, 'text_encoder') and self.model.text_encoder is not None:
            prompt = DEFAULT_CT_PROMPT
            B = batch['ct_video'].shape[0] if 'ct_video' in batch else 1
            cosmos_batch["ai_caption"] = [prompt] * B
            cosmos_batch["t5_text_embeddings"] = self.model.text_encoder.compute_text_embeddings_online(
                data_batch={"ai_caption": cosmos_batch["ai_caption"], "images": None},
                input_caption_key="ai_caption",
            )
        
        # Add metadata
        B = batch['ct_video'].shape[0] if 'ct_video' in batch else 1
        H, W = self.resolution
        cosmos_batch["fps"] = torch.ones(B, device=self.device) * 30.0
        cosmos_batch["padding_mask"] = torch.zeros(B, 1, H, W, device=self.device)
        cosmos_batch["num_conditional_frames"] = 1
        
        # Move video tensors to device (keep as uint8)
        if "video" in cosmos_batch:
            cosmos_batch["video"] = cosmos_batch["video"].to(device=self.device)
        if "conditioning_image" in cosmos_batch:
            cosmos_batch["conditioning_image"] = cosmos_batch["conditioning_image"].to(device=self.device)
        
        return cosmos_batch
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        if self.model is None:
            raise RuntimeError("Model not initialized. Call setup() first.")
        
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        try:
            from apex.optimizers import FusedAdam
            optimizer = FusedAdam(
                trainable_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
            log.info("Using FusedAdam optimizer from apex")
        except ImportError:
            log.warning("apex not available, using AdamW instead")
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=2000,
            T_mult=1,
            eta_min=self.learning_rate * 0.2,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when saving checkpoint."""
        checkpoint["num_frames"] = self.num_frames
        checkpoint["resolution"] = self.resolution
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when loading checkpoint."""
        if "num_frames" in checkpoint:
            self.num_frames = checkpoint["num_frames"]
        if "resolution" in checkpoint:
            self.resolution = checkpoint["resolution"]


def create_inference_wrapper(
    checkpoint_path: Optional[str] = None,
    num_frames: int = 121,
    resolution: tuple[int, int] = (256, 256),
    **kwargs,
) -> CosmedModelWrapper:
    """Create a CosmedModelWrapper configured for inference."""
    wrapper = CosmedModelWrapper(
        checkpoint_path=checkpoint_path,
        num_frames=num_frames,
        resolution=resolution,
        offload_diffusion_model=True,
        offload_text_encoder=True,
        offload_tokenizer=True,
        **kwargs,
    )
    wrapper.setup(stage='predict')
    return wrapper


if __name__ == "__main__":
    """Test the model wrapper."""
    print("Testing CosmedModelWrapper...")
    print("=" * 80)
    
    model = CosmedModelWrapper(
        num_frames=121,
        resolution=(256, 256),
        learning_rate=2 ** (-14.5),
    )
    
    print(f"Num frames: {model.num_frames}")
    print(f"Resolution: {model.resolution}")
    print(f"Learning rate: {model.learning_rate:.2e}")
    print(f"Guidance scale: {model.guidance_scale}")
    print(f"Inference steps: {model.num_inference_steps}")
    
    print("=" * 80)
    print("✓ Model wrapper test complete")
