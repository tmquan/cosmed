"""
Cosmed PyTorch Lightning Module for 360° CT/XR rotation video generation.

This module loads the Cosmos Predict 2.5 DiT network for medical imaging:
1. CT frontal projection -> 360° rotation video  
2. Real XR frontal image -> 360° rotation video with cycle consistency
"""

import sys
import os

# Add cosmos-predict2.5 to Python path
COSMOS_PATH = "/workspace/cosmos-predict2.5"
if os.path.exists(COSMOS_PATH) and COSMOS_PATH not in sys.path:
    sys.path.insert(0, COSMOS_PATH)

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Any
from lightning import LightningModule
from einops import rearrange, repeat, reduce

try:
    from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
    from cosmos_predict2._src.predict2.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT
    from cosmos_predict2._src.predict2.networks.selective_activation_checkpoint import SACConfig, CheckpointMode
    from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface
    from transformers import T5EncoderModel, T5Tokenizer
    
    COSMOS_AVAILABLE = True
    print("✓ Cosmos Predict 2.5 modules imported successfully")
except ImportError as e:
    COSMOS_AVAILABLE = False
    print(f"⚠ Cosmos Predict 2.5 not available: {e}")
    print(f"Python path: {sys.path[:3]}")
    import traceback
    traceback.print_exc()


class CosmedModel(LightningModule):
    """
    Cosmed model using Cosmos Predict 2.5 DiT for medical 360° rotation generation.
    
    This model:
    - Loads the pretrained DiT network directly (bypassing tokenizer requirements)
    - Fine-tunes on medical imaging tasks
    - Supports CT and XR rotation generation
    """
    
    def __init__(
        self,
        # Cosmos checkpoint paths
        dit_checkpoint_path: str = None,
        tokenizer_checkpoint_path: str = None,
        
        # Text encoder
        text_encoder_path: str = "google-t5/t5-11b",
        
        # Model params
        num_frames: int = 121,
        img_height: int = 256,
        img_width: int = 256,
        
        # Training params
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        
        # Loss weights
        ct_video_loss_weight: float = 1.0,
        xr_cycle_loss_weight: float = 0.5,
        xr_input_loss_weight: float = 0.5,
        
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.dit_checkpoint_path = dit_checkpoint_path
        self.tokenizer_checkpoint_path = tokenizer_checkpoint_path
        self.text_encoder_path = text_encoder_path
        self.num_frames = num_frames
        self.img_height = img_height
        self.img_width = img_width
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.ct_video_loss_weight = ct_video_loss_weight
        self.xr_cycle_loss_weight = xr_cycle_loss_weight
        self.xr_input_loss_weight = xr_input_loss_weight
        
        # Cache for text embeddings
        self.text_embeddings_cache = {}
        
        # Assert Cosmos is available
        if not COSMOS_AVAILABLE:
            raise RuntimeError("Cosmos Predict 2.5 is not available. Please install it first.")
        
        # ========================================
        # Setup DiT checkpoint
        # ========================================
        if not dit_checkpoint_path:
            print("⚠ DiT checkpoint: Not provided, downloading from Hugging Face...")
            dit_checkpoint_path = self._download_dit_checkpoint()
        elif not os.path.exists(dit_checkpoint_path):
            print(f"⚠ DiT checkpoint: Not found at {dit_checkpoint_path}")
            print("  Downloading from Hugging Face...")
            dit_checkpoint_path = self._download_dit_checkpoint()
        else:
            print(f"✓ DiT checkpoint: Using local file at {dit_checkpoint_path}")
        
        self.dit_checkpoint_path = dit_checkpoint_path
        
        # ========================================
        # Setup Tokenizer checkpoint
        # ========================================
        if not tokenizer_checkpoint_path:
            print("⚠ Tokenizer checkpoint: Not provided, downloading from Hugging Face...")
            tokenizer_checkpoint_path = self._download_tokenizer_checkpoint()
        elif not os.path.exists(tokenizer_checkpoint_path):
            print(f"⚠ Tokenizer checkpoint: Not found at {tokenizer_checkpoint_path}")
            print("  Downloading from Hugging Face...")
            tokenizer_checkpoint_path = self._download_tokenizer_checkpoint()
        else:
            print(f"✓ Tokenizer checkpoint: Using local file at {tokenizer_checkpoint_path}")
        
        self.tokenizer_checkpoint_path = tokenizer_checkpoint_path
        
        # ========================================
        # Setup Text Encoder checkpoint
        # ========================================
        if not text_encoder_path:
            print("⚠ Text encoder: Not provided, using default...")
            text_encoder_path = self._download_text_encoder_checkpoint()
        else:
            print(f"✓ Text encoder: Using {text_encoder_path}")
        
        self.text_encoder_path = text_encoder_path
        
        # ========================================
        # Validate all components
        # ========================================
        if not self.dit_checkpoint_path:
            raise ValueError("DiT checkpoint path is required but not available")
        if not self.tokenizer_checkpoint_path:
            raise ValueError("Tokenizer checkpoint path is required but not available")
        if not self.text_encoder_path:
            raise ValueError("Text encoder path is required but not available")
        
        print("\n" + "="*80)
        print("All checkpoints ready:")
        print(f"  • DiT: {self.dit_checkpoint_path}")
        print(f"  • Tokenizer: {self.tokenizer_checkpoint_path}")
        print(f"  • Text Encoder: {self.text_encoder_path}")
        print("="*80 + "\n")
        
        # Initialize Cosmos DiT
        self._init_cosmos_dit()
    
    
    def _download_dit_checkpoint(self) -> str:
        """
        Download DiT checkpoint from Hugging Face.
        
        Returns:
            str: Path to the downloaded DiT checkpoint
            
        Raises:
            RuntimeError: If download fails
        """
        try:
            from huggingface_hub import hf_hub_download
            
            print("Downloading DiT checkpoint from Hugging Face...")
            print("  Repository: nvidia/Cosmos-Predict2.5-2B")
            print("  File: base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt")
            
            dit_path = hf_hub_download(
                repo_id="nvidia/Cosmos-Predict2.5-2B",
                filename="base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt",
                local_dir="/workspace/checkpoints",
            )
            
            print(f"✓ DiT checkpoint downloaded to: {dit_path}")
            return dit_path
            
        except Exception as e:
            print(f"✗ Failed to download DiT checkpoint: {e}")
            print("\nManual download instructions:")
            print("  1. Login: huggingface-cli login")
            print("  2. Download: huggingface-cli download nvidia/Cosmos-Predict2.5-2B \\")
            print("               --include 'base/post-trained/*' --local-dir /workspace/checkpoints")
            raise RuntimeError("Failed to download DiT checkpoint from Hugging Face") from e
    
    def _download_tokenizer_checkpoint(self) -> str:
        """
        Download tokenizer checkpoint from Hugging Face.
        
        Returns:
            str: Path to the downloaded tokenizer checkpoint
            
        Raises:
            RuntimeError: If download fails
        """
        try:
            from huggingface_hub import hf_hub_download
            
            print("Downloading tokenizer checkpoint from Hugging Face...")
            print("  Repository: nvidia/Cosmos-Predict2.5-2B")
            print("  File: tokenizer.pth")
            
            tokenizer_path = hf_hub_download(
                repo_id="nvidia/Cosmos-Predict2.5-2B",
                filename="tokenizer.pth",
                local_dir="/workspace/checkpoints",
            )
            
            print(f"✓ Tokenizer checkpoint downloaded to: {tokenizer_path}")
            return tokenizer_path
            
        except Exception as e:
            print(f"✗ Failed to download tokenizer checkpoint: {e}")
            print("\nManual download instructions:")
            print("  1. Login: huggingface-cli login")
            print("  2. Download: huggingface-cli download nvidia/Cosmos-Predict2.5-2B \\")
            print("               tokenizer.pth --local-dir /workspace/checkpoints")
            raise RuntimeError("Failed to download tokenizer checkpoint from Hugging Face") from e
    
    def _download_text_encoder_checkpoint(self) -> str:
        """
        Download text encoder checkpoint from Hugging Face.
        
        Returns:
            str: Path to the downloaded text encoder checkpoint
            
        Raises:
            RuntimeError: If download fails
        """
        try:
            from huggingface_hub import snapshot_download
            
            print("Downloading text encoder checkpoint from Hugging Face...")
            print("  Repository: google-t5/t5-11b")
            print("  Note: This is a large model (~11GB), download may take several minutes")
            
            text_encoder_path = snapshot_download(
                repo_id="google-t5/t5-11b",
                local_dir="/workspace/checkpoints/google-t5/t5-11b",
            )
            
            print(f"✓ Text encoder checkpoint downloaded to: {text_encoder_path}")
            return text_encoder_path
            
        except Exception as e:
            print(f"✗ Failed to download text encoder checkpoint: {e}")
            print("\nManual download instructions:")
            print("  1. Login: huggingface-cli login")
            print("  2. Download: huggingface-cli download google-t5/t5-11b \\")
            print("               --local-dir /workspace/checkpoints/google-t5/t5-11b")
            print("\nFallback: Using HuggingFace model ID (will auto-download on first use)")
            return "google-t5/t5-11b"
    
    def _init_cosmos_dit(self):
        """Initialize Cosmos Predict 2.5 DiT network directly."""
        try:
            print("Initializing Cosmos Predict 2.5 DiT network...")
            
            # Create DiT network config (2B model configuration)
            # Note: MinimalV1LVGDiT automatically adds +1 to in_channels for condition mask
            dit_config = dict(
                max_img_h=self.img_height,
                max_img_w=self.img_width,
                max_frames=self.num_frames,
                in_channels=16,  # lat channels (MinimalV1LVGDiT will add +1 internally)
                out_channels=16,
                patch_spatial=2,
                patch_temporal=1,
                concat_padding_mask=True,  # Required by checkpoint (adds +1 channel)
                # 2B model architecture
                model_channels=2048,
                num_blocks=28,
                num_heads=16,
                atten_backend="minimal_a2a",
                # Positional embeddings
                pos_emb_cls="rope3d",
                pos_emb_learnable=True,
                pos_emb_interpolation="crop",
                use_adaln_lora=True,
                adaln_lora_dim=256,
                rope_h_extrapolation_ratio=2.0,
                rope_w_extrapolation_ratio=2.0,
                rope_t_extrapolation_ratio=1.0,
                extra_per_block_abs_pos_emb=False,
                rope_enable_fps_modulation=False,
                sac_config=SACConfig(
                    every_n_blocks=1,
                    mode=CheckpointMode.NONE,  # Use NONE mode to disable selective checkpoint
                ),
            )
            
            # Initialize DiT network
            self.dit = MinimalV1LVGDiT(**dit_config)
            
            # Load checkpoint
            if self.dit_checkpoint_path:
                print(f"Loading DiT checkpoint from: {self.dit_checkpoint_path}")
                state_dict = torch.load(self.dit_checkpoint_path, map_location="cpu")
                
                # Extract DiT weights (remove "net." prefix if present)
                if isinstance(state_dict, dict):
                    # Try different possible keys
                    if 'net' in state_dict and isinstance(state_dict['net'], dict):
                        model_dict = state_dict['net']
                    elif any(k.startswith('net.') for k in state_dict.keys()):
                        model_dict = {k.replace('net.', ''): v for k, v in state_dict.items() if k.startswith('net.')}
                    else:
                        model_dict = state_dict
                    
                    # Load weights
                    missing_keys, unexpected_keys = self.dit.load_state_dict(model_dict, strict=False)
                    print(f"Loaded checkpoint: {len(model_dict)} keys")
                    if missing_keys:
                        print(f"Missing keys: {len(missing_keys)}")
                    if unexpected_keys:
                        print(f"Unexpected keys: {len(unexpected_keys)}")
                
                # Print dtype information
                param_dtypes = set(p.dtype for p in self.dit.parameters())
                buffer_dtypes = set(b.dtype for b in self.dit.buffers())
                print(f"✓ DiT loaded in natural dtype:")
                print(f"  Parameter dtypes: {param_dtypes}")
                print(f"  Buffer dtypes: {buffer_dtypes}")
                
                # Count parameters
                num_params = sum(p.numel() for p in self.dit.parameters()) / 1e9
                trainable_params = sum(p.numel() for p in self.dit.parameters() if p.requires_grad) / 1e9
                print(f"✓ DiT loaded: {num_params:.2f}B total params, {trainable_params:.2f}B trainable")
            
            # Initialize Cosmos tokenizer (VAE for pixel↔latent conversion)
            print(f"Loading Cosmos tokenizer from: {self.tokenizer_checkpoint_path}")
            self.tokenizer = Wan2pt1VAEInterface(
                name="wan2pt1_tokenizer",
                vae_pth=self.tokenizer_checkpoint_path,
            )
            # Freeze tokenizer
            self.tokenizer.model.model.requires_grad_(False)
            self.tokenizer.model.model.eval()
            print("✓ Tokenizer loaded and frozen")
            
            # Initialize text encoder for conditioning
            print(f"Loading T5 text encoder from: {self.text_encoder_path}")
            try:
                # Work around modelopt patching issues with HuggingFace
                # Temporarily restore original _load_pretrained_model if it was patched
                import transformers.modeling_utils
                original_load_method = None
                if hasattr(transformers.modeling_utils.PreTrainedModel, '_modelopt_cache'):
                    # modelopt has patched the method, restore the original
                    cache = transformers.modeling_utils.PreTrainedModel._modelopt_cache
                    if '_load_pretrained_model' in cache:
                        original_load_method = transformers.modeling_utils.PreTrainedModel._load_pretrained_model
                        transformers.modeling_utils.PreTrainedModel._load_pretrained_model = cache['_load_pretrained_model']
                
                try:
                    self.text_tokenizer = T5Tokenizer.from_pretrained(self.text_encoder_path, legacy=False)
                    self.text_encoder = T5EncoderModel.from_pretrained(
                        self.text_encoder_path,
                        # dtype=torch.float32,  # Use float32 to avoid APEX fused norm issues
                    )
                finally:
                    # Restore the patched method if we changed it
                    if original_load_method is not None:
                        transformers.modeling_utils.PreTrainedModel._load_pretrained_model = original_load_method
                
                # Freeze text encoder
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
                self.text_encoder.eval()
                
                # Print text encoder dtype
                text_param_dtypes = set(p.dtype for p in self.text_encoder.parameters())
                print(f"✓ Text encoder loaded in natural dtype:")
                print(f"  Parameter dtypes: {text_param_dtypes}")
            except Exception as e:
                print(f"⚠ Warning: Could not load T5 encoder: {e}")
                print("Creating dummy text encoder for placeholder embeddings")
                # Create a simple embedding layer as fallback
                self.text_tokenizer = None
                self.text_encoder = None
            
            print("✓ Cosmos DiT network initialized with text conditioning")
            
        except Exception as e:
            print(f"✗ Error initializing Cosmos DiT: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError("Failed to initialize Cosmos DiT. Cannot proceed without it.") from e
    
    def get_text_embeddings(self, prompt: str) -> torch.Tensor:
        """
        Get text embeddings for a prompt (with caching).
        
        Args:
            prompt: Text prompt
            
        Returns:
            Text embeddings (B, seq_len, hidden_dim)
        """
        if prompt in self.text_embeddings_cache:
            return self.text_embeddings_cache[prompt]
        
        # If no text encoder, return dummy embeddings
        if self.text_encoder is None:
            # Create fixed dummy embeddings (compatible with DiT cross-attention)
            # T5-base has 768 hidden dimensions
            embeddings = torch.zeros(1, 77, 768, device=self.device)
            self.text_embeddings_cache[prompt] = embeddings
            return embeddings
        
        # Tokenize
        inputs = self.text_tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Encode
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            embeddings = outputs.last_hidden_state  # (1, seq_len, hidden_dim)
        
        # Cache
        self.text_embeddings_cache[prompt] = embeddings
        
        return embeddings
    
    def forward(self, image: torch.Tensor, prompt: str = "360 degree rotation of medical scan") -> torch.Tensor:
        """
        Generate 360° rotation video from a single frontal image using Cosmos DiT.
        
        Args:
            image: Input frontal image (B, 1, H, W) - grayscale medical image
            prompt: Text prompt for conditioning
            
        Returns:
            Generated video (B, T, H, W) - grayscale
        """
        B = image.shape[0]
        
        # Get model dtype and device
        _device = image.device
        
        # Convert input image to regular tensor (strip MetaTensor if present from MONAI)
        if hasattr(image, 'as_tensor'):
            image = image.as_tensor()
        image = torch.as_tensor(image, device=_device)
        
        # Convert grayscale to RGB for tokenizer (pretrained on RGB images)
        # Medical images are grayscale (1 channel) but tokenizer expects RGB (3 channels)
        image_rgb = repeat(image, 'b 1 h w -> b 3 h w')  # (B, 1, H, W) -> (B, 3, H, W)
        
        # Convert pixel space to latent space using Cosmos tokenizer
        image_5d = rearrange(image_rgb, 'b c h w -> b c 1 h w')  # (B, 3, 1, H, W)
        with torch.no_grad():
            lat_img = self.tokenizer.encode(image_5d)  # (B, 16, 1, H', W')
            # Convert to regular tensor immediately after encoding
            lat_img = torch.as_tensor(lat_img.data if hasattr(lat_img, 'data') else lat_img, device=_device)
        lat_img = rearrange(lat_img, 'b c 1 h w -> b c h w')  # (B, 16, H', W')
        
        # Expand to video format (B, 16, T, H', W')
        lat_video = repeat(lat_img, 'b c h w -> b c t h w', t=self.num_frames)
        
        # Get latent dimensions
        _, _, _, H_lat, W_lat = lat_video.shape
        
        # Create inputs for DiT
        timesteps = torch.zeros(B, self.num_frames, device=_device)
        
        # Get text embeddings (will be in bfloat16 from T5 encoder)
        text_emb = self.get_text_embeddings(prompt)
        crossattn_emb = repeat(text_emb, '1 n d -> b n d', b=B)
        
        # Condition mask and padding mask
        condition_mask = torch.zeros(B, 1, self.num_frames, H_lat, W_lat, device=_device)
        condition_mask[:, :, 0, :, :] = 1.0  # First frame is conditional
        
        padding_mask = torch.ones(B, 1, H_lat, W_lat, device=_device)
        
        # Run DiT forward
        lat_output = self.dit(
            x_B_C_T_H_W=lat_video,
            timesteps_B_T=timesteps,
            crossattn_emb=crossattn_emb,
            condition_video_input_mask_B_C_T_H_W=condition_mask,
            padding_mask=padding_mask,
        )
        
        with torch.no_grad():
            T_lat = lat_output.shape[2]  # Temporal dimension of latent
            max_decode_frames = 50  # Tokenizer's max supported frames
            
            if T_lat <= max_decode_frames:
                # Can decode in one pass
                video_5d_rgb = self.tokenizer.decode(lat_output)  # (B, 3, T, H, W) RGB output
            else:
                # Sliding window decode with overlap for smoother transitions
                overlap = 10  # Overlap frames in latent space
                stride = max_decode_frames - overlap
                
                video_chunks = []
                decoded_up_to = 0  # Track how many output frames we've kept
                
                start_idx = 0
                while start_idx < T_lat:
                    end_idx = min(start_idx + max_decode_frames, T_lat)
                    lat_chunk = lat_output[:, :, start_idx:end_idx, :, :]
                    video_chunk = self.tokenizer.decode(lat_chunk)  # (B, 3, T_chunk, H, W)
                    
                    if start_idx == 0:
                        # First chunk: keep everything
                        video_chunks.append(video_chunk)
                        # Estimate temporal upsampling ratio from first chunk
                        t_upsample = video_chunk.shape[2] / lat_chunk.shape[2]
                    else:
                        # Subsequent chunks: skip the overlapping decoded frames
                        # Calculate how many decoded frames correspond to the overlap
                        overlap_decoded = int(overlap * t_upsample)
                        video_chunks.append(video_chunk[:, :, overlap_decoded:, :, :])
                    
                    # Move window
                    if end_idx >= T_lat:
                        break
                    start_idx += stride
                
                video_5d_rgb = torch.cat(video_chunks, dim=2)  # Concatenate along temporal dim
        
        # Convert RGB back to grayscale (average across channels)
        video = reduce(video_5d_rgb, 'b c t h w -> b t h w', 'mean')  # (B, T, H, W) grayscale
        
        # Interpolate to target frame count if needed (VAE has temporal upsampling)
        T_out = video.shape[1]
        if T_out != self.num_frames:
            # Temporal interpolation to match expected frame count
            video = rearrange(video, 'b t h w -> b 1 t h w')  # Add channel dim for interpolate
            video = F.interpolate(
                video, 
                size=(self.num_frames, video.shape[3], video.shape[4]),
                mode='trilinear',
                align_corners=False
            )
            video = rearrange(video, 'b 1 t h w -> b t h w')  # Remove channel dim
        
        return torch.sigmoid(video)  # Ensure output is in [0, 1]
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with CT and XR losses."""
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        losses = {}
        
        # CT video reconstruction loss
        if 'ct_frontal' in batch and 'ct_video' in batch:
            ct_frontal = batch['ct_frontal']
            ct_video_target = batch['ct_video']
            
            # Use CT-specific prompt
            ct_prompt = "360 degree rotation of chest CT scan"
            ct_video_pred = self(ct_frontal, prompt=ct_prompt)
            
            loss_ct = F.l1_loss(ct_video_pred, ct_video_target)
            total_loss = total_loss + self.ct_video_loss_weight * loss_ct
            losses['train/ct_video_loss'] = loss_ct
        
        # XR cycle consistency loss
        if 'xr_image' in batch:
            xr_image = batch['xr_image']
            
            # Use XR-specific prompt
            xr_prompt = "360 degree rotation of chest X-ray"
            xr_video = self(xr_image, prompt=xr_prompt)
            
            first_frame = xr_video[:, :1, :, :]
            last_frame = xr_video[:, -1:, :, :]
            
            loss_input = F.l1_loss(first_frame, xr_image)
            loss_cycle = F.l1_loss(last_frame, first_frame)
            
            total_loss = total_loss + self.xr_input_loss_weight * loss_input
            total_loss = total_loss + self.xr_cycle_loss_weight * loss_cycle
            
            losses['train/xr_input_loss'] = loss_input
            losses['train/xr_cycle_loss'] = loss_cycle
        
        losses['train/total_loss'] = total_loss
        self.log_dict(losses, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        total_loss = 0.0
        losses = {}
        
        if 'ct_frontal' in batch and 'ct_video' in batch:
            ct_frontal = batch['ct_frontal']
            ct_video_target = batch['ct_video']
            ct_video_pred = self(ct_frontal, prompt="360 degree rotation of chest CT scan")
            loss_ct = F.l1_loss(ct_video_pred, ct_video_target)
            total_loss += loss_ct
            losses['val/ct_video_loss'] = loss_ct
        
        if 'xr_image' in batch:
            xr_image = batch['xr_image']
            xr_video = self(xr_image, prompt="360 degree rotation of chest X-ray")
            first_frame = xr_video[:, :1, :, :]
            last_frame = xr_video[:, -1:, :, :]
            
            loss_input = F.l1_loss(first_frame, xr_image)
            loss_cycle = F.l1_loss(last_frame, first_frame)
            
            total_loss += loss_input + loss_cycle
            losses['val/xr_input_loss'] = loss_input
            losses['val/xr_cycle_loss'] = loss_cycle
        
        losses['val/total_loss'] = total_loss
        self.log_dict(losses, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        return total_loss
    
    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.learning_rate * 0.01,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
