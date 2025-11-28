#!/usr/bin/env python
"""
Training script for Cosmed Model using Cosmos-Predict2.5 Video2World pipeline.

This script provides two training modes:

1. PyTorch Lightning Mode (default):
   - Uses Lightning Trainer with CosmedModelWrapper
   - Supports standard Lightning callbacks, logging, and checkpointing
   - Easier to customize and debug
   
2. Cosmos Native Mode (--use_cosmos_trainer):
   - Uses Cosmos's ImaginaireTrainer directly
   - Better integration with Cosmos's distributed training features
   - Supports FSDP, context parallelism, etc.

Usage:
    # PyTorch Lightning mode (default):
    python cosmed_train.py \
        --data_dir /workspace/datasets/ChestMedicalData \
        --cache_dir /workspace/datasets/ChestMedicalDataCache \
        --output_dir /workspace/cosmed/outputs \
        --num_frames 121 \
        --batch_size 1 \
        --max_epochs 100

    # Cosmos Native mode:
    python cosmed_train.py \
        --use_cosmos_trainer \
        --max_iter 10000 \
        --checkpoint_save_iter 1000
    
    # With torchrun for distributed training:
    torchrun --nproc_per_node=8 --master_port=12341 cosmed_train.py \
        --use_cosmos_trainer \
        --max_iter 10000

    # Inside Docker container:
    docker run --gpus all --ipc=host -it \
        -v "$HOME/cosmed:/workspace/cosmed" \
        -v "$HOME/cosmos-predict2.5:/workspace/cosmos-predict2.5" \
        -v "$HOME/datasets:/workspace/datasets" \
        -v "$HOME/checkpoints:/workspace/checkpoints" \
        nvcr.io/nvidia/nemo:25.09 \
        bash -c "cd /workspace/cosmed && python cosmed_train.py --data_dir /workspace/datasets/ChestMedicalData"
"""

import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import torch

# Add cosmos-predict2.5 to PYTHONPATH before importing anything else
COSMOS_ROOT = os.getenv("COSMOS_ROOT", "/workspace/cosmos-predict2.5")
if COSMOS_ROOT not in sys.path:
    sys.path.insert(0, COSMOS_ROOT)

# Now import Lightning and other modules
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy

# Import local modules
from cosmed_datamodule import CosmedDataModule
from cosmed_model_wrapper import CosmedModelWrapper
from cosmed_callbacks import (
    CosmedVisualizationCallback,
    CosmedMetricsCallback,
    CosmedRichProgressBar,
)

# =============================================================================
# Default Configuration Constants
# =============================================================================

# Default config file path for cosmos-predict2.5
DEFAULT_CONFIG_FILE = "cosmos_predict2/_src/predict2/configs/video2world/config.py"

# Default hyperparameters
DEFAULT_NUM_FRAMES = 121  # 360° rotation
DEFAULT_RESOLUTION = (256, 256)
DEFAULT_LEARNING_RATE = 2 ** (-14.5)  # ~4.3e-5
DEFAULT_WEIGHT_DECAY = 0.001
DEFAULT_GUIDANCE_SCALE = 7.0
DEFAULT_NUM_INFERENCE_STEPS = 35

# Default data settings
DEFAULT_DATA_DIR = "/workspace/datasets/ChestMedicalData"
DEFAULT_CACHE_DIR = "/workspace/datasets/ChestMedicalDataCache"
DEFAULT_OUTPUT_DIR = "/workspace/cosmed/outputs"

# Training settings
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_WORKERS = 8
DEFAULT_TRAIN_SAMPLES = 1000
DEFAULT_VAL_SAMPLES = 100
DEFAULT_MAX_EPOCHS = 100
DEFAULT_MAX_ITER = 10000


def setup_environment(args):
    """Setup environment variables for training."""
    cosmos_root = os.getenv("COSMOS_ROOT", "/workspace/cosmos-predict2.5")
    if cosmos_root not in sys.path:
        sys.path.insert(0, cosmos_root)
    
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # Avoid fork warning
    
    if args.gpus != "auto" and args.gpus.isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(int(args.gpus))])
    
    os.environ.setdefault("COSMOS_INTERNAL", "0")
    
    print(f"COSMOS_ROOT: {cosmos_root}")
    print(f"PYTHONPATH includes: {cosmos_root}")
    print(f"HF_HOME: {os.environ.get('HF_HOME', 'not set')}")


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser(description="Train Cosmed model with Cosmos-Predict2.5")
    
    # Training mode selection
    parser.add_argument(
        "--use_cosmos_trainer",
        action="store_true",
        help="Use Cosmos's native ImaginaireTrainer instead of PyTorch Lightning",
    )
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Root directory for datasets")
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR, help="Directory for cached preprocessed data")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for checkpoints and logs")
    
    # Model arguments
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to pretrained checkpoint (None for auto-download from HF)")
    parser.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES, help="Number of output frames (121 for 360° rotation)")
    parser.add_argument("--resolution", type=int, nargs=2, default=list(DEFAULT_RESOLUTION), help="Output resolution [H, W]")
    
    # Training arguments (Lightning mode)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size per GPU")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of data loading workers")
    parser.add_argument("--train_samples", type=int, default=DEFAULT_TRAIN_SAMPLES, help="Number of training samples per epoch")
    parser.add_argument("--val_samples", type=int, default=DEFAULT_VAL_SAMPLES, help="Number of validation samples per epoch")
    parser.add_argument("--max_epochs", type=int, default=DEFAULT_MAX_EPOCHS, help="Maximum number of training epochs (Lightning mode)")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate (default: 2^-14.5)")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight decay")
    parser.add_argument("--guidance_scale", type=float, default=DEFAULT_GUIDANCE_SCALE, help="Classifier-free guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS, help="Number of inference steps for validation")
    
    # Training arguments (Cosmos mode)
    parser.add_argument("--max_iter", type=int, default=DEFAULT_MAX_ITER, help="Maximum iterations (Cosmos trainer mode)")
    parser.add_argument("--checkpoint_save_iter", type=int, default=1000, help="Save checkpoint every N iterations (Cosmos trainer mode)")
    parser.add_argument("--logging_iter", type=int, default=100, help="Log every N iterations (Cosmos trainer mode)")
    
    # Hardware arguments
    parser.add_argument("--gpus", type=str, default="auto", help="Number of GPUs to use (or 'auto')")
    parser.add_argument("--precision", type=str, default="bf16-mixed", choices=["32", "16", "bf16", "bf16-mixed"], help="Training precision")
    parser.add_argument("--strategy", type=str, default="auto", choices=["auto", "ddp", "fsdp", "deepspeed"], help="Distributed training strategy")
    parser.add_argument("--context_parallel_size", type=int, default=1, help="Context parallel size for distributed inference")
    
    # Logging arguments
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="cosmed-cosmos-predict", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity/team name")
    parser.add_argument("--experiment_tag", type=str, default=None, help="Tag for this experiment run")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable W&B logging even if --use_wandb is set")
    
    # Checkpoint arguments
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--save_top_k", type=int, default=3, help="Save top K checkpoints")
    parser.add_argument("--save_every_n_epochs", type=int, default=5, help="Save checkpoint every N epochs")
    
    # Debug arguments
    parser.add_argument("--fast_dev_run", action="store_true", help="Run a fast development run for debugging")
    parser.add_argument("--limit_train_batches", type=int, default=None, help="Limit number of training batches (for debugging)")
    parser.add_argument("--limit_val_batches", type=int, default=None, help="Limit number of validation batches (for debugging)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    return parser.parse_args()


def create_callbacks(args):
    """Create Lightning callbacks."""
    callbacks = []
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="cosmed-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=args.save_top_k,
        save_last=True,
        every_n_epochs=args.save_every_n_epochs,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    progress_bar = CosmedRichProgressBar()
    callbacks.append(progress_bar)
    
    viz_callback = CosmedVisualizationCallback(
        output_dir=os.path.join(args.output_dir, "visualizations"),
        log_every_n_epochs=5,
        num_samples=4,
        save_to_disk=True,
    )
    callbacks.append(viz_callback)
    
    metrics_callback = CosmedMetricsCallback()
    callbacks.append(metrics_callback)
    
    return callbacks


def create_loggers(args):
    """Create Lightning loggers."""
    loggers = []
    
    tb_logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name="tensorboard",
        version=args.experiment_tag or "default",
    )
    loggers.append(tb_logger)
    
    if args.use_wandb and not args.disable_wandb:
        try:
            wandb_logger = WandbLogger(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.experiment_tag or f"cosmed-{args.num_frames}frames",
                save_dir=args.output_dir,
                log_model=True,
            )
            loggers.append(wandb_logger)
            print("✓ Weights & Biases logger initialized")
        except Exception as e:
            print(f"⚠ Warning: Could not initialize W&B logger: {e}")
    
    return loggers


def train_with_lightning(args):
    """Train using PyTorch Lightning."""
    print("\n" + "=" * 80)
    print("Cosmed Training with Cosmos-Predict2.5 (PyTorch Lightning Mode)")
    print("=" * 80)
    print(f"Data directory:     {args.data_dir}")
    print(f"Cache directory:    {args.cache_dir}")
    print(f"Output directory:   {args.output_dir}")
    print(f"Number of frames:   {args.num_frames}")
    print(f"Resolution:         {args.resolution}")
    print(f"Batch size:         {args.batch_size}")
    print(f"GPUs:               {args.gpus}")
    print(f"Precision:          {args.precision}")
    print(f"Max epochs:         {args.max_epochs}")
    print(f"Learning rate:      {args.learning_rate:.2e}")
    print("=" * 80 + "\n")
    
    seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize data module
    print("Initializing data module...")
    datamodule = CosmedDataModule(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        img_shape=args.resolution[0],
        vol_shape=256,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print("✓ Data module initialized")
    
    # Initialize model
    print("Initializing model...")
    model = CosmedModelWrapper(
        checkpoint_path=args.checkpoint_path,
        num_frames=args.num_frames,
        resolution=tuple(args.resolution),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        context_parallel_size=args.context_parallel_size,
    )
    print("✓ Model initialized")
    
    callbacks = create_callbacks(args)
    loggers = create_loggers(args)
    
    # Setup distributed strategy
    if args.strategy == "ddp":
        strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)
    elif args.strategy == "fsdp":
        from lightning.pytorch.strategies import FSDPStrategy
        strategy = FSDPStrategy(sharding_strategy="FULL_SHARD", cpu_offload=False)
    else:
        strategy = args.strategy
    
    devices = "auto" if args.gpus == "auto" else (int(args.gpus) if args.gpus.isdigit() else args.gpus)
    
    print("Initializing trainer...")
    trainer = Trainer(
        default_root_dir=args.output_dir,
        accelerator="auto",
        devices=devices,
        strategy=strategy,
        precision=args.precision,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=args.limit_train_batches if args.limit_train_batches else 1.0,
        limit_val_batches=args.limit_val_batches if args.limit_val_batches else 1.0,
        num_sanity_val_steps=2,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,
    )
    print("✓ Trainer initialized")
    
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=args.resume_from_checkpoint)
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
    if trainer.checkpoint_callback and hasattr(trainer.checkpoint_callback, 'best_model_path'):
        print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
        if hasattr(trainer.checkpoint_callback, 'best_model_score') and trainer.checkpoint_callback.best_model_score is not None:
            print(f"Best val loss:   {trainer.checkpoint_callback.best_model_score:.4f}")
    print("=" * 80 + "\n")


def train_with_cosmos_trainer(args):
    """Train using Cosmos's native ImaginaireTrainer."""
    print("\n" + "=" * 80)
    print("Cosmed Training with Cosmos-Predict2.5 (Cosmos Native Mode)")
    print("=" * 80)
    
    from cosmos_predict2._src.imaginaire.lazy_config import LazyConfig, instantiate
    from cosmos_predict2._src.imaginaire.trainer import ImaginaireTrainer
    from cosmos_predict2._src.imaginaire.utils import distributed, log, misc
    from cosmos_predict2._src.imaginaire.utils.config_helper import get_config_module, override
    from cosmos_predict2._src.predict2.configs.video2world.config import make_config
    from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey
    
    # Get default checkpoint for experiment name
    default_checkpoint = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]
    experiment_name = default_checkpoint.experiment
    
    print(f"Data directory:     {args.data_dir}")
    print(f"Cache directory:    {args.cache_dir}")
    print(f"Output directory:   {args.output_dir}")
    print(f"Experiment name:    {experiment_name}")
    print(f"Max iterations:     {args.max_iter}")
    print(f"Checkpoint interval: {args.checkpoint_save_iter}")
    print(f"Logging interval:   {args.logging_iter}")
    print("=" * 80 + "\n")
    
    config = make_config()
    
    override_opts = [
        "--",
        f"experiment={experiment_name}",
        f"trainer.max_iter={args.max_iter}",
        f"checkpoint.save_iter={args.checkpoint_save_iter}",
        f"trainer.logging_iter={args.logging_iter}",
    ]
    
    if args.disable_wandb:
        override_opts.append("job.wandb_mode=disabled")
    
    config = override(config, override_opts)
    config.validate()
    config.freeze()
    
    misc.set_random_seed(seed=args.seed, by_rank=True)
    
    torch.backends.cudnn.deterministic = config.trainer.cudnn.deterministic
    torch.backends.cudnn.benchmark = config.trainer.cudnn.benchmark
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
    
    print("Creating Cosmos trainer...")
    trainer = ImaginaireTrainer(config)
    print("✓ Trainer created")
    
    print("Loading model...")
    model = instantiate(config.model).cuda()
    model.on_train_start()
    print("✓ Model loaded")
    
    print("Creating dataloaders...")
    dataloader_train = instantiate(config.dataloader_train)
    dataloader_val = instantiate(config.dataloader_val) if config.dataloader_val else None
    print("✓ Dataloaders created")
    
    print("\n" + "=" * 80)
    print("Starting training with Cosmos trainer...")
    print("=" * 80 + "\n")
    
    trainer.train(model, dataloader_train, dataloader_val)
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80 + "\n")


def main():
    """Main entry point."""
    args = parse_args()
    setup_environment(args)
    
    if args.use_cosmos_trainer:
        train_with_cosmos_trainer(args)
    else:
        train_with_lightning(args)


if __name__ == "__main__":
    main()
