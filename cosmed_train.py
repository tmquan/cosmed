"""
Cosmed Training Script for 360° rotation video generation.

Trains the Cosmed model to generate 360° rotation videos from frontal images:
1. CT: frontal projection -> 360° rotation video (supervised)
2. XR: frontal image -> 360° rotation video (cycle consistency)

Usage:
    python cosmed_train.py \\
        --batch_size 2 \\
        --max_epochs 100 \\
        --learning_rate 1e-4 \\
        --gpus 1
"""

import sys
import os

# Add cosmos-predict2.5 to Python path
COSMOS_PATH = "/workspace/cosmos-predict2.5"
if os.path.exists(COSMOS_PATH) and COSMOS_PATH not in sys.path:
    sys.path.insert(0, COSMOS_PATH)

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger

from cosmed_callbacks import CosmedVisualizationCallback
from cosmed_datamodule import CosmedDataModule
from cosmed_model import CosmedModel


def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(description="Train Cosmed 360° rotation video generation model")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="/workspace/datasets/ChestMedicalData")
    parser.add_argument("--cache_dir", type=str, default="/workspace/datasets/ChestMedicalDataCache")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_shape", type=int, default=256)
    parser.add_argument("--vol_shape", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--train_samples", type=int, default=1000)
    parser.add_argument("--val_samples", type=int, default=100)
    
    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dit_checkpoint_path", type=str, 
                       default=None,  # Will auto-download from HF if not provided
                       help="Path to Cosmos Predict 2.5 DiT checkpoint (auto-downloads if not provided)")
    parser.add_argument("--tokenizer_checkpoint_path", type=str,
                       default=None,  # Will auto-download from HF if not provided
                       help="Path to Cosmos Predict 2.5 tokenizer checkpoint (auto-downloads if not provided)")
    parser.add_argument("--text_encoder_path", type=str, default="google-t5/t5-11b",
                       help="Path or HF model ID for text encoder (T5)")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_epochs", type=int, default=100)
    
    # Loss weights
    parser.add_argument("--ct_video_loss_weight", type=float, default=1.0)
    parser.add_argument("--xr_cycle_loss_weight", type=float, default=0.5)
    parser.add_argument("--xr_input_loss_weight", type=float, default=1.0)
    
    # Output and logging
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--experiment_name", type=str, default="cosmed")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    
    # Trainer parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--precision", type=str, default="bf16-mixed", choices=["32", "16-mixed", "bf16-mixed"])
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--val_check_interval", type=float, default=1.0)
    
    # Visualization
    parser.add_argument("--num_visualization_samples", type=int, default=4)
    parser.add_argument("--visualize_every_n_epochs", type=int, default=1)
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    seed_everything(args.seed, workers=True)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = output_dir / f"{args.experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Cosmed 360° Rotation Video Generation Training")
    print("=" * 80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Output directory: {experiment_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Number of frames: {args.num_frames}")
    print(f"Image shape: {args.img_shape}")
    print(f"Max epochs: {args.max_epochs}")
    print("=" * 80)
    
    # Initialize data module
    print("\nInitializing data module...")
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
    
    # Initialize model
    print("Initializing model...")
    model = CosmedModel(
        dit_checkpoint_path=args.dit_checkpoint_path,
        tokenizer_checkpoint_path=args.tokenizer_checkpoint_path,
        text_encoder_path=args.text_encoder_path,
        num_frames=args.num_frames,
        img_height=args.img_shape,
        img_width=args.img_shape,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        ct_video_loss_weight=args.ct_video_loss_weight,
        xr_cycle_loss_weight=args.xr_cycle_loss_weight,
        xr_input_loss_weight=args.xr_input_loss_weight,
    )
    
    # Initialize logger
    tb_logger = TensorBoardLogger(
        save_dir=experiment_dir,
        name="tensorboard",
        version="",
    )
    
    # Initialize callbacks
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=experiment_dir / "checkpoints",
        filename="cosmed-{epoch:03d}-{val/total_loss:.4f}",
        monitor="val/total_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Visualization callback (now fixed for bf16-mixed precision)
    viz_callback = CosmedVisualizationCallback(
        num_samples=args.num_visualization_samples,
        log_every_n_epochs=args.visualize_every_n_epochs,
        save_to_disk=True,
        output_dir=experiment_dir / "validation_samples",
    )
    callbacks.append(viz_callback)
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        default_root_dir=experiment_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.gpus,
        precision=args.precision,
        max_epochs=args.max_epochs,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
        logger=tb_logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Print model summary
    print("\nModel summary:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Start training
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=args.resume_from_checkpoint,
    )
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
    print(f"Checkpoints saved to: {experiment_dir / 'checkpoints'}")
    print(f"Logs saved to: {experiment_dir}")
    
    # Run final validation
    if trainer.is_global_zero:
        print("\nRunning final validation...")
        trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
