# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Cosmed Post-training Pipeline for Cosmos Predict 2.5

This script provides a complete post-training pipeline for medical imaging datasets
(XR images and CT videos) using the NVIDIA Cosmos Predict 2.5 framework.

Based on NVIDIA's post-training examples:
- https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/examples/posttraining/nemo/post_training_nemo_assets.py
- https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/examples/posttraining/groot/post_training_groot.py

Usage:
    python cosmed_posttrain.py --max_iters 1000 --datasets_dir /workspace/datasets/ChestMedicalDataCache
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()


def run(cmd, shell=False, check=True, cwd=None, env=None):
    """Utility function to run commands with consistent logging."""
    if cwd is None:
        cwd = os.getcwd()
    # Merge with current environment if env is provided
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    print(f"Running: {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    result = subprocess.run(cmd, shell=shell, check=check, cwd=cwd, env=run_env)
    return result


def copy_files_to_directory(source_path, dest_directory, copy_contents=True):
    """
    Generic function to copy files or directories to a destination directory.

    Args:
        source_path (str or Path): Path to source file or directory
        dest_directory (str or Path): Destination directory path
        copy_contents (bool): If True and source is a directory, copy its contents.
                             If False, copy the directory itself.
    """
    source_path = Path(source_path)
    dest_dir = Path(dest_directory)

    if not source_path.exists():
        print(f"Warning: Source path {source_path} does not exist. Skipping copy operation.")
        return

    # Create destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)

    if source_path.is_file():
        # Copy single file
        dest_file = dest_dir / source_path.name
        shutil.copy2(source_path, dest_file)
        print(f"Copied file: {source_path.name} to {dest_dir}")
    elif source_path.is_dir():
        if copy_contents:
            # Copy all contents from source directory to destination
            for item in source_path.iterdir():
                if item.is_file():
                    dest_file = dest_dir / item.name
                    shutil.copy2(item, dest_file)
                elif item.is_dir():
                    dest_subdir = dest_dir / item.name
                    shutil.copytree(item, dest_subdir, dirs_exist_ok=True)
            print(f"Successfully copied contents from {source_path} to {dest_dir}")
        else:
            # Copy the entire directory as a subdirectory
            dest_subdir = dest_dir / source_path.name
            shutil.copytree(source_path, dest_subdir, dirs_exist_ok=True)
            print(f"Copied directory: {source_path} to {dest_subdir}")
    else:
        print(f"Warning: {source_path} is neither a file nor a directory.")


class CosmedPostTrain:
    """
    A class to manage the complete post-training pipeline for Cosmed medical imaging datasets.

    This class provides a structured way to handle the entire workflow from
    preparing datasets to running inference, with configurable parameters
    and state tracking.
    
    Supports:
    - CT volume rotation videos (video2world task)
    - XR to CT conversion (image2world task)
    """

    def __init__(
        self,
        max_iters: int = None,
        checkpoint_save_iter: int = None,
        datasets_dir: str = "/workspace/datasets/ChestMedicalDataCache",
        checkpoints_base_dir: str = "/workspace/checkpoints",
        cosmos_repo_dir: str = "/workspace/cosmos-predict2.5",
        output_dir: str = "outputs/cosmed_posttraining",
        temp_checkpoint_base_dir: Optional[str] = None,
        ct_prompt: str = "A 360-degree rotational view of a chest CT scan showing anatomical structures from all angles.",
        xr_prompt: str = "A chest X-ray image showing anatomical structures of the thorax.",
        experiment_name: str = "predict2_video2world_training_2b_cosmed_ct",
        num_gpus: int = 8,
        video_size: tuple = (704, 1280),
        num_frames: int = 93,
        modality: str = "ct",  # "ct" for video or "xr" for images
    ):
        """
        Initialize the CosmedPostTrain pipeline.

        Args:
            max_iters (int): Maximum training iterations (defaults to env MAX_ITERS or 1000)
            checkpoint_save_iter (int): Frequency to save checkpoint
            datasets_dir (str): Directory containing the ChestMedicalDataCache dataset
            checkpoints_base_dir (str): Base directory for model checkpoints
            cosmos_repo_dir (str): Path to the cosmos-predict2.5 repository
            output_dir (str): Directory for inference outputs
            temp_checkpoint_base_dir (str): Base temporary directory for training checkpoints
            ct_prompt (str): Text prompt for CT video generation
            xr_prompt (str): Text prompt for XR image processing
            experiment_name (str): Name of the experiment configuration
            num_gpus (int): Number of GPUs for distributed training
            video_size (tuple): Target video resolution (height, width)
            num_frames (int): Number of frames in the video
            modality (str): Either "ct" for video or "xr" for images
        """
        # Environment variable MAX_ITERS takes precedence over command line argument
        self.max_iters = int(os.getenv("MAX_ITERS", max_iters or 100000))
        self.checkpoint_save_iter = checkpoint_save_iter or max(1, self.max_iters // 2)
        self.datasets_dir = Path(datasets_dir)
        self.checkpoints_base_dir = Path(checkpoints_base_dir)
        self.cosmos_repo_dir = Path(cosmos_repo_dir)
        self.output_dir = Path(output_dir)
        self.ct_prompt = ct_prompt
        self.xr_prompt = xr_prompt
        self.experiment_name = experiment_name
        self.num_gpus = num_gpus
        self.video_size = video_size
        self.num_frames = num_frames
        self.modality = modality

        # Generate unique job name with timestamp
        self.job_name = f"cosmed_{modality}_posttraining_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Use IMAGINAIRE_OUTPUT_ROOT if set, otherwise use default
        imaginaire_output_root = os.getenv("IMAGINAIRE_OUTPUT_ROOT", "/tmp/imaginaire4-output")
        if temp_checkpoint_base_dir is None:
            self.temp_checkpoint_base_dir = Path(
                f"{imaginaire_output_root}/cosmos_predict_v2p5/video2world/{self.job_name}/checkpoints"
            )
        else:
            self.temp_checkpoint_base_dir = Path(temp_checkpoint_base_dir)

        # Derived paths for dataset structure
        # ChestMedicalDataCache has ct/ and xr/ subdirectories
        self.ct_dir = self.datasets_dir / "ct"
        self.xr_dir = self.datasets_dir / "xr"
        self.videos_dir = self.ct_dir / "vid"
        self.images_dir = self.ct_dir / "img"
        self.metas_dir = self.ct_dir / "txt"
        self.xr_images_dir = self.xr_dir / "img"
        
        # Cosmos-compatible dataset directories (will be created)
        self.cosmos_base_dir = self.datasets_dir / "cosmos_format"
        
        # CT directories
        self.cosmos_ct_train_dir = self.cosmos_base_dir / "ct" / "train"
        self.cosmos_ct_test_dir = self.cosmos_base_dir / "ct" / "test"
        
        # XR directories  
        self.cosmos_xr_test_dir = self.cosmos_base_dir / "xr" / "test"
        
        # Default training directory (used by experiment config)
        self.cosmos_dataset_dir = self.cosmos_ct_train_dir
        self.cosmos_videos_dir = self.cosmos_dataset_dir / "videos"
        self.cosmos_metas_dir = self.cosmos_dataset_dir / "metas"
        self.cosmos_images_dir = self.cosmos_dataset_dir / "images"

        # Base checkpoint paths
        self.base_checkpoints_dir = self.checkpoints_base_dir / "base/pre-trained"
        self.tokenizer_path = self.checkpoints_base_dir / "tokenizer.pth"

        # Build temp checkpoint dir with proper 9-digit formatting
        self.temp_checkpoint_dir = self.temp_checkpoint_base_dir / f"iter_{self.max_iters:09d}"

        # Pipeline state tracking
        self.pipeline_state = {
            "dataset_prepared": False,
            "prompts_created": False,
            "experiment_registered": False,
            "training_completed": False,
            "checkpoints_converted": False,
            "inference_completed": False,
        }

        print(f"CosmedPostTrain initialized with max_iters={self.max_iters}")
        print(f"Dataset dir: {self.datasets_dir}")
        print(f"Checkpoints dir: {self.checkpoints_base_dir}")
        print(f"Cosmos repo dir: {self.cosmos_repo_dir}")
        print(f"Temp checkpoint dir: {self.temp_checkpoint_dir}")
        print(f"Modality: {self.modality}")

    def verify_dataset(self):
        """Verify that the dataset exists and has the expected structure."""
        print("Verifying dataset structure...")
        
        if self.modality == "ct":
            required_paths = [self.ct_dir, self.videos_dir, self.images_dir, self.metas_dir]
        else:  # xr
            required_paths = [self.xr_dir, self.xr_images_dir]
        
        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(f"Required path not found: {path}")
            print(f"  ✓ Found: {path}")
        
        # Count files
        if self.modality == "ct":
            video_count = len(list(self.videos_dir.glob("*.mp4")))
            image_count = len(list(self.images_dir.glob("*.png")))
            meta_count = len(list(self.metas_dir.glob("*.txt")))
            print(f"  Videos: {video_count}, Images: {image_count}, Metas: {meta_count}")
        else:
            image_count = len(list(self.xr_images_dir.glob("*.png")))
            print(f"  XR Images: {image_count}")
        
        return self

    def _link_or_copy(self, src: Path, dest: Path) -> bool:
        """Try to create a relative symlink, fall back to copying if that fails."""
        if dest.exists():
            return False
        try:
            # Use relative path for symlink (works better in Docker)
            # e.g., cosmos_format/videos/file.mp4 -> ../../ct/vid/file.mp4
            rel_path = os.path.relpath(src, dest)
            dest.symlink_to(rel_path)
            return True
        except (OSError, NotImplementedError):
            # Symlinks not supported - fall back to copying
            shutil.copy2(src, dest)
            return True

    def _prepare_split(self, src_vid_dir: Path, src_txt_dir: Path, dest_dir: Path, prefix: str, modality: str = "ct", progress: Progress = None, task_id = None):
        """Prepare a single split (train or test) for a modality."""
        videos_dir = dest_dir / "videos"
        metas_dir = dest_dir / "metas"
        
        videos_dir.mkdir(parents=True, exist_ok=True)
        metas_dir.mkdir(parents=True, exist_ok=True)
        
        video_count = 0
        meta_count = 0
        
        # Link video files
        if src_vid_dir.exists():
            video_files = list(src_vid_dir.glob(f"{prefix}*.mp4"))
            for video_file in video_files:
                dest = videos_dir / video_file.name
                if self._link_or_copy(video_file, dest):
                    video_count += 1
                if progress and task_id:
                    progress.advance(task_id)
        
        # Link meta files
        if src_txt_dir.exists():
            meta_files = list(src_txt_dir.glob(f"{prefix}*.txt"))
            for meta_file in meta_files:
                dest = metas_dir / meta_file.name
                if self._link_or_copy(meta_file, dest):
                    meta_count += 1
                if progress and task_id:
                    progress.advance(task_id)
        
        return video_count, meta_count

    def prepare_cosmos_dataset(self, prefix: str = "train_"):
        """Prepare dataset in Cosmos-compatible format for all splits.
        
        Creates:
        - cosmos_format/ct/train/ - CT training data (train_* files)
        - cosmos_format/ct/test/  - CT test data (test_* files)
        - cosmos_format/xr/test/  - XR test data (test_* files)
        
        The Cosmos VideoDataset expects in each directory:
        - videos/*.mp4
        - metas/*.txt
        """
        print("Preparing Cosmos-compatible dataset structure for all splits...")
        
        # Clear existing cosmos_format directory to ensure clean state
        if self.cosmos_base_dir.exists():
            print(f"  Clearing existing {self.cosmos_base_dir}...")
            shutil.rmtree(self.cosmos_base_dir)
        
        # Prepare CT train split (train_* files)
        print(f"\n  Preparing CT train split (prefix='train_')...")
        ct_train_vids, ct_train_metas = self._prepare_split(
            self.videos_dir, self.metas_dir, self.cosmos_ct_train_dir, "train_"
        )
        print(f"    CT train: {ct_train_vids} videos, {ct_train_metas} metas")
        
        # Prepare CT test split (test_* files)
        print(f"\n  Preparing CT test split (prefix='test_')...")
        ct_test_vids, ct_test_metas = self._prepare_split(
            self.videos_dir, self.metas_dir, self.cosmos_ct_test_dir, "test_"
        )
        print(f"    CT test: {ct_test_vids} videos, {ct_test_metas} metas")
        
        # Prepare XR test split (test_* files from xr/img)
        print(f"\n  Preparing XR test split (prefix='test_')...")
        xr_test_dir = self.cosmos_xr_test_dir
        xr_images_dir = xr_test_dir / "images"
        xr_metas_dir = xr_test_dir / "metas"
        xr_images_dir.mkdir(parents=True, exist_ok=True)
        xr_metas_dir.mkdir(parents=True, exist_ok=True)
        
        xr_image_count = 0
        if self.xr_images_dir.exists():
            for img_file in self.xr_images_dir.glob("test_*.png"):
                dest = xr_images_dir / img_file.name
                if self._link_or_copy(img_file, dest):
                    xr_image_count += 1
        print(f"    XR test: {xr_image_count} images")
        
        # Summary
        print(f"\n  Dataset prepared at: {self.cosmos_base_dir}")
        print(f"    CT train: {self.cosmos_ct_train_dir}")
        print(f"    CT test:  {self.cosmos_ct_test_dir}")
        print(f"    XR test:  {self.cosmos_xr_test_dir}")
        
        self.pipeline_state["dataset_prepared"] = True
        return self

    def create_prompts_for_dataset(self):
        """Create or verify prompt files for the dataset."""
        print("Creating/verifying prompts for dataset...")
        
        if self.modality == "ct":
            # For CT videos, prompts should match video filenames
            video_files = list(self.cosmos_videos_dir.glob("*.mp4"))
            prompt = self.ct_prompt
        else:
            # For XR images
            video_files = list(self.cosmos_images_dir.glob("*.png"))
            prompt = self.xr_prompt
        
        created_count = 0
        for video_file in video_files:
            meta_filename = self.cosmos_metas_dir / f"{video_file.stem}.txt"
            if not meta_filename.exists():
                with open(meta_filename, "w") as fp:
                    fp.write(prompt)
                created_count += 1
        
        print(f"Created {created_count} new prompt files")
        print(f"Total prompt files: {len(list(self.cosmos_metas_dir.glob('*.txt')))}")
        self.pipeline_state["prompts_created"] = True
        return self

    def create_experiment_config(self):
        """Create the experiment configuration file for Cosmos training."""
        print("Creating experiment configuration...")
        
        experiment_config = f'''# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Auto-generated Cosmed experiment configuration for Cosmos Predict 2.5
# Generated on: {datetime.now().isoformat()}

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import (
    VideoDataset,
    get_generic_dataloader,
    get_sampler,
)
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]


# Cosmed CT dataset and dataloader
cosmed_ct_video_dataset = L(VideoDataset)(
    dataset_dir="{self.cosmos_dataset_dir}",
    num_frames={self.num_frames},
    video_size={self.video_size},
)

dataloader_train_cosmed_ct = L(get_generic_dataloader)(
    dataset=cosmed_ct_video_dataset,
    sampler=L(get_sampler)(dataset=cosmed_ct_video_dataset),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)

# Video2World post-training configuration for 2B model with Cosmed CT data
# torchrun --nproc_per_node={self.num_gpus} --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- experiment={self.experiment_name}
{self.experiment_name} = dict(
    defaults=[
        f"/experiment/{{DEFAULT_CHECKPOINT.experiment}}",
        {{"override /data_train": "mock"}},
        {{"override /data_val": "mock"}},
        "_self_",
    ],
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="{self.job_name}",
    ),
    dataloader_train=dataloader_train_cosmed_ct,
    checkpoint=dict(
        save_iter={self.checkpoint_save_iter},
        # pyrefly: ignore  # missing-attribute
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
        load_from_object_store=dict(
            enabled=False,
        ),
        save_to_object_store=dict(
            enabled=False,
        ),
    ),
    optimizer=dict(
        lr=2 ** (-14.5),
        weight_decay=0.001,
    ),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[min(2000, {self.max_iters} // 5)],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=1000,
        max_iter={self.max_iters},
        callbacks=dict(
            heart_beat=dict(
                save_s3=False,
            ),
            iter_speed=dict(
                hit_thres=1000,
                save_s3=False,
            ),
            device_monitor=dict(
                save_s3=False,
            ),
            every_n_sample_reg=dict(
                every_n=200,
                save_s3=False,
            ),
            every_n_sample_ema=dict(
                every_n=200,
                save_s3=False,
            ),
            wandb=dict(
                save_s3=False,
            ),
            wandb_10x=dict(
                save_s3=False,
            ),
            dataloader_speed=dict(
                save_s3=False,
            ),
        ),
    ),
    model_parallel=dict(
        context_parallel_size=1,
    ),
)

cs = ConfigStore.instance()

# Register the configuration with Hydra ConfigStore
for _item in [
    {self.experiment_name},
]:
    # Get the experiment name from the global variable
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
'''
        
        # Write to cosmos experiments directory
        experiments_dir = self.cosmos_repo_dir / "cosmos_predict2/experiments/base"
        experiments_dir.mkdir(parents=True, exist_ok=True)
        
        config_path = experiments_dir / "cosmed.py"
        with open(config_path, "w") as f:
            f.write(experiment_config)
        
        print(f"Created experiment config: {config_path}")
        self.pipeline_state["experiment_registered"] = True
        return self

    def setup_environment_variables(self):
        """Set up required environment variables for training."""
        # Find tokenizer files
        tokenizer_pth = self.checkpoints_base_dir / "tokenizer.pth"
        
        # Look for mean_std.pt in various locations
        mean_std_candidates = [
            self.checkpoints_base_dir / "base/pre-trained/mean_std.pt",
            self.checkpoints_base_dir / "tokenizer/mean_std.pt",
            self.checkpoints_base_dir / "mean_std.pt",
        ]
        mean_std_path = None
        for candidate in mean_std_candidates:
            if candidate.exists():
                mean_std_path = candidate
                break
        
        if mean_std_path is None:
            # If not found, use a default path (will be handled by the training script)
            mean_std_path = self.checkpoints_base_dir / "mean_std.pt"
        
        # Add cosmos repo to PYTHONPATH
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        cosmos_path = str(self.cosmos_repo_dir)
        if cosmos_path not in current_pythonpath:
            new_pythonpath = f"{cosmos_path}:{current_pythonpath}" if current_pythonpath else cosmos_path
            os.environ["PYTHONPATH"] = new_pythonpath
        
        env_vars = {
            "COSMOS_INTERNAL": "0",
            "COSMOS_WAN2PT1_VAE_PATH": str(tokenizer_pth),
            "COSMOS_WAN2PT1_VAE_MEAN_STD_PATH": str(mean_std_path),
        }

        for key, value in env_vars.items():
            os.environ[key] = value
            print(f"Set {key}={value}")
        
        print(f"Set PYTHONPATH={os.environ.get('PYTHONPATH', '')}")

        return self

    def run_training(self):
        """Execute the training process."""
        print("Starting training...")

        # Set up environment
        self.setup_environment_variables()

        # Build training command - run script directly instead of -m
        cosmos_path = str(self.cosmos_repo_dir)
        script_path = f"{cosmos_path}/scripts/train.py"
        # Config path must be relative (from cosmos repo root) for the module import to work
        config_path = "cosmos_predict2/_src/predict2/configs/video2world/config.py"
        
        # Use export to ensure PYTHONPATH is available in the subprocess
        cmd_parts = [
            f"export PYTHONPATH={cosmos_path}:$PYTHONPATH &&",
            f"torchrun --nproc_per_node={self.num_gpus} --master_port=12341",
            script_path,
            f"--config={config_path}",
            "--",
            f"experiment={self.experiment_name}",
            f"trainer.max_iter={self.max_iters}",
            f"checkpoint.save_iter={self.checkpoint_save_iter}",
            "job.wandb_mode=disabled",
            f"job.name={self.job_name}",
        ]
        command = " ".join(cmd_parts)
        
        # Run from cosmos repo directory (important for relative config path)
        run(command, shell=True, cwd=str(self.cosmos_repo_dir))
        print("Training completed.")
        self.pipeline_state["training_completed"] = True
        return self

    def list_checkpoints(self):
        """List the trained checkpoints."""
        print("Listing trained checkpoints:")
        
        # List the base checkpoints directory
        if self.temp_checkpoint_base_dir.exists():
            run(["ls", "-la", str(self.temp_checkpoint_base_dir)])
        else:
            print(f"Warning: Checkpoint base directory {self.temp_checkpoint_base_dir} not found")

        # List specific iteration checkpoint directory
        if self.temp_checkpoint_dir.exists():
            run(["ls", "-la", str(self.temp_checkpoint_dir)])
        else:
            print(f"Warning: Checkpoint directory {self.temp_checkpoint_dir} not found")
        return self

    def convert_distcp_to_pt(self):
        """Convert distributed checkpoint to PyTorch format."""
        print("Converting distributed checkpoint to PyTorch format...")
        
        model_dir = self.temp_checkpoint_dir / "model"
        if not model_dir.exists():
            print(f"Warning: Model checkpoint directory {model_dir} not found")
            return self
        
        try:
            # Run the conversion script
            run(
                [
                    sys.executable,
                    "scripts/convert_distcp_to_pt.py",
                    str(model_dir),
                    str(self.temp_checkpoint_dir),
                ],
                cwd=str(self.cosmos_repo_dir)
            )
            print("Checkpoint conversion completed.")
            self.pipeline_state["checkpoints_converted"] = True
        except subprocess.CalledProcessError as e:
            print(f"Warning: Checkpoint conversion failed: {e}")

        return self

    def create_inference_config(self):
        """Create inference configuration JSON file."""
        print("Creating inference configuration...")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get sample input images/videos
        if self.modality == "ct":
            input_files = list(self.cosmos_images_dir.glob("*.png"))[:4]
            if not input_files and self.images_dir.exists():
                input_files = list(self.images_dir.glob("*.png"))[:4]
        else:
            input_files = list(self.cosmos_images_dir.glob("*.png"))[:4]
        
        if not input_files:
            print("Warning: No input files found for inference")
            return self
        
        # Create inference samples
        samples = []
        for img_path in input_files:
            sample = {
                "input_image": str(img_path.resolve()),
                "prompt": self.ct_prompt if self.modality == "ct" else self.xr_prompt,
                "num_input_frames": 1,
                "num_output_frames": self.num_frames,
            }
            samples.append(sample)
        
        # Write inference config
        inference_config_path = self.output_dir / "cosmed_inference.json"
        with open(inference_config_path, "w") as f:
            json.dump(samples, f, indent=2)
        
        print(f"Created inference config: {inference_config_path}")
        return inference_config_path

    def run_inference(self, input_json: Optional[str] = None):
        """
        Run inference with the trained model.

        Args:
            input_json (str): Path to the input JSON configuration file
        """
        print("Running inference...")
        
        # Create inference config if not provided
        if input_json is None:
            input_json = self.create_inference_config()
            if input_json is None:
                print("Warning: Could not create inference config")
                return self
        
        # Check for converted checkpoint
        ema_checkpoint = self.temp_checkpoint_dir / "model_ema_bf16.pt"
        if not ema_checkpoint.exists():
            print(f"Warning: EMA checkpoint not found at {ema_checkpoint}")
            # Try alternate paths
            alternate_paths = [
                self.temp_checkpoint_dir / "model_ema.pt",
                self.temp_checkpoint_dir / "model.pt",
            ]
            for alt_path in alternate_paths:
                if alt_path.exists():
                    ema_checkpoint = alt_path
                    break
            else:
                print("Warning: No checkpoint found for inference")
                return self
        
        run(
            [
                "torchrun",
                f"--nproc_per_node={self.num_gpus}",
                "examples/inference.py",
                "-i",
                str(input_json),
                "-o",
                str(self.output_dir),
                "--checkpoint-path",
                str(ema_checkpoint),
                "--experiment",
                self.experiment_name,
            ],
            cwd=str(self.cosmos_repo_dir)
        )

        print("Inference completed.")
        self.pipeline_state["inference_completed"] = True
        return self

    def get_pipeline_status(self):
        """Get the current status of the pipeline."""
        print("\n" + "=" * 40)
        print("          PIPELINE STATUS")
        print("=" * 40)
        for step, completed in self.pipeline_state.items():
            status = "✓" if completed else "✗"
            step_name = step.replace("_", " ").title()
            print(f"  {status} {step_name}")
        print("=" * 40 + "\n")
        return self.pipeline_state

    def list_dataset(self):
        """List contents of the dataset directory."""
        print("\nListing dataset directory:")
        run(["ls", "-la", str(self.datasets_dir)])
        
        if self.ct_dir.exists():
            print("\nListing CT directory:")
            run(["ls", "-la", str(self.ct_dir)])
        
        if self.xr_dir.exists():
            print("\nListing XR directory:")
            run(["ls", "-la", str(self.xr_dir)])
        
        if self.cosmos_dataset_dir.exists():
            print("\nListing Cosmos format directory:")
            run(["ls", "-la", str(self.cosmos_dataset_dir)])
        
        return self


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Cosmed Post-training Pipeline for Cosmos Predict 2.5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--max_iters",
        type=int,
        default=100000,
        help="Maximum training iterations (can be overridden by MAX_ITERS env variable)",
    )

    parser.add_argument(
        "--checkpoint_save_iter",
        type=int,
        default=None,
        help="Frequency to save model. Defaults to max_iters//2"
    )

    parser.add_argument(
        "--datasets_dir",
        type=str,
        default="/workspace/datasets/ChestMedicalDataCache",
        help="Directory containing the ChestMedicalDataCache dataset"
    )

    parser.add_argument(
        "--checkpoints_base_dir",
        type=str,
        default="/workspace/checkpoints",
        help="Base directory for model checkpoints"
    )

    parser.add_argument(
        "--cosmos_repo_dir",
        type=str,
        default="/workspace/cosmos-predict2.5",
        help="Path to the cosmos-predict2.5 repository"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/cosmed_posttraining",
        help="Directory for inference outputs"
    )

    parser.add_argument(
        "--temp_checkpoint_base_dir",
        type=str,
        default=None,
        help="Base temporary directory for training checkpoints"
    )

    parser.add_argument(
        "--ct_prompt",
        type=str,
        default="A 360-degree rotational view of a chest CT scan showing anatomical structures from all angles.",
        help="Text prompt for CT video generation"
    )

    parser.add_argument(
        "--xr_prompt",
        type=str,
        default="A chest X-ray image showing anatomical structures of the thorax.",
        help="Text prompt for XR image processing"
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="predict2_video2world_training_2b_cosmed_ct",
        help="Name of the experiment configuration"
    )

    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs for distributed training"
    )

    parser.add_argument(
        "--video_height",
        type=int,
        default=256,
        help="Target video height"
    )

    parser.add_argument(
        "--video_width",
        type=int,
        default=256,
        help="Target video width"
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=121,
        help="Number of frames to sample from each video (must be <= video length)"
    )

    parser.add_argument(
        "--modality",
        type=str,
        default="ct",
        choices=["ct", "xr"],
        help="Modality: 'ct' for video or 'xr' for images"
    )

    parser.add_argument(
        "--skip_prepare",
        action="store_true",
        help="Skip dataset preparation step"
    )

    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training step"
    )

    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Skip inference step"
    )

    parser.add_argument(
        "--only_verify",
        action="store_true",
        help="Only verify dataset structure, don't run training"
    )

    parser.add_argument(
        "--file_prefix",
        type=str,
        default="train_",
        help="Only include files starting with this prefix (e.g., 'train_' or 'test_')"
    )

    return parser.parse_args()


def main():
    """Main function with command-line argument support."""
    args = parse_arguments()

    # Create pipeline with parsed arguments
    pipeline = CosmedPostTrain(
        max_iters=args.max_iters,
        checkpoint_save_iter=args.checkpoint_save_iter,
        datasets_dir=args.datasets_dir,
        checkpoints_base_dir=args.checkpoints_base_dir,
        cosmos_repo_dir=args.cosmos_repo_dir,
        output_dir=args.output_dir,
        temp_checkpoint_base_dir=args.temp_checkpoint_base_dir,
        ct_prompt=args.ct_prompt,
        xr_prompt=args.xr_prompt,
        experiment_name=args.experiment_name,
        num_gpus=args.num_gpus,
        video_size=(args.video_height, args.video_width),
        num_frames=args.num_frames,
        modality=args.modality,
    )

    print(f"\n{'=' * 60}")
    print("  COSMED POST-TRAINING PIPELINE")
    print(f"  Max iterations: {pipeline.max_iters}")
    print(f"  Modality: {pipeline.modality.upper()}")
    print(f"  GPUs: {pipeline.num_gpus}")
    print(f"{'=' * 60}\n")

    # Verify dataset
    try:
        pipeline.verify_dataset()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the dataset is properly structured.")
        sys.exit(1)

    if args.only_verify:
        pipeline.list_dataset()
        print("Dataset verification completed.")
        return

    # Prepare dataset
    if not args.skip_prepare:
        pipeline.prepare_cosmos_dataset()
        pipeline.create_prompts_for_dataset()
        pipeline.create_experiment_config()
        pipeline.list_dataset()

    # Training step
    if not args.skip_training:
        pipeline.run_training()
        pipeline.list_checkpoints()
        pipeline.convert_distcp_to_pt()

    # Inference step
    if not args.skip_inference:
        pipeline.run_inference()

    print("\nPipeline completed!")
    pipeline.get_pipeline_status()


if __name__ == "__main__":
    main()

