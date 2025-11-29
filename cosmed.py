# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Auto-generated Cosmed experiment configuration for Cosmos Predict 2.5
# Generated on: 2025-11-29T16:20:11.844603
# cosmos-predict2.5/cosmos_predict2/experiments/base/cosmed.py
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
    dataset_dir="/workspace/datasets/ChestMedicalDataCache/cosmos_format/ct/train",
    num_frames=120,
    video_size=(256, 256),
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
# torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- experiment=predict2_video2world_training_2b_cosmed_ct
predict2_video2world_training_2b_cosmed_ct = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="cosmed_ct_posttraining_20251129_160829",
    ),
    dataloader_train=dataloader_train_cosmed_ct,
    checkpoint=dict(
        save_iter=500,
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
        warm_up_steps=[min(2000, 1000 // 5)],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=1000,
        callbacks=dict(
            heart_beat=dict(
                save_s3=False,
            ),
            iter_speed=dict(
                hit_thres=100,
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
    predict2_video2world_training_2b_cosmed_ct,
]:
    # Get the experiment name from the global variable
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
