# Cosmed - 360° Medical Imaging Rotation Video Generation

Post-training system for Cosmos Predict 2.5 to generate 360° rotation videos from frontal medical images (CT/XR).

## Quick Start

### 1. Start Docker Container

```bash
tmux # Start a new session

docker run \
  --gpus all \
  --ipc=host \
  -it \
  -v "$HOME/cosmed:/workspace/cosmed" \
  -v "$HOME/cosmos-predict2.5:/workspace/cosmos-predict2.5" \
  -v "$HOME/datasets:/workspace/datasets" \
  -v "$HOME/checkpoints:/workspace/checkpoints" \
  nvcr.io/nvidia/nemo:25.09 \
  bash
```

### 2. Install System Dependencies

```bash
apt update
apt install -y fish libgl1 libglib2.0-0t64
```

### 3. Quick Setup (Automated)

Use the provided setup script for automated installation:

```bash
cd /workspace/cosmed
sh setup_environment.sh
```

This script will:
- ✓ Install Lightning framework
- ✓ Install Cosmed requirements (MONAI, PyTorch3D, etc.)
- ✓ Install Cosmos Predict 2.5 with CUDA 12.8 extras using uv
- ✓ Create virtual environment at `/workspace/cosmos-predict2.5/.venv`
- ✓ Verify all installations

**Important:** Cosmos Predict 2.5 uses a separate virtual environment. To use Cosmos:

```bash
# Activate the Cosmos venv
source /workspace/cosmos-predict2.5/.venv/bin/activate

# Test Cosmos
python -c 'from cosmos_predict2._src.predict2.models.video2world_model import Video2WorldModel; print("✓ Cosmos OK")'
```

### 3. Manual Setup (Alternative)

If you prefer manual installation with more control:

```bash
cd /workspace/cosmed

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Remove conflicting package
uv pip uninstall pynvml

# Install cosmed requirements
uv pip install -r requirements_docker.txt
uv pip install tyro
uv pip install lightning  # Modern lightning package
uv pip install nibabel SimpleITK scikit-image tqdm tensorboard

# Install MONAI (medical imaging library)
uv pip install --no-deps monai[all]

# Install PyTorch3D (for volume rendering)
uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install Kaolin (optional, for advanced rendering)
uv pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.0_cu128.html

# Install Cosmos OSS (local dependency)
cd /workspace/cosmos-predict2.5/packages/cosmos-oss
pip install -e .

# Install Cosmos Predict 2.5 with CUDA 12.8 extras
cd /workspace/cosmos-predict2.5
/root/.local/bin/uv sync --extra=cu128

# Verify installations
python -c "import lightning; print(f'✓ Lightning: {lightning.__version__}')"
python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"
python -c "import monai; print(f'✓ MONAI: {monai.__version__}')"

# Test Cosmos (from venv)
/workspace/cosmos-predict2.5/.venv/bin/python -c 'from cosmos_predict2._src.predict2.models.video2world_model import Video2WorldModel; print("✓ Cosmos OK")'
```

### 4. Download Cosmos Checkpoint

```bash
cd /workspace/cosmed

# Set your Hugging Face token (get from https://huggingface.co/settings/tokens)
export HF_TOKEN="your_token_here"

# Download checkpoint
python download_cosmos_checkpoint.py

# Checkpoint will be saved to /workspace/checkpoints/base/
```

### 5. Generate Cache

```bash
cd /workspace/cosmed

# Test with 10 files first (quick test)
python cosmed_datamodule.py \
  --mode cache \
  --data_dir /workspace/datasets/ChestMedicalData \
  --cache_dir /workspace/datasets/ChestMedicalDataCache \
  --max_files 10 \
  --img_shape 256 \
  --num_frames 121

# Generate full cache (CT + XR, may take hours)
python cosmed_datamodule.py \
  --mode cache \
  --data_dir /workspace/datasets/ChestMedicalData \
  --cache_dir /workspace/datasets/ChestMedicalDataCache \
  --img_shape 256 \
  --vol_shape 256 \
  --num_frames 121

# Or generate CT only
python cosmed_datamodule.py --mode cache --skip_xr

# Or generate XR only
python cosmed_datamodule.py --mode cache --skip_ct
```

### 6. Test Data Loading

```bash
python cosmed_datamodule.py \
  --mode test \
  --data_dir /workspace/datasets/ChestMedicalData \
  --cache_dir /workspace/datasets/ChestMedicalDataCache \
  --batch_size 2
```

### 7. Train Model

```bash
cd /workspace/cosmed

# First epoch test
python cosmed_train.py \
  --data_dir /workspace/datasets/ChestMedicalData \
  --cache_dir /workspace/datasets/ChestMedicalDataCache \
  --dit_checkpoint_path /workspace/checkpoints/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt \
  --batch_size 2 \
  --max_epochs 1 \
  --train_samples 100 \
  --val_samples 20 \
  --img_shape 256 \
  --num_frames 121 \
  --learning_rate 1e-4 \
  --gpus 1 \
  --precision 16-mixed

# Full training with config
python cosmed_train.py \
  --config cosmed_config.yaml \
  --dit_checkpoint_path /workspace/checkpoints/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt
```

### 8. Monitor Training

```bash
# In another terminal
tensorboard --logdir /workspace/cosmed/outputs/
# Open browser to http://localhost:6006
```

## Cache Structure

The cache is organized by modality:

```
ChestMedicalDataCache/
├── ct/                          # CT volume data
│   ├── vol/                     # Preprocessed volumes
│   │   ├── train_<hash>.nii.gz
│   │   └── test_<hash>.nii.gz
│   ├── vid/                     # Rotation videos (121 frames)
│   │   ├── train_<hash>.mp4
│   │   └── test_<hash>.mp4
│   ├── img/                     # Frontal projections
│   │   ├── train_<hash>.png
│   │   └── test_<hash>.png
│   └── txt/                     # Text prompts
│       ├── train_<hash>.txt
│       └── test_<hash>.txt
└── xr/                          # XR image data
    └── img/                     # Preprocessed XR images
        ├── train_<hash>.png
        └── test_<hash>.png
```

## Key Scripts

- **`cosmed_datamodule.py`** - Data loading + cache generation
  - Mode `cache`: Pre-generate cache files
  - Mode `test`: Test data loading
  
- **`cosmed_train.py`** - Training script
  - Supports multi-GPU (DDP, FSDP)
  - Mixed precision training
  - TensorBoard logging
  
- **`cosmed_model.py`** - Model wrapper
  - Loads Cosmos Predict 2.5
  - Implements dual loss (CT + XR)
  
- **`cosmed_inference.py`** - Inference script
  - Generate from single image
  - Batch processing

## Troubleshooting

**No module named 'lightning'**
```bash
pip install lightning
```

**CUDA extra not installed**
```bash
# Use uv to properly install Cosmos with CUDA extras
cd /workspace/cosmos-predict2.5
/root/.local/bin/uv sync --extra=cu128

# Test from venv
/workspace/cosmos-predict2.5/.venv/bin/python -c 'from cosmos_predict2 import __version__; print(f"✓ Cosmos {__version__}")'
```

**Cannot import Cosmos modules**
```bash
# Make sure to use the venv Python or activate it
source /workspace/cosmos-predict2.5/.venv/bin/activate
python -c 'from cosmos_predict2._src.predict2.models.video2world_model import Video2WorldModel; print("✓ OK")'
```

**Model shows 808K params instead of 2B**
- Cosmos not loading properly
- Check CUDA extras are installed
- Verify checkpoint path is correct

**Out of memory**
```bash
# Reduce batch size
python cosmed_train.py --batch_size 1 ...
```

**Cache not found**
```bash
# Generate cache first
python cosmed_datamodule.py --mode cache
```

## Documentation

For more details, see:
- **COSMED_CACHE_GENERATION.md** - Complete cache generation guide
- **COSMED_README.md** - Full project documentation
- **QUICK_START.md** - Step-by-step quick start
- **IMPLEMENTATION_SUMMARY.md** - Technical implementation details

## Expected Timeline

- Setup: 30 minutes
- Cache generation (6000 CT): 10-20 hours
- Cache generation (18000 XR): 1-3 hours
- Training (1 epoch): 30 minutes
- Training (100 epochs): 50 hours

## Hardware Requirements

- Minimum: 1x GPU with 24GB VRAM
- Recommended: 4x GPU with 40-80GB VRAM
- Storage: 500GB+ for cache
- RAM: 64GB+
