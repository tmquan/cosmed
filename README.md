# cosmed

```bash
tmux # Start a new session

docker run \
  --gpus all \
  --ipc=host \
  -it \
  -v "$HOME/cosmos-predict2.5:/workspace/cosmos-predict2" \
  -v "$HOME/cosmos-ct-multiview:/workspace/cosmos-ct-multiview" \
  -v "$HOME/data:/workspace/data" \
  -v "$HOME/checkpoints:/workspace/checkpoints" \
  nvcr.io/nvidia/nemo:25.09 \
  bash
```

```bash
apt update
apt install -y fish
```

```bash
cd cosmos-ct-multiview
uv pip uninstall pynvml

uv pip install -r requirements_docker.txt

uv pip install nibabel SimpleITK scikit-image tqdm tensorboard
uv pip install --no-deps monai[all]==1.5.1
uv pip install --no-deps lightning==2.0.0
uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable" 
uv pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu129.html

```

```bash
# Cosmos CT Multiview - Docker Requirements
# For use with nvcr.io/nvidia/nemo:25.09 container
# Note: PyTorch may be in system Python but not in venv, install explicitly

# PyTorch with CUDA 12.9 (for NeMo 25.09)
# Using --extra-index-url to add PyTorch repo while keeping default PyPI
# --extra-index-url https://download.pytorch.org/whl/cu129
# torch>=2.0.0
# torchvision>=0.15.0
# torchaudio>=2.0.0

# Core dependencies (lightweight, not in base container)
einops>=0.7.0
nvidia-ml-py>=12.535.133  # Replaces deprecated pynvml

# Medical imaging (NOT in NeMo container)
# monai[all]>=1.5.1

# 3D rendering dependencies (PyTorch3D installed separately)
# NOTE: PyTorch3D must be installed SEPARATELY with --no-build-isolation flag
# Run: uv pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
# fvcore
# iopath

# Computer vision extras
opencv-python>=4.8.0
Pillow>=10.0.0
imageio>=2.31.0

# Scientific computing (may need upgrade)
scipy>=1.11.0

# Configuration management
omegaconf>=2.3.0
hydra-core>=1.3.0

# Utilities
tqdm>=4.66.0
rich>=13.0.0
loguru>=0.7.0

# Visualization (tensorboard likely in NeMo, but ensure version)
wandb>=0.15.0
matplotlib>=3.7.0

# Data processing
pandas>=2.0.0
h5py>=3.9.0

# Configuration files
pyyaml>=6.0

```

```bash
# Test imports (you may see a pynvml deprecation warning - this is safe to ignore)
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

python -c "import torchvision; print(f'torchvision version: {torchvision.__version__}')"
```

```bash
# Test datamodule
python cm_datamodule.py --datadir /workspace/data
```

```bash
# Generate 360* rotation videos
python ct_to_video_cache.py --ct_folders=/workspace/data/ChestXRLungSegmentation/

```

```bash

```