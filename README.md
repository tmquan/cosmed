# cosmed

```bash
tmux # Start a new session

docker run \
  --gpus all \
  --ipc=host \
  -it \
  -v "$HOME/cosmed:/workspace/cosmed" \
  -v "$HOME/cosmos-predict2.5:/workspace/cosmos-predict2" \
  -v "$HOME/datasets:/workspace/datasets" \
  -v "$HOME/checkpoints:/workspace/checkpoints" \
  nvcr.io/nvidia/nemo:25.09 \
  bash
```

```bash
apt update
apt install -y fish
apt install -y libgl1
apt install -y libglib2.0-0t64
```

```bash
cd cosmed

uv pip uninstall pynvml

uv pip install -r requirements_docker.txt
uv pip install tyro
uv pip install --no-deps lightning
uv pip install nibabel SimpleITK scikit-image tqdm tensorboard
uv pip install --no-deps monai[all]
uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable" 
uv pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu129.html
```


```bash
# Test imports (you may see a pynvml deprecation warning - this is safe to ignore)
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

python -c "import torchvision; print(f'torchvision version: {torchvision.__version__}')"
```

```bash
# Generate 360* rotation videos
python process_nifty_to_video.py \
--datadir=/workspace/datasets/ChestXRLungSegmentation \
--destination=/workspace/datasets/cache
```