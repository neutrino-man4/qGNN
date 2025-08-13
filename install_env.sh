#!/bin/bash

# Minimal qGNN Environment Setup Script
# Author: Aritra Bal, ETP
# Date: XIII Idibus Sextilibus anno ab urbe condita MMDCCLXXVIII

set -e
echo -e "This (archaic CUDA 11.8) installation is meant to work on the machine Deepthought, set up \033[9mon the planet of Magrathea 7.5M years ago\033[0m at the ETP quite some years back."
echo "Setting up qGNN environment..."

# Remove existing environment if it exists
if conda env list | grep -q "^qGNN "; then
    echo "Removing existing qGNN environment..."
    conda env remove -n qGNN -y
fi

# Create environment
echo "Creating qGNN environment with Python 3.10..."
conda create -n qGNN python=3.10 -y

# Install PyTorch
echo "Installing PyTorch 2.4.1 with CUDA 11.8..."
conda run -n qGNN pip install packaging>=20.0
conda run -n qGNN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

# Install all other packages
echo "Installing PyTorch Geometric and other packages..."
conda run -n qGNN pip install torch-geometric==2.5.3 torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.1+cu118.html && conda run -n qGNN pip install numpy==1.26.4 scipy==1.12.0 matplotlib==3.8.4 scikit-learn==1.4.2 pandas==2.2.2 awkward==2.6.8 uproot==5.3.7 h5py==3.11.0 loguru==0.7.2 tqdm==4.66.4 pyyaml==6.0.1 seaborn==0.13.2 jupyter==1.0.0 ipython==8.25.0 tensorboard==2.17.0

# Quick verification
echo "Verifying installation..."
conda run -n qGNN python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
conda run -n qGNN python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"

echo "✅ qGNN environment ready!"
echo "Activate with: conda activate qGNN"