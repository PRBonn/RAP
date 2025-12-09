#!/usr/bin/env bash
set -e

# Install Dependencies
# PyTorch 2.5.1
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Diffusers
pip install diffusers==0.33.0

# torch-cluster, torch-scatter, torch-sparse, spline-conv (For PointTransformer V3)
pip install ninja
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

# Pytorch3D
pip install https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt251/pytorch3d-0.7.8-cp310-cp310-linux_x86_64.whl

# FlashAttention 2.7.4 (work for PyTorch 2.5)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Trimesh
pip install trimesh==4.6.4

# Lightning
pip install lightning==2.5.2

# Torchmetrics
pip install torchmetrics==1.6.3

# Muon
pip install git+https://github.com/KellerJordan/Muon

# install other dependencies
pip install --ignore-installed -r requirements_other.txt 
