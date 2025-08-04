#!/bin/bash

# LION Training Environment Setup Script
# This script sets up the complete conda environment needed for LION 3D object detection with Linear RNN operators

set -e  # Exit on any error

echo "=== LION Training Environment Setup ==="
echo "This script will create a conda environment called 'lion' with all required dependencies."
echo

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first"
    exit 1
fi

# Environment name
ENV_NAME="lion"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove it and create a new one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Aborted."
        exit 1
    fi
fi

echo "Creating conda environment with Python 3.8.20..."
conda create -n ${ENV_NAME} python=3.8.20 -y

echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

echo "Installing CUDA toolkit 11.8..."
conda install cuda=11.8.0 -c nvidia/label/cuda-11.8.0 -y

echo "Installing PyTorch 2.1.0 with CUDA 11.8 support..."
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118

echo "Installing core dependencies with exact versions..."
pip install numpy==1.24.1
pip install scipy==1.10.1
pip install Pillow==10.2.0
pip install matplotlib==3.7.5
pip install opencv-python==4.11.0.86

echo "Installing sparse convolution and related packages..."
pip install spconv-cu118==2.3.6
pip install cumm-cu118==0.4.11
pip install pccm==0.4.16

echo "Installing causal-conv1d with exact version..."
pip install causal-conv1d==1.2.0.post2

echo "Installing PyTorch ecosystem packages..."
pip install tensorboardx==2.6.2.2
pip install torch-scatter==2.1.2+pt21cu118 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install timm==1.0.15
pip install triton==2.1.0

echo "Installing computer vision and image processing..."
pip install kornia==0.5.8
pip install kornia-rs==0.1.9
pip install scikit-image==0.21.0
pip install imageio==2.35.1
pip install tifffile==2023.7.10
pip install pywavelets==1.4.1

echo "Installing ML and data processing libraries..."
pip install transformers==4.38.2
pip install tokenizers==0.15.2
pip install safetensors==0.5.3
pip install huggingface-hub==0.33.0
pip install einops==0.8.1
pip install regex==2024.11.6

echo "Installing data handling and utilities..."
pip install pandas==2.0.3
pip install pyarrow==17.0.0
pip install polars==1.8.2
pip install av2==0.2.0
pip install av==12.3.0
pip install pyproj==3.5.0
pip install SharedArray==3.2.4

echo "Installing system and utility packages..."
pip install easydict==1.13
pip install pyyaml==6.0.2
pip install fire==0.7.0
pip install tqdm==4.67.1
pip install protobuf==5.29.5
pip install requests==2.28.1
pip install urllib3==1.26.13
pip install click==8.1.8
pip install colorlog==6.9.0
pip install termcolor==2.4.0
pip install rich==14.0.0

echo "Installing scientific computing packages..."
pip install numba==0.58.1
pip install llvmlite==0.41.1
pip install networkx==3.0
pip install sympy==1.13.3
pip install mpmath==1.3.0

echo "Installing development and build tools..."
pip install ninja==1.11.1.4
pip install pybind11==2.13.6
pip install packaging==25.0
pip install setuptools
pip install wheel

echo "Installing plotting and visualization..."
pip install contourpy==1.1.1
pip install cycler==0.12.1
pip install fonttools==4.57.0
pip install kiwisolver==1.4.7
pip install pyparsing==3.1.4

echo "Installing additional dependencies..."
pip install argcomplete==3.6.2
pip install attrs==25.3.0
pip install ccimport==0.4.4
pip install certifi==2022.12.7
pip install charset-normalizer==2.1.1
pip install dependency-groups==1.3.1
pip install distlib==0.3.9
pip install filelock==3.13.1
pip install fsspec==2024.6.1
pip install hf-xet==1.1.4
pip install importlib-metadata==8.5.0
pip install importlib-resources==6.4.5
pip install jinja2==3.1.4
pip install joblib==1.4.2
pip install lark==1.2.2
pip install lazy-loader==0.4
pip install markdown-it-py==3.0.0
pip install markupsafe==2.1.5
pip install mdurl==0.1.2
pip install nox==2025.5.1
pip install platformdirs==4.3.6
pip install portalocker==3.0.0
pip install pygments==2.19.1
pip install python-dateutil==2.9.0.post0
pip install pytz==2025.2
pip install six==1.17.0
pip install tomli==2.2.1
pip install typing-extensions==4.12.2
pip install tzdata==2025.2
pip install universal-pathlib==0.2.6
pip install virtualenv==20.31.2
pip install zipp==3.20.2

echo "Installing LION (pcdet) in development mode..."
python setup.py develop

echo "Installing mamba custom ops..."
cd pcdet/ops/mamba
python setup.py install
cd ../../..

echo "Verifying PyTorch CUDA availability..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Number of GPUs: {torch.cuda.device_count()}')"

echo
echo "=== Environment Setup Complete ==="
echo "Environment name: ${ENV_NAME}"
echo "Python version: 3.8.20"
echo
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo
echo "To verify the installation, you can run:"
echo "  python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\""
echo
echo "Next steps:"
echo "1. Activate the environment: conda activate ${ENV_NAME}"
echo "2. Download and prepare your dataset (KITTI, Waymo, etc.)"
echo "3. Start training with the provided training scripts"
echo
echo "Available training scripts in tools/:"
echo "- run_train_lion_for_kitti.sh"
echo "- run_train_lion_for_waymo.sh"  
echo "- run_train_lion_for_nus.sh"
echo "- run_train_lion_for_once.sh"
echo "- run_train_lion_for_argov2.sh"
echo
echo "To monitor training with tensorboard:"
echo "tensorboard --logdir=output/cfgs --port=6007 --host=0.0.0.0"
echo "Then open http://localhost:6007 in your browser"