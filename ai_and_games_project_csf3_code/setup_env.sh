#!/bin/bash

# Stop on error (e) and undefined variables (u)
set -eu

# 1. Load Modules
echo "Loading Conda module..."
module purge
module load apps/binapps/conda/miniforge3

# 2. Define Env Name
ENV_NAME="hex_env"

# 3. Initialize Conda
eval "$(conda shell.bash hook)"

# 4. Create/Update Environment
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists. Activating..."
else
    echo "Creating environment '$ENV_NAME'..."
    # Create with Python 3.11
    conda create -n $ENV_NAME python=3.11 -y
fi

# 5. Activate
conda activate $ENV_NAME

# 6. Install PyTorch 2.6.0 + CUDA 12.4 (Using PIP for latest version)
# Note: We use pip here because Conda channels often lag behind for v2.6.0
echo "Installing PyTorch 2.6.0 (CUDA 12.4)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 7. Install other dependencies
echo "Installing NumPy and others..."
conda install numpy -y
# Or strictly use pip for consistency: pip install numpy

echo "=================================================================="
echo "Environment '$ENV_NAME' is ready!"
echo "PyTorch Version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA Available:  $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "=================================================================="