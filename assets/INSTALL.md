# Installation

This codebase is tested on Ubuntu 20.04.2 LTS with python 3.10. Follow the below steps to create environment and install dependencies.

* Setup conda environment (recommended).
```bash
# Create a conda environment
conda create -n EviSteer python=3.10 -y

# Activate the environment
conda activate EviSteer

# Install torch (requires version >= 2.0.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

* Clone Evi-Steer code repository and install requirements
```bash
# Clone Evi-Steer code base
git clone https://github.com/HealthX-Lab/Evi-Steer

cd Evi-Steer/
# Install requirements

pip install -r requirements.txt

```
