# ESCA: Enabling Seamless Codec Avatar Execution through Algorithm and Hardware Co-Optimization for Virtual Reality

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the implementation of ESCA, a full-stack optimization framework for accelerating Photorealistic Codec Avatar (PCA) inference on edge AR/VR platforms through efficient post-training quantization.

## Table of Contents

- [ESCA: Enabling Seamless Codec Avatar Execution through Algorithm and Hardware Co-Optimization for Virtual Reality](#esca-enabling-seamless-codec-avatar-execution-through-algorithm-and-hardware-co-optimization-for-virtual-reality)
  - [Table of Contents](#table-of-contents)
  - [Abstract](#abstract)
  - [Key Features](#key-features)
  - [Performance Results](#performance-results)
  - [Quick Start](#quick-start)
  - [Repository Structure](#repository-structure)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup Steps](#setup-steps)
    - [Data Preparation](#data-preparation)
  - [Pre-trained \& Quantized Model Checkpoints](#pre-trained--quantized-model-checkpoints)
    - [📥 Download from Google Drive](#-download-from-google-drive)
      - [What's Included](#whats-included)
    - [Directory Organization](#directory-organization)
    - [Loading Models](#loading-models)
  - [Usage](#usage)
    - [Post-Training Quantization](#post-training-quantization)
      - [Using the Training Script](#using-the-training-script)

## Abstract

Photorealistic Codec Avatars (PCA) enable high-fidelity human face rendering for immersive AR/VR communication but impose significant computational demands. ESCA addresses this challenge by providing an efficient post-training quantization (PTQ) method tailored for Codec Avatar models, enabling low-precision execution without compromising output quality, along with a custom hardware accelerator design.

## Key Features

- **Input Channel-wise Activation Smoothing (ICAS)**: Novel smoothing module to reduce extreme inter-channel activation disparities
- **Facial-Feature-Aware Smoothing (FFAS)**: Region-aware smoothing strategy using facial masks
- **UV-weighted Hessian-Based Weight Quantization**: Weight quantization guided by UV-weighted Hessian matrix


## Performance Results

- **Quality**: Up to +0.39 FovVideoVDP quality score improvement over best 4-bit baseline
- **Speed**: Up to 3.36× latency reduction
- **Frame Rate**: 100 FPS end-to-end rendering in real-time VR requirements

## Quick Start

Get started with ESCA in 3 simple steps:

```bash
# 1. Clone and setup environment
git clone https://github.com/zmzfpc/ESCA.git
cd ESCA
conda create -n esca python=3.13
conda activate esca
pip install -r requirement.txt

# 2. Download pre-trained models
# Place pre-trained models in: pretrained_model/<subject_id>/<arch>/best_model.pth

# 3. Run quantization
bash scripts/ptq_training_testing.sh
```

**For detailed setup and usage, see the [Installation](#installation) and [Usage](#usage) sections below.**

## Repository Structure

```
├── qlib/                   # Quantization library
│   ├── base.py             # Base quantization modules
│   ├── gptq.py             # GPTQ implementation
│   ├── ptq.py              # Post-training quantization
│   ├── ptq_trainer.py      # PTQ training framework
│   ├── quant.py            # Quantization utilities
│   ├── qwrap.py            # Model wrapper utilities
│   └── utils.py            # Utility functions
├── configs/                # Configuration files
│   └── camera_configs/     # Camera configuration files for different views
├── scripts/                # Training and evaluation scripts
│   └── ptq_training_testing.sh  # PTQ training script
├── runs/                   # Quantization experiment outputs (git-ignored)
├── pretrained_model/       # Pre-trained full-precision models
├── docs/                   # Documentation and assets
│   └── assets/             # Images, videos, and media files
├── profiling_plot/         # Profiling and visualization tools
├── test_segments_*/        # Test segment configurations
├── dataset.py              # Dataset handling
├── datasetlite.py          # Lightweight dataset loader
├── models.py               # Model architectures
├── quantize.py             # Main quantization script
├── visualize.py            # Visualization and evaluation
├── psnr.py                 # Image quality metrics
└── utils.py                # General utilities 
```

## Installation

### Prerequisites

- CUDA-compatible GPU (NVIDIA)
- CUDA Toolkit 11.8 or later
- Conda or Python 3.13+

### Setup Steps

1. **Create Environment**
   ```bash
   conda create -n esca python=3.13
   conda activate esca
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirement.txt
   ```

3. **Install PyTorch with CUDA Support**
   ```bash
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # Or for CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install nvdiffrast for Rendering**
   ```bash
   git clone https://github.com/NVlabs/nvdiffrast
   cd nvdiffrast
   pip install .
   cd ..
   ```

### Data Preparation

1. **Download MultiFace Dataset**
   
   Download the MultiFace dataset from the [official repository](https://github.com/facebookresearch/multiface):
   ```bash
   # Follow instructions from MultiFace repo to download dataset
   # Organize data in the following structure:
   # /path/to/multiface/
   # └── m--YYYYMMDD--HHMM--SUBJECT_ID--GHS/
   #     ├── KRT/
   #     ├── unwrapped_uv_1024/
   #     ├── tracked_mesh/
   #     └── frame_list.txt
   ```

2. **Download Pre-trained Models**
   
   The pre-trained full-precision models can be found in the [MultiFace](https://github.com/facebookresearch/multiface) repository. Place them in the `pretrained_model/<subject_id>/<arch>/` directory.
   
   ```bash
   mkdir -p pretrained_model/002643814/warp
   # Download and place best_model.pth in the above directory
   ```

## Pre-trained & Quantized Model Checkpoints

### 📥 Download from Google Drive

We provide our best quantized models for easy reproduction:

> **[📥 Download All Checkpoints from Google Drive](https://drive.google.com/drive/folders/YOUR_FOLDER_ID_HERE)**


#### What's Included

The checkpoint package includes:

| Type | Precision | Size | Description |
|------|-----------|------|-------------|
| **Full-precision models** | FP32 | ~120 MB | Pre-trained on MultiFace dataset |
| **W4A4 quantized models** | 4-bit / 4-bit | ~15 MB | ESCA optimized (recommended) |
| **W8A8 quantized models** | 8-bit / 8-bit | ~30 MB | High-quality baseline |


### Directory Organization

After downloading, organize the models as follows:

**Pre-trained Models** (required for quantization):
```
pretrained_model/
├── <subject_id>/
│   ├── warp/
│       └── best_model.pth          # Full-precision pre-trained model
```

**Quantized Models** (outputs from quantization, stored in `runs/`):
```
runs/                         # Optional: for storing best quantized models
├── <subject_id>/
│   └── <checkpoint_name>/
│       ├── model.pth 
```

### Loading Models

```python
import torch
from models import WarpFieldVAE

# Load pre-trained full-precision model
model_fp32 = WarpFieldVAE()
checkpoint = torch.load('pretrained_model/002643814/warp/best_model.pth')
model_fp32.load_state_dict(checkpoint)

# Load quantized model (from runs/ directory after quantization)
model_w4a4 = WarpFieldVAE()
checkpoint = torch.load('runs/experiment_002643814/PTQ_warp_w4a4_*/model.pth')
model_w4a4.load_state_dict(checkpoint)

# Or load from checkpoints/ if you saved the best quantized model there
checkpoint = torch.load('checkpoints/002643814/warp/quantized_w4a4.pth')
model_w4a4.load_state_dict(checkpoint)
```

## Usage

### Post-Training Quantization

#### Using the Training Script 

Run the pre-configured quantization script:

```bash
bash scripts/ptq_training_testing.sh
```

This script is pre-configured with optimal parameters for W4A4 quantization.

<!-- ## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{esca2024,
  title={ESCA: Enabling Seamless Codec Avatar Execution through Algorithm and Hardware Co-Optimization for Virtual Reality},
  author={Your Name and Others},
  booktitle={NeurIPS},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MultiFace](https://github.com/facebookresearch/multiface) - For providing the base dataset and models
- The ESCA team for developing this optimization framework -->









