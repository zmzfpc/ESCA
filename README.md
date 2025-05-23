# ESCA: Enabling Seamless Codec Avatar Execution through Algorithm and Hardware Co-Optimization for Virtual Reality

This repository contains the implementation of ESCA, a full-stack optimization framework for accelerating Photorealistic Codec Avatar (PCA) inference on edge AR/VR platforms through efficient post-training quantization and custom hardware acceleration.

## Abstract

Photorealistic Codec Avatars (PCA) enable high-fidelity human face rendering for immersive AR/VR communication but impose significant computational demands. ESCA addresses this challenge by providing an efficient post-training quantization (PTQ) method tailored for Codec Avatar models, enabling low-precision execution without compromising output quality, along with a custom hardware accelerator design.

## Key Features

- **Input Channel-wise Activation Smoothing (ICAS)**: Novel smoothing module to reduce extreme inter-channel activation disparities
- **Facial-Feature-Aware Smoothing (FFAS)**: Region-aware smoothing strategy using facial masks
- **UV-weighted Hessian-Based Weight Quantization**: Weight quantization guided by UV-weighted Hessian matrix
- **Custom Hardware Accelerator**: Specialized DNN accelerator with 4-bit and 8-bit operations support
- **Real-time Performance**: Achieves 100+ FPS rendering on AR/VR headsets


## Performance Results

- **Quality**: Up to +0.39 FovVideoVDP quality score improvement over best 4-bit baseline
- **Speed**: Up to 3.36× latency reduction
- **Frame Rate**: 100 FPS end-to-end rendering in real-time VR requirements

## Repository Structure

```
├── camera_configs/          # Camera configuration files for different view
├── profiling_plot/         # Profiling and visualization tools
├── qlib/                   # Quantization library
│   ├── base.py             # Base quantization modules
│   ├── gptq.py             # GPTQ implementation
│   ├── ptq.py              # Post-training quantization
│   ├── ptq_trainer.py      # PTQ training framework
│   ├── quant.py            # Quantization utilities
│   ├── qwrap.py            # Model wrapper utilities
│   └── utils.py            # Utility functions
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

1. **Environment Setup**
   ```bash
   conda create -n esca python=3.13
   conda activate esca
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirement.txt
   ```

3. **Necessary Requirements**
   - CUDA-compatible GPU
   - PyTorch with CUDA support
   - OpenCV
   - nvdiffrast for rendering

### Data Preparation

1. Download multiface dataset and place in the appropriate directory structure
2. The pre-trained full-precision model can be found in the [MultiFace](https://github.com/facebookresearch/multiface) repo.

### Post-Training Quantization

Run the quantization process:

```bash
bash ptq_training_testing.sh
```

Or run directly with custom parameters:

```bash
python quantize.py \
    --data_dir /path/to/multiface/data \
    --krt_dir /path/to/KRT \
    --model_ckpt ./pretrained_model/subject/arch/best_model.pth \
    --arch warp \
    --wbit 4 \
    --abit 4 \
    --mask True \
    --mask_weighted True \
    --result_path ./runs/experiment/
```

### Visualization and Evaluation

```bash
python visualize.py \
    --data_dir /path/to/data \
    --model_path /path/to/quantized/model.pth \
    --camera_config /path/to/camera/config.json \
    --test_segment /path/to/test/segments.json \
    --report_psnr True
```

## Key Parameters

### Quantization Parameters
- `--wbit`: Weight precision
- `--abit`: Activation precision
- `--mask`: Enable facial mask weighting
- `--omni`: Enable omnidirectional smoothing
- `--clip_ratio`: Clipping ratio for quantization

### Model Architectures
- `base`: Base DeepAppearance VAE
- `warp`: WarpField VAE with texture warping
- `res`: ResNet-based architecture
- `non`: Non-residual architecture
- `bilinear`: Bilinear upsampling architecture




