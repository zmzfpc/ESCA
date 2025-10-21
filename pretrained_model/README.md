# Pre-trained Models Directory

This directory stores **full-precision (FP32) pre-trained models** from the MultiFace dataset. These models are required as input for the quantization process.

## Directory Structure

Organize pre-trained models by subject ID and architecture:

```
pretrained_model/
├── 002643814/
│   ├── warp/
│   │   └── best_model.pth          # Full-precision WarpField VAE
│   └── base/
│       └── best_model.pth          # Full-precision 
└── ...
```

## Download Pre-trained Models


Download full-precision models following the [MultiFace](https://github.com/facebookresearch/multiface/blob/main/pretrained_model/index.html) repository instructions.

## Training Your Own Models

If you want to train your own pre-trained models:

1. Follow the [MultiFace training instructions](https://github.com/facebookresearch/multiface)
2. Train on your subject's data
3. Save the trained model as `best_model.pth`
4. Place in `pretrained_model/<subject_id>/<arch>/`

