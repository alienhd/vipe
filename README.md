# ViPE: Video Pose Engine for Geometric 3D Perception

<p align="center">
  <img src="assets/teaser.gif" alt="teaser"/>
</p>

**TL;DR: ViPE is a useful open-source spatial AI tool for annotating camera poses and dense depth maps from raw videos! Now featuring AI-powered video colorization with temporal consistency and semantic understanding.**

**Contributors**: NVIDIA (Spatial Intelligence Lab, Dynamic Vision Lab, NVIDIA Issac, NVIDIA Research).

**Full Abstract**: Accurate 3D geometric perception is an important prerequisite for a wide range of spatial AI systems. While state-of-the-art methods depend on large-scale training data, acquiring consistent and precise 3D annotations from in-the-wild videos remains a key challenge. In this work, we introduce ViPE, a handy and versatile video processing engine designed to bridge this gap. ViPE efficiently estimates camera intrinsics, camera motion, and dense, near-metric depth maps from unconstrained raw videos. It is robust to diverse scenarios, including dynamic selfie videos, cinematic shots, or dashcams, and supports various camera models such as pinhole, wide-angle, and 360° panoramas. 

**NEW: Video Colorization Integration**: ViPE now includes comprehensive AI-based video colorization capabilities that leverage the robust 3D geometric perception to achieve temporally consistent and semantically plausible colorization of black and white videos. The colorization pipeline integrates semantic segmentation, depth-guided processing, and advanced deep learning models for realistic color generation.

We use ViPE to annotate a large-scale collection of videos. This collection includes around 100K real-world internet videos, 1M high-quality AI-generated videos, and 2K panoramic videos, totaling approximately 96M frames -- all annotated with accurate camera poses and dense depth maps. We open source ViPE and the annotated dataset with the hope to accelerate the development of spatial AI systems.

**[Technical Whitepaper](https://research.nvidia.com/labs/toronto-ai/vipe/assets/paper.pdf), [Project Page](https://research.nvidia.com/labs/toronto-ai/vipe), [Dataset](https://huggingface.co/) (Coming Soon)**

## Installation

To ensure the reproducibility, we recommend creating the runtime environment using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).

```bash
# Create a new conda environment and install 3rd-party dependencies
conda env create -f envs/base.yml
conda activate vipe
pip install -r envs/requirements.txt

# For video colorization functionality, install additional dependencies
pip install torch torchvision transformers diffusers timm open3d

# Build the project and install it into the current environment
# Omit the -e flag to install the project as a regular package
pip install --no-build-isolation -e .
```

## Usage

### Using the ViPE CLI

Once the python package is installed, you can use the `vipe` CLI to process raw videos in mp4 format.

#### Standard ViPE Processing

```bash
# Replace YOUR_VIDEO.mp4 with the path to your video. We provide sample videos in assets/examples.
vipe infer YOUR_VIDEO.mp4
# Additional options:
#   --output: Output directory (default: vipe_results)
#   --visualize: Enable visualization of intermediate and final results (default: false)
#   --pipeline: Pipeline configuration to use (default: default)
```

#### Video Colorization

```bash
# Basic colorization of black and white videos
vipe colorize YOUR_BW_VIDEO.mp4

# Colorization with custom settings
vipe colorize YOUR_BW_VIDEO.mp4 --output colorization_results/ --visualize --device cuda

# Using different colorization pipeline configurations
vipe colorize YOUR_BW_VIDEO.mp4 --pipeline colorization

# CPU-only colorization (slower but works without GPU)
vipe colorize YOUR_BW_VIDEO.mp4 --device cpu
```

![vipe-vis](assets/vipe-vis.gif)

One can visualize the results that ViPE produces by running (supported by `viser`):
```bash
vipe visualize vipe_results/
# Please modify the above vipe_results/ path to the output directory of your choice.
```

![vipe-viser](assets/vipe-viser.gif)

> We found that running [video-depth-anything](https://github.com/DepthAnything/Video-Depth-Anything) might eat up too much of GPU memory. To that end we provide a `no_vda` config that produces less temporally-stable depth (but empirically more 3D consistent) maps. This can be triggered by adding `--pipeline no_vda` to the `vipe infer` command.

### Using the `run.py` script

The `run.py` script is a more flexible way to run ViPE. Compared to the CLI, the script supports running on multiple videos at once and allows more fine-grained control over the pipeline with `hydra` configs. It also provides an example of using `vipe` as a library in your own project.

Example usages:

```bash
# Running the full pipeline.
python run.py pipeline=default streams=raw_mp4_stream streams.base_path=YOUR_VIDEO_OR_DIR_PATH

# Running the pose-only pipeline without depth estimation.
python run.py pipeline=default streams=raw_mp4_stream streams.base_path=YOUR_VIDEO_OR_DIR_PATH pipeline.post.depth_align_model=null

# Running video colorization pipeline
python run.py pipeline=colorization streams=raw_mp4_stream streams.base_path=YOUR_BW_VIDEO_OR_DIR_PATH

# Colorization with custom model settings
python run.py pipeline=colorization streams=raw_mp4_stream streams.base_path=YOUR_BW_VIDEO_OR_DIR_PATH colorization.inference_steps=30 colorization.guidance_scale=8.0
```

### Video Colorization Features

ViPE's colorization capabilities combine state-of-the-art AI models with robust 3D geometric understanding:

- **Temporal Consistency**: Leverages ViPE's SLAM system for stable colorization across frames
- **Semantic Awareness**: Uses object understanding to apply contextually appropriate colors  
- **Depth-Guided Processing**: Incorporates 3D geometry for realistic lighting and shading
- **Multiple Pre-trained Models**: Supports various colorization models for different use cases
- **Batch Processing**: Efficient processing of multiple videos with parallel GPU utilization
- **Custom Model Training**: Comprehensive training framework for developing specialized models
- **Model Evaluation**: Advanced metrics and benchmarking tools for quality assessment

For detailed colorization documentation, see [COLORIZATION_README.md](COLORIZATION_README.md).

## Colorization Model Training

ViPE provides comprehensive training capabilities for custom video colorization models. You can train models from scratch, fine-tune pre-trained models, or adapt existing models for specific use cases.

### Training Environment Setup

First, install the additional training dependencies:

```bash
# Install deep learning and training dependencies
pip install torch torchvision transformers diffusers timm open3d
pip install accelerate torchmetrics matplotlib seaborn pandas scipy
pip install tensorboard wandb  # For training monitoring (optional)
```

### Dataset Preparation

Prepare your training data in the following directory structure:

```
data/colorization_dataset/
├── train/
│   ├── grayscale/          # Grayscale input frames (PNG format)
│   ├── color/              # Corresponding color target frames (PNG format)
│   ├── depth/              # Optional: Depth maps (NPY format)
│   └── semantic/           # Optional: Semantic segmentation (NPY format)
└── val/
    ├── grayscale/
    ├── color/
    ├── depth/              # Optional
    └── semantic/           # Optional
```

To prepare datasets from videos:

```bash
# Extract frame pairs from color videos (creates grayscale/color pairs)
python -c "
from train_colorization import VideoColorizationDataset
import cv2
from pathlib import Path

def prepare_dataset_from_videos(video_dir, output_dir):
    video_files = list(Path(video_dir).glob('*.mp4'))
    output_path = Path(output_dir)
    
    (output_path / 'train' / 'grayscale').mkdir(parents=True, exist_ok=True)
    (output_path / 'train' / 'color').mkdir(parents=True, exist_ok=True)
    
    for video_file in video_files:
        cap = cv2.VideoCapture(str(video_file))
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Save color frame
            color_path = output_path / 'train' / 'color' / f'{video_file.stem}_{frame_idx:06d}.png'
            cv2.imwrite(str(color_path), frame)
            
            # Create and save grayscale frame
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_rgb = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            gray_path = output_path / 'train' / 'grayscale' / f'{video_file.stem}_{frame_idx:06d}.png'
            cv2.imwrite(str(gray_path), gray_rgb)
            
            frame_idx += 1
        cap.release()

# Usage: prepare_dataset_from_videos('path/to/videos', 'data/colorization_dataset')
"
```

### Training Configuration

Training is configured through YAML files. Several pre-configured setups are provided:

```bash
# Available training configurations
ls configs/training/
# stable_diffusion_finetune.yaml    - Fine-tune Stable Diffusion for colorization
# custom_unet.yaml                  - Train custom U-Net from scratch
# high_quality_colorization.yaml   - High-quality training settings
# realtime_colorization.yaml       - Fast inference-optimized training
```

Example configuration (`configs/training/stable_diffusion_finetune.yaml`):

```yaml
# Data configuration
data_dir: "./data/colorization_dataset"
output_dir: "./training_outputs/stable_diffusion_finetune"

# Model configuration  
model:
  type: "stable_diffusion"
  name: "runwayml/stable-diffusion-inpainting"
  params:
    use_depth_conditioning: true
    use_semantic_conditioning: true
    temporal_layers: true

# Training parameters
training:
  batch_size: 4
  epochs: 50
  mixed_precision: true
  gradient_accumulation_steps: 4

# Optimizer configuration
optimizer:
  type: "adamw"
  lr: 1e-5
  weight_decay: 0.01

# Loss function weights
loss:
  perceptual_weight: 1.0
  temporal_weight: 0.5
  semantic_weight: 0.3
```

### Training Commands

#### Train from Scratch

```bash
# Train a custom U-Net model
python train_colorization.py --config configs/training/custom_unet.yaml

# Train with custom data directory
python train_colorization.py --config configs/training/custom_unet.yaml \
    --override data_dir=./my_data output_dir=./my_training_output
```

#### Fine-tune Pre-trained Models

```bash
# Fine-tune Stable Diffusion for colorization
python train_colorization.py --config configs/training/stable_diffusion_finetune.yaml

# Resume training from checkpoint
python train_colorization.py --config configs/training/stable_diffusion_finetune.yaml \
    --resume training_outputs/stable_diffusion_finetune/checkpoint_epoch_10.pth
```

#### Distributed Training

```bash
# Multi-GPU training
accelerate launch --multi_gpu --num_processes=4 train_colorization.py \
    --config configs/training/stable_diffusion_finetune.yaml
```

### Model Saving and Loading

#### Model Storage

Trained models are automatically saved in the specified output directory:

```
training_outputs/your_experiment/
├── checkpoints/
│   ├── checkpoint_epoch_5.pth      # Regular checkpoints
│   ├── checkpoint_epoch_10.pth
│   └── best_model.pth              # Best model based on validation metrics
├── tensorboard/                    # Training logs for visualization
├── config.yaml                     # Training configuration used
└── training.log                    # Training progress log
```

#### Model File Contents

Each checkpoint contains:
- Model state dictionary (weights and biases)
- Optimizer state
- Training metadata (epoch, metrics, configuration)
- Learning rate scheduler state (if used)

#### Loading Trained Models

```python
# Loading a trained model for inference
import torch
from video_colorization_program import VideoColorizationPipeline

# Load model checkpoint
checkpoint = torch.load('training_outputs/my_model/best_model.pth')
model_state = checkpoint['model_state_dict']

# Initialize pipeline with custom model
pipeline = VideoColorizationPipeline(output_dir="./results")
pipeline.load_custom_model(model_state)

# Use for colorization
result = pipeline.process_video("input_video.mp4")
```

#### Using Trained Models with CLI

```bash
# Use trained model with ViPE CLI
vipe colorize input_video.mp4 --model-path training_outputs/my_model/best_model.pth

# Use with run.py script
python run.py pipeline=colorization streams=raw_mp4_stream \
    streams.base_path=input_video.mp4 \
    colorization.model_path=training_outputs/my_model/best_model.pth
```

### Training Monitoring

#### TensorBoard Visualization

```bash
# Start TensorBoard to monitor training
tensorboard --logdir training_outputs/your_experiment/tensorboard

# View metrics in browser at http://localhost:6006
```

Training metrics logged:
- Loss components (reconstruction, perceptual, temporal, semantic)
- Quality metrics (SSIM, PSNR)
- Learning rate progression
- Sample colorization results

#### Weights & Biases Integration

```bash
# Optional: Setup W&B for advanced experiment tracking
pip install wandb
wandb login

# Enable in training config
# wandb:
#   enabled: true
#   project: "vipe-colorization"
#   experiment_name: "stable-diffusion-finetune"
```

### Model Evaluation and Comparison

#### Evaluate Trained Models

```bash
# Evaluate model on test dataset
python evaluate_models.py \
    --dataset-name custom \
    --dataset-path data/test_videos/ \
    --models training_outputs/my_model/best_model.pth \
    --output-dir evaluation_results/

# Compare multiple trained models
python evaluate_models.py \
    --dataset-name custom \
    --dataset-path data/test_videos/ \
    --models \
        training_outputs/model_v1/best_model.pth \
        training_outputs/model_v2/best_model.pth \
        training_outputs/model_v3/best_model.pth \
    --output-dir model_comparison/
```

#### Enhanced Inference with Custom Models

```bash
# Use enhanced inference with trained model
python enhanced_inference.py \
    --input test_video.mp4 \
    --output results/ \
    --model-path training_outputs/my_model/best_model.pth \
    --ground-truth ground_truth_video.mp4  # Optional: for quality evaluation
```

### Training Tips and Best Practices

#### Dataset Preparation
- **Diverse Content**: Include various scenes, lighting conditions, and object types
- **High Quality**: Use high-resolution source videos (720p or higher)
- **Balanced Distribution**: Ensure good coverage of different video characteristics
- **Temporal Consistency**: Prefer videos with smooth motion and stable lighting

#### Training Configuration
- **Start Small**: Begin with lower resolution (256x256) then progressively increase
- **Transfer Learning**: Fine-tune from pre-trained models when possible
- **Batch Size**: Adjust based on GPU memory (typically 2-8 for diffusion models)
- **Learning Rate**: Use lower rates (1e-5 to 1e-4) for fine-tuning

#### Memory Optimization
```bash
# For GPU memory constraints
python train_colorization.py --config configs/training/stable_diffusion_finetune.yaml \
    --override training.batch_size=2 training.gradient_accumulation_steps=8 training.mixed_precision=true
```

#### Custom Loss Functions
```python
# Example: Add custom loss component in training config
loss:
  perceptual_weight: 1.0
  temporal_weight: 0.5
  semantic_weight: 0.3
  edge_weight: 0.2        # Custom edge preservation loss
  style_weight: 0.1       # Custom style consistency loss
```

### Troubleshooting

#### Common Issues

**Out of Memory Errors:**
```bash
# Reduce batch size and enable memory optimizations
python train_colorization.py --config configs/training/stable_diffusion_finetune.yaml \
    --override training.batch_size=1 training.mixed_precision=true
```

**Slow Training:**
```bash
# Use mixed precision and optimize data loading
python train_colorization.py --config configs/training/realtime_colorization.yaml \
    --override training.num_workers=8 training.mixed_precision=true
```

**Poor Quality Results:**
- Increase training epochs or learning rate
- Add more diverse training data
- Adjust loss function weights
- Use higher resolution training

For comprehensive training documentation, advanced techniques, and research details, see [COLORIZATION_README.md](COLORIZATION_README.md).


## Acknowledgments

ViPE is built on top of many great open-source research projects and codebases. Some of these include (not exhaustive):
- [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [Metric3Dv2](https://github.com/YvanYin/Metric3D)
- [PriorDA](https://github.com/SpatialVision/Prior-Depth-Anything)
- [UniDepth](https://github.com/lpiccinelli-eth/UniDepth)
- [VideoDepthAnything](https://github.com/DepthAnything/Video-Depth-Anything)
- [GeoCalib](https://github.com/cvg/GeoCalib)
- [Segment and Track Anything](https://github.com/z-x-yang/Segment-and-Track-Anything)

Please refer to the [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for a full list of projects and their licenses.

We thank useful discussions from Aigul Dzhumamuratova, Viktor Kuznetsov, Soha Pouya, and Ming-Yu Liu, as well as release support from Vishal Kulkarni.

## TODO

- [x] Initial code released under Apache 2.0 license.
- [x] Add comprehensive colorization model training documentation.
- [ ] Full dataset uploaded to Hugging Face for download.
- [ ] Add instructions to run inference on wide-angle and 360° videos.
- [ ] Add instructions for benchmarking.

## Citation

If you find ViPE useful in your research or application, please consider citing the following whitepaper:

```
@inproceedings{huang2025vipe,
    title={ViPE: Video Pose Engine for 3D Geometric Perception},
    author={Huang, Jiahui and Zhou, Qunjie and Rabeti, Hesam and Korovko, Aleksandr and Ling, Huan and Ren, Xuanchi and Shen, Tianchang and Gao, Jun and Slepichev, Dmitry and Lin, Chen-Hsuan and Ren, Jiawei and Xie, Kevin and Biswas, Joydeep and Leal-Taixe, Laura and Fidler, Sanja},
    booktitle={NVIDIA Research Whitepapers},
    year={2025}
}
```

## License

This project will download and install additional third-party **models and softwares**. Note that these models or softwares are not distributed by NVIDIA. Review the license terms of these models and projects before use. This source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).
