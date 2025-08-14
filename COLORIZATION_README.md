# Video Colorization with ViPE

This implementation extends ViPE (Video Pose Engine) with comprehensive AI-based video colorization capabilities. The system leverages ViPE's robust 3D geometric perception to achieve temporally consistent and semantically plausible colorization of black and white videos.

## Overview

The video colorization pipeline combines:
- **ViPE's 3D Geometric Perception**: Robust camera pose estimation and metric depth maps
- **Semantic Understanding**: Video semantic segmentation for object-level context
- **Advanced Colorization**: Multi-modal deep learning models for realistic color generation
- **Temporal Consistency**: Leveraging 3D structure for stable colorization across frames

## Architecture

### 1. Core Components

#### Standalone Program (`video_colorization_program.py`)
A complete, standalone implementation that can be used independently:
```python
from video_colorization_program import VideoColorizationPipeline

pipeline = VideoColorizationPipeline(output_dir="./colorization_output")
colorized_video = pipeline.process_video("input_bw_video.mp4")
```

#### ViPE-Integrated Pipeline (`vipe/pipeline/colorization.py`)
Full integration with ViPE's architecture for enhanced 3D understanding:
```bash
vipe colorize input_video.mp4 --output colorization_results/
```

### 2. Processing Stages

1. **Video Pre-processing**: Frame extraction and preparation
2. **3D Geometric Perception**: ViPE's SLAM system for camera poses and depth
3. **Semantic Segmentation**: Object-level understanding with temporal consistency
4. **Point Cloud Generation**: 3D semantic point clouds for spatial understanding
5. **Multi-Modal Colorization**: AI-powered color generation using all available cues
6. **Video Reconstruction**: Final colorized video assembly

### 3. Key Features

- **Temporal Consistency**: Leverages 3D structure to maintain color consistency across frames
- **Semantic Awareness**: Uses object understanding to apply contextually appropriate colors
- **Depth-Guided Colorization**: Incorporates 3D geometry for realistic lighting and shading
- **Modular Design**: Easy to extend and customize for different use cases
- **Multiple Output Formats**: Supports various visualization and analysis outputs

## Usage

### Quick Start with ViPE CLI

```bash
# Basic colorization
vipe colorize input_video.mp4

# With visualization and custom output
vipe colorize input_video.mp4 --output my_results/ --visualize --device cuda

# Using different pipeline configurations
vipe colorize input_video.mp4 --pipeline colorization
```

### Standalone Usage

```python
from video_colorization_program import VideoColorizationPipeline

# Initialize pipeline
pipeline = VideoColorizationPipeline(
    output_dir="./colorization_output",
    device="cuda"  # or "cpu"
)

# Process video
colorized_video_path = pipeline.process_video(
    video_path="input_video.mp4",
    output_video_path="colorized_output.mp4"
)

print(f"Colorized video saved to: {colorized_video_path}")
```

### Advanced Configuration

The colorization pipeline can be customized through YAML configuration files:

```yaml
# configs/pipeline/colorization.yaml
colorization:
  depth_model: "LiheYoung/depth-anything-small-hf"
  semantic_model: "openmmlab/upernet-convnext-small"
  colorization_model: "runwayml/stable-diffusion-inpainting"
  
  inference_steps: 20
  guidance_scale: 7.5
  temporal_weight: 0.3
  
  depth_conditioning: true
  semantic_consistency: true
```

## Installation

### Basic Dependencies
```bash
pip install opencv-python numpy omegaconf tqdm
```

### Full Deep Learning Dependencies
```bash
pip install torch torchvision transformers diffusers timm
pip install open3d  # For point cloud processing
```

### ViPE Installation
Follow the main ViPE installation instructions in the repository README.

## Model Dependencies

The colorization pipeline uses several state-of-the-art models:

1. **Depth Estimation**: Depth Anything V2 (`LiheYoung/depth-anything-small-hf`)
2. **Semantic Segmentation**: UperNet with ConvNeXt (`openmmlab/upernet-convnext-small`)
3. **Colorization**: Stable Diffusion Inpainting (`runwayml/stable-diffusion-inpainting`)

These models are automatically downloaded when first used.

## Output Structure

The pipeline generates comprehensive outputs:

```
colorization_results/
├── frames/                 # Extracted grayscale frames
├── depth_maps/            # Metric depth maps from ViPE
├── semantic_masks/        # Semantic segmentation results
├── point_clouds/          # 3D semantic point clouds
├── colorized_frames/      # Individual colorized frames
├── final_colorized_video.mp4  # Final output video
└── visualization/         # Comparison videos and analysis
```

## Technical Details

### Frame Attributes Extended

The implementation extends ViPE's `VideoFrame` class with new attributes:
- `semantic_mask`: Pixel-level semantic labels
- `semantic_segments`: Detailed segmentation results
- `point_cloud`: 3D semantic point cloud data
- `colorized_rgb`: Generated color information

### Processing Pipeline

1. **ViPE SLAM Processing**: Robust 3D structure estimation
2. **Temporal Depth Alignment**: Consistent metric depth maps
3. **Video Semantic Segmentation**: Object-level understanding
4. **Dynamic Point Cloud Generation**: 3D semantic representation
5. **Multi-Modal Colorization**: AI-powered color generation
6. **Temporal Consistency Enforcement**: Smooth color transitions

### Processor Architecture

- `SemanticSegmentationProcessor`: Handles video semantic segmentation
- `PointCloudProcessor`: Generates 3D semantic point clouds
- `ColorizationProcessor`: Performs multi-modal colorization

## Performance Considerations

- **GPU Acceleration**: Recommended for real-time processing
- **Memory Usage**: Scales with video resolution and length
- **Processing Speed**: ~3-5 FPS on modern GPUs (following ViPE's performance)
- **Model Loading**: First-time model downloads may take time

## Limitations and Future Work

### Current Limitations
- Requires significant GPU memory for high-resolution videos
- Processing time scales with video length
- Dependent on pre-trained model quality

### Future Enhancements
- Real-time colorization support
- User-guided colorization with reference images
- Video-specific model fine-tuning
- Enhanced temporal consistency algorithms

## Research Foundation

This implementation is based on extensive research synthesis covering:
- AI-based video colorization techniques
- 3D geometric perception and SLAM
- Semantic video understanding
- Temporal consistency in video processing

Key research areas incorporated:
- Dynamic Point Maps (DPM) for 4D scene understanding
- Multi-modal fusion for enhanced colorization
- Transformer-based architectures for temporal modeling
- Diffusion models for high-quality color generation

## Testing

Run the included test suite to verify the implementation:

```bash
python test_colorization.py
```

This validates:
- Code structure and syntax
- Component integration
- Dependency availability
- Pipeline configuration

## Contributing

When contributing to the colorization functionality:

1. Follow ViPE's coding standards and architecture patterns
2. Add tests for new components
3. Update documentation for new features
4. Consider performance implications for video processing

## License

This colorization extension follows the same Apache 2.0 license as the main ViPE project.

## Citation

If you use this colorization implementation in your research, please cite both the ViPE paper and acknowledge the colorization extension:

```bibtex
@inproceedings{huang2025vipe,
    title={ViPE: Video Pose Engine for 3D Geometric Perception},
    author={Huang, Jiahui and Zhou, Qunjie and Rabeti, Hesam and Korovko, Aleksandr and Ling, Huan and Ren, Xuanchi and Shen, Tianchang and Gao, Jun and Slepichev, Dmitry and Lin, Chen-Hsuan and Ren, Jiawei and Xie, Kevin and Biswas, Joydeep and Leal-Taixe, Laura and Fidler, Sanja},
    booktitle={NVIDIA Research Whitepapers},
    year={2025}
}
```