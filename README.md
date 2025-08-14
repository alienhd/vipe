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

For detailed colorization documentation, see [COLORIZATION_README.md](COLORIZATION_README.md).


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
