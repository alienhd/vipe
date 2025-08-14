#!/usr/bin/env python3
"""
Quick Start Guide for Video Colorization

This script provides a step-by-step guide for getting started with
the video colorization functionality.
"""

import os
import sys
from pathlib import Path

def print_banner():
    """Print a nice banner."""
    print("🎨" + "="*60 + "🎨")
    print("  ViPE Video Colorization - Quick Start Guide")
    print("🎨" + "="*60 + "🎨")
    print()

def check_installation():
    """Check if required components are installed."""
    print("📋 Checking Installation...")
    print()
    
    # Check basic dependencies
    basic_deps = {
        'cv2': 'OpenCV (opencv-python)',
        'numpy': 'NumPy',
        'omegaconf': 'OmegaConf'
    }
    
    print("Basic Dependencies:")
    for module, name in basic_deps.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - Install with: pip install {module.replace('cv2', 'opencv-python')}")
    
    print()
    
    # Check AI/ML dependencies
    ml_deps = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'diffusers': 'Diffusers',
        'open3d': 'Open3D'
    }
    
    print("AI/ML Dependencies (required for full functionality):")
    missing_ml = []
    for module, name in ml_deps.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name}")
            missing_ml.append(module)
    
    if missing_ml:
        print(f"\n  To install missing ML dependencies:")
        print(f"  pip install {' '.join(missing_ml)}")
    
    print()

def show_usage_examples():
    """Show usage examples."""
    print("🚀 Usage Examples")
    print("="*50)
    print()
    
    print("1. Standalone Python Script:")
    print("-" * 30)
    print("""
from video_colorization_program import VideoColorizationPipeline

# Initialize pipeline
pipeline = VideoColorizationPipeline(
    output_dir="./colorization_output",
    device="cuda"  # or "cpu"
)

# Process video
colorized_video = pipeline.process_video(
    video_path="my_bw_video.mp4",
    output_video_path="colorized_video.mp4"
)

print(f"Colorized video saved: {colorized_video}")
""")
    
    print("2. ViPE CLI (after full ViPE installation):")
    print("-" * 45)
    print("""
# Basic colorization
vipe colorize my_bw_video.mp4

# With custom output and visualization
vipe colorize my_bw_video.mp4 \\
    --output ./my_results/ \\
    --visualize \\
    --device cuda

# Using custom configuration
vipe colorize my_bw_video.mp4 \\
    --pipeline colorization
""")
    
    print("3. Advanced Configuration:")
    print("-" * 30)
    print("""
Create a custom config file: configs/pipeline/my_colorization.yaml

colorization:
  # Models
  depth_model: "LiheYoung/depth-anything-small-hf"
  semantic_model: "openmmlab/upernet-convnext-small"
  colorization_model: "runwayml/stable-diffusion-inpainting"
  
  # Parameters
  device: "cuda"
  inference_steps: 25    # Higher = better quality, slower
  guidance_scale: 8.0    # Conditioning strength
  temporal_weight: 0.4   # Temporal consistency
  
  # Features
  depth_conditioning: true
  semantic_consistency: true
  save_point_clouds: true

output:
  save_artifacts: true
  save_viz: true
  save_intermediate: true
""")

def show_output_structure():
    """Show what outputs are generated."""
    print("📂 Output Structure")
    print("="*50)
    print("""
colorization_results/
├── frames/                      # Extracted grayscale frames
│   ├── frame_000000.png
│   ├── frame_000001.png
│   └── ...
├── depth_maps/                  # Metric depth maps from ViPE
│   ├── depth_000000.npy
│   ├── depth_000001.npy
│   └── ...
├── semantic_masks/              # Semantic segmentation results
│   ├── semantic_000000.npy
│   ├── semantic_000001.npy
│   └── ...
├── point_clouds/                # 3D semantic point clouds
│   ├── pointcloud_000000.ply
│   ├── pointcloud_000001.ply
│   └── ...
├── colorized_frames/            # Individual colorized frames
│   ├── colorized_000000.png
│   ├── colorized_000001.png
│   └── ...
├── final_colorized_video.mp4    # Final output video
├── video_metadata.pkl           # Video metadata
├── camera_poses.npy             # Camera poses from ViPE
└── visualization/               # Comparison videos
    ├── before_after_comparison.mp4
    └── depth_visualization.mp4
""")

def show_installation_steps():
    """Show complete installation steps."""
    print("⚙️  Complete Installation Guide")
    print("="*50)
    print()
    
    print("1. Install ViPE (follow main README):")
    print("-" * 35)
    print("""
# Create conda environment
conda env create -f envs/base.yml
conda activate vipe

# Install dependencies
pip install -r envs/requirements.txt

# Build and install ViPE
pip install --no-build-isolation -e .
""")
    
    print("2. Verify Colorization Installation:")
    print("-" * 35)
    print("""
# Run the test suite
python test_colorization.py

# Try the examples
python example_colorization.py
""")
    
    print("3. First Run:")
    print("-" * 15)
    print("""
# Test with the CLI
vipe colorize --help

# Or use standalone
python -c "from video_colorization_program import VideoColorizationPipeline; print('Ready!')"
""")

def show_performance_tips():
    """Show performance optimization tips."""
    print("⚡ Performance Tips")
    print("="*50)
    print("""
🖥️  Hardware Recommendations:
   • GPU: NVIDIA RTX 3080 or better (8GB+ VRAM)
   • RAM: 16GB+ system memory
   • Storage: SSD for faster I/O

⚙️  Configuration Tips:
   • Use lower inference_steps (10-15) for faster processing
   • Reduce guidance_scale (5-6) if quality is acceptable
   • Set batch_size=1 for memory efficiency
   • Use CPU for small/test videos

📺 Video Recommendations:
   • Resolution: 720p-1080p for best balance
   • Length: Start with short clips (10-30 seconds)
   • Format: MP4 with standard codecs
   • Frame rate: 24-30 FPS

🔧 Troubleshooting:
   • Out of memory: Reduce video resolution or use CPU
   • Slow processing: Check GPU utilization
   • Poor quality: Increase inference_steps or guidance_scale
   • Flickering: Increase temporal_weight (0.4-0.6)
""")

def main():
    """Main function."""
    print_banner()
    
    sections = [
        check_installation,
        show_installation_steps,
        show_usage_examples,
        show_output_structure,
        show_performance_tips
    ]
    
    for section in sections:
        section()
        print()
    
    print("🎯 Quick Commands to Get Started:")
    print("="*50)
    print("# Test the implementation")
    print("python test_colorization.py")
    print()
    print("# See examples")
    print("python example_colorization.py") 
    print()
    print("# After full installation:")
    print("vipe colorize your_video.mp4")
    print()
    print("📚 For detailed documentation, see: COLORIZATION_README.md")
    print()
    print("Happy colorizing! 🌈")

if __name__ == "__main__":
    main()