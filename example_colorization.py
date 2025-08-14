#!/usr/bin/env python3
"""
Example usage of the video colorization pipeline.

This script demonstrates how to use the video colorization functionality
both as a standalone program and integrated with ViPE.
"""

import sys
import logging
from pathlib import Path

def example_standalone_usage():
    """Example of using the standalone colorization program."""
    print("=== Standalone Video Colorization Example ===")
    
    try:
        # Import the standalone program
        from video_colorization_program import VideoColorizationPipeline
        
        print("✓ Standalone colorization program imported successfully")
        
        # Initialize the pipeline
        pipeline = VideoColorizationPipeline(
            output_dir="./example_colorization_output",
            device="cpu"  # Use CPU for demonstration
        )
        
        print("✓ Pipeline initialized")
        
        # Example of processing a video (commented out since we don't have a real video)
        """
        # Process a video file
        colorized_video = pipeline.process_video(
            video_path="input_bw_video.mp4",
            output_video_path="colorized_output.mp4"
        )
        
        print(f"✓ Video colorized successfully: {colorized_video}")
        """
        
        print("✓ Standalone example completed (no actual video processed)")
        return True
        
    except ImportError as e:
        print(f"✗ Could not import colorization program: {e}")
        print("  This is expected if deep learning dependencies are missing.")
        return False
    except Exception as e:
        print(f"✗ Error in standalone example: {e}")
        return False

def example_vipe_integration():
    """Example of using the ViPE-integrated colorization."""
    print("\n=== ViPE-Integrated Colorization Example ===")
    
    try:
        # Test if ViPE integration is available
        from vipe.pipeline.colorization import VideoColorizationPipeline as ViPEColorization
        
        print("✓ ViPE-integrated colorization imported successfully")
        
        # Example configuration (this would normally be loaded from YAML)
        from omegaconf import DictConfig
        
        example_config = {
            'init': {
                'camera_type': 'pinhole',
                'intrinsics': 'geocalib',
                'instance': {
                    'kf_gap_sec': 2.0,
                    'phrases': ['person', 'car'],
                    'add_sky': True
                }
            },
            'slam': {
                'keyframe_depth': 'unidepth-l',
                'optimize_intrinsics': True
            },
            'post': {
                'depth_align_model': 'adaptive_unidepth-l_svda'
            },
            'colorization': {
                'depth_model': 'LiheYoung/depth-anything-small-hf',
                'semantic_model': 'openmmlab/upernet-convnext-small',
                'colorization_model': 'runwayml/stable-diffusion-inpainting',
                'device': 'cpu',
                'inference_steps': 20,
                'guidance_scale': 7.5
            },
            'output': {
                'path': './vipe_colorization_output',
                'save_artifacts': True,
                'save_viz': True
            }
        }
        
        config = DictConfig(example_config)
        
        # Initialize the ViPE colorization pipeline
        vipe_pipeline = ViPEColorization(
            init=config.init,
            slam=config.slam,
            post=config.post,
            colorization=config.colorization,
            output=config.output
        )
        
        print("✓ ViPE colorization pipeline initialized")
        
        # Example of processing (commented out since we don't have ViPE fully set up)
        """
        # This would be the actual usage with ViPE
        from vipe.streams.raw_mp4_stream import RawMp4Stream
        from vipe.streams.base import ProcessedVideoStream
        
        video_stream = ProcessedVideoStream(RawMp4Stream("input.mp4"), [])
        result = vipe_pipeline.run(video_stream)
        
        print("✓ ViPE colorization completed")
        """
        
        print("✓ ViPE integration example completed (no actual processing)")
        return True
        
    except ImportError as e:
        print(f"✗ Could not import ViPE components: {e}")
        print("  This is expected if PyTorch/ViPE dependencies are missing.")
        return False
    except Exception as e:
        print(f"✗ Error in ViPE integration example: {e}")
        return False

def example_cli_usage():
    """Example of CLI usage."""
    print("\n=== CLI Usage Examples ===")
    
    print("Once ViPE is fully installed, you can use these CLI commands:")
    print()
    print("# Basic colorization:")
    print("vipe colorize input_video.mp4")
    print()
    print("# With custom output directory:")
    print("vipe colorize input_video.mp4 --output my_colorization_results/")
    print()
    print("# With visualization enabled:")
    print("vipe colorize input_video.mp4 --visualize")
    print()
    print("# Using CPU instead of GPU:")
    print("vipe colorize input_video.mp4 --device cpu")
    print()
    print("# Using a different pipeline configuration:")
    print("vipe colorize input_video.mp4 --pipeline colorization")
    print()
    
    return True

def example_configuration():
    """Example of custom configuration."""
    print("\n=== Configuration Example ===")
    
    print("You can customize the colorization pipeline using YAML configuration:")
    print()
    
    config_example = """
# configs/pipeline/my_colorization.yaml
defaults:
  - /slam: default

instance: vipe.pipeline.colorization.VideoColorizationPipeline

colorization:
  # Model configurations
  depth_model: "LiheYoung/depth-anything-small-hf"
  semantic_model: "openmmlab/upernet-convnext-small" 
  colorization_model: "runwayml/stable-diffusion-inpainting"
  
  # Processing parameters
  device: "cuda"
  inference_steps: 30  # Higher quality, slower
  guidance_scale: 8.0  # Stronger conditioning
  temporal_weight: 0.5  # More temporal consistency
  
  # Feature toggles
  depth_conditioning: true
  semantic_consistency: true
  save_point_clouds: true

output:
  path: custom_colorization_results/
  save_artifacts: true
  save_intermediate: true
  save_viz: true
"""
    
    print(config_example)
    print("Then use it with:")
    print("vipe colorize input_video.mp4 --pipeline my_colorization")
    print()
    
    return True

def main():
    """Run all examples."""
    print("🎨 Video Colorization Examples\n")
    
    examples = [
        example_standalone_usage,
        example_vipe_integration,
        example_cli_usage,
        example_configuration
    ]
    
    results = []
    for example in examples:
        try:
            result = example()
            results.append(result)
        except Exception as e:
            print(f"✗ Example failed: {e}")
            results.append(False)
    
    print("\n" + "="*50)
    if any(results):
        print("🎉 Examples completed! Some functionality demonstrated.")
        print("\nNote: Full functionality requires installing deep learning dependencies:")
        print("  pip install torch transformers diffusers open3d")
        return 0
    else:
        print("❌ All examples failed. Please check your installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())