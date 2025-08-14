#!/usr/bin/env python3
"""
Example scripts demonstrating various use cases of the video colorization system.

This file contains practical examples for:
- Basic video colorization
- Batch processing workflows
- Model comparison and evaluation
- Custom training pipelines
- Integration with ViPE

Author: ViPE Team
"""

import os
import sys
from pathlib import Path
import logging

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import colorization components
try:
    from video_colorization_program import VideoColorizationPipeline
    from enhanced_inference import EnhancedColorizationInference, PretrainedModelAssessor
    from train_colorization import ColorizationTrainer
    from evaluate_models import ComprehensiveEvaluator, BenchmarkDataset
except ImportError as e:
    print(f"Error importing colorization components: {e}")
    print("Please ensure all dependencies are installed and ViPE is properly set up.")
    sys.exit(1)


def example_basic_colorization():
    """Example 1: Basic video colorization."""
    print("=== Example 1: Basic Video Colorization ===")
    
    # Initialize the standalone colorization pipeline
    pipeline = VideoColorizationPipeline(
        output_dir="./examples/basic_colorization",
        device="cuda"  # Change to "cpu" if no GPU available
    )
    
    # Example video path (replace with actual video)
    input_video = "assets/examples/sample_bw_video.mp4"
    
    if not Path(input_video).exists():
        print(f"Sample video not found: {input_video}")
        print("Please provide a black and white video file.")
        return
    
    try:
        # Process the video
        print(f"Processing video: {input_video}")
        colorized_video = pipeline.process_video(
            video_path=input_video,
            output_video_path="./examples/basic_colorization/colorized_video.mp4"
        )
        
        print(f"✓ Colorization completed!")
        print(f"  Input: {input_video}")
        print(f"  Output: {colorized_video}")
        
    except Exception as e:
        print(f"✗ Colorization failed: {e}")


def example_model_comparison():
    """Example 2: Compare different colorization models."""
    print("\n=== Example 2: Model Comparison ===")
    
    # Initialize the inference engine
    inference_engine = EnhancedColorizationInference(device="cuda")
    
    # Models to compare
    models_to_test = [
        'stable_diffusion_inpaint',
        'stable_diffusion_xl'
    ]
    
    # Test video
    test_video = "assets/examples/sample_bw_video.mp4"
    
    if not Path(test_video).exists():
        print(f"Test video not found: {test_video}")
        return
    
    print(f"Comparing models: {models_to_test}")
    
    results = {}
    for model_id in models_to_test:
        print(f"\nTesting model: {model_id}")
        
        try:
            output_dir = f"./examples/model_comparison/{model_id}"
            result = inference_engine.process_video_with_evaluation(
                video_path=test_video,
                output_dir=output_dir,
                model_id=model_id
            )
            
            results[model_id] = result
            print(f"✓ {model_id} completed in {result['processing_time']:.2f}s")
            
        except Exception as e:
            print(f"✗ {model_id} failed: {e}")
    
    # Print comparison summary
    print("\n--- Model Comparison Summary ---")
    for model_id, result in results.items():
        if 'processing_time' in result:
            print(f"{model_id}:")
            print(f"  Processing time: {result['processing_time']:.2f}s")
            print(f"  Output: {result['output_video']}")


def example_batch_processing():
    """Example 3: Batch processing multiple videos."""
    print("\n=== Example 3: Batch Processing ===")
    
    # Create example video list
    video_list = [
        "assets/examples/video1.mp4",
        "assets/examples/video2.mp4", 
        "assets/examples/video3.mp4"
    ]
    
    # Filter to existing videos
    existing_videos = [v for v in video_list if Path(v).exists()]
    
    if not existing_videos:
        print("No test videos found. Please add videos to assets/examples/")
        print("Creating mock video list...")
        # Create a simple example with one video repeated
        if Path("assets/examples/sample_bw_video.mp4").exists():
            existing_videos = ["assets/examples/sample_bw_video.mp4"]
    
    if not existing_videos:
        print("No videos available for batch processing example.")
        return
    
    print(f"Processing {len(existing_videos)} videos in batch...")
    
    # Initialize inference engine
    inference_engine = EnhancedColorizationInference(device="cuda")
    
    try:
        # Batch process
        batch_results = inference_engine.batch_process_videos(
            video_list=existing_videos,
            output_base_dir="./examples/batch_processing",
            model_id="stable_diffusion_inpaint"
        )
        
        print(f"✓ Batch processing completed!")
        print(f"  Successful: {batch_results['successful']}")
        print(f"  Failed: {batch_results['failed']}")
        print(f"  Results saved to: ./examples/batch_processing/")
        
    except Exception as e:
        print(f"✗ Batch processing failed: {e}")


def example_model_evaluation():
    """Example 4: Comprehensive model evaluation."""
    print("\n=== Example 4: Model Evaluation ===")
    
    # Set up test dataset structure
    test_dataset_path = "./examples/test_dataset"
    
    # Check if test dataset exists
    if not (Path(test_dataset_path) / "grayscale").exists():
        print(f"Test dataset not found at: {test_dataset_path}")
        print("Creating minimal test dataset structure...")
        
        # Create directory structure
        (Path(test_dataset_path) / "grayscale").mkdir(parents=True, exist_ok=True)
        (Path(test_dataset_path) / "color").mkdir(parents=True, exist_ok=True)
        
        print("Please add test images to:")
        print(f"  Grayscale: {test_dataset_path}/grayscale/")
        print(f"  Color: {test_dataset_path}/color/")
        return
    
    try:
        # Initialize evaluation components
        dataset = BenchmarkDataset("custom", test_dataset_path)
        evaluator = ComprehensiveEvaluator(device="cuda")
        
        # Models to evaluate
        models_to_evaluate = ['stable_diffusion_inpaint']
        
        print(f"Evaluating models on {len(dataset.video_list)} video sequences...")
        
        # Perform evaluation
        comparison_results = evaluator.compare_models(
            model_ids=models_to_evaluate,
            dataset=dataset,
            max_videos=5  # Limit for example
        )
        
        print("✓ Evaluation completed!")
        print("Results:")
        
        for model_id, results in comparison_results['individual_results'].items():
            if 'aggregate_statistics' in results:
                stats = results['aggregate_statistics']
                print(f"  {model_id}:")
                print(f"    SSIM: {stats.get('ssim_mean', 'N/A'):.4f}")
                print(f"    PSNR: {stats.get('psnr_mean', 'N/A'):.2f}")
                
    except Exception as e:
        print(f"✗ Model evaluation failed: {e}")


def example_custom_training():
    """Example 5: Custom model training setup."""
    print("\n=== Example 5: Custom Training Setup ===")
    
    # Training dataset path
    training_data_path = "./examples/training_data"
    
    # Check if training data exists
    if not (Path(training_data_path) / "train" / "grayscale").exists():
        print(f"Training dataset not found at: {training_data_path}")
        print("Creating training dataset structure...")
        
        # Create directory structure
        for split in ["train", "val"]:
            for data_type in ["grayscale", "color"]:
                (Path(training_data_path) / split / data_type).mkdir(parents=True, exist_ok=True)
        
        print("Please add training data to:")
        print(f"  Training grayscale: {training_data_path}/train/grayscale/")
        print(f"  Training color: {training_data_path}/train/color/")
        print(f"  Validation grayscale: {training_data_path}/val/grayscale/")
        print(f"  Validation color: {training_data_path}/val/color/")
        return
    
    print("Training dataset structure found.")
    print("To start training, run:")
    print("  python train_colorization.py --config configs/training/stable_diffusion_finetune.yaml")
    print("  # Or create a custom config file and modify the data_dir path")


def example_vipe_integration():
    """Example 6: Integration with ViPE pipeline."""
    print("\n=== Example 6: ViPE Integration ===")
    
    # This example shows how to use colorization within ViPE
    print("ViPE Integration example:")
    print("1. Install ViPE following the main README instructions")
    print("2. Use the colorization pipeline:")
    print("   vipe colorize input_video.mp4 --output colorization_results/")
    print("3. Or use the Python API:")
    
    example_code = """
    from vipe.pipeline.colorization import VideoColorizationPipeline
    from omegaconf import DictConfig
    
    # Configuration
    config = DictConfig({
        'init': {'camera_type': 'pinhole', 'intrinsics': 'geocalib'},
        'slam': {'keyframe_depth': 'unidepth-l'},
        'post': {'depth_align_model': 'adaptive_unidepth-l_svda'},
        'colorization': {
            'depth_model': 'LiheYoung/depth-anything-small-hf',
            'colorization_model': 'runwayml/stable-diffusion-inpainting',
            'device': 'cuda'
        },
        'output': {'path': 'vipe_colorization_results/'}
    })
    
    # Initialize ViPE colorization pipeline
    pipeline = VideoColorizationPipeline(**config)
    
    # Process video with full ViPE capabilities
    result = pipeline.run(video_stream)
    """
    
    print(example_code)


def main():
    """Run all examples."""
    print("🎨 Video Colorization Examples")
    print("=" * 50)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create examples directory
    Path("./examples").mkdir(exist_ok=True)
    
    # Run examples
    examples = [
        example_basic_colorization,
        example_model_comparison,
        example_batch_processing,
        example_model_evaluation,
        example_custom_training,
        example_vipe_integration
    ]
    
    for example_func in examples:
        try:
            example_func()
        except KeyboardInterrupt:
            print("\nExamples interrupted by user.")
            break
        except Exception as e:
            print(f"Example failed: {e}")
            continue
    
    print("\n" + "=" * 50)
    print("🎉 Examples completed!")
    print("\nNext steps:")
    print("1. Review the generated outputs in ./examples/")
    print("2. Modify the examples for your specific use case")
    print("3. Check the documentation for advanced features")
    print("4. Report any issues on the GitHub repository")


if __name__ == "__main__":
    main()