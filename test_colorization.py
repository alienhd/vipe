#!/usr/bin/env python3
"""
Test script for the video colorization implementation.

This script validates the colorization pipeline structure and components
without requiring the full deep learning dependencies.
"""

import sys
import ast
from pathlib import Path

def test_colorization_program():
    """Test the main colorization program structure."""
    print("Testing video_colorization_program.py...")
    
    try:
        # Test syntax
        with open("video_colorization_program.py", "r") as f:
            source = f.read()
        
        ast.parse(source)
        print("✓ Syntax validation passed")
        
        # Test that key classes and methods are defined
        tree = ast.parse(source)
        
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        required_classes = ["VideoColorizationPipeline"]
        required_methods = ["extract_frames", "estimate_depth_and_poses", "segment_frames", 
                          "generate_semantic_point_clouds", "colorize_frames", "reconstruct_video", "process_video"]
        
        for cls in required_classes:
            if cls in classes:
                print(f"✓ Class {cls} found")
            else:
                print(f"✗ Class {cls} missing")
                return False
        
        for method in required_methods:
            if method in functions:
                print(f"✓ Method {method} found")
            else:
                print(f"✗ Method {method} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing colorization program: {e}")
        return False

def test_vipe_integration():
    """Test the ViPE integration components."""
    print("\nTesting ViPE integration...")
    
    try:
        # Test colorization pipeline
        colorization_path = Path("vipe/pipeline/colorization.py")
        if colorization_path.exists():
            with open(colorization_path, "r") as f:
                source = f.read()
            
            ast.parse(source)
            print("✓ Colorization pipeline syntax valid")
            
            tree = ast.parse(source)
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            
            if "VideoColorizationPipeline" in classes:
                print("✓ ViPE-integrated VideoColorizationPipeline found")
            else:
                print("✗ ViPE-integrated VideoColorizationPipeline missing")
                return False
        else:
            print("✗ Colorization pipeline file missing")
            return False
        
        # Test processors
        processors_path = Path("vipe/pipeline/processors.py")
        if processors_path.exists():
            with open(processors_path, "r") as f:
                source = f.read()
            
            tree = ast.parse(source)
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            
            required_processors = ["SemanticSegmentationProcessor", "PointCloudProcessor", "ColorizationProcessor"]
            for processor in required_processors:
                if processor in classes:
                    print(f"✓ Processor {processor} found")
                else:
                    print(f"✗ Processor {processor} missing")
                    return False
        
        # Test configuration
        config_path = Path("configs/pipeline/colorization.yaml")
        if config_path.exists():
            print("✓ Colorization configuration file found")
        else:
            print("✗ Colorization configuration file missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing ViPE integration: {e}")
        return False

def test_dependencies():
    """Test availability of key dependencies."""
    print("\nTesting dependencies...")
    
    # Test basic dependencies
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__} available")
    except ImportError:
        print("✗ OpenCV not available")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__} available")
    except ImportError:
        print("✗ NumPy not available")
        return False
    
    try:
        from omegaconf import OmegaConf
        print("✓ OmegaConf available")
    except ImportError:
        print("✗ OmegaConf not available")
        return False
    
    # Test optional deep learning dependencies
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} available")
    except ImportError:
        print("⚠ PyTorch not available (required for full functionality)")
    
    try:
        import transformers
        print("✓ Transformers available")
    except ImportError:
        print("⚠ Transformers not available (required for semantic segmentation)")
    
    try:
        import diffusers
        print("✓ Diffusers available")
    except ImportError:
        print("⚠ Diffusers not available (required for colorization)")
    
    try:
        import open3d
        print("✓ Open3D available")
    except ImportError:
        print("⚠ Open3D not available (required for point cloud processing)")
    
    return True

def main():
    """Run all tests."""
    print("=== Video Colorization Implementation Test ===\n")
    
    tests = [
        test_colorization_program,
        test_vipe_integration,
        test_dependencies
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    if all(results):
        print("🎉 All core tests passed!")
        print("\nNote: Some optional dependencies may be missing, but core structure is valid.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())