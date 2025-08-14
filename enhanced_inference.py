#!/usr/bin/env python3
"""
Enhanced Inference and Evaluation Script for Video Colorization

This script provides comprehensive inference capabilities including:
- Batch processing of multiple videos
- Model comparison and evaluation
- Performance benchmarking
- Quality metrics computation
- Pre-trained model assessment

Features:
- Support for multiple pre-trained models
- Automatic model selection based on video characteristics
- Comprehensive quality metrics (SSIM, PSNR, LPIPS, FID)
- Temporal consistency evaluation
- User-guided colorization options
- Real-time processing optimization

Author: ViPE Team
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Quality metrics
try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from torchmetrics.image.fid import FrechetInceptionDistance
    import torchvision.transforms as transforms
except ImportError as e:
    warnings.warn(f"Missing evaluation dependencies: {e}")

# Model imports
try:
    from transformers import pipeline
    from diffusers import StableDiffusionInpaintPipeline
    from video_colorization_program import VideoColorizationPipeline
except ImportError as e:
    warnings.warn(f"Missing model dependencies: {e}")


class ModelEvaluator:
    """
    Comprehensive model evaluation framework for video colorization.
    
    Evaluates models on various metrics including perceptual quality,
    temporal consistency, and computational efficiency.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Initialize quality metrics
        self.ssim = StructuralSimilarityIndexMeasure().to(self.device)
        self.psnr = PeakSignalNoiseRatio().to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(self.device)
        self.fid = FrechetInceptionDistance(feature=2048).to(self.device)
        
        # Temporal consistency metrics
        self.temporal_metrics = TemporalConsistencyMetrics()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def evaluate_single_frame(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Evaluate a single frame pair."""
        # Ensure tensors are in correct format [B, C, H, W] and range [0, 1]
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)
        
        # Normalize to [0, 1] if needed
        if pred.max() > 1.0:
            pred = pred / 255.0
        if target.max() > 1.0:
            target = target / 255.0
        
        pred = pred.to(self.device)
        target = target.to(self.device)
        
        metrics = {}
        
        try:
            metrics['ssim'] = self.ssim(pred, target).item()
            metrics['psnr'] = self.psnr(pred, target).item()
            metrics['lpips'] = self.lpips(pred, target).item()
            
            # Update FID (requires batch processing)
            self.fid.update(target, real=True)
            self.fid.update(pred, real=False)
            
        except Exception as e:
            self.logger.warning(f"Error computing metrics: {e}")
            metrics = {'ssim': 0.0, 'psnr': 0.0, 'lpips': 1.0}
        
        return metrics
    
    def evaluate_video_sequence(self, pred_frames: List[np.ndarray], 
                              target_frames: List[np.ndarray]) -> Dict[str, Any]:
        """Evaluate a complete video sequence."""
        frame_metrics = []
        temporal_metrics = {}
        
        # Evaluate individual frames
        for pred_frame, target_frame in zip(pred_frames, target_frames):
            pred_tensor = torch.from_numpy(pred_frame).permute(2, 0, 1).float()
            target_tensor = torch.from_numpy(target_frame).permute(2, 0, 1).float()
            
            frame_result = self.evaluate_single_frame(pred_tensor, target_tensor)
            frame_metrics.append(frame_result)
        
        # Compute temporal consistency metrics
        temporal_metrics = self.temporal_metrics.evaluate_sequence(pred_frames, target_frames)
        
        # Aggregate frame metrics
        aggregate_metrics = {
            'ssim_mean': np.mean([m['ssim'] for m in frame_metrics]),
            'ssim_std': np.std([m['ssim'] for m in frame_metrics]),
            'psnr_mean': np.mean([m['psnr'] for m in frame_metrics]),
            'psnr_std': np.std([m['psnr'] for m in frame_metrics]),
            'lpips_mean': np.mean([m['lpips'] for m in frame_metrics]),
            'lpips_std': np.std([m['lpips'] for m in frame_metrics]),
        }
        
        # Compute FID for the sequence
        try:
            fid_score = self.fid.compute().item()
            aggregate_metrics['fid'] = fid_score
            self.fid.reset()
        except:
            aggregate_metrics['fid'] = float('inf')
        
        return {
            'frame_metrics': frame_metrics,
            'aggregate_metrics': aggregate_metrics,
            'temporal_metrics': temporal_metrics
        }


class TemporalConsistencyMetrics:
    """Metrics for evaluating temporal consistency in video colorization."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def optical_flow_consistency(self, frames: List[np.ndarray]) -> float:
        """Compute temporal consistency using optical flow."""
        if len(frames) < 2:
            return 1.0
        
        consistencies = []
        
        for i in range(1, len(frames)):
            prev_frame = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
            curr_frame = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            
            # Compute optical flow
            flow = cv2.calcOpticalFlowPyrLK(
                prev_frame, curr_frame, None, None
            )
            
            if flow is not None:
                # Compute warping error
                warped = cv2.remap(frames[i-1], flow[..., 0], flow[..., 1], cv2.INTER_LINEAR)
                error = np.mean(np.abs(warped.astype(float) - frames[i].astype(float)))
                consistency = max(0, 1 - error / 255.0)
                consistencies.append(consistency)
        
        return np.mean(consistencies) if consistencies else 1.0
    
    def frame_difference_consistency(self, frames: List[np.ndarray]) -> float:
        """Compute consistency based on frame differences."""
        if len(frames) < 2:
            return 1.0
        
        differences = []
        
        for i in range(1, len(frames)):
            diff = np.mean(np.abs(frames[i].astype(float) - frames[i-1].astype(float)))
            differences.append(diff)
        
        # Lower differences indicate better temporal consistency
        avg_diff = np.mean(differences)
        return max(0, 1 - avg_diff / 255.0)
    
    def evaluate_sequence(self, pred_frames: List[np.ndarray], 
                         target_frames: List[np.ndarray]) -> Dict[str, float]:
        """Evaluate temporal consistency for a video sequence."""
        metrics = {}
        
        # Optical flow consistency
        pred_flow_consistency = self.optical_flow_consistency(pred_frames)
        target_flow_consistency = self.optical_flow_consistency(target_frames)
        
        metrics['optical_flow_consistency'] = pred_flow_consistency
        metrics['flow_consistency_ratio'] = pred_flow_consistency / max(target_flow_consistency, 1e-6)
        
        # Frame difference consistency
        pred_diff_consistency = self.frame_difference_consistency(pred_frames)
        target_diff_consistency = self.frame_difference_consistency(target_frames)
        
        metrics['frame_diff_consistency'] = pred_diff_consistency
        metrics['diff_consistency_ratio'] = pred_diff_consistency / max(target_diff_consistency, 1e-6)
        
        return metrics


class PretrainedModelAssessor:
    """
    Assessment framework for pre-trained colorization models.
    
    Evaluates different pre-trained models on various criteria:
    - Quality metrics on standard datasets
    - Computational efficiency
    - Robustness to different video types
    - Temporal consistency
    """
    
    AVAILABLE_MODELS = {
        'stable_diffusion_inpaint': {
            'name': 'runwayml/stable-diffusion-inpainting',
            'type': 'diffusion',
            'description': 'Stable Diffusion Inpainting for colorization',
            'memory_usage': 'high',
            'speed': 'slow',
            'quality': 'high'
        },
        'stable_diffusion_xl': {
            'name': 'stabilityai/stable-diffusion-xl-base-1.0',
            'type': 'diffusion',
            'description': 'Stable Diffusion XL for high-resolution colorization',
            'memory_usage': 'very_high',
            'speed': 'very_slow',
            'quality': 'very_high'
        },
        'controlnet_colorization': {
            'name': 'lllyasviel/sd-controlnet-canny',
            'type': 'controlnet',
            'description': 'ControlNet-based colorization with edge guidance',
            'memory_usage': 'high',
            'speed': 'slow',
            'quality': 'high'
        }
    }
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.evaluator = ModelEvaluator(device)
        self.logger = logging.getLogger(__name__)
        
        # Model cache
        self.loaded_models = {}
    
    def load_model(self, model_id: str):
        """Load a pre-trained model."""
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]
        
        if model_id not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_id}")
        
        model_info = self.AVAILABLE_MODELS[model_id]
        self.logger.info(f"Loading model: {model_info['description']}")
        
        try:
            if model_info['type'] == 'diffusion':
                model = StableDiffusionInpaintPipeline.from_pretrained(
                    model_info['name'],
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
            else:
                raise NotImplementedError(f"Model type {model_info['type']} not implemented")
            
            self.loaded_models[model_id] = model
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {e}")
            return None
    
    def assess_model_on_dataset(self, model_id: str, test_dataset_path: str) -> Dict[str, Any]:
        """Assess a model on a test dataset."""
        model = self.load_model(model_id)
        if model is None:
            return {'error': f'Failed to load model {model_id}'}
        
        test_path = Path(test_dataset_path)
        gray_dir = test_path / "grayscale"
        color_dir = test_path / "color"
        
        if not gray_dir.exists() or not color_dir.exists():
            return {'error': 'Invalid test dataset structure'}
        
        results = {
            'model_id': model_id,
            'model_info': self.AVAILABLE_MODELS[model_id],
            'test_dataset': str(test_dataset_path),
            'frame_results': [],
            'performance_metrics': {}
        }
        
        gray_files = sorted(gray_dir.glob("*.png"))[:50]  # Limit to 50 frames for assessment
        
        total_inference_time = 0
        total_memory_used = 0
        
        for gray_file in tqdm(gray_files, desc=f"Assessing {model_id}"):
            color_file = color_dir / gray_file.name
            if not color_file.exists():
                continue
            
            # Load images
            gray_img = cv2.imread(str(gray_file), cv2.IMREAD_GRAYSCALE)
            color_img = cv2.imread(str(color_file), cv2.IMREAD_COLOR)
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            
            # Convert grayscale to RGB
            gray_rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
            
            # Measure inference time and memory
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            
            start_time = time.time()
            
            # Perform colorization
            pred_img = self.colorize_with_model(model, gray_rgb, model_id)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            inference_time = time.time() - start_time
            
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
            else:
                memory_used = 0
            
            total_inference_time += inference_time
            total_memory_used = max(total_memory_used, memory_used)
            
            # Evaluate quality
            pred_tensor = torch.from_numpy(pred_img).permute(2, 0, 1).float()
            target_tensor = torch.from_numpy(color_img).permute(2, 0, 1).float()
            
            frame_metrics = self.evaluator.evaluate_single_frame(pred_tensor, target_tensor)
            frame_metrics['inference_time'] = inference_time
            frame_metrics['memory_used'] = memory_used
            
            results['frame_results'].append(frame_metrics)
        
        # Aggregate performance metrics
        if results['frame_results']:
            metrics = results['frame_results']
            results['performance_metrics'] = {
                'avg_ssim': np.mean([m['ssim'] for m in metrics]),
                'avg_psnr': np.mean([m['psnr'] for m in metrics]),
                'avg_lpips': np.mean([m['lpips'] for m in metrics]),
                'avg_inference_time': np.mean([m['inference_time'] for m in metrics]),
                'total_inference_time': total_inference_time,
                'max_memory_used_mb': total_memory_used,
                'frames_per_second': len(metrics) / total_inference_time
            }
        
        return results
    
    def colorize_with_model(self, model, gray_image: np.ndarray, model_id: str) -> np.ndarray:
        """Colorize an image using a specific model."""
        try:
            if model_id.startswith('stable_diffusion'):
                from PIL import Image
                
                # Convert to PIL
                gray_pil = Image.fromarray(gray_image)
                mask = Image.fromarray(np.ones(gray_image.shape[:2], dtype=np.uint8) * 255)
                
                # Generate colorized image
                result = model(
                    prompt="a realistic colorized photograph",
                    image=gray_pil,
                    mask_image=mask,
                    num_inference_steps=20,
                    guidance_scale=7.5
                ).images[0]
                
                return np.array(result)
            
            else:
                # Fallback: return grayscale as RGB
                return gray_image
                
        except Exception as e:
            self.logger.warning(f"Colorization failed: {e}")
            return gray_image
    
    def compare_models(self, model_ids: List[str], test_dataset_path: str) -> Dict[str, Any]:
        """Compare multiple models on the same test dataset."""
        comparison_results = {
            'test_dataset': test_dataset_path,
            'models': {},
            'comparison_metrics': {}
        }
        
        # Assess each model
        for model_id in model_ids:
            self.logger.info(f"Assessing model: {model_id}")
            model_results = self.assess_model_on_dataset(model_id, test_dataset_path)
            comparison_results['models'][model_id] = model_results
        
        # Generate comparison metrics
        comparison_results['comparison_metrics'] = self._generate_comparison_metrics(
            comparison_results['models']
        )
        
        return comparison_results
    
    def _generate_comparison_metrics(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparative metrics across models."""
        metrics = {}
        
        for metric in ['avg_ssim', 'avg_psnr', 'avg_lpips', 'avg_inference_time', 'max_memory_used_mb']:
            metric_values = {}
            for model_id, results in model_results.items():
                if 'performance_metrics' in results and metric in results['performance_metrics']:
                    metric_values[model_id] = results['performance_metrics'][metric]
            
            if metric_values:
                best_model = max(metric_values.keys(), key=lambda k: metric_values[k]) if 'time' not in metric and 'memory' not in metric and metric != 'avg_lpips' else min(metric_values.keys(), key=lambda k: metric_values[k])
                
                metrics[metric] = {
                    'values': metric_values,
                    'best_model': best_model,
                    'best_value': metric_values[best_model]
                }
        
        return metrics
    
    def recommend_model(self, video_characteristics: Dict[str, Any]) -> str:
        """Recommend the best model based on video characteristics."""
        # Simple heuristic-based recommendation
        resolution = video_characteristics.get('resolution', 'medium')
        duration = video_characteristics.get('duration', 'medium')
        complexity = video_characteristics.get('complexity', 'medium')
        
        # High quality, low speed requirements
        if resolution == 'high' and duration == 'short':
            return 'stable_diffusion_xl'
        
        # Balanced quality and speed
        elif resolution == 'medium' or duration == 'medium':
            return 'stable_diffusion_inpaint'
        
        # Fast processing requirements
        else:
            return 'stable_diffusion_inpaint'  # Fallback to most reliable


class EnhancedColorizationInference:
    """
    Enhanced inference engine with model selection and optimization.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model_assessor = PretrainedModelAssessor(device)
        self.evaluator = ModelEvaluator(device)
        self.logger = logging.getLogger(__name__)
        
        # Performance optimization settings
        self.optimization_settings = {
            'use_mixed_precision': True,
            'use_model_compilation': True,
            'batch_size': 1,
            'memory_efficient': True
        }
    
    def process_video_with_evaluation(self, video_path: str, output_dir: str, 
                                   model_id: Optional[str] = None,
                                   ground_truth_path: Optional[str] = None) -> Dict[str, Any]:
        """Process a video with comprehensive evaluation."""
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Analyze video characteristics
        video_chars = self.analyze_video_characteristics(video_path)
        
        # Select model if not specified
        if model_id is None:
            model_id = self.model_assessor.recommend_model(video_chars)
            self.logger.info(f"Auto-selected model: {model_id}")
        
        # Load model
        model = self.model_assessor.load_model(model_id)
        if model is None:
            raise ValueError(f"Failed to load model: {model_id}")
        
        # Process video
        self.logger.info(f"Processing video: {video_path}")
        
        # Use the standalone colorization pipeline
        pipeline = VideoColorizationPipeline(
            output_dir=str(output_dir),
            device=self.device
        )
        
        # Override the model in the pipeline
        pipeline.colorization_model = model
        
        # Process the video
        start_time = time.time()
        colorized_video_path = pipeline.process_video(str(video_path))
        processing_time = time.time() - start_time
        
        results = {
            'input_video': str(video_path),
            'output_video': colorized_video_path,
            'model_used': model_id,
            'video_characteristics': video_chars,
            'processing_time': processing_time,
            'performance_metrics': {}
        }
        
        # Evaluate against ground truth if available
        if ground_truth_path:
            evaluation_results = self.evaluate_against_ground_truth(
                colorized_video_path, ground_truth_path
            )
            results['evaluation'] = evaluation_results
        
        # Save results
        results_path = output_dir / 'inference_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def analyze_video_characteristics(self, video_path: str) -> Dict[str, Any]:
        """Analyze video characteristics for model selection."""
        cap = cv2.VideoCapture(str(video_path))
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        # Categorize characteristics
        resolution = 'high' if width * height > 1920 * 1080 else 'medium' if width * height > 1280 * 720 else 'low'
        duration = 'long' if frame_count / fps > 60 else 'medium' if frame_count / fps > 10 else 'short'
        
        return {
            'width': width,
            'height': height,
            'frame_count': frame_count,
            'fps': fps,
            'duration_seconds': frame_count / fps,
            'resolution': resolution,
            'duration': duration,
            'file_size_mb': Path(video_path).stat().st_size / (1024 * 1024)
        }
    
    def evaluate_against_ground_truth(self, pred_video_path: str, gt_video_path: str) -> Dict[str, Any]:
        """Evaluate colorized video against ground truth."""
        # Load frames from both videos
        pred_frames = self.load_video_frames(pred_video_path)
        gt_frames = self.load_video_frames(gt_video_path)
        
        # Ensure same number of frames
        min_frames = min(len(pred_frames), len(gt_frames))
        pred_frames = pred_frames[:min_frames]
        gt_frames = gt_frames[:min_frames]
        
        # Evaluate
        return self.evaluator.evaluate_video_sequence(pred_frames, gt_frames)
    
    def load_video_frames(self, video_path: str) -> List[np.ndarray]:
        """Load all frames from a video."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()
        return frames
    
    def batch_process_videos(self, video_list: List[str], output_base_dir: str,
                           model_id: Optional[str] = None) -> Dict[str, Any]:
        """Process multiple videos in batch."""
        output_base_dir = Path(output_base_dir)
        batch_results = {
            'total_videos': len(video_list),
            'successful': 0,
            'failed': 0,
            'results': {}
        }
        
        for i, video_path in enumerate(tqdm(video_list, desc="Processing videos")):
            try:
                video_name = Path(video_path).stem
                output_dir = output_base_dir / f"video_{i:03d}_{video_name}"
                
                result = self.process_video_with_evaluation(
                    video_path, str(output_dir), model_id
                )
                
                batch_results['results'][video_name] = result
                batch_results['successful'] += 1
                
            except Exception as e:
                self.logger.error(f"Failed to process {video_path}: {e}")
                batch_results['results'][Path(video_path).stem] = {'error': str(e)}
                batch_results['failed'] += 1
        
        # Save batch results
        batch_results_path = output_base_dir / 'batch_results.json'
        with open(batch_results_path, 'w') as f:
            json.dump(batch_results, f, indent=2, default=str)
        
        return batch_results


def main():
    """Main function for enhanced inference and evaluation."""
    parser = argparse.ArgumentParser(description="Enhanced Video Colorization Inference and Evaluation")
    
    # Basic inference options
    parser.add_argument('--input', type=str, help='Input video path or directory')
    parser.add_argument('--output', type=str, default='inference_results', help='Output directory')
    parser.add_argument('--model', type=str, help='Model ID to use for colorization')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    
    # Evaluation options
    parser.add_argument('--ground-truth', type=str, help='Ground truth video for evaluation')
    parser.add_argument('--evaluate-models', nargs='+', help='List of models to evaluate and compare')
    parser.add_argument('--test-dataset', type=str, help='Test dataset path for model evaluation')
    
    # Batch processing
    parser.add_argument('--batch', action='store_true', help='Batch process multiple videos')
    parser.add_argument('--video-list', type=str, help='File containing list of video paths')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference_engine = EnhancedColorizationInference(device=args.device)
    
    if args.evaluate_models and args.test_dataset:
        # Model comparison mode
        print("Evaluating and comparing models...")
        results = inference_engine.model_assessor.compare_models(
            args.evaluate_models, args.test_dataset
        )
        
        # Save comparison results
        output_path = Path(args.output) / 'model_comparison.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Model comparison results saved to: {output_path}")
        
        # Print summary
        print("\nModel Comparison Summary:")
        for metric, data in results['comparison_metrics'].items():
            print(f"{metric}: Best model is {data['best_model']} with value {data['best_value']:.4f}")
    
    elif args.batch and args.video_list:
        # Batch processing mode
        with open(args.video_list, 'r') as f:
            video_paths = [line.strip() for line in f if line.strip()]
        
        print(f"Processing {len(video_paths)} videos in batch...")
        results = inference_engine.batch_process_videos(
            video_paths, args.output, args.model
        )
        
        print(f"Batch processing completed: {results['successful']} successful, {results['failed']} failed")
    
    elif args.input:
        # Single video processing mode
        if Path(args.input).is_dir():
            # Process all videos in directory
            video_files = list(Path(args.input).glob("*.mp4")) + list(Path(args.input).glob("*.avi"))
            results = inference_engine.batch_process_videos(
                [str(f) for f in video_files], args.output, args.model
            )
        else:
            # Process single video
            results = inference_engine.process_video_with_evaluation(
                args.input, args.output, args.model, args.ground_truth
            )
        
        print(f"Processing completed. Results saved to: {args.output}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()