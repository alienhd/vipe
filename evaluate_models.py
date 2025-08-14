#!/usr/bin/env python3
"""
Model Evaluation and Benchmarking Script for Video Colorization

This script provides comprehensive evaluation capabilities for colorization models:
- Performance benchmarking on standard datasets
- Quality metrics computation and analysis
- Model comparison and ranking
- Temporal consistency evaluation
- Computational efficiency analysis

Features:
- Support for multiple evaluation datasets
- Comprehensive quality metrics (SSIM, PSNR, LPIPS, FID)
- Temporal consistency analysis
- Performance profiling
- Statistical analysis and visualization
- Report generation

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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm

# Statistical analysis
from scipy import stats
import statsmodels.api as sm

# Import our evaluation components
try:
    from enhanced_inference import PretrainedModelAssessor, ModelEvaluator, TemporalConsistencyMetrics
    from video_colorization_program import VideoColorizationPipeline
except ImportError as e:
    warnings.warn(f"Missing colorization components: {e}")


class BenchmarkDataset:
    """
    Standard benchmark dataset for video colorization evaluation.
    
    Supports various dataset formats and provides standardized evaluation protocols.
    """
    
    STANDARD_DATASETS = {
        'davis_2017': {
            'name': 'DAVIS 2017',
            'description': 'Video object segmentation dataset adapted for colorization',
            'url': 'https://davischallenge.org/',
            'frames': 4219,
            'videos': 90,
            'resolution': '480p'
        },
        'youtube_vos': {
            'name': 'YouTube-VOS',
            'description': 'Large-scale video object segmentation dataset',
            'url': 'https://youtube-vos.org/',
            'frames': 197000,
            'videos': 4453,
            'resolution': 'variable'
        },
        'sintel': {
            'name': 'MPI Sintel',
            'description': 'Synthetic movie dataset with ground truth optical flow',
            'url': 'http://sintel.is.tue.mpg.de/',
            'frames': 1041,
            'videos': 23,
            'resolution': '1024x436'
        },
        'custom': {
            'name': 'Custom Dataset',
            'description': 'User-provided custom dataset',
            'url': 'N/A',
            'frames': 'variable',
            'videos': 'variable',
            'resolution': 'variable'
        }
    }
    
    def __init__(self, dataset_name: str, dataset_path: str):
        self.dataset_name = dataset_name
        self.dataset_path = Path(dataset_path)
        self.dataset_info = self.STANDARD_DATASETS.get(dataset_name, self.STANDARD_DATASETS['custom'])
        
        # Validate dataset structure
        self.validate_dataset()
        
        # Load dataset metadata
        self.video_list = self.load_video_list()
        
    def validate_dataset(self):
        """Validate dataset structure and files."""
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {self.dataset_path}")
        
        # Check for required subdirectories
        required_dirs = ['grayscale', 'color']
        for dir_name in required_dirs:
            dir_path = self.dataset_path / dir_name
            if not dir_path.exists():
                raise ValueError(f"Missing required directory: {dir_path}")
    
    def load_video_list(self) -> List[Dict[str, Any]]:
        """Load list of videos with metadata."""
        video_list = []
        
        # Find video sequences
        gray_dir = self.dataset_path / 'grayscale'
        color_dir = self.dataset_path / 'color'
        
        # Group frames by video sequence
        video_sequences = {}
        
        for gray_file in gray_dir.glob('*.png'):
            # Extract video sequence identifier (assuming format: video_id_frame_id.png)
            parts = gray_file.stem.split('_')
            if len(parts) >= 2:
                video_id = '_'.join(parts[:-1])
                frame_id = parts[-1]
            else:
                video_id = 'default'
                frame_id = gray_file.stem
            
            color_file = color_dir / gray_file.name
            if color_file.exists():
                if video_id not in video_sequences:
                    video_sequences[video_id] = []
                
                video_sequences[video_id].append({
                    'frame_id': frame_id,
                    'gray_path': str(gray_file),
                    'color_path': str(color_file)
                })
        
        # Sort frames within each video
        for video_id, frames in video_sequences.items():
            frames.sort(key=lambda x: x['frame_id'])
            
            video_list.append({
                'video_id': video_id,
                'frame_count': len(frames),
                'frames': frames
            })
        
        return video_list


class ComprehensiveEvaluator:
    """
    Comprehensive evaluation framework for video colorization models.
    
    Provides detailed analysis including:
    - Quality metrics
    - Temporal consistency
    - Perceptual quality
    - Statistical significance testing
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model_evaluator = ModelEvaluator(device)
        self.temporal_evaluator = TemporalConsistencyMetrics()
        self.model_assessor = PretrainedModelAssessor(device)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.evaluation_results = {}
    
    def evaluate_model_on_dataset(self, model_id: str, dataset: BenchmarkDataset, 
                                max_videos: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate a model on a complete dataset."""
        self.logger.info(f"Evaluating {model_id} on {dataset.dataset_name}")
        
        # Load model
        model = self.model_assessor.load_model(model_id)
        if model is None:
            return {'error': f'Failed to load model {model_id}'}
        
        video_results = []
        overall_metrics = {
            'frame_metrics': [],
            'temporal_metrics': [],
            'computational_metrics': []
        }
        
        videos_to_process = dataset.video_list[:max_videos] if max_videos else dataset.video_list
        
        for video_info in tqdm(videos_to_process, desc=f"Evaluating {model_id}"):
            video_result = self.evaluate_model_on_video(model, model_id, video_info)
            video_results.append(video_result)
            
            # Aggregate metrics
            if 'frame_metrics' in video_result:
                overall_metrics['frame_metrics'].extend(video_result['frame_metrics'])
            if 'temporal_metrics' in video_result:
                overall_metrics['temporal_metrics'].append(video_result['temporal_metrics'])
            if 'computational_metrics' in video_result:
                overall_metrics['computational_metrics'].append(video_result['computational_metrics'])
        
        # Compute aggregate statistics
        aggregate_stats = self.compute_aggregate_statistics(overall_metrics)
        
        results = {
            'model_id': model_id,
            'dataset': dataset.dataset_name,
            'dataset_info': dataset.dataset_info,
            'videos_evaluated': len(video_results),
            'video_results': video_results,
            'aggregate_metrics': overall_metrics,
            'aggregate_statistics': aggregate_stats
        }
        
        return results
    
    def evaluate_model_on_video(self, model, model_id: str, video_info: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a model on a single video sequence."""
        frames = video_info['frames']
        
        pred_frames = []
        gt_frames = []
        inference_times = []
        memory_usage = []
        
        # Process each frame
        for frame_info in frames:
            # Load images
            gray_img = cv2.imread(frame_info['gray_path'], cv2.IMREAD_GRAYSCALE)
            color_img = cv2.imread(frame_info['color_path'], cv2.IMREAD_COLOR)
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            gray_rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
            
            # Measure inference time and memory
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            
            start_time = time.time()
            pred_img = self.model_assessor.colorize_with_model(model, gray_rgb, model_id)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            inference_time = time.time() - start_time
            
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
            else:
                memory_used = 0
            
            pred_frames.append(pred_img)
            gt_frames.append(color_img)
            inference_times.append(inference_time)
            memory_usage.append(memory_used)
        
        # Evaluate frame-level metrics
        frame_metrics = []
        for pred, gt in zip(pred_frames, gt_frames):
            pred_tensor = torch.from_numpy(pred).permute(2, 0, 1).float()
            gt_tensor = torch.from_numpy(gt).permute(2, 0, 1).float()
            
            metrics = self.model_evaluator.evaluate_single_frame(pred_tensor, gt_tensor)
            frame_metrics.append(metrics)
        
        # Evaluate temporal consistency
        temporal_metrics = self.temporal_evaluator.evaluate_sequence(pred_frames, gt_frames)
        
        # Computational metrics
        computational_metrics = {
            'avg_inference_time': np.mean(inference_times),
            'total_inference_time': sum(inference_times),
            'max_memory_usage': max(memory_usage) if memory_usage else 0,
            'fps': len(frames) / sum(inference_times) if sum(inference_times) > 0 else 0
        }
        
        return {
            'video_id': video_info['video_id'],
            'frame_count': len(frames),
            'frame_metrics': frame_metrics,
            'temporal_metrics': temporal_metrics,
            'computational_metrics': computational_metrics
        }
    
    def compute_aggregate_statistics(self, metrics: Dict[str, List]) -> Dict[str, Any]:
        """Compute aggregate statistics across all frames and videos."""
        stats = {}
        
        # Frame-level statistics
        if metrics['frame_metrics']:
            frame_data = pd.DataFrame(metrics['frame_metrics'])
            
            for metric in ['ssim', 'psnr', 'lpips']:
                if metric in frame_data.columns:
                    values = frame_data[metric].values
                    stats[f'{metric}_mean'] = float(np.mean(values))
                    stats[f'{metric}_std'] = float(np.std(values))
                    stats[f'{metric}_median'] = float(np.median(values))
                    stats[f'{metric}_min'] = float(np.min(values))
                    stats[f'{metric}_max'] = float(np.max(values))
                    stats[f'{metric}_p25'] = float(np.percentile(values, 25))
                    stats[f'{metric}_p75'] = float(np.percentile(values, 75))
        
        # Temporal consistency statistics
        if metrics['temporal_metrics']:
            temporal_data = pd.DataFrame(metrics['temporal_metrics'])
            
            for metric in temporal_data.columns:
                values = temporal_data[metric].values
                stats[f'temporal_{metric}_mean'] = float(np.mean(values))
                stats[f'temporal_{metric}_std'] = float(np.std(values))
        
        # Computational statistics
        if metrics['computational_metrics']:
            comp_data = pd.DataFrame(metrics['computational_metrics'])
            
            for metric in comp_data.columns:
                values = comp_data[metric].values
                stats[f'comp_{metric}_mean'] = float(np.mean(values))
                stats[f'comp_{metric}_std'] = float(np.std(values))
        
        return stats
    
    def compare_models(self, model_ids: List[str], dataset: BenchmarkDataset,
                      max_videos: Optional[int] = None) -> Dict[str, Any]:
        """Compare multiple models on the same dataset."""
        comparison_results = {
            'dataset': dataset.dataset_name,
            'models_compared': model_ids,
            'individual_results': {},
            'statistical_comparison': {},
            'rankings': {}
        }
        
        # Evaluate each model
        for model_id in model_ids:
            self.logger.info(f"Evaluating model: {model_id}")
            model_results = self.evaluate_model_on_dataset(model_id, dataset, max_videos)
            comparison_results['individual_results'][model_id] = model_results
        
        # Perform statistical comparisons
        comparison_results['statistical_comparison'] = self.perform_statistical_comparison(
            comparison_results['individual_results']
        )
        
        # Generate rankings
        comparison_results['rankings'] = self.generate_model_rankings(
            comparison_results['individual_results']
        )
        
        return comparison_results
    
    def perform_statistical_comparison(self, model_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Perform statistical significance testing between models."""
        statistical_tests = {}
        
        # Extract metric values for each model
        model_metrics = {}
        for model_id, results in model_results.items():
            if 'aggregate_metrics' in results and 'frame_metrics' in results['aggregate_metrics']:
                frame_data = pd.DataFrame(results['aggregate_metrics']['frame_metrics'])
                model_metrics[model_id] = frame_data
        
        # Perform pairwise t-tests for each metric
        for metric in ['ssim', 'psnr', 'lpips']:
            statistical_tests[metric] = {}
            
            models = list(model_metrics.keys())
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    model1, model2 = models[i], models[j]
                    
                    if (metric in model_metrics[model1].columns and 
                        metric in model_metrics[model2].columns):
                        
                        values1 = model_metrics[model1][metric].values
                        values2 = model_metrics[model2][metric].values
                        
                        # Perform t-test
                        t_stat, p_value = stats.ttest_ind(values1, values2)
                        
                        # Effect size (Cohen's d)
                        pooled_std = np.sqrt(((len(values1) - 1) * np.var(values1) + 
                                            (len(values2) - 1) * np.var(values2)) / 
                                           (len(values1) + len(values2) - 2))
                        cohens_d = (np.mean(values1) - np.mean(values2)) / pooled_std
                        
                        statistical_tests[metric][f'{model1}_vs_{model2}'] = {
                            't_statistic': float(t_stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05,
                            'cohens_d': float(cohens_d),
                            'effect_size': 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'
                        }
        
        return statistical_tests
    
    def generate_model_rankings(self, model_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate model rankings based on various metrics."""
        rankings = {}
        
        # Extract aggregate statistics for ranking
        model_stats = {}
        for model_id, results in model_results.items():
            if 'aggregate_statistics' in results:
                model_stats[model_id] = results['aggregate_statistics']
        
        # Rank by individual metrics
        metrics_to_rank = ['ssim_mean', 'psnr_mean', 'lpips_mean', 'comp_fps_mean']
        
        for metric in metrics_to_rank:
            if all(metric in stats for stats in model_stats.values()):
                # Sort models by metric (higher is better for ssim/psnr/fps, lower is better for lpips)
                reverse_sort = metric != 'lpips_mean'
                
                sorted_models = sorted(
                    model_stats.keys(),
                    key=lambda m: model_stats[m][metric],
                    reverse=reverse_sort
                )
                
                rankings[metric] = [
                    {
                        'rank': i + 1,
                        'model': model,
                        'value': model_stats[model][metric]
                    }
                    for i, model in enumerate(sorted_models)
                ]
        
        # Compute overall ranking (weighted combination)
        if all(metric in rankings for metric in ['ssim_mean', 'psnr_mean', 'lpips_mean']):
            overall_scores = {}
            
            for model in model_stats.keys():
                # Normalize scores and compute weighted average
                ssim_rank = next(r['rank'] for r in rankings['ssim_mean'] if r['model'] == model)
                psnr_rank = next(r['rank'] for r in rankings['psnr_mean'] if r['model'] == model)
                lpips_rank = next(r['rank'] for r in rankings['lpips_mean'] if r['model'] == model)
                
                # Lower rank is better, so invert and weight
                overall_score = (
                    (len(model_stats) - ssim_rank + 1) * 0.4 +  # 40% weight
                    (len(model_stats) - psnr_rank + 1) * 0.3 +  # 30% weight
                    (len(model_stats) - lpips_rank + 1) * 0.3   # 30% weight
                )
                
                overall_scores[model] = overall_score
            
            sorted_overall = sorted(overall_scores.keys(), key=lambda m: overall_scores[m], reverse=True)
            
            rankings['overall'] = [
                {
                    'rank': i + 1,
                    'model': model,
                    'score': overall_scores[model]
                }
                for i, model in enumerate(sorted_overall)
            ]
        
        return rankings


class BenchmarkReportGenerator:
    """Generate comprehensive benchmark reports with visualizations."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generate_comparison_report(self, comparison_results: Dict[str, Any]) -> str:
        """Generate a comprehensive comparison report."""
        report_path = self.output_dir / 'model_comparison_report.html'
        
        # Create visualizations
        self.create_performance_plots(comparison_results)
        self.create_statistical_plots(comparison_results)
        
        # Generate HTML report
        html_content = self.create_html_report(comparison_results)
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return str(report_path)
    
    def create_performance_plots(self, comparison_results: Dict[str, Any]):
        """Create performance comparison plots."""
        models = list(comparison_results['individual_results'].keys())
        
        # Extract performance data
        performance_data = {}
        for model in models:
            results = comparison_results['individual_results'][model]
            if 'aggregate_statistics' in results:
                stats = results['aggregate_statistics']
                performance_data[model] = stats
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # SSIM comparison
        if all('ssim_mean' in data for data in performance_data.values()):
            ssim_values = [performance_data[m]['ssim_mean'] for m in models]
            ssim_stds = [performance_data[m].get('ssim_std', 0) for m in models]
            
            axes[0, 0].bar(models, ssim_values, yerr=ssim_stds, capsize=5)
            axes[0, 0].set_title('SSIM (Structural Similarity)')
            axes[0, 0].set_ylabel('SSIM Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # PSNR comparison
        if all('psnr_mean' in data for data in performance_data.values()):
            psnr_values = [performance_data[m]['psnr_mean'] for m in models]
            psnr_stds = [performance_data[m].get('psnr_std', 0) for m in models]
            
            axes[0, 1].bar(models, psnr_values, yerr=psnr_stds, capsize=5)
            axes[0, 1].set_title('PSNR (Peak Signal-to-Noise Ratio)')
            axes[0, 1].set_ylabel('PSNR (dB)')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # LPIPS comparison
        if all('lpips_mean' in data for data in performance_data.values()):
            lpips_values = [performance_data[m]['lpips_mean'] for m in models]
            lpips_stds = [performance_data[m].get('lpips_std', 0) for m in models]
            
            axes[1, 0].bar(models, lpips_values, yerr=lpips_stds, capsize=5)
            axes[1, 0].set_title('LPIPS (Perceptual Distance)')
            axes[1, 0].set_ylabel('LPIPS Score')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # FPS comparison
        if all('comp_fps_mean' in data for data in performance_data.values()):
            fps_values = [performance_data[m]['comp_fps_mean'] for m in models]
            
            axes[1, 1].bar(models, fps_values)
            axes[1, 1].set_title('Processing Speed')
            axes[1, 1].set_ylabel('FPS')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_statistical_plots(self, comparison_results: Dict[str, Any]):
        """Create statistical significance plots."""
        if 'statistical_comparison' not in comparison_results:
            return
        
        statistical_data = comparison_results['statistical_comparison']
        
        # Create significance heatmap for each metric
        for metric in statistical_data.keys():
            comparisons = statistical_data[metric]
            
            # Extract p-values for heatmap
            models = list(comparison_results['individual_results'].keys())
            n_models = len(models)
            
            p_value_matrix = np.ones((n_models, n_models))
            
            for comparison, data in comparisons.items():
                model1, model2 = comparison.split('_vs_')
                i, j = models.index(model1), models.index(model2)
                p_value = data['p_value']
                
                p_value_matrix[i, j] = p_value
                p_value_matrix[j, i] = p_value
            
            # Create heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                -np.log10(p_value_matrix + 1e-10),  # Use -log10(p) for better visualization
                annot=True,
                fmt='.2f',
                xticklabels=models,
                yticklabels=models,
                cmap='viridis',
                cbar_kws={'label': '-log10(p-value)'}
            )
            plt.title(f'Statistical Significance: {metric.upper()}')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'significance_{metric}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_html_report(self, comparison_results: Dict[str, Any]) -> str:
        """Create HTML report content."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Video Colorization Model Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metrics-table {{ border-collapse: collapse; width: 100%; }}
                .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                .metrics-table th {{ background-color: #f2f2f2; }}
                .ranking {{ background-color: #e8f4fd; padding: 15px; border-radius: 5px; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Video Colorization Model Comparison Report</h1>
                <p><strong>Dataset:</strong> {comparison_results['dataset']}</p>
                <p><strong>Models Compared:</strong> {', '.join(comparison_results['models_compared'])}</p>
                <p><strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        # Performance summary table
        html += self.create_performance_table(comparison_results)
        
        # Rankings section
        if 'rankings' in comparison_results:
            html += self.create_rankings_section(comparison_results['rankings'])
        
        # Visualizations
        html += """
            <div class="section">
                <h2>Performance Visualizations</h2>
                <img src="performance_comparison.png" alt="Performance Comparison">
            </div>
        """
        
        # Statistical analysis
        if 'statistical_comparison' in comparison_results:
            html += self.create_statistical_section(comparison_results['statistical_comparison'])
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def create_performance_table(self, comparison_results: Dict[str, Any]) -> str:
        """Create performance summary table."""
        models = comparison_results['models_compared']
        
        html = """
            <div class="section">
                <h2>Performance Summary</h2>
                <table class="metrics-table">
                    <tr>
                        <th>Model</th>
                        <th>SSIM ↑</th>
                        <th>PSNR ↑</th>
                        <th>LPIPS ↓</th>
                        <th>FPS ↑</th>
                    </tr>
        """
        
        for model in models:
            results = comparison_results['individual_results'][model]
            if 'aggregate_statistics' in results:
                stats = results['aggregate_statistics']
                
                ssim = stats.get('ssim_mean', 'N/A')
                psnr = stats.get('psnr_mean', 'N/A')
                lpips = stats.get('lpips_mean', 'N/A')
                fps = stats.get('comp_fps_mean', 'N/A')
                
                html += f"""
                    <tr>
                        <td><strong>{model}</strong></td>
                        <td>{ssim:.4f if isinstance(ssim, float) else ssim}</td>
                        <td>{psnr:.2f if isinstance(psnr, float) else psnr}</td>
                        <td>{lpips:.4f if isinstance(lpips, float) else lpips}</td>
                        <td>{fps:.2f if isinstance(fps, float) else fps}</td>
                    </tr>
                """
        
        html += """
                </table>
            </div>
        """
        
        return html
    
    def create_rankings_section(self, rankings: Dict[str, Any]) -> str:
        """Create rankings section."""
        html = """
            <div class="section">
                <h2>Model Rankings</h2>
        """
        
        if 'overall' in rankings:
            html += """
                <div class="ranking">
                    <h3>Overall Ranking</h3>
                    <ol>
            """
            
            for rank_info in rankings['overall']:
                html += f"<li><strong>{rank_info['model']}</strong> (Score: {rank_info['score']:.2f})</li>"
            
            html += """
                    </ol>
                </div>
            """
        
        # Individual metric rankings
        for metric, ranking_list in rankings.items():
            if metric != 'overall':
                html += f"""
                    <div class="ranking">
                        <h4>{metric.replace('_', ' ').title()}</h4>
                        <ol>
                """
                
                for rank_info in ranking_list:
                    html += f"<li>{rank_info['model']} ({rank_info['value']:.4f})</li>"
                
                html += """
                        </ol>
                    </div>
                """
        
        html += "</div>"
        return html
    
    def create_statistical_section(self, statistical_data: Dict[str, Any]) -> str:
        """Create statistical analysis section."""
        html = """
            <div class="section">
                <h2>Statistical Analysis</h2>
                <p>Statistical significance testing between model pairs (p < 0.05 indicates significant difference):</p>
        """
        
        for metric, comparisons in statistical_data.items():
            html += f"<h3>{metric.upper()}</h3>"
            html += f'<img src="significance_{metric}.png" alt="Statistical Significance for {metric}">'
            
            html += """
                <table class="metrics-table">
                    <tr>
                        <th>Comparison</th>
                        <th>p-value</th>
                        <th>Significant</th>
                        <th>Effect Size</th>
                    </tr>
            """
            
            for comparison, data in comparisons.items():
                significant = "Yes" if data['significant'] else "No"
                html += f"""
                    <tr>
                        <td>{comparison.replace('_vs_', ' vs ')}</td>
                        <td>{data['p_value']:.4f}</td>
                        <td>{significant}</td>
                        <td>{data['effect_size']}</td>
                    </tr>
                """
            
            html += "</table><br>"
        
        html += "</div>"
        return html


def main():
    """Main function for model evaluation and benchmarking."""
    parser = argparse.ArgumentParser(description="Comprehensive Video Colorization Model Evaluation")
    
    parser.add_argument('--dataset-name', type=str, required=True, 
                       choices=list(BenchmarkDataset.STANDARD_DATASETS.keys()),
                       help='Name of the benchmark dataset')
    parser.add_argument('--dataset-path', type=str, required=True,
                       help='Path to the dataset directory')
    parser.add_argument('--models', nargs='+', required=True,
                       help='List of model IDs to evaluate and compare')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory for results and reports')
    parser.add_argument('--max-videos', type=int,
                       help='Maximum number of videos to evaluate (for quick testing)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # Initialize components
    dataset = BenchmarkDataset(args.dataset_name, args.dataset_path)
    evaluator = ComprehensiveEvaluator(device=args.device)
    report_generator = BenchmarkReportGenerator(args.output_dir)
    
    print(f"Loaded dataset: {dataset.dataset_name}")
    print(f"Found {len(dataset.video_list)} video sequences")
    print(f"Evaluating models: {args.models}")
    
    # Perform model comparison
    comparison_results = evaluator.compare_models(args.models, dataset, args.max_videos)
    
    # Save detailed results
    results_path = Path(args.output_dir) / 'detailed_results.json'
    with open(results_path, 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    
    # Generate report
    report_path = report_generator.generate_comparison_report(comparison_results)
    
    print(f"\nEvaluation completed!")
    print(f"Detailed results saved to: {results_path}")
    print(f"HTML report generated: {report_path}")
    
    # Print summary
    if 'rankings' in comparison_results and 'overall' in comparison_results['rankings']:
        print("\nOverall Rankings:")
        for rank_info in comparison_results['rankings']['overall']:
            print(f"{rank_info['rank']}. {rank_info['model']} (Score: {rank_info['score']:.2f})")


if __name__ == "__main__":
    main()