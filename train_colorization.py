#!/usr/bin/env python3
"""
Comprehensive Training Script for Video Colorization Models

This script provides a complete framework for training custom colorization models,
fine-tuning pre-trained models, and evaluating colorization performance.

Features:
- Support for multiple model architectures
- Temporal consistency training
- Semantic-guided colorization  
- Depth-conditioned training
- Model evaluation and benchmarking
- Distributed training support

Author: ViPE Team
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import cv2
from tqdm import tqdm
import yaml

# Deep learning imports
try:
    from transformers import AutoModel, AutoProcessor
    from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
    from accelerate import Accelerator
    import torchvision.transforms as transforms
    from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
except ImportError as e:
    print(f"Missing training dependencies: {e}")
    print("Please install: pip install torch transformers diffusers accelerate torchmetrics")
    exit(1)

# ViPE imports for depth and semantic processing
try:
    from vipe.utils import io
    from vipe.pipeline.processors import SemanticSegmentationProcessor, PointCloudProcessor
    from video_colorization_program import VideoColorizationPipeline
except ImportError as e:
    print(f"ViPE components not available: {e}")


class VideoColorizationDataset(Dataset):
    """
    Dataset class for training video colorization models.
    
    Supports loading video frames with ground truth colors, depth maps,
    and semantic segmentation for multi-modal training.
    """
    
    def __init__(self, data_dir: str, transform=None, include_depth=True, include_semantic=True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.include_depth = include_depth
        self.include_semantic = include_semantic
        
        # Load frame pairs (grayscale input, color target)
        self.frame_pairs = self._load_frame_pairs()
        
    def _load_frame_pairs(self) -> List[Tuple[str, str]]:
        """Load pairs of grayscale and color frames for training."""
        pairs = []
        
        # Look for structured data directory:
        # data_dir/
        #   grayscale/
        #   color/
        #   depth/ (optional)
        #   semantic/ (optional)
        
        gray_dir = self.data_dir / "grayscale"
        color_dir = self.data_dir / "color"
        
        if not gray_dir.exists() or not color_dir.exists():
            raise ValueError(f"Missing grayscale or color directories in {self.data_dir}")
        
        gray_files = sorted(gray_dir.glob("*.png"))
        for gray_file in gray_files:
            color_file = color_dir / gray_file.name
            if color_file.exists():
                pairs.append((str(gray_file), str(color_file)))
        
        return pairs
    
    def __len__(self):
        return len(self.frame_pairs)
    
    def __getitem__(self, idx):
        gray_path, color_path = self.frame_pairs[idx]
        
        # Load images
        gray_img = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
        color_img = cv2.imread(color_path, cv2.IMREAD_COLOR)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        
        # Convert grayscale to 3-channel
        gray_rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
        
        sample = {
            'input': gray_rgb,
            'target': color_img,
            'frame_id': Path(gray_path).stem
        }
        
        # Load depth if available
        if self.include_depth:
            depth_path = self.data_dir / "depth" / f"{Path(gray_path).stem}.npy"
            if depth_path.exists():
                depth = np.load(depth_path)
                sample['depth'] = depth
        
        # Load semantic segmentation if available
        if self.include_semantic:
            semantic_path = self.data_dir / "semantic" / f"{Path(gray_path).stem}.npy"
            if semantic_path.exists():
                semantic = np.load(semantic_path)
                sample['semantic'] = semantic
        
        # Apply transforms
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class ColorizationLoss(nn.Module):
    """
    Multi-component loss function for video colorization training.
    
    Combines:
    - Perceptual loss (VGG features)
    - L1/L2 reconstruction loss
    - Temporal consistency loss
    - Semantic consistency loss
    """
    
    def __init__(self, perceptual_weight=1.0, temporal_weight=0.5, semantic_weight=0.3):
        super().__init__()
        self.perceptual_weight = perceptual_weight
        self.temporal_weight = temporal_weight
        self.semantic_weight = semantic_weight
        
        # Load pre-trained VGG for perceptual loss
        from torchvision.models import vgg19
        self.vgg = vgg19(pretrained=True).features[:16]
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
    
    def perceptual_loss(self, pred, target):
        """Compute perceptual loss using VGG features."""
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        return self.mse_loss(pred_features, target_features)
    
    def temporal_consistency_loss(self, pred_current, pred_previous, target_current, target_previous):
        """Compute temporal consistency loss between consecutive frames."""
        if pred_previous is None or target_previous is None:
            return torch.tensor(0.0, device=pred_current.device)
        
        pred_diff = pred_current - pred_previous
        target_diff = target_current - target_previous
        return self.mse_loss(pred_diff, target_diff)
    
    def semantic_consistency_loss(self, pred, target, semantic_mask):
        """Compute semantic consistency loss within object regions."""
        if semantic_mask is None:
            return torch.tensor(0.0, device=pred.device)
        
        # Compute loss per semantic region
        unique_labels = torch.unique(semantic_mask)
        semantic_loss = 0.0
        
        for label in unique_labels:
            if label == 0:  # Skip background
                continue
            
            mask = (semantic_mask == label).float()
            if mask.sum() > 0:
                pred_region = pred * mask.unsqueeze(1)
                target_region = target * mask.unsqueeze(1)
                semantic_loss += self.mse_loss(pred_region, target_region)
        
        return semantic_loss / len(unique_labels) if len(unique_labels) > 1 else torch.tensor(0.0, device=pred.device)
    
    def forward(self, pred, target, pred_prev=None, target_prev=None, semantic_mask=None):
        """Compute total loss."""
        # Reconstruction loss
        recon_loss = self.l1_loss(pred, target)
        
        # Perceptual loss
        perc_loss = self.perceptual_loss(pred, target) * self.perceptual_weight
        
        # Temporal consistency loss
        temp_loss = self.temporal_consistency_loss(pred, pred_prev, target, target_prev) * self.temporal_weight
        
        # Semantic consistency loss
        sem_loss = self.semantic_consistency_loss(pred, target, semantic_mask) * self.semantic_weight
        
        total_loss = recon_loss + perc_loss + temp_loss + sem_loss
        
        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'perceptual': perc_loss,
            'temporal': temp_loss,
            'semantic': sem_loss
        }


class ColorizationTrainer:
    """
    Main trainer class for video colorization models.
    
    Supports various model architectures and training strategies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        
        # Setup logging
        self.setup_logging()
        
        # Initialize model, optimizer, and loss
        self.model = self.build_model()
        self.optimizer = self.build_optimizer()
        self.criterion = ColorizationLoss(**config.get('loss', {}))
        self.scheduler = self.build_scheduler()
        
        # Metrics
        self.ssim = StructuralSimilarityIndexMeasure().to(self.device)
        self.psnr = PeakSignalNoiseRatio().to(self.device)
        
        # Prepare for distributed training
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = 0.0
        
        # Setup tensorboard
        if self.accelerator.is_main_process:
            self.writer = SummaryWriter(config['output_dir'] / 'tensorboard')
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config['output_dir'] / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def build_model(self):
        """Build the colorization model."""
        model_type = self.config['model']['type']
        
        if model_type == 'stable_diffusion':
            # Fine-tune Stable Diffusion for colorization
            model = StableDiffusionInpaintPipeline.from_pretrained(
                self.config['model']['name'],
                torch_dtype=torch.float32,  # Use float32 for training
            )
            return model.unet  # Return just the UNet for training
            
        elif model_type == 'custom_unet':
            # Custom U-Net architecture
            from .models import CustomColorizationUNet
            return CustomColorizationUNet(**self.config['model']['params'])
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def build_optimizer(self):
        """Build optimizer."""
        opt_config = self.config['optimizer']
        
        if opt_config['type'] == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 0)
            )
        elif opt_config['type'] == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 0.01)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_config['type']}")
    
    def build_scheduler(self):
        """Build learning rate scheduler."""
        sched_config = self.config.get('scheduler', {})
        
        if sched_config.get('type') == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs']
            )
        elif sched_config.get('type') == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get('step_size', 10),
                gamma=sched_config.get('gamma', 0.1)
            )
        else:
            return None
    
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_ssim = 0.0
        total_psnr = 0.0
        
        prev_batch = None
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Forward pass
            with self.accelerator.autocast():
                pred = self.model(batch['input'])
                
                # Prepare temporal data
                pred_prev = prev_batch['pred'] if prev_batch is not None else None
                target_prev = prev_batch['target'] if prev_batch is not None else None
                
                # Compute loss
                loss_dict = self.criterion(
                    pred, batch['target'],
                    pred_prev, target_prev,
                    batch.get('semantic')
                )
                
                loss = loss_dict['total']
            
            # Backward pass
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Update metrics
            with torch.no_grad():
                ssim_score = self.ssim(pred, batch['target'])
                psnr_score = self.psnr(pred, batch['target'])
                
                total_loss += loss.item()
                total_ssim += ssim_score.item()
                total_psnr += psnr_score.item()
            
            # Log to tensorboard
            if self.accelerator.is_main_process and batch_idx % 10 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/SSIM', ssim_score.item(), self.global_step)
                self.writer.add_scalar('Train/PSNR', psnr_score.item(), self.global_step)
                
                for loss_name, loss_value in loss_dict.items():
                    if loss_name != 'total':
                        self.writer.add_scalar(f'Train/Loss_{loss_name}', loss_value.item(), self.global_step)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ssim': f"{ssim_score.item():.4f}",
                'psnr': f"{psnr_score.item():.2f}"
            })
            
            # Store for temporal consistency
            prev_batch = {
                'pred': pred.detach(),
                'target': batch['target']
            }
            
            self.global_step += 1
        
        # Update learning rate
        if self.scheduler:
            self.scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        avg_ssim = total_ssim / len(dataloader)
        avg_psnr = total_psnr / len(dataloader)
        
        return {
            'loss': avg_loss,
            'ssim': avg_ssim,
            'psnr': avg_psnr
        }
    
    def validate(self, dataloader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_ssim = 0.0
        total_psnr = 0.0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                pred = self.model(batch['input'])
                loss_dict = self.criterion(pred, batch['target'])
                
                ssim_score = self.ssim(pred, batch['target'])
                psnr_score = self.psnr(pred, batch['target'])
                
                total_loss += loss_dict['total'].item()
                total_ssim += ssim_score.item()
                total_psnr += psnr_score.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_ssim = total_ssim / len(dataloader)
        avg_psnr = total_psnr / len(dataloader)
        
        return {
            'loss': avg_loss,
            'ssim': avg_ssim,
            'psnr': avg_psnr
        }
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.config['output_dir'] / f'checkpoint_epoch_{self.epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.config['output_dir'] / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with SSIM: {metrics['ssim']:.4f}")
    
    def train(self, train_dataloader, val_dataloader):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        for epoch in range(self.config['training']['epochs']):
            self.epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_dataloader)
            
            # Validation
            val_metrics = self.validate(val_dataloader)
            
            # Log metrics
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Epoch {epoch}: "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val SSIM: {val_metrics['ssim']:.4f}, "
                    f"Val PSNR: {val_metrics['psnr']:.2f}"
                )
                
                # Tensorboard logging
                self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
                self.writer.add_scalar('Val/SSIM', val_metrics['ssim'], epoch)
                self.writer.add_scalar('Val/PSNR', val_metrics['psnr'], epoch)
                
                # Save checkpoint
                is_best = val_metrics['ssim'] > self.best_metric
                if is_best:
                    self.best_metric = val_metrics['ssim']
                
                if epoch % self.config['training'].get('save_freq', 5) == 0 or is_best:
                    self.save_checkpoint(val_metrics, is_best)
        
        self.logger.info("Training completed!")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert paths to Path objects
    config['output_dir'] = Path(config['output_dir'])
    config['data_dir'] = Path(config['data_dir'])
    
    # Create output directory
    config['output_dir'].mkdir(parents=True, exist_ok=True)
    
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Video Colorization Models")
    parser.add_argument('--config', type=str, required=True, help='Path to training configuration file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume training')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = VideoColorizationDataset(
        config['data_dir'] / 'train',
        transform=transform,
        include_depth=config.get('use_depth', True),
        include_semantic=config.get('use_semantic', True)
    )
    
    val_dataset = VideoColorizationDataset(
        config['data_dir'] / 'val',
        transform=transform,
        include_depth=config.get('use_depth', True),
        include_semantic=config.get('use_semantic', True)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4)
    )
    
    # Initialize trainer
    trainer = ColorizationTrainer(config)
    
    # Resume training if specified
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.epoch = checkpoint['epoch']
        trainer.global_step = checkpoint['global_step']
        trainer.best_metric = checkpoint['metrics'].get('ssim', 0.0)
        print(f"Resumed training from epoch {trainer.epoch}")
    
    # Start training
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()