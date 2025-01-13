import torch
import wandb
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import logging

class TrainingMonitor:
    def __init__(self, config: Dict[str, Any], save_dir: Path):
        self.config = config
        self.save_dir = save_dir
        self.logger = logging.getLogger(__name__)
        
        # Initialize wandb
        wandb.init(
            project="ocean_heat_transport",
            name=f"train_{str(save_dir).replace('/', '_')}",
            config=config['training']
        )
        
        # Setup GPU monitoring
        self.start_gpu_monitor()
    
    def start_gpu_monitor(self):
        """Monitor GPU usage"""
        if torch.cuda.is_available():
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def log_batch(self, metrics: Dict[str, float], step: int):
        """Log batch-level metrics"""
        wandb.log({f"batch/{k}": v for k, v in metrics.items()}, step=step)
    
    def log_epoch(self, metrics: Dict[str, float], epoch: int):
        """Log epoch-level metrics"""
        wandb.log({f"epoch/{k}": v for k, v in metrics.items()}, step=epoch)
        
        # Save training curves
        self.plot_training_curves(metrics, epoch)
    
    def plot_training_curves(self, metrics: Dict[str, float], epoch: int):
        """Plot and save training curves"""
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        
        if not hasattr(self, 'history'):
            self.history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_r2': []}
        
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(metrics['train_loss'])
        self.history['val_loss'].append(metrics['val_loss'])
        self.history['val_r2'].append(metrics['val_r2'])
        
        axes[0].plot(self.history['epoch'], self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['epoch'], self.history['val_loss'], label='Val Loss')
        axes[0].set_title('Loss Curves')
        axes[0].legend()
        
        axes[1].plot(self.history['epoch'], self.history['val_r2'], label='Validation R2')
        axes[1].set_title('Validation R2 Score')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'training_curves_epoch_{epoch}.png')
        plt.close()

        wandb.log({
            'plots/training_curves': wandb.Image(str(self.save_dir / f'training_curves_epoch_{epoch}.png')),
            'epoch/train_loss': metrics['train_loss'],
            'epoch/val_loss': metrics['val_loss'],
            'epoch/val_r2': metrics['val_r2'],
            'epoch': epoch,
            **{f'epoch/{k}': v for k, v in metrics.items() if k not in ['train_loss', 'val_loss', 'val_r2']}
        })
    
    def log_model_summary(self, model: torch.nn.Module):
        """Log model architecture summary"""
        wandb.watch(model, log='all')
    
    def log_attention_maps(self, attention_maps: torch.Tensor, epoch: int):
        """Log attention visualizations"""
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(attention_maps[0].cpu(), ax=ax)
        wandb.log({"attention_maps": wandb.Image(fig)}, step=epoch)
        plt.close()
    
    def finish(self):
        """Cleanup monitoring"""
        wandb.finish()