import torch
import wandb
from pathlib import Path
import numpy as np
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
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        
        if not hasattr(self, 'history'):
            self.history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_r2': []}
        
        self.history['epoch'].append(float(epoch))
        self.history['train_loss'].append(float(metrics['train_loss']))
        self.history['val_loss'].append(float(metrics['val_loss']))
        self.history['val_r2'].append(float(metrics.get('val_r2', 0)))

        epochs = np.array(self.history['epoch'])
        train_losses = np.array(self.history['train_loss'])
        val_losses = np.array(self.history['val_loss'])
        val_r2s = np.array(self.history['val_r2'])

        axes[0].plot(epochs, train_losses, label='Train Loss', marker='o')
        axes[0].plot(epochs, val_losses, label='Val Loss', marker='o')
        axes[0].set_title('Loss Curves')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True)
        axes[0].legend()

        # Plot R2 curve
        axes[1].plot(epochs, val_r2s, label='Validation R2', marker='o')
        axes[1].set_title('Validation R2 Score')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('R2 Score')
        axes[1].grid(True)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(self.save_dir / f'training_curves_epoch_{epoch}.png')
        plt.close()

        wandb.log({
            'plots/training_curves': wandb.Image(str(self.save_dir / f'training_curves_epoch_{epoch}.png')),
            'epoch/train_loss': metrics['train_loss'],
            'epoch/val_loss': metrics['val_loss'],
            'epoch/val_r2': metrics['val_r2'],
            'epoch': epoch
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