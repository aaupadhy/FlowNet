import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
from pathlib import Path
import wandb
from tqdm import tqdm
import numpy as np
from typing import Dict, Any, Tuple
import time
from datetime import datetime

class OceanTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        save_dir: Path,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.config = config
        self.save_dir = Path(save_dir)
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self._init_wandb()
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['training']['learning_rate'],
            epochs=config['training']['epochs'],
            steps_per_epoch=1,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=10000.0
        )
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging"""
        run_name = f"ocean_transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project="ocean_heat_transport",
            name=run_name,
            config=self.config
        )
    
    def create_dataloaders(
        self,
        ssh_data: np.ndarray,
        sst_data: np.ndarray,
        heat_transport: np.ndarray
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/val/test dataloaders"""
        self.logger.info("Creating dataloaders...")
        self.logger.info(f"Total samples: {len(ssh_data)}")
        
        ssh_tensor = torch.FloatTensor(ssh_data)
        sst_tensor = torch.FloatTensor(sst_data)
        target_tensor = torch.FloatTensor(heat_transport)
        
        n_samples = len(ssh_tensor)
        train_size = int(0.7 * n_samples)
        val_size = int(0.15 * n_samples)
        
        indices = torch.randperm(n_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_dataset = TensorDataset(
            ssh_tensor[train_indices],
            sst_tensor[train_indices],
            target_tensor[train_indices]
        )
        
        val_dataset = TensorDataset(
            ssh_tensor[val_indices],
            sst_tensor[val_indices],
            target_tensor[val_indices]
        )
        
        test_dataset = TensorDataset(
            ssh_tensor[test_indices],
            sst_tensor[test_indices],
            target_tensor[test_indices]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        self.scheduler.total_steps = len(train_loader) * self.config['training']['epochs']
        
        self.logger.info(f"Train samples: {len(train_dataset)}")
        self.logger.info(f"Val samples: {len(val_dataset)}")
        self.logger.info(f"Test samples: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        start_time = time.time()
        
        with tqdm(train_loader, desc="Training") as pbar:
            for i, (ssh, sst, target) in enumerate(pbar):
                ssh = ssh.to(self.device)
                sst = sst.to(self.device)
                target = target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(ssh, sst)
                loss = self.criterion(output, target)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )
                self.optimizer.step()
                self.scheduler.step()
                
                total_loss += loss.item()
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
        
        epoch_time = time.time() - start_time
        metrics = {
            'train_loss': total_loss / len(train_loader),
            'learning_rate': self.scheduler.get_last_lr()[0],
            'epoch_time': epoch_time
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        for ssh, sst, target in val_loader:
            ssh = ssh.to(self.device)
            sst = sst.to(self.device)
            target = target.to(self.device)
            
            output = self.model(ssh, sst)
            loss = self.criterion(output, target)
            
            total_loss += loss.item()
            predictions.extend(output.cpu().numpy())
            targets.extend(target.cpu().numpy())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        r2 = 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - targets.mean()) ** 2)
        rmse = np.sqrt(np.mean((targets - predictions) ** 2))
        mae = np.mean(np.abs(targets - predictions))
        
        metrics = {
            'val_loss': total_loss / len(val_loader),
            'val_r2': r2,
            'val_rmse': rmse,
            'val_mae': mae
        }
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        checkpoint_path = self.save_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.save_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model with val_loss: {metrics['val_loss']:.6f}")
        
        # Save numbered checkpoint every N epochs
        if epoch % self.config['training']['save_freq'] == 0:
            epoch_path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, epoch_path)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ):
        """Train the model"""
        self.logger.info("Starting training...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(self.config['training']['epochs']):
            epoch_start_time = time.time()
            
            train_metrics = self.train_epoch(train_loader)
            
            val_metrics = self.validate(val_loader)
            
            metrics = {**train_metrics, **val_metrics}
            metrics['epoch'] = epoch
            metrics['epoch_time'] = time.time() - epoch_start_time
            
            wandb.log(metrics)
            self.logger.info(
                f"Epoch {epoch}/{self.config['training']['epochs']} - "
                f"Train Loss: {metrics['train_loss']:.6f} - "
                f"Val Loss: {metrics['val_loss']:.6f} - "
                f"Val R²: {metrics['val_r2']:.4f} - "
                f"Time: {metrics['epoch_time']:.2f}s"
            )
            
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                patience_counter = 0
                self.save_checkpoint(epoch, metrics, is_best=True)
            else:
                patience_counter += 1
            
            self.save_checkpoint(epoch, metrics)
            
            if patience_counter >= self.config['training']['patience']:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")
        wandb.finish()
    
    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on test set"""
        self.logger.info("Evaluating model on test set...")
        self.model.eval()
        
        total_loss = 0
        predictions = []
        targets = []
        attention_maps = []
        
        for ssh, sst, target in tqdm(test_loader, desc="Testing"):
            ssh = ssh.to(self.device)
            sst = sst.to(self.device)
            target = target.to(self.device)
            
            # Get predictions and attention maps
            output = self.model(ssh, sst)
            attn_map, _ = self.model.get_attention_maps(ssh, sst)
            
            loss = self.criterion(output, target)
            
            total_loss += loss.item()
            predictions.extend(output.cpu().numpy())
            targets.extend(target.cpu().numpy())
            attention_maps.extend(attn_map.cpu().numpy())
        
        # Calculate metrics
        predictions = np.array(predictions)
        targets = np.array(targets)
        attention_maps = np.array(attention_maps)
        
        metrics = {
            'test_loss': total_loss / len(test_loader),
            'test_r2': 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - targets.mean()) ** 2),
            'test_rmse': np.sqrt(np.mean((targets - predictions) ** 2)),
            'test_mae': np.mean(np.abs(targets - predictions))
        }
        
        self.logger.info(f"Test metrics: {metrics}")
        
        return metrics, predictions, attention_maps
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch'], checkpoint['metrics']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                