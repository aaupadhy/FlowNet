import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import traceback
import os
from pathlib import Path
import wandb
import numpy as np
import time
from datetime import datetime
from src.architecture.transformer import OceanTransformer
from src.utils.visualization import OceanVisualizer
from torch.cuda.amp import autocast, GradScaler
from src.data.process_data import aggregate_vnt


class SmoothHuberLoss(nn.Module):
    """
    Smooth L1 (Huber) loss - more robust to outliers than MSE
    """
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        
    def forward(self, pred, target):
        return nn.functional.smooth_l1_loss(pred, target, beta=self.beta)


class VNTLoss(nn.Module):
    """
    Loss function for VNT field prediction with mask support
    """
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        
    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: Predicted VNT [B, D, H, W]
            target: Target VNT [B, D, H, W]
            mask: Optional mask [B, H, W]
            
        Returns:
            Weighted MSE loss for VNT prediction
        """
        if mask is not None:
            # Expand mask to match VNT dimensions
            expanded_mask = mask.unsqueeze(1).expand_as(pred)
            
            # Apply mask and calculate MSE on valid points only
            valid_elements = expanded_mask.sum()
            if valid_elements > 0:
                return self.weight * nn.functional.mse_loss(
                    pred * expanded_mask, 
                    target * expanded_mask,
                    reduction='sum'
                ) / valid_elements
            else:
                return torch.tensor(0.0, device=pred.device)
        else:
            # Standard MSE if no mask provided
            return self.weight * nn.functional.mse_loss(pred, target)


def setup_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class OceanTrainer:
    """
    Trainer class for ocean heat transport models
    
    Handles training, validation, evaluation and visualization
    with memory-efficient processing and optional VNT supervision
    """
    def __init__(self, model, config, save_dir, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.save_dir = Path(save_dir)
        self.logger = logging.getLogger(__name__)
        
        # Create necessary directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb for experiment tracking
        self._setup_wandb()
        
        # Set up loss functions
        self.criterion = SmoothHuberLoss(beta=config['training'].get('huber_beta', 1.0))
        self.vnt_criterion = VNTLoss(weight=config['training'].get('vnt_loss_weight', 0.1))
        
        # Set up optimizer with weight decay
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler (to be created in setup_scheduler)
        self.scheduler = None
        
        # Initialize mixed precision scaler for faster training
        self.scaler = GradScaler()
        
        # Set up target transformation parameters
        self.log_target = config['training'].get('log_target', False)
        self.target_scale = config['training'].get('target_scale', 10.0)
        
        # Initialize tracking metrics
        self.best_val_loss = float('inf')
        self.best_val_r2 = float('-inf')
        self.patience_counter = 0
        
        self.logger.info(f"Trainer initialized on device: {device}")
        
    def _setup_wandb(self):
        """Initialize Weights & Biases for experiment tracking"""
        run_name = f"ocean_transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb_mode = self.config.get("wandb_mode", "online")
        
        if wandb_mode.lower() == "offline":
            wandb.init(
                project="ocean_heat_transport", 
                name=run_name, 
                config=self.config['training'], 
                mode="offline"
            )
        else:
            if not wandb.run:
                wandb.init(
                    project="ocean_heat_transport", 
                    name=run_name, 
                    config=self.config['training']
                )
        
        # Log model architecture details
        wandb.run.summary['model_params'] = sum(p.numel() for p in self.model.parameters())
        wandb.run.summary['trainable_params'] = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        
    def _create_optimizer(self):
        """Create optimizer with appropriate parameters"""
        # Get parameters from config
        opt_name = self.config['training'].get('optimizer', {}).get('name', 'adamw')
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training'].get('weight_decay', 0.01)
        
        # Create optimizer based on type
        if opt_name.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif opt_name.lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif opt_name.lower() == 'sgd':
            momentum = self.config['training'].get('optimizer', {}).get('momentum', 0.9)
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            self.logger.warning(f"Unknown optimizer {opt_name}, defaulting to AdamW")
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        
    def create_dataloaders(self, ssh_data, sst_data, ht_data, ht_mean, ht_std,
                          ssh_mean, ssh_std, sst_mean, sst_std, shape, 
                          vnt_data=None, tarea=None, dz=None, ref_lat_index=0):
        """
        Create training, validation and test dataloaders
        
        Args:
            ssh_data: SSH data array
            sst_data: SST data array
            ht_data: Heat transport data array
            ht_mean, ht_std: Heat transport statistics
            ssh_mean, ssh_std: SSH statistics
            sst_mean, sst_std: SST statistics
            shape: Data shape
            vnt_data: Optional VNT data for supervision
            tarea, dz: Grid information for heat transport calculation
            ref_lat_index: Reference latitude index
            
        Returns:
            train_loader, val_loader, test_loader
        """
        from src.data.process_data import OceanDataset
        
        # Check for debug mode
        debug_mode = self.config['training'].get('debug_mode', False)
        
        # Create dataset
        dataset = OceanDataset(
            ssh_data, sst_data, ht_data, ht_mean, ht_std,
            ssh_mean, ssh_std, sst_mean, sst_std, shape,
            vnt_data=vnt_data, tarea=tarea, dz=dz, ref_lat_index=ref_lat_index,
            debug=debug_mode, log_target=self.log_target, target_scale=self.target_scale
        )
        
        # Determine split sizes
        total_samples = len(dataset)
        train_frac = 1.0 - self.config['training'].get('validation_split', 0.15) - self.config['training'].get('test_split', 0.15)
        train_size = int(train_frac * total_samples)
        val_size = int(self.config['training'].get('validation_split', 0.15) * total_samples)
        test_size = total_samples - train_size - val_size
        
        # Compute indices for each split
        indices = np.arange(total_samples)
        
        if self.config['training'].get('shuffle_split', True):
            # Shuffle indices for random splits
            np.random.shuffle(indices)
            
        # Create subsets
        from torch.utils.data import Subset
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]
        
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)
        
        # Create dataloaders
        batch_size = self.config['training']['batch_size']
        num_workers = self.config['training'].get('num_workers', 4)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            pin_memory=True,
            num_workers=num_workers, 
            persistent_workers=True, 
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            pin_memory=True, 
            num_workers=max(1, num_workers//2)
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            pin_memory=True, 
            num_workers=max(1, num_workers//2)
        )
        
        self.logger.info(f"Created dataloaders - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
        
    def setup_scheduler(self, train_loader):
        """Set up learning rate scheduler"""
        scheduler_config = self.config['training'].get('scheduler', {})
        scheduler_name = scheduler_config.get('name', 'onecycle')
        
        # Calculate steps per epoch and total steps
        steps_per_epoch = len(train_loader)
        total_epochs = self.config['training']['epochs']
        total_steps = steps_per_epoch * total_epochs
        
        # Create scheduler based on type
        if scheduler_name.lower() == 'onecycle':
            max_lr = scheduler_config.get('max_lr', self.config['training']['learning_rate'])
            pct_start = scheduler_config.get('pct_start', 0.3)
            
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lr,
                total_steps=total_steps,
                pct_start=pct_start,
                div_factor=scheduler_config.get('div_factor', 10.0),
                final_div_factor=scheduler_config.get('final_div_factor', 100.0)
            )
            
            self.logger.info(f"Initialized OneCycleLR scheduler with {steps_per_epoch} steps per epoch")
            
        elif scheduler_name.lower() == 'cosine':
            # Cosine annealing with warm restarts
            t_0 = scheduler_config.get('t_0', 10) * steps_per_epoch  # in steps
            t_mult = scheduler_config.get('t_mult', 2)
            
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=t_0,
                T_mult=t_mult,
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
            
            self.logger.info(f"Initialized CosineAnnealingWarmRestarts scheduler with T_0={t_0} steps")
            
        elif scheduler_name.lower() == 'step':
            # Step LR with gamma decay
            step_size = scheduler_config.get('step_size', 10) * steps_per_epoch
            gamma = scheduler_config.get('gamma', 0.1)
            
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
            
            self.logger.info(f"Initialized StepLR scheduler with step_size={step_size} steps, gamma={gamma}")
            
        else:
            # Default to OneCycleLR if unknown scheduler
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config['training']['learning_rate'],
                total_steps=total_steps,
                pct_start=0.3,
                div_factor=10.0,
                final_div_factor=100.0
            )
            
            self.logger.info(f"Using default OneCycleLR scheduler with {steps_per_epoch} steps per epoch")
        
    def train(self, train_loader, val_loader, start_epoch=0):
        """
        Train the model
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            start_epoch: Starting epoch (for resuming training)
            
        Returns:
            None
        """
        # Clear GPU cache before starting
        torch.cuda.empty_cache()
        
        # Reset peak memory stats for monitoring
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device if isinstance(self.device, int) else 0)
        
        # Set up gradient accumulation
        effective_batch_size = self.config['training'].get('effective_batch_size', self.config['training']['batch_size'])
        accumulation_steps = self.config['training'].get('accumulation_steps', 1)
        actual_batch_size = self.config['training']['batch_size']
        
        if accumulation_steps > 1:
            self.logger.info(f"Using gradient accumulation: {accumulation_steps} steps")
            self.logger.info(f"Effective batch size: {effective_batch_size} (actual: {actual_batch_size})")
        
        try:
            # Create scheduler if not already initialized
            if self.scheduler is None:
                self.setup_scheduler(train_loader)
                
            # Set up early stopping parameters
            best_val = float('inf')
            patience = self.config['training'].get('early_stopping_patience', 10)
            patience_counter = 0
            
            # Main training loop
            for epoch in range(start_epoch, self.config['training']['epochs']):
                # Training phase
                self.model.train()
                train_loss = 0.0
                train_direct_loss = 0.0
                train_vnt_loss = 0.0
                epoch_start_time = time.time()
                
                # Reset optimizer gradients at the start of each epoch
                self.optimizer.zero_grad(set_to_none=True)
                
                # Process batches
                for batch_idx, batch in enumerate(train_loader):
                    # Check if we need to accumulate gradients
                    is_accumulation_step = (batch_idx + 1) % accumulation_steps != 0
                    
                    # Handle different batch formats based on VNT data availability
                    vnt_target = None
                    if len(batch) == 4:
                        ssh, sst, mask, target = batch
                        ssh = ssh.to(self.device)
                        sst = sst.to(self.device)
                        mask = mask.to(self.device)
                        target = target.to(self.device)
                    elif len(batch) == 5:
                        ssh, sst, mask, target, vnt_target = batch
                        ssh = ssh.to(self.device)
                        sst = sst.to(self.device)
                        mask = mask.to(self.device)
                        target = target.to(self.device)
                        vnt_target = vnt_target.to(self.device)
                    
                    # Run forward pass with autocast for mixed precision
                    with autocast():
                        heat_pred, vnt_pred, _ = self.model(ssh, sst, mask)
                        
                        # Direct heat transport loss
                        direct_loss = self.criterion(heat_pred, target)
                        
                        # VNT prediction loss if available
                        if vnt_target is not None:
                            vnt_loss = self.vnt_criterion(vnt_pred, vnt_target, mask)
                            loss = direct_loss + vnt_loss
                        else:
                            vnt_loss = torch.tensor(0.0, device=self.device)
                            loss = direct_loss
                        
                        # Scale loss for gradient accumulation
                        if accumulation_steps > 1:
                            loss = loss / accumulation_steps
                    
                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                    
                    # Only update weights after accumulation or at the end of epoch
                    if not is_accumulation_step or batch_idx == len(train_loader) - 1:
                        # Unscale for gradient clipping
                        if self.config['training'].get('grad_clip', 0) > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 
                                self.config['training']['grad_clip']
                            )
                            
                        # Optimizer and scheduler step
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                    
                    # Accumulate losses for logging
                    train_loss += loss.item() * (accumulation_steps if is_accumulation_step else 1)
                    train_direct_loss += direct_loss.item()
                    train_vnt_loss += vnt_loss.item() if vnt_target is not None else 0
                    
                    # Log batch metrics periodically
                    if batch_idx % 10 == 0:
                        # Calculate learning rate
                        current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]['lr']
                        
                        # Log to wandb
                        wandb.log({
                            'batch/loss': loss.item() * (accumulation_steps if is_accumulation_step else 1),
                            'batch/direct_loss': direct_loss.item(),
                            'batch/vnt_loss': vnt_loss.item() if vnt_target is not None else 0,
                            'batch/lr': current_lr,
                            'batch/gpu_memory': torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                        }, step=epoch * len(train_loader) + batch_idx)
                
                # Calculate average losses for the epoch
                avg_train_loss = train_loss / len(train_loader)
                avg_train_direct_loss = train_direct_loss / len(train_loader)
                avg_train_vnt_loss = train_vnt_loss / len(train_loader)
                
                # Calculate epoch time
                epoch_time = time.time() - epoch_start_time
                
                # Run validation
                val_metrics = self.validate(val_loader)
                
                # Log epoch metrics
                self.logger.info(
                    f"Epoch {epoch+1}/{self.config['training']['epochs']} Summary:\n"
                    f"  Train Loss: {avg_train_loss:.4f}\n"
                    f"  Train Direct Loss: {avg_train_direct_loss:.4f}\n"
                    f"  Train VNT Loss: {avg_train_vnt_loss:.4f}\n"
                    f"  Val Loss: {val_metrics['val_loss']:.4f}\n"
                    f"  Val R²: {val_metrics['val_r2']:.4f}\n"
                    f"  Val RMSE: {val_metrics['val_rmse']:.4f}\n"
                    f"  Time: {epoch_time:.2f}s"
                )
                
                # Log to wandb
                wandb.log({
                    'epoch': epoch,
                    'epoch/train_loss': avg_train_loss,
                    'epoch/train_direct_loss': avg_train_direct_loss,
                    'epoch/train_vnt_loss': avg_train_vnt_loss,
                    'epoch/val_loss': val_metrics['val_loss'],
                    'epoch/val_r2': val_metrics['val_r2'],
                    'epoch/val_rmse': val_metrics['val_rmse'],
                    'epoch/val_mape': val_metrics['val_mape'],
                    'epoch/explained_variance': val_metrics['explained_variance'],
                    'epoch/time': epoch_time
                })
                
                # Check for best model
                if val_metrics['val_loss'] < best_val:
                    best_val = val_metrics['val_loss']
                    patience_counter = 0
                    
                    # Save best model
                    torch.save(
                        {
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                            'val_loss': val_metrics['val_loss'],
                            'val_r2': val_metrics['val_r2']
                        },
                        self.save_dir / 'checkpoints' / 'best_model.pt'
                    )
                    
                    wandb.run.summary['best_epoch'] = epoch
                    wandb.run.summary['best_val_loss'] = best_val
                    wandb.run.summary['best_val_r2'] = val_metrics['val_r2']
                    
                    self.logger.info(f"New best model saved at epoch {epoch+1}")
                else:
                    patience_counter += 1
                
                # Save periodic checkpoints
                if (epoch + 1) % self.config['training'].get('save_freq', 5) == 0:
                    torch.save(
                        {
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                            'val_loss': val_metrics['val_loss']
                        },
                        self.save_dir / f"checkpoints/checkpoint_epoch_{epoch+1}.pt"
                    )
                    
                    self.logger.info(f"Checkpoint saved at epoch {epoch+1}")
                
                # Early stopping check
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
                    
            self.logger.info("Training completed successfully")
            
        except Exception as ex:
            self.logger.error(f"Error during training: {ex}")
            self.logger.error(traceback.format_exc())
            raise
            
    def validate(self, loader):
        """
        Validate model on validation set
        
        Args:
            loader: DataLoader for validation data
            
        Returns:
            Dictionary of validation metrics
        """
        self.logger.info("Starting validation...")
        self.model.eval()
        
        # Initialize accumulators
        total_loss = 0.0
        direct_loss = 0.0
        vnt_loss = 0.0
        all_preds = []
        all_targets = []
        
        # Process validation batches
        with torch.no_grad(), autocast():
            for batch in loader:
                # Handle different batch formats
                vnt_target = None
                if len(batch) == 4:
                    ssh, sst, mask, target = batch
                    ssh = ssh.to(self.device)
                    sst = sst.to(self.device)
                    mask = mask.to(self.device)
                    target = target.to(self.device)
                elif len(batch) == 5:
                    ssh, sst, mask, target, vnt_target = batch
                    ssh = ssh.to(self.device)
                    sst = sst.to(self.device)
                    mask = mask.to(self.device)
                    target = target.to(self.device)
                    vnt_target = vnt_target.to(self.device)
                
                # Forward pass
                heat_pred, vnt_pred, _ = self.model(ssh, sst, mask)
                
                # Calculate direct heat transport loss
                batch_direct_loss = self.criterion(heat_pred, target)
                direct_loss += batch_direct_loss.item()
                
                # Calculate VNT loss if available
                if vnt_target is not None:
                    batch_vnt_loss = self.vnt_criterion(vnt_pred, vnt_target, mask)
                    vnt_loss += batch_vnt_loss.item()
                    total_loss += batch_direct_loss.item() + batch_vnt_loss.item()
                else:
                    total_loss += batch_direct_loss.item()
                
                # Collect predictions for metrics calculation
                all_preds.extend(heat_pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate average losses
        avg_loss = total_loss / len(loader)
        avg_direct_loss = direct_loss / len(loader)
        avg_vnt_loss = vnt_loss / len(loader) if vnt_loss > 0 else 0
        
        # Convert to numpy arrays for metric calculation
        predictions = np.array(all_preds)
        targets = np.array(all_targets)
        
        # Reverse log transformation if applied
        if self.log_target:
            # Safely reverse log transform
            predictions = np.clip(predictions * self.target_scale, -100, 100)  # Prevent overflow
            targets = np.clip(targets * self.target_scale, -100, 100)
            
            predictions = np.exp(predictions) - 1e-8
            targets = np.exp(targets) - 1e-8
        
        # Calculate metrics
        epsilon = 1e-8  # Small value to prevent division by zero
        
        # Explained variance
        target_var = np.var(targets) + epsilon
        explained_variance = 1 - np.var(targets - predictions) / target_var
        
        # RMSE
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        
        # MAPE (Mean Absolute Percentage Error)
        non_zero_mask = np.abs(targets) > epsilon
        if np.sum(non_zero_mask) > 0:
            mape = np.mean(np.abs((predictions[non_zero_mask] - targets[non_zero_mask]) / targets[non_zero_mask])) * 100
        else:
            mape = float('nan')
        
        # R-squared coefficient of determination
        r2 = 1.0 - np.sum((predictions - targets) ** 2) / (np.sum((targets - np.mean(targets)) ** 2) + epsilon)
        
        # Log validation summary
        self.logger.info(
            f"Validation Summary:\n"
            f"  Loss: {avg_loss:.4f}\n"
            f"  Direct Loss: {avg_direct_loss:.4f}\n"
            f"  VNT Loss: {avg_vnt_loss:.4f}\n"
            f"  R²: {r2:.4f}\n"
            f"  RMSE: {rmse:.4f}\n"
            f"  MAPE: {mape:.4f}\n"
            f"  Explained Variance: {explained_variance:.4f}"
        )
        
        # Return metrics dictionary
        return {
            'val_loss': avg_loss,
            'val_direct_loss': avg_direct_loss,
            'val_vnt_loss': avg_vnt_loss,
            'val_r2': r2,
            'val_rmse': rmse,
            'val_mape': mape,
            'explained_variance': explained_variance,
            'predictions': predictions,
            'targets': targets
        }
    
    def get_attention_maps(self, ssh, sst, mask):
        """
        Extract attention maps from model for visualization
        
        Args:
            ssh: SSH tensor [B, 1, H, W]
            sst: SST tensor [B, 1, H, W]
            mask: Mask tensor [B, H, W]
            
        Returns:
            Dictionary with attention maps and metadata
        """
        self.model.eval()
        with torch.no_grad(), autocast():
            _, _, out_dict = self.model(
                ssh.to(self.device),
                sst.to(self.device),
                mask.to(self.device)
            )
        return out_dict
    
    def evaluate(self, test_loader, ht_mean=None, ht_std=None, return_vnt=False):
        """
        Evaluate model on test set
        
        Args:
            test_loader: DataLoader for test data
            ht_mean: Mean of heat transport data (optional)
            ht_std: Standard deviation of heat transport data (optional)
            return_vnt: Whether to return VNT predictions
            
        Returns:
            metrics: Dictionary of evaluation metrics
            predictions: Array of model predictions
            truth: Array of ground truth values
            attention_maps: Attention visualizations from first batch
            vnt_predictions: VNT predictions if return_vnt=True
        """
        self.logger.info("Starting model evaluation...")
        self.model.eval()
        
        # Load best model if available
        best_model_path = self.save_dir / 'checkpoints' / 'best_model.pt'
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}")
        
        # Initialize accumulators
        all_heat_preds = []
        all_targets = []
        all_vnt_preds = []
        all_vnt_targets = []
        attention_maps = None
        
        try:
            with torch.no_grad(), autocast():
                for batch_idx, batch in enumerate(test_loader):
                    # Handle different batch formats
                    vnt_target = None
                    if len(batch) == 4:
                        ssh, sst, mask, target = batch
                        ssh = ssh.to(self.device)
                        sst = sst.to(self.device)
                        mask = mask.to(self.device)
                        target = target.to(self.device)
                    elif len(batch) == 5:
                        ssh, sst, mask, target, vnt_target = batch
                        ssh = ssh.to(self.device)
                        sst = sst.to(self.device)
                        mask = mask.to(self.device)
                        target = target.to(self.device)
                        vnt_target = vnt_target.to(self.device)
                    
                    # Forward pass
                    heat_pred, vnt_pred, out_dict = self.model(ssh, sst, mask)
                    
                    # Collect predictions
                    all_heat_preds.extend(heat_pred.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
                    
                    # Collect VNT predictions if requested
                    if return_vnt and vnt_target is not None:
                        all_vnt_preds.append(vnt_pred.cpu().numpy())
                        all_vnt_targets.append(vnt_target.cpu().numpy())
                    
                    # Store attention maps from first batch for visualization
                    if batch_idx == 0:
                        attention_maps = out_dict
            
            # Convert to numpy arrays
            predictions = np.array(all_heat_preds)
            truth = np.array(all_targets)
            
            # Reverse log transformation if applied
            if self.log_target:
                # Clip values to prevent overflow
                predictions = np.clip(predictions * self.target_scale, -100, 100)
                truth = np.clip(truth * self.target_scale, -100, 100)
                
                # Apply exponential function to reverse log transform
                predictions = np.exp(predictions) - 1e-8
                truth = np.exp(truth) - 1e-8
            
            # Calculate metrics
            epsilon = 1e-8
            
            # Basic error metrics
            mse = np.mean((predictions - truth) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - truth))
            
            # Mean Absolute Percentage Error
            non_zero_mask = np.abs(truth) > epsilon
            if np.sum(non_zero_mask) > 0:
                mape = np.mean(np.abs((predictions[non_zero_mask] - truth[non_zero_mask]) / truth[non_zero_mask])) * 100
            else:
                mape = float('nan')
            
            # R-squared coefficient of determination
            r2 = 1 - np.sum((predictions - truth) ** 2) / (np.sum((truth - np.mean(truth)) ** 2) + epsilon)
            
            # Explained variance
            explained_var = 1 - np.var(predictions - truth) / (np.var(truth) + epsilon)
            
            # Compile metrics
            metrics = {
                'test_mse': mse,
                'test_rmse': rmse,
                'test_mae': mae,
                'test_r2': r2,
                'test_mape': mape,
                'test_explained_var': explained_var
            }
            
            # Log metrics
            self.logger.info(f"Test Metrics:")
            for metric_name, value in metrics.items():
                self.logger.info(f"  {metric_name}: {value:.4f}")
                wandb.run.summary[metric_name] = value
            
            # Process VNT predictions if available
            if all_vnt_preds and all_vnt_targets:
                self.logger.info("Calculating VNT prediction metrics...")
                # Simple MSE for VNT field prediction
                vnt_mse = np.mean([(pred - target) ** 2 for pred, target in zip(all_vnt_preds, all_vnt_targets)])
                metrics['vnt_mse'] = vnt_mse
                wandb.run.summary['vnt_mse'] = vnt_mse
                self.logger.info(f"  VNT MSE: {vnt_mse:.4f}")
            
            # Return results
            if return_vnt and all_vnt_preds:
                return metrics, predictions, truth, attention_maps, all_vnt_preds
            else:
                return metrics, predictions, truth, attention_maps
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def calculate_heat_transport_from_vnt(self, loader, tarea, dz, ref_lat_index):
        """
        Calculate and evaluate heat transport by aggregating predicted VNT fields
        
        Args:
            loader: DataLoader for test data
            tarea: Area tensor for aggregation
            dz: Depth tensor for aggregation
            ref_lat_index: Reference latitude index
            
        Returns:
            Dictionary of metrics comparing aggregated VNT with ground truth
        """
        self.logger.info("Starting VNT-based heat transport evaluation...")
        self.model.eval()
        
        # Load best model if available
        best_model_path = self.save_dir / 'checkpoints' / 'best_model.pt'
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}")
        
        # Prepare tensors for torch-based aggregation
        if isinstance(tarea, np.ndarray):
            tarea_tensor = torch.tensor(tarea, device=self.device, dtype=torch.float32)
        elif isinstance(tarea, torch.Tensor):
            tarea_tensor = tarea.to(self.device)
        else:
            tarea_tensor = torch.tensor(tarea.values, device=self.device, dtype=torch.float32)
            
        if isinstance(dz, np.ndarray):
            dz_tensor = torch.tensor(dz, device=self.device, dtype=torch.float32)
        elif isinstance(dz, torch.Tensor):
            dz_tensor = dz.to(self.device)
        else:
            dz_tensor = torch.tensor(dz.values, device=self.device, dtype=torch.float32)
        
        # Collect predictions and targets
        preds = []
        targets = []
        
        with torch.no_grad(), autocast():
            for batch in loader:
                # Handle different batch formats
                if len(batch) == 4:
                    ssh, sst, mask, target = batch
                    ssh = ssh.to(self.device)
                    sst = sst.to(self.device)
                    mask = mask.to(self.device)
                    target = target.to(self.device)
                elif len(batch) == 5:
                    ssh, sst, mask, target, _ = batch
                    ssh = ssh.to(self.device)
                    sst = sst.to(self.device)
                    mask = mask.to(self.device)
                    target = target.to(self.device)
                
                # Predict VNT field
                _, predicted_vnt, _ = self.model(ssh, sst, mask)
                
                # Calculate heat transport from predicted VNT
                if self.model.heat_aggregator is not None:
                    # Use model's built-in aggregator if available
                    agg_pred = self.model.heat_aggregator(predicted_vnt)
                else:
                    # Otherwise use the function from process_data
                    agg_pred = aggregate_vnt(
                        predicted_vnt, 
                        tarea_tensor, 
                        dz_tensor, 
                        ref_lat_index=ref_lat_index
                    )
                
                # Collect results
                preds.extend(agg_pred.cpu().numpy())
                targets.extend(target.cpu().numpy())
        
        # Convert to numpy arrays
        preds = np.array(preds)
        targets = np.array(targets)
        
        # Reverse log transformation if applied
        if self.log_target:
            # Clip values to prevent overflow
            preds = np.clip(preds * self.target_scale, -100, 100)
            targets = np.clip(targets * self.target_scale, -100, 100)
            
            # Apply exponential function to reverse log transform
            preds = np.exp(preds) - 1e-8
            targets = np.exp(targets) - 1e-8
        
        # Calculate metrics
        epsilon = 1e-8
        
        # Basic error metrics
        mse = np.mean((preds - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(preds - targets))
        
        # Mean Absolute Percentage Error
        non_zero_mask = np.abs(targets) > epsilon
        if np.sum(non_zero_mask) > 0:
            mape = np.mean(np.abs((preds[non_zero_mask] - targets[non_zero_mask]) / targets[non_zero_mask])) * 100
        else:
            mape = float('nan')
        
        # R-squared coefficient of determination
        r2 = 1 - np.sum((preds - targets) ** 2) / (np.sum((targets - np.mean(targets)) ** 2) + epsilon)
        
        # Explained variance
        explained_var = 1 - np.var(preds - targets) / (np.var(targets) + epsilon)
        
        # Log metrics
        self.logger.info(
            f"VNT-based Heat Transport Evaluation:\n"
            f"  MSE: {mse:.4f}\n"
            f"  RMSE: {rmse:.4f}\n"
            f"  MAE: {mae:.4f}\n"
            f"  R²: {r2:.4f}\n"
            f"  MAPE: {mape:.4f}\n"
            f"  Explained Variance: {explained_var:.4f}"
        )
        
        # Log to wandb
        wandb.log({
            'vnt_eval/mse': mse,
            'vnt_eval/rmse': rmse,
            'vnt_eval/mae': mae,
            'vnt_eval/r2': r2,
            'vnt_eval/mape': mape,
            'vnt_eval/explained_var': explained_var
        })
        
        # Return complete metrics and predictions for further analysis
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'explained_var': explained_var,
            'predictions': preds,
            'targets': targets
        }
    
    def load_checkpoint(self, checkpoint_path=None):
        """
        Load model from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file, defaults to best model
            
        Returns:
            Epoch number of loaded checkpoint
        """
        if checkpoint_path is None:
            checkpoint_path = self.save_dir / 'checkpoints' / 'best_model.pt'
        
        if not os.path.exists(checkpoint_path):
            self.logger.error(f"Checkpoint not found at {checkpoint_path}")
            return 0
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Optionally load optimizer and scheduler states
            if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None and self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            epoch = checkpoint.get('epoch', -1) + 1
            self.logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch})")
            
            # Log best validation metrics if available
            if 'val_loss' in checkpoint:
                self.logger.info(f"Checkpoint validation loss: {checkpoint['val_loss']:.4f}")
                
            if 'val_r2' in checkpoint:
                self.logger.info(f"Checkpoint validation R²: {checkpoint['val_r2']:.4f}")
                
            return epoch
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            return 0
    
    def save_checkpoint(self, epoch, val_metrics=None, is_best=False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch number
            val_metrics: Validation metrics dictionary
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
        }
        
        # Add validation metrics if available
        if val_metrics is not None:
            for k, v in val_metrics.items():
                if isinstance(v, (int, float)):
                    checkpoint[k] = v
        
        # Determine save path
        if is_best:
            save_path = self.save_dir / 'checkpoints' / 'best_model.pt'
        else:
            save_path = self.save_dir / 'checkpoints' / f'checkpoint_epoch_{epoch}.pt'
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save checkpoint
        torch.save(checkpoint, save_path)
        self.logger.info(f"Saved checkpoint to {save_path}")
        
        return save_path