import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from src.data.process_data import OceanDataProcessor, OceanDataset
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
from src.utils.dask_utils import dask_monitor
from torch.cuda.amp import autocast

class SmoothHuberLoss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
    def forward(self, pred, target):
        return nn.functional.smooth_l1_loss(pred, target, beta=self.beta)

def setup_random_seeds():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

class OceanTrainer:
    def __init__(self, model, config, save_dir, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.save_dir = Path(save_dir)
        self.logger = logging.getLogger(__name__)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        run_name = f"ocean_transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not wandb.run:
            wandb.init(
                project="ocean_heat_transport",
                name=run_name,
                config=self.config,
                settings=wandb.Settings(start_method="thread")
            )
        self.criterion = SmoothHuberLoss(beta=1.0)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler()
    def create_dataloaders(self, ssh_data, sst_data, heat_transport_data, heat_transport_mean, heat_transport_std,
                           ssh_mean, ssh_std, sst_mean, sst_std, shape):
        debug_mode = self.config['training'].get('debug_mode', False)
        debug_samples = self.config['training'].get('debug_samples', 32)
        dataset = OceanDataset(
            ssh_data, sst_data, heat_transport_data, heat_transport_mean, heat_transport_std,
            ssh_mean, ssh_std, sst_mean, sst_std, shape,
            debug=debug_mode
        )
        total_size = len(dataset)
        if debug_mode:
            train_size = debug_samples // 2
            val_size = debug_samples // 4
            test_size = debug_samples // 4
        else:
            train_size = int(0.7 * total_size)
            val_size = int(0.15 * total_size)
            test_size = total_size - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        batch_size = self.config['training']['batch_size']
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=2
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=2
        )
        self.logger.info(f"Created dataloaders - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
        return train_loader, val_loader, test_loader
    def setup_scheduler(self, train_loader):
        steps_per_epoch = len(train_loader) // self.config['training'].get('accumulation_steps', 4)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config['training']['learning_rate'],
            epochs=self.config['training']['epochs'],
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            div_factor=10.0,
            final_div_factor=1000.0
        )
        self.logger.info(f"Initialized OneCycleLR scheduler with {steps_per_epoch} steps per epoch")
    def train(self, train_loader, val_loader, start_epoch=0):
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(0)
        try:
            if self.scheduler is None:
                self.setup_scheduler(train_loader)
            best_val_loss = float('inf')
            patience_counter = 0
            accumulation_steps = self.config['training'].get('accumulation_steps', 4)
            for epoch in range(start_epoch, self.config['training']['epochs']):
                self.current_epoch = epoch
                self.model.train()
                epoch_loss = 0
                epoch_start = time.time()
                for i, (ssh, sst, attention_mask, target) in enumerate(train_loader):
                    ssh = ssh.to(self.device, non_blocking=True)
                    sst = sst.to(self.device, non_blocking=True)
                    attention_mask = attention_mask.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)
                    self.optimizer.zero_grad(set_to_none=True)
                    with autocast():
                        output, _ = self.model(ssh, sst, attention_mask)
                        loss = self.criterion(output, target)
                        loss = loss / accumulation_steps
                    self.scaler.scale(loss).backward()
                    if (i + 1) % accumulation_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        if self.scheduler is not None:
                            self.scheduler.step()
                    epoch_loss += loss.item() * accumulation_steps
                    torch.cuda.empty_cache()
                epoch_loss = epoch_loss / len(train_loader)
                epoch_time = time.time() - epoch_start
                val_metrics = self.validate(val_loader)
                self.logger.info(f"Epoch {epoch+1} Summary:\nTrain Loss: {epoch_loss:.4f}\nVal Loss: {val_metrics['val_loss']:.4f}\nVal R²: {val_metrics['val_r2']:.4f}\nVal RMSE: {val_metrics['val_rmse']:.4f}\nTime: {epoch_time:.2f}s")
                wandb.log({
                    'epoch': epoch,
                    'epoch/train_loss': epoch_loss,
                    'epoch/val_loss': val_metrics['val_loss'],
                    'epoch/val_r2': val_metrics['val_r2'],
                    'epoch/val_rmse': val_metrics['val_rmse'],
                    'epoch/time': epoch_time
                })
                viz = OceanVisualizer(self.save_dir)
                if val_metrics.get('attention_maps', None) is not None:
                    viz.plot_attention_maps(val_metrics['attention_maps'],
                                            self.config.get('data', {}).get('tlat', None),
                                            self.config.get('data', {}).get('tlong', None),
                                            save_path=f'attention_epoch_{epoch}')
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    patience_counter = 0
                    self.save_checkpoint(epoch, val_metrics, is_best=True)
                else:
                    patience_counter += 1
                    if epoch % self.config['training']['save_freq'] == 0:
                        self.save_checkpoint(epoch, val_metrics)
                if patience_counter >= self.config['training']['patience']:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            self.logger.error(traceback.format_exc())
            if wandb.run:
                wandb.finish()
            raise
        torch.cuda.empty_cache()
        self.logger.info("Training completed successfully")
    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        predictions = []
        targets = []
        n_samples = 0
        all_attention_maps = None
        self.logger.info("Starting validation...")
        for ssh, sst, attention_mask, target in val_loader:
            ssh = ssh.to(self.device, non_blocking=True)
            sst = sst.to(self.device, non_blocking=True)
            attention_mask = attention_mask.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            with autocast():
                output, attn_maps = self.model(ssh, sst, attention_mask)
                loss = self.criterion(output, target)
            val_loss += loss.item()
            predictions.extend(output.detach().cpu().numpy())
            targets.extend(target.detach().cpu().numpy())
            batch_size = ssh.shape[0]
            if all_attention_maps is None:
                all_attention_maps = {key: attn_maps[key].cpu().numpy().mean(axis=0) for key in attn_maps}
            else:
                for key in attn_maps:
                    all_attention_maps[key] = (
                        (all_attention_maps[key] * n_samples) + (attn_maps[key].cpu().numpy().mean(axis=0) * batch_size)
                    ) / (n_samples + batch_size)
            n_samples += batch_size
            torch.cuda.empty_cache()
        predictions = np.array(predictions)
        targets = np.array(targets)
        eps = 1e-8
        ss_total = np.sum((targets - np.mean(targets)) ** 2) + eps
        ss_residual = np.sum((targets - predictions) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        val_loss = val_loss / len(val_loader)
        rmse = np.sqrt(np.mean((targets - predictions) ** 2))
        metrics = {'val_loss': val_loss, 'val_r2': r2, 'val_rmse': rmse, 'attention_maps': all_attention_maps}
        self.logger.info(f"Validation Summary: Loss: {val_loss:.4f}, R²: {r2:.4f}, RMSE: {rmse:.4f}")
        wandb.log({
            'epoch/val_loss': val_loss,
            'epoch/val_r2': r2,
            'epoch/val_rmse': rmse
        })
        return metrics
    def save_checkpoint(self, epoch, metrics, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config,
            'scaler': self.scaler.state_dict(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        }
        checkpoint_dir = self.save_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        latest_path = checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)
        if is_best:
            best_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            wandb.run.summary["best_val_loss"] = metrics['val_loss']
            wandb.run.summary["best_epoch"] = epoch
            wandb.save(str(best_path))
        if epoch % self.config['training']['save_freq'] == 0:
            periodic_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, periodic_path)
            wandb.save(str(periodic_path))
    def resume_from_checkpoint(self, checkpoint_path):
        self.logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        torch.set_rng_state(checkpoint['rng_state'])
        if torch.cuda.is_available() and checkpoint['cuda_rng_state'] is not None:
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        wandb.config.update({"resumed_from_epoch": checkpoint['epoch']})
        self.logger.info(f"Resumed training from epoch {checkpoint['epoch']}")
        return checkpoint['epoch'], checkpoint['metrics']
    @torch.no_grad()
    def evaluate(self, test_loader, heat_transport_mean: float, heat_transport_std: float):
        self.model.eval()
        predictions = []
        targets = []
        n_samples = 0
        test_losses = []
        all_attention_maps = None
        self.logger.info("Starting model evaluation...")
        for ssh, sst, attention_mask, target in test_loader:
            ssh = ssh.to(self.device, non_blocking=True)
            sst = sst.to(self.device, non_blocking=True)
            attention_mask = attention_mask.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            with autocast():
                output, attn_maps = self.model(ssh, sst, attention_mask)
                loss = self.criterion(output, target)
            predictions.append((output.cpu() * heat_transport_std + heat_transport_mean).numpy())
            targets.append((target.cpu() * heat_transport_std + heat_transport_mean).numpy())
            test_losses.append(loss.item())
            batch_size = ssh.shape[0]
            if all_attention_maps is None:
                all_attention_maps = {key: attn_maps[key].cpu().numpy().mean(axis=0) for key in attn_maps}
            else:
                for key in attn_maps:
                    all_attention_maps[key] = (
                        (all_attention_maps[key] * n_samples) + (attn_maps[key].cpu().numpy().mean(axis=0) * batch_size)
                    ) / (n_samples + batch_size)
            n_samples += batch_size
            torch.cuda.empty_cache()
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        for key in all_attention_maps:
            all_attention_maps[key] /= n_samples
        eps = 1e-8
        ss_total = np.sum((targets - np.mean(targets)) ** 2) + eps
        ss_residual = np.sum((targets - predictions) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        metrics = {
            'test_loss': np.mean(test_losses),
            'test_loss_std': np.std(test_losses),
            'test_r2': r2,
            'test_rmse': np.sqrt(np.mean((targets - predictions) ** 2)),
            'test_mae': np.mean(np.abs(targets - predictions))
        }
        self.logger.info("Test Metrics:")
        for k, v in metrics.items():
            self.logger.info(f"{k}: {v:.4f}")
        wandb.run.summary.update({
            'test/loss': metrics['test_loss'],
            'test/loss_std': metrics['test_loss_std'],
            'test/r2': metrics['test_r2'],
            'test/rmse': metrics['test_rmse'],
            'test/mae': metrics['test_mae']
        })
        from src.utils.visualization import OceanVisualizer
        viz = OceanVisualizer(self.save_dir)
        self.logger.info("Plotting final predictions...")
        viz.plot_predictions(
            predictions,
            targets,
            time_indices=np.arange(len(predictions)),
            save_path='final_predictions'
        )
        self.logger.info("Plotting final attention maps...")
        viz.plot_attention_maps(
            all_attention_maps,
            self.config.get('data', {}).get('tlat', None),
            self.config.get('data', {}).get('tlong', None),
            save_path='final_attention_maps'
        )
        results = {
            'metrics': metrics,
            'predictions': predictions,
            'targets': targets,
            'attention_maps': all_attention_maps
        }
        results_path = self.save_dir / 'test_results.pt'
        torch.save(results, results_path)
        wandb.save(str(results_path))
        self.logger.info("Evaluation completed")
        return metrics, predictions, all_attention_maps
