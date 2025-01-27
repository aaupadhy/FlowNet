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
from typing import Dict, Any, Tuple
import time
from datetime import datetime
from src.utils.visualization import OceanVisualizer
from src.utils.training_monitor import TrainingMonitor
import torch.cuda.amp
from torch.cuda.amp import autocast

def setup_random_seeds():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

class OceanTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        save_dir: Path,
        device: str = 'cuda'
    ):
        self.device = torch.device(device)
        setup_random_seeds()
        self.model = model.to(self.device)
        self.config = config
        self.save_dir = Path(save_dir)
        self.logger = logging.getLogger(__name__)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        run_name = f"ocean_transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not wandb.run:
            wandb.init(
                project="FlowNet",
                name=run_name,
                config=self.config,
                settings=wandb.Settings(start_method="thread")
            )
            self.monitor = TrainingMonitor(config, save_dir)
            self.monitor.log_model_summary(model)

        self.criterion = nn.SmoothL1Loss(reduction='mean')
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        self.scaler = torch.cuda.amp.GradScaler()
        self.scheduler = None

    def setup_scheduler(self, train_loader: DataLoader):
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

    def create_dataloaders(self, ssh_data, sst_data, heat_transport_data, heat_transport_mean,
                          ssh_mean, ssh_std, sst_mean, sst_std, shape):
        debug_mode = self.config['training'].get('debug_mode', False)
        debug_samples = self.config['training'].get('debug_samples', 32)
        
        dataset = OceanDataset(
            ssh_data, sst_data, heat_transport_data, heat_transport_mean,
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
            drop_last=True,
            prefetch_factor=2
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

        self.logger.info(f"Created dataloaders - Train: {len(train_loader.dataset)}, "
                        f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
        return train_loader, val_loader, test_loader

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        val_loss = 0
        predictions = []
        targets = []
        batch_times = []
        
        # For progressive averaging
        attention_sum = None
        n_samples = 0

        for ssh, sst, attention_mask, target in val_loader:
            batch_start = time.time()
            ssh = ssh.to(self.device, non_blocking=True)
            sst = sst.to(self.device, non_blocking=True)
            attention_mask = attention_mask.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            try:
                with autocast():
                    output, attn_maps = self.model(ssh, sst, attention_mask)
                    loss = self.criterion(output, target)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    self.logger.error(f"OOM in validation: {str(e)}")
                raise

            val_loss += loss.item()
            predictions.extend(output.cpu().numpy())
            targets.extend(target.cpu().numpy())
            batch_times.append(time.time() - batch_start)
            
            # Process attention maps for the current batch
            attn_numpy = attn_maps.cpu().numpy()
            current_attention = attn_numpy.mean(axis=(0, 1))  # Mean over batch and heads
            
            # Update running average
            if attention_sum is None:
                attention_sum = current_attention
                n_samples = 1
            else:
                attention_sum = attention_sum + current_attention
                n_samples += 1

            del output, attn_maps, loss, current_attention
            torch.cuda.empty_cache()

        avg_attention_map = attention_sum / n_samples

        attention_dir = self.save_dir / 'attention_maps'
        attention_dir.mkdir(parents=True, exist_ok=True)
        attention_path = attention_dir / f'epoch_{self.current_epoch}.png'
        
        viz = OceanVisualizer(self.save_dir)
        viz.plot_attention_maps(
            avg_attention_map,
            self.config.get('data', {}).get('tlat', None),
            self.config.get('data', {}).get('tlong', None),
            save_path=str(attention_path)
        )

        val_loss = val_loss / len(val_loader)
        predictions = np.array(predictions)
        targets = np.array(targets)
        r2 = 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - targets.mean()) ** 2)
        rmse = np.sqrt(np.mean((targets - predictions) ** 2))

        if attention_path.exists():
            wandb.log({
                f'attention_maps/epoch_{self.current_epoch}': wandb.Image(str(attention_path)),
                'attention/avg_magnitude': float(np.mean(np.abs(avg_attention_map))),
                'attention/max_value': float(np.max(avg_attention_map)),
                'attention/sparsity': float(np.sum(avg_attention_map < 0.01) / avg_attention_map.size)
            })

        return {
            'val_loss': val_loss,
            'val_r2': r2,
            'val_rmse': rmse,
            'val_time': np.mean(batch_times)
        }
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader, start_epoch: int = 0):
        attention_dir = self.save_dir / 'attention_maps'
        attention_dir.mkdir(parents=True, exist_ok=True)
        
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(0)

        try:
            if self.scheduler is None:
                self.setup_scheduler(train_loader)

            best_val_loss = float('inf')
            patience_counter = 0
            accumulation_steps = self.config['training'].get('accumulation_steps', 4)
            total_batches = len(train_loader)

            self.logger.info(f"Starting training from epoch {start_epoch}")
            self.logger.info(f"Total epochs: {self.config['training']['epochs']}")
            self.logger.info(f"Total batches per epoch: {total_batches}")
            self.logger.info(f"Device: {self.device}")
            self.logger.info(f"Batch size: {self.config['training']['batch_size']}")

            for epoch in range(start_epoch, self.config['training']['epochs']):
                self.current_epoch = epoch
                self.model.train()
                epoch_loss = 0
                batch_times = []
                epoch_start = time.time()

                self.logger.info(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}")

                for i, (ssh, sst, attention_mask, target) in enumerate(train_loader):
                    try:
                        batch_start = time.time()
                        ssh = ssh.to(self.device, non_blocking=True)
                        sst = sst.to(self.device, non_blocking=True)
                        attention_mask = attention_mask.to(self.device, non_blocking=True)
                        target = target.to(self.device, non_blocking=True)

                        self.optimizer.zero_grad(set_to_none=True)

                        try:
                            with autocast():
                                output, attention_maps = self.model(ssh, sst, attention_mask)
                                loss = self.criterion(output, target)
                                loss = loss / accumulation_steps
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                torch.cuda.empty_cache()
                                self.logger.error(f"OOM in forward pass: {str(e)}")
                            raise

                        self.scaler.scale(loss).backward()

                        if (i + 1) % accumulation_steps == 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config['training']['grad_clip']
                            )
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.scheduler.step()

                        batch_loss = loss.item() * accumulation_steps
                        epoch_loss += batch_loss
                        batch_time = time.time() - batch_start
                        batch_times.append(batch_time)

                        if i % 10 == 0:
                            with torch.no_grad():
                                predictions = output.detach().cpu().numpy()
                                targets = target.detach().cpu().numpy()
                                batch_r2 = 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - targets.mean()) ** 2)
                                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                                memory_reserved = torch.cuda.memory_reserved() / 1024**3

                                self.logger.info(
                                    f"Batch {i}/{total_batches} | "
                                    f"Loss: {batch_loss:.4f} | "
                                    f"R²: {batch_r2:.4f} | "
                                    f"LR: {self.scheduler.get_last_lr()[0]:.2e} | "
                                    f"Time: {batch_time:.2f}s | "
                                    f"GPU Memory: {memory_allocated:.2f}GB allocated/{memory_reserved:.2f}GB reserved"
                                )

                                wandb.log({
                                    'batch/loss': batch_loss,
                                    'batch/r2': batch_r2,
                                    'batch/learning_rate': self.scheduler.get_last_lr()[0],
                                    'batch/time': batch_time,
                                    'batch/memory_allocated': memory_allocated,
                                    'batch/memory_reserved': memory_reserved
                                })

                        del output, attention_maps, loss
                        torch.cuda.empty_cache()

                    except Exception as e:
                        self.logger.error(f"Error in batch {i}: {str(e)}")
                        self.logger.error(traceback.format_exc())
                        raise

                epoch_loss = epoch_loss / len(train_loader)
                epoch_time = time.time() - epoch_start
                
                val_metrics = self.validate(val_loader)

                self.logger.info(
                    f"\nEpoch {epoch+1} Summary:\n"
                    f"Train Loss: {epoch_loss:.4f}\n"
                    f"Val Loss: {val_metrics['val_loss']:.4f}\n"
                    f"Val R²: {val_metrics['val_r2']:.4f}\n"
                    f"Val RMSE: {val_metrics['val_rmse']:.4f}\n"
                    f"Time: {epoch_time:.2f}s"
                )

                wandb.log({
                    'epoch': epoch,
                    'epoch/train_loss': epoch_loss,
                    'epoch/val_loss': val_metrics['val_loss'],
                    'epoch/val_r2': val_metrics['val_r2'],
                    'epoch/val_rmse': val_metrics['val_rmse'],
                    'epoch/time': epoch_time
                })

                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    patience_counter = 0
                    self.save_checkpoint(epoch, val_metrics, is_best=True)
                    self.logger.info(f"New best validation loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                    if epoch % self.config['training']['save_freq'] == 0:
                        self.save_checkpoint(epoch, val_metrics)

                if patience_counter >= self.config['training']['patience']:
                    self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            self.logger.error(traceback.format_exc())
            if wandb.run:
                wandb.finish()
            raise

        torch.cuda.empty_cache()
        self.logger.info("Training completed successfully")
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        checkpoint_dir = self.save_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)

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

    def resume_from_checkpoint(self, checkpoint_path: str):
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
    def evaluate(self, test_loader: DataLoader, heat_transport_mean: float):
        self.model.eval()
        predictions = []
        targets = []
        attention_maps = []
        test_losses = []

        self.logger.info("Starting model evaluation...")

        for ssh, sst, attention_mask, target in test_loader:
            ssh = ssh.to(self.device, non_blocking=True)
            sst = sst.to(self.device, non_blocking=True)
            attention_mask = attention_mask.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            try:
                with autocast():
                    output, attn_map = self.model(ssh, sst, attention_mask)
                    loss = self.criterion(output, target)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    self.logger.error(f"OOM in evaluation: {str(e)}")
                raise

            predictions.extend((output.cpu() + heat_transport_mean).numpy())
            targets.extend((target.cpu() + heat_transport_mean).numpy())
            
            curr_batch_size = ssh.size(0)
            attn_numpy = attn_map.cpu().numpy()
            reshaped_attn = attn_numpy.reshape(curr_batch_size, -1, attn_numpy.shape[-2], attn_numpy.shape[-1])
            attention_maps.append(reshaped_attn)
            
            test_losses.append(loss.item())

            del output, attn_map, loss
            torch.cuda.empty_cache()

        predictions = np.array(predictions)
        targets = np.array(targets)
        attention_maps = np.concatenate(attention_maps, axis=0)
        test_losses = np.array(test_losses)

        metrics = {
            'test_loss': np.mean(test_losses),
            'test_loss_std': np.std(test_losses),
            'test_r2': 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - targets.mean()) ** 2),
            'test_rmse': np.sqrt(np.mean((targets - predictions) ** 2)),
            'test_mae': np.mean(np.abs(targets - predictions))
        }

        self.logger.info(f"Test Metrics:")
        for k, v in metrics.items():
            self.logger.info(f"{k}: {v:.4f}")

        wandb.run.summary.update({
            'test/loss': metrics['test_loss'],
            'test/loss_std': metrics['test_loss_std'],
            'test/r2': metrics['test_r2'],
            'test/rmse': metrics['test_rmse'],
            'test/mae': metrics['test_mae']
        })

        viz = OceanVisualizer(self.save_dir)
        self.logger.info("Plotting final predictions...")
        viz.plot_predictions(
            predictions,
            targets,
            save_path='final_predictions'
        )

        self.logger.info("Plotting final attention maps...")
        viz.plot_attention_maps(
            attention_maps,
            self.config.get('data', {}).get('tlat', None),
            self.config.get('data', {}).get('tlong', None),
            save_path='final_attention_maps'
        )

        results = {
            'metrics': metrics,
            'predictions': predictions.tolist(),
            'targets': targets.tolist(),
            'attention_maps': attention_maps.tolist()
        }
        results_path = self.save_dir / 'test_results.pt'
        torch.save(results, results_path)
        wandb.save(str(results_path))

        self.logger.info("Evaluation completed")
        return metrics, predictions, attention_maps