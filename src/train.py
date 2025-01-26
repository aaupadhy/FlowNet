import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from src.data.process_data import OceanDataProcessor, OceanDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
from torch.utils.data.distributed import DistributedSampler
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

def setup_distributed():
    if torch.cuda.device_count() > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return rank, world_size
    return 0, 1

class OceanTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        save_dir: Path,
        device: str = 'cuda'
    ):
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f'cuda:{self.local_rank}')
        if not dist.is_initialized() and torch.cuda.device_count() > 1:
            dist.init_process_group(backend='nccl')
        
        if torch.cuda.device_count() > 1:
            self.model = DDP(model.to(self.device), device_ids=[self.local_rank])
        else:
            self.model = model.to(self.device)
        self.config = config
        self.save_dir = Path(save_dir)
        self.rank, self.world_size = setup_distributed()
        self.logger = logging.getLogger(__name__)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.local_rank == 0:
            self._init_wandb()
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

    def _init_wandb(self):
        run_name = f"ocean_transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            wandb.init(
                project="FlowNet",
                name=run_name,
                config=self.config,
                settings=wandb.Settings(start_method="thread")
            )
            wandb.run.log_code(".")
            self.logger.info(f"Initialized WandB run: {run_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize WandB: {str(e)}")
            raise

    def setup_scheduler(self, train_loader: DataLoader):
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=1e-3,
            epochs=self.config['training']['epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            div_factor=10.0,
            final_div_factor=1000.0
        )
        self.logger.info(f"Initialized OneCycleLR scheduler with {len(train_loader)} steps per epoch")

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
            dataset, [train_size, val_size, test_size]
        )
        train_sampler = DistributedSampler(train_dataset) if torch.cuda.device_count() > 1 else None

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            pin_memory=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size']
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['training']['batch_size']
        )
        
        self.logger.info(f"Created dataloaders - Train: {len(train_loader.dataset)}, "
                        f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
        
        return train_loader, val_loader, test_loader

    @torch.no_grad()
    def get_validation_predictions(self, val_loader):
        self.model.eval()
        predictions = []
        targets = []
        
        for ssh, sst, attention_mask, target in val_loader:
            ssh = ssh.to(self.device)
            sst = sst.to(self.device)
            attention_mask = attention_mask.to(self.device)
            output, _ = self.model(ssh, sst, attention_mask)
            predictions.extend(output.cpu().numpy())
            targets.extend(target.cpu().numpy())
            
        return np.array(predictions), np.array(targets)

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        val_loss = 0
        predictions = []
        targets = []
        batch_times = []
        
        for ssh, sst, attention_mask, target in val_loader:
            batch_start = time.time()
            
            ssh = ssh.to(self.device)
            sst = sst.to(self.device)
            attention_mask = attention_mask.to(self.device)
            target = target.to(self.device)
            
            with autocast():
                output, attention_maps = self.model(ssh, sst, attention_mask)
                loss = self.criterion(output, target)
                
            val_loss += loss.item()
            predictions.extend(output.cpu().numpy())
            targets.extend(target.cpu().numpy())
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
        val_loss = val_loss / len(val_loader)
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        r2 = 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - targets.mean()) ** 2)
        rmse = np.sqrt(np.mean((targets - predictions) ** 2))
        
        return {
            'val_loss': val_loss,
            'val_r2': r2,
            'val_rmse': rmse,
            'val_time': np.mean(batch_times)
        }

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        checkpoint_dir = Path(self.save_dir) / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config,
            'scaler': self.scaler.state_dict(),
            'rng_state': {
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state(),
                'numpy': np.random.get_state()
            }
        }
        
        latest_path = checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)
        self.logger.info(f"Saved checkpoint at epoch {epoch}")
        
        if is_best:
            best_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model with val_loss: {metrics['val_loss']:.6f}")
            wandb.run.summary["best_val_loss"] = metrics['val_loss']
            wandb.run.summary["best_epoch"] = epoch
            wandb.save(str(best_path))
            
        if epoch % self.config['training']['save_freq'] == 0:
            periodic_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, periodic_path)
            wandb.save(str(periodic_path))
            self.logger.info(f"Saved periodic checkpoint at epoch {epoch}")
            
        self._cleanup_old_checkpoints(checkpoint_dir)

    def _cleanup_old_checkpoints(self, checkpoint_dir: Path):
        checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if len(checkpoints) > self.config['training'].get('keep_n_checkpoints', 3):
            checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
            for checkpoint in checkpoints[:-self.config['training']['keep_n_checkpoints']]:
                checkpoint.unlink()
                self.logger.info(f"Removed old checkpoint: {checkpoint}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader, start_epoch: int = 0):
        try:
            # if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            #     train_loader.sampler.set_epoch(epoch)
            if self.scheduler is None:
                self.setup_scheduler(train_loader)
                
            global_step = start_epoch * len(train_loader)
            best_val_loss = float('inf')
            patience_counter = 0
            accumulation_steps = 4
            total_batches = len(train_loader)
            
            self.logger.info(f"Starting training from epoch {start_epoch}")
            self.logger.info(f"Total epochs: {self.config['training']['epochs']}")
            self.logger.info(f"Total batches per epoch: {total_batches}")
            self.logger.info(f"Device: {self.device}")
            self.logger.info(f"Batch size: {self.config['training']['batch_size']}")
            
            wandb.config.update({
                "accumulation_steps": accumulation_steps,
                "total_batches_per_epoch": total_batches,
                "device": str(self.device),
                "patience": self.config['training']['patience']
            })
            
            grad_norms = []
            
            for epoch in range(start_epoch, self.config['training']['epochs']):
                self.model.train()
                epoch_loss = 0
                batch_times = []
                epoch_start = time.time()
                
                self.logger.info(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}")
                
                for i, (ssh, sst, attention_mask, target) in enumerate(train_loader):
                    torch.cuda.empty_cache()
                    batch_start = time.time()
                    
                    try:
                        ssh = ssh.to(self.device)
                        sst = sst.to(self.device)
                        attention_mask = attention_mask.to(self.device)
                        target = target.to(self.device)
                        
                        output, attention_maps = self.model(ssh, sst, attention_mask)
                        loss = self.criterion(output, target)
                        loss = loss / accumulation_steps
                        loss.backward()
                        
                        total_norm = 0
                        for p in self.model.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5
                        grad_norms.append(total_norm)
                        
                        if i % 100 == 0:
                            wandb.log({
                                'gradient/total_norm': total_norm,
                                'gradient/mean_norm': np.mean(grad_norms[-100:])
                            }, step=global_step)
                            
                        batch_loss = loss.item() * accumulation_steps
                        predictions = output.detach().cpu().numpy()
                        targets = target.detach().cpu().numpy()
                        batch_r2 = 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - targets.mean()) ** 2)
                        
                        if i % 10 == 0:
                            self.logger.info(
                                f"Batch {i}/{total_batches} | "
                                f"Loss: {batch_loss:.4f} | "
                                f"R²: {batch_r2:.4f} | "
                                f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
                            )
                            
                        if (i + 1) % accumulation_steps == 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config['training']['grad_clip']
                            )
                            self.optimizer.step()
                            self.optimizer.zero_grad(set_to_none=True)
                            self.scheduler.step()
                            
                        epoch_loss += batch_loss
                        batch_time = time.time() - batch_start
                        batch_times.append(batch_time)
                        
                        wandb.log({
                            'batch/loss': batch_loss,
                            'batch/r2': batch_r2,
                            'batch/learning_rate': self.scheduler.get_last_lr()[0],
                            'batch/time': batch_time,
                            'batch/throughput': len(ssh) / batch_time,
                            'batch/memory_allocated': torch.cuda.memory_allocated() / 1024**3,
                            'batch/memory_cached': torch.cuda.memory_reserved() / 1024**3
                        }, step=global_step)
                        
                        self.monitor.log_batch({
                            'loss': batch_loss,
                            'r2': batch_r2,
                            'lr': self.scheduler.get_last_lr()[0],
                            'batch_time': batch_time,
                            'memory_allocated': torch.cuda.memory_allocated() / 1024**3
                        }, global_step)
                        
                        global_step += 1
                        
                    except Exception as e:
                        self.logger.error(f"Error in batch {i}: {str(e)}")
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
                    f"Time: {epoch_time:.2f}s\n"
                    f"Avg Batch Time: {np.mean(batch_times):.3f}s\n"
                    f"Learning Rate: {self.scheduler.get_last_lr()[0]:.2e}"
                )
                
                wandb.log({
                    'epoch': epoch,
                    'epoch/train_loss': epoch_loss,
                    'epoch/val_loss': val_metrics['val_loss'],
                    'epoch/val_r2': val_metrics['val_r2'],
                    'epoch/val_rmse': val_metrics['val_rmse'],
                    'epoch/time': epoch_time,
                    'epoch/avg_batch_time': np.mean(batch_times),
                    'epoch/learning_rate': self.scheduler.get_last_lr()[0],
                    'epoch/throughput': len(train_loader.dataset) / epoch_time,
                    'epoch/memory_peak': torch.cuda.max_memory_allocated() / 1024**3
                }, step=epoch)
                
                metrics = {
                    'epoch': epoch,
                    'train_loss': epoch_loss,
                    'val_loss': val_metrics['val_loss'],
                    'val_r2': val_metrics['val_r2'],
                    'val_rmse': val_metrics['val_rmse'],
                    'epoch_time': epoch_time,
                    'avg_batch_time': np.mean(batch_times),
                    'memory_allocated': torch.cuda.memory_allocated() / 1024**3
                }
                
                self.monitor.log_epoch(metrics, epoch)
                
                if epoch % 5 == 0:
                    output_dir = Path(self.config['paths']['output_dir'])
                    viz = OceanVisualizer(output_dir)
                    
                    self.logger.info("Plotting attention maps...")
                    viz.plot_attention_maps(
                        attention_maps.detach().cpu().numpy(),
                        self.config['data']['tlat'],
                        self.config['data']['tlong'],
                        save_path=f'epoch_{epoch}'
                    )
                    
                    self.logger.info("Getting validation predictions...")
                    val_preds, val_targets = self.get_validation_predictions(val_loader)
                    
                    self.logger.info("Plotting predictions...")
                    viz.plot_predictions(
                        val_preds,
                        val_targets,
                        save_path=f'epoch_{epoch}'
                    )
                
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    patience_counter = 0
                    self.save_checkpoint(epoch, metrics, is_best=True)
                else:
                    patience_counter += 1
                    if epoch % self.config['training']['save_freq'] == 0:
                        self.save_checkpoint(epoch, metrics)
                        
                if patience_counter >= self.config['training']['patience']:
                    self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
                    
            if self.local_rank == 0:
                self.save_checkpoint(epoch, metrics, is_best=True)
            self.monitor.finish()
            wandb.finish()
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def resume_from_checkpoint(self, checkpoint_path: str):
        self.logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
                    
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.scaler.load_state_dict(checkpoint['scaler'])
        
        torch.set_rng_state(checkpoint['rng_state']['torch'])
        torch.cuda.set_rng_state(checkpoint['rng_state']['cuda'])
        np.random.set_state(checkpoint['rng_state']['numpy'])
        
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
            ssh = ssh.to(self.device)
            sst = sst.to(self.device)
            attention_mask = attention_mask.to(self.device)
            target = target.to(self.device)
            
            with autocast():
                output, attn_map = self.model(ssh, sst, attention_mask)
                loss = self.criterion(output, target)
                
            predictions.extend((output.cpu() + heat_transport_mean).numpy())
            targets.extend((target.cpu() + heat_transport_mean).numpy())
            attention_maps.append(attn_map.detach().cpu().numpy())
            test_losses.append(loss.item())
            
        predictions = np.array(predictions)
        targets = np.array(targets)
        attention_maps = np.concatenate(attention_maps, axis=0)
        
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
        tlat = self.config.get('data', {}).get('tlat', None)
        tlong = self.config.get('data', {}).get('tlong', None)
        viz.plot_attention_maps(
            attention_maps,
            tlat, tlong,
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
        
        self.logger.info("Evaluation completed and results saved")
        return metrics, predictions, attention_maps

    def profile_model(self, loader: DataLoader):
        self.model.eval()
        profile_dir = self.save_dir / 'profiler'
        profile_dir.mkdir(exist_ok=True)
        self.logger.info("Starting model profiling...")
        batch_limit = 5
        
        try:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=2,
                ),
                record_shapes=True,
                with_stack=True
            ) as prof:
                for i, (ssh, sst, attention_mask, _) in enumerate(loader):
                    if i >= batch_limit:
                        break
                        
                    ssh = ssh.to(self.device)
                    sst = sst.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    
                    with autocast():
                        output, attention_maps = self.model(ssh, sst, attention_mask)
                        
                    prof.step()
            
            profile_table = prof.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=10
            )
            
            with open(profile_dir / 'profile_summary.txt', 'w') as f:
                f.write(profile_table)
                
            wandb.run.summary.update({
                'profile/cuda_time_total': prof.key_averages().total_average.cuda_time_total / 1e9,
                'profile/cpu_time_total': prof.key_averages().total_average.cpu_time_total / 1e9
            })
            
            print(profile_table)
            
        except Exception as e:
            self.logger.error(f"Error during profiling: {str(e)}")
            raise
