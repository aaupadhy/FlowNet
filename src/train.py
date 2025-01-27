import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, random_split
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

def setup_random_seeds(rank):
    seed = 42
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    np.random.seed(seed + rank)

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
        self.rank, self.world_size = setup_distributed()
        setup_random_seeds(self.rank)
        if self.world_size > 1:
            model = model.to(self.device)
            self.model = DDP(model,
                           device_ids=[self.local_rank],
                           gradient_as_bucket_view=True,
                           find_unused_parameters=False)
        else:
            self.model = model.to(self.device)
        self.sync_model()
        self.config = config
        self.save_dir = Path(save_dir)
        self.logger = logging.getLogger(__name__)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        if self.rank == 0:
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
        if self.rank == 0:
            self.logger.info(f"Initialized OneCycleLR scheduler with {steps_per_epoch} steps per epoch")


    def sync_model(self):
        if self.world_size > 1:
            for param in self.model.parameters():
                dist.broadcast(param.data, src=0)

    def create_dataloaders(self, ssh_data, sst_data, heat_transport_data, heat_transport_mean,
                          ssh_mean, ssh_std, sst_mean, sst_std, shape):
        debug_mode = self.config['training'].get('debug_mode', False)
        debug_samples = self.config['training'].get('debug_samples', 32)
        
        dataset = OceanDataset(
            ssh_data, sst_data, heat_transport_data, heat_transport_mean,
            ssh_mean, ssh_std, sst_mean, sst_std, shape,
            debug=debug_mode
        )

        if self.world_size > 1:
            generator = torch.Generator()
            generator.manual_seed(self.config.get('seed', 42))
        else:
            generator = None

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
            generator=generator
        )

        global_batch_size = self.config['training']['batch_size']
        local_batch_size = global_batch_size // self.world_size if self.world_size > 1 else global_batch_size

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            seed=self.config.get('seed', 42)
        ) if self.world_size > 1 else None

        train_loader = DataLoader(
            train_dataset,
            batch_size=local_batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
            drop_last=True,
            prefetch_factor=2
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=local_batch_size,
            pin_memory=True,
            num_workers=2
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=local_batch_size,
            pin_memory=True,
            num_workers=2
        )

        if self.rank == 0:
            self.logger.info(f"Created dataloaders - Train: {len(train_loader.dataset)}, "
                           f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
            self.logger.info(f"Global batch size: {global_batch_size}, Local batch size: {local_batch_size}")

        return train_loader, val_loader, test_loader

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        val_loss = 0
        predictions = []
        targets = []
        batch_times = []

        for ssh, sst, attention_mask, target in val_loader:
            batch_start = time.time()
            ssh = ssh.to(self.device, non_blocking=True)
            sst = sst.to(self.device, non_blocking=True)
            attention_mask = attention_mask.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            try:
                with autocast():
                    output, attention_maps = self.model(ssh, sst, attention_mask)
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

            # Clear cache after each batch
            del output, attention_maps, loss
            torch.cuda.empty_cache()

        val_loss = val_loss / len(val_loader)
        predictions = np.array(predictions)
        targets = np.array(targets)

        if self.world_size > 1:
            val_metrics = torch.tensor([val_loss, len(predictions)], device=self.device)
            dist.all_reduce(val_metrics)
            val_loss = val_metrics[0].item() / self.world_size
            
            all_predictions = [None for _ in range(self.world_size)]
            all_targets = [None for _ in range(self.world_size)]
            
            dist.all_gather_object(all_predictions, predictions.tolist())
            dist.all_gather_object(all_targets, targets.tolist())
            
            predictions = np.concatenate([np.array(p) for p in all_predictions])
            targets = np.concatenate([np.array(t) for t in all_targets])

        r2 = 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - targets.mean()) ** 2)
        rmse = np.sqrt(np.mean((targets - predictions) ** 2))

        return {
            'val_loss': val_loss,
            'val_r2': r2,
            'val_rmse': rmse,
            'val_time': np.mean(batch_times)
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader, start_epoch: int = 0):
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            for dev in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(dev)
        try:
            if self.scheduler is None:
                self.setup_scheduler(train_loader)

            best_val_loss = float('inf')
            patience_counter = 0
            accumulation_steps = self.config['training'].get('accumulation_steps', 4)
            total_batches = len(train_loader)

            if self.rank == 0:
                self.logger.info(f"Starting training from epoch {start_epoch}")
                self.logger.info(f"Total epochs: {self.config['training']['epochs']}")
                self.logger.info(f"Total batches per epoch: {total_batches}")
                self.logger.info(f"Device: {self.device}")
                self.logger.info(f"Batch size: {self.config['training']['batch_size']}")

            for epoch in range(start_epoch, self.config['training']['epochs']):
                if hasattr(train_loader.sampler, 'set_epoch'):
                    train_loader.sampler.set_epoch(epoch)
                    
                if self.world_size > 1:
                    dist.barrier() 
                if train_loader.sampler and isinstance(train_loader.sampler, DistributedSampler):
                    train_loader.sampler.set_epoch(epoch)

                self.model.train()
                epoch_loss = 0
                batch_times = []
                epoch_start = time.time()

                if self.rank == 0:
                    self.logger.info(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}")

                for i, (ssh, sst, attention_mask, target) in enumerate(train_loader):
                    try:
                        batch_start = time.time()
                        
                        # Move data to GPU with non_blocking=True
                        ssh = ssh.to(self.device, non_blocking=True)
                        sst = sst.to(self.device, non_blocking=True)
                        attention_mask = attention_mask.to(self.device, non_blocking=True)
                        target = target.to(self.device, non_blocking=True)

                        # Clear gradients
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

                        if self.world_size > 1:
                            torch.distributed.barrier() 
                        self.scaler.scale(loss).backward()

                        if (i + 1) % accumulation_steps == 0:
                            if dist.is_initialized():
                                dist.barrier()

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

                        # Logging for rank 0
                        if self.rank == 0 and i % 10 == 0:
                            with torch.no_grad():
                                predictions = output.detach().cpu().numpy()
                                targets = target.detach().cpu().numpy()
                                batch_r2 = 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - targets.mean()) ** 2)
                                
                                self.logger.info(
                                    f"Batch {i}/{total_batches} | "
                                    f"Loss: {batch_loss:.4f} | "
                                    f"R²: {batch_r2:.4f} | "
                                    f"LR: {self.scheduler.get_last_lr()[0]:.2e} | "
                                    f"Time: {batch_time:.2f}s"
                                )

                                wandb.log({
                                    'batch/loss': batch_loss,
                                    'batch/r2': batch_r2,
                                    'batch/learning_rate': self.scheduler.get_last_lr()[0],
                                    'batch/time': batch_time,
                                    'batch/memory_allocated': torch.cuda.memory_allocated() / 1024**3
                                })

                        # Clear memory after each batch
                        del output, attention_maps, loss
                        torch.cuda.empty_cache()

                    except Exception as e:
                        self.logger.error(f"Error in batch {i}: {str(e)}")
                        raise

                if self.world_size > 1:
                    epoch_loss_tensor = torch.tensor([epoch_loss], device=self.device)
                    dist.all_reduce(epoch_loss_tensor)
                    epoch_loss = epoch_loss_tensor.item() / self.world_size

                if self.rank == 0:
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
                    else:
                        patience_counter += 1
                        if epoch % self.config['training']['save_freq'] == 0:
                            self.save_checkpoint(epoch, val_metrics)

                    if patience_counter >= self.config['training']['patience']:
                        self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break

                # Synchronize at the end of each epoch
                if self.world_size > 1:
                    dist.barrier()

        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            self.logger.error(traceback.format_exc())
            if self.rank == 0 and wandb.run:
                wandb.finish()
            raise

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        if self.rank != 0:
            return

        checkpoint_dir = self.save_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict(),
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
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.local_rank}
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
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

        if self.rank == 0:
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

        if self.rank == 0:
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
            attention_maps.append(attn_map.cpu().numpy())
            test_losses.append(loss.item())

            # Clear memory after each batch
            del output, attn_map, loss
            torch.cuda.empty_cache()

        if self.world_size > 1:
            all_predictions = [None for _ in range(self.world_size)]
            all_targets = [None for _ in range(self.world_size)]
            all_attention_maps = [None for _ in range(self.world_size)]
            all_losses = [None for _ in range(self.world_size)]

            dist.all_gather_object(all_predictions, predictions)
            dist.all_gather_object(all_targets, targets)
            dist.all_gather_object(all_attention_maps, attention_maps)
            dist.all_gather_object(all_losses, test_losses)

            predictions = np.concatenate(all_predictions)
            targets = np.concatenate(all_targets)
            attention_maps = np.concatenate(all_attention_maps)
            test_losses = np.concatenate(all_losses)
        else:
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

        if self.rank == 0:
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

        if self.rank == 0:
            self.logger.info("Evaluation completed")
        
        return metrics, predictions, attention_maps