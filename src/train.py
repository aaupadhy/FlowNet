import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
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

class SmoothHuberLoss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
    def forward(self, pred, target):
        return nn.functional.smooth_l1_loss(pred, target, beta=self.beta)

def setup_random_seeds():
    s = 42
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(s)
    np.random.seed(s)

class OceanTrainer:
    def __init__(self, model, config, save_dir, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.save_dir = Path(save_dir)
        self.logger = logging.getLogger(__name__)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        rn = f"ocean_transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb_mode = config.get("wandb_mode", "online")
        if wandb_mode.lower() == "offline":
            wandb.init(project="ocean_heat_transport", name=rn, config=config['training'], mode="offline")
        else:
            if not wandb.run:
                wandb.init(project="ocean_heat_transport", name=rn, config=config['training'])
        self.criterion = SmoothHuberLoss(beta=1.0)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
        self.scheduler = None
        self.scaler = GradScaler()
    def create_dataloaders(self, ssh_data, sst_data, ht_data, ht_mean, ht_std, ssh_mean, ssh_std, sst_mean, sst_std, shape):
        dbg = self.config['training'].get('debug_mode', False)
        dbg_s = self.config['training'].get('debug_samples', 32)
        from src.data.process_data import OceanDataset
        log_target = self.config['training'].get('log_target', False)
        target_scale = self.config['training'].get('target_scale', 10.0)
        ds = OceanDataset(ssh_data, sst_data, ht_data, ht_mean, ht_std, ssh_mean, ssh_std, sst_mean, sst_std, shape, debug=dbg, log_target=log_target, target_scale=target_scale)
        ts = len(ds)
        if dbg:
            tr = dbg_s // 2
            v = dbg_s // 4
            te = dbg_s // 4
        else:
            tr = int(0.7 * ts)
            v = int(0.15 * ts)
            te = ts - tr - v
        g = torch.Generator().manual_seed(42)
        tr_ds, v_ds, te_ds = random_split(ds, [tr, v, te], generator=g)
        bs = self.config['training']['batch_size']
        tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True, drop_last=True)
        va_dl = DataLoader(v_ds, batch_size=bs, pin_memory=True, num_workers=2)
        te_dl = DataLoader(te_ds, batch_size=bs, pin_memory=True, num_workers=2)
        self.logger.info(f"Created dataloaders - Train: {len(tr_dl.dataset)}, Val: {len(va_dl.dataset)}, Test: {len(te_dl.dataset)}")
        return tr_dl, va_dl, te_dl
    def setup_scheduler(self, train_loader):
        sps = len(train_loader) // self.config['training'].get('accumulation_steps', 4)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.config['training']['learning_rate'],
                                                        epochs=self.config['training']['epochs'], steps_per_epoch=sps,
                                                        pct_start=0.3, div_factor=10.0, final_div_factor=1000.0)
        self.logger.info(f"Initialized OneCycleLR scheduler with {sps} steps per epoch")
    def train(self, train_loader, val_loader, start_epoch=0):
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(0)
        try:
            if self.scheduler is None:
                self.setup_scheduler(train_loader)
            best_val = float('inf')
            pat = 0
            acs = self.config['training'].get('accumulation_steps', 8)
            for e in range(start_epoch, self.config['training']['epochs']):
                self.model.train()
                el = 0
                est = time.time()
                for i, (ssh, sst, mask, tgt) in enumerate(train_loader):
                    ssh = ssh.to(self.device)
                    sst = sst.to(self.device)
                    mask = mask.to(self.device)
                    tgt = tgt.to(self.device)
                    self.optimizer.zero_grad(set_to_none=True)
                    with autocast():
                        o, _ = self.model(ssh, sst, mask)
                        ls = self.criterion(o, tgt) / acs
                    self.scaler.scale(ls).backward()
                    if (i + 1) % acs == 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                        if self.scheduler is not None:
                            self.scheduler.step()
                    el += ls.item() * acs
                el /= len(train_loader)
                et = time.time() - est
                v = self.validate(val_loader)
                self.logger.info(f"Epoch {e+1} Summary:\nTrain Loss: {el:.4f}\nVal Loss: {v['val_loss']:.4f}\nVal R²: {v['val_r2']:.4f}\nVal RMSE: {v['val_rmse']:.4f}\nVal MAPE: {v['val_mape']:.4f}\nExplained Var: {v['explained_variance']:.4f}\nTime: {et:.2f}s")
                wandb.log({'epoch': e, 'epoch/train_loss': el, 'epoch/val_loss': v['val_loss'], 'epoch/val_r2': v['val_r2'], 
                           'epoch/val_rmse': v['val_rmse'], 'epoch/val_mape': v['val_mape'], 'epoch/explained_variance': v['explained_variance'], 
                           'epoch/time': et})
                if v['val_loss'] < best_val:
                    best_val = v['val_loss']
                    pat = 0
                    torch.save(self.model.state_dict(), self.save_dir / 'checkpoints' / 'best_model.pt')
                    wandb.run.summary['best_epoch'] = e
                    wandb.run.summary['best_val_loss'] = best_val
                else:
                    pat += 1
                if e % 5 == 0:
                    torch.save(self.model.state_dict(), self.save_dir / f"checkpoints/checkpoint_epoch_{e}.pt")
                if pat > self.config['training']['patience']:
                    break
            self.logger.info("Training completed successfully")
        except Exception as ex:
            self.logger.error(f"Error during training: {ex}")
            self.logger.error(traceback.format_exc())
            raise
    def validate(self, loader):
        self.logger.info("Starting validation...")
        self.model.eval()
        vs = 0
        ct = 0
        ps = []
        ts = []
        with torch.no_grad(), autocast():
            for ssh, sst, m, tgt in loader:
                ssh = ssh.to(self.device)
                sst = sst.to(self.device)
                m = m.to(self.device)
                tgt = tgt.to(self.device)
                o, _ = self.model(ssh, sst, m)
                ls = self.criterion(o, tgt)
                vs += ls.item() * tgt.size(0)
                ct += tgt.size(0)
                ps.extend(o.cpu().numpy())
                ts.extend(tgt.cpu().numpy())
        vs /= ct
        ps = np.array(ps)
        ts = np.array(ts)
        epsilon = 1e-8
        target_var = np.var(ts) + epsilon
        explained_variance = 1 - np.var(ts - ps) / target_var
        rm = np.sqrt(np.mean((ps - ts) ** 2))
        mape = np.mean(np.abs((ps - ts) / np.where(ts == 0, 1, ts))) * 100
        rr = 1.0 - np.sum((ps - ts) ** 2) / (np.sum((ts - np.mean(ts)) ** 2) + epsilon)
        self.logger.info(f"Validation Summary: Loss: {vs:.4f}, R²: {rr:.4f}, RMSE: {rm:.4f}, MAPE: {mape:.4f}, Explained Var: {explained_variance:.4f}")
        return {'val_loss': vs, 'val_r2': rr, 'val_rmse': rm, 'val_mape': mape, 'explained_variance': explained_variance}
    def evaluate(self, loader, ht_mean, ht_std):
        self.logger.info("Starting model evaluation...")
        self.model.eval()
        ps = []
        ts = []
        attn_maps_list = []
        with torch.no_grad(), autocast():
            for ssh, sst, m, tgt in loader:
                ssh = ssh.to(self.device)
                sst = sst.to(self.device)
                m = m.to(self.device)
                tgt = tgt.to(self.device)
                o, out_dict = self.model(ssh, sst, m)
                ps.extend(o.cpu().numpy())
                ts.extend(tgt.cpu().numpy())
                if not attn_maps_list:
                    attn_maps_list = out_dict.get('attn', [])
        ps = np.array(ps)
        ts = np.array(ts)
        if self.config['training'].get('log_target', False):
            if hasattr(loader.dataset, 'target_scale'):
                t_scale = loader.dataset.target_scale
            elif hasattr(loader.dataset, 'dataset'):
                t_scale = loader.dataset.dataset.target_scale
            else:
                raise AttributeError("Cannot find target_scale in dataset")
            ps = np.exp(np.clip(ps * t_scale, -100, 100))
            ts = np.exp(np.clip(ts * t_scale, -100, 100))
        epsilon = 1e-8
        e = (ps - ts) ** 2
        ms = np.mean(e)
        sd = np.std(e)
        meae = np.mean(np.abs(ps - ts))
        target_var = np.var(ts) + epsilon
        explained_variance = 1 - np.var(ts - ps) / target_var
        rm = np.sqrt(np.mean((ps - ts) ** 2))
        mape = np.mean(np.abs((ps - ts) / np.where(ts == 0, 1, ts))) * 100
        rr = 1.0 - np.sum((ps - ts) ** 2) / (np.sum((ts - np.mean(ts)) ** 2) + epsilon)
        self.logger.info("Test Metrics:")
        self.logger.info(f"test_loss: {ms:.4f}")
        self.logger.info(f"test_loss_std: {sd:.4f}")
        self.logger.info(f"test_r2: {rr:.4f}")
        self.logger.info(f"test_rmse: {rm:.4f}")
        self.logger.info(f"test_mae: {meae:.4f}")
        self.logger.info(f"test_mape: {mape:.4f}")
        self.logger.info(f"test_explained_variance: {explained_variance:.4f}")
        return {'loss': ms, 'loss_std': sd, 'mae': meae, 'r2': rr, 'rmse': rm, 'mape': mape, 'explained_variance': explained_variance}, ps, ts, attn_maps_list
    def plot_attention_maps(self, sample_ssh, sample_sst, sample_mask):
        self.model.eval()
        with torch.no_grad(), autocast():
            out, out_dict = self.model(sample_ssh.to(self.device), sample_sst.to(self.device), sample_mask.to(self.device))
        return out_dict.get('attn', [])

