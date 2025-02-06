import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import math
from typing import Optional, Tuple, Union
from torch.cuda.amp import autocast
import logging
import numpy as np

class MultiModalFusion(nn.Module):
   def __init__(self, d_model: int):
       super().__init__()
       self.logger = logging.getLogger(__name__)
       self.d_model = d_model
       
       self.ssh_proj = nn.Sequential(
           nn.Linear(d_model, d_model),
           nn.ReLU(),
           nn.LayerNorm(d_model)
       )
       
       self.sst_proj = nn.Sequential(
           nn.Linear(d_model, d_model),
           nn.ReLU(),
           nn.LayerNorm(d_model)
       )
       
       self.fusion = nn.Sequential(
           nn.Linear(2 * d_model, d_model),
           nn.LayerNorm(d_model),
           nn.ReLU()
       )
       
   def forward(self, ssh_feat: torch.Tensor, sst_feat: torch.Tensor) -> torch.Tensor:
       ssh_proj = self.ssh_proj(ssh_feat)
       sst_proj = self.sst_proj(sst_feat)
       fused = self.fusion(torch.cat([ssh_proj, sst_proj], dim=-1))
       return fused

class SpatialPositionalEncoding(nn.Module):
   def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
       super().__init__()
       self.dropout = nn.Dropout(p=dropout)
       pe = torch.zeros(1, max_len, d_model)
       position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
       div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
       pe[0, :, 0::2] = torch.sin(position * div_term)
       pe[0, :, 1::2] = torch.cos(position * div_term)
       self.register_buffer('pe', pe)
       
   def forward(self, x: torch.Tensor) -> torch.Tensor:
       x = x + self.pe[:, :x.size(1)]
       return self.dropout(x)

class OceanTransformer(nn.Module):
   def __init__(self, spatial_size, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
       super().__init__()
       self.logger = logging.getLogger(__name__)
       self.d_model = d_model
       
       self.ssh_encoder = nn.Sequential(
           nn.Conv2d(1, d_model//4, kernel_size=3, padding=1),
           nn.BatchNorm2d(d_model//4),
           nn.GELU(),
           nn.Conv2d(d_model//4, d_model//2, kernel_size=3, padding=1),
           nn.BatchNorm2d(d_model//2),
           nn.GELU()
       )
       
       self.sst_encoder = nn.Sequential(
           nn.Conv2d(1, d_model//4, kernel_size=3, padding=1),
           nn.BatchNorm2d(d_model//4),
           nn.GELU(),
           nn.Conv2d(d_model//4, d_model//2, kernel_size=3, padding=1),
           nn.BatchNorm2d(d_model//2),
           nn.GELU()
       )
       
       self.ssh_norm = nn.LayerNorm([d_model//2, spatial_size[0]//2, spatial_size[1]//2])
       self.sst_norm = nn.LayerNorm([d_model//2, spatial_size[0]//2, spatial_size[1]//2])
       
       patch_size = 16
       self.patch_dim = (d_model // 2) * patch_size * patch_size
       self.ssh_proj = nn.Linear(self.patch_dim, d_model)
       self.sst_proj = nn.Linear(self.patch_dim, d_model)
       
       h = (spatial_size[0] + patch_size - 1) // patch_size
       w = (spatial_size[1] + patch_size - 1) // patch_size
       n_patches = h * w
       
       self.fusion = MultiModalFusion(d_model)
       self.pos_encoding = SpatialPositionalEncoding(d_model, n_patches * 2, dropout)
       
       encoder_layer = nn.TransformerEncoderLayer(
           d_model=d_model,
           nhead=nhead,
           dim_feedforward=dim_feedforward,
           dropout=dropout,
           activation=F.gelu,
           batch_first=True,
           norm_first=True
       )
       self.transformer = nn.TransformerEncoder(encoder_layer, num_layers, enable_nested_tensor=False)
       
       self.pre_fc = nn.Sequential(
           nn.Linear(d_model, d_model//2),
           nn.GELU(),
           nn.LayerNorm(d_model//2),
           nn.Dropout(dropout/2)
       )
       self.final = nn.Linear(d_model//2, 1)
       self.init_weights()
   
   def init_weights(self):
       for m in self.modules():
           if isinstance(m, (nn.Linear, nn.Conv2d)):
               nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
               if m.bias is not None:
                   nn.init.zeros_(m.bias)
           elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
               nn.init.ones_(m.weight)
               nn.init.zeros_(m.bias)
   
   def _log_tensor_stats(self, tensor, name):
       with torch.no_grad():
           self.logger.info(
               f"{name} stats - "
               f"Mean: {tensor.mean():.6f}, "
               f"Std: {tensor.std():.6f}, "
               f"Min: {tensor.min():.6f}, "
               f"Max: {tensor.max():.6f}"
           )

   def forward(self, ssh, sst, attention_mask=None):
       self.logger.info(f"Input shapes - SSH: {ssh.shape}, SST: {sst.shape}")
       self._log_tensor_stats(ssh, "Input SSH")
       self._log_tensor_stats(sst, "Input SST")
       
       ssh_encoded = self.ssh_encoder(ssh)
       sst_encoded = self.sst_encoder(sst)
       
       self._log_tensor_stats(ssh_encoded, "SSH after encoder")
       self._log_tensor_stats(sst_encoded, "SST after encoder")
       
       ssh = self.ssh_norm(ssh_encoded)
       sst = self.sst_norm(sst_encoded)
       
       self._log_tensor_stats(ssh, "SSH after norm")
       self._log_tensor_stats(sst, "SST after norm")
       
       B, C, H, W = ssh.shape
       P = 16
       
       H_pad = ((H + P - 1) // P) * P - H
       W_pad = ((W + P - 1) // P) * P - W
       
       ssh = F.pad(ssh, (0, W_pad, 0, H_pad))
       sst = F.pad(sst, (0, W_pad, 0, H_pad))
       
       H, W = ssh.shape[2], ssh.shape[3]
       
       # Create patches
       ssh = ssh.unfold(2, P, P).unfold(3, P, P)
       ssh = ssh.permute(0, 2, 3, 1, 4, 5).contiguous()
       ssh = ssh.view(B, -1, C * P * P)
       
       sst = sst.unfold(2, P, P).unfold(3, P, P)
       sst = sst.permute(0, 2, 3, 1, 4, 5).contiguous()
       sst = sst.view(B, -1, C * P * P)
       
       # Project patches
       ssh = self.ssh_proj(ssh)
       sst = self.sst_proj(sst)
       
       self._log_tensor_stats(ssh, "SSH after projection")
       self._log_tensor_stats(sst, "SST after projection")
       
       fused_features = self.fusion(ssh, sst)
       self._log_tensor_stats(fused_features, "After fusion")
       
       x = F.layer_norm(fused_features, [fused_features.size(-1)])
       self._log_tensor_stats(x, "After fusion norm")
       
       x = self.pos_encoding(x)
       self._log_tensor_stats(x, "After positional encoding")
       self._log_tensor_stats(x, "Before transformer")
       
       if self.training:
           x = torch.utils.checkpoint.checkpoint(self.transformer, x, use_reentrant=False)
       else:
           x = self.transformer(x)
           
       self._log_tensor_stats(x, "After transformer")

       attention_weights = F.softmax(x @ x.transpose(-2, -1) / np.sqrt(x.size(-1)), dim=-1)
       self._log_tensor_stats(attention_weights, "Attention maps")

       x = torch.matmul(attention_weights, x)
       x = x.mean(dim=1)
       
       x = F.layer_norm(x, [x.size(-1)])
       self._log_tensor_stats(x, "After final norm")
       
       x = self.pre_fc(x)
       self._log_tensor_stats(x, "After pre_fc")
       
       x = self.final(x)
       x = x.squeeze(-1)
       self._log_tensor_stats(x, "Final output")


       with torch.no_grad():
           ssh_attn = F.softmax(ssh @ ssh.transpose(-2, -1) / np.sqrt(ssh.size(-1)), dim=-1)
           sst_attn = F.softmax(sst @ sst.transpose(-2, -1) / np.sqrt(sst.size(-1)), dim=-1)
           cross_attn = F.softmax(ssh @ sst.transpose(-2, -1) / np.sqrt(ssh.size(-1)), dim=-1)
           
           attention_maps = {
               'combined': attention_weights,
               'ssh': ssh_attn,
               'sst': sst_attn,
               'cross': cross_attn
           }
       
       return x, attention_maps