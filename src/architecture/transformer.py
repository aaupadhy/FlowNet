import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import math
from typing import Optional, Tuple, Union
from torch.cuda.amp import autocast
import logging

class SpatialPositionalEncoding(nn.Module):
   def __init__(self, d_model: int, max_h: int, max_w: int, dropout: float = 0.1):
       super().__init__()
       self.dropout = nn.Dropout(p=dropout)
       pe = torch.zeros(1, max_h, max_w, d_model)
       h_pos = torch.arange(0, max_h).unsqueeze(1).unsqueeze(2).float()
       w_pos = torch.arange(0, max_w).unsqueeze(0).unsqueeze(2).float()
       div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
       pe[0, :, :, 0::2] = torch.sin(h_pos * div_term)
       pe[0, :, :, 1::2] = torch.cos(h_pos * div_term)
       pe[0, :, :, 0::2] += torch.sin(w_pos * div_term)
       pe[0, :, :, 1::2] += torch.cos(w_pos * div_term)
       self.register_buffer('pe', pe)

   def forward(self, x: torch.Tensor) -> torch.Tensor:
       x = x + self.pe[:, :x.size(1), :x.size(2)]
       return self.dropout(x)

class MultiModalFusion(nn.Module):
    def __init__(self, ssh_dim: int, sst_dim: int, fusion_dim: int):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        mid_dim = fusion_dim // 2
        self.ssh_proj = nn.Sequential(
            nn.Linear(ssh_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, fusion_dim)
        )
        self.sst_proj = nn.Sequential(
            nn.Linear(sst_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, fusion_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU()
        )

    def forward(self, ssh_feat: torch.Tensor, sst_feat: torch.Tensor) -> torch.Tensor:
        def log_tensor_stats(tensor, name):
            with torch.no_grad():
                if torch.isnan(tensor).any():
                    self.logger.warning(f"{name} contains NaN values!")
                self.logger.info(f"{name} stats - Mean: {tensor.mean():.6f}, Std: {tensor.std():.6f}, Min: {tensor.min():.6f}, Max: {tensor.max():.6f}")
        
        ssh_proj = self.ssh_proj(ssh_feat)
        log_tensor_stats(ssh_proj, "SSH after projection")
        
        sst_proj = self.sst_proj(sst_feat)
        log_tensor_stats(sst_proj, "SST after projection")
        
        fused = self.fusion(torch.cat([ssh_proj, sst_proj], dim=-1)) + 0.1*(ssh_proj + sst_proj)
        log_tensor_stats(fused, "After fusion")
        
        return fused

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
            nn.GELU(),
            nn.Identity()
        )
        
        self.sst_encoder = nn.Sequential(
            nn.Conv2d(1, d_model//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model//4),
            nn.GELU(),
            nn.Conv2d(d_model//4, d_model//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model//2),
            nn.GELU(),
            nn.Identity()
        )

        self.ssh_norm = nn.LayerNorm([d_model//2, *[s//2 for s in spatial_size]])
        self.sst_norm = nn.LayerNorm([d_model//2, *[s//2 for s in spatial_size]])
        
        self.ssh_proj = nn.Sequential(
            nn.Linear(d_model//2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        self.sst_proj = nn.Sequential(
            nn.Linear(d_model//2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        self.fusion = nn.Sequential(
            nn.Linear(2*d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, spatial_size[0]*spatial_size[1]//4, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True,
            norm_first=True 
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.pre_fc = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.LayerNorm(d_model//2),
            nn.Dropout(dropout/2)
        )
        
        self.final = nn.Linear(d_model//2, 1)
        
        self._init_weights()

    def _init_weights(self):
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

            # Encoder paths
            ssh_encoded = self.ssh_encoder(ssh)
            sst_encoded = self.sst_encoder(sst)
            
            self._log_tensor_stats(ssh_encoded, "SSH after encoder")
            self._log_tensor_stats(sst_encoded, "SST after encoder")

            # Apply layer normalization
            ssh = self.ssh_norm(ssh_encoded)
            sst = self.sst_norm(sst_encoded)
            
            self._log_tensor_stats(ssh, "SSH after norm")
            self._log_tensor_stats(sst, "SST after norm")

            # Reshape and project
            batch_size = ssh.shape[0]
            ssh = ssh.flatten(2).transpose(1, 2)
            sst = sst.flatten(2).transpose(1, 2)
            
            ssh = self.ssh_proj(ssh)
            sst = self.sst_proj(sst)
            
            self._log_tensor_stats(ssh, "SSH after projection")
            self._log_tensor_stats(sst, "SST after projection")

            # Concatenate and fuse modalities
            x = torch.cat([ssh, sst], dim=-1)
            x = self.fusion(x)
            
            self._log_tensor_stats(x, "After fusion")
            x = F.layer_norm(x, [x.size(-1)])
            self._log_tensor_stats(x, "After fusion norm")

            # Add positional embeddings
            x = x + self.pos_embedding[:, :x.size(1)]
            self._log_tensor_stats(x, "After positional encoding")
            
            # Apply transformer
            self._log_tensor_stats(x, "Before transformer")
            if self.training:
                x = torch.utils.checkpoint.checkpoint(self.transformer, x, use_reentrant=False)
            else:
                x = self.transformer(x)
            self._log_tensor_stats(x, "After transformer")

            # Compute attention weights from transformer output
            attention_weights = F.softmax(x @ x.transpose(-2, -1) / np.sqrt(x.size(-1)), dim=-1)
            self._log_tensor_stats(attention_weights, "Attention maps")
            
            # Apply attention pooling
            x = torch.matmul(attention_weights, x)
            
            # Global pooling
            x = x.mean(dim=1)
            x = F.layer_norm(x, [x.size(-1)])
            self._log_tensor_stats(x, "After final norm")

            # Final prediction layers
            x = self.pre_fc(x)
            self._log_tensor_stats(x, "After pre_fc")
            
            x = self.final(x)
            self._log_tensor_stats(x, "Final output")

            return x, attention_weights

    def get_attention_map(self):
        """Returns the attention map from the last forward pass"""
        if not hasattr(self, '_last_attention'):
            raise RuntimeError("No attention map available. Run forward pass first.")
        return self._last_attention