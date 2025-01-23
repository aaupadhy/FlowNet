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
   def __init__(self, spatial_size: Tuple[int, int], d_model: int = 256, nhead: int = 8,
               num_layers: int = 4, dim_feedforward: int = 512, dropout: float = 0.1):
       super().__init__()
       height, width = spatial_size
       self.d_model = d_model
       self.logger = logging.getLogger(__name__)
       self.dropout = nn.Dropout(dropout)

       h1, w1 = (height + 2*1 - 3) // 2 + 1, (width + 2*1 - 3) // 2 + 1  
       h2, w2 = (h1 + 2*1 - 3) // 2 + 1, (w1 + 2*1 - 3) // 2 + 1        
       h3, w3 = (h2 + 2*1 - 3) // 2 + 1, (w2 + 2*1 - 3) // 2 + 1        
       h4, w4 = (h3 + 2*1 - 3) // 2 + 1, (w3 + 2*1 - 3) // 2 + 1

       self.logger.info(f"Spatial dimensions after convolutions: {h4}x{w4}")

       self.ssh_encoder = nn.Sequential(
           nn.Conv2d(1, 32, 3, stride=2, padding=1),
           nn.BatchNorm2d(32),
           nn.ReLU(),
           nn.Conv2d(32, 48, 3, stride=2, padding=1),
           nn.BatchNorm2d(48),
           nn.ReLU(),
           nn.Conv2d(48, 64, 3, stride=2, padding=1),
           nn.BatchNorm2d(64),
           nn.ReLU(),
           nn.Conv2d(64, 64, 3, stride=2, padding=1),
           nn.BatchNorm2d(64),
           nn.ReLU()
       )

       self.sst_encoder = nn.Sequential(
           nn.Conv2d(1, 32, 3, stride=2, padding=1),
           nn.BatchNorm2d(32),
           nn.ReLU(),
           nn.Conv2d(32, 48, 3, stride=2, padding=1),
           nn.BatchNorm2d(48),
           nn.ReLU(),
           nn.Conv2d(48, 64, 3, stride=2, padding=1),
           nn.BatchNorm2d(64),
           nn.ReLU(),
           nn.Conv2d(64, 64, 3, stride=2, padding=1),
           nn.BatchNorm2d(64),
           nn.ReLU()
       )

       self.ssh_norm = nn.LayerNorm(64)
       self.sst_norm = nn.LayerNorm(64)
       
       self.fusion = MultiModalFusion(64, 64, d_model)
       self.fusion_norm = nn.LayerNorm(d_model)
       
       self.pos_encoder = SpatialPositionalEncoding(d_model, h4, w4, dropout)

       encoder_layer = nn.TransformerEncoderLayer(
           d_model=d_model,
           nhead=nhead,
           dim_feedforward=dim_feedforward,
           dropout=dropout,
           batch_first=True,
           norm_first=True
       )
       self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
       self.final_norm = nn.LayerNorm(d_model)
       
       self.pre_fc = nn.Sequential(
           nn.Linear(d_model, d_model // 2),
           nn.ReLU(),
           nn.LayerNorm(d_model // 2),
           nn.Dropout(dropout)
       )
       self.fc = nn.Sequential(nn.Linear(d_model // 2, d_model // 4),
        nn.ReLU(),
        nn.Linear(d_model // 4, 1))

   def _get_transformer_output_and_attention(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
       attention_maps = []
       encoded = x
       
       for layer in self.transformer_encoder.layers:
           self_attn = layer.self_attn
           attn_output, attn_weights = self_attn(encoded, encoded, encoded, need_weights=True)
           attention_maps.append(attn_weights)
           encoded = layer.norm1(encoded + self.dropout(attn_output))
           encoded = layer.norm2(encoded + self.dropout(layer.linear2(layer.activation(layer.linear1(encoded)))))
       
       return encoded, torch.stack(attention_maps)

   def forward(self, ssh: torch.Tensor, sst: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
       def log_tensor_stats(tensor, name):
           with torch.no_grad():
               if torch.isnan(tensor).any():
                   self.logger.warning(f"{name} contains NaN values!")
               self.logger.info(f"{name} stats - Mean: {tensor.mean():.6f}, Std: {tensor.std():.6f}, Min: {tensor.min():.6f}, Max: {tensor.max():.6f}")
       
       self.logger.info(f"Input shapes - SSH: {ssh.shape}, SST: {sst.shape}")
       log_tensor_stats(ssh, "Input SSH")
       log_tensor_stats(sst, "Input SST")
       
       if ssh.dim() == 3:
           ssh = ssh.unsqueeze(1)
           sst = sst.unsqueeze(1)
       
       ssh_feat = self.ssh_encoder(ssh)
       log_tensor_stats(ssh_feat, "SSH after encoder")
       
       sst_feat = self.sst_encoder(sst)
       log_tensor_stats(sst_feat, "SST after encoder")
       
       B, C, H, W = ssh_feat.shape
       ssh_feat = self.ssh_norm(ssh_feat.permute(0, 2, 3, 1))
       log_tensor_stats(ssh_feat, "SSH after norm")
       
       sst_feat = self.sst_norm(sst_feat.permute(0, 2, 3, 1))
       log_tensor_stats(sst_feat, "SST after norm")
       
       fused = self.fusion(ssh_feat, sst_feat)
       log_tensor_stats(fused, "After fusion")
       
       fused = self.fusion_norm(fused)
       log_tensor_stats(fused, "After fusion norm")
       
       fused = self.pos_encoder(fused)
       log_tensor_stats(fused, "After positional encoding")
       
       fused = fused.reshape(B, H * W, self.d_model)
       log_tensor_stats(fused, "Before transformer")
       
       encoded, attention_maps = self._get_transformer_output_and_attention(fused, mask)
       log_tensor_stats(encoded, "After transformer")
       log_tensor_stats(attention_maps, "Attention maps")
       
       out = self.final_norm(encoded.mean(dim=1))
       log_tensor_stats(out, "After final norm")
       
       out = self.pre_fc(out)
       log_tensor_stats(out, "After pre_fc")
       
       out = self.fc(out)
       log_tensor_stats(out, "Final output")
       
       return out.squeeze(-1), attention_maps
       
   @torch.no_grad()
   def get_attention_maps(self, ssh: torch.Tensor, sst: torch.Tensor) -> torch.Tensor:
       _, attention_maps = self.forward(ssh, sst)
       return attention_maps