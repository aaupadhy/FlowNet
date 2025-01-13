import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import math
from typing import Optional, Tuple
from torch.cuda.amp import autocast
import logging


class SpatialPositionalEncoding(nn.Module):
   def __init__(self, d_model: int, max_h: int, max_w: int, dropout: float = 0.1):
       super().__init__()
       self.dropout = nn.Dropout(p=dropout)
       
       # Compute position encoding once and cache it
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
       ssh_proj = self.ssh_proj(ssh_feat)
       sst_proj = self.sst_proj(sst_feat)
       return self.fusion(torch.cat([ssh_proj, sst_proj], dim=-1))


class OceanTransformer(nn.Module):
   def __init__(self, spatial_size: Tuple[int, int], d_model: int = 256, nhead: int = 8,
               num_layers: int = 4, dim_feedforward: int = 512, dropout: float = 0.1):
       super().__init__()
       height, width = spatial_size
       self.d_model = d_model
       self.logger = logging.getLogger(__name__)
       self.logger.info(f"Initializing transformer with spatial size: {spatial_size}")

       # Calculate reduced dimensions after each conv layer
       h1, w1 = (height + 2*1 - 3) // 2 + 1, (width + 2*1 - 3) // 2 + 1
       h2, w2 = (h1 + 2*1 - 3) // 2 + 1, (w1 + 2*1 - 3) // 2 + 1
       h3, w3 = (h2 + 2*1 - 3) // 2 + 1, (w2 + 2*1 - 3) // 2 + 1
       
       self.logger.info(f"Spatial dimensions after convolutions: {h3}x{w3}")

       # Downsample with strided convs - using smaller stride for Atlantic region
       self.ssh_encoder = nn.Sequential(
           nn.Conv2d(1, 32, 3, stride=2, padding=1),
           nn.BatchNorm2d(32),
           nn.ReLU(),
           nn.Conv2d(32, 64, 3, stride=2, padding=1), 
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
           nn.Conv2d(32, 64, 3, stride=2, padding=1),
           nn.BatchNorm2d(64),
           nn.ReLU(),
           nn.Conv2d(64, 64, 3, stride=2, padding=1),
           nn.BatchNorm2d(64),
           nn.ReLU()
       )

       self.fusion = MultiModalFusion(64, 64, d_model)
       self.pos_encoder = SpatialPositionalEncoding(d_model, h3, w3, dropout)

       encoder_layer = nn.TransformerEncoderLayer(
           d_model=d_model,
           nhead=nhead,
           dim_feedforward=dim_feedforward,
           dropout=dropout,
           batch_first=True
       )
       self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

       self.final_norm = nn.LayerNorm(d_model)
       self.fc = nn.Linear(d_model, 1)
       self.dropout = nn.Dropout(dropout)
       self.num_layers = num_layers
       self.nhead = nhead

   def _get_transformer_output_and_attention(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
       attention_maps = []
       encoded = x
       
       for layer in self.transformer_encoder.layers:
           # Get the MultiheadAttention module
           self_attn = layer.self_attn
           
           # Use the forward method directly to get attention weights
           attn_output, attn_weights = self_attn(encoded, encoded, encoded, need_weights=True)
           attention_maps.append(attn_weights)
           
           # Apply attention output
           encoded = layer.norm1(encoded + self.dropout(attn_output))
           encoded = layer.norm2(encoded + self.dropout(layer.linear2(layer.activation(layer.linear1(encoded)))))
       
       return encoded, torch.stack(attention_maps)

   def _forward(self, ssh: torch.Tensor, sst: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
       self.logger.info(f"Input shapes - SSH: {ssh.shape}, SST: {sst.shape}")
       
       if ssh.dim() == 3:
           ssh = ssh.unsqueeze(1)
           sst = sst.unsqueeze(1)
       self.logger.info(f"After unsqueeze - SSH: {ssh.shape}, SST: {sst.shape}")
       
       # Encode inputs
       ssh_feat = self.ssh_encoder(ssh)
       sst_feat = self.sst_encoder(sst)
       self.logger.info(f"After encoding - SSH: {ssh_feat.shape}, SST: {sst_feat.shape}")
       
       B, C, H, W = ssh_feat.shape
       ssh_feat = ssh_feat.permute(0, 2, 3, 1)
       sst_feat = sst_feat.permute(0, 2, 3, 1)
       self.logger.info(f"After reshape - SSH: {ssh_feat.shape}, SST: {sst_feat.shape}")
       
       # Fuse features
       fused = self.fusion(ssh_feat, sst_feat)
       self.logger.info(f"After fusion: {fused.shape}")
       
       fused = self.pos_encoder(fused)
       
       # Reshape for attention
       fused = fused.reshape(B, H * W, self.d_model)
       self.logger.info(f"Before transformer: {fused.shape}")
       
       # Get both output and attention maps
       encoded, attention_maps = self._get_transformer_output_and_attention(fused, mask)
       self.logger.info(f"After transformer: {encoded.shape}")
       self.logger.info(f"Attention maps shape: {attention_maps.shape}")
       
       out = self.final_norm(encoded.mean(dim=1))
       self.logger.info(f"After pooling: {out.shape}")
       
       out = self.fc(out)
       self.logger.info(f"Final output: {out.shape}")
       
       # Reshape attention maps to (batch, num_layers, heads, height, width)
    #    attention_maps = attention_maps.view(B, self.num_layers, self.nhead, H, W)
       
       return out.squeeze(-1), None

   def forward(self, ssh, sst, mask=None):
       return self._forward(ssh, sst, mask)

   @torch.no_grad()
   def get_attention_maps(self, ssh: torch.Tensor, sst: torch.Tensor) -> torch.Tensor:
       _, attention_maps = self.forward(ssh, sst)
       return attention_maps