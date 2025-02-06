import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import math
from typing import Optional, Tuple, Union
from torch.cuda.amp import autocast
import logging
import numpy as np

class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H', W']
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        return x

class MultiModalFusion(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.3):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    def forward(self, ssh_feat: torch.Tensor, sst_feat: torch.Tensor) -> torch.Tensor:
        fused = self.fusion(torch.cat([ssh_feat, sst_feat], dim=-1))
        return fused

class SpatialPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.3):
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
    def __init__(self, spatial_size, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.3):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.d_model = d_model
        # Encoder for SSH and SST
        self.ssh_encoder = nn.Sequential(
            nn.Conv2d(1, d_model//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model//4),
            nn.GELU(),
            nn.Conv2d(d_model//4, d_model//2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(d_model//2),
            nn.GELU()
        )
        self.sst_encoder = nn.Sequential(
            nn.Conv2d(1, d_model//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model//4),
            nn.GELU(),
            nn.Conv2d(d_model//4, d_model//2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(d_model//2),
            nn.GELU()
        )
        # Compute encoder output dimensions using convolution formula:
        # output = floor((L + 2*padding - kernel_size)/stride + 1)
        def conv_out_dim(L):
            return (L + 2 - 3) // 2 + 1
        h_conv = conv_out_dim(spatial_size[0])
        w_conv = conv_out_dim(spatial_size[1])
        self.ssh_norm = nn.LayerNorm([d_model//2, h_conv, w_conv])
        self.sst_norm = nn.LayerNorm([d_model//2, h_conv, w_conv])
        # Channel-wise L2 normalization will be applied in forward.
        # Patch embedding using a dedicated module.
        self.patch_size = 16
        self.ssh_patch_embed = PatchEmbed(d_model//2, d_model, self.patch_size)
        self.sst_patch_embed = PatchEmbed(d_model//2, d_model, self.patch_size)
        # Fusion block on patch embeddings.
        self.fusion = MultiModalFusion(d_model, dropout)
        n_patches = (h_conv // self.patch_size) * (w_conv // self.patch_size)
        self.pos_encoding = SpatialPositionalEncoding(d_model, n_patches, dropout)
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
    def forward(self, ssh, sst, attention_mask=None):
        self.logger.info(f"Input shapes - SSH: {ssh.shape}, SST: {sst.shape}")
        # Encode raw inputs.
        ssh_encoded = self.ssh_encoder(ssh)
        sst_encoded = self.sst_encoder(sst)
        ssh = self.ssh_norm(ssh_encoded)
        sst = self.sst_norm(sst_encoded)
        # Apply channel-wise L2 normalization.
        ssh = F.normalize(ssh, p=2, dim=1)
        sst = F.normalize(sst, p=2, dim=1)
        # Patch embedding.
        ssh_patch = self.ssh_patch_embed(ssh)  # [B, N, d_model]
        sst_patch = self.sst_patch_embed(sst)
        # Fuse modalities at patch level.
        fused_features = self.fusion(ssh_patch, sst_patch)
        x = self.pos_encoding(fused_features)
        if self.training:
            x = torch.utils.checkpoint.checkpoint(self.transformer, x, use_reentrant=False)
        else:
            x = self.transformer(x)
        # Compute self-attention (standard scaling).
        attention_weights = F.softmax(x @ x.transpose(-2, -1) / math.sqrt(self.d_model), dim=-1)
        self.logger.info(f"Attention map shape: {attention_weights.shape}")
        x = x.mean(dim=1)
        x = self.pre_fc(x)
        x = self.final(x)
        x = x.squeeze(-1)
        # Compute individual attention maps for diagnostics.
        with torch.no_grad():
            try:
                ssh_attn = F.softmax(ssh_patch @ ssh_patch.transpose(-2, -1) / math.sqrt(self.d_model), dim=-1)
                sst_attn = F.softmax(sst_patch @ sst_patch.transpose(-2, -1) / math.sqrt(self.d_model), dim=-1)
                cross_attn = F.softmax(ssh_patch @ sst_patch.transpose(-2, -1) / math.sqrt(self.d_model), dim=-1)
            except Exception as e:
                self.logger.error(f"Failed to generate attention maps: {e}")
                ssh_attn, sst_attn, cross_attn = None, None, None
        attention_maps = {
            'combined': attention_weights,
            'ssh': ssh_attn if ssh_attn is not None else torch.zeros_like(attention_weights),
            'sst': sst_attn if sst_attn is not None else torch.zeros_like(attention_weights),
            'cross': cross_attn if cross_attn is not None else torch.zeros_like(attention_weights)
        }
        self.logger.info(f"Returning attention maps with keys: {attention_maps.keys()}")
        return x, attention_maps
