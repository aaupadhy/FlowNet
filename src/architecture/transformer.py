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

    def forward(self, x, mask=None):
        x = self.proj(x)
        if mask is not None:
            mask = F.avg_pool2d(mask.float().unsqueeze(1), 
                              kernel_size=self.proj.kernel_size,
                              stride=self.proj.stride)
            mask = (mask > 0.5).squeeze(1)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        if mask is not None:
            mask = mask.view(B, -1)
        return x, mask

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
        self.nhead = nhead

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

        def conv_out_dim(L):
            return (L + 2 - 3) // 2 + 1

        h_conv = conv_out_dim(spatial_size[0])
        w_conv = conv_out_dim(spatial_size[1])

        self.ssh_norm = nn.LayerNorm([d_model//2, h_conv, w_conv])
        self.sst_norm = nn.LayerNorm([d_model//2, h_conv, w_conv])

        self.patch_size = 16
        self.ssh_patch_embed = PatchEmbed(d_model//2, d_model, self.patch_size)
        self.sst_patch_embed = PatchEmbed(d_model//2, d_model, self.patch_size)

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

    def compute_multihead_attention(self, q, k, v, mask=None):
        B, L, E = q.shape
        
        q = q.view(B, L, self.nhead, E // self.nhead).transpose(1, 2)  # [B, H, L, E/H]
        k = k.view(B, L, self.nhead, E // self.nhead).transpose(1, 2)  # [B, H, L, E/H]
        v = v.view(B, L, self.nhead, E // self.nhead).transpose(1, 2)  # [B, H, L, E/H]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(E // self.nhead)  # [B, H, L, L]

        if mask is not None:
            expanded_mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            expanded_mask = expanded_mask.expand(-1, self.nhead, L, -1)  # [B, H, L, L]
            scores = scores.masked_fill(~expanded_mask, float('-inf'))

        attn_probs = F.softmax(scores, dim=-1)  # [B, H, L, L]

        context = torch.matmul(attn_probs, v)  # [B, H, L, E/H]
        
        context = context.transpose(1, 2).contiguous()  # [B, L, H, E/H]
        context = context.view(B, L, E)  # [B, L, E]

        return context, attn_probs

    def forward(self, ssh, sst, attention_mask=None):
        B = ssh.size(0)
        
        ssh_encoded = self.ssh_encoder(ssh)
        sst_encoded = self.sst_encoder(sst)

        ssh = self.ssh_norm(ssh_encoded)
        sst = self.sst_norm(sst_encoded)

        ssh = F.normalize(ssh, p=2, dim=1)
        sst = F.normalize(sst, p=2, dim=1)

        ssh_patch, ssh_mask = self.ssh_patch_embed(ssh, attention_mask)
        sst_patch, sst_mask = self.sst_patch_embed(sst, attention_mask)

        fused_features = self.fusion(ssh_patch, sst_patch)
        x = self.pos_encoding(fused_features)

        if ssh_mask is not None and sst_mask is not None:
            transformer_mask = ssh_mask & sst_mask
        else:
            transformer_mask = None

        if self.training:
            x = torch.utils.checkpoint.checkpoint(self.transformer, x, use_reentrant=False)
        else:
            x = self.transformer(x)

        Q = x
        K = x
        V = x

        L = Q.size(1)
        if transformer_mask is not None:
            transformer_mask = transformer_mask.view(B, L)

        combined_attention, attention_weights = self.compute_multihead_attention(Q, K, V, transformer_mask)

        valid_tokens = transformer_mask.sum(dim=1, keepdim=True) if transformer_mask is not None else L
        x = x.sum(dim=1) / valid_tokens
        x = self.pre_fc(x)
        x = self.final(x)
        x = x.squeeze(-1)

        attention_maps = {}
        with torch.no_grad():
            try:
                ssh_attn, _ = self.compute_multihead_attention(ssh_patch, ssh_patch, ssh_patch, ssh_mask)
                sst_attn, _ = self.compute_multihead_attention(sst_patch, sst_patch, sst_patch, sst_mask)
                cross_attn, _ = self.compute_multihead_attention(ssh_patch, sst_patch, sst_patch, ssh_mask)

                attention_maps = {
                    'combined': attention_weights.mean(1),
                    'ssh': ssh_attn.mean(1),
                    'sst': sst_attn.mean(1),
                    'cross': cross_attn.mean(1)
                }
            except Exception as e:
                self.logger.error(f"Failed to generate attention maps: {e}")
                attention_maps = None

        return x, attention_maps