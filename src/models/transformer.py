import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class SpatialPositionalEncoding(nn.Module):
    """
    Positional encoding for 2D spatial data (SSH/SST fields)
    """
    def __init__(self, d_model: int, max_h: int, max_w: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(1, max_h, max_w, d_model)
        h_pos = torch.arange(0, max_h).unsqueeze(1).unsqueeze(2)
        w_pos = torch.arange(0, max_w).unsqueeze(0).unsqueeze(2)
        
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe[0, :, :, 0::2] = torch.sin(h_pos * div_term)
        pe[0, :, :, 1::2] = torch.cos(h_pos * div_term)
        pe[0, :, :, 0::2] += torch.sin(w_pos * div_term)
        pe[0, :, :, 1::2] += torch.cos(w_pos * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input"""
        x = x + self.pe[:, :x.size(1), :x.size(2)]
        return self.dropout(x)

class MultiModalFusion(nn.Module):
    """
    Fusion module for combining SSH and SST features
    """
    def __init__(self, ssh_dim: int, sst_dim: int, fusion_dim: int):
        super().__init__()
        self.ssh_proj = nn.Linear(ssh_dim, fusion_dim)
        self.sst_proj = nn.Linear(sst_dim, fusion_dim)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU()
        )
    
    def forward(self, ssh_feat: torch.Tensor, sst_feat: torch.Tensor) -> torch.Tensor:
        ssh_proj = self.ssh_proj(ssh_feat)
        sst_proj = self.sst_proj(sst_feat)
        combined = torch.cat([ssh_proj, sst_proj], dim=-1)
        return self.fusion(combined)

class OceanTransformer(nn.Module):
    def __init__(
        self,
        spatial_size: Tuple[int, int],
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        height, width = spatial_size
        self.d_model = d_model
        
        self.ssh_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        
        self.sst_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        
        self.fusion = MultiModalFusion(64, 64, d_model)
        
        self.pos_encoder = SpatialPositionalEncoding(d_model, height, width, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        ssh: torch.Tensor,
        sst: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        ssh_feat = self.ssh_encoder(ssh)  # B, 64, H, W
        sst_feat = self.sst_encoder(sst)  # B, 64, H, W
        
        # Reshape for fusion
        B, C, H, W = ssh_feat.shape
        ssh_feat = ssh_feat.permute(0, 2, 3, 1)  # B, H, W, C
        sst_feat = sst_feat.permute(0, 2, 3, 1)  # B, H, W, C
        
        fused = self.fusion(ssh_feat, sst_feat)  # B, H, W, d_model
        
        fused = self.pos_encoder(fused)
        
        fused = fused.reshape(B, H * W, self.d_model)
        
        if mask is not None:
            mask = mask.reshape(B, H * W)
        out = self.transformer(fused, src_key_padding_mask=mask)
        
        out = out.mean(dim=1)
        
        out = self.dropout(F.relu(self.fc1(out)))
        return self.fc2(out)
    
    def get_attention_maps(
        self,
        ssh: torch.Tensor,
        sst: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get attention maps for visualization"""
        self.eval()
        with torch.no_grad():
            ssh_feat = self.ssh_encoder(ssh)
            sst_feat = self.sst_encoder(sst)
            
            B, C, H, W = ssh_feat.shape
            ssh_feat = ssh_feat.permute(0, 2, 3, 1)
            sst_feat = sst_feat.permute(0, 2, 3, 1)
            
            fused = self.fusion(ssh_feat, sst_feat)
            fused = self.pos_encoder(fused)
            fused = fused.reshape(B, H * W, self.d_model)
            
            attention_maps = []
            for layer in self.transformer.layers:
                attn_output, attn_weights = layer.self_attn(
                    fused, fused, fused,
                    need_weights=True,
                    average_attn_weights=False
                )
                attention_maps.append(attn_weights)
            
            avg_attention = torch.stack(attention_maps).mean(dim=(0, 1))  # B, L, L
            
            spatial_attention = avg_attention.mean(dim=1).reshape(B, H, W)
            
            return spatial_attention, attention_maps