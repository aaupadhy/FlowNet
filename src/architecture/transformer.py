import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

class PatchEmbed2D(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x, (H, W)

class SpatialPositionalEncoding2D(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim
    def forward(self, x, h, w):
        B, N, C = x.shape
        if N != h * w:
            raise ValueError("Number of patches does not match spatial dims")
        pe = self._build_pe(h, w, C).to(x.device)  # (1, h*w, C)
        x = x + pe
        return self.dropout(x)
    def _build_pe(self, h, w, d_model):
        pe = torch.zeros(1, d_model, h, w)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_h = torch.arange(0, h).unsqueeze(1).float()  # (h, 1)
        pos_w = torch.arange(0, w).unsqueeze(1).float()  # (w, 1)
        pe[0, 0::2, :, :] = torch.sin((pos_h * div_term).t()).unsqueeze(2).repeat(1,1,w)
        pe[0, 1::2, :, :] = torch.cos((pos_h * div_term).t()).unsqueeze(2).repeat(1,1,w)
        pe_w = torch.zeros(1, d_model, h, w)
        pe_w[0, 0::2, :, :] = torch.sin((pos_w * div_term).t()).unsqueeze(1).repeat(1,h,1)
        pe_w[0, 1::2, :, :] = torch.cos((pos_w * div_term).t()).unsqueeze(1).repeat(1,h,1)
        pe = pe + pe_w
        pe = pe.flatten(2).transpose(1, 2)  # (1, h*w, d_model)
        return pe

class MultiModalFusion(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=-1)
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        return self.dropout(x)

class OceanTransformer(nn.Module):
    def __init__(self, spatial_size, d_model=256, nhead=8, num_layers=6, dim_feedforward=512, dropout=0.1, patch_size=8):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.d_model = d_model
        self.patch_size = patch_size
        self.ssh_cnn = nn.Sequential(
            nn.Conv2d(1, d_model // 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model // 4),
            nn.ReLU(),
            nn.Conv2d(d_model // 4, d_model // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.ReLU()
        )
        self.sst_cnn = nn.Sequential(
            nn.Conv2d(1, d_model // 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model // 4),
            nn.ReLU(),
            nn.Conv2d(d_model // 4, d_model // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.ReLU()
        )
        self.ssh_patch = PatchEmbed2D(d_model // 2, d_model, patch_size=patch_size)
        self.sst_patch = PatchEmbed2D(d_model // 2, d_model, patch_size=patch_size)
        self.fusion = MultiModalFusion(d_model, dropout=dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_encoder = SpatialPositionalEncoding2D(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        # New regressor: aggregates three signals
        self.regressor = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, ssh, sst, mask=None):
        B = ssh.shape[0]
        ssh_feat = self.ssh_cnn(ssh)  # (B, d_model//2, H1, W1)
        ssh_patches, (h1, w1) = self.ssh_patch(ssh_feat)  # (B, N, d_model)
        sst_feat = self.sst_cnn(sst)  # (B, d_model//2, H1, W1)
        sst_patches, _ = self.sst_patch(sst_feat)  # (B, N, d_model)
        fused = self.fusion(ssh_patches, sst_patches)  # (B, N, d_model)
        skip = fused.mean(dim=1)  # (B, d_model)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x_input = torch.cat([cls_tokens, fused], dim=1)  # (B, 1+N, d_model)
        pe = self.pos_encoder(x_input[:,1:], h1, w1)
        x_input = torch.cat([x_input[:,0:1], pe], dim=1)
        x_trans = self.transformer(x_input)  # (B, 1+N, d_model)
        cls_token = x_trans[:,0]  # (B, d_model)
        patch_avg = x_trans[:,1:].mean(dim=1)  # (B, d_model)
        # Aggregate three signals: (CLS + skip), patch average, and skip.
        rep = torch.cat([cls_token + skip, patch_avg, skip], dim=1)  # (B, 3*d_model)
        out = self.regressor(rep).squeeze(-1)  # (B,)
        sim = F.cosine_similarity(cls_token.unsqueeze(1), x_trans[:,1:], dim=-1)  # (B, N)
        attn_list = [sim.unsqueeze(1)]
        return out, {'attn': attn_list, 'patch_dims': (h1, w1)}

