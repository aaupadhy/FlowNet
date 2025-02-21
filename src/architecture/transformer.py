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
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
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
        pe = self._build_pe(h, w, C).to(x.device)
        return self.dropout(x + pe)

    def _build_pe(self, h, w, d_model):
        pe = torch.zeros(1, d_model, h, w)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_h = torch.arange(0, h).unsqueeze(1).float()
        pos_w = torch.arange(0, w).unsqueeze(1).float()
        pe[0, 0::2, :, :] = torch.sin((pos_h * div_term).t()).unsqueeze(2).repeat(1, 1, w)
        pe[0, 1::2, :, :] = torch.cos((pos_h * div_term).t()).unsqueeze(2).repeat(1, 1, w)
        pe_w = torch.zeros(1, d_model, h, w)
        pe_w[0, 0::2, :, :] = torch.sin((pos_w * div_term).t()).unsqueeze(1).repeat(1, h, 1)
        pe_w[0, 1::2, :, :] = torch.cos((pos_w * div_term).t()).unsqueeze(1).repeat(1, h, 1)
        pe = pe + pe_w
        return pe.flatten(2).transpose(1, 2)

class VNTDecoder(nn.Module):
    def __init__(self, in_channels, target_channels, target_h, target_w):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels // 8, in_channels // 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels // 16, target_channels, kernel_size=4, stride=2, padding=1)
        )
        self.target_h = target_h
        self.target_w = target_w

    def forward(self, x):
        # x: (B, d_model, h, w)
        x = self.decoder(x)
        if x.shape[-2:] != (self.target_h, self.target_w):
            x = nn.functional.interpolate(x, size=(self.target_h, self.target_w), mode='bilinear', align_corners=False)
        return x

class OceanTransformer(nn.Module):
    def __init__(self, spatial_size, d_model=256, nhead=8, num_layers=6,
                 dim_feedforward=512, dropout=0.1, patch_size=8, target_nlat=None, target_nlon=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.d_model = d_model
        self.patch_size = patch_size
        self.target_nlat = target_nlat
        self.target_nlon = target_nlon
        if target_nlat is None or target_nlon is None:
            raise ValueError("target_nlat and target_nlon must be provided for VNT prediction")

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
        self.ssh_pos_encoder = SpatialPositionalEncoding2D(d_model, dropout=dropout)
        self.sst_pos_encoder = SpatialPositionalEncoding2D(d_model, dropout=dropout)
        self.ssh_cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.sst_cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.ssh_transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.sst_transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.regressor = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        self.vnt_decoder = VNTDecoder(in_channels=d_model, target_channels=62,
                                      target_h=target_nlat, target_w=target_nlon)
        self.residual_scale = nn.Parameter(torch.ones(1))
        self.residual_bias = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def forward(self, ssh, sst, mask=None):
        B = ssh.shape[0]
        # SSH branch
        ssh_feat = self.ssh_cnn(ssh)
        ssh_patches, (h1, w1) = self.ssh_patch(ssh_feat)
        ssh_cls = self.ssh_cls_token.expand(B, -1, -1)
        ssh_input = torch.cat([ssh_cls, ssh_patches], dim=1)
        ssh_pe = self.ssh_pos_encoder(ssh_input[:, 1:], h1, w1)
        ssh_input = torch.cat([ssh_input[:, :1], ssh_pe], dim=1)
        ssh_trans = self.ssh_transformer(ssh_input)
        ssh_cls_out = ssh_trans[:, 0]
        ssh_patch_avg = ssh_trans[:, 1:].mean(dim=1)
        ssh_sim = F.cosine_similarity(ssh_cls_out.unsqueeze(1), ssh_trans[:, 1:], dim=-1)

        # SST branch
        sst_feat = self.sst_cnn(sst)
        sst_patches, (h2, w2) = self.sst_patch(sst_feat)
        sst_cls = self.sst_cls_token.expand(B, -1, -1)
        sst_input = torch.cat([sst_cls, sst_patches], dim=1)
        sst_pe = self.sst_pos_encoder(sst_input[:, 1:], h2, w2)
        sst_input = torch.cat([sst_input[:, :1], sst_pe], dim=1)
        sst_trans = self.sst_transformer(sst_input)
        sst_cls_out = sst_trans[:, 0]
        sst_patch_avg = sst_trans[:, 1:].mean(dim=1)
        sst_sim = F.cosine_similarity(sst_cls_out.unsqueeze(1), sst_trans[:, 1:], dim=-1)

        # Aggregated prediction branch (fallback)
        rep = torch.cat([ssh_cls_out, ssh_patch_avg, sst_cls_out, sst_patch_avg], dim=1)
        predicted_residual = self.regressor(rep).squeeze(-1)
        agg_pred = self.residual_bias + self.residual_scale * predicted_residual

        # VNT prediction branch
        fused_tokens = (ssh_patches + sst_patches) / 2.0
        B, N, C = fused_tokens.shape
        spatial_map = fused_tokens.transpose(1, 2).reshape(B, C, h1, w1)
        predicted_vnt = self.vnt_decoder(spatial_map)
        attn_list = [ssh_sim.unsqueeze(1), sst_sim.unsqueeze(1)]
        return agg_pred, predicted_vnt, {'attn': attn_list, 'patch_dims': (h1, w1)}

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
