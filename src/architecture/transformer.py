import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

class PatchEmbed2D(nn.Module):
    """
    2D Patch Embedding layer
    Converts 2D image data into sequence of patch embeddings
    """
    def __init__(self, in_channels, embed_dim, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        """
        Args:
            x: Input feature map [B, C, H, W]
            
        Returns:
            Patch embeddings [B, N, D] and spatial output dimensions (H', W')
        """
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, C, H*W] -> [B, H*W, C]
        return x, (H, W)

class SpatialPositionalEncoding2D(nn.Module):
    """
    2D Positional Encoding for spatial data
    Adds position information to patch embeddings
    """
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim
        
    def forward(self, x, h, w):
        """
        Args:
            x: Patch embeddings [B, N, D]
            h: Height in patches
            w: Width in patches
            
        Returns:
            Embeddings with positional information added
        """
        B, N, C = x.shape
        if N != h * w:
            raise ValueError(f"Number of patches ({N}) does not match spatial dims ({h}x{w}={h*w})")
        
        pe = self._build_pe(h, w, C).to(x.device)
        return self.dropout(x + pe)
        
    def _build_pe(self, h, w, d_model):
        """Build 2D positional encoding with sinusoidal functions"""
        # Initialize positional encoding
        pe = torch.zeros(1, d_model, h, w)
        
        # Use log-space frequencies for better generalization
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Generate position values
        pos_h = torch.arange(0, h).unsqueeze(1).float()
        pos_w = torch.arange(0, w).unsqueeze(1).float()
        
        # Apply sinusoidal encoding for height dimension
        pe[0, 0::2, :, :] = torch.sin((pos_h * div_term).t()).unsqueeze(2).repeat(1, 1, w)
        pe[0, 1::2, :, :] = torch.cos((pos_h * div_term).t()).unsqueeze(2).repeat(1, 1, w)
        
        # Apply sinusoidal encoding for width dimension
        pe_w = torch.zeros(1, d_model, h, w)
        pe_w[0, 0::2, :, :] = torch.sin((pos_w * div_term).t()).unsqueeze(1).repeat(1, h, 1)
        pe_w[0, 1::2, :, :] = torch.cos((pos_w * div_term).t()).unsqueeze(1).repeat(1, h, 1)
        
        # Combine height and width encodings
        pe = pe + pe_w
        
        # Reshape to match sequence format
        return pe.flatten(2).transpose(1, 2)  # [1, C, H*W] -> [1, H*W, C]


class VNTDecoder(nn.Module):
    """
    Decoder module for VNT field prediction
    Upsamples encoded features to target resolution
    """
    def __init__(self, in_channels, target_channels, target_h, target_w):
        super().__init__()
        
        # Calculate number of upsampling layers needed
        self.decoder = nn.Sequential(
            # Progressive upsampling with decreasing channel count
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
            
            nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
            
            nn.ConvTranspose2d(in_channels // 4, in_channels // 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(),
            
            nn.ConvTranspose2d(in_channels // 8, in_channels // 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(in_channels // 16),
            nn.ReLU(),
            
            # Final upsampling to target channels
            nn.ConvTranspose2d(in_channels // 16, target_channels, kernel_size=4, stride=2, padding=1)
        )
        
        self.target_h = target_h
        self.target_w = target_w
        
    def forward(self, x):
        """
        Args:
            x: Input feature map [B, C, H, W]
            
        Returns:
            Decoded VNT field [B, target_channels, target_h, target_w]
        """
        x = self.decoder(x)
        
        # Ensure output matches target dimensions exactly
        if x.shape[-2:] != (self.target_h, self.target_w):
            x = F.interpolate(x, size=(self.target_h, self.target_w), 
                            mode='bilinear', align_corners=False)
        return x


class HeatTransportAggregator(nn.Module):
    """
    Aggregates VNT field predictions to heat transport values
    Implements the physical aggregation formula
    """
    def __init__(self, tarea, dz, tarea_conversion=0.0001, dz_conversion=0.01, ref_lat_index=None):
        super().__init__()
        
        # Register tensors as buffers (not trained)
        if isinstance(tarea, torch.Tensor):
            self.register_buffer('tarea', tarea)
        else:
            self.register_buffer('tarea', torch.tensor(tarea.values, dtype=torch.float32))
            
        if isinstance(dz, torch.Tensor):
            self.register_buffer('dz', dz)
        else:
            self.register_buffer('dz', torch.tensor(dz.values, dtype=torch.float32))
        
        # Store conversion factors
        self.tarea_conversion = tarea_conversion
        self.dz_conversion = dz_conversion
        self.ref_lat_index = ref_lat_index
        
    def forward(self, vnt):
        """
        Args:
            vnt: VNT field tensor [B, D, H, W]
            
        Returns:
            Heat transport values [B]
        """
        # Match dimensions for broadcasting
        batch_size = vnt.shape[0]
        
        # Expand tarea to match vnt dimensions
        tarea_expanded = self.tarea.view(1, 1, 1, -1).expand(batch_size, -1, -1, -1)
        
        # Expand dz to match vnt dimensions
        dz_expanded = self.dz.view(1, -1, 1, 1).expand(batch_size, -1, -1, vnt.shape[-1])
        
        # Calculate heat transport with proper broadcasting
        heat_transport = (
            vnt * 
            tarea_expanded * self.tarea_conversion * 
            dz_expanded * self.dz_conversion
        )
        
        # Sum over depth and longitude dimensions
        heat_transport = heat_transport.sum(dim=[1, 3])
        
        # Select reference latitude if provided
        if self.ref_lat_index is not None:
            heat_transport = heat_transport[:, self.ref_lat_index]
            
        return heat_transport


class OceanTransformer(nn.Module):
    """
    Transformer model for ocean heat transport prediction
    
    Features:
    - Separate encoders for SSH and SST data
    - Memory-efficient transformer with gradient checkpointing
    - Dual prediction pathways: direct and VNT-based
    - Attention visualization for interpretability
    """
    def __init__(self, spatial_size, d_model=256, nhead=8, num_layers=6,
                 dim_feedforward=512, dropout=0.1, patch_size=8, target_nlat=None, target_nlon=None,
                 vnt_depth=62, tarea=None, dz=None, ref_lat_index=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.d_model = d_model
        self.patch_size = patch_size
        self.target_nlat = target_nlat
        self.target_nlon = target_nlon
        
        # Validate inputs
        if target_nlat is None or target_nlon is None:
            raise ValueError("target_nlat and target_nlon must be provided for VNT prediction")
        
        # SSH processing branch
        self.ssh_cnn = nn.Sequential(
            nn.Conv2d(1, d_model // 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model // 4),
            nn.ReLU(),
            nn.Conv2d(d_model // 4, d_model // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.ReLU()
        )
        
        # SST processing branch
        self.sst_cnn = nn.Sequential(
            nn.Conv2d(1, d_model // 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model // 4),
            nn.ReLU(),
            nn.Conv2d(d_model // 4, d_model // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.ReLU()
        )
        
        # Patch embedding modules
        self.ssh_patch = PatchEmbed2D(d_model // 2, d_model, patch_size=patch_size)
        self.sst_patch = PatchEmbed2D(d_model // 2, d_model, patch_size=patch_size)
        
        # Positional encoding modules
        self.ssh_pos_encoder = SpatialPositionalEncoding2D(d_model, dropout=dropout)
        self.sst_pos_encoder = SpatialPositionalEncoding2D(d_model, dropout=dropout)
        
        # Class tokens for global representation
        self.ssh_cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.sst_cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Transformer encoders with memory-efficient configuration
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.ssh_transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.sst_transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Direct heat transport prediction head
        self.regressor = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.LayerNorm(d_model),  # LayerNorm instead of BatchNorm for more stable training
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
        # VNT decoder for spatial heat transport prediction
        self.vnt_decoder = VNTDecoder(
            in_channels=d_model, 
            target_channels=vnt_depth,
            target_h=target_nlat,  # or appropriate subset for memory efficiency
            target_w=target_nlon
        )
        
        # Heat transport aggregation layer
        if tarea is not None and dz is not None:
            self.heat_aggregator = HeatTransportAggregator(
                tarea=tarea,
                dz=dz,
                ref_lat_index=ref_lat_index
            )
        else:
            self.heat_aggregator = None
            
        # Parameters for residual connections
        self.residual_scale = nn.Parameter(torch.ones(1))
        self.residual_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        self._init_weights()
        
    def forward(self, ssh, sst, mask=None):
        """
        Forward pass
        
        Args:
            ssh: Sea Surface Height [B, 1, H, W]
            sst: Sea Surface Temperature [B, 1, H, W]
            mask: Optional mask for valid data [B, H, W]
            
        Returns:
            heat_transport: Predicted heat transport [B]
            vnt: Predicted VNT field [B, D, H, W]
            metadata: Dictionary with attention maps and dimensions
        """
        B = ssh.shape[0]
        
        # Create attention mask from data mask if provided
        attn_mask = None
        if mask is not None:
            # Downsample mask to match patch grid
            mask_float = mask.float()
            # Account for CNN downsampling (factor of 4) and patching
            patch_size_factor = 4 * self.patch_size
            
            # Downsample mask using average pooling
            mask_downsampled = F.avg_pool2d(
                mask_float.unsqueeze(1), 
                kernel_size=patch_size_factor, 
                stride=patch_size_factor
            ).squeeze(1)
            
            # Create boolean attention mask (1=attend, 0=ignore)
            attn_mask = mask_downsampled > 0.5
        
        # Process SSH through CNN and create patches
        ssh_feat = self.ssh_cnn(ssh)
        ssh_patches, (h1, w1) = self.ssh_patch(ssh_feat)
        
        # Add class token and positional encoding for SSH
        ssh_cls = self.ssh_cls_token.expand(B, -1, -1)
        ssh_input = torch.cat([ssh_cls, ssh_patches], dim=1)
        ssh_pe = self.ssh_pos_encoder(ssh_input[:, 1:], h1, w1)
        ssh_input = torch.cat([ssh_input[:, :1], ssh_pe], dim=1)
        
        # Use gradient checkpointing for memory efficiency
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(inputs[0])
            return custom_forward
            
        # Apply transformer with memory optimization when training
        if self.training and ssh_input.requires_grad:
            ssh_trans = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.ssh_transformer),
                ssh_input
            )
        else:
            ssh_trans = self.ssh_transformer(ssh_input)
            
        # Extract features from SSH transformer
        ssh_cls_out = ssh_trans[:, 0]  # CLS token output
        ssh_patch_avg = ssh_trans[:, 1:].mean(dim=1)  # Average patch features
        
        # Calculate attention similarity for visualization
        # This shows which regions the model focuses on
        ssh_sim = F.cosine_similarity(ssh_cls_out.unsqueeze(1), ssh_trans[:, 1:], dim=-1)
        
        # Process SST through CNN and create patches
        sst_feat = self.sst_cnn(sst)
        sst_patches, (h2, w2) = self.sst_patch(sst_feat)
        
        # Add class token and positional encoding for SST
        sst_cls = self.sst_cls_token.expand(B, -1, -1)
        sst_input = torch.cat([sst_cls, sst_patches], dim=1)
        sst_pe = self.sst_pos_encoder(sst_input[:, 1:], h2, w2)
        sst_input = torch.cat([sst_input[:, :1], sst_pe], dim=1)
        
        # Apply transformer with memory optimization
        if self.training and sst_input.requires_grad:
            sst_trans = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.sst_transformer),
                sst_input
            )
        else:
            sst_trans = self.sst_transformer(sst_input)
            
        # Extract features from SST transformer
        sst_cls_out = sst_trans[:, 0]
        sst_patch_avg = sst_trans[:, 1:].mean(dim=1)
        
        # Calculate attention similarity for SST
        sst_sim = F.cosine_similarity(sst_cls_out.unsqueeze(1), sst_trans[:, 1:], dim=-1)
        
        # Concatenate features for direct heat transport prediction
        rep = torch.cat([ssh_cls_out, ssh_patch_avg, sst_cls_out, sst_patch_avg], dim=1)
        predicted_residual = self.regressor(rep).squeeze(-1)
        
        # Apply scaling and bias for flexibility
        direct_pred = self.residual_bias + self.residual_scale * predicted_residual
        
        # Fuse SSH and SST features for VNT prediction
        # Use weighted fusion based on mask coverage
        if mask is not None:
            # Calculate per-channel weights based on valid data percentage
            ssh_weight = mask.float().mean().unsqueeze(0)
            sst_weight = mask.float().mean().unsqueeze(0)
            
            # Normalize weights
            total = ssh_weight + sst_weight
            ssh_weight = ssh_weight / total
            sst_weight = sst_weight / total
            
            # Apply weighted fusion
            fused_tokens = ssh_patches * ssh_weight + sst_patches * sst_weight
        else:
            # Simple averaging if no mask provided
            fused_tokens = (ssh_patches + sst_patches) / 2.0
            
        # Reshape for decoder
        B, N, C = fused_tokens.shape
        spatial_map = fused_tokens.transpose(1, 2).reshape(B, C, h1, w1)
        
        # Predict VNT field
        predicted_vnt = self.vnt_decoder(spatial_map)
        
        # Aggregate VNT to heat transport if aggregator is available
        if self.heat_aggregator is not None:
            agg_pred = self.heat_aggregator(predicted_vnt)
        else:
            # Fall back to direct prediction
            agg_pred = direct_pred
        
        # Prepare attention maps for visualization
        attn_list = [ssh_sim.unsqueeze(1), sst_sim.unsqueeze(1)]
        
        # Return predictions and metadata
        return agg_pred, predicted_vnt, {'attn': attn_list, 'patch_dims': (h1, w1)}
        
    def _init_weights(self):
        """Initialize model weights properly"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                # Kaiming initialization for ReLU networks
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                # Unit variance for normalization layers
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)