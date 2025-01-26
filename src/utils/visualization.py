import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
from pathlib import Path
import cmocean
import logging
import torch
from datetime import datetime
from typing import Optional, Tuple, Union
import matplotlib.colors as colors
import wandb

class OceanVisualizer:
    def __init__(self, output_dir: str, fig_size: tuple = (12, 8), dpi: int = 300):
        self.output_dir = Path(output_dir)
        self.fig_size = fig_size
        self.dpi = dpi
        self.logger = logging.getLogger(__name__)
        self.projection = ccrs.PlateCarree()
        
        for dir_name in ['plots', 'attention_maps', 'predictions']:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)
            
        self._setup_plotting_style()
        self._log_config_to_wandb()

    def _setup_plotting_style(self):
        try:
            sns.set_theme(style="whitegrid", font_scale=1.2)
            sns.set_palette("colorblind")
            
            plt.rcParams.update({
                'figure.figsize': self.fig_size,
                'figure.dpi': self.dpi,
                'axes.grid': True,
                'grid.alpha': 0.3,
                'axes.linewidth': 1.0,
                'axes.spines.top': False,
                'axes.spines.right': False,
                'xtick.major.size': 6,
                'ytick.major.size': 6,
                'xtick.minor.size': 3,
                'ytick.minor.size': 3,
                'image.cmap': 'viridis',
            })
            
        except Exception as e:
            self.logger.warning(f"Error setting up plot style: {str(e)}")
            self.logger.info("Falling back to basic matplotlib settings")

    def _log_config_to_wandb(self):
        """Log visualization config to wandb if initialized."""
        try:
            if wandb.run is not None:
                wandb.config.update({
                    "visualization": {
                        "fig_size": self.fig_size,
                        "dpi": self.dpi,
                        "output_dir": str(self.output_dir)
                    }
                }, allow_val_change=True)
        except Exception as e:
            self.logger.warning(f"Could not log to wandb: {str(e)}")
            pass

    def _setup_ocean_map(self, ax):
        ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        
        # Gyre region focus 
        lon_min, lon_max = -80, 0  # Atlantic focus
        lat_min, lat_max = 10, 60  # Main gyre region
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=self.projection)
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        return gl

    def plot_attention_maps(self, attention_maps, tlat, tlong, save_path=None):
        self.logger.info(f"Plotting attention maps with shape: {attention_maps.shape}")
        
        if not isinstance(attention_maps, np.ndarray):
            attention_maps = np.array(attention_maps)
            
        if len(attention_maps.shape) < 3:
            raise ValueError(f"Expected attention maps with shape (n_layers, n_heads, ...); got {attention_maps.shape}")
            
        if np.isnan(attention_maps).any():
            self.logger.warning("Found NaN values in attention maps - replacing with zeros")
            attention_maps = np.nan_to_num(attention_maps, 0.0)
        
        n_layers = len(attention_maps)
        n_heads = attention_maps[0].shape[1]
        
        fig = plt.figure(figsize=(20, 10))
        gs = plt.GridSpec(2, 2, figure=fig, width_ratios=[1, 1])
        
        ax_heat = fig.add_subplot(gs[:, 0], projection=self.projection)
        ax_attn = fig.add_subplot(gs[:, 1], projection=self.projection)
        
        mean_attention = np.mean(attention_maps, axis=(0,1))
        
        if len(mean_attention.shape) > 2:
            mean_attention = mean_attention[0]
            
        if len(mean_attention.shape) != 2:
            grid_h = int(np.floor(np.sqrt(mean_attention.shape[0])))
            grid_w = int(np.ceil(mean_attention.shape[0] / grid_h))
            mean_attention = mean_attention[:grid_h * grid_w].reshape(grid_h, grid_w)
            
        self.logger.info(f"Reshaped attention map to size: {mean_attention.shape}")
        
        tlat_grid = np.linspace(tlat.min(), tlat.max(), mean_attention.shape[0])
        tlong_grid = np.linspace(tlong.min(), tlong.max(), mean_attention.shape[1])
        
        # Normalize attention weights to make patterns more visible
        mean_attention = (mean_attention - mean_attention.min()) / (mean_attention.max() - mean_attention.min() + 1e-6)
        
        self._setup_ocean_map(ax_heat)
        heat_map = ax_heat.pcolormesh(
            tlong_grid, tlat_grid, mean_attention,
            transform=self.projection,
            cmap='YlOrRd',  # Changed colormap for better visibility
            norm=colors.LogNorm(vmin=1e-3, vmax=1.0),  # Log scale to highlight variations
            shading='auto'
        )
        plt.colorbar(heat_map, ax=ax_heat, label='Heat Transport Contribution (normalized)')
        ax_heat.set_title('Ocean Heat Transport Pattern')
        
        self._setup_ocean_map(ax_attn)
        attention_map = ax_attn.pcolormesh(
            tlong_grid, tlat_grid, mean_attention,
            transform=self.projection,
            cmap='viridis',
            shading='auto'
        )
        plt.colorbar(attention_map, ax=ax_attn, label='Attention Weight')
        ax_attn.set_title('Spatial Attention Distribution')
        
        plt.tight_layout()
        
        if save_path:
            if isinstance(save_path, Path):
                save_path = str(save_path)
            save_path = Path(save_path)
            if not save_path.is_absolute():
                save_path = self.output_dir / 'attention_maps' / f"{save_path.stem}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Saving attention map to: {save_path}")
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            wandb.log({
                f"attention_maps/{Path(save_path).stem}": wandb.Image(str(save_path))
            })
            
            self.logger.info(f"Saved attention map visualization to {save_path}")
            return None
            
        return fig

    def plot_spatial_pattern(self, data, tlat, tlong, title, cmap='cmo.thermal', save_path=None):
        self.logger.info(f"Plotting spatial pattern for {title}")
        
        fig = plt.figure(figsize=self.fig_size)
        ax = plt.axes(projection=self.projection)
        
        self._setup_ocean_map(ax)
        
        if isinstance(cmap, str) and cmap.startswith('cmo.'):
            cmap = getattr(cmocean.cm, cmap[4:])
            
        data_masked = np.ma.masked_invalid(data)
        vmin, vmax = np.nanpercentile(data_masked, [2, 98])
        
        mesh = ax.pcolormesh(
            tlong, tlat, data_masked,
            transform=self.projection,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading='auto'
        )
        
        cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05)
        cbar.set_label(title)
        
        ax.set_title(title)
        
        if save_path:
            save_path = self.output_dir / 'plots' / f"{save_path}.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            wandb.log({
                f"spatial_patterns/{Path(save_path).stem}": wandb.Image(str(save_path))
            })
            
            self.logger.info(f"Saved spatial pattern plot to {save_path}")
            return None
            
        return fig

    def plot_predictions(self, predictions, targets, time_indices=None, save_path=None):
        self.logger.info("Plotting model predictions vs targets")
        
        if time_indices is None:
            time_indices = np.arange(len(predictions))
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        scatter = ax1.scatter(targets, predictions, alpha=0.5, c='blue')
        ax1.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
        ax1.set_xlabel('True Heat Transport (Normalized)')
        ax1.set_ylabel('Predicted Heat Transport (Normalized)')
        ax1.set_title('Prediction Accuracy')
        
        time_series = ax2.plot(time_indices, targets, 'b-', label='True', alpha=0.7)
        ax2.plot(time_indices, predictions, 'r--', label='Predicted', alpha=0.7)
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Heat Transport (Normalized)')
        ax2.set_title('Heat Transport Time Series')
        ax2.legend()

        note_text = ("Note: Values shown are normalized. Raw values in degC*m³/s\n"
                    "To convert to Watts: multiply by 4184 * 1025")
        fig.text(0.1, 0.01, note_text, fontsize=8, style='italic')
        
        plt.tight_layout()
        
        if save_path:
            save_path = self.output_dir / 'predictions' / f"{save_path}.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            wandb.log({
                f"predictions/{Path(save_path).stem}": wandb.Image(str(save_path))
            })
            
            self.logger.info(f"Saved predictions plot to {save_path}")
            return None
            
        return fig