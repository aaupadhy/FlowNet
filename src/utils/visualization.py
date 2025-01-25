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
        for dir_name in ['plots', 'attention_maps', 'predictions']:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)
        self._setup_plotting_style()
        self._log_config_to_wandb()

    def _setup_plotting_style(self):
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            'figure.figsize': self.fig_size,
            'figure.dpi': self.dpi,
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'axes.grid': True,
            'grid.alpha': 0.3
        })

    def _log_config_to_wandb(self):
        wandb.config.update({
            "visualization": {
                "fig_size": self.fig_size,
                "dpi": self.dpi,
                "output_dir": str(self.output_dir)
            }
        }, allow_val_change=True)


    def plot_attention_maps(self, attention_maps, tlat, tlong, save_path=None) -> Optional[plt.Figure]:
        
        n_layers = len(attention_maps)
        n_heads = attention_maps[0].shape[1]
        attention_size = attention_maps[0][0, 0].shape[0]
        
        grid_size = int(np.sqrt(attention_size))
        if grid_size * grid_size != attention_size:
            grid_size = int(np.ceil(np.sqrt(attention_size)))
        
        tlat = tlat.values if hasattr(tlat, 'values') else tlat
        tlong = tlong.values if hasattr(tlong, 'values') else tlong
        
        lat_range = np.linspace(tlat.min(), tlat.max(), grid_size)
        lon_range = np.linspace(tlong.min(), tlong.max(), grid_size)
        
        fig, axes = plt.subplots(n_layers, n_heads,
                                figsize=(4*n_heads, 4*n_layers),
                                subplot_kw={'projection': ccrs.PlateCarree()})
        axes = np.atleast_2d(axes)
        
        for layer_idx, layer_attn in enumerate(attention_maps):
            for head_idx in range(n_heads):
                ax = axes[layer_idx, head_idx]
                
                attention = layer_attn[0, head_idx]
                padded_attention = np.zeros((grid_size * grid_size))
                padded_attention[:attention_size] = attention
                attention_grid = padded_attention.reshape(grid_size, grid_size)
                
                ax.coastlines(resolution='50m')
                ax.add_feature(cfeature.BORDERS, linestyle=':')
                
                im = ax.imshow(
                    attention_grid,
                    extent=[lon_range[0], lon_range[-1], lat_range[0], lat_range[-1]],
                    transform=ccrs.PlateCarree(),
                    cmap='viridis',
                    aspect='auto',
                    origin='lower'
                )
                
                ax.set_title(f'Layer {layer_idx+1}, Head {head_idx+1}')
                plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
        
        plt.tight_layout()
        
        if save_path:
            save_path = self.output_dir / 'attention_maps' / f"{save_path}.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            wandb.log({
                f"attention_maps/{Path(save_path).stem}": wandb.Image(str(save_path))
            })
            return None
        
        return fig


    def plot_spatial_pattern(self, data, tlat, tlong, title, cmap='cmo.thermal', save_path=None):
        fig = plt.figure(figsize=self.fig_size)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines(resolution='50m')
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.gridlines(draw_labels=True)
        
        data_subset = data[::2, ::2]
        tlat_subset = tlat[::2, ::2]
        tlong_subset = tlong[::2, ::2]
        
        vmin, vmax = np.nanpercentile(data_subset, [2, 98])
        im = ax.pcolormesh(
            tlong_subset, tlat_subset, data_subset,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
        plt.title(title)

        if save_path:
            save_path = self.output_dir / 'plots' / f"{save_path}.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            wandb.log({
                f"spatial_patterns/{Path(save_path).stem}": wandb.Image(str(save_path))
            })
            self.logger.info(f"Saved spatial pattern plot to {save_path}")

        return fig