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
    def __init__(self, output_dir: Union[str, Path], fig_size: tuple = (20, 8), dpi: int = 300):
        self.output_dir = Path(output_dir)
        self.fig_size = fig_size
        self.dpi = dpi
        self.logger = logging.getLogger(__name__)
        self.projection = ccrs.PlateCarree()
        
        for dir_name in ['plots', 'attention_maps', 'predictions']:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)

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
        self.logger.info(f"Plotting attention maps")
        
        if attention_maps.ndim > 2:
            attention_maps = attention_maps.mean(axis=0)
        
        h, w = tlat.shape
        attention_maps = np.array(attention_maps)
        attention_maps = attention_maps[:h-1, :w-1] 

        fig = plt.figure(figsize=self.fig_size)
        gs = plt.GridSpec(1, 2, width_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[0], projection=self.projection)
        ax2 = fig.add_subplot(gs[1], projection=self.projection)

        lon_min, lon_max = -80, 0
        lat_min, lat_max = 0, 65

        ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=self.projection)
        ax1.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black')
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.8)
        gl1 = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl1.top_labels = False
        gl1.right_labels = False

        tlong_mesh, tlat_mesh = np.meshgrid(tlong[:-1], tlat[:-1])
        
        mesh1 = ax1.pcolormesh(
            tlong_mesh, tlat_mesh, attention_maps,
            transform=self.projection,
            cmap='RdBu_r',
            shading='nearest'
        )
        
        plt.colorbar(mesh1, ax=ax1, orientation='horizontal', pad=0.05,
                    label='Meridional Heat Transport')
        ax1.set_title('Ocean Heat Transport Pattern')

        attention_norm = attention_maps / np.max(attention_maps)
        ax2.set_extent([lon_min, lon_max, lat_min, lat_max], crs=self.projection)
        ax2.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black')
        ax2.add_feature(cfeature.COASTLINE, linewidth=0.8)
        gl2 = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl2.top_labels = False
        gl2.right_labels = False

        mesh2 = ax2.pcolormesh(
            tlong_mesh, tlat_mesh, attention_norm,
            transform=self.projection,
            cmap='viridis',
            shading='nearest'
        )

        plt.colorbar(mesh2, ax=ax2, orientation='horizontal', pad=0.05,
                    label='Normalized Attention')
        ax2.set_title('Spatial Attention Distribution')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved attention map to: {save_path}")
            return None

        return fig

    def plot_spatial_pattern(self, data, tlat, tlong, title, cmap='cmo.thermal', save_path=None):
        self.logger.info(f"Plotting spatial pattern for {title}")
        fig = plt.figure(figsize=self.fig_size)
        ax = plt.axes(projection=self.projection)
        self._setup_ocean_map(ax)
        
        # Properly mask the data
        data_masked = np.ma.masked_invalid(data)
        
        # Calculate robust percentiles for colorbar range
        valid_data = data_masked.compressed()  # Get non-masked values
        vmin, vmax = np.nanpercentile(valid_data, [2, 98])
        self.logger.info(f"Data range - Min: {np.nanmin(valid_data):.4f}, Max: {np.nanmax(valid_data):.4f}")
        self.logger.info(f"Using colorbar range: [{vmin:.4f}, {vmax:.4f}]")
        
        if isinstance(cmap, str) and cmap.startswith('cmo.'):
            cmap = getattr(cmocean.cm, cmap[4:])
        
        mesh = ax.pcolormesh(
            tlong, tlat, data_masked,
            transform=self.projection,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading='auto'
        )
        
        cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, extend='both')
        cbar.set_label(title)
        
        units = ""
        if "SSH" in title:
            units = "(meters)"
        elif "SST" in title:
            units = "(°C)"
        if units:
            ax.set_title(f"{title} {units}")
        else:
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

    def _setup_ocean_map(self, ax):
        """Helper function to set up a clean ocean map."""
        ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        
        lon_min, lon_max = -80, 0
        lat_min, lat_max = 10, 60
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=self.projection)
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        
        return gl

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