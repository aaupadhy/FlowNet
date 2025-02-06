import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from pathlib import Path
import logging
import json
from scipy.ndimage import zoom

class OceanVisualizer:
    def __init__(self, output_dir, fig_size=(20, 8), dpi=300):
        self.output_dir = Path(output_dir)
        self.fig_size = fig_size
        self.dpi = dpi
        self.logger = logging.getLogger(__name__)
        for subdir in ['plots', 'attention_maps', 'predictions']:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)
        self.projection = ccrs.PlateCarree()
    
    def _get_geographic_grid(self, attn_shape, tlat, tlong):
        # Use provided tlat/tlong or defaults.
        if tlat is not None and tlong is not None:
            lat_min, lat_max = np.min(tlat), np.max(tlat)
            lon_min, lon_max = np.min(tlong), np.max(tlong)
        else:
            lat_min, lat_max = 0, 65
            lon_min, lon_max = -80, 0
        lat_grid = np.linspace(lat_min, lat_max, attn_shape[0] + 1)
        lon_grid = np.linspace(lon_min, lon_max, attn_shape[1] + 1)
        return lat_grid, lon_grid

    def _regrid_attention(self, attn, target_shape):
        # Upsample attention map to target_shape using linear interpolation.
        zoom_factors = (target_shape[0] / attn.shape[0], target_shape[1] / attn.shape[1])
        return zoom(attn, zoom_factors, order=1)

    def plot_attention_maps(self, attention_maps, tlat, tlong, save_path=None):
        self.logger.info("Plotting attention maps")
        expected_keys = ['ssh', 'sst', 'cross']
        missing = [k for k in expected_keys if k not in attention_maps]
        if missing:
            self.logger.error(f"Missing keys in attention maps: {missing}")
            return
        fig = plt.figure(figsize=(20, 6))
        titles = ['SSH Attention', 'SST Attention', 'Cross-Modal Attention']
        # Determine target grid dimensions from tlat/tlong or use defaults.
        if tlat is not None and tlong is not None:
            target_shape = (len(tlat), len(tlong))
        else:
            target_shape = (300, 300)
        lat_grid, lon_grid = self._get_geographic_grid(target_shape, tlat, tlong)
        for idx, key in enumerate(expected_keys):
            ax = fig.add_subplot(1, 3, idx+1, projection=self.projection)
            ax.set_extent([lon_grid[0], lon_grid[-1], lat_grid[0], lat_grid[-1]], crs=self.projection)
            ax.add_feature(cfeature.LAND, facecolor='lightgray')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            attn = attention_maps[key]
            if attn is None:
                self.logger.warning(f"No attention data for {key}")
                continue
            # Regrid the patch-level attention map to the geographic grid.
            attn_regridded = self._regrid_attention(attn, target_shape)
            vmin, vmax = np.nanpercentile(attn_regridded, [2, 98])
            mesh = ax.pcolormesh(lon_grid, lat_grid, attn_regridded, cmap='viridis', shading='auto',
                                   vmin=vmin, vmax=vmax, transform=self.projection)
            plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05)
            ax.set_title(titles[idx])
        plt.tight_layout()
        if save_path:
            sp = self.output_dir / 'attention_maps' / f"{save_path}.png"
            plt.savefig(sp, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved attention map to: {sp}")
        else:
            return fig

    def plot_predictions(self, predictions, targets, time_indices=None, save_path=None):
        self.logger.info("Plotting predictions vs targets")
        if time_indices is None:
            time_indices = np.arange(len(predictions))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        ax1.scatter(targets, predictions, alpha=0.6, color='blue')
        lims = [min(np.min(targets), np.min(predictions)), max(np.max(targets), np.max(predictions))]
        ax1.plot(lims, lims, 'r--')
        ax1.set_xlabel('True Heat Transport')
        ax1.set_ylabel('Predicted Heat Transport')
        ax1.set_title('Prediction Scatter')
        ax2.plot(time_indices, targets, 'b-', label='True', alpha=0.7)
        ax2.plot(time_indices, predictions, 'r--', label='Predicted', alpha=0.7)
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Heat Transport')
        ax2.set_title('Time Series Comparison')
        ax2.legend()
        plt.tight_layout()
        if save_path:
            sp = self.output_dir / 'predictions' / f"{save_path}.png"
            plt.savefig(sp, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved predictions plot to: {sp}")
        else:
            return fig

    def plot_error_histogram(self, predictions, targets, save_path=None):
        self.logger.info("Plotting error histogram")
        errors = predictions - targets
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(errors, kde=True, ax=ax, color='orange')
        ax.set_title("Error Distribution")
        ax.set_xlabel("Prediction Error")
        if save_path:
            sp = self.output_dir / 'plots' / f"{save_path}.png"
            plt.savefig(sp, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved error histogram to: {sp}")
        else:
            return fig
