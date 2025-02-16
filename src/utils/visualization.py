import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from pathlib import Path
import logging
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
        if tlat is not None and tlong is not None:
            lat_min, lat_max = float(np.min(tlat)), float(np.max(tlat))
            lon_min, lon_max = float(np.min(tlong)), float(np.max(tlong))
        else:
            lat_min, lat_max = 0, 65
            lon_min, lon_max = -80, 0
        lat_grid = np.linspace(lat_min, lat_max, attn_shape[0])
        lon_grid = np.linspace(lon_min, lon_max, attn_shape[1])
        return lat_grid, lon_grid

    def _create_ocean_mask(self, shape, tlat, tlong):
        lat_grid, lon_grid = self._get_geographic_grid(shape, tlat, tlong)
        ocean = np.ones(shape)
        
        land_mask = cfeature.NaturalEarthFeature(
            'physical', 'land', '50m').intersecting_geometries([lon_grid[0], lon_grid[-1], 
                                                              lat_grid[0], lat_grid[-1]])
        
        for geom in land_mask:
            pts = np.array(geom.exterior.coords)
            lon_idx = np.digitize(pts[:, 0], lon_grid) - 1
            lat_idx = np.digitize(pts[:, 1], lat_grid) - 1
            valid_idx = (lon_idx >= 0) & (lon_idx < shape[1]) & (lat_idx >= 0) & (lat_idx < shape[0])
            if np.any(valid_idx):
                ocean[lat_idx[valid_idx], lon_idx[valid_idx]] = 0
                
        ocean[:, :70] = 0  # Mask regions west of -70°W
        ocean[:, 340:] = 0  # Mask regions east of 20°E
        ocean[:20, :] = 0   # Mask tropical regions
        ocean[160:, :] = 0  # Mask southern regions
        
        return ocean

    def _regrid_attention(self, attn, target_shape, ocean_mask=None):
        if attn.shape == target_shape:
            regridded = attn
        else:
            zoom_factors = (target_shape[0] / attn.shape[0], 
                          target_shape[1] / attn.shape[1])
            regridded = zoom(attn, zoom_factors, order=1, mode='nearest')
        
        if ocean_mask is not None:
            regridded = regridded * ocean_mask
            
        return regridded

    def plot_attention_maps(self, attention_maps, tlat, tlong, save_path=None):
        self.logger.info("Plotting attention maps")
        if attention_maps is None:
            self.logger.error("No attention maps provided")
            return

        expected_keys = ['ssh', 'sst', 'cross']
        missing = [k for k in expected_keys if k not in attention_maps]
        if missing:
            self.logger.error(f"Missing keys in attention maps: {missing}")
            return

        fig = plt.figure(figsize=(20, 6))
        titles = ['SSH Attention', 'SST Attention', 'Cross-Modal Attention']

        target_shape = (180, 360) if tlat is None else (len(tlat), len(tlong))
        lat_grid, lon_grid = self._get_geographic_grid(target_shape, tlat, tlong)
        ocean_mask = self._create_ocean_mask(target_shape, tlat, tlong)

        for idx, key in enumerate(expected_keys):
            ax = fig.add_subplot(1, 3, idx+1, projection=self.projection)
            
            ax.set_extent([lon_grid[0], lon_grid[-1], lat_grid[0], lat_grid[-1]], 
                         crs=self.projection)
            
            ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=2)
            ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

            attn = attention_maps[key].squeeze()
            if len(attn.shape) > 2:
                attn = attn.mean(axis=0)

            attn_regridded = self._regrid_attention(attn, target_shape, ocean_mask)
            
            # Focus on North Atlantic
            na_mask = np.zeros_like(ocean_mask)
            na_mask[40:120, 120:280] = 1  # Approximate North Atlantic region
            attn_regridded = attn_regridded * na_mask
            
            valid_attention = attn_regridded[attn_regridded > 0]
            if len(valid_attention) > 0:
                vmin, vmax = np.percentile(valid_attention, [5, 95])
            else:
                vmin, vmax = 0, 1

            mesh = ax.pcolormesh(lon_grid, lat_grid, attn_regridded,
                               transform=self.projection,
                               cmap='viridis',
                               vmin=vmin, vmax=vmax,
                               shading='auto')
            
            plt.colorbar(mesh, ax=ax, orientation='horizontal', 
                        pad=0.05, label='Attention Weight')
            
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
        lims = [
            min(np.min(targets), np.min(predictions)),
            max(np.max(targets), np.max(predictions))
        ]
        ax1.plot(lims, lims, 'r--')
        ax1.set_xlabel('True Heat Transport')
        ax1.set_ylabel('Predicted Heat Transport')
        ax1.set_title('Prediction Scatter')
        ax1.grid(True)

        ax2.plot(time_indices, targets, 'b-', label='True', alpha=0.7)
        ax2.plot(time_indices, predictions, 'r--', label='Predicted', alpha=0.7)
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Heat Transport')
        ax2.set_title('Time Series Comparison')
        ax2.legend()
        ax2.grid(True)

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
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.set_title("Error Distribution")
        ax.set_xlabel("Prediction Error")
        ax.grid(True)

        if save_path:
            sp = self.output_dir / 'plots' / f"{save_path}.png"
            plt.savefig(sp, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved error histogram to: {sp}")
        else:
            return fig