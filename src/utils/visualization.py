import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from pathlib import Path
import logging
from scipy.interpolate import griddata
import xarray as xr

class OceanVisualizer:
    def __init__(self, output_dir, fig_size=(12,8), dpi=300):
        self.output_dir = Path(output_dir)
        self.fig_size = fig_size
        self.dpi = dpi
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        for s in ['plots', 'attention_maps', 'predictions', 'temporal']:
            (self.output_dir / s).mkdir(parents=True, exist_ok=True)
            
        self.projection = ccrs.PlateCarree()
        self.atlantic_bounds = [-80, 0, 30, 65]  # [lon_min, lon_max, lat_min, lat_max]

    def _get_patch_grid(self, attn_length, patch_dims=None):
        if patch_dims is not None:
            return patch_dims
        side = int(round(np.sqrt(attn_length)))
        if side * side != attn_length:
            side = int(np.ceil(np.sqrt(attn_length)))
        return (side, side)

    def _r(self, a, sh):
        """Resample attention map to target shape using linear interpolation."""
        hi, wi = a.shape
        ho, wo = sh
        yi = np.linspace(0, 1, hi)
        xi = np.linspace(0, 1, wi)
        yo = np.linspace(0, 1, ho)
        xo = np.linspace(0, 1, wo)
        
        xi2, yi2 = np.meshgrid(xi, yi)
        pts = np.vstack([xi2.ravel(), yi2.ravel()]).T
        vals = a.ravel()
        
        xo2, yo2 = np.meshgrid(xo, yo)
        opts = np.vstack([xo2.ravel(), yo2.ravel()]).T
        
        r = griddata(pts, vals, opts, method='linear')
        return r.reshape(ho, wo)

    def plot_attention_maps(self, attn_dict, save_path=None, show_colorbar=True):
        """Plot attention maps with improved Atlantic Ocean visualization."""
        if not attn_dict or 'attn' not in attn_dict:
            self.logger.info("No attention maps provided.")
            return
            
        attn_list = attn_dict['attn']
        patch_dims = attn_dict.get('patch_dims', None)
        num_layers = len(attn_list)

        if num_layers == 1:
            fig, ax = plt.subplots(figsize=(6,6), 
                                 subplot_kw={'projection': self.projection})
            axs = [ax]
        else:
            fig, axs = plt.subplots(1, num_layers, 
                                  figsize=(6*num_layers,6), 
                                  subplot_kw={'projection': self.projection})
            if num_layers == 1:
                axs = [axs]

        for ax_idx, (ax, attn) in enumerate(zip(axs, attn_list)):
            attn_mean = attn[0].detach().cpu().numpy()
            
            if patch_dims is not None:
                grid_shape = patch_dims
            else:
                grid_shape = self._get_patch_grid(attn_mean.size)
                
            try:
                attn_map = attn_mean.reshape(grid_shape)
            except Exception as e:
                self.logger.error(
                    "Reshape error: cannot reshape array of size %d into shape %s", 
                    attn_mean.size, str(grid_shape)
                )
                return

            attn_map_resized = self._r(attn_map, (100, 100))
            
            im = ax.imshow(
                attn_map_resized, 
                extent=self.atlantic_bounds,
                origin='upper',
                transform=self.projection,
                cmap='viridis'
            )

            # Add geographic features
            ax.coastlines(resolution='50m')
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.add_feature(cfeature.LAND, alpha=0.3)
            ax.add_feature(cfeature.OCEAN)
            
            # Set extent to Atlantic Ocean
            ax.set_extent(self.atlantic_bounds, crs=self.projection)
            
            # Add gridlines
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False
            
            if num_layers > 1:
                ax.set_title(f"Attention Layer {ax_idx + 1}")
            else:
                ax.set_title("Attention Map")

        if show_colorbar:
            fig.colorbar(im, ax=axs, orientation='horizontal', pad=0.05,
                        label='Attention Weight')

        plt.tight_layout()

        if save_path:
            sp = self.output_dir / 'attention_maps' / f'{save_path}.png'
            plt.savefig(sp, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved attention map to {sp}")
        else:
            plt.show()

    def plot_predictions(self, preds, tg, time_indices=None, save_path=None):
        """Plot prediction results with scatter and time series plots."""
        if time_indices is None:
            time_indices = np.arange(len(preds))
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.fig_size)
        
        # Scatter plot
        ax1.scatter(tg, preds, alpha=0.5, color='blue')
        l1 = min(np.min(tg), np.min(preds))
        l2 = max(np.max(tg), np.max(preds))
        ax1.plot([l1, l2], [l1, l2], 'r--')
        ax1.set_xlabel("True Heat Transport")
        ax1.set_ylabel("Predicted Heat Transport")
        ax1.grid(True)
        
        # Time series plot
        ax2.plot(time_indices, tg, 'b-', label='True', alpha=0.7)
        ax2.plot(time_indices, preds, 'r--', label='Predicted', alpha=0.7)
        ax2.set_xlabel("Time Index")
        ax2.set_ylabel("Heat Transport")
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            sp = self.output_dir / 'predictions' / f'{save_path}.png'
            plt.savefig(sp, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved predictions plot to {sp}")
        else:
            plt.show()

    def plot_error_histogram(self, preds, tg, save_path=None):
        """Plot histogram of prediction errors."""
        e = preds - tg
        fig, ax = plt.subplots(figsize=(8,6))
        
        sns.histplot(e, kde=True, color='orange', ax=ax)
        ax.axvline(x=0, color='r', linestyle='--')
        ax.set_xlabel("Prediction Error")
        ax.set_ylabel("Frequency")
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            sp = self.output_dir / 'plots' / f'{save_path}.png'
            plt.savefig(sp, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved error histogram to {sp}")
        else:
            plt.show()

    def plot_temporal_trends(self, times, true_ht, pred_ht=None, save_path=None):
        """Plot temporal trends of heat transport."""
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        ax.plot(times, true_ht, 'b-', label='True Heat Transport', alpha=0.7)
        if pred_ht is not None:
            ax.plot(times, pred_ht, 'r--', label='Predicted Heat Transport', alpha=0.7)
            
        ax.set_xlabel("Time")
        ax.set_ylabel("Heat Transport")
        ax.set_title("Temporal Trends of Heat Transport")
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            sp = self.output_dir / 'temporal' / f'{save_path}.png'
            plt.savefig(sp, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved temporal trends plot to {sp}")
        else:
            plt.show()