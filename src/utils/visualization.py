import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from pathlib import Path
import logging
from scipy.interpolate import griddata

class OceanVisualizer:
    def __init__(self, output_dir, fig_size=(12, 8), dpi=300):
        self.output_dir = Path(output_dir)
        self.fig_size = fig_size
        self.dpi = dpi
        self.logger = logging.getLogger(__name__)
        for sub in ['plots', 'attention_maps', 'predictions']:
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)
        self.projection = ccrs.PlateCarree()

    def _get_geographic_grid(self, shape, tlat, tlong):
        # shape = (H, W)
        if tlat is not None and tlong is not None:
            lat_min, lat_max = float(np.min(tlat)), float(np.max(tlat))
            lon_min, lon_max = float(np.min(tlong)), float(np.max(tlong))
        else:
            lat_min, lat_max = 0, 65
            lon_min, lon_max = -80, 0
        lat_grid = np.linspace(lat_min, lat_max, shape[0])
        lon_grid = np.linspace(lon_min, lon_max, shape[1])
        return lat_grid, lon_grid

    def _regrid_attention(self, attn, out_shape):
        # attn: [H_in, W_in]
        # regrid to out_shape via griddata
        H_in, W_in = attn.shape
        H_out, W_out = out_shape
        y_in = np.linspace(0, 1, H_in)
        x_in = np.linspace(0, 1, W_in)
        y_out = np.linspace(0, 1, H_out)
        x_out = np.linspace(0, 1, W_out)
        # Flatten
        xx_in, yy_in = np.meshgrid(x_in, y_in)
        points = np.vstack([xx_in.ravel(), yy_in.ravel()]).T
        values = attn.ravel()
        xx_out, yy_out = np.meshgrid(x_out, y_out)
        out_points = np.vstack([xx_out.ravel(), yy_out.ravel()]).T
        attn_out = griddata(points, values, out_points, method='linear')
        attn_out = attn_out.reshape(H_out, W_out)
        return attn_out

    def plot_attention_maps(self, attention_maps, tlat=None, tlong=None, save_path=None):
        if not attention_maps:
            self.logger.error("No attention maps provided.")
            return
        keys = list(attention_maps.keys())
        fig, axs = plt.subplots(1, len(keys), figsize=(5*len(keys), 5),
                                subplot_kw={'projection': self.projection})
        if len(keys) == 1:
            axs = [axs]
        # Decide the final shape to regrid onto
        # e.g. if you want 180x360 or if you have lat/lon arrays
        # We'll do 120x200 just as an example
        out_shape = (120, 200)
        lat_grid, lon_grid = self._get_geographic_grid(out_shape, tlat, tlong)
        for ax, k in zip(axs, keys):
            attn = attention_maps[k]
            if attn.ndim > 2:
                attn = attn.mean(axis=0)
            # Suppose attn is [n_tokens,] or [n_tokens, n_tokens], we keep a 2D final
            # Here we assume it's [H_in, W_in]. If it's 1D, you need to sqrt, etc.
            # We'll assume you've carefully shaped attn outside or it's 2D already
            regridded = self._regrid_attention(attn, out_shape)
            im = ax.pcolormesh(lon_grid, lat_grid, regridded, transform=self.projection,
                               cmap='viridis', shading='auto')
            ax.add_feature(cfeature.LAND, zorder=99, edgecolor='black', facecolor='lightgray')
            ax.coastlines()
            ax.set_title(k.upper())
            plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
        plt.tight_layout()
        if save_path:
            sp = self.output_dir / "attention_maps" / f"{save_path}.png"
            plt.savefig(sp, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved attention map to {sp}")
        else:
            plt.show()

    def plot_predictions(self, predictions, targets, time_indices=None, save_path=None):
        self.logger.info("Plotting predictions vs. targets")
        if time_indices is None:
            time_indices = np.arange(len(predictions))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        # scatter
        ax1.scatter(targets, predictions, alpha=0.5, color='blue')
        lims = [min(np.min(targets), np.min(predictions)),
                max(np.max(targets), np.max(predictions))]
        ax1.plot(lims, lims, 'r--')
        ax1.set_xlabel("True Heat Transport")
        ax1.set_ylabel("Predicted Heat Transport")
        ax1.set_title("Scatter: True vs Predicted")
        ax1.grid(True)
        # time series
        ax2.plot(time_indices, targets, 'b-', label='True')
        ax2.plot(time_indices, predictions, 'r--', label='Predicted')
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Heat Transport")
        ax2.set_title("Time Series: True vs Predicted")
        ax2.grid(True)
        ax2.legend()
        plt.tight_layout()
        if save_path:
            sp = self.output_dir / "predictions" / f"{save_path}.png"
            plt.savefig(sp, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved predictions plot to {sp}")
        else:
            plt.show()

    def plot_error_histogram(self, predictions, targets, save_path=None):
        errors = predictions - targets
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(errors, kde=True, color='orange', ax=ax)
        ax.axvline(x=0, color='r', linestyle='--')
        ax.set_title("Error Distribution")
        ax.set_xlabel("Prediction Error")
        ax.grid(True)
        if save_path:
            sp = self.output_dir / "plots" / f"{save_path}.png"
            plt.savefig(sp, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved error histogram to {sp}")
        else:
            plt.show()
