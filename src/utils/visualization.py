import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from pathlib import Path
import logging
from scipy.interpolate import griddata
import xarray as xr
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

class OceanVisualizer:
    def __init__(self, output_dir, fig_size=(12,8), dpi=300):
        self.output_dir = Path(output_dir)
        self.fig_size = fig_size
        self.dpi = dpi
        self.logger = logging.getLogger(__name__)
        
        for s in ['plots', 'attention_maps', 'predictions', 'temporal', 'vnt_plots', 'comparison']:
            (self.output_dir / s).mkdir(parents=True, exist_ok=True)
            
        self.projection = ccrs.PlateCarree()
        
        self.atlantic_bounds = [-80, 0, 30, 65]
        
        self._setup_colormaps()
        
    def _setup_colormaps(self):
        attention_colors = plt.cm.viridis(np.linspace(0, 1, 256))
        attention_colors[:10, 3] = np.linspace(0, 1, 10)
        self.attention_cmap = LinearSegmentedColormap.from_list('attention_cmap', attention_colors)
        
        self.diff_cmap = plt.cm.RdBu_r
        
        self.heat_cmap = plt.cm.plasma
        
    def _get_patch_grid(self, attn_length, patch_dims=None):
        if patch_dims is not None:
            return patch_dims
            
        side = int(round(np.sqrt(attn_length)))
        if side * side != attn_length:
            side = int(np.ceil(np.sqrt(attn_length)))
            
        return (side, side)
        
    def _r(self, a, sh):
        hi, wi = a.shape
        ho, wo = sh
        
        yi = np.linspace(0, 1, hi)
        xi = np.linspace(0, 1, wi)
        xi2, yi2 = np.meshgrid(xi, yi)
        pts = np.vstack([xi2.ravel(), yi2.ravel()]).T
        vals = a.ravel()
        
        yo = np.linspace(0, 1, ho)
        xo = np.linspace(0, 1, wo)
        xo2, yo2 = np.meshgrid(xo, yo)
        opts = np.vstack([xo2.ravel(), yo2.ravel()]).T
        
        r = griddata(pts, vals, opts, method='linear')
        return r.reshape(ho, wo)
        
    def plot_attention_maps(self, attn_dict, save_path=None, show_colorbar=True, title=None):
        if not attn_dict or 'attn' not in attn_dict:
            self.logger.info("No attention maps provided.")
            return
            
        attn_list = attn_dict['attn']
        patch_dims = attn_dict.get('patch_dims', None)
        
        num_layers = len(attn_list)
        if num_layers == 1:
            fig, ax = plt.subplots(figsize=(10,8),
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
                
                attn_map_resized = self._r(attn_map, (100, 100))
                
                vmin = 0
                vmax = attn_mean.max()
                
                im = ax.imshow(
                    attn_map_resized,
                    extent=self.atlantic_bounds,
                    origin='upper',
                    transform=self.projection,
                    cmap=self.attention_cmap,
                    vmin=vmin,
                    vmax=vmax
                )
                
                ax.coastlines(resolution='50m', linewidth=0.8, color='black')
                ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.6)
                ax.add_feature(cfeature.LAND, alpha=0.3, color='tan')
                ax.add_feature(cfeature.OCEAN, alpha=0.1, color='lightblue')
                
                ax.set_extent(self.atlantic_bounds, crs=self.projection)
                
                gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
                gl.top_labels = False
                gl.right_labels = False
                gl.xlabel_style = {'size': 9}
                gl.ylabel_style = {'size': 9}
                
                if num_layers == 1 or ax_idx == 0:
                    ax.text(-0.12, 0.55, 'Latitude', va='bottom', ha='center',
                            rotation='vertical', rotation_mode='anchor',
                            transform=ax.transAxes, fontsize=10)
                if ax_idx == len(axs) - 1 or ax_idx == 0:
                    ax.text(0.5, -0.1, 'Longitude', va='bottom', ha='center',
                            rotation='horizontal', rotation_mode='anchor',
                            transform=ax.transAxes, fontsize=10)
                
                if num_layers > 1:
                    layer_title = f"Attention Layer {ax_idx + 1}"
                    if ax_idx == 0 and title:
                        layer_title = title
                    ax.set_title(layer_title, fontsize=12)
                else:
                    ax.set_title(title if title else "Attention Map", fontsize=12)
                    
            except Exception as e:
                self.logger.error(
                    "Reshape error: cannot reshape array of size %d into shape %s: %s",
                    attn_mean.size, str(grid_shape), str(e)
                )
                return
                
        if show_colorbar:
            cbar = fig.colorbar(im, ax=axs, orientation='horizontal', pad=0.05,
                              label='Attention Weight', shrink=0.8)
            cbar.ax.tick_params(labelsize=9)
                        
        plt.tight_layout()
        
        if save_path:
            sp = self.output_dir / 'attention_maps' / f'{save_path}.png'
            plt.savefig(sp, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved attention map to {sp}")
        else:
            plt.show()
            
    def plot_predictions(self, preds, tg, time_indices=None, save_path=None, title=None):
        if time_indices is None:
            time_indices = np.arange(len(preds))
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.fig_size, height_ratios=[1, 1.5])
        
        scatter = ax1.scatter(tg, preds, alpha=0.6, color='royalblue', edgecolor='navy', s=25)
        
        l1 = min(np.min(tg), np.min(preds))
        l2 = max(np.max(tg), np.max(preds))
        ax1.plot([l1, l2], [l1, l2], 'r--', alpha=0.8, label='Perfect Prediction')
        
        ax1.set_xlabel('True Heat Transport (PW)', fontsize=12)
        ax1.set_ylabel('Predicted Heat Transport (PW)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        if title:
            ax1.set_title(title, fontsize=14, pad=10)
            
        ax2.plot(time_indices, tg, 'b-', label='True', alpha=0.7)
        ax2.plot(time_indices, preds, 'r--', label='Predicted', alpha=0.7)
        
        ax2.set_xlabel('Time Index', fontsize=12)
        ax2.set_ylabel('Heat Transport (PW)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            sp = self.output_dir / 'predictions' / f'{save_path}.png'
            plt.savefig(sp, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved prediction plot to {sp}")
        else:
            plt.show()
            
    def plot_error_histogram(self, preds, tg, save_path=None, title=None):
        errors = preds - tg
        
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        sns.histplot(errors, kde=True, ax=ax, color='royalblue', alpha=0.6)
        
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.8)
        
        ax.set_xlabel('Prediction Error (PW)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        if title:
            ax.set_title(title, fontsize=14)
            
        plt.tight_layout()
        
        if save_path:
            sp = self.output_dir / 'plots' / f'{save_path}.png'
            plt.savefig(sp, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved error histogram to {sp}")
        else:
            plt.show()
            
    def plot_temporal_trends(self, times, true_ht, pred_ht=None, save_path=None, title=None):
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        ax.plot(times, true_ht, 'b-', label='True', alpha=0.7)
        if pred_ht is not None:
            ax.plot(times, pred_ht, 'r--', label='Predicted', alpha=0.7)
            
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Heat Transport (PW)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        if title:
            ax.set_title(title, fontsize=14)
            
        plt.tight_layout()
        
        if save_path:
            sp = self.output_dir / 'temporal' / f'{save_path}.png'
            plt.savefig(sp, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved temporal trends plot to {sp}")
        else:
            plt.show()
            
    def plot_method_comparison(self, direct_pred, vnt_pred, true_values, save_path=None, title=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        scatter1 = ax1.scatter(true_values, direct_pred, alpha=0.6, color='royalblue', edgecolor='navy', s=25)
        scatter2 = ax2.scatter(true_values, vnt_pred, alpha=0.6, color='forestgreen', edgecolor='darkgreen', s=25)
        
        l1 = min(np.min(true_values), np.min(direct_pred), np.min(vnt_pred))
        l2 = max(np.max(true_values), np.max(direct_pred), np.max(vnt_pred))
        
        ax1.plot([l1, l2], [l1, l2], 'r--', alpha=0.8, label='Perfect Prediction')
        ax2.plot([l1, l2], [l1, l2], 'r--', alpha=0.8, label='Perfect Prediction')
        
        ax1.set_xlabel('True Heat Transport (PW)', fontsize=12)
        ax1.set_ylabel('Direct Prediction (PW)', fontsize=12)
        ax2.set_xlabel('True Heat Transport (PW)', fontsize=12)
        ax2.set_ylabel('VNT-based Prediction (PW)', fontsize=12)
        
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        ax1.legend(fontsize=10)
        ax2.legend(fontsize=10)
        
        if title:
            fig.suptitle(title, fontsize=14, y=1.05)
            
        plt.tight_layout()
        
        if save_path:
            sp = self.output_dir / 'comparison' / f'{save_path}.png'
            plt.savefig(sp, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved method comparison plot to {sp}")
        else:
            plt.show()
            
    def plot_vnt_field(self, vnt_pred, vnt_true=None, lat=None, lon=None, depth_idx=0, 
                      save_path=None, title=None):
        if lat is None or lon is None:
            self.logger.error("Latitude and longitude arrays are required for VNT field plotting")
            return
            
        fig, ax = plt.subplots(figsize=self.fig_size,
                              subplot_kw={'projection': self.projection})
        
        if vnt_true is not None:
            vmin = min(np.min(vnt_pred), np.min(vnt_true))
            vmax = max(np.max(vnt_pred), np.max(vnt_true))
        else:
            vmin = np.min(vnt_pred)
            vmax = np.max(vnt_pred)
            
        im = ax.pcolormesh(lon, lat, vnt_pred[depth_idx],
                          transform=self.projection,
                          cmap=self.heat_cmap,
                          vmin=vmin,
                          vmax=vmax)
        
        ax.coastlines(resolution='50m', linewidth=0.8, color='black')
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.6)
        ax.add_feature(cfeature.LAND, alpha=0.3, color='tan')
        ax.add_feature(cfeature.OCEAN, alpha=0.1, color='lightblue')
        
        ax.set_extent(self.atlantic_bounds, crs=self.projection)
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 9}
        gl.ylabel_style = {'size': 9}
        
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05,
                           label='VNT Field', shrink=0.8)
        cbar.ax.tick_params(labelsize=9)
        
        if title:
            ax.set_title(title, fontsize=14)
            
        plt.tight_layout()
        
        if save_path:
            sp = self.output_dir / 'vnt_plots' / f'{save_path}.png'
            plt.savefig(sp, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved VNT field plot to {sp}")
        else:
            plt.show()