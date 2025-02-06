import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import shapely
from pathlib import Path
import cmocean
import logging
from typing import Optional, Tuple, Union

class OceanVisualizer:
    def __init__(self, output_dir: Union[str, Path], fig_size: tuple = (20, 8), dpi: int = 300):
        self.output_dir = Path(output_dir)
        self.fig_size = fig_size
        self.dpi = dpi
        self.logger = logging.getLogger(__name__)
        self.projection = ccrs.PlateCarree()
        for dir_name in ['plots', 'attention_maps', 'predictions']:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)

    def _setup_ocean_map(self, ax):
        ax.set_extent([-80, 0, 10, 60], crs=self.projection)
        land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                            edgecolor='black', facecolor='lightgray')
        ax.add_feature(land, zorder=2)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                        alpha=0.5, clip_box=ax.bbox)
        gl.top_labels = False
        gl.right_labels = False
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8, zorder=3)
        return gl

    def plot_attention_maps(self, attention_maps, tlat, tlong, save_path=None):
        self.logger.info(f"Plotting attention maps")
        
        ssh_attn = attention_maps['ssh'].mean(0).cpu().numpy()
        sst_attn = attention_maps['sst'].mean(0).cpu().numpy()
        cross_attn = attention_maps['cross'].mean(0).cpu().numpy()
        
        h = int(np.sqrt(ssh_attn.shape[0]))
        w = h
        
        lons = np.linspace(-80, 0, w)
        lats = np.linspace(0, 65, h)
        lon_mesh, lat_mesh = np.meshgrid(lons, lats)
        
        land = cfeature.NaturalEarthFeature('physical', 'land', '50m')
        ocean_mask = np.ones((h, w), dtype=bool)
        points = np.column_stack((lon_mesh.ravel(), lat_mesh.ravel()))
        for geom in land.geometries():
            ocean_mask.ravel()[:] &= ~np.array([shapely.Point(p).within(geom) for p in points])
        
        # Setup figure
        fig = plt.figure(figsize=(20, 6))
        gs = plt.GridSpec(1, 3, width_ratios=[1, 1, 1])
        
        titles = ['SSH Attention', 'SST Attention', 'Cross-Modal Attention']
        attention_data = [
            ssh_attn.reshape(h, w),
            sst_attn.reshape(h, w),
            cross_attn.reshape(h, w)
        ]
        
        for idx, (attn, title) in enumerate(zip(attention_data, titles)):
            attn = np.ma.masked_array(attn, mask=~ocean_mask)
            
            ax = fig.add_subplot(gs[idx], projection=self.projection)
            ax.set_extent([-80, 0, 0, 65], crs=self.projection)
            ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False
            
            vmin, vmax = np.nanpercentile(attn.compressed(), [2, 98])
            mesh = ax.pcolormesh(
                lon_mesh, lat_mesh, attn,
                transform=self.projection,
                cmap='viridis',
                vmin=vmin,
                vmax=vmax,
                shading='auto'
            )
            plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05,
                        label='Attention Strength')
            ax.set_title(title)
        
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
        data_np = data.values if hasattr(data, 'values') else np.array(data)
        tlat_np = tlat.values if hasattr(tlat, 'values') else np.array(tlat)
        tlong_np = tlong.values if hasattr(tlong, 'values') else np.array(tlong)
        data_masked = np.ma.masked_invalid(data_np)
        kernel_size = 3
        for i in range(1, data_np.shape[0]-1):
            for j in range(1, data_np.shape[1]-1):
                neighborhood = data_np[i-1:i+2, j-1:j+2]
                if not np.any(np.isnan(neighborhood)):
                    diag1 = np.abs(neighborhood[0,0] - neighborhood[2,2])
                    diag2 = np.abs(neighborhood[0,2] - neighborhood[2,0])
                    horiz = np.abs(neighborhood[1,0] - neighborhood[1,2])
                    vert = np.abs(neighborhood[0,1] - neighborhood[2,1])
                    if max(diag1, diag2) > 5 * min(horiz, vert):
                        data_masked.mask[i,j] = True
                        
        valid_data = data_masked.compressed()
        vmin, vmax = np.nanpercentile(valid_data, [2, 98])
        self._setup_ocean_map(ax)
        
        mesh = ax.pcolormesh(
            tlong_np, tlat_np, data_masked,
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
        
        ax2.plot(time_indices, targets, 'b-', label='True', alpha=0.7)
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
            self.logger.info(f"Saved predictions plot to {save_path}")
            return None
        return fig