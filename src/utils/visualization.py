import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
from pathlib import Path
import cmocean
import logging
from datetime import datetime
import matplotlib.colors as colors

class OceanVisualizer:
    def __init__(self, output_dir: str, fig_size: tuple = (12, 8), dpi: int = 300):
        """Initialize the ocean data visualizer."""
        self.output_dir = Path(output_dir)
        self.fig_size = fig_size
        self.dpi = dpi
        self.logger = logging.getLogger(__name__)
        
        self.plot_dir = self.output_dir / 'plots'
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_plotting_style()
    
    def _setup_plotting_style(self):
        """Configure matplotlib styling"""
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
        
        sns.set_palette("deep")
        
        self.logger.info("Plotting style configured")

    def plot_spatial_pattern(self, data, tlat, tlong, title, cmap='cmo.thermal', save_path=None):
        """Plot spatial pattern with TLAT/TLONG coordinates for Atlantic Ocean"""
        self.logger.info(f"Creating spatial plot: {title}")
        self.logger.info(f"Data shape: {data.shape}")
        
        try:
            with data.load() as loaded_data:
                valid_data = loaded_data.where(~loaded_data.isnull())
                vmin = float(valid_data.min())
                vmax = float(valid_data.max())
                self.logger.info(f"Data range: {vmin} to {vmax}")
            
            fig = plt.figure(figsize=self.fig_size)
            ax = plt.axes(projection=ccrs.PlateCarree())
            
            ax.coastlines(resolution='50m')
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.gridlines(draw_labels=True)
            
            valid_mask = ~np.isnan(data)
            valid_tlat = tlat[valid_mask]
            valid_tlong = tlong[valid_mask]
            valid_data = data[valid_mask]
            
            self.logger.info(f"TLAT range of valid data: {float(valid_tlat.min())} to {float(valid_tlat.max())}")
            self.logger.info(f"TLONG range of valid data: {float(valid_tlong.min())} to {float(valid_tlong.max())}")
            
            im = ax.scatter(
                valid_tlong, valid_tlat, 
                c=valid_data,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                s=1  # point size
            )
            
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
            cbar.set_label(title)
            
            plt.title(title)
            
            if save_path:
                save_path = self.plot_dir / f"{save_path}.png"
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                self.logger.info(f"Plot saved to {save_path}")
            
            return fig
                
        except Exception as e:
            self.logger.error(f"Error in spatial plotting: {str(e)}", exc_info=True)
            raise

    def plot_time_series(self, data, title, ylabel, save_path=None):
        """Plot time series with enhanced error handling"""
        self.logger.info(f"Creating time series plot: {title}")
        
        try:
            fig, ax = plt.subplots(figsize=self.fig_size)
            
            ax.plot(data.time, data.values, linewidth=2, label='Data')
            
            z = np.polyfit(range(len(data)), data.values, 1)
            p = np.poly1d(z)
            ax.plot(data.time, p(range(len(data))), 
                    '--', color='red', 
                    label=f'Trend: {z[0]:.2e} per year')
            
            std = data.values.std()
            ax.fill_between(
                data.time,
                data.values - 2*std,
                data.values + 2*std,
                alpha=0.2,
                label='95% Confidence'
            )
            
            ax.set_title(title)
            ax.set_xlabel('Time')
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if save_path:
                save_path = self.plot_dir / f"{save_path}.png"
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                self.logger.info(f"Plot saved to {save_path}")
            return fig
            
        except Exception as e:
            self.logger.error(f"Error in time series plotting: {str(e)}", exc_info=True)
            raise                                                                                                                                                                                                                                                                    