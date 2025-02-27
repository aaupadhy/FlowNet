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
    """
    Visualization tools for ocean data and model outputs
    
    Features:
    - Attention map visualization with geographic context
    - Prediction performance visualization
    - Error analysis visualizations
    - VNT field visualization
    - Time series analysis
    """
    def __init__(self, output_dir, fig_size=(12,8), dpi=300):
        self.output_dir = Path(output_dir)
        self.fig_size = fig_size
        self.dpi = dpi
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        for s in ['plots', 'attention_maps', 'predictions', 'temporal', 'vnt_plots', 'comparison']:
            (self.output_dir / s).mkdir(parents=True, exist_ok=True)
            
        # Set up map projection for geospatial plots
        self.projection = ccrs.PlateCarree()
        
        # Default region of interest (North Atlantic)
        self.atlantic_bounds = [-80, 0, 30, 65]
        
        # Set up custom colormaps
        self._setup_colormaps()
        
    def _setup_colormaps(self):
        """Set up custom colormaps for different visualization types"""
        # Custom colormap for attention visualization
        attention_colors = plt.cm.viridis(np.linspace(0, 1, 256))
        attention_colors[:10, 3] = np.linspace(0, 1, 10)  # Add alpha gradient at low values
        self.attention_cmap = LinearSegmentedColormap.from_list('attention_cmap', attention_colors)
        
        # Custom diverging colormap for difference plots
        self.diff_cmap = plt.cm.RdBu_r
        
        # Custom thermal colormap for ocean heat
        self.heat_cmap = plt.cm.plasma
        
    def _get_patch_grid(self, attn_length, patch_dims=None):
        """
        Determine grid dimensions for reshaping attention maps
        
        Args:
            attn_length: Length of flattened attention map
            patch_dims: Optional explicit patch dimensions
            
        Returns:
            Tuple of (height, width) dimensions
        """
        if patch_dims is not None:
            return patch_dims
            
        # Estimate grid dimensions from attention length
        side = int(round(np.sqrt(attn_length)))
        if side * side != attn_length:
            side = int(np.ceil(np.sqrt(attn_length)))
            
        return (side, side)
        
    def _r(self, a, sh):
        """
        Resize/Interpolate a 2D array to target shape using griddata
        
        Args:
            a: Input 2D array
            sh: Target shape (height, width)
            
        Returns:
            Interpolated array of shape sh
        """
        hi, wi = a.shape
        ho, wo = sh
        
        # Create source coordinates (normalized grid)
        yi = np.linspace(0, 1, hi)
        xi = np.linspace(0, 1, wi)
        xi2, yi2 = np.meshgrid(xi, yi)
        pts = np.vstack([xi2.ravel(), yi2.ravel()]).T
        vals = a.ravel()
        
        # Create target coordinates (normalized grid)
        yo = np.linspace(0, 1, ho)
        xo = np.linspace(0, 1, wo)
        xo2, yo2 = np.meshgrid(xo, yo)
        opts = np.vstack([xo2.ravel(), yo2.ravel()]).T
        
        # Interpolate using griddata
        r = griddata(pts, vals, opts, method='linear')
        return r.reshape(ho, wo)
        
    def plot_attention_maps(self, attn_dict, save_path=None, show_colorbar=True, title=None):
        """
        Plot attention maps with proper geographic projection and coastlines
        
        Args:
            attn_dict: Dictionary containing attention maps and dimensions
            save_path: Path to save the plot
            show_colorbar: Whether to show colorbar
            title: Optional title for the plot
        """
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
            # Extract attention weights
            attn_mean = attn[0].detach().cpu().numpy()
            
            # Determine grid shape for reshaping
            if patch_dims is not None:
                grid_shape = patch_dims
            else:
                grid_shape = self._get_patch_grid(attn_mean.size)
                
            try:
                # Reshape attention weights to 2D grid
                attn_map = attn_mean.reshape(grid_shape)
                
                # Interpolate to higher resolution for smoother visualization
                attn_map_resized = self._r(attn_map, (100, 100))
                
                # Get attention value range for consistent color scaling
                vmin = 0
                vmax = attn_mean.max()
                
                # Plot attention map with proper projection
                im = ax.imshow(
                    attn_map_resized,
                    extent=self.atlantic_bounds,
                    origin='upper',
                    transform=self.projection,
                    cmap=self.attention_cmap,
                    vmin=vmin,
                    vmax=vmax
                )
                
                # Add geographic features
                ax.coastlines(resolution='50m', linewidth=0.8, color='black')
                ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.6)
                ax.add_feature(cfeature.LAND, alpha=0.3, color='tan')
                ax.add_feature(cfeature.OCEAN, alpha=0.1, color='lightblue')
                
                # Set map extent
                ax.set_extent(self.atlantic_bounds, crs=self.projection)
                
                # Add gridlines with labels
                gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
                gl.top_labels = False
                gl.right_labels = False
                gl.xlabel_style = {'size': 9}
                gl.ylabel_style = {'size': 9}
                
                # Add lat/lon labels
                if num_layers == 1 or ax_idx == 0:
                    ax.text(-0.12, 0.55, 'Latitude', va='bottom', ha='center',
                            rotation='vertical', rotation_mode='anchor',
                            transform=ax.transAxes, fontsize=10)
                if ax_idx == len(axs) - 1 or ax_idx == 0:
                    ax.text(0.5, -0.1, 'Longitude', va='bottom', ha='center',
                            rotation='horizontal', rotation_mode='anchor',
                            transform=ax.transAxes, fontsize=10)
                
                # Add title
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
                
        # Add colorbar if requested
        if show_colorbar:
            cbar = fig.colorbar(im, ax=axs, orientation='horizontal', pad=0.05,
                              label='Attention Weight', shrink=0.8)
            cbar.ax.tick_params(labelsize=9)
                        
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            sp = self.output_dir / 'attention_maps' / f'{save_path}.png'
            plt.savefig(sp, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved attention map to {sp}")
        else:
            plt.show()
            
    def plot_predictions(self, preds, tg, time_indices=None, save_path=None, title=None):
        """
        Plot scatter and time series of predictions vs ground truth
        
        Args:
            preds: Predicted values
            tg: Ground truth values
            time_indices: Optional time indices for x-axis
            save_path: Path to save the plot
            title: Optional title for the plot
        """
        if time_indices is None:
            time_indices = np.arange(len(preds))
            
        # Create figure with two subplots: scatter and time series
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.fig_size, height_ratios=[1, 1.5])
        
        # Scatter plot of predictions vs ground truth
        scatter = ax1.scatter(tg, preds, alpha=0.6, color='royalblue', edgecolor='navy', s=25)
        
        # Add diagonal line for perfect predictions
        l1 = min(np.min(tg), np.min(preds))
        l2 = max(np.max(tg), np.max(preds))
        margin = (l2 - l1) * 0.05  # Add 5% margin
        ax1.plot([l1-margin, l2+margin], [l1-margin, l2+margin], 'r--', alpha=0.7, linewidth=1.5)
        
        # Set axis limits with margin
        ax1.set_xlim(l1-margin, l2+margin)
        ax1.set_ylim(l1-margin, l2+margin)
        
        # Add R² value to scatter plot
        r2 = 1 - np.sum((preds - tg) ** 2) / np.sum((tg - np.mean(tg)) ** 2)
        rmse = np.sqrt(np.mean((preds - tg) ** 2))
        
        # Add stats text box
        stats_text = f"$R^2 = {r2:.4f}$\nRMSE = {rmse:.4f}"
        ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, 
                 va='top', ha='left', bbox=dict(boxstyle='round,pad=0.5', 
                                               facecolor='white', alpha=0.8),
                 fontsize=10)
        
        # Configure scatter plot
        ax1.set_xlabel("True Heat Transport", fontsize=10)
        ax1.set_ylabel("Predicted Heat Transport", fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=9)
        
        # Time series plot
        ax2.plot(time_indices, tg, 'b-', label='True', alpha=0.7, linewidth=1.5)
        ax2.plot(time_indices, preds, 'r--', label='Predicted', alpha=0.7, linewidth=1.5)
        
        # Highlight errors with shaded region
        ax2.fill_between(time_indices, tg, preds, alpha=0.2, color='purple')
        
        # Configure time series plot
        ax2.set_xlabel("Time Index", fontsize=10)
        ax2.set_ylabel("Heat Transport", fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=9)
        ax2.legend(frameon=True, fontsize=9)
        
        # Add overall title if provided
        if title:
            fig.suptitle(title, fontsize=14, y=0.98)
            
        # Adjust spacing between subplots
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.3)
        
        # Save or show plot
        if save_path:
            sp = self.output_dir / 'predictions' / f'{save_path}.png'
            plt.savefig(sp, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved predictions plot to {sp}")
        else:
            plt.show()
            
    def plot_error_histogram(self, preds, tg, save_path=None, title=None):
        """
        Plot histogram of prediction errors
        
        Args:
            preds: Predicted values
            tg: Ground truth values
            save_path: Path to save the plot
            title: Optional title for the plot
        """
        # Calculate errors
        errors = preds - tg
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram with kernel density estimate
        sns.histplot(errors, kde=True, color='royalblue', ax=ax, bins=30,
                    edgecolor='navy', alpha=0.7, line_kws={'linewidth': 2})
        
        # Add vertical line at zero
        ax.axvline(x=0, color='r', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Add statistics
        mean_err = np.mean(errors)
        std_err = np.std(errors)
        median_err = np.median(errors)
        
        # Add stats text box
        stats_text = (f"Mean error: {mean_err:.4f}\n"
                     f"Median error: {median_err:.4f}\n"
                     f"Std error: {std_err:.4f}")
        
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                va='top', ha='right', bbox=dict(boxstyle='round,pad=0.5', 
                                              facecolor='white', alpha=0.8),
                fontsize=10)
        
        # Configure plot
        ax.set_xlabel("Prediction Error (Predicted - True)", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
        
        # Set x-axis limits symmetrically
        max_abs_error = max(abs(np.min(errors)), abs(np.max(errors)))
        margin = max_abs_error * 0.1  # Add 10% margin
        ax.set_xlim(-max_abs_error-margin, max_abs_error+margin)
        
        # Add title if provided
        if title:
            ax.set_title(title, fontsize=14)
            
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            sp = self.output_dir / 'plots' / f'{save_path}.png'
            plt.savefig(sp, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved error histogram to {sp}")
        else:
            plt.show()
            
    def plot_temporal_trends(self, times, true_ht, pred_ht=None, save_path=None, title=None):
        """
        Plot temporal trends of heat transport
        
        Args:
            times: Time indices or timestamps
            true_ht: True heat transport values
            pred_ht: Predicted heat transport values (optional)
            save_path: Path to save the plot
            title: Optional title for the plot
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Plot true heat transport
        ax.plot(times, true_ht, 'b-', label='True Heat Transport', alpha=0.7, linewidth=1.5)
        
        # Plot predicted heat transport if provided
        if pred_ht is not None:
            ax.plot(times, pred_ht, 'r--', label='Predicted Heat Transport', alpha=0.7, linewidth=1.5)
            
            # Add error statistics
            rmse = np.sqrt(np.mean((pred_ht - true_ht) ** 2))
            r2 = 1 - np.sum((pred_ht - true_ht) ** 2) / np.sum((true_ht - np.mean(true_ht)) ** 2)
            corr = np.corrcoef(true_ht, pred_ht)[0, 1]
            
            # Add stats text box
            stats_text = (f"RMSE: {rmse:.4f}\n"
                         f"R²: {r2:.4f}\n"
                         f"Correlation: {corr:.4f}")
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                    va='top', ha='left', bbox=dict(boxstyle='round,pad=0.5', 
                                                 facecolor='white', alpha=0.8),
                    fontsize=10)
        
        # Configure plot
        ax.set_xlabel("Time", fontsize=11)
        ax.set_ylabel("Heat Transport", fontsize=11)
        ax.set_title(title if title else "Temporal Trends of Heat Transport", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
        ax.legend(frameon=True, fontsize=10)
        
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            sp = self.output_dir / 'temporal' / f'{save_path}.png'
            plt.savefig(sp, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved temporal trends plot to {sp}")
        else:
            plt.show()
            
    def plot_method_comparison(self, direct_pred, vnt_pred, true_values, save_path=None, title=None):
        """
        Compare direct prediction with VNT-based prediction
        
        Args:
            direct_pred: Direct model predictions
            vnt_pred: VNT-based predictions
            true_values: Ground truth values
            save_path: Path to save the plot
            title: Optional title for the plot
        """
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), height_ratios=[1, 1, 1])
        
        # Time indices
        times = np.arange(len(true_values))
        
        # Plot 1: Direct prediction vs ground truth
        ax1.plot(times, true_values, 'b-', label='True Values', alpha=0.7, linewidth=1.5)
        ax1.plot(times, direct_pred, 'r--', label='Direct Prediction', alpha=0.7, linewidth=1.5)
        
        # Direct prediction stats
        direct_rmse = np.sqrt(np.mean((direct_pred - true_values) ** 2))
        direct_r2 = 1 - np.sum((direct_pred - true_values) ** 2) / np.sum((true_values - np.mean(true_values)) ** 2)
        
        # Add stats text box
        direct_stats = (f"Direct Prediction:\n"
                       f"RMSE: {direct_rmse:.4f}\n"
                       f"R²: {direct_r2:.4f}")
        
        ax1.text(0.05, 0.95, direct_stats, transform=ax1.transAxes, 
                va='top', ha='left', bbox=dict(boxstyle='round,pad=0.5', 
                                            facecolor='white', alpha=0.8),
                fontsize=9)
        
        ax1.set_title("Direct Prediction vs Ground Truth", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=True, fontsize=9)
        
        # Plot 2: VNT-based prediction vs ground truth
        ax2.plot(times, true_values, 'b-', label='True Values', alpha=0.7, linewidth=1.5)
        ax2.plot(times, vnt_pred, 'g--', label='VNT-Based Prediction', alpha=0.7, linewidth=1.5)
        
        # VNT prediction stats
        vnt_rmse = np.sqrt(np.mean((vnt_pred - true_values) ** 2))
        vnt_r2 = 1 - np.sum((vnt_pred - true_values) ** 2) / np.sum((true_values - np.mean(true_values)) ** 2)
        
        # Add stats text box
        vnt_stats = (f"VNT-Based Prediction:\n"
                    f"RMSE: {vnt_rmse:.4f}\n"
                    f"R²: {vnt_r2:.4f}")
        
        ax2.text(0.05, 0.95, vnt_stats, transform=ax2.transAxes, 
                va='top', ha='left', bbox=dict(boxstyle='round,pad=0.5', 
                                             facecolor='white', alpha=0.8),
                fontsize=9)
        
        ax2.set_title("VNT-Based Prediction vs Ground Truth", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(frameon=True, fontsize=9)
        
        # Plot 3: Difference between methods
        method_diff = direct_pred - vnt_pred
        ax3.plot(times, method_diff, 'k-', label='Direct - VNT', alpha=0.7, linewidth=1.5)
        ax3.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.7)
        
        # Difference stats
        mean_diff = np.mean(method_diff)
        std_diff = np.std(method_diff)
        
        # Add stats text box
        diff_stats = (f"Method Difference:\n"
                     f"Mean: {mean_diff:.4f}\n"
                     f"Std: {std_diff:.4f}")
        
        ax3.text(0.05, 0.95, diff_stats, transform=ax3.transAxes, 
                va='top', ha='left', bbox=dict(boxstyle='round,pad=0.5', 
                                             facecolor='white', alpha=0.8),
                fontsize=9)
        
        ax3.set_title("Difference Between Prediction Methods", fontsize=12)
        ax3.set_xlabel("Time Index", fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.legend(frameon=True, fontsize=9)
        
        # Y-axis labels
        ax1.set_ylabel("Heat Transport", fontsize=10)
        ax2.set_ylabel("Heat Transport", fontsize=10)
        ax3.set_ylabel("Difference", fontsize=10)
        
        # Add overall title if provided
        if title:
            fig.suptitle(title, fontsize=14, y=0.98)
            
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.3)
        
        # Save or show plot
        if save_path:
            sp = self.output_dir / 'comparison' / f'{save_path}.png'
            plt.savefig(sp, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved method comparison plot to {sp}")
        else:
            plt.show()
            
    def plot_vnt_field(self, vnt_pred, vnt_true=None, lat=None, lon=None, depth_idx=0, 
                      save_path=None, title=None):
        """
        Plot predicted VNT field with optional comparison to ground truth
        
        Args:
            vnt_pred: Predicted VNT field [batch, depth, lat, lon]
            vnt_true: True VNT field (optional)
            lat: Latitude values
            lon: Longitude values
            depth_idx: Depth index to plot (default: 0 for surface)
            save_path: Path to save the plot
            title: Optional title for the plot
        """
        # Select the first sample if batch dimension is present
        if len(vnt_pred.shape) > 3:
            vnt_pred = vnt_pred[0]
            if vnt_true is not None and len(vnt_true.shape) > 3:
                vnt_true = vnt_true[0]
        
        # Select the specified depth level
        vnt_pred_level = vnt_pred[depth_idx]
        if vnt_true is not None:
            vnt_true_level = vnt_true[depth_idx]
        
        # Create latitude and longitude grids if not provided
        if lat is None or lon is None:
            lat = np.arange(vnt_pred_level.shape[0])
            lon = np.arange(vnt_pred_level.shape[1])
            lon_grid, lat_grid = np.meshgrid(lon, lat)
        else:
            lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # Determine plot layout based on whether ground truth is provided
        if vnt_true is not None:
            # Create figure with three subplots: predicted, true, and difference
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6),
                                               subplot_kw={'projection': self.projection})
            axes = [ax1, ax2, ax3]
            titles = ["Predicted VNT", "True VNT", "Difference"]
            data = [vnt_pred_level, vnt_true_level, vnt_pred_level - vnt_true_level]
        else:
            # Create figure with single subplot for prediction only
            fig, ax = plt.subplots(figsize=(10, 8),
                                 subplot_kw={'projection': self.projection})
            axes = [ax]
            titles = ["Predicted VNT"]
            data = [vnt_pred_level]
        
        # Create color scales for consistent visualization
        if vnt_true is not None:
            # Use same scale for predicted and true, different for difference
            vmin = min(vnt_pred_level.min(), vnt_true_level.min())
            vmax = max(vnt_pred_level.max(), vnt_true_level.max())
            
            # Symmetric scale for difference plot
            vmin_diff = -max(abs(vnt_pred_level - vnt_true_level).max(), 0.1)
            vmax_diff = abs(vmin_diff)
            
            # Use different colormaps for each plot
            cmaps = [self.heat_cmap, self.heat_cmap, self.diff_cmap]
            vmins = [vmin, vmin, vmin_diff]
            vmaxs = [vmax, vmax, vmax_diff]
        else:
            # Single plot with thermal colormap
            vmin = vnt_pred_level.min()
            vmax = vnt_pred_level.max()
            cmaps = [self.heat_cmap]
            vmins = [vmin]
            vmaxs = [vmax]
            
        # Plot each panel
        for i, (ax, d, t, cmap, vmin_i, vmax_i) in enumerate(zip(axes, data, titles, cmaps, vmins, vmaxs)):
            # Create pcolormesh plot for data
            im = ax.pcolormesh(
                lon_grid, lat_grid, d,
                transform=self.projection,
                cmap=cmap,
                vmin=vmin_i,
                vmax=vmax_i,
                shading='auto'
            )
            
            # Add geographic features
            ax.coastlines(resolution='50m', linewidth=0.8, color='black')
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.6)
            ax.add_feature(cfeature.LAND, alpha=0.3, color='tan')
            
            # Set map extent to North Atlantic region
            ax.set_extent(self.atlantic_bounds, crs=self.projection)
            
            # Add gridlines with labels
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False
            
            # Add title
            ax.set_title(t, fontsize=12)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, label='VNT')
            cbar.ax.tick_params(labelsize=9)
            
        # Add overall title if provided
        if title:
            fig.suptitle(title, fontsize=14, y=0.95)
            
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            sp = self.output_dir / 'vnt_plots' / f'{save_path}.png'
            plt.savefig(sp, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved VNT field plot to {sp}")
        else:
            plt.show()