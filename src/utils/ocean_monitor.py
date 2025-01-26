import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import holoviews as hv
from bokeh.layouts import layout
from bokeh.plotting import figure, save
from bokeh.io import output_file
import time
import logging
from typing import Dict, Any, Optional, Tuple
import json
import xarray as xr

class OceanDataMonitor:
    def __init__(self, output_dir: str = 'monitoring'):
        self.output_dir = Path(output_dir)
        self.metrics_dir = self.output_dir / 'metrics'
        self.plots_dir = self.output_dir / 'plots'
        self.dashboard_dir = self.output_dir / 'dashboards'
        
        # Create directories
        for dir_path in [self.metrics_dir, self.plots_dir, self.dashboard_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'spatial': [],
            'temporal': [],
            'performance': [],
            'validation': []
        }
        
        self.logger = logging.getLogger(__name__)
        hv.extension('bokeh')
    
    def analyze_spatial_coverage(self, data: xr.DataArray, var_name: str) -> Dict[str, Any]:
        valid_points = ~np.isnan(data.values)
        
        spatial_metrics = {
            'variable': var_name,
            'coverage_pct': float(np.sum(valid_points) / valid_points.size * 100),
            'lat_range': [float(data.nlat.min()), float(data.nlat.max())],
            'lon_range': [float(data.nlon.min()), float(data.nlon.max())]
        }
        
        if hasattr(data, 'TLAT') and hasattr(data, 'TLONG'):
            spatial_metrics.update({
                'tlat_range': [float(data.TLAT.min()), float(data.TLAT.max())],
                'tlong_range': [float(data.TLONG.min()), float(data.TLONG.max())]
            })
        
        self.metrics['spatial'].append(spatial_metrics)
        return spatial_metrics
    
    def analyze_temporal_patterns(self, data: xr.DataArray, var_name: str) -> Dict[str, Any]:
        """Analyze temporal patterns in the data"""
        self.logger.info(f"Analyzing temporal patterns for {var_name}")
        
        # Calculate temporal statistics
        temporal_metrics = {
            'variable': var_name,
            'n_timesteps': len(data.time),
            'temporal_coverage_pct': (np.sum(~np.isnan(data), axis=0) / len(data.time) * 100).mean(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.metrics['temporal'].append(temporal_metrics)
        return temporal_metrics
    
    def validate_data(self, data: xr.DataArray, var_name: str) -> Dict[str, Any]:
        """Validate data quality and consistency"""
        self.logger.info(f"Validating data for {var_name}")
        
        validation_metrics = {
            'variable': var_name,
            'min_value': float(data.min()),
            'max_value': float(data.max()),
            'mean_value': float(data.mean()),
            'std_value': float(data.std()),
            'missing_pct': (np.sum(np.isnan(data)) / data.size * 100),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.metrics['validation'].append(validation_metrics)
        return validation_metrics
    
    def plot_spatial_coverage(self, data: xr.DataArray, var_name: str):
        """Create spatial coverage plot"""
        self.logger.info(f"Plotting spatial coverage for {var_name}")
        
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Plot data availability
        valid_mask = ~np.isnan(data.isel(time=0))
        im = ax.pcolormesh(data.nlon, data.nlat, valid_mask, cmap='viridis')
        
        # Add coastlines if available
        try:
            import cartopy.feature as cfeature
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.gridlines(draw_labels=True)
        except ImportError:
            self.logger.warning("Cartopy not available, plotting without coastlines")
            ax.grid(True)
        
        plt.colorbar(im, label='Data Availability')
        plt.title(f'Spatial Coverage - {var_name}')
        
        save_path = self.plots_dir / f'spatial_coverage_{var_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Spatial coverage plot saved to {save_path}")
    
    def create_performance_dashboard(self):
        """Create interactive performance dashboard"""
        self.logger.info("Creating performance dashboard")
        
        # Convert metrics to DataFrames
        spatial_df = pd.DataFrame(self.metrics['spatial'])
        temporal_df = pd.DataFrame(self.metrics['temporal'])
        validation_df = pd.DataFrame(self.metrics['validation'])
        
        # Create performance plots
        spatial_plot = figure(title='Spatial Coverage by Variable',
                            x_range=spatial_df['variable'].unique())
        spatial_plot.vbar(x='variable', top='coverage_pct', source=spatial_df)
        
        temporal_plot = figure(title='Temporal Coverage by Variable',
                             x_range=temporal_df['variable'].unique())
        temporal_plot.vbar(x='variable', top='temporal_coverage_pct', source=temporal_df)
        
        # Create dashboard layout
        dashboard = layout([
            [spatial_plot],
            [temporal_plot]
        ])
        
        # Save dashboard
        output_file(self.dashboard_dir / 'performance_dashboard.html')
        save(dashboard)
        
        self.logger.info("Performance dashboard created")
    
    def save_metrics(self):
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        for metric_type, metrics in self.metrics.items():
            if metrics:
                df = pd.DataFrame(metrics)
                save_path = self.metrics_dir / f'{metric_type}_metrics_{timestamp}.csv'
                self.metrics_dir.mkdir(parents=True, exist_ok=True)
                df.to_csv(save_path, index=False)
                self.logger.info(f"Saved {metric_type} metrics to {save_path}")
        
        summary = {
            'timestamp': timestamp,
            'metrics_collected': list(self.metrics.keys()),
            'total_records': {k: len(v) for k, v in self.metrics.items()},
            'variables_monitored': list(set(
                m['variable'] for metrics in self.metrics.values()
                for m in metrics if 'variable' in m
            ))
        }
        
        summary_path = self.metrics_dir / f'summary_{timestamp}.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info(f"Metrics summary saved to {summary_path}")
    
    def create_monitoring_report(self):
        """Create comprehensive monitoring report"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f'monitoring_report_{timestamp}.txt'
        
        with open(report_path, 'w') as f:
            f.write("Ocean Data Monitoring Report\n")
            f.write("==========================\n\n")
            
            for metric_type, metrics in self.metrics.items():
                if metrics:
                    f.write(f"\n{metric_type.upper()} METRICS\n")
                    df = pd.DataFrame(metrics)
                    f.write(df.describe().to_string())
                    f.write("\n\n")
        
        self.logger.info(f"Monitoring report saved to {report_path}")

ocean_monitor = OceanDataMonitor()