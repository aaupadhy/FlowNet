import xarray as xr
import numpy as np
import dask
import dask.array as da
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import logging
from pathlib import Path
import gc
from tqdm import tqdm
from ..utils.dask_utils import dask_monitor

class OceanDataProcessor:
    def __init__(self, ssh_path: str, sst_path: str, vnt_path: str):
        self.logger = logging.getLogger(__name__)
        
        self.cache_dir = Path('/scratch/user/aaupadhy/college/RA/final_data/cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created/verified cache directory at {self.cache_dir}")
        
        with dask_monitor.profile_task("data_loading"):
            self.logger.info("Loading datasets...")
            self._load_datasets(ssh_path, sst_path, vnt_path)

    def _cleanup_memory(self):
        gc.collect()
        if hasattr(self, 'client'): 
            self.client.cancel(self.client.get_current_task())
            
        torch.cuda.empty_cache()
        self.logger.info(f"Memory cleaned up. Current GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    def log_chunk_stats(self, data_array, array_name: str = "data_array"):
        if not hasattr(data_array, 'chunks') or data_array.chunks is None:
            self.logger.info(f"{array_name} is not dask-backed or has no chunks. Shape: {data_array.shape}")
            return {}

        chunk_tuples = data_array.chunks
        dtype_size = data_array.dtype.itemsize

        total_elems = np.prod(data_array.shape)
        total_mem_gb = (total_elems * dtype_size) / (1024**3)
        
        chunks_per_dim = [len(dim_c) for dim_c in chunk_tuples]
        n_chunks = np.prod(chunks_per_dim)
        
        avg_shape_per_dim = [np.mean(dim_chunk_sizes) for dim_chunk_sizes in chunk_tuples]
        avg_chunk_elems = int(np.prod(avg_shape_per_dim))
        avg_chunk_mem_gb = (avg_chunk_elems * dtype_size) / (1024**3)
        
        stats = {
            "shape": data_array.shape,
            "dtype": str(data_array.dtype),
            "total_size_gb": total_mem_gb,
            "n_chunks": n_chunks,
            "chunks_per_dim": chunks_per_dim,
            "avg_chunk_size_gb": avg_chunk_mem_gb
        }
        
        self.logger.info(f"\nChunk stats for {array_name}:")
        for key, value in stats.items():
            self.logger.info(f"  {key}: {value}")
        
        return stats
    
    
    def _load_datasets(self, ssh_path, sst_path, vnt_path):
        """
        Load and initialize all required datasets with proper chunking and statistics computation.
        """
        try:
            # Define base chunking strategy
            base_chunks = {'time': 180, 'nlat': 'auto', 'nlon': 'auto'}
            
            # Load SSH dataset
            self.logger.info("Loading SSH dataset...")
            self.ssh_ds = xr.open_dataset(ssh_path, engine="zarr", chunks=None)
            self.ssh_ds = self.ssh_ds.chunk(base_chunks)
            
            # Load SST dataset
            self.logger.info("Loading SST dataset...")
            self.sst_ds = xr.open_dataset(sst_path, engine="zarr", chunks=None)
            self.sst_ds = self.sst_ds.chunk(base_chunks)
            
            # Load VNT dataset with different chunking
            self.logger.info("Loading VNT dataset...")
            vnt_chunks = {'time': 12, 'nlat': 300, 'nlon': 300, 'z_t': None}
            self.vnt_ds = xr.open_dataset(vnt_path, engine="zarr", chunks=None)
            self.vnt_ds = self.vnt_ds.chunk(vnt_chunks)
            
            # Find reference latitude for heat transport calculations
            self.reference_latitude = self._find_reference_latitude()
            
            # Compute statistics for normalization
            with dask_monitor.profile_task("compute_statistics"):
                # Analyze SSH sample
                ssh_sample = self.ssh_ds["SSH"][0].compute()  # First timestep only for sample
                self.logger.info(f"SSH sample stats:")
                self.logger.info(f"Shape: {ssh_sample.shape}")
                self.logger.info(f"Any NaNs: {np.any(np.isnan(ssh_sample))}")
                self.logger.info(f"Min/Max: {float(ssh_sample.min())}/{float(ssh_sample.max())}")
                
                # Compute SSH statistics
                self.logger.info("Computing SSH statistics...")
                self.ssh_mean, self.ssh_std = self._compute_mean_std(
                    self.ssh_ds["SSH"],
                    var_name="SSH"
                )
                self.logger.info(f"Final SSH stats:")
                self.logger.info(f"Mean: {self.ssh_mean:.6f}")
                self.logger.info(f"Std: {self.ssh_std:.6f}")
                
                # Compute SST statistics
                self.logger.info("Computing SST statistics...")
                self.sst_mean, self.sst_std = self._compute_mean_std(
                    self.sst_ds["SST"],
                    var_name="SST"
                )
                self.logger.info(f"Final SST stats:")
                self.logger.info(f"Mean: {self.sst_mean:.6f}")
                self.logger.info(f"Std: {self.sst_std:.6f}")
            
            # Store dataset properties
            self.tarea_conversion = 0.0001  # Area conversion factor
            self.dz_conversion = 0.01       # Depth conversion factor
            self.shape = self.sst_ds["SST"].shape
            
            # Store essential statistics
            self.stats = {
                'grid_shape': self.shape,
                'ssh_mean': self.ssh_mean,
                'ssh_std': self.ssh_std,
                'sst_mean': self.sst_mean,
                'sst_std': self.sst_std,
                'spatial_dims': {
                    'nlat': self.shape[1],
                    'nlon': self.shape[2]
                },
                'time_steps': self.shape[0]
            }
            
            # Validate statistics
            self._validate_statistics()
            
            self.logger.info("Successfully loaded all datasets")
            
        except Exception as e:
            self.logger.error(f"Error loading datasets: {str(e)}")
            self._cleanup_memory()
            raise

    def _validate_data_consistency(self):
        """Validate consistency between datasets."""
        # Check temporal alignment
        if not (len(self.ssh_ds.time) == len(self.sst_ds.time) == len(self.vnt_ds.time)):
            raise ValueError("Temporal dimensions do not match across datasets")
        
        # Check spatial alignment
        ssh_shape = self.ssh_ds.SSH.shape
        sst_shape = self.sst_ds.SST.shape
        
        if ssh_shape != sst_shape:
            raise ValueError(f"Spatial dimensions mismatch: SSH {ssh_shape} vs SST {sst_shape}")
        
        # Validate coordinates
        if not np.allclose(self.ssh_ds.nlat, self.sst_ds.nlat) or \
        not np.allclose(self.ssh_ds.nlon, self.sst_ds.nlon):
            raise ValueError("Coordinate systems do not match between datasets")

    def _validate_statistics(self):
        """Validate computed statistics."""
        stat_checks = {
            'ssh_mean': (-500, 500),    # Reasonable SSH range in cm
            'sst_mean': (-5, 35),       # Reasonable SST range in Celsius
            'ssh_std': (0, 100),        # Reasonable SSH variability
            'sst_std': (0, 20)          # Reasonable SST variability
        }
        
        for stat_name, (min_val, max_val) in stat_checks.items():
            value = self.stats[stat_name]
            if not isinstance(value, (int, float)):
                raise ValueError(f"{stat_name} is not a scalar value")
            if not np.isfinite(value):
                raise ValueError(f"Invalid {stat_name}: {value}")
            if value == 0:
                raise ValueError(f"Zero {stat_name} detected")
            if not min_val <= value <= max_val:
                raise ValueError(
                    f"{stat_name} ({value:.2f}) outside reasonable range "
                    f"[{min_val}, {max_val}]"
                )
        
        self.logger.info("Statistics validation passed:")
        for key, value in self.stats.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"{key}: {value:.4f}")
            else:
                self.logger.info(f"{key}: {value}")
                
    def _find_reference_latitude(self) -> int:
        """
        Find the reference latitude index closest to 40°N for heat transport calculations.
        
        Returns:
            int: Index of the reference latitude
            
        Raises:
            ValueError: If no suitable latitude points are found or if data is invalid
        """
        cache_file = Path('/scratch/user/aaupadhy/college/RA/final_data/cache/ref_lat_index.npy')
        
        # Try to load from cache first
        if cache_file.exists():
            try:
                ref_lat_idx = int(np.load(cache_file))
                tlat = self.vnt_ds['TLAT']
                cached_lats = tlat[ref_lat_idx].compute()
                valid_points = ((cached_lats >= 39.9) & (cached_lats <= 40.1)).sum()
                
                if valid_points > 0:
                    self.logger.info(f"Loaded cached reference latitude index {ref_lat_idx} "
                                f"with {valid_points} points near 40°N")
                    return ref_lat_idx
                else:
                    self.logger.warning("Cached index has no valid points near 40°N. Recomputing...")
            except Exception as e:
                self.logger.warning(f"Failed to load cached reference latitude: {str(e)}. Recomputing...")
        
        try:
            # Get raw TLAT values
            tlat = self.vnt_ds['TLAT']
            
            # Log the full latitude range
            lat_min = float(tlat.min().compute())
            lat_max = float(tlat.max().compute())
            self.logger.info(f"Full latitude range: {lat_min:.2f}°N to {lat_max:.2f}°N")
            
            if not (lat_min <= 40 <= lat_max):
                raise ValueError(f"40°N outside available range [{lat_min:.2f}°N, {lat_max:.2f}°N]")
            
            # Convert to numpy array for processing
            lat_array = tlat.compute()
            
            # Find all points close to 40°N
            target_mask = (lat_array >= 39.9) & (lat_array <= 40.1)
            valid_rows, valid_cols = np.where(target_mask)
            
            if len(valid_rows) == 0:
                # If no exact matches, try finding closest points
                distances = np.abs(lat_array - 40.0)
                min_distance = float(distances.min())
                self.logger.info(f"No points in target range. Closest point is {min_distance:.4f}° away from 40°N")
                raise ValueError("No points found between 39.9°N and 40.1°N")
                
            # Group points by row and analyze each row
            row_stats = {}
            for row, col in zip(valid_rows, valid_cols):
                if row not in row_stats:
                    row_stats[row] = {
                        'points': [],
                        'lats': []
                    }
                row_stats[row]['points'].append(col)
                row_stats[row]['lats'].append(lat_array[row, col])
            
            # Calculate statistics for each row
            analyzed_rows = []
            for row, stats in row_stats.items():
                lats = np.array(stats['lats'])
                analyzed_rows.append({
                    'row': row,
                    'num_points': len(stats['points']),
                    'mean_lat': float(np.mean(lats)),
                    'std_lat': float(np.std(lats)),
                    'min_lat': float(np.min(lats)),
                    'max_lat': float(np.max(lats)),
                    'columns': stats['points']
                })
            
            # Log detailed information about found points
            self.logger.info(f"\nFound points near 40°N in {len(analyzed_rows)} rows:")
            for row_data in analyzed_rows:
                self.logger.info(
                    f"Row {row_data['row']}: {row_data['num_points']} points, "
                    f"mean={row_data['mean_lat']:.4f}°N ± {row_data['std_lat']:.4f}°, "
                    f"range=[{row_data['min_lat']:.4f}°N, {row_data['max_lat']:.4f}°N]"
                )
            
            # Select best row based on number of points and mean latitude
            best_row = max(analyzed_rows, key=lambda x: x['num_points'])
            ref_lat_idx = best_row['row']
            
            # Verify selected row is suitable
            if best_row['num_points'] < 10:
                raise ValueError(f"Best row only has {best_row['num_points']} points near 40°N")
            
            if abs(best_row['mean_lat'] - 40.0) > 0.2:
                raise ValueError(f"Best row mean latitude {best_row['mean_lat']:.4f}°N too far from 40°N")
            
            # Log final selection details
            self.logger.info(f"\nSelected row {ref_lat_idx}:")
            self.logger.info(f"  Number of points: {best_row['num_points']}")
            self.logger.info(f"  Mean latitude: {best_row['mean_lat']:.4f}°N")
            self.logger.info(f"  Latitude std dev: {best_row['std_lat']:.4f}°")
            self.logger.info(f"  Latitude range: [{best_row['min_lat']:.4f}°N, {best_row['max_lat']:.4f}°N]")
            self.logger.info(f"  Columns: {best_row['columns'][:5]}... (showing first 5)")
            
            # Cache the validated result
            np.save(cache_file, ref_lat_idx)
            
            return ref_lat_idx
            
        except Exception as e:
            self.logger.error(f"Error finding reference latitude: {str(e)}")
            raise ValueError(f"Failed to find reference latitude: {str(e)}")
            
        finally:
            self._cleanup_memory()
        
    def _compute_mean_std(self, data_array, var_name: str, dim="time"):
        """
        Compute global mean and standard deviation for data normalization.
        
        Args:
            data_array: xarray.DataArray to compute statistics for
            var_name: Name of the variable (for logging)
            dim: Dimension to compute statistics over
            
        Returns:
            tuple: (mean, std) as scalar float values
        """
        cache_file = self.cache_dir / f'{var_name}_stats.npy'
        if cache_file.exists():
            try:
                stats = np.load(cache_file)
                mean = float(stats[0])
                std = float(stats[1])
                self.logger.info(f"Loaded cached statistics for {var_name}")
                self.logger.info(f"Mean: {mean:.6f}, Std: {std:.6f}")
                return mean, std
            except Exception as e:
                self.logger.warning(f"Failed to load cached stats: {str(e)}. Recomputing...")

        self.logger.info(f"Computing statistics for {var_name}")
        self.logger.info(f"Input shape: {data_array.shape}")

        try:
            # Define missing value constant
            MISSING_VALUE = 9.96921e+36
            
            # Create mask for valid values
            data_masked = data_array.where(
                (data_array != MISSING_VALUE) & 
                (~np.isnan(data_array))
            )
            
            # Compute global statistics
            with dask_monitor.profile_task(f"compute_{var_name}_stats"):
                mean = float(data_masked.mean().compute())
                std = float(data_masked.std().compute())

                # Validate statistics
                if not np.isfinite(mean) or not np.isfinite(std):
                    raise ValueError(f"Invalid statistics computed: mean={mean}, std={std}")
                
                if std < 1e-6:
                    self.logger.warning(f"Very small std dev ({std}) for {var_name}, using 1.0")
                    std = 1.0

            # Log statistics
            self.logger.info(f"Computed global statistics for {var_name}:")
            self.logger.info(f"Mean: {mean:.6f}")
            self.logger.info(f"Std: {std:.6f}")

            # Cache the results
            try:
                np.save(cache_file, np.array([mean, std]))
                self.logger.info(f"Cached statistics to {cache_file}")
            except Exception as e:
                self.logger.warning(f"Failed to cache statistics: {str(e)}")

            return mean, std

        except Exception as e:
            self.logger.error(f"Error computing statistics for {var_name}: {str(e)}")
            raise

    def get_spatial_data(self):
        return (
            self.ssh_ds["SSH"],
            self.sst_ds["SST"],
            self.vnt_ds["TLAT"],
            self.vnt_ds["TLONG"]
        )

    def get_dataset_info(self):
        return {
            'ssh_shape': self.ssh_ds["SSH"].shape,
            'sst_shape': self.sst_ds["SST"].shape,
            'vnt_shape': self.vnt_ds["VNT"].shape,
            'time_steps': len(self.ssh_ds["time"]),
            'ssh_chunks': self.ssh_ds["SSH"].chunks,
            'sst_chunks': self.sst_ds["SST"].chunks,
            'vnt_chunks': self.vnt_ds["VNT"].chunks
        }

    def calculate_heat_transport(self, latitude_index=None):
        if latitude_index is None:
            latitude_index = self.reference_latitude
            
        cache_file = Path(f'/scratch/user/aaupadhy/college/RA/final_data/cache/heat_transport_lat{latitude_index}.npy')
        cache_mean_file = Path(f'/scratch/user/aaupadhy/college/RA/final_data/cache/heat_transport_lat{latitude_index}_mean.npy')
        
        if cache_file.exists() and cache_mean_file.exists():
            try:
                heat_transport = np.load(cache_file)
                mean_transport = np.load(cache_mean_file)
                self.logger.info(f"Loaded cached heat transport data:")
                self.logger.info(f"  Shape: {heat_transport.shape}")
                self.logger.info(f"  Range: {heat_transport.min():.2e} to {heat_transport.max():.2e}")
                self.logger.info(f"  Mean: {mean_transport:.2e}")
                return heat_transport, mean_transport
            except Exception as e:
                self.logger.warning(f"Failed to load cached data: {str(e)}. Recomputing...")
        
        try:
            self.logger.info(f"Calculating heat transport for latitude index {latitude_index}")
            
            vnt = self.vnt_ds["VNT"].isel(nlat=latitude_index)
            self.logger.info(f"VNT shape after lat selection: {vnt.shape}")
            
            tarea = self.vnt_ds["TAREA"].isel(nlat=latitude_index)
            self.logger.info(f"TAREA shape: {tarea.shape}")
            
            dz = self.vnt_ds["dz"]
            self.logger.info(f"dz shape: {dz.shape}")

            chunk_size = 60
            total_times = vnt.shape[0]
            heat_transport_list = []
            
            for start_idx in range(0, total_times, chunk_size):
                end_idx = min(start_idx + chunk_size, total_times)
                vnt_chunk = vnt.isel(time=slice(start_idx, end_idx))
                
                chunk_transport = (
                    vnt_chunk *
                    tarea * self.tarea_conversion *
                    dz * self.dz_conversion
                ).sum(dim=['z_t', 'nlon']).compute()
                
                heat_transport_list.append(chunk_transport)
            
            heat_transport = np.concatenate(heat_transport_list)
            mean_transport = float(np.mean(heat_transport))
            
            np.save(cache_file, heat_transport)
            np.save(cache_mean_file, mean_transport)
            
            self.logger.info(f"Heat transport statistics:")
            self.logger.info(f"  Range: {float(heat_transport.min()):.2e} to {float(heat_transport.max()):.2e}")
            self.logger.info(f"  Mean: {mean_transport:.2e}")
            
            return heat_transport, mean_transport
            
        except Exception as e:
            self.logger.error(f"Error calculating heat transport: {str(e)}")
            raise
        finally:
            self._cleanup_memory()

class OceanDataset(Dataset):
    """Dataset class for ocean data with standardization."""
    
    def __init__(self, ssh, sst, heat_transport, heat_transport_mean,
                 ssh_mean, ssh_std, sst_mean, sst_std, shape):
        """
        Initialize dataset with raw data and statistics for normalization.
        
        Args:
            ssh: SSH data array
            sst: SST data array
            heat_transport: Heat transport data array
            heat_transport_mean: Mean value for heat transport
            ssh_mean: Global mean for SSH (scalar)
            ssh_std: Global std for SSH (scalar)
            sst_mean: Global mean for SST (scalar)
            sst_std: Global std for SST (scalar)
            shape: Shape of the data arrays
        """
        self.ssh = ssh
        self.sst = sst
        self.logger = logging.getLogger(__name__)
        
        # Standardize heat transport
        heat_transport_std = np.std(heat_transport)
        self.heat_transport = (heat_transport - heat_transport_mean) / heat_transport_std
        
        self.logger.info(f"Heat Transport statistics after standardization:")
        self.logger.info(f"  Min: {np.min(self.heat_transport):.2f}")
        self.logger.info(f"  Max: {np.max(self.heat_transport):.2f}")
        self.logger.info(f"  Mean: {np.mean(self.heat_transport):.2f}")
        self.logger.info(f"  Std: {np.std(self.heat_transport):.2f}")
        
        self.ssh_mean = float(ssh_mean) 
        self.ssh_std = float(ssh_std)   
        self.sst_mean = float(sst_mean) 
        self.sst_std = float(sst_std)    
        
        self.logger.info(f"Normalization parameters:")
        self.logger.info(f"  SSH mean: {self.ssh_mean:.4f}")
        self.logger.info(f"  SSH std:  {self.ssh_std:.4f}")
        self.logger.info(f"  SST mean: {self.sst_mean:.4f}")
        self.logger.info(f"  SST std:  {self.sst_std:.4f}")
        
        self.MISSING_VALUE = 9.96921e+36
        self.ssh_bounds, self.sst_bounds = self.analyze_distributions(ssh, sst)
        
        ssh_vals = ssh.isel(time=0).values
        sst_vals = sst.isel(time=0).values
        ssh_valid = ssh_vals != self.MISSING_VALUE
        sst_valid = sst_vals != self.MISSING_VALUE
        self.valid_mask = ssh_valid & sst_valid
        
        self.length = shape[0]

    def analyze_distributions(self, ssh, sst):
        """
        Analyze SSH and SST distributions to determine appropriate clipping bounds.
        
        Args:
            ssh: xarray DataArray of SSH data
            sst: xarray DataArray of SST data
        
        Returns:
            tuple: (ssh_bounds, sst_bounds) in normalized units
        """
        try:
            # Sample a subset of timesteps for efficiency (e.g., every 10th timestep)
            sample_freq = 10
            timesteps = range(0, ssh.shape[0], sample_freq)
            
            # Initialize lists to store valid values
            ssh_valid_values = []
            sst_valid_values = []
            
            # Collect valid values across timesteps
            for t in timesteps:
                ssh_slice, sst_slice = dask.compute(
                    ssh.isel(time=t),
                    sst.isel(time=t)
                )
                
                # Get valid SSH values
                ssh_data = ssh_slice.values
                ssh_valid = ssh_data[
                    (ssh_data != self.MISSING_VALUE) & 
                    np.isfinite(ssh_data)
                ]
                ssh_valid_values.extend(ssh_valid)
                
                # Get valid SST values
                sst_data = sst_slice.values
                sst_valid = sst_data[
                    (sst_data != self.MISSING_VALUE) & 
                    np.isfinite(sst_data)
                ]
                sst_valid_values.extend(sst_valid)
            
            # Convert to numpy arrays
            ssh_valid = np.array(ssh_valid_values)
            sst_valid = np.array(sst_valid_values)
            
            # Compute distribution statistics
            percentiles = [0.1, 1, 5, 95, 99, 99.9]
            ssh_percentiles = np.percentile(ssh_valid, percentiles)
            sst_percentiles = np.percentile(sst_valid, percentiles)
            
            # Log raw data statistics
            self.logger.info("\nDistribution Analysis:")
            self.logger.info("\nSSH Statistics (cm):")
            self.logger.info(f"  Mean: {np.mean(ssh_valid):.2f}")
            self.logger.info(f"  Std:  {np.std(ssh_valid):.2f}")
            self.logger.info(f"  Min:  {np.min(ssh_valid):.2f}")
            self.logger.info(f"  Max:  {np.max(ssh_valid):.2f}")
            self.logger.info("  Percentiles:")
            for p, v in zip(percentiles, ssh_percentiles):
                self.logger.info(f"    {p:>5.1f}%: {v:.2f}")
            
            self.logger.info("\nSST Statistics (°C):")
            self.logger.info(f"  Mean: {np.mean(sst_valid):.2f}")
            self.logger.info(f"  Std:  {np.std(sst_valid):.2f}")
            self.logger.info(f"  Min:  {np.min(sst_valid):.2f}")
            self.logger.info(f"  Max:  {np.max(sst_valid):.2f}")
            self.logger.info("  Percentiles:")
            for p, v in zip(percentiles, sst_percentiles):
                self.logger.info(f"    {p:>5.1f}%: {v:.2f}")
            
            # Define bounds based on physics and distribution
            # SSH: Allow range between 1st and 99th percentile
            ssh_norm_min = (ssh_percentiles[1] - self.ssh_mean) / (self.ssh_std + 1e-6)
            ssh_norm_max = (ssh_percentiles[-2] - self.ssh_mean) / (self.ssh_std + 1e-6)
            
            # SST: Physical constraints
            # Minimum: Cannot be below freezing (-2°C for seawater)
            # Maximum: Rarely exceeds 32°C in Atlantic
            sst_norm_min = max((-2 - self.sst_mean) / (self.sst_std + 1e-6),
                            (sst_percentiles[1] - self.sst_mean) / (self.sst_std + 1e-6))
            sst_norm_max = min((32 - self.sst_mean) / (self.sst_std + 1e-6),
                            (sst_percentiles[-2] - self.sst_mean) / (self.sst_std + 1e-6))
            
            # Log normalized bounds
            self.logger.info("\nSelected Clipping Bounds (normalized units):")
            self.logger.info(f"  SSH: [{ssh_norm_min:.2f}, {ssh_norm_max:.2f}]")
            self.logger.info(f"  SST: [{sst_norm_min:.2f}, {sst_norm_max:.2f}]")
            
            # Convert back to physical units to verify
            ssh_phys_min = ssh_norm_min * self.ssh_std + self.ssh_mean
            ssh_phys_max = ssh_norm_max * self.ssh_std + self.ssh_mean
            sst_phys_min = sst_norm_min * self.sst_std + self.sst_mean
            sst_phys_max = sst_norm_max * self.sst_std + self.sst_mean
            
            self.logger.info("\nSelected Clipping Bounds (physical units):")
            self.logger.info(f"  SSH: [{ssh_phys_min:.2f} cm, {ssh_phys_max:.2f} cm]")
            self.logger.info(f"  SST: [{sst_phys_min:.2f}°C, {sst_phys_max:.2f}°C]")
            
            return (ssh_norm_min, ssh_norm_max), (sst_norm_min, sst_norm_max)
            
        except Exception as e:
            self.logger.error(f"Error in analyze_distributions: {str(e)}")
            raise
   
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        try:
            ssh_data, sst_data = dask.compute(
                self.ssh.isel(time=idx),
                self.sst.isel(time=idx)
            )
            
            ssh_vals = ssh_data.values
            sst_vals = sst_data.values
            
            ssh_valid = (ssh_vals != self.MISSING_VALUE) & np.isfinite(ssh_vals)
            sst_valid = (sst_vals != self.MISSING_VALUE) & np.isfinite(sst_vals)
            
            ssh_norm = np.zeros_like(ssh_vals, dtype=np.float32)
            sst_norm = np.zeros_like(sst_vals, dtype=np.float32)
            
            ssh_norm = np.where(ssh_valid,
                              (ssh_vals - self.ssh_mean) / (self.ssh_std + 1e-6),
                              0)
            sst_norm = np.where(sst_valid,
                              (sst_vals - self.sst_mean) / (self.sst_std + 1e-6),
                              0)
            
            ssh_norm = np.clip(ssh_norm, self.ssh_bounds[0], self.ssh_bounds[1])
            ssh_norm = np.clip(ssh_norm, self.ssh_bounds[0], self.ssh_bounds[1])
            
            ssh_tensor = torch.from_numpy(ssh_norm).float().unsqueeze(0)
            sst_tensor = torch.from_numpy(sst_norm).float().unsqueeze(0)
            
            mask_tensor = torch.from_numpy(ssh_valid & sst_valid).float()
            
            ssh_downsampled = F.avg_pool2d(ssh_tensor.unsqueeze(0), kernel_size=2, stride=2).squeeze(0)
            sst_downsampled = F.avg_pool2d(sst_tensor.unsqueeze(0), kernel_size=2, stride=2).squeeze(0)
            mask_downsampled = F.avg_pool2d(mask_tensor.unsqueeze(0).unsqueeze(0), 
                                          kernel_size=2, stride=2).squeeze(0).squeeze(0)
            
            if idx % 500 == 0:
                self.logger.info(f"Batch {idx} statistics:")
                self.logger.info(f"  SSH range: [{float(ssh_downsampled.min()):.2f}, {float(ssh_downsampled.max()):.2f}]")
                self.logger.info(f"  SST range: [{float(sst_downsampled.min()):.2f}, {float(sst_downsampled.max()):.2f}]")
                self.logger.info(f"  Valid pixels: {float(mask_downsampled.mean()):.1%}")
            
            return (
                ssh_downsampled,
                sst_downsampled,
                mask_downsampled,
                torch.tensor(self.heat_transport[idx]).float()
            )
            
        except Exception as e:
            self.logger.error(f"Error in __getitem__ at index {idx}: {str(e)}")
            raise