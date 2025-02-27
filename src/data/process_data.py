import xarray as xr
import numpy as np
import dask
import dask.array as da
import torch
import torch.nn.functional as F
import logging
from pathlib import Path
import gc
from tqdm import tqdm
import time
from ..utils.dask_utils import dask_monitor
import pandas as pd

logger = logging.getLogger(__name__)

class OceanDataProcessor:
    """
    Processes oceanographic data for model training and evaluation.
    Efficiently handles large SSH, SST, and VNT datasets.
    """
    def __init__(self, ssh_path: str, sst_path: str, vnt_path: str, preload_data: bool = False, cache_data: bool = True):
        logger.info("Initializing OceanDataProcessor")
        self.cache_dir = Path('/scratch/user/aaupadhy/college/RA/final_data/cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up chunking strategy for efficient processing
        time_chunk = 180  # Time chunks for balanced memory usage
        base_chunks = {'time': time_chunk, 'nlat': 'auto', 'nlon': 'auto'}
        
        with dask_monitor.profile_task("load_datasets"):
            logger.info("Opening SSH dataset")
            self.ssh_ds = xr.open_zarr(ssh_path, chunks=base_chunks)
            
            logger.info("Opening SST dataset")
            self.sst_ds = xr.open_zarr(sst_path, chunks=base_chunks)
            
            logger.info("Opening VNT dataset")
            # Different chunking strategy for VNT - optimize for specific latitude access
            vnt_chunks = {'time': time_chunk, 'nlat': 1, 'nlon': 'auto', 'z_t': -1}
            self.vnt_ds = xr.open_zarr(vnt_path, chunks=vnt_chunks)
        
        # Load and cache necessary data
        if preload_data:
            self._preload_data(cache_data)
        
        # Compute key statistics
        self._compute_and_cache_stats()
        
        # Set conversion constants
        self.tarea_conversion = 0.0001
        self.dz_conversion = 0.01
        
        # Validate data consistency
        self._validate_data_consistency()
        logger.info("Successfully loaded all datasets")
    
    def _preload_data(self, cache_data):
        """Efficiently preload SSH and SST data"""
        t_preload = time.time()
        logger.info("Preloading SSH data")
        
        # Load SSH first - more memory efficient than loading both simultaneously
        self.ssh_ds = self.ssh_ds.load()
        logger.info("SSH data loaded")
        
        logger.info("Preloading SST data")
        self.sst_ds = self.sst_ds.load()
        logger.info("SST data loaded")
        
        logger.info("Preloading completed in %.2f seconds", time.time() - t_preload)
        
        # Log data statistics
        logger.info("SSH data: min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
                    float(self.ssh_ds["SSH"].min()), float(self.ssh_ds["SSH"].max()),
                    float(self.ssh_ds["SSH"].mean()), float(self.ssh_ds["SSH"].std()))
        logger.info("SST data: min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
                    float(self.sst_ds["SST"].min()), float(self.sst_ds["SST"].max()),
                    float(self.sst_ds["SST"].mean()), float(self.sst_ds["SST"].std()))
        
        # Cache arrays if needed - provides faster access but uses more memory
        if cache_data:
            self.ssh_array = self.ssh_ds["SSH"].values
            self.sst_array = self.sst_ds["SST"].values
            logger.info("Cached SSH and SST arrays in memory.")
            
            # Force garbage collection to clean up temporary arrays
            gc.collect()
    
    def _compute_and_cache_stats(self):
        """Compute and cache essential statistics for training"""
        t0 = time.time()
        
        with dask_monitor.profile_task("compute_statistics"):
            logger.info("Computing SSH mean and std")
            self.ssh_mean, self.ssh_std = self._compute_mean_std(self.ssh_ds["SSH"], "SSH")
            
            logger.info("Computing SST mean and std")
            self.sst_mean, self.sst_std = self._compute_mean_std(self.sst_ds["SST"], "SST")
            
            # Store grid shape
            self.shape = self.ssh_ds["SSH"].shape
            
            logger.info("Computing reference latitude")
            self.reference_latitude = self._find_reference_latitude()
        
        t_elapsed = time.time() - t0
        logger.info("Statistics computed in %.2f seconds", t_elapsed)
        
        # Store statistics for later use
        self.stats = {
            'grid_shape': self.shape,
            'ssh_mean': self.ssh_mean,
            'ssh_std': self.ssh_std,
            'sst_mean': self.sst_mean,
            'sst_std': self.sst_std,
            'spatial_dims': {'nlat': self.shape[1], 'nlon': self.shape[2]},
            'time_steps': self.shape[0],
            'reference_latitude': self.reference_latitude
        }
        
        # Validate statistics
        self._validate_statistics()
    
    def _cleanup_memory(self):
        """Force memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _validate_data_consistency(self):
        """Ensure data is consistent across datasets"""
        if not (len(self.ssh_ds.time) == len(self.sst_ds.time) == len(self.vnt_ds.time)):
            raise ValueError("Temporal dimensions do not match across datasets")
        
        ssh_shape = self.ssh_ds.SSH.shape
        sst_shape = self.sst_ds.SST.shape
        if ssh_shape != sst_shape:
            raise ValueError(f"Spatial dimensions mismatch: SSH {ssh_shape} vs SST {sst_shape}")
        
        if not np.allclose(self.ssh_ds.nlat, self.sst_ds.nlat):
            raise ValueError("Latitude coordinates do not match between SSH and SST datasets")
            
        if not np.allclose(self.ssh_ds.nlon, self.sst_ds.nlon):
            raise ValueError("Longitude coordinates do not match between SSH and SST datasets")
    
    def validate_coordinates(self):
        """Validate coordinate systems match across datasets"""
        # For custom coordinate validation beyond basic shape checks
        for coord in ['nlat', 'nlon', 'time']:
            ssh_coord = self.ssh_ds.coords.get(coord)
            sst_coord = self.sst_ds.coords.get(coord)
            vnt_coord = self.vnt_ds.coords.get(coord)
            
            if ssh_coord is None or sst_coord is None or vnt_coord is None:
                raise ValueError(f"Missing coordinate: {coord}")
                
        logger.info("All spatial and temporal coordinates are valid.")
    
    def _validate_statistics(self):
        """Validate computed statistics are reasonable"""
        stat_checks = {
            'ssh_mean': (-500, 500),    # Reasonable SSH mean range in cm
            'sst_mean': (-5, 35),       # Reasonable SST mean range in °C
            'ssh_std': (0, 100),        # Reasonable SSH std range
            'sst_std': (0, 20)          # Reasonable SST std range
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
                raise ValueError(f"{stat_name} ({value:.2f}) outside reasonable range [{min_val}, {max_val}]")
    
    def _find_reference_latitude(self) -> int:
        """Find reference latitude around 40°N for heat transport calculation"""
        cache_file = self.cache_dir / 'ref_lat_index.npy'
        
        # Try to load from cache first
        if cache_file.exists():
            try:
                ref_lat_idx = int(np.load(cache_file))
                tlat = self.vnt_ds['TLAT']
                cached_lats = tlat[ref_lat_idx].compute()
                valid_points = ((cached_lats >= 39.9) & (cached_lats <= 40.1)).sum()
                
                if valid_points > 0:
                    logger.info("Using cached reference latitude: %d", ref_lat_idx)
                    return ref_lat_idx
            except Exception as e:
                logger.warning("Failed to load cached reference latitude: %s", str(e))
        
        # Compute reference latitude if not in cache
        try:
            # Find all latitudes between 39.9°N and 40.1°N
            tlat = self.vnt_ds['TLAT'].compute()
            target_mask = (tlat >= 39.9) & (tlat <= 40.1)
            valid_rows = np.where(target_mask.any(axis=1))[0]
            
            if len(valid_rows) == 0:
                raise ValueError("No points found between 39.9°N and 40.1°N")
            
            # Use middle index for stability
            ref_lat_idx = valid_rows[len(valid_rows)//2]
            
            # Cache for future use
            np.save(cache_file, ref_lat_idx)
            logger.info("Computed and cached reference latitude: %d", ref_lat_idx)
            return ref_lat_idx
        except Exception as e:
            logger.error("Error finding reference latitude: %s", str(e))
            raise
        finally:
            self._cleanup_memory()
    
    def _compute_mean_std(self, data_array, var_name: str):
        """Compute mean and standard deviation with caching"""
        cache_file = self.cache_dir / f'{var_name}_stats.npy'
        
        # Try to load from cache first
        if cache_file.exists():
            try:
                stats = np.load(cache_file)
                logger.info("Loaded cached %s stats", var_name)
                return float(stats[0]), float(stats[1])
            except Exception as e:
                logger.warning("Failed to load cached stats: %s", str(e))
        
        # Compute statistics if not in cache
        t0 = time.time()
        # Properly handle NaN values
        data_masked = data_array.where(~np.isnan(data_array))
        
        with dask_monitor.profile_task(f"compute_{var_name}_stats"):
            mean = float(data_masked.mean().compute())
            std = float(data_masked.std().compute())
        
        t_elapsed = time.time() - t0
        logger.info("%s mean and std computed in %.2f seconds", var_name, t_elapsed)
        
        # Validate statistics
        if not np.isfinite(mean) or not np.isfinite(std):
            raise ValueError(f"Invalid statistics for {var_name}: mean={mean}, std={std}")
        
        # Avoid division by zero
        if std < 1e-6:
            std = 1.0
            logger.warning("Very small std detected for %s, using std=1.0", var_name)
        
        # Cache statistics
        try:
            np.save(cache_file, np.array([mean, std]))
            logger.info("Cached %s stats", var_name)
        except Exception as e:
            logger.warning("Failed to cache statistics: %s", str(e))
        
        return mean, std
    
    def calculate_heat_transport(self, latitude_index=None):
        """
        Calculate heat transport at specified latitude
        
        Args:
            latitude_index: Index of latitude to calculate heat transport at,
                            defaults to reference latitude
        
        Returns:
            heat_transport: Array of heat transport values
            mean_transport: Mean heat transport
            std_transport: Standard deviation of heat transport
        """
        if latitude_index is None:
            latitude_index = self.reference_latitude
        
        # Check for cached results
        cache_file = self.cache_dir / f'heat_transport_lat{latitude_index}.npy'
        cache_mean_file = self.cache_dir / f'heat_transport_lat{latitude_index}_mean.npy'
        cache_std_file = self.cache_dir / f'heat_transport_lat{latitude_index}_std.npy'
        
        if cache_file.exists() and cache_mean_file.exists() and cache_std_file.exists():
            try:
                heat_transport = np.load(cache_file)
                mean_transport = np.load(cache_mean_file)
                std_transport = np.load(cache_std_file)
                logger.info("Loaded cached heat transport for latitude index %d", latitude_index)
                return heat_transport, mean_transport, std_transport
            except Exception as e:
                logger.warning("Failed to load cached heat transport data: %s", str(e))
        
        # Calculate heat transport if not cached
        try:
            # Extract required variables
            vnt = self.vnt_ds["VNT"].isel(nlat=latitude_index)
            tarea = self.vnt_ds["TAREA"].isel(nlat=latitude_index)
            dz = self.vnt_ds["dz"]
            
            # Process in chunks to avoid memory issues
            chunk_size = 60
            total_times = vnt.shape[0]
            heat_transport_list = []
            
            logger.info("Calculating heat transport in %d-chunk steps", chunk_size)
            for start_idx in tqdm(range(0, total_times, chunk_size), desc="Heat transport chunks"):
                end_idx = min(start_idx + chunk_size, total_times)
                
                vnt_chunk = vnt.isel(time=slice(start_idx, end_idx))
                
                # Efficient calculation with proper broadcasting
                chunk_transport = (
                    vnt_chunk *
                    tarea * self.tarea_conversion *
                    dz * self.dz_conversion
                ).sum(dim=['z_t', 'nlon']).compute()
                
                heat_transport_list.append(chunk_transport)
            
            # Combine results
            heat_transport = np.concatenate(heat_transport_list)
            mean_transport = float(np.mean(heat_transport))
            std_transport = float(np.std(heat_transport))
            
            # Cache results
            np.save(cache_file, heat_transport)
            np.save(cache_mean_file, mean_transport)
            np.save(cache_std_file, std_transport)
            logger.info("Cached heat transport for latitude index %d", latitude_index)
            
            return heat_transport, mean_transport, std_transport
        except Exception as e:
            logger.error("Error calculating heat transport: %s", str(e))
            raise
        finally:
            self._cleanup_memory()
    
    def get_vnt_slice(self, time_indices=None, lat_indices=None, depth_indices=None):
        """
        Get a slice of VNT data efficiently
        
        Args:
            time_indices: Time indices to select
            lat_indices: Latitude indices to select
            depth_indices: Depth indices to select
            
        Returns:
            vnt_slice: xarray DataArray with selected VNT data
        """
        # Use all times if not specified
        if time_indices is None:
            time_indices = slice(None)
        
        # Default to reference latitude if not specified
        if lat_indices is None:
            lat_indices = [self.reference_latitude]
        
        # Use all depths if not specified
        if depth_indices is None:
            depth_indices = slice(None)
        
        # Get slice with optimal chunking
        vnt_slice = self.vnt_ds["VNT"].isel(
            time=time_indices,
            nlat=lat_indices,
            z_t=depth_indices
        )
        
        logger.info(f"Created VNT slice with shape {vnt_slice.shape}")
        return vnt_slice
    
    def get_aggregation_data(self):
        """Return data needed for heat transport aggregation"""
        tarea = self.vnt_ds["TAREA"].isel(nlat=self.reference_latitude)
        dz = self.vnt_ds["dz"]
        return tarea, dz
    
    def get_spatial_data(self):
        """Return spatial data for visualization and model input"""
        return (self.ssh_ds["SSH"], self.sst_ds["SST"],
                self.vnt_ds["TLAT"], self.vnt_ds["TLONG"])


def aggregate_vnt(predicted_vnt, tarea, dz, tarea_conversion=0.0001, dz_conversion=0.01, ref_lat_index=0):
    """
    Aggregate VNT to calculate heat transport
    
    Args:
        predicted_vnt: Predicted VNT tensor [batch, depth, lat, lon]
        tarea: Area of each grid cell
        dz: Depth of each layer
        tarea_conversion: Conversion factor for tarea
        dz_conversion: Conversion factor for dz
        ref_lat_index: Reference latitude index
        
    Returns:
        Aggregated heat transport
    """
    # Handle different input types (torch.Tensor or xarray.DataArray)
    if isinstance(predicted_vnt, torch.Tensor):
        # Convert xarray inputs to tensors if needed
        if not isinstance(tarea, torch.Tensor):
            tarea = torch.tensor(tarea.values, device=predicted_vnt.device, dtype=predicted_vnt.dtype)
        
        if not isinstance(dz, torch.Tensor):
            dz = torch.tensor(dz.values, device=predicted_vnt.device, dtype=predicted_vnt.dtype)
        
        # Proper broadcasting for tensors
        # [batch, depth, lat, lon] * [lon] * [depth] -> [batch, lat]
        result = (predicted_vnt * 
                 tarea.view(1, 1, 1, -1) * tarea_conversion * 
                 dz.view(1, -1, 1, 1) * dz_conversion).sum(dim=[1, 3])
        
        # Select reference latitude if needed
        if ref_lat_index is not None and result.shape[1] > 1:
            result = result[:, ref_lat_index]
            
        return result
    else:
        # Handle xarray DataArrays
        return (predicted_vnt * 
                tarea * tarea_conversion * 
                dz * dz_conversion).sum(dim=['z_t', 'nlon'])


class OceanDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for ocean data processing
    
    Efficient handling of SSH, SST and optional VNT data
    for heat transport prediction
    """
    def __init__(self, ssh, sst, heat_transport, heat_transport_mean, heat_transport_std,
                 ssh_mean, ssh_std, sst_mean, sst_std, shape, vnt_data=None, 
                 tarea=None, dz=None, ref_lat_index=0, debug=False, 
                 log_target=False, target_scale=10.0):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing OceanDataset")
        
        # Store target transformation parameters
        self.log_target = log_target
        self.target_scale = target_scale
        self.ref_lat_index = ref_lat_index
        
        # Store aggregation data
        self.vnt_data = vnt_data
        self.tarea = tarea
        self.dz = dz
        
        # Debug mode uses a small subset of data
        if debug:
            self.logger.info("Debug mode enabled, using reduced dataset (first 32 samples)")
            self.ssh = ssh.isel(time=slice(0, 32))
            self.sst = sst.isel(time=slice(0, 32))
            self.heat_transport = heat_transport[:32]
            self.length = 32
            
            # Also subset VNT data if available
            if self.vnt_data is not None:
                self.vnt_data = self.vnt_data.isel(time=slice(0, 32))
            
            # Extract month information
            self._extract_months(self.ssh.coords['time'])
        else:
            self.logger.info("Using full dataset with %d samples", shape[0])
            self.ssh = ssh
            self.sst = sst
            self.heat_transport = heat_transport
            self.length = shape[0]
            
            # Extract month information for seasonality modeling
            self._extract_months(self.ssh.coords['time'])
        
        # Store normalization parameters
        self.heat_transport_mean = float(heat_transport_mean)
        self.heat_transport_std = float(heat_transport_std)
        self.ssh_mean = float(ssh_mean)
        self.ssh_std = float(ssh_std)
        self.sst_mean = float(sst_mean)
        self.sst_std = float(sst_std)
        
        # Apply log transform to heat transport if needed
        self._process_targets()
        
        # Log normalization parameters
        self.logger.info("Normalization parameters:")
        self.logger.info("  SSH mean: %.4f, SSH std: %.4f", self.ssh_mean, self.ssh_std)
        self.logger.info("  SST mean: %.4f, SST std: %.4f", self.sst_mean, self.sst_std)
    
    def _extract_months(self, time_coord):
        """Extract month information from time coordinate"""
        try:
            # Try to extract month information properly
            units = time_coord.attrs.get('units')
            calendar = time_coord.attrs.get('calendar')
            time_vals = time_coord.values
            
            if units is None or calendar is None:
                # Fall back to string parsing if CF attributes not available
                self.months = np.array([pd.Timestamp(str(t)).month for t in time_vals])
            else:
                # Use xarray's CF-compliant time decoder
                decoded = xr.conventions.times.decode_cf_datetime(time_vals, units, calendar)
                if hasattr(decoded[0], "strftime"):
                    self.months = pd.DatetimeIndex(decoded).month.to_numpy()
                else:
                    self.months = np.array([pd.Timestamp(str(t)).month for t in decoded])
        except Exception as e:
            self.logger.warning("Failed to decode time: %s", str(e))
            # Fall back to basic parsing
            time_vals = time_coord.values
            self.months = np.array([pd.Timestamp(str(t)).month for t in time_vals])
    
    def _process_targets(self):
        """Process heat transport targets (apply log transform if needed)"""
        if self.log_target:
            self.logger.info("Applying log transform to heat transport")
            
            # Handle possible negative values
            min_value = np.min(self.heat_transport)
            if min_value <= 0:
                offset = abs(min_value) + 1e-8
                self.logger.info(f"Adding offset of {offset} to ensure positive values")
                self.heat_transport = np.log(self.heat_transport + offset)
                self.log_offset = offset
            else:
                self.heat_transport = np.log(self.heat_transport)
                self.log_offset = 0
                
            # Scale the log values
            self.logger.info("Using scaling factor: %.4f", self.target_scale)
            self.heat_transport = self.heat_transport / self.target_scale
            
            # Log statistics
            self.logger.info("Transformed heat transport: min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
                            float(np.min(self.heat_transport)), float(np.max(self.heat_transport)),
                            float(np.mean(self.heat_transport)), float(np.std(self.heat_transport)))
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        try:
            # Fetch SSH and SST data
            ssh_vals = np.asarray(self.ssh[idx].values, dtype=np.float32)
            sst_vals = np.asarray(self.sst[idx].values, dtype=np.float32)
            
            # Create valid masks (where data is not NaN)
            ssh_valid = ~np.isnan(ssh_vals)
            sst_valid = ~np.isnan(sst_vals)
            mask = ssh_valid & sst_valid
            
            # Apply normalization, replacing NaNs with zeros
            ssh_norm = np.where(ssh_valid, 
                            (ssh_vals - self.ssh_mean) / (self.ssh_std + 1e-8), 
                            0)
            sst_norm = np.where(sst_valid, 
                            (sst_vals - self.sst_mean) / (self.sst_std + 1e-8), 
                            0)
            
            # Get target value
            target = self.heat_transport[idx]
            
            # Create tensors
            ssh_tensor = torch.from_numpy(ssh_norm).float().unsqueeze(0)
            sst_tensor = torch.from_numpy(sst_norm).float().unsqueeze(0)
            mask_tensor = torch.from_numpy(mask).float()
            target_tensor = torch.tensor(target, dtype=torch.float32)
            
            # Include VNT data if available
            if self.vnt_data is not None:
                try:
                    vnt_slice = self.vnt_data.isel(time=idx).values
                    vnt_tensor = torch.from_numpy(vnt_slice).float()
                    return ssh_tensor, sst_tensor, mask_tensor, target_tensor, vnt_tensor
                except Exception as e:
                    self.logger.warning(f"Error fetching VNT data for idx {idx}: {e}")
            
            # Return without VNT data
            return ssh_tensor, sst_tensor, mask_tensor, target_tensor
                
        except Exception as e:
            self.logger.error("Error in __getitem__ at index %d: %s", idx, str(e))
            import traceback
            self.logger.error(traceback.format_exc())
            raise